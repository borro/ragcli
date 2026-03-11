package hybrid

import (
	"cmp"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/tools/filetools"
	"github.com/borro/ragcli/internal/verbose"

	mapmode "github.com/borro/ragcli/internal/map"
	"github.com/sashabaranov/go-openai"
)

type Options struct {
	TopK           int
	FinalK         int
	MapK           int
	ReadWindow     int
	Fallback       string
	ChunkSize      int
	ChunkOverlap   int
	IndexTTL       time.Duration
	IndexDir       string
	EmbeddingModel string
}

type DocumentProfile string

const (
	ProfilePlain    DocumentProfile = "plain"
	ProfileMarkdown DocumentProfile = "markdown"
	ProfileLogs     DocumentProfile = "logs"
	ProfileUnknown  DocumentProfile = "unknown"
)

type Segment struct {
	ID        int    `json:"id"`
	Level     string `json:"level"`
	Kind      string `json:"kind"`
	StartLine int    `json:"start_line"`
	EndLine   int    `json:"end_line"`
	ByteStart int    `json:"byte_start"`
	ByteEnd   int    `json:"byte_end"`
	Text      string `json:"text"`
}

type RetrievedHit struct {
	SegmentID    int
	Source       string
	Score        float64
	MatchedLine  int
	MatchedLines []int
}

type Region struct {
	SegmentIDs  []int
	StartLine   int
	EndLine     int
	ByteStart   int
	ByteEnd     int
	Text        string
	Score       float64
	BestLine    int
	HitSources  []string
	SourcePath  string
	DisplayName string
}

type RegionFacts struct {
	Region Region
	Facts  []string
}

type CoverageReport struct {
	Covered       bool
	Conflicts     []string
	Gaps          []string
	FollowUpQuery string
}

type EvidenceBlock struct {
	SourcePath string
	StartLine  int
	EndLine    int
	BestLine   int
	Text       string
	AroundText string
}

type cachedIndex struct {
	Manifest   indexManifest
	Segments   []Segment
	Embeddings [][]float32
}

type indexManifest = retrieval.Manifest

const (
	hybridIndexSchemaVersion = 2
	hybridManifestFilename   = "manifest.json"
	hybridSegmentsFilename   = "segments.jsonl"
	hybridEmbeddingsFilename = "embeddings.jsonl"
)

var (
	markdownHeadingPattern = regexp.MustCompile(`^\s{0,3}#{1,6}\s+\S`)
	logTimestampPattern    = regexp.MustCompile(`^(?:\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}|[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})`)
	logLevelPattern        = regexp.MustCompile(`\b(?:TRACE|DEBUG|INFO|WARN|WARNING|ERROR|FATAL|PANIC)\b`)
	errEmptyChatContent    = errors.New("chat response content is empty")
	errReasoningOnlyOutput = errors.New("chat response returned reasoning-only output")
)

func Run(ctx context.Context, chat llm.ChatAutoContextRequester, embedder llm.EmbeddingRequester, source input.Source, opts Options, question string, plan *verbose.Plan) (string, error) {
	prepareMeter := plan.Stage("prepare").Slice(2, 2)
	segmentsMeter := plan.Stage("segments")
	retrievalMeter := plan.Stage("retrieval")
	groundingMeter := plan.Stage("grounding")
	factMeter := plan.Stage("facts")
	coverageMeter := plan.Stage("coverage")
	finalMeter := plan.Stage("final")
	if source.Reader == nil {
		return "", errors.New("hybrid source reader is required")
	}

	startedAt := time.Now()
	prepareMeter.Start(localize.T("progress.hybrid.read_input"))
	displayName := normalizeSourceDisplayName(source)
	snapshot, err := spoolSourceSnapshot(source.Reader, displayName, opts)
	if err != nil {
		return "", fmt.Errorf("failed to spool source: %w", err)
	}
	defer func() {
		_ = os.Remove(snapshot.Path)
	}()
	prepareMeter.Done(localize.T("progress.hybrid.read_done", localize.Data{"Bytes": snapshot.Bytes}))
	if snapshot.Bytes == 0 {
		slog.Info("hybrid processing finished", "reason", "empty_input")
		return localize.T("map.answer.empty_file"), nil
	}

	readPath := normalizeSourceReadPath(source)
	slog.Info("hybrid processing started",
		"source", displayName,
		"has_file_path", strings.TrimSpace(readPath) != "",
		"content_bytes", snapshot.Bytes,
		"top_k", opts.TopK,
		"final_k", opts.FinalK,
		"map_k", opts.MapK,
		"read_window", opts.ReadWindow,
		"fallback", opts.Fallback,
	)

	profile := snapshot.Profile
	microCount, meso, err := segmentDocumentFromFile(snapshot.Path, profile, opts)
	if err != nil {
		return "", fmt.Errorf("failed to segment source: %w", err)
	}
	if len(meso) == 0 {
		slog.Info("hybrid processing finished", "reason", "no_segments", "source", displayName)
		return localize.T("map.answer.no_info"), nil
	}

	slog.Info("hybrid document prepared",
		"profile", profile,
		"micro_segments", microCount,
		"meso_segments", len(meso),
	)
	segmentsMeter.Start(localize.T("progress.hybrid.segment_start"))
	segmentsMeter.Done(localize.T("progress.hybrid.segment_done", localize.Data{"Profile": profile, "Count": len(meso)}))

	slog.Debug("hybrid lexical retrieval started", "segment_count", len(meso))
	retrievalMeter.Start(localize.T("progress.hybrid.lexical_start"))
	lexicalHits := lexicalRetrieve(question, meso)
	slog.Debug("hybrid lexical retrieval finished", "hits", len(lexicalHits))
	retrievalMeter.Note(localize.T("progress.hybrid.lexical_done", localize.Data{"Count": len(lexicalHits)}))

	retrievalMeter.Note(localize.T("progress.hybrid.semantic_start"))
	slog.Debug("hybrid semantic retrieval started", "segment_count", len(meso))
	semanticHits, semanticErr := semanticRetrieve(ctx, embedder, displayName, snapshot.Hash, meso, opts, question)
	if semanticErr != nil {
		slog.Warn("hybrid semantic retrieval failed", "fallback", opts.Fallback, "error", semanticErr)
		switch opts.Fallback {
		case "map":
			slog.Info("hybrid switching to map fallback", "reason", "semantic_retrieval_failed")
			return fallbackToMap(ctx, chat, source, opts, question, snapshot.Path, nil)
		case "rag-only":
			slog.Warn("hybrid semantic retrieval unavailable, continuing with lexical retrieval only", "error", semanticErr)
		default:
			return "", semanticErr
		}
	}
	if semanticErr == nil {
		slog.Debug("hybrid semantic retrieval finished", "hits", len(semanticHits))
		retrievalMeter.Note(localize.T("progress.hybrid.semantic_done", localize.Data{"Count": len(semanticHits)}))
	} else {
		retrievalMeter.Note(localize.T("progress.hybrid.semantic_unavailable"))
	}

	slog.Debug("hybrid fusion started", "lexical_hits", len(lexicalHits), "semantic_hits", len(semanticHits))
	regions := fuseAndExpandRegions(lexicalHits, semanticHits, meso, readPath, displayName, opts)
	slog.Info("hybrid fusion finished", "regions", len(regions))
	retrievalMeter.Done(localize.T("progress.hybrid.regions_found", localize.Data{"Count": len(regions)}))
	if len(regions) == 0 || retrievalTooWeak(regions) {
		slog.Warn("hybrid retrieval too weak", "regions", len(regions), "fallback", opts.Fallback)
		if opts.Fallback == "map" {
			slog.Info("hybrid switching to map fallback", "reason", "weak_retrieval")
			return fallbackToMap(ctx, chat, source, opts, question, snapshot.Path, nil)
		}
		if len(regions) == 0 {
			slog.Info("hybrid processing finished", "reason", "no_evidence", "duration", roundDuration(time.Since(startedAt)))
			return localize.T("hybrid.answer.insufficient"), nil
		}
	}

	finalRegions := slices.Clone(regions)
	if len(finalRegions) > opts.FinalK {
		finalRegions = finalRegions[:opts.FinalK]
	}
	mapRegions := slices.Clone(regions)
	if len(mapRegions) > opts.MapK {
		mapRegions = mapRegions[:opts.MapK]
	}

	slog.Debug("hybrid grounding started", "regions", len(finalRegions))
	groundingMeter.Start(localize.T("progress.hybrid.grounding_start"))
	evidence, err := groundEvidence(finalRegions, opts.ReadWindow, groundingMeter)
	if err != nil {
		return "", err
	}
	groundingMeter.Done(localize.T("progress.hybrid.grounding_done", localize.Data{"Count": len(evidence)}))
	slog.Info("hybrid grounding finished", "evidence_blocks", len(evidence))

	slog.Debug("hybrid map extraction started", "regions", len(mapRegions))
	factMeter.Start(localize.T("progress.hybrid.fact_start"))
	factBlocks, reducedFacts, err := mapExtractFacts(ctx, chat, mapRegions, question, factMeter)
	if err != nil {
		return "", err
	}
	slog.Info("hybrid map extraction finished",
		"region_fact_blocks", len(factBlocks),
		"fact_lines", countFactLines(reducedFacts),
	)

	slog.Debug("hybrid coverage check started")
	coverageMeter.Start(localize.T("progress.hybrid.coverage_start"))
	coverage, err := checkCoverage(ctx, chat, question, reducedFacts, evidence)
	if err != nil {
		return "", err
	}
	coverageMeter.Done(localize.T("progress.hybrid.coverage_done"))
	slog.Info("hybrid coverage check finished",
		"covered", coverage.Covered,
		"gaps", len(coverage.Gaps),
		"conflicts", len(coverage.Conflicts),
		"has_follow_up", hasFollowUpQuery(deriveFollowUpQuery(question, coverage)),
	)

	followUpQuery := deriveFollowUpQuery(question, coverage)
	if !coverage.Covered && hasFollowUpQuery(followUpQuery) {
		slog.Info("hybrid targeted reread started", "follow_up_query", followUpQuery)
		existing := make(map[int]struct{}, len(regions))
		for _, region := range regions {
			for _, segmentID := range region.SegmentIDs {
				existing[segmentID] = struct{}{}
			}
		}

		extraLexical := lexicalRetrieve(followUpQuery, meso)
		extraSemantic := []RetrievedHit(nil)
		if semanticErr == nil {
			extraSemantic, _ = semanticRetrieve(ctx, embedder, displayName, snapshot.Hash, meso, opts, followUpQuery)
		}
		extra := fuseAndExpandRegions(extraLexical, extraSemantic, meso, readPath, displayName, opts)
		extra = filterRegionsBySegments(extra, existing)
		if len(extra) > 0 {
			if len(extra) > opts.FinalK {
				extra = extra[:opts.FinalK]
			}
			slog.Debug("hybrid targeted reread grounding started", "regions", len(extra))
			groundingMeter.Start(localize.T("progress.hybrid.extra_start"))
			extraEvidence, err := groundEvidence(extra, opts.ReadWindow, groundingMeter)
			if err != nil {
				return "", err
			}
			groundingMeter.Done(localize.T("progress.hybrid.extra_done", localize.Data{"Count": len(extra)}))
			evidence = appendEvidence(evidence, extraEvidence)

			slog.Debug("hybrid targeted reread map extraction started", "regions", len(extra))
			_, extraReduced, err := mapExtractFacts(ctx, chat, extra, followUpQuery, factMeter)
			if err != nil {
				return "", err
			}
			reducedFacts = mergeFactBlocks(reducedFacts, extraReduced)
			slog.Info("hybrid targeted reread finished",
				"extra_regions", len(extra),
				"total_evidence_blocks", len(evidence),
				"fact_lines", countFactLines(reducedFacts),
			)
		} else {
			slog.Info("hybrid targeted reread finished", "extra_regions", 0)
		}
	}

	slog.Debug("hybrid answer synthesis started", "evidence_blocks", len(evidence), "fact_lines", countFactLines(reducedFacts))
	finalMeter.Start(localize.T("progress.hybrid.final_start", localize.Data{"Evidence": len(evidence), "Facts": countFactLines(reducedFacts)}))
	answer, err := synthesizeAnswer(ctx, chat, question, reducedFacts, evidence)
	if err != nil {
		if (errors.Is(err, errEmptyChatContent) || errors.Is(err, errReasoningOnlyOutput)) && opts.Fallback == "map" {
			slog.Warn("hybrid answer synthesis returned empty content, switching to map fallback")
			return fallbackToMap(ctx, chat, source, opts, question, snapshot.Path, nil)
		}
		return "", err
	}
	result := appendSources(answer, evidence)
	finalMeter.Note(localize.T("progress.hybrid.final_citations"))
	finalMeter.Done(localize.T("progress.hybrid.final_done"))
	slog.Info("hybrid processing finished",
		"reason", "success",
		"duration", roundDuration(time.Since(startedAt)),
		"evidence_blocks", len(evidence),
		"fact_lines", countFactLines(reducedFacts),
		"result_empty", strings.TrimSpace(result) == "",
	)
	return result, nil
}

func profileDocument(sourcePath, content string) DocumentProfile {
	lines := strings.Split(strings.ReplaceAll(content, "\r\n", "\n"), "\n")
	stats := profileStats{}
	for _, line := range lines {
		collectProfileStats(line, &stats)
	}
	return detectProfile(sourcePath, stats.headingCount, stats.logLikeCount, stats.nonEmpty)
}

type lineInfo struct {
	text      string
	lineNo    int
	byteStart int
	byteEnd   int
}

type sourceSnapshot struct {
	Path    string
	Bytes   int
	Profile DocumentProfile
	Hash    string
}

type profileStats struct {
	headingCount int
	logLikeCount int
	nonEmpty     bool
}

type segmentGroup struct {
	lines []lineInfo
	kind  string
}

func segmentDocument(content []byte, profile DocumentProfile, opts Options) ([]Segment, []Segment) {
	lines := splitLines(content)
	return segmentLines(lines, profile, opts)
}

func segmentDocumentFromFile(path string, profile DocumentProfile, opts Options) (int, []Segment, error) {
	lines, err := readLines(path)
	if err != nil {
		return 0, nil, err
	}
	return microSegmentCount(lines, opts.ReadWindow), buildMesoSegments(lines, profile, opts), nil
}

func segmentLines(lines []lineInfo, profile DocumentProfile, opts Options) ([]Segment, []Segment) {
	if len(lines) == 0 {
		return nil, nil
	}

	micro := buildMicroSegments(lines, opts.ReadWindow)
	meso := buildMesoSegments(lines, profile, opts)
	return micro, meso
}

func microSegmentCount(lines []lineInfo, readWindow int) int {
	if len(lines) == 0 {
		return 0
	}
	window := max(readWindow*2+1, 8)
	return (len(lines) + window - 1) / window
}

func splitLines(content []byte) []lineInfo {
	rawLines := strings.Split(string(content), "\n")
	if len(rawLines) > 0 && rawLines[len(rawLines)-1] == "" && len(content) > 0 && content[len(content)-1] == '\n' {
		rawLines = rawLines[:len(rawLines)-1]
	}

	lines := make([]lineInfo, 0, len(rawLines))
	offset := 0
	for i, line := range rawLines {
		byteLen := len(line)
		if i < len(rawLines)-1 {
			byteLen++
		}
		lines = append(lines, lineInfo{
			text:      line,
			lineNo:    i + 1,
			byteStart: offset,
			byteEnd:   offset + byteLen,
		})
		offset += byteLen
	}
	return lines
}

func collectProfileStats(text string, stats *profileStats) {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return
	}
	stats.nonEmpty = true
	if markdownHeadingPattern.MatchString(trimmed) {
		stats.headingCount++
	}
	if logTimestampPattern.MatchString(trimmed) || logLevelPattern.MatchString(trimmed) {
		stats.logLikeCount++
	}
}

func readLines(path string) ([]lineInfo, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer func() { _ = file.Close() }()

	lines := make([]lineInfo, 0, 128)
	_, err = retrieval.WalkLines(file, func(_ string, text string, lineNo, byteStart, byteEnd int) error {
		lines = append(lines, lineInfo{
			text:      text,
			lineNo:    lineNo,
			byteStart: byteStart,
			byteEnd:   byteEnd,
		})
		return nil
	})
	if err != nil {
		return nil, err
	}
	return lines, nil
}

func spoolSourceSnapshot(reader io.Reader, sourcePath string, opts Options) (*sourceSnapshot, error) {
	stats := profileStats{}
	spooled, err := retrieval.SpoolSource(reader, "ragcli-hybrid-source-*", []string{
		sourcePath,
		fmt.Sprintf("%d", hybridIndexSchemaVersion),
		fmt.Sprintf("%d", opts.ChunkSize),
		fmt.Sprintf("%d", opts.ChunkOverlap),
		opts.EmbeddingModel,
	}, func(_ string, text string, _ int, _, _ int) error {
		collectProfileStats(text, &stats)
		return nil
	})
	if err != nil {
		return nil, err
	}

	return &sourceSnapshot{
		Path:    spooled.TempPath,
		Bytes:   spooled.Bytes,
		Profile: detectProfile(sourcePath, stats.headingCount, stats.logLikeCount, stats.nonEmpty),
		Hash:    spooled.Hash,
	}, nil
}

func detectProfile(sourcePath string, headingCount, logLikeCount int, nonEmpty bool) DocumentProfile {
	lowerPath := strings.ToLower(strings.TrimSpace(sourcePath))
	switch {
	case strings.HasSuffix(lowerPath, ".md"), strings.HasSuffix(lowerPath, ".markdown"):
		return ProfileMarkdown
	case strings.HasSuffix(lowerPath, ".log"), strings.HasSuffix(lowerPath, ".out"):
		return ProfileLogs
	}

	switch {
	case headingCount >= 2:
		return ProfileMarkdown
	case logLikeCount >= 3:
		return ProfileLogs
	case !nonEmpty:
		return ProfileUnknown
	default:
		return ProfilePlain
	}
}

func buildMicroSegments(lines []lineInfo, readWindow int) []Segment {
	window := max(readWindow*2+1, 8)
	segments := make([]Segment, 0, microSegmentCount(lines, readWindow))
	for start := 0; start < len(lines); start += window {
		end := min(start+window, len(lines))
		segments = append(segments, buildSegment(len(segments), "micro", "window", lines[start:end]))
	}
	return segments
}

func buildMesoSegments(lines []lineInfo, profile DocumentProfile, opts Options) []Segment {
	var groups []segmentGroup

	switch profile {
	case ProfileMarkdown:
		groups = splitMarkdownGroups(lines, opts.ChunkSize)
	case ProfileLogs:
		groups = splitLogGroups(lines, opts.ChunkSize)
	default:
		groups = splitPlainGroups(lines, opts.ChunkSize)
	}

	segments := make([]Segment, 0, len(groups))
	for _, group := range groups {
		if len(group.lines) == 0 {
			continue
		}
		kind := group.kind
		if kind == "" {
			kind = "section"
		}
		segments = append(segments, buildSegment(len(segments), "meso", kind, group.lines))
	}
	return segments
}

func splitPlainGroups(lines []lineInfo, targetSize int) []segmentGroup {
	blocks := collectParagraphBlocks(lines, false)
	return mergeBlocks(blocks, targetSize, "section")
}

func splitMarkdownGroups(lines []lineInfo, targetSize int) []segmentGroup {
	blocks := collectParagraphBlocks(lines, true)
	return mergeBlocks(blocks, targetSize, "section")
}

func splitLogGroups(lines []lineInfo, targetSize int) []segmentGroup {
	blocks := collectLogBlocks(lines)
	return mergeBlocks(blocks, targetSize, "incident")
}

func collectParagraphBlocks(lines []lineInfo, markdown bool) []segmentGroup {
	blocks := make([]segmentGroup, 0, max(1, len(lines)/6))
	current := make([]lineInfo, 0, 12)
	currentKind := "paragraph"

	flush := func() {
		if len(current) == 0 {
			return
		}
		blocks = append(blocks, segmentGroup{
			lines: slices.Clone(current),
			kind:  currentKind,
		})
		current = current[:0]
		currentKind = "paragraph"
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line.text)
		isHeading := markdown && markdownHeadingPattern.MatchString(trimmed)
		switch {
		case isHeading:
			flush()
			current = append(current, line)
			currentKind = "heading"
		case trimmed == "":
			if len(current) > 0 {
				current = append(current, line)
				flush()
			}
		default:
			current = append(current, line)
		}
	}
	flush()
	return blocks
}

func collectLogBlocks(lines []lineInfo) []segmentGroup {
	blocks := make([]segmentGroup, 0, max(1, len(lines)/10))
	current := make([]lineInfo, 0, 24)

	flush := func() {
		if len(current) == 0 {
			return
		}
		blocks = append(blocks, segmentGroup{
			lines: slices.Clone(current),
			kind:  "incident",
		})
		current = current[:0]
	}

	for _, line := range lines {
		trimmed := strings.TrimSpace(line.text)
		newIncident := len(current) > 0 && logTimestampPattern.MatchString(trimmed)
		if newIncident {
			flush()
		}
		current = append(current, line)
		if trimmed == "" {
			flush()
		}
	}
	flush()
	return blocks
}

func mergeBlocks(blocks []segmentGroup, targetSize int, defaultKind string) []segmentGroup {
	if len(blocks) == 0 {
		return nil
	}

	targetRunes := max(targetSize, 240)
	maxRunes := int(float64(targetRunes) * 1.5)
	groups := make([]segmentGroup, 0, len(blocks))
	current := make([]lineInfo, 0, 24)
	currentRunes := 0
	currentKind := defaultKind

	flush := func() {
		if len(current) == 0 {
			return
		}
		groups = append(groups, segmentGroup{
			lines: slices.Clone(current),
			kind:  currentKind,
		})
		current = current[:0]
		currentRunes = 0
		currentKind = defaultKind
	}

	for _, block := range blocks {
		blockRunes := groupRuneCount(block.lines)
		blockKind := block.kind
		if blockKind == "" {
			blockKind = defaultKind
		}

		if len(current) == 0 {
			current = append(current, block.lines...)
			currentRunes = blockRunes
			currentKind = blockKind
			continue
		}

		shouldFlush := currentRunes >= targetRunes
		if currentRunes+blockRunes > maxRunes && currentRunes >= targetRunes/2 {
			shouldFlush = true
		}
		if block.kind == "heading" {
			shouldFlush = true
		}
		if shouldFlush {
			flush()
		}
		current = append(current, block.lines...)
		currentRunes += blockRunes
		if currentKind == defaultKind && blockKind != "" {
			currentKind = blockKind
		}
	}

	flush()
	return groups
}

func groupRuneCount(lines []lineInfo) int {
	total := 0
	for _, line := range lines {
		total += utf8.RuneCountInString(line.text) + 1
	}
	return total
}

func buildSegment(id int, level, kind string, lines []lineInfo) Segment {
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		parts = append(parts, line.text)
	}
	return Segment{
		ID:        id,
		Level:     level,
		Kind:      kind,
		StartLine: lines[0].lineNo,
		EndLine:   lines[len(lines)-1].lineNo,
		ByteStart: lines[0].byteStart,
		ByteEnd:   lines[len(lines)-1].byteEnd,
		Text:      strings.Join(parts, "\n"),
	}
}

func lexicalRetrieve(question string, segments []Segment) []RetrievedHit {
	lowerQuestion := strings.ToLower(strings.TrimSpace(question))
	queryText := strings.Join(retrieval.Tokenize(question), " ")
	if lowerQuestion == "" || len(segments) == 0 {
		return nil
	}

	hits := make([]RetrievedHit, 0, len(segments))
	for _, segment := range segments {
		score, lineNo, matchedLines := lexicalScoreSegment(lowerQuestion, queryText, segment)
		if score <= 0 {
			continue
		}
		hits = append(hits, RetrievedHit{
			SegmentID:    segment.ID,
			Source:       "lexical",
			Score:        score,
			MatchedLine:  lineNo,
			MatchedLines: matchedLines,
		})
	}

	slices.SortStableFunc(hits, func(a, b RetrievedHit) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		default:
			return cmp.Compare(a.SegmentID, b.SegmentID)
		}
	})
	return hits
}

func lexicalScoreSegment(lowerQuestion, queryText string, segment Segment) (float64, int, []int) {
	lowerText := strings.ToLower(segment.Text)
	bestScore := 0.0
	bestLine := segment.StartLine
	matchedLines := make([]int, 0, 4)

	if strings.Contains(lowerText, lowerQuestion) {
		bestScore = 1.75
	}

	for offset, line := range strings.Split(segment.Text, "\n") {
		lowerLine := strings.ToLower(line)
		literal := 0.0
		if strings.Contains(lowerLine, lowerQuestion) {
			literal = 1.5
		}
		overlap := retrieval.TokenOverlapRatio(queryText, line)
		score := literal + overlap
		if score > 0 {
			matchedLines = append(matchedLines, segment.StartLine+offset)
		}
		if score > bestScore {
			bestScore = score
			bestLine = segment.StartLine + offset
		}
	}

	if bestScore == 0 && queryText != "" {
		overlap := retrieval.TokenOverlapRatio(queryText, segment.Text)
		if overlap > 0 {
			bestScore = overlap
		}
	}

	return bestScore, bestLine, matchedLines
}

func semanticRetrieve(ctx context.Context, embedder llm.EmbeddingRequester, sourcePath, inputHash string, segments []Segment, opts Options, question string) ([]RetrievedHit, error) {
	index, err := buildOrLoadIndex(ctx, embedder, sourcePath, inputHash, segments, opts)
	if err != nil {
		return nil, err
	}

	queryEmbedding, _, err := embedQuery(ctx, embedder, question)
	if err != nil {
		return nil, err
	}

	hits := make([]RetrievedHit, 0, len(index.Segments))
	for i, segment := range index.Segments {
		score := retrieval.CosineSimilarity(queryEmbedding, index.Embeddings[i])
		if score <= 0 {
			continue
		}
		hits = append(hits, RetrievedHit{
			SegmentID: segment.ID,
			Source:    "semantic",
			Score:     score,
		})
	}

	slices.SortStableFunc(hits, func(a, b RetrievedHit) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		default:
			return cmp.Compare(a.SegmentID, b.SegmentID)
		}
	})
	return trimSemanticHits(hits, opts.TopK), nil
}

func trimSemanticHits(hits []RetrievedHit, topK int) []RetrievedHit {
	if len(hits) == 0 {
		return nil
	}

	limit := max(topK*6, 24)
	if limit > len(hits) {
		limit = len(hits)
	}
	best := hits[0].Score
	minScore := maxFloat(0.18, best*0.55)

	trimmed := make([]RetrievedHit, 0, limit)
	for _, hit := range hits {
		if len(trimmed) >= limit {
			break
		}
		if hit.Score < minScore {
			break
		}
		trimmed = append(trimmed, hit)
	}
	if len(trimmed) == 0 {
		trimmed = append(trimmed, hits[0])
	}
	return trimmed
}

func buildOrLoadIndex(ctx context.Context, embedder llm.EmbeddingRequester, sourcePath, inputHash string, segments []Segment, opts Options) (*cachedIndex, error) {
	if err := os.MkdirAll(opts.IndexDir, 0o700); err != nil {
		return nil, fmt.Errorf("failed to create hybrid index dir: %w", err)
	}
	retrieval.CleanupExpiredIndexes(opts.IndexDir, opts.IndexTTL)

	indexDir := filepath.Join(opts.IndexDir, "hybrid-"+inputHash)
	if cached, err := loadIndex(indexDir, opts.IndexTTL); err == nil {
		return cached, nil
	}

	embeddings, err := embedSegments(ctx, embedder, segments)
	if err != nil {
		return nil, err
	}

	index := &cachedIndex{
		Manifest: indexManifest{
			SchemaVersion:  hybridIndexSchemaVersion,
			CreatedAt:      time.Now().UTC(),
			InputHash:      inputHash,
			SourcePath:     sourcePath,
			EmbeddingModel: opts.EmbeddingModel,
			ChunkSize:      opts.ChunkSize,
			ChunkOverlap:   opts.ChunkOverlap,
			ItemCount:      len(segments),
		},
		Segments:   segments,
		Embeddings: embeddings,
	}

	if err := persistIndex(indexDir, index); err != nil {
		return nil, err
	}
	return index, nil
}

func loadIndex(indexDir string, ttl time.Duration) (*cachedIndex, error) {
	manifest, segments, embeddings, err := retrieval.LoadIndex[Segment](indexDir, ttl, hybridIndexSchemaVersion, retrieval.Filenames{
		Manifest:   hybridManifestFilename,
		Items:      hybridSegmentsFilename,
		Embeddings: hybridEmbeddingsFilename,
	})
	if err != nil {
		return nil, retrieval.WrapIndexError("hybrid", err, "")
	}
	return &cachedIndex{
		Manifest:   manifest,
		Segments:   segments,
		Embeddings: embeddings,
	}, nil
}

func persistIndex(indexDir string, index *cachedIndex) error {
	return retrieval.PersistIndex(indexDir, index.Manifest, index.Segments, index.Embeddings, retrieval.Filenames{
		Manifest:   hybridManifestFilename,
		Items:      hybridSegmentsFilename,
		Embeddings: hybridEmbeddingsFilename,
	})
}

func embedSegments(ctx context.Context, embedder llm.EmbeddingRequester, segments []Segment) ([][]float32, error) {
	const batchSize = 32
	vectors := make([][]float32, 0, len(segments))
	for start := 0; start < len(segments); start += batchSize {
		end := min(start+batchSize, len(segments))
		inputs := make([]string, 0, end-start)
		for _, segment := range segments[start:end] {
			inputs = append(inputs, segment.Text)
		}
		batch, _, err := retrieval.EmbedTexts(ctx, embedder, inputs)
		if err != nil {
			return nil, fmt.Errorf("failed to embed hybrid segments: %w", err)
		}
		vectors = append(vectors, batch...)
	}
	return vectors, nil
}

func embedQuery(ctx context.Context, embedder llm.EmbeddingRequester, question string) ([]float32, llm.EmbeddingMetrics, error) {
	return retrieval.EmbedQuery(ctx, embedder, question, "hybrid query")
}

func fuseAndExpandRegions(lexicalHits, semanticHits []RetrievedHit, segments []Segment, sourcePath, displayName string, opts Options) []Region {
	type fusionState struct {
		Score      float64
		BestLine   int
		HitSources map[string]struct{}
	}

	fusion := make(map[int]*fusionState)
	applyHits := func(hits []RetrievedHit) {
		for rank, hit := range hits {
			state := fusion[hit.SegmentID]
			if state == nil {
				state = &fusionState{HitSources: make(map[string]struct{})}
				fusion[hit.SegmentID] = state
			}
			state.Score += 1.0/float64(rank+60) + 0.15*hit.Score
			if hit.MatchedLine > 0 && state.BestLine == 0 {
				state.BestLine = hit.MatchedLine
			}
			state.HitSources[hit.Source] = struct{}{}
		}
	}
	applyHits(lexicalHits)
	applyHits(semanticHits)

	type scoredSegment struct {
		ID       int
		Score    float64
		BestLine int
		Sources  []string
	}

	scored := make([]scoredSegment, 0, len(fusion))
	for id, state := range fusion {
		sources := make([]string, 0, len(state.HitSources))
		for source := range state.HitSources {
			sources = append(sources, source)
		}
		slices.Sort(sources)
		scored = append(scored, scoredSegment{
			ID:       id,
			Score:    state.Score,
			BestLine: state.BestLine,
			Sources:  sources,
		})
	}
	slices.SortStableFunc(scored, func(a, b scoredSegment) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		default:
			return cmp.Compare(a.ID, b.ID)
		}
	})
	if len(scored) > opts.TopK {
		scored = scored[:opts.TopK]
	}

	selected := make(map[int]scoredSegment, len(scored)*3)
	for _, item := range scored {
		for _, id := range []int{item.ID - 1, item.ID, item.ID + 1} {
			if id < 0 || id >= len(segments) {
				continue
			}
			if current, ok := selected[id]; !ok || item.Score > current.Score {
				selected[id] = scoredSegment{
					ID:       id,
					Score:    item.Score,
					BestLine: item.BestLine,
					Sources:  item.Sources,
				}
			}
		}
	}

	ids := make([]int, 0, len(selected))
	for id := range selected {
		ids = append(ids, id)
	}
	slices.Sort(ids)

	regions := make([]Region, 0, len(ids))
	for i := 0; i < len(ids); {
		start := i
		end := i + 1
		for end < len(ids) && ids[end] == ids[end-1]+1 {
			end++
		}

		segmentIDs := ids[start:end]
		texts := make([]string, 0, len(segmentIDs))
		score := 0.0
		bestLine := 0
		sourceSet := make(map[string]struct{})
		for _, id := range segmentIDs {
			segment := segments[id]
			texts = append(texts, segment.Text)
			score += selected[id].Score
			if bestLine == 0 && selected[id].BestLine > 0 {
				bestLine = selected[id].BestLine
			}
			for _, source := range selected[id].Sources {
				sourceSet[source] = struct{}{}
			}
		}
		hitSources := make([]string, 0, len(sourceSet))
		for source := range sourceSet {
			hitSources = append(hitSources, source)
		}
		slices.Sort(hitSources)

		first := segments[segmentIDs[0]]
		last := segments[segmentIDs[len(segmentIDs)-1]]
		regions = append(regions, Region{
			SegmentIDs:  slices.Clone(segmentIDs),
			StartLine:   first.StartLine,
			EndLine:     last.EndLine,
			ByteStart:   first.ByteStart,
			ByteEnd:     last.ByteEnd,
			Text:        strings.Join(texts, "\n"),
			Score:       score,
			BestLine:    bestLine,
			HitSources:  hitSources,
			SourcePath:  sourcePath,
			DisplayName: displayName,
		})
		i = end
	}

	regions = dedupeRegions(regions)
	slices.SortStableFunc(regions, func(a, b Region) int {
		switch {
		case a.Score > b.Score:
			return -1
		case a.Score < b.Score:
			return 1
		default:
			return cmp.Compare(a.StartLine, b.StartLine)
		}
	})
	return regions
}

func dedupeRegions(regions []Region) []Region {
	if len(regions) <= 1 {
		return regions
	}
	kept := make([]Region, 0, len(regions))
	for _, region := range regions {
		duplicate := false
		for _, existing := range kept {
			if overlapRatio(region, existing) >= 0.8 {
				duplicate = true
				break
			}
		}
		if !duplicate {
			kept = append(kept, region)
		}
	}
	return kept
}

func overlapRatio(a, b Region) float64 {
	start := max(a.StartLine, b.StartLine)
	end := min(a.EndLine, b.EndLine)
	if end < start {
		return 0
	}
	overlap := end - start + 1
	union := max(a.EndLine, b.EndLine) - min(a.StartLine, b.StartLine) + 1
	return float64(overlap) / float64(union)
}

func retrievalTooWeak(regions []Region) bool {
	if len(regions) == 0 {
		return true
	}
	return regions[0].Score < 0.18
}

func groundEvidence(regions []Region, readWindow int, meter verbose.Meter) ([]EvidenceBlock, error) {
	evidence := make([]EvidenceBlock, 0, len(regions))
	for index, region := range regions {
		linesRaw, err := filetools.ReadLines(region.SourcePath, region.StartLine, region.EndLine)
		if err != nil {
			return nil, err
		}
		var linesResult filetools.ReadLinesResult
		if err := json.Unmarshal([]byte(linesRaw), &linesResult); err != nil {
			return nil, fmt.Errorf("failed to decode read_lines result: %w", err)
		}

		block := EvidenceBlock{
			SourcePath: region.DisplayNameOrPath(),
			StartLine:  linesResult.StartLine,
			EndLine:    linesResult.EndLine,
			BestLine:   region.BestLine,
			Text:       lineSlicesText(linesResult.Lines),
		}

		if region.BestLine > 0 {
			aroundRaw, err := filetools.ReadAround(region.SourcePath, region.BestLine, readWindow, readWindow)
			if err != nil {
				return nil, err
			}
			var aroundResult filetools.ReadAroundResult
			if err := json.Unmarshal([]byte(aroundRaw), &aroundResult); err != nil {
				return nil, fmt.Errorf("failed to decode read_around result: %w", err)
			}
			block.AroundText = lineSlicesText(aroundResult.Lines)
		}

		evidence = append(evidence, block)
		meter.Report(index+1, len(regions), localize.T("progress.hybrid.regions_read", localize.Data{"Index": index + 1, "Total": len(regions)}))
	}
	return evidence, nil
}

func (r Region) DisplayNameOrPath() string {
	if strings.TrimSpace(r.DisplayName) != "" {
		return strings.TrimSpace(r.DisplayName)
	}
	return strings.TrimSpace(r.SourcePath)
}

func lineSlicesText(lines []filetools.LineSlice) string {
	if len(lines) == 0 {
		return ""
	}
	parts := make([]string, 0, len(lines))
	for _, line := range lines {
		parts = append(parts, fmt.Sprintf("%d: %s", line.LineNumber, line.Content))
	}
	return strings.Join(parts, "\n")
}

func mapExtractFacts(ctx context.Context, chat llm.ChatRequester, regions []Region, question string, meter verbose.Meter) ([]RegionFacts, string, error) {
	if len(regions) == 0 {
		return nil, "", nil
	}

	result := make([]RegionFacts, 0, len(regions))
	blocks := make([]string, 0, len(regions))
	for index, region := range regions {
		meter.Note(localize.T("progress.hybrid.region_send", localize.Data{"Index": index + 1, "Total": len(regions)}))
		facts, err := callChat(ctx, chat, chatCallOptions{
			Stage:                   "map_extract_region",
			MaxCompletionTokens:     220,
			EmptyContentRetryPrompt: localize.T("hybrid.prompt.region_retry"),
		}, []openai.ChatCompletionMessage{
			{Role: "system", Content: hybridMapPrompt()},
			{Role: "user", Content: localize.T("hybrid.prompt.region_user", localize.Data{"Question": strings.TrimSpace(question), "Region": sanitize(region.Text)})},
		})
		if err != nil {
			if errors.Is(err, errEmptyChatContent) || errors.Is(err, errReasoningOnlyOutput) {
				slog.Warn("hybrid map extraction skipped empty region response",
					"region_start_line", region.StartLine,
					"region_end_line", region.EndLine,
				)
				continue
			}
			return nil, "", err
		}
		normalized := normalizeFacts(facts)
		if normalized == "" || strings.EqualFold(normalized, "SKIP") {
			meter.Report(index+1, len(regions), localize.T("progress.hybrid.regions_processed", localize.Data{"Index": index + 1, "Total": len(regions)}))
			continue
		}
		factList := strings.Split(normalized, "\n")
		result = append(result, RegionFacts{Region: region, Facts: factList})
		blocks = append(blocks, normalized)
		meter.Report(index+1, len(regions), localize.T("progress.hybrid.regions_processed", localize.Data{"Index": index + 1, "Total": len(regions)}))
	}
	if len(blocks) == 0 {
		return result, "", nil
	}
	if len(blocks) == 1 {
		return result, blocks[0], nil
	}

	reduced, err := callChat(ctx, chat, chatCallOptions{
		Stage:                   "map_extract_reduce",
		MaxCompletionTokens:     320,
		EmptyContentRetryPrompt: localize.T("hybrid.prompt.reduce_retry"),
	}, []openai.ChatCompletionMessage{
		{Role: "system", Content: hybridReducePrompt()},
		{Role: "user", Content: localize.T("hybrid.prompt.reduce_user", localize.Data{"Question": strings.TrimSpace(question), "Facts": strings.Join(blocks, "\n\n---\n\n")})},
	})
	if err != nil {
		if errors.Is(err, errEmptyChatContent) || errors.Is(err, errReasoningOnlyOutput) {
			return result, strings.Join(blocks, "\n"), nil
		}
		return nil, "", err
	}
	return result, normalizeFacts(reduced), nil
}

func checkCoverage(ctx context.Context, chat llm.ChatRequester, question, facts string, evidence []EvidenceBlock) (CoverageReport, error) {
	if strings.TrimSpace(facts) == "" && len(evidence) == 0 {
		return CoverageReport{Covered: false}, nil
	}

	evidenceSummary := make([]string, 0, len(evidence))
	for i, block := range evidence {
		evidenceSummary = append(evidenceSummary, fmt.Sprintf("[source %d] %s:%d-%d", i+1, block.SourcePath, block.StartLine, block.EndLine))
	}
	content, err := callChat(ctx, chat, chatCallOptions{
		Stage:                   "coverage_check",
		MaxCompletionTokens:     220,
		EmptyContentRetryPrompt: localize.T("hybrid.prompt.coverage_retry"),
	}, []openai.ChatCompletionMessage{
		{Role: "system", Content: coveragePrompt()},
		{Role: "user", Content: localize.T("hybrid.prompt.coverage_user", localize.Data{"Question": strings.TrimSpace(question), "Facts": sanitize(facts), "Evidence": strings.Join(evidenceSummary, "\n")})},
	})
	if err != nil {
		if errors.Is(err, errEmptyChatContent) || errors.Is(err, errReasoningOnlyOutput) {
			return CoverageReport{
				Covered:       false,
				Gaps:          []string{localize.T("hybrid.coverage.empty")},
				FollowUpQuery: question,
			}, nil
		}
		return CoverageReport{}, err
	}
	return parseCoverageReport(content), nil
}

func synthesizeAnswer(ctx context.Context, chat llm.ChatRequester, question, facts string, evidence []EvidenceBlock) (string, error) {
	contextBlocks := make([]string, 0, len(evidence))
	for i, block := range evidence {
		extra := block.Text
		if block.AroundText != "" && block.AroundText != block.Text {
			extra += localize.T("hybrid.prompt.local_window", localize.Data{"Text": block.AroundText})
		}
		contextBlocks = append(contextBlocks, fmt.Sprintf("[source %d] %s:%d-%d\n%s", i+1, block.SourcePath, block.StartLine, block.EndLine, sanitize(extra)))
	}

	answer, err := callChat(ctx, chat, chatCallOptions{
		Stage:                   "synthesize_answer",
		MaxCompletionTokens:     1400,
		EmptyContentRetryPrompt: localize.T("hybrid.prompt.answer_retry"),
	}, []openai.ChatCompletionMessage{
		{Role: "system", Content: synthesizePrompt()},
		{
			Role:    "user",
			Content: localize.T("hybrid.prompt.answer_user", localize.Data{"Question": strings.TrimSpace(question), "Facts": sanitize(facts), "Sources": strings.Join(contextBlocks, "\n\n---\n\n")}),
		},
	})
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(answer), nil
}

func appendSources(answer string, evidence []EvidenceBlock) string {
	citations := make([]retrieval.Citation, 0, len(evidence))
	for _, block := range evidence {
		citations = append(citations, retrieval.Citation{
			SourcePath: block.SourcePath,
			StartLine:  block.StartLine,
			EndLine:    block.EndLine,
		})
	}
	return retrieval.AppendSources(answer, citations)
}

func appendEvidence(base, extra []EvidenceBlock) []EvidenceBlock {
	seen := make(map[string]struct{}, len(base))
	for _, block := range base {
		seen[fmt.Sprintf("%s:%d-%d", block.SourcePath, block.StartLine, block.EndLine)] = struct{}{}
	}
	for _, block := range extra {
		key := fmt.Sprintf("%s:%d-%d", block.SourcePath, block.StartLine, block.EndLine)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		base = append(base, block)
	}
	return base
}

func filterRegionsBySegments(regions []Region, existing map[int]struct{}) []Region {
	filtered := make([]Region, 0, len(regions))
	for _, region := range regions {
		allSeen := true
		for _, segmentID := range region.SegmentIDs {
			if _, ok := existing[segmentID]; !ok {
				allSeen = false
				break
			}
		}
		if !allSeen {
			filtered = append(filtered, region)
		}
	}
	return filtered
}

func mergeFactBlocks(left, right string) string {
	combined := strings.TrimSpace(strings.Join([]string{strings.TrimSpace(left), strings.TrimSpace(right)}, "\n"))
	return normalizeFacts(combined)
}

func fallbackToMap(ctx context.Context, chat llm.ChatAutoContextRequester, source input.Source, opts Options, question string, sourcePath string, plan *verbose.Plan) (string, error) {
	slog.Debug("hybrid map fallback started", "chunk_length_mode", "auto_or_default")
	file, err := os.Open(sourcePath)
	if err != nil {
		return "", fmt.Errorf("failed to open fallback source: %w", err)
	}
	defer func() { _ = file.Close() }()

	return mapmode.Run(ctx, chat, input.Source{
		Reader:      file,
		Path:        source.Path,
		DisplayName: source.DisplayName,
	}, mapmode.Options{
		Concurrency:    1,
		ChunkLength:    10000,
		LengthExplicit: false,
	}, question, plan)
}

type chatCallOptions struct {
	Stage                   string
	MaxCompletionTokens     int
	EmptyContentRetryPrompt string
	AllowReasoningFallback  bool
}

func callChat(ctx context.Context, chat llm.ChatRequester, opts chatCallOptions, messages []openai.ChatCompletionMessage) (string, error) {
	return callChatAttempt(ctx, chat, opts, messages, true)
}

func callChatAttempt(ctx context.Context, chat llm.ChatRequester, opts chatCallOptions, messages []openai.ChatCompletionMessage, allowRetry bool) (string, error) {
	req := openai.ChatCompletionRequest{
		Messages: messages,
		ChatTemplateKwargs: localize.Data{
			"enable_thinking": false,
		},
	}
	if opts.MaxCompletionTokens > 0 {
		req.MaxCompletionTokens = opts.MaxCompletionTokens
	}
	slog.Debug("hybrid llm call started", "stage", opts.Stage, "message_count", len(messages), "max_completion_tokens", opts.MaxCompletionTokens)
	resp, metrics, err := chat.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return "", err
	}
	if resp == nil || len(resp.Choices) == 0 {
		return "", errors.New("empty chat response")
	}
	content, contentSource := extractMessageText(resp.Choices[0].Message, opts.AllowReasoningFallback)
	slog.Debug("hybrid llm call finished",
		"stage", opts.Stage,
		"prompt_tokens", metrics.PromptTokens,
		"completion_tokens", metrics.CompletionTokens,
		"total_tokens", metrics.TotalTokens,
		"empty_content", content == "",
		"content_source", contentSource,
		"has_reasoning_content", strings.TrimSpace(resp.Choices[0].Message.ReasoningContent) != "",
		"has_refusal", strings.TrimSpace(resp.Choices[0].Message.Refusal) != "",
	)
	if contentSource == "reasoning" && !opts.AllowReasoningFallback {
		if allowRetry && strings.TrimSpace(opts.EmptyContentRetryPrompt) != "" {
			retryMessages := append(slices.Clone(messages), openai.ChatCompletionMessage{
				Role:    "user",
				Content: strings.TrimSpace(opts.EmptyContentRetryPrompt),
			})
			slog.Warn("hybrid llm call returned reasoning-only output, retrying", "stage", opts.Stage)
			return callChatAttempt(ctx, chat, opts, retryMessages, false)
		}
		return "", errReasoningOnlyOutput
	}
	if content == "" {
		if allowRetry && strings.TrimSpace(opts.EmptyContentRetryPrompt) != "" {
			retryMessages := append(slices.Clone(messages), openai.ChatCompletionMessage{
				Role:    "user",
				Content: strings.TrimSpace(opts.EmptyContentRetryPrompt),
			})
			slog.Warn("hybrid llm call returned empty content, retrying", "stage", opts.Stage)
			return callChatAttempt(ctx, chat, opts, retryMessages, false)
		}
		return "", errEmptyChatContent
	}
	if looksLikeReasoningTrace(content) {
		if allowRetry && strings.TrimSpace(opts.EmptyContentRetryPrompt) != "" {
			retryMessages := append(slices.Clone(messages), openai.ChatCompletionMessage{
				Role:    "user",
				Content: strings.TrimSpace(opts.EmptyContentRetryPrompt) + localize.T("hybrid.prompt.no_reasoning_suffix"),
			})
			slog.Warn("hybrid llm call returned reasoning-like trace, retrying", "stage", opts.Stage)
			return callChatAttempt(ctx, chat, opts, retryMessages, false)
		}
		return "", errReasoningOnlyOutput
	}
	return content, nil
}

func parseCoverageReport(raw string) CoverageReport {
	report := CoverageReport{Covered: false}
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	section := ""
	parsedCovered := false
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		lower := strings.ToLower(trimmed)
		switch {
		case strings.HasPrefix(lower, "covered:"):
			report.Covered = strings.Contains(lower, "yes")
			parsedCovered = true
		case strings.HasPrefix(lower, "gaps:"):
			section = "gaps"
		case strings.HasPrefix(lower, "conflicts:"):
			section = "conflicts"
		case strings.HasPrefix(lower, "follow_up_query:"):
			report.FollowUpQuery = strings.TrimSpace(trimmed[len("follow_up_query:"):])
			section = ""
		case strings.HasPrefix(trimmed, "- "):
			switch section {
			case "gaps":
				report.Gaps = append(report.Gaps, strings.TrimSpace(strings.TrimPrefix(trimmed, "- ")))
			case "conflicts":
				report.Conflicts = append(report.Conflicts, strings.TrimSpace(strings.TrimPrefix(trimmed, "- ")))
			}
		}
	}
	if !parsedCovered && strings.TrimSpace(raw) != "" {
		report.Gaps = append(report.Gaps, "coverage response was unstructured")
	}
	return report
}

func normalizeFacts(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if strings.EqualFold(trimmed, "SKIP") {
		return "SKIP"
	}

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	seen := make(map[string]struct{}, len(lines))
	facts := make([]string, 0, len(lines))
	for _, line := range lines {
		candidate := strings.TrimSpace(line)
		candidate = bulletPrefixPattern.ReplaceAllString(candidate, "")
		candidate = strings.TrimSpace(candidate)
		if candidate == "" || isInstructionLike(candidate) {
			continue
		}
		key := strings.ToLower(candidate)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		facts = append(facts, candidate)
	}
	return strings.Join(facts, "\n")
}

func sanitize(raw string) string {
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	clean := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			continue
		}
		if isInstructionLike(trimmed) {
			continue
		}
		clean = append(clean, trimmed)
	}
	return strings.TrimSpace(strings.Join(clean, "\n"))
}

var bulletPrefixPattern = regexp.MustCompile(`^\s*(?:[-*•]|\d+[.)])\s*`)

func isInstructionLike(line string) bool {
	lower := strings.ToLower(strings.TrimSpace(line))
	if lower == "" {
		return false
	}
	for _, prefix := range []string{
		"ignore previous instructions",
		"system:",
		"assistant:",
		"user:",
		"developer:",
		"tool:",
	} {
		if strings.HasPrefix(lower, prefix) {
			return true
		}
	}
	return false
}

func normalizeSourceDisplayName(source input.Source) string {
	if strings.TrimSpace(source.DisplayName) != "" {
		displayName := strings.TrimSpace(source.DisplayName)
		if strings.Contains(displayName, string(os.PathSeparator)) {
			return filepath.Base(displayName)
		}
		return displayName
	}
	if strings.TrimSpace(source.Path) != "" {
		return strings.TrimSpace(filepath.Base(source.Path))
	}
	return "stdin"
}

func normalizeSourceReadPath(source input.Source) string {
	if strings.TrimSpace(source.Path) != "" {
		return strings.TrimSpace(source.Path)
	}
	if strings.TrimSpace(source.DisplayName) != "" {
		return strings.TrimSpace(source.DisplayName)
	}
	return ""
}

func hasFollowUpQuery(query string) bool {
	trimmed := strings.TrimSpace(query)
	return trimmed != "" && !strings.EqualFold(trimmed, "none")
}

func countFactLines(raw string) int {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" || strings.EqualFold(trimmed, "SKIP") {
		return 0
	}
	return strings.Count(strings.ReplaceAll(trimmed, "\r\n", "\n"), "\n") + 1
}

func roundDuration(duration time.Duration) float64 {
	return float64(duration.Round(time.Millisecond)) / float64(time.Second)
}

func extractMessageText(message openai.ChatCompletionMessage, allowReasoningFallback bool) (string, string) {
	content := strings.TrimSpace(message.Content)
	if content != "" {
		return content, "content"
	}
	if allowReasoningFallback {
		reasoning := strings.TrimSpace(message.ReasoningContent)
		if reasoning != "" {
			return reasoning, "reasoning"
		}
	}
	refusal := strings.TrimSpace(message.Refusal)
	if refusal != "" {
		return refusal, "refusal"
	}
	return "", "empty"
}

func looksLikeReasoningTrace(content string) bool {
	lower := strings.ToLower(strings.TrimSpace(content))
	if lower == "" {
		return false
	}
	markers := []string{
		"thinking process",
		"analyze the request",
		"chain-of-thought",
		"reasoning trace",
		"let me think",
		"the user is asking",
		"wait,",
	}
	for _, marker := range markers {
		if strings.Contains(lower, marker) {
			return true
		}
	}
	return false
}

func deriveFollowUpQuery(question string, coverage CoverageReport) string {
	if hasFollowUpQuery(coverage.FollowUpQuery) {
		return strings.TrimSpace(coverage.FollowUpQuery)
	}

	parts := make([]string, 0, len(coverage.Gaps)+1)
	for _, gap := range coverage.Gaps {
		gap = strings.TrimSpace(gap)
		if gap == "" || strings.EqualFold(gap, "none") {
			continue
		}
		parts = append(parts, gap)
		if len(parts) >= 3 {
			break
		}
	}
	if len(parts) == 0 {
		tokens := retrieval.Tokenize(question)
		if len(tokens) > 6 {
			tokens = tokens[:6]
		}
		return strings.Join(tokens, " ")
	}
	return strings.Join(parts, " ")
}

func maxFloat(left, right float64) float64 {
	if left > right {
		return left
	}
	return right
}

func hybridMapPrompt() string {
	return localize.T("hybrid.prompt.map_system")
}

func hybridReducePrompt() string {
	return localize.T("hybrid.prompt.reduce_system")
}

func coveragePrompt() string {
	return localize.T("hybrid.prompt.coverage_system")
}

func synthesizePrompt() string {
	return localize.T("hybrid.prompt.answer_system")
}
