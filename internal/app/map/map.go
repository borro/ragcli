package mapmode

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"regexp"
	"sort"
	"strings"
	"sync"
	"time"
	"unicode/utf8"

	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
	openai "github.com/sashabaranov/go-openai"
)

type Options struct {
	InputPath      string
	Concurrency    int
	ChunkLength    int
	LengthExplicit bool
}

const defaultChunkLengthFallback = 10000

const separator = "\n\n---\n\n"

type llmCallContext struct {
	Stage         string
	ChunkIndex    int
	ChunkCount    int
	ChunkBytes    int
	ChunkTokens   int
	RequestTokens int
	BatchIndex    int
	BatchCount    int
	InputItems    int
}

type pipelineStats struct {
	LLMCalls         int
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	MapSuccesses     int
	MapSkips         int
	MapErrors        int
	MapFactLinesRaw  int
	MapFactLinesKept int
	ReduceIterations int
}

//
// PROMPTS
//

func sharedAntiInjectionPolicy() string {
	return localize.T("map.prompt.shared_policy")
}

func mapPrompt() string {
	return localize.T("map.prompt.map", localize.Data{"Policy": sharedAntiInjectionPolicy()})
}

func reducePrompt() string {
	return localize.T("map.prompt.reduce", localize.Data{"Policy": sharedAntiInjectionPolicy()})
}

func finalPrompt() string {
	return localize.T("map.prompt.final", localize.Data{"Policy": sharedAntiInjectionPolicy()})
}

func critiquePrompt() string {
	return localize.T("map.prompt.critique", localize.Data{"Policy": sharedAntiInjectionPolicy()})
}

func refinePrompt() string {
	return localize.T("map.prompt.refine", localize.Data{"Policy": sharedAntiInjectionPolicy()})
}

//
// LLM helper
//

func callLLM(ctx context.Context, client llm.ChatRequester, messages []openai.ChatCompletionMessage, callCtx llmCallContext, stats *pipelineStats) (string, error) {
	req := openai.ChatCompletionRequest{
		Messages: messages,
	}

	resp, metrics, err := client.SendRequestWithMetrics(ctx, req)
	if err != nil {
		return "", err
	}

	if stats != nil {
		stats.LLMCalls++
		stats.PromptTokens += metrics.PromptTokens
		stats.CompletionTokens += metrics.CompletionTokens
		stats.TotalTokens += metrics.TotalTokens
	}

	logArgs := []any{"stage", callCtx.Stage}
	switch callCtx.Stage {
	case "map_chunk":
		logArgs = append(logArgs,
			"chunk_index", callCtx.ChunkIndex,
			"chunk_count", callCtx.ChunkCount,
			"chunk_tokens", callCtx.ChunkTokens,
			"request_tokens", callCtx.RequestTokens,
			"chunk_bytes", callCtx.ChunkBytes,
		)
	case "reduce_batch":
		logArgs = append(logArgs,
			"batch_index", callCtx.BatchIndex,
			"batch_count", callCtx.BatchCount,
			"input_items", callCtx.InputItems,
		)
	}
	slog.Debug("map-reduce llm call completed", logArgs...)

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("empty response")
	}

	return strings.TrimSpace(resp.Choices[0].Message.Content), nil
}

//
// message builders
//

func mapMessages(question, chunk string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeChunk := sanitizeUntrustedBlock(chunk)
	return []openai.ChatCompletionMessage{
		{Role: "system", Content: mapPrompt()},
		{
			Role: "user",
			Content: buildUserMessage(
				safeQuestion,
				formatUntrustedBlock(localize.T("map.label.text"), safeChunk),
			),
		},
	}
}

func reduceMessages(question, facts string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeFacts := prepareReduceFacts(facts)
	return []openai.ChatCompletionMessage{
		{Role: "system", Content: reducePrompt()},
		{
			Role: "user",
			Content: buildUserMessage(
				safeQuestion,
				formatUntrustedBlock(localize.T("map.label.facts"), safeFacts),
			),
		},
	}
}

func finalMessages(question, facts string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeFacts := normalizeFactOutput(facts, 10)
	return []openai.ChatCompletionMessage{
		{Role: "system", Content: finalPrompt()},
		{
			Role: "user",
			Content: buildUserMessage(
				safeQuestion,
				formatUntrustedBlock(localize.T("map.label.facts"), safeFacts),
			),
		},
	}
}

func critiqueMessages(question, facts, answer string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeFacts := normalizeFactOutput(facts, 10)
	safeAnswer := sanitizeUntrustedBlock(answer)
	return []openai.ChatCompletionMessage{
		{Role: "system", Content: critiquePrompt()},
		{
			Role: "user",
			Content: buildUserMessage(
				safeQuestion,
				formatUntrustedBlock(localize.T("map.label.facts"), safeFacts),
				formatUntrustedBlock(localize.T("map.label.draft"), safeAnswer),
			),
		},
	}
}

func refineMessages(question, facts, answer, critique string) []openai.ChatCompletionMessage {
	safeQuestion := strings.TrimSpace(question)
	safeFacts := normalizeFactOutput(facts, 10)
	safeAnswer := sanitizeUntrustedBlock(answer)
	safeCritique := sanitizeUntrustedBlock(critique)
	return []openai.ChatCompletionMessage{
		{Role: "system", Content: refinePrompt()},
		{
			Role: "user",
			Content: buildUserMessage(
				safeQuestion,
				formatUntrustedBlock(localize.T("map.label.facts"), safeFacts),
				formatUntrustedBlock(localize.T("map.label.draft"), safeAnswer),
				formatUntrustedBlock(localize.T("map.label.remarks"), safeCritique),
			),
		},
	}
}

func buildUserMessage(question string, blocks ...string) string {
	parts := make([]string, 0, len(blocks)+1)
	parts = append(parts, fmt.Sprintf("Вопрос:\n%s", question))
	parts = append(parts, blocks...)
	return strings.Join(parts, "\n\n")
}

func formatUntrustedBlock(label, content string) string {
	if content == "" {
		content = "(пусто)"
	}
	return fmt.Sprintf("%s:\n```\n%s\n```", label, content)
}

func sanitizeUntrustedBlock(raw string) string {
	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	cleaned := make([]string, 0, len(lines))
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed == "" {
			if len(cleaned) == 0 || cleaned[len(cleaned)-1] == "" {
				continue
			}
			cleaned = append(cleaned, "")
			continue
		}
		if isInstructionLikeLine(trimmed) {
			continue
		}
		cleaned = append(cleaned, trimmed)
	}

	return strings.TrimSpace(strings.Join(cleaned, "\n"))
}

func normalizeFactOutput(raw string, maxFacts int) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if strings.EqualFold(trimmed, "SKIP") {
		return "SKIP"
	}

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	capHint := len(lines)
	if maxFacts > 0 {
		capHint = min(len(lines), maxFacts)
	}
	facts := make([]string, 0, capHint)
	for _, line := range lines {
		candidate := normalizeFactLine(line)
		if candidate == "" {
			continue
		}
		facts = append(facts, candidate)
		if maxFacts > 0 && len(facts) >= maxFacts {
			break
		}
	}

	return strings.Join(facts, "\n")
}

func prepareReduceFacts(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if strings.EqualFold(trimmed, "SKIP") {
		return "SKIP"
	}

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	facts := make([]string, 0, len(lines))
	for _, line := range lines {
		candidate := normalizeReduceFactLine(line)
		if candidate == "" {
			continue
		}
		facts = append(facts, candidate)
	}

	return strings.Join(facts, "\n")
}

const maxMapFactsPerChunk = 7

var mapLeadInPrefixes = []string{
	"в тексте",
	"в материале",
	"в документе",
	"автор считает",
	"автор отмечает",
	"автор пишет",
	"источник показывает",
	"источник указывает",
}

func normalizeMapOutput(raw string) string {
	trimmed := strings.TrimSpace(raw)
	if trimmed == "" {
		return ""
	}
	if strings.EqualFold(trimmed, "SKIP") {
		return "SKIP"
	}

	lines := strings.Split(strings.ReplaceAll(raw, "\r\n", "\n"), "\n")
	seen := make(map[string]struct{}, len(lines))
	candidates := make([]string, 0, len(lines))
	for _, line := range lines {
		candidate := normalizeFactLine(line)
		if candidate == "" {
			continue
		}
		key := strings.ToLower(candidate)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}
		candidates = append(candidates, candidate)
	}

	if len(candidates) <= maxMapFactsPerChunk {
		return strings.Join(candidates, "\n")
	}

	sort.SliceStable(candidates, func(i, j int) bool {
		left, right := candidates[i], candidates[j]
		leftPenalty, rightPenalty := mapFactPenalty(left), mapFactPenalty(right)
		if leftPenalty != rightPenalty {
			return leftPenalty < rightPenalty
		}
		leftLen, rightLen := utf8.RuneCountInString(left), utf8.RuneCountInString(right)
		if leftLen != rightLen {
			return leftLen < rightLen
		}
		return left < right
	})

	return strings.Join(candidates[:maxMapFactsPerChunk], "\n")
}

func mapFactPenalty(line string) int {
	normalized := strings.ToLower(strings.TrimSpace(line))
	penalty := 0
	for _, prefix := range mapLeadInPrefixes {
		if strings.HasPrefix(normalized, prefix) {
			penalty += 100
			break
		}
	}
	if strings.Contains(normalized, "считает, что") || strings.Contains(normalized, "отмечает, что") || strings.Contains(normalized, "указывает, что") {
		penalty += 25
	}
	return penalty
}

var bulletPrefixPattern = regexp.MustCompile(`^\s*(?:[-*•]|\d+[.)])\s*`)

func normalizeFactLine(line string) string {
	trimmed := strings.TrimSpace(line)
	if trimmed == "" {
		return ""
	}

	trimmed = bulletPrefixPattern.ReplaceAllString(trimmed, "")
	trimmed = strings.TrimSpace(trimmed)
	if trimmed == "" || isInstructionLikeLine(trimmed) {
		return ""
	}

	return trimmed
}

func normalizeReduceFactLine(line string) string {
	trimmed := normalizeFactLine(line)
	if trimmed == "" {
		return ""
	}
	if trimmed == "---" {
		return ""
	}

	return trimmed
}

func countPreparedFactLines(raw string, prepare func(string) string) int {
	prepared := prepare(raw)
	if prepared == "" || strings.EqualFold(prepared, "SKIP") {
		return 0
	}

	return strings.Count(prepared, "\n") + 1
}

func countFactLines(raw string) int {
	if strings.TrimSpace(raw) == "" || strings.EqualFold(strings.TrimSpace(raw), "SKIP") {
		return 0
	}
	return strings.Count(strings.ReplaceAll(raw, "\r\n", "\n"), "\n") + 1
}

func isInstructionLikeLine(line string) bool {
	normalized := strings.ToLower(strings.TrimSpace(line))
	if normalized == "" {
		return false
	}

	prefixes := []string{
		"ignore previous instructions",
		"ignore all previous instructions",
		"disregard previous instructions",
		"follow these instructions",
		"system:",
		"assistant:",
		"user:",
		"tool:",
		"developer:",
	}
	for _, prefix := range prefixes {
		if strings.HasPrefix(normalized, prefix) {
			return true
		}
	}

	return false
}

func runMapParallel(ctx context.Context, client llm.ChatRequester, chunks []textChunk, question string, workers int, stats *pipelineStats, chunkMeter verbose.Meter) ([]string, error) {
	slog.Debug("map phase started", "chunk_count", len(chunks), "workers", workers)

	type mapJob struct {
		index int
		chunk textChunk
	}
	type mapResult struct {
		output       string
		skipped      bool
		err          error
		rawFactLines int
		keptFactLine int
	}

	jobs := make(chan mapJob)
	resultMeta := make(chan mapResult, len(chunks))

	var wg sync.WaitGroup

	for i := 0; i < workers; i++ {

		wg.Add(1)

		go func() {

			defer wg.Done()

			for job := range jobs {
				res, err := callLLM(ctx, client, mapMessages(question, job.chunk.Text), llmCallContext{
					Stage:         "map_chunk",
					ChunkIndex:    job.index + 1,
					ChunkCount:    len(chunks),
					ChunkBytes:    job.chunk.ByteCount,
					ChunkTokens:   job.chunk.TokenCount,
					RequestTokens: job.chunk.RequestTokens,
				}, stats)
				if err != nil {
					slog.Debug("map chunk failed", "chunk_index", job.index+1, "chunk_count", len(chunks), "error", err)
					resultMeta <- mapResult{err: err}
					continue
				}

				rawFactLines := countFactLines(res)
				normalized := normalizeMapOutput(res)
				keptFactLines := countPreparedFactLines(normalized, func(raw string) string { return raw })
				slog.Debug("map chunk normalized",
					"chunk_index", job.index+1,
					"chunk_count", len(chunks),
					"fact_lines_raw", rawFactLines,
					"fact_lines_kept", keptFactLines,
				)

				if normalized != "" && normalized != "SKIP" {
					resultMeta <- mapResult{output: normalized, rawFactLines: rawFactLines, keptFactLine: keptFactLines}
				} else {
					resultMeta <- mapResult{skipped: true, rawFactLines: rawFactLines}
				}
			}

		}()
	}

	go func() {

		for i, c := range chunks {
			jobs <- mapJob{index: i, chunk: c}
		}

		close(jobs)

		wg.Wait()
		close(resultMeta)
	}()

	var collected []string

	completed := 0
	for result := range resultMeta {
		completed++
		switch {
		case result.err != nil:
			stats.MapErrors++
		case result.skipped:
			stats.MapSkips++
		default:
			stats.MapSuccesses++
			if result.output != "" {
				collected = append(collected, result.output)
			}
		}
		stats.MapFactLinesRaw += result.rawFactLines
		stats.MapFactLinesKept += result.keptFactLine
		chunkMeter.Report(completed, len(chunks), fmt.Sprintf("обработано %d/%d chunks", completed, len(chunks)))
	}

	slog.Debug("map phase finished",
		"chunk_count", len(chunks),
		"workers", workers,
		"results_count", len(collected),
		"fact_line_count", countPreparedFactLines(strings.Join(collected, "\n"), prepareReduceFacts),
		"skip_count", stats.MapSkips,
		"error_count", stats.MapErrors,
	)

	return collected, nil
}

func batchReduceInputs(ctx context.Context, items []string, question string, maxTokens int) ([][]string, error) {
	if len(items) == 0 {
		return nil, nil
	}

	safeMaxTokens := effectiveReduceApproxTokenBudget(maxTokens)
	normalizedItems := make([]string, 0, len(items))
	for _, item := range items {
		pieces, err := splitReduceItemByApproxTokens(ctx, item, question, safeMaxTokens)
		if err != nil {
			return nil, err
		}
		normalizedItems = append(normalizedItems, pieces...)
	}

	var batches [][]string
	var current []string
	currentText := ""

	for _, item := range normalizedItems {
		if currentText == "" {
			current = []string{item}
			currentText = item
			continue
		}

		candidate := currentText + separator + item
		if estimateReduceRequestTokens(question, candidate) <= safeMaxTokens {
			current = append(current, item)
			currentText = candidate
			continue
		}

		batches = append(batches, current)
		current = []string{item}
		currentText = item
	}

	if len(current) > 0 {
		batches = append(batches, current)
	}

	return batches, nil
}

func splitReduceItemByApproxTokens(ctx context.Context, item string, question string, safeMaxTokens int) ([]string, error) {
	normalized := prepareReduceFacts(item)
	if normalized == "" {
		return nil, nil
	}
	if estimateReduceRequestTokens(question, normalized) <= safeMaxTokens {
		return []string{normalized}, nil
	}

	requestOverhead := estimateReduceRequestOverheadTokens(question)
	maxChunkTokens := safeMaxTokens - requestOverhead
	if maxChunkTokens < 1 {
		return []string{normalized}, nil
	}

	lines := strings.Split(strings.ReplaceAll(normalized, "\r\n", "\n"), "\n")
	var pieces []string
	var current strings.Builder
	currentRuneCount := 0

	flush := func() {
		if current.Len() == 0 {
			return
		}
		pieces = append(pieces, current.String())
		current.Reset()
		currentRuneCount = 0
	}

	for _, line := range lines {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		lineRuneCount := utf8.RuneCountInString(line)
		candidateRuneCount := lineRuneCount
		if current.Len() > 0 {
			candidateRuneCount = currentRuneCount + 1 + lineRuneCount
		}

		if requestOverhead+approxTokenCountFromRunes(candidateRuneCount) <= safeMaxTokens {
			if current.Len() > 0 {
				current.WriteByte('\n')
			}
			current.WriteString(line)
			currentRuneCount = candidateRuneCount
			continue
		}

		flush()

		if estimateReduceRequestTokens(question, line) <= safeMaxTokens {
			current.WriteString(line)
			currentRuneCount = utf8.RuneCountInString(line)
			continue
		}

		lineChunks, err := splitOversizedApproxLine(ctx, line, maxChunkTokens, requestOverhead)
		if err != nil {
			return nil, err
		}
		for _, chunk := range lineChunks {
			if chunk.Text != "" {
				pieces = append(pieces, chunk.Text)
			}
		}
	}

	flush()
	return pieces, nil
}

//
// REDUCE
//

func fanInReduce(ctx context.Context, client llm.ChatRequester, results []string, question string, opts Options, stats *pipelineStats, reduceMeter verbose.Meter) (string, error) {
	iteration := 0
	for len(results) > 1 {
		iteration++
		stats.ReduceIterations = iteration

		batches, err := batchReduceInputs(ctx, results, question, opts.ChunkLength)
		if err != nil {
			return "", err
		}
		reduceMeter.Note(localize.T("progress.map.reduce_iteration", localize.Data{"Iteration": iteration, "Batches": len(batches)}))
		slog.Debug("reduce iteration started",
			"iteration", iteration,
			"input_items", len(results),
			"input_fact_lines", countPreparedFactLines(strings.Join(results, "\n"), prepareReduceFacts),
			"batch_count", len(batches),
			"max_request_tokens", opts.ChunkLength,
		)

		var next []string
		inputItems := len(results)
		inputFactLines := countPreparedFactLines(strings.Join(results, "\n"), prepareReduceFacts)

		for batchIndex, batch := range batches {
			input := strings.Join(batch, separator)
			rawFactLines := countPreparedFactLines(input, func(raw string) string { return raw })
			preparedInput := prepareReduceFacts(input)
			preparedFactLines := countPreparedFactLines(preparedInput, func(raw string) string { return raw })
			reduceMeter.Note(localize.T("progress.map.send_batch", localize.Data{"Index": batchIndex + 1, "Total": len(batches), "Lines": preparedFactLines}))
			slog.Debug("reduce batch prepared",
				"iteration", iteration,
				"batch_index", batchIndex+1,
				"batch_count", len(batches),
				"input_items", len(batch),
				"fact_lines_before_prepare", rawFactLines,
				"fact_lines_after_prepare", preparedFactLines,
				"request_tokens_estimate", estimateReduceRequestTokens(question, preparedInput),
			)

			summary, err := callLLM(ctx, client, reduceMessages(question, preparedInput), llmCallContext{
				Stage:      "reduce_batch",
				BatchIndex: batchIndex + 1,
				BatchCount: len(batches),
				InputItems: len(batch),
			}, stats)
			if err != nil {
				return "", err
			}
			reduceMeter.Report(batchIndex+1, len(batches), localize.T("progress.map.batch_done", localize.Data{"Index": batchIndex + 1, "Total": len(batches)}))

			next = append(next, summary)
		}

		results = next
		slog.Debug("reduce iteration finished",
			"iteration", iteration,
			"input_items", inputItems,
			"input_fact_lines", inputFactLines,
			"output_items", len(results),
			"output_fact_lines", countPreparedFactLines(strings.Join(next, "\n"), prepareReduceFacts),
		)
	}

	return results[0], nil
}

func resolveChunkLength(ctx context.Context, client llm.AutoContextLengthResolver, opts Options) int {
	if opts.LengthExplicit {
		return opts.ChunkLength
	}

	length, err := client.ResolveAutoContextLength(ctx)
	if err != nil || length < 1 {
		length = defaultChunkLengthFallback
	}

	return length
}

//
// SELF REFINEMENT
//

func selfRefine(ctx context.Context, client llm.ChatRequester, question, facts, answer string, stats *pipelineStats, verifyMeter verbose.Meter) (string, error) {
	verifyMeter.Start(localize.T("progress.map.verify_start"))
	slog.Debug("self-refine critique started")

	critique, err := callLLM(ctx, client, critiqueMessages(question, facts, answer), llmCallContext{Stage: "self_refine_critique"}, stats)
	if err != nil {
		return "", err
	}

	if strings.Contains(critique, "NONE") {
		slog.Debug("self-refine skipped", "reason", "critique_none")
		verifyMeter.Done(localize.T("progress.map.verify_skip"))
		return answer, nil
	}

	slog.Debug("self-refine finalize started")
	verifyMeter.Note(localize.T("progress.map.verify_refine"))
	refined, err := callLLM(ctx, client, refineMessages(question, facts, answer, critique), llmCallContext{Stage: "self_refine_finalize"}, stats)
	if err != nil {
		return "", err
	}
	verifyMeter.Done(localize.T("progress.map.verify_done"))
	return refined, nil
}

//
// PIPELINE
//

func Run(ctx context.Context, client llm.ChatAutoContextRequester, input io.Reader, opts Options, question string, plan *verbose.Plan) (string, error) {
	startedAt := time.Now()
	inputMode := "stdin"
	if opts.InputPath != "" {
		inputMode = "file"
	}
	stats := &pipelineStats{}
	if plan == nil {
		plan = verbose.NewPlan(nil, localize.T("progress.mode.map"),
			verbose.StageDef{Key: "prepare", Label: localize.T("progress.stage.prepare"), Slots: 2},
			verbose.StageDef{Key: "chunking", Label: localize.T("progress.stage.chunking"), Slots: 4},
			verbose.StageDef{Key: "chunks", Label: localize.T("progress.stage.chunks"), Slots: 9},
			verbose.StageDef{Key: "reduce", Label: localize.T("progress.stage.reduce"), Slots: 5},
			verbose.StageDef{Key: "verify", Label: localize.T("progress.stage.verify"), Slots: 2},
			verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 2},
		)
	}
	chunkingMeter := plan.Stage("chunking")
	chunkMeter := plan.Stage("chunks")
	reduceMeter := plan.Stage("reduce")
	verifyMeter := plan.Stage("verify")
	finalMeter := plan.Stage("final")
	chunkingMeter.Start(localize.T("progress.map.chunking_start"))
	opts.ChunkLength = resolveChunkLength(ctx, client, opts)

	slog.Debug("map-reduce pipeline started",
		"input_mode", inputMode,
		"chunk_length", opts.ChunkLength,
		"chunk_length_unit", "approx_tokens",
		"concurrency", opts.Concurrency,
		"map_token_budget", effectiveMapApproxTokenBudget(opts.ChunkLength),
		"reduce_token_budget", effectiveReduceApproxTokenBudget(opts.ChunkLength),
	)

	chunks, err := SplitByApproxTokens(ctx, input, question, opts.ChunkLength)
	if err != nil {
		return "", err
	}
	slog.Debug("split into chunks", "chunk_count", len(chunks))
	chunkingMeter.Done(localize.T("progress.map.chunking_done", localize.Data{"Count": len(chunks)}))

	if len(chunks) == 0 {
		slog.Debug("map-reduce pipeline finished", "reason", "empty_input", "duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second))
		return localize.T("map.answer.empty_file"), nil
	}

	mapResults, err := runMapParallel(ctx, client, chunks, question, opts.Concurrency, stats, chunkMeter)
	if err != nil {
		return "", err
	}

	if len(mapResults) == 0 {
		slog.Debug("map-reduce pipeline finished",
			"reason", "no_map_results",
			"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
			"llm_calls", stats.LLMCalls,
			"prompt_tokens", stats.PromptTokens,
			"completion_tokens", stats.CompletionTokens,
			"total_tokens", stats.TotalTokens,
			"map_successes", stats.MapSuccesses,
			"map_skips", stats.MapSkips,
			"map_errors", stats.MapErrors,
			"map_fact_lines_raw", stats.MapFactLinesRaw,
			"map_fact_lines_kept", stats.MapFactLinesKept,
			"reduce_iterations", stats.ReduceIterations,
		)
		return localize.T("map.answer.no_info"), nil
	}

	slog.Debug("reduce phase started", "input_items", len(mapResults))
	reduceMeter.Start(localize.T("progress.map.reduce_start"))
	facts, err := fanInReduce(ctx, client, mapResults, question, opts, stats, reduceMeter)
	if err != nil {
		return "", err
	}
	reduceMeter.Done(localize.T("progress.map.reduce_done"))

	slog.Debug("final answer generation started")
	finalMeter.Start(localize.T("progress.map.final_start"))
	answer, err := callLLM(ctx, client, finalMessages(question, facts), llmCallContext{Stage: "final_answer"}, stats)
	if err != nil {
		return "", err
	}
	finalMeter.Note(localize.T("progress.map.final_draft"))

	result, err := selfRefine(ctx, client, question, facts, answer, stats, verifyMeter)
	if err != nil {
		return "", err
	}
	finalMeter.Done(localize.T("progress.map.final_done"))

	slog.Debug("map-reduce pipeline finished",
		"reason", "success",
		"duration", float64(time.Since(startedAt).Round(time.Millisecond))/float64(time.Second),
		"llm_calls", stats.LLMCalls,
		"prompt_tokens", stats.PromptTokens,
		"completion_tokens", stats.CompletionTokens,
		"total_tokens", stats.TotalTokens,
		"map_successes", stats.MapSuccesses,
		"map_skips", stats.MapSkips,
		"map_errors", stats.MapErrors,
		"map_fact_lines_raw", stats.MapFactLinesRaw,
		"map_fact_lines_kept", stats.MapFactLinesKept,
		"reduce_iterations", stats.ReduceIterations,
	)
	return result, nil
}
