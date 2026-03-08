package mapmode

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"strings"
	"unicode/utf8"

	openai "github.com/sashabaranov/go-openai"
)

type textChunk struct {
	Text          string
	TokenCount    int
	ByteCount     int
	RequestTokens int
}

const (
	approxMessageOverheadTokens = 12
	approxMapRequestReserve     = 64
	approxBudgetNumerator       = 3
	approxBudgetDenominator     = 5
)

func SplitByApproxTokens(ctx context.Context, r io.Reader, question string, maxTokens int) ([]textChunk, error) {
	if maxTokens < 1 {
		return nil, fmt.Errorf("max tokens must be >= 1")
	}
	safeMaxTokens := effectiveApproxTokenBudget(maxTokens)
	if safeMaxTokens < 1 {
		return nil, fmt.Errorf("max tokens must be >= 1")
	}
	requestOverhead := estimateMapRequestOverheadTokens(question)
	maxChunkTokens := safeMaxTokens - requestOverhead

	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	var chunks []textChunk
	var buf strings.Builder
	bufRuneCount := 0

	flush := func() {
		if buf.Len() == 0 {
			return
		}
		text := buf.String()
		tokenCount := approxTokenCountFromRunes(bufRuneCount)
		chunks = append(chunks, textChunk{
			Text:          text,
			TokenCount:    tokenCount,
			ByteCount:     len(text),
			RequestTokens: requestOverhead + tokenCount,
		})
		buf.Reset()
		bufRuneCount = 0
	}

	for scanner.Scan() {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		line := scanner.Text()
		lineRuneCount := utf8.RuneCountInString(line)
		if maxChunkTokens < 1 && line != "" {
			return nil, fmt.Errorf("map prompt and question exceed chunk token limit")
		}
		candidateRuneCount := lineRuneCount
		if buf.Len() > 0 {
			candidateRuneCount = bufRuneCount + 1 + lineRuneCount
		}

		if requestOverhead+approxTokenCountFromRunes(candidateRuneCount) <= safeMaxTokens {
			if buf.Len() > 0 {
				buf.WriteByte('\n')
			}
			buf.WriteString(line)
			bufRuneCount = candidateRuneCount
			continue
		}

		flush()

		if requestOverhead+approxTokenCountFromRunes(lineRuneCount) <= safeMaxTokens {
			buf.WriteString(line)
			bufRuneCount = lineRuneCount
			continue
		}

		lineChunks, err := splitOversizedApproxLine(ctx, line, maxChunkTokens, requestOverhead)
		if err != nil {
			return nil, err
		}
		chunks = append(chunks, lineChunks...)
	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}

	flush()
	return chunks, nil
}

func approxTokenCount(text string) int {
	if text == "" {
		return 0
	}

	return approxTokenCountFromRunes(utf8.RuneCountInString(text))
}

func approxTokenCountFromRunes(runes int) int {
	if runes <= 0 {
		return 0
	}

	return (runes + 2) / 3
}

func splitOversizedApproxLine(ctx context.Context, line string, maxChunkTokens int, requestOverhead int) ([]textChunk, error) {
	if line == "" {
		return nil, nil
	}
	if maxChunkTokens < 1 {
		return nil, fmt.Errorf("map prompt and question exceed chunk token limit")
	}

	maxRunes := maxChunkTokens * 3
	runeOffsets := make([]int, 0, utf8.RuneCountInString(line)+1)
	for idx := range line {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		runeOffsets = append(runeOffsets, idx)
	}
	runeOffsets = append(runeOffsets, len(line))

	chunkCount := (len(runeOffsets) - 1 + maxRunes - 1) / maxRunes
	chunks := make([]textChunk, 0, max(1, chunkCount))
	for startRune := 0; startRune < len(runeOffsets)-1; startRune += maxRunes {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		endRune := startRune + maxRunes
		if endRune > len(runeOffsets)-1 {
			endRune = len(runeOffsets) - 1
		}
		text := line[runeOffsets[startRune]:runeOffsets[endRune]]
		tokenCount := approxTokenCountFromRunes(endRune - startRune)
		chunks = append(chunks, textChunk{
			Text:          text,
			TokenCount:    tokenCount,
			ByteCount:     len(text),
			RequestTokens: requestOverhead + tokenCount,
		})
	}

	return chunks, nil
}

func estimateMapRequestTokens(question, chunk string) int {
	return estimateMapRequestOverheadTokens(question) + approxTokenCount(chunk)
}

func estimateReduceRequestTokens(question, facts string) int {
	return estimateReduceRequestOverheadTokens(question) + approxTokenCount(facts)
}

func effectiveApproxTokenBudget(maxTokens int) int {
	if maxTokens <= 0 {
		return 0
	}

	budget := (maxTokens * approxBudgetNumerator) / approxBudgetDenominator
	if budget < 1 {
		return 1
	}

	return budget
}

func estimateMapRequestOverheadTokens(question string) int {
	return estimateRequestOverheadTokens(question, mapMessages)
}

func estimateReduceRequestOverheadTokens(question string) int {
	return estimateRequestOverheadTokens(question, reduceMessages)
}

func estimateRequestOverheadTokens(question string, buildMessages func(string, string) []openai.ChatCompletionMessage) int {
	placeholder := "x"
	total := approxMapRequestReserve
	for _, message := range buildMessages(question, placeholder) {
		total += approxMessageOverheadTokens
		total += approxTokenCount(message.Role)
		total += approxTokenCount(message.Content)
	}
	return total - approxTokenCount(placeholder)
}
