package hybrid

import (
	"context"
	"log/slog"
	"strings"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/ragcore"
	"github.com/borro/ragcli/internal/retrieval"
	"github.com/borro/ragcli/internal/tools"
	"github.com/borro/ragcli/internal/verbose"
)

func Run(
	ctx context.Context,
	client llm.ChatRequester,
	embedder llm.EmbeddingRequester,
	source input.FileBackedSource,
	opts Options,
	prompt string,
	plan *verbose.Plan,
) (string, error) {
	slog.Debug("hybrid processing started", "input_path", source.InputPath(), "file_count", source.FileCount())

	session, err := tools.PrepareSession(ctx, source, embedder, tools.Options{
		EnableRAG: true,
		RAG:       opts.Search,
	}, tools.SessionOptions{
		Prompt:                 prompt,
		AdditionalSystemPrompt: strings.TrimSpace(localize.T("hybrid.prompt.system")),
		AdditionalUserPrompt:   strings.TrimSpace(localize.T("hybrid.prompt.user")),
		FirstTurnAnswerPolicy:  tools.FirstTurnAnswerAllow,
	}, plan.Stage("index"))
	if err != nil {
		return "", err
	}

	retrievalMeter := plan.Stage("retrieval")
	retrievalMeter.Start(localize.T("progress.hybrid.seed_start"))
	retrievalMeter.Note(localize.T("progress.hybrid.seed_search"))
	seedBundle, err := ragcore.BuildSeedBundle(ctx, session, prompt, opts.Search)
	if err != nil {
		return "", err
	}
	if seedBundle.Weak {
		session.SetFirstTurnAnswerPolicy(tools.FirstTurnAnswerRequireToolUse)
	}
	retrievalMeter.Done(localize.T("progress.hybrid.seed_done"))

	result, err := session.Run(ctx, client, seedBundle.History, plan)
	if err != nil {
		return "", err
	}

	slog.Debug("hybrid processing finished", "evidence", len(result.Evidence), "seed_hits", len(seedBundle.Hits), "weak_seed", seedBundle.Weak)
	return retrieval.AppendEvidenceSources(result.Answer, result.Evidence), nil
}
