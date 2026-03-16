package mapmode

import (
	"context"
	"strings"

	"github.com/borro/ragcli/internal/input"
	"github.com/borro/ragcli/internal/llm"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
)

type conversationTurn struct {
	Question string
	Answer   string
}

type Conversation struct {
	client   llm.ChatAutoContextRequester
	source   input.SnapshotSource
	opts     Options
	turns    []conversationTurn
	baseline []conversationTurn
}

func StartConversation(
	ctx context.Context,
	client llm.ChatAutoContextRequester,
	source input.SnapshotSource,
	opts Options,
	prompt string,
	plan *verbose.Plan,
) (*Conversation, string, error) {
	answer, err := Run(ctx, client, source, opts, prompt, plan)
	if err != nil {
		return nil, "", err
	}
	turns := []conversationTurn{{Question: strings.TrimSpace(prompt), Answer: answer}}
	return &Conversation{
		client:   client,
		source:   source,
		opts:     opts,
		turns:    append([]conversationTurn(nil), turns...),
		baseline: append([]conversationTurn(nil), turns...),
	}, answer, nil
}

func (c *Conversation) Ask(ctx context.Context, prompt string, plan *verbose.Plan) (string, error) {
	question := strings.TrimSpace(prompt)
	answer, err := Run(ctx, c.client, c.source, c.opts, buildFollowUpPrompt(c.turns, question), plan)
	if err != nil {
		return "", err
	}
	c.turns = append(c.turns, conversationTurn{Question: question, Answer: answer})
	return answer, nil
}

func (c *Conversation) FollowUpPlan(reporter verbose.Reporter) *verbose.Plan {
	return verbose.NewPlan(reporter, localize.T("progress.mode.map"),
		verbose.StageDef{Key: "chunking", Label: localize.T("progress.stage.chunking"), Slots: 4},
		verbose.StageDef{Key: "chunks", Label: localize.T("progress.stage.chunks"), Slots: 9},
		verbose.StageDef{Key: "reduce", Label: localize.T("progress.stage.reduce"), Slots: 5},
		verbose.StageDef{Key: "verify", Label: localize.T("progress.stage.verify"), Slots: 2},
		verbose.StageDef{Key: "final", Label: localize.T("progress.stage.final"), Slots: 2},
	)
}

func (c *Conversation) Reset() {
	if c == nil {
		return
	}
	c.turns = append([]conversationTurn(nil), c.baseline...)
}

func buildFollowUpPrompt(turns []conversationTurn, question string) string {
	if len(turns) == 0 {
		return strings.TrimSpace(question)
	}
	parts := []string{strings.TrimSpace(localize.T("map.prompt.follow_up_intro"))}
	for index, turn := range turns {
		parts = append(parts, localize.T("map.prompt.follow_up_turn", localize.Data{
			"Index":    index + 1,
			"Question": strings.TrimSpace(turn.Question),
			"Answer":   strings.TrimSpace(turn.Answer),
		}))
	}
	parts = append(parts, localize.T("map.prompt.follow_up_question", localize.Data{
		"Question": strings.TrimSpace(question),
	}))
	return strings.Join(parts, "\n\n")
}
