package verbose

import (
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/borro/ragcli/internal/localize"
)

type Reporter interface {
	ModeStarted(mode string, totalSteps int)
	StageChanged(stage string, detail string, completedSteps int)
}

func New(writer io.Writer, enabled bool, debug bool) Reporter {
	if !enabled || debug || writer == nil {
		return nopReporter{}
	}
	return &stateReporter{writer: writer}
}

func Safe(reporter Reporter) Reporter {
	if reporter == nil {
		return nopReporter{}
	}
	return reporter
}

type nopReporter struct{}

func (nopReporter) ModeStarted(string, int)          {}
func (nopReporter) StageChanged(string, string, int) {}

type stateReporter struct {
	mu           sync.Mutex
	writer       io.Writer
	mode         string
	stage        string
	detail       string
	total        int
	cur          int
	lastRendered string
}

func (r *stateReporter) ModeStarted(mode string, totalSteps int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.mode = strings.TrimSpace(mode)
	r.total = max(totalSteps, 0)
	r.cur = 0
}

func (r *stateReporter) StageChanged(stage string, detail string, completedSteps int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.stage = strings.TrimSpace(stage)
	r.detail = strings.TrimSpace(detail)
	if completedSteps >= 0 {
		r.cur = completedSteps
	}
	r.renderLocked()
}

func (r *stateReporter) renderLocked() {
	if r.writer == nil {
		return
	}
	content := renderLine(r.mode, r.stage, r.detail, r.cur, r.total)
	if content == r.lastRendered {
		return
	}
	r.lastRendered = content
	_, _ = fmt.Fprintln(r.writer, content)
}

func renderLine(mode string, stage string, detail string, current int, total int) string {
	return fmt.Sprintf(
		"%s %5s: %s (%s)",
		defaultText(mode, localize.T("verbose.fallback.mode")),
		formatPercentField(current, total),
		defaultText(stage, localize.T("verbose.fallback.stage")),
		defaultText(detail, localize.T("verbose.fallback.detail")),
	)
}

func formatPercentField(current int, total int) string {
	return fmt.Sprintf("(%s)", renderPercent(current, total))
}

func renderPercent(current int, total int) string {
	if total <= 0 {
		total = 1
	}
	cur := min(max(current, 0), total)
	return fmt.Sprintf("%d%%", int((float64(cur)/float64(total))*100.0+0.5))
}

func defaultText(value string, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return strings.TrimSpace(value)
}
