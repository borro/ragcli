package testutil

import "sync"

type ProgressRecorder struct {
	mu        sync.Mutex
	currents  []int
	totals    []int
	progressN int
	total     int
}

func NewProgressRecorder(total int) *ProgressRecorder {
	return &ProgressRecorder{total: total}
}

func (r *ProgressRecorder) ModeStarted(string, int) {}

func (r *ProgressRecorder) StageChanged(_ string, _ string, current int) {
	if current < 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.progressN++
	r.currents = append(r.currents, current)
	if r.total > 0 {
		r.totals = append(r.totals, r.total)
	}
}

func (r *ProgressRecorder) Snapshot() (int, []int, []int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	currents := append([]int(nil), r.currents...)
	totals := append([]int(nil), r.totals...)
	return r.progressN, currents, totals
}
