package verbose

import (
	"testing"

	"github.com/borro/ragcli/internal/testutil"
)

func TestScaleProgressClampsBoundaries(t *testing.T) {
	cases := []struct {
		name  string
		index int
		total int
		slots int
		want  int
	}{
		{name: "zero total", index: 1, total: 0, slots: 4, want: 0},
		{name: "negative index", index: -1, total: 10, slots: 4, want: 0},
		{name: "zero slots", index: 2, total: 10, slots: 0, want: 0},
		{name: "rounded up", index: 1, total: 10, slots: 4, want: 1},
		{name: "overflow", index: 99, total: 10, slots: 4, want: 4},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := ScaleProgress(tc.index, tc.total, tc.slots); got != tc.want {
				t.Fatalf("ScaleProgress(%d, %d, %d) = %d, want %d", tc.index, tc.total, tc.slots, got, tc.want)
			}
		})
	}
}

func TestMeterReportClampsValues(t *testing.T) {
	recorder := testutil.NewProgressRecorder(10)
	plan := NewPlan(recorder, "rag", StageDef{Key: "index", Label: "index", Slots: 10})
	meter := plan.Stage("index").Slice(1, 1)
	meter.start = 2
	meter.slots = 4

	meter.Report(-5, 10, "negative")
	meter.Report(15, 10, "overflow")

	_, currents, _ := recorder.Snapshot()
	if len(currents) != 2 {
		t.Fatalf("StageChanged() calls = %d, want 2", len(currents))
	}
	if currents[0] != 2 {
		t.Fatalf("first current = %d, want 2", currents[0])
	}
	if currents[1] != 6 {
		t.Fatalf("second current = %d, want 6", currents[1])
	}
}

func TestSafeNilReporterReturnsNopReporter(t *testing.T) {
	reporter := Safe(nil)
	reporter.ModeStarted("rag", 10)
	reporter.StageChanged("index", "done", 10)
}

func TestMeterSliceAllocatesSubrange(t *testing.T) {
	recorder := testutil.NewProgressRecorder(8)
	plan := NewPlan(recorder, "rag",
		StageDef{Key: "prepare", Label: "prepare", Slots: 2},
		StageDef{Key: "loop", Label: "loop", Slots: 6},
	)

	slice := plan.Stage("loop").Slice(2, 3)
	slice.Start("working")
	slice.Done("done")

	_, currents, _ := recorder.Snapshot()
	if len(currents) != 2 {
		t.Fatalf("currents len = %d, want 2", len(currents))
	}
	if currents[0] != 4 {
		t.Fatalf("slice start = %d, want 4", currents[0])
	}
	if currents[1] != 6 {
		t.Fatalf("slice done = %d, want 6", currents[1])
	}
}
