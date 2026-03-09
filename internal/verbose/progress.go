package verbose

type StageDef struct {
	Key   string
	Label string
	Slots int
}

type Plan struct {
	stages map[string]Meter
	total  int
}

func NewPlan(reporter Reporter, mode string, defs ...StageDef) *Plan {
	reporter = Safe(reporter)
	total := 0
	for _, def := range defs {
		total += max(def.Slots, 0)
	}
	reporter.ModeStarted(mode, total)

	stages := make(map[string]Meter, len(defs))
	progress := 0
	for _, def := range defs {
		slots := max(def.Slots, 0)
		label := def.Label
		if label == "" {
			label = def.Key
		}
		stages[def.Key] = Meter{
			reporter: reporter,
			stage:    label,
			start:    progress,
			slots:    slots,
		}
		progress += slots
	}

	return &Plan{stages: stages, total: total}
}
func (p *Plan) Stage(key string) Meter {
	if p == nil {
		return Meter{}
	}
	return p.stages[key]
}

func (p *Plan) Total() int {
	if p == nil {
		return 0
	}
	return p.total
}

func ScaleProgress(index int, total int, slots int) int {
	if total <= 0 || slots <= 0 || index <= 0 {
		return 0
	}
	scaled := (index*slots + total - 1) / total
	if scaled > slots {
		return slots
	}
	return scaled
}

type Meter struct {
	reporter Reporter
	stage    string
	start    int
	slots    int
}

func (m Meter) WithStage(stage string) Meter {
	m.stage = stage
	return m
}

func (m Meter) Slice(index int, total int) Meter {
	if m.reporter == nil {
		return Meter{}
	}
	start := m.start + ScaleProgress(index-1, total, m.slots)
	end := m.start + ScaleProgress(index, total, m.slots)
	return Meter{
		reporter: m.reporter,
		stage:    m.stage,
		start:    start,
		slots:    max(end-start, 0),
	}
}

func (m Meter) Note(detail string) {
	if m.reporter == nil {
		return
	}
	m.reporter.StageChanged(m.stage, detail, -1)
}

func (m Meter) Start(detail string) {
	if m.reporter == nil {
		return
	}
	m.reporter.StageChanged(m.stage, detail, m.start)
}

func (m Meter) Done(detail string) {
	if m.reporter == nil {
		return
	}
	m.reporter.StageChanged(m.stage, detail, m.start+m.slots)
}

func (m Meter) Report(index int, total int, detail string) {
	if m.reporter == nil {
		return
	}
	m.reporter.StageChanged(m.stage, detail, m.start+ScaleProgress(index, total, m.slots))
}
