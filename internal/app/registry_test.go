package app

import "testing"

func TestCommandRegistryIsComplete(t *testing.T) {
	seen := make(map[string]struct{})

	for _, spec := range commandSpecs() {
		if spec == nil {
			t.Fatal("commandSpecs() contains nil entry")
		}
		if spec.name == "" {
			t.Fatal("command spec name must not be empty")
		}
		if _, exists := seen[spec.name]; exists {
			t.Fatalf("duplicate command spec %q", spec.name)
		}
		seen[spec.name] = struct{}{}

		if spec.bind == nil {
			t.Fatalf("command %q bind must not be nil", spec.name)
		}
		if spec.execute == nil {
			t.Fatalf("command %q execute must not be nil", spec.name)
		}
		if spec.template.modeKey == "" {
			t.Fatalf("command %q template modeKey must not be empty", spec.name)
		}
		if len(spec.template.stages) == 0 {
			t.Fatalf("command %q must define at least one stage", spec.name)
		}

		template := TemplateFor(spec.name)
		if template.Mode == "" {
			t.Fatalf("TemplateFor(%q).Mode = empty", spec.name)
		}
		if len(template.Stages) != len(spec.template.stages) {
			t.Fatalf("TemplateFor(%q) stages = %d, want %d", spec.name, len(template.Stages), len(spec.template.stages))
		}
	}
}
