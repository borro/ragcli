package app

import (
	"strings"
	"testing"
)

func TestApplyPromptArgCompatibilityRecognizesSharedValueFlags(t *testing.T) {
	for _, spec := range commandSpecs() {
		flags := append([]flagSpec{}, globalFlagSpecs()...)
		flags = append(flags, spec.flagSpecs...)

		for _, flag := range flags {
			if !flag.takesValue {
				continue
			}

			token := flag.names[0]
			got := applyPromptArgCompatibility([]string{spec.name, token, "VALUE", "question"})
			want := []string{spec.name, token, "VALUE", "--", "question"}
			if strings.Join(got, "\x00") != strings.Join(want, "\x00") {
				t.Fatalf("%s %s compatibility = %q, want %q", spec.name, token, got, want)
			}
		}
	}
}
