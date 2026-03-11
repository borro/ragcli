package app

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strconv"
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

func TestRunHybridVerboseProgressIsMonotonic(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/embeddings" {
			http.NotFound(w, r)
			return
		}
		http.Error(w, "embed fail", http.StatusInternalServerError)
	}))
	defer server.Close()

	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{
		"--verbose",
		"--api-url", server.URL + "/v1",
		"hybrid",
		"--embedding-model", "embed",
		"--hybrid-fallback", "fail",
		"what changed?",
	}, &stdout, &stderr, bytes.NewBufferString("retry policy is enabled\n"), "test-version")
	if exitCode != 1 {
		t.Fatalf("Run(hybrid --verbose) exit code = %d, want 1", exitCode)
	}

	percents := extractProgressPercents(stderr.String())
	if len(percents) < 4 {
		t.Fatalf("stderr progress = %q, want multiple progress updates", stderr.String())
	}
	for i := 1; i < len(percents); i++ {
		if percents[i] < percents[i-1] {
			t.Fatalf("progress percents = %v, want monotonic non-decreasing sequence", percents)
		}
	}
}

func extractProgressPercents(output string) []int {
	matches := regexp.MustCompile(`\((\d+)%\)`).FindAllStringSubmatch(output, -1)
	percents := make([]int, 0, len(matches))
	for _, match := range matches {
		value, err := strconv.Atoi(match[1])
		if err != nil {
			continue
		}
		percents = append(percents, value)
	}
	return percents
}
