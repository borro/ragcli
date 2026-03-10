package localize

import "testing"

func TestNormalize(t *testing.T) {
	tests := []struct {
		raw   string
		want  string
		valid bool
	}{
		{raw: "ru", want: RU, valid: true},
		{raw: "ru_RU.UTF-8", want: RU, valid: true},
		{raw: "en_US.UTF-8", want: EN, valid: true},
		{raw: "de_DE.UTF-8", want: EN, valid: false},
	}

	for _, tt := range tests {
		got, valid := Normalize(tt.raw)
		if got != tt.want || valid != tt.valid {
			t.Fatalf("Normalize(%q) = (%q, %v), want (%q, %v)", tt.raw, got, valid, tt.want, tt.valid)
		}
	}
}

func TestDetectPrecedence(t *testing.T) {
	origLookup := lookupEnv
	origDetect := detectSystemLocale
	t.Cleanup(func() {
		lookupEnv = origLookup
		detectSystemLocale = origDetect
	})

	lookupEnv = func(key string) (string, bool) {
		if key == "RAGCLI_LANG" {
			return "ru_RU.UTF-8", true
		}
		return "", false
	}
	detectSystemLocale = func() (string, error) {
		return "en_US.UTF-8", nil
	}

	if got, ok := Detect([]string{"--lang", "en"}); got != EN || !ok {
		t.Fatalf("Detect(flag) = (%q, %v), want (%q, true)", got, ok, EN)
	}
	if got, ok := Detect(nil); got != RU || !ok {
		t.Fatalf("Detect(env) = (%q, %v), want (%q, true)", got, ok, RU)
	}

	lookupEnv = func(string) (string, bool) { return "", false }
	if got, ok := Detect(nil); got != EN || !ok {
		t.Fatalf("Detect(system) = (%q, %v), want (%q, true)", got, ok, EN)
	}
}

func TestDetectInvalidOverride(t *testing.T) {
	origLookup := lookupEnv
	t.Cleanup(func() { lookupEnv = origLookup })

	if got, ok := Detect([]string{"--lang", "de"}); got != EN || ok {
		t.Fatalf("Detect(invalid flag) = (%q, %v), want (%q, false)", got, ok, EN)
	}

	lookupEnv = func(key string) (string, bool) {
		if key == "RAGCLI_LANG" {
			return "de_DE.UTF-8", true
		}
		return "", false
	}
	if got, ok := Detect(nil); got != EN || ok {
		t.Fatalf("Detect(invalid env) = (%q, %v), want (%q, false)", got, ok, EN)
	}
}
