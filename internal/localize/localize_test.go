package localize

import (
	"fmt"
	"io/fs"
	"strings"
	"sync"
	"testing"
	"testing/fstest"

	"github.com/nicksnyder/go-i18n/v2/i18n"
	"golang.org/x/text/language"
)

func TestNormalize(t *testing.T) {
	tests := []struct {
		name  string
		raw   string
		want  string
		valid bool
	}{
		{name: "ru", raw: "ru", want: RU, valid: true},
		{name: "locale with underscore", raw: "ru_RU.UTF-8", want: RU, valid: true},
		{name: "english locale", raw: "en_US.UTF-8", want: EN, valid: true},
		{name: "unsupported locale", raw: "de_DE.UTF-8", want: EN, valid: false},
		{name: "empty value", raw: "   ", want: EN, valid: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, valid := Normalize(tt.raw)
			if got != tt.want || valid != tt.valid {
				t.Fatalf("Normalize(%q) = (%q, %v), want (%q, %v)", tt.raw, got, valid, tt.want, tt.valid)
			}
		})
	}
}

func TestDetect(t *testing.T) {
	tests := []struct {
		name      string
		args      []string
		envValue  string
		envOK     bool
		system    string
		systemErr error
		want      string
		wantValid bool
	}{
		{
			name:      "flag wins over env and system",
			args:      []string{"--lang", "en"},
			envValue:  "ru_RU.UTF-8",
			envOK:     true,
			system:    "ru_RU.UTF-8",
			want:      EN,
			wantValid: true,
		},
		{
			name:      "env wins over system",
			envValue:  "ru_RU.UTF-8",
			envOK:     true,
			system:    "en_US.UTF-8",
			want:      RU,
			wantValid: true,
		},
		{
			name:      "system fallback",
			system:    "en_US.UTF-8",
			want:      EN,
			wantValid: true,
		},
		{
			name:      "system error falls back to english",
			systemErr: fmt.Errorf("boom"),
			want:      EN,
			wantValid: true,
		},
		{
			name:      "invalid flag is explicit error",
			args:      []string{"--lang", "de"},
			want:      EN,
			wantValid: false,
		},
		{
			name:      "invalid env is explicit error",
			envValue:  "de_DE.UTF-8",
			envOK:     true,
			want:      EN,
			wantValid: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			stubLocaleDetection(t, tt.envValue, tt.envOK, tt.system, tt.systemErr)

			got, valid := Detect(tt.args)
			if got != tt.want || valid != tt.wantValid {
				t.Fatalf("Detect(%v) = (%q, %v), want (%q, %v)", tt.args, got, valid, tt.want, tt.wantValid)
			}
		})
	}
}

func TestLangFlagValue(t *testing.T) {
	tests := []struct {
		name string
		args []string
		want string
		ok   bool
	}{
		{name: "separate value", args: []string{"--lang", "ru"}, want: "ru", ok: true},
		{name: "equals syntax", args: []string{"--lang=en"}, want: "en", ok: true},
		{name: "stop on double dash", args: []string{"--", "--lang", "ru"}, want: "", ok: false},
		{name: "missing value", args: []string{"--lang"}, want: "", ok: false},
		{name: "no flag", args: []string{"map", "question"}, want: "", ok: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := langFlagValue(tt.args)
			if got != tt.want || ok != tt.ok {
				t.Fatalf("langFlagValue(%v) = (%q, %v), want (%q, %v)", tt.args, got, ok, tt.want, tt.ok)
			}
		})
	}
}

func TestT(t *testing.T) {
	t.Run("without localizer returns id", func(t *testing.T) {
		resetLocalizationState(t)

		if got := T("missing.id"); got != "missing.id" {
			t.Fatalf("T(missing.id) = %q, want %q", got, "missing.id")
		}
	})

	t.Run("localizes with template data and falls back on unknown id", func(t *testing.T) {
		resetLocalizationState(t)
		registerCatalog("owner-a", testFS(
			"i18n/en.toml", "[hello]\nmessage = \"Hello, {{.Name}}!\"\n",
			"i18n/ru.toml", "[hello]\nmessage = \"Привет, {{.Name}}!\"\n",
		), "i18n/en.toml", "i18n/ru.toml")

		if err := SetCurrent(RU); err != nil {
			t.Fatalf("SetCurrent(ru) error = %v", err)
		}
		if got := T("hello.message", Data{"Name": "Борис"}); got != "Привет, Борис!" {
			t.Fatalf("T(hello.message) = %q, want %q", got, "Привет, Борис!")
		}
		if got := T("unknown.message"); got != "unknown.message" {
			t.Fatalf("T(unknown.message) = %q, want %q", got, "unknown.message")
		}
	})
}

func TestBundleLoadsRegisteredCatalogs(t *testing.T) {
	resetLocalizationState(t)

	registerCatalog("owner-b", testFS(
		"i18n/en.toml", "[bundle]\nvalue = \"value from B\"\n",
		"i18n/ru.toml", "[bundle]\nvalue = \"значение от B\"\n",
	), "i18n/ru.toml", "i18n/en.toml")
	registerCatalog("owner-a", testFS(
		"i18n/en.toml", "[alpha]\nvalue = \"value from A\"\n",
		"i18n/ru.toml", "[alpha]\nvalue = \"значение от A\"\n",
	), "i18n/en.toml", "i18n/ru.toml")

	assertLocalizedValue(t, EN, "alpha.value", "value from A")
	assertLocalizedValue(t, EN, "bundle.value", "value from B")
	assertLocalizedValue(t, RU, "alpha.value", "значение от A")
	assertLocalizedValue(t, RU, "bundle.value", "значение от B")
}

func TestFreezeCatalogFilesSortsDeterministically(t *testing.T) {
	resetLocalizationState(t)

	registerCatalog("owner-b", testFS(), "i18n/z.toml", "i18n/a.toml")
	registerCatalog("owner-a", testFS(), "i18n/c.toml", "i18n/b.toml")

	files := freezeCatalogFiles()
	got := make([]string, 0, len(files))
	for _, file := range files {
		got = append(got, file.owner+":"+file.path)
	}

	want := []string{
		"owner-a:i18n/b.toml",
		"owner-a:i18n/c.toml",
		"owner-b:i18n/a.toml",
		"owner-b:i18n/z.toml",
	}
	if strings.Join(got, "|") != strings.Join(want, "|") {
		t.Fatalf("freezeCatalogFiles() = %v, want %v", got, want)
	}
}

func TestMustRegisterValidationPanics(t *testing.T) {
	tests := []struct {
		name string
		fn   func()
		want string
	}{
		{
			name: "empty owner",
			fn: func() {
				MustRegister("   ", testFS("i18n/en.toml", ""), "i18n/en.toml")
			},
			want: "owner must not be empty",
		},
		{
			name: "nil fs",
			fn: func() {
				var nilFS fs.FS
				MustRegister("owner-a", nilFS, "i18n/en.toml")
			},
			want: `owner "owner-a" provided nil fs`,
		},
		{
			name: "no paths",
			fn: func() {
				MustRegister("owner-a", testFS())
			},
			want: `owner "owner-a" must register at least one catalog path`,
		},
		{
			name: "empty path",
			fn: func() {
				MustRegister("owner-a", testFS(), " ")
			},
			want: `owner "owner-a" provided empty catalog path`,
		},
		{
			name: "duplicate owner",
			fn: func() {
				registerCatalog("owner-a", testFS(), "i18n/en.toml")
				registerCatalog("owner-a", testFS(), "i18n/ru.toml")
			},
			want: `duplicate catalog owner "owner-a"`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			resetLocalizationState(t)
			mustPanicContains(t, tt.want, tt.fn)
		})
	}
}

func TestBundleFailsOnDuplicateMessageID(t *testing.T) {
	resetLocalizationState(t)

	registerCatalog("owner-a", testFS("i18n/en.toml", "[shared]\nvalue = \"one\"\n"), "i18n/en.toml")
	registerCatalog("owner-b", testFS("i18n/en.toml", "[shared]\nvalue = \"two\"\n"), "i18n/en.toml")

	_, err := bundle()
	if err == nil {
		t.Fatal("bundle() error = nil, want duplicate message id error")
	}
	if !strings.Contains(err.Error(), "duplicate locale message en/shared.value") {
		t.Fatalf("bundle() error = %q, want duplicate message id", err)
	}
}

func TestLoadCatalogFileErrors(t *testing.T) {
	resetLocalizationState(t)

	t.Run("missing file", func(t *testing.T) {
		err := loadCatalogFile(i18n.NewBundle(language.English), catalogFile{
			owner: "owner-a",
			fsys:  testFS(),
			path:  "i18n/en.toml",
		}, map[messageKey]string{})
		if err == nil || !strings.Contains(err.Error(), "read locale catalog owner-a (i18n/en.toml)") {
			t.Fatalf("loadCatalogFile() error = %v, want read error", err)
		}
	})

	t.Run("invalid toml", func(t *testing.T) {
		err := loadCatalogFile(i18n.NewBundle(language.English), catalogFile{
			owner: "owner-a",
			fsys:  testFS("i18n/en.toml", "[broken"),
			path:  "i18n/en.toml",
		}, map[messageKey]string{})
		if err == nil || !strings.Contains(err.Error(), "parse locale catalog owner-a (i18n/en.toml)") {
			t.Fatalf("loadCatalogFile() error = %v, want parse error", err)
		}
	})
}

func TestMustRegisterAfterBundleInitPanics(t *testing.T) {
	resetLocalizationState(t)
	registerCatalog("owner-a", testFS("i18n/en.toml", "[bundle]\nvalue = \"ok\"\n"), "i18n/en.toml")

	if _, err := bundle(); err != nil {
		t.Fatalf("bundle() error = %v", err)
	}

	mustPanicContains(t, `catalogs already initialized; cannot register "owner-b"`, func() {
		registerCatalog("owner-b", testFS(), "i18n/en.toml")
	})
}

func stubLocaleDetection(t *testing.T, envValue string, envOK bool, system string, systemErr error) {
	t.Helper()

	origLookup := lookupEnv
	origDetect := detectSystemLocale
	t.Cleanup(func() {
		lookupEnv = origLookup
		detectSystemLocale = origDetect
	})

	lookupEnv = func(key string) (string, bool) {
		if key != "RAGCLI_LANG" {
			return "", false
		}
		return envValue, envOK
	}
	detectSystemLocale = func() (string, error) {
		return system, systemErr
	}
}

func registerCatalog(owner string, fsys fstest.MapFS, paths ...string) {
	MustRegister(owner, fsys, paths...)
}

func testFS(entries ...string) fstest.MapFS {
	files := make(fstest.MapFS, len(entries)/2)
	for i := 0; i+1 < len(entries); i += 2 {
		files[entries[i]] = &fstest.MapFile{Data: []byte(entries[i+1])}
	}
	return files
}

func assertLocalizedValue(t *testing.T, lang string, id string, want string) {
	t.Helper()

	if err := SetCurrent(lang); err != nil {
		t.Fatalf("SetCurrent(%s) error = %v", lang, err)
	}
	if got := T(id); got != want {
		t.Fatalf("T(%s) = %q, want %q", id, got, want)
	}
}

func mustPanicContains(t *testing.T, want string, fn func()) {
	t.Helper()

	defer func() {
		t.Helper()
		if got := recover(); got == nil {
			t.Fatalf("panic = nil, want substring %q", want)
		} else if text := fmt.Sprint(got); !strings.Contains(text, want) {
			t.Fatalf("panic = %q, want substring %q", text, want)
		}
	}()

	fn()
}

func resetLocalizationState(t *testing.T) {
	t.Helper()

	bundleOnce = sync.Once{}
	bundleInst = nil
	bundleErr = nil
	currentLocal.Store(nil)

	registryMu.Lock()
	registryFrozen = false
	registeredOwners = nil
	registeredFiles = nil
	registryMu.Unlock()
}
