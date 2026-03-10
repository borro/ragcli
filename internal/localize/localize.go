package localize

import (
	"embed"
	"fmt"
	"io/fs"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/BurntSushi/toml"
	"github.com/Xuanwo/go-locale"
	"github.com/nicksnyder/go-i18n/v2/i18n"
	"golang.org/x/text/language"
)

const (
	EN = "en"
	RU = "ru"
)

type Data map[string]any

//go:embed catalogs/*.toml
var catalogFS embed.FS

var (
	bundleOnce sync.Once
	bundleInst *i18n.Bundle
	bundleErr  error

	currentLocal atomic.Pointer[i18n.Localizer]

	detectSystemLocale = func() (string, error) {
		tag, err := locale.Detect()
		if err != nil {
			return "", err
		}
		return tag.String(), nil
	}
)

func bundle() (*i18n.Bundle, error) {
	bundleOnce.Do(func() {
		b := i18n.NewBundle(language.English)
		for _, spec := range []struct {
			path string
			tag  language.Tag
		}{
			{path: "catalogs/active.en.toml", tag: language.English},
			{path: "catalogs/active.ru.toml", tag: language.Russian},
		} {
			if err := loadCatalog(b, spec.tag, spec.path); err != nil {
				bundleErr = fmt.Errorf("load locale catalog %s: %w", spec.path, err)
				return
			}
		}
		bundleInst = b
	})
	return bundleInst, bundleErr
}

func loadCatalog(bundle *i18n.Bundle, tag language.Tag, path string) error {
	raw, err := fs.ReadFile(catalogFS, path)
	if err != nil {
		return err
	}

	var data map[string]any
	if err := toml.Unmarshal(raw, &data); err != nil {
		return err
	}

	messages := make([]*i18n.Message, 0, 128)
	flattenMessages("", data, &messages)
	return bundle.AddMessages(tag, messages...)
}

func flattenMessages(prefix string, node map[string]any, messages *[]*i18n.Message) {
	for key, value := range node {
		id := key
		if prefix != "" {
			id = prefix + "." + key
		}

		switch typed := value.(type) {
		case string:
			*messages = append(*messages, &i18n.Message{ID: id, Other: typed})
		case map[string]any:
			flattenMessages(id, typed, messages)
		}
	}
}

func SetCurrent(code string) error {
	normalized, _ := Normalize(code)
	b, err := bundle()
	if err != nil {
		return err
	}
	currentLocal.Store(i18n.NewLocalizer(b, normalized))
	return nil
}

func Detect(args []string) (string, bool) {
	if value, ok := langFlagValue(args); ok {
		normalized, valid := Normalize(value)
		if !valid {
			return EN, false
		}
		return normalized, true
	}

	if value, ok := envLangValue(); ok {
		normalized, valid := Normalize(value)
		if !valid {
			return EN, false
		}
		return normalized, true
	}

	raw, err := detectSystemLocale()
	if err != nil {
		return EN, true
	}
	normalized, valid := Normalize(raw)
	if !valid {
		return EN, true
	}
	return normalized, true
}

func Normalize(raw string) (string, bool) {
	value := strings.TrimSpace(raw)
	if value == "" {
		return EN, false
	}
	if dot := strings.Index(value, "."); dot >= 0 {
		value = value[:dot]
	}
	value = strings.ReplaceAll(value, "_", "-")
	tag, err := language.Parse(value)
	if err != nil {
		return EN, false
	}
	base, _ := tag.Base()
	switch base.String() {
	case EN:
		return EN, true
	case RU:
		return RU, true
	default:
		return EN, false
	}
}

func T(id string, data ...Data) string {
	if len(data) > 0 {
		return localize(id, data[0])
	}
	return localize(id, nil)
}

func localize(id string, data Data) string {
	localizer := currentLocal.Load()
	if localizer == nil {
		return id
	}

	cfg := &i18n.LocalizeConfig{MessageID: id, TemplateData: data}
	value, err := localizer.Localize(cfg)
	if err == nil {
		return value
	}
	return id
}

func langFlagValue(args []string) (string, bool) {
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "--" {
			break
		}
		switch {
		case arg == "--lang":
			if i+1 >= len(args) {
				return "", false
			}
			return args[i+1], true
		case strings.HasPrefix(arg, "--lang="):
			return strings.TrimPrefix(arg, "--lang="), true
		}
	}
	return "", false
}

func envLangValue() (string, bool) {
	raw, ok := lookupEnv("RAGCLI_LANG")
	if !ok {
		return "", false
	}
	return raw, true
}

var lookupEnv = func(key string) (string, bool) {
	return syscallLookupEnv(key)
}
