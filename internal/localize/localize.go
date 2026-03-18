package localize

import (
	"fmt"
	"io/fs"
	"os"
	"sort"
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

var (
	bundleOnce sync.Once
	bundleInst *i18n.Bundle
	errBundle  error

	currentLocal atomic.Pointer[i18n.Localizer]

	registryMu       sync.Mutex
	registryFrozen   bool
	registeredOwners map[string]struct{}
	registeredFiles  []catalogFile

	detectSystemLocale = func() (string, error) {
		tag, err := locale.Detect()
		if err != nil {
			return "", err
		}
		return tag.String(), nil
	}
)

type catalogFile struct {
	owner string
	fsys  fs.FS
	path  string
}

type messageKey struct {
	tag language.Tag
	id  string
}

var tomlUnmarshalFuncs = map[string]i18n.UnmarshalFunc{
	"toml": toml.Unmarshal,
}

func MustRegister(owner string, fsys fs.FS, paths ...string) {
	owner = strings.TrimSpace(owner)
	if owner == "" {
		panic("localize: owner must not be empty")
	}
	if fsys == nil {
		panic(fmt.Sprintf("localize: owner %q provided nil fs", owner))
	}
	if len(paths) == 0 {
		panic(fmt.Sprintf("localize: owner %q must register at least one catalog path", owner))
	}

	sortedPaths := make([]string, 0, len(paths))
	for _, path := range paths {
		path = strings.TrimSpace(path)
		if path == "" {
			panic(fmt.Sprintf("localize: owner %q provided empty catalog path", owner))
		}
		sortedPaths = append(sortedPaths, path)
	}
	sort.Strings(sortedPaths)

	registryMu.Lock()
	defer registryMu.Unlock()

	if registryFrozen {
		panic(fmt.Sprintf("localize: catalogs already initialized; cannot register %q", owner))
	}
	if registeredOwners == nil {
		registeredOwners = make(map[string]struct{})
	}
	if _, exists := registeredOwners[owner]; exists {
		panic(fmt.Sprintf("localize: duplicate catalog owner %q", owner))
	}
	registeredOwners[owner] = struct{}{}
	for _, path := range sortedPaths {
		registeredFiles = append(registeredFiles, catalogFile{owner: owner, fsys: fsys, path: path})
	}
}

func bundle() (*i18n.Bundle, error) {
	bundleOnce.Do(func() {
		b := i18n.NewBundle(language.English)
		b.RegisterUnmarshalFunc("toml", toml.Unmarshal)

		seenMessages := make(map[messageKey]string)
		for _, catalog := range freezeCatalogFiles() {
			if err := loadCatalogFile(b, catalog, seenMessages); err != nil {
				errBundle = err
				return
			}
		}
		bundleInst = b
	})
	return bundleInst, errBundle
}

func freezeCatalogFiles() []catalogFile {
	registryMu.Lock()
	defer registryMu.Unlock()

	registryFrozen = true
	files := append([]catalogFile(nil), registeredFiles...)
	sort.Slice(files, func(i, j int) bool {
		if files[i].owner == files[j].owner {
			return files[i].path < files[j].path
		}
		return files[i].owner < files[j].owner
	})
	return files
}

func loadCatalogFile(bundle *i18n.Bundle, catalog catalogFile, seen map[messageKey]string) error {
	raw, err := fs.ReadFile(catalog.fsys, catalog.path)
	if err != nil {
		return fmt.Errorf("read locale catalog %s (%s): %w", catalog.owner, catalog.path, err)
	}

	messageFile, err := i18n.ParseMessageFileBytes(raw, catalog.path, tomlUnmarshalFuncs)
	if err != nil {
		return fmt.Errorf("parse locale catalog %s (%s): %w", catalog.owner, catalog.path, err)
	}

	for _, message := range messageFile.Messages {
		key := messageKey{tag: messageFile.Tag, id: message.ID}
		source := catalog.owner + ":" + catalog.path
		if first, exists := seen[key]; exists {
			return fmt.Errorf("duplicate locale message %s/%s in %s; already defined in %s", key.tag, key.id, source, first)
		}
		seen[key] = source
	}

	if err := bundle.AddMessages(messageFile.Tag, messageFile.Messages...); err != nil {
		return fmt.Errorf("add locale catalog %s (%s): %w", catalog.owner, catalog.path, err)
	}
	return nil
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
		return Normalize(value)
	}

	if value, ok := lookupEnv("RAGCLI_LANG"); ok {
		return Normalize(value)
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

	tag, err := language.Parse(strings.ReplaceAll(value, "_", "-"))
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
	localizer := currentLocal.Load()
	if localizer == nil {
		return id
	}

	cfg := &i18n.LocalizeConfig{MessageID: id}
	if len(data) > 0 {
		cfg.TemplateData = data[0]
	}
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

var lookupEnv = os.LookupEnv
