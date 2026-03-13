package app

import (
	"embed"

	"github.com/borro/ragcli/internal/localize"
)

//go:embed i18n/*.toml
var i18nFS embed.FS

func init() {
	localize.MustRegister("internal/app", i18nFS, "i18n/en.toml", "i18n/ru.toml")
}
