package main

import (
	"os"

	"github.com/borro/ragcli/internal/app"
)

var version = "dev"

func main() {
	os.Exit(app.Run(os.Args[1:], os.Stdout, os.Stdin, version))
}
