package app

import (
	"net/url"
	"strings"
)

func llmProxyLogFields(proxyValue string, noProxy bool) (mode string, target string) {
	switch {
	case noProxy:
		return "disabled", "off"
	case strings.TrimSpace(proxyValue) == "":
		return "environment", "environment"
	default:
		parsed, err := url.Parse(strings.TrimSpace(proxyValue))
		if err != nil || parsed.Scheme == "" || parsed.Host == "" {
			return "fixed", "invalid"
		}
		return "fixed", parsed.Scheme + "://" + parsed.Host
	}
}
