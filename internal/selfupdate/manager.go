package selfupdate

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"regexp"
	"runtime"
	"strings"

	"github.com/Masterminds/semver/v3"
	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
	upstream "github.com/creativeprojects/go-selfupdate"
	upstreamupdate "github.com/creativeprojects/go-selfupdate/update"
)

const (
	defaultRepositorySlug = "borro/ragcli"
	checksumAssetName     = "checksums.txt"
)

var (
	ErrInvalidCurrentVersion = errors.New("invalid current version")
	ErrNoStableRelease       = errors.New("no stable release found")
	versionPattern           = regexp.MustCompile(`\d+\.\d+\.\d+`)
)

type Config struct {
	Repository     upstream.Repository
	Source         upstream.Source
	ExecutablePath func() (string, error)
	ApplyRelease   func(context.Context, *upstream.Release, string) error
	RollbackError  func(error) error
	OS             string
	Arch           string
	Arm            uint8
}

type Manager struct {
	repository     upstream.Repository
	source         upstream.Source
	updater        *upstream.Updater
	executablePath func() (string, error)
	applyRelease   func(context.Context, *upstream.Release, string) error
	rollbackError  func(error) error
	os             string
	arch           string
}

type releaseCandidate struct {
	tag     string
	version *semver.Version
}

func New(config Config) (*Manager, error) {
	repository := config.Repository
	if repository == nil {
		repository = upstream.ParseSlug(defaultRepositorySlug)
	}

	source := config.Source
	if source == nil {
		var err error
		source, err = upstream.NewGitHubSource(upstream.GitHubConfig{})
		if err != nil {
			return nil, err
		}
	}

	osName := config.OS
	if osName == "" {
		osName = runtime.GOOS
	}
	archName := config.Arch
	if archName == "" {
		archName = runtime.GOARCH
	}

	updater, err := upstream.NewUpdater(upstream.Config{
		Source:    source,
		Validator: &upstream.ChecksumValidator{UniqueFilename: checksumAssetName},
		OS:        osName,
		Arch:      archName,
		Arm:       config.Arm,
	})
	if err != nil {
		return nil, err
	}

	manager := &Manager{
		repository:     repository,
		source:         source,
		updater:        updater,
		executablePath: config.ExecutablePath,
		rollbackError:  config.RollbackError,
		applyRelease:   config.ApplyRelease,
		os:             osName,
		arch:           archName,
	}
	if manager.executablePath == nil {
		manager.executablePath = upstream.ExecutablePath
	}
	if manager.rollbackError == nil {
		manager.rollbackError = upstreamupdate.RollbackError
	}
	if manager.applyRelease == nil {
		manager.applyRelease = manager.updater.UpdateTo
	}

	return manager, nil
}

func (m *Manager) Run(ctx context.Context, current string, opts Options, meter verbose.Meter) (Result, error) {
	meter.Start(localize.T("selfupdate.progress.checking"))

	currentVersion, err := parseCurrentVersion(current)
	if err != nil {
		return Result{}, invalidCurrentVersionError(current)
	}
	currentDisplay := displayVersion(currentVersion)

	latest, err := m.latestStableRelease(ctx)
	if err != nil {
		return Result{}, err
	}
	if latest == nil {
		return Result{}, noStableReleaseError()
	}

	latestDisplay := displayVersion(latest.version)
	if !latest.version.GreaterThan(currentVersion) {
		meter.Done(localize.T("selfupdate.progress.up_to_date"))
		return Result{
			CurrentVersion: currentDisplay,
			LatestVersion:  latestDisplay,
			Checked:        opts.CheckOnly,
			Output: localize.T("selfupdate.status.up_to_date", localize.Data{
				"Current": currentDisplay,
			}),
		}, nil
	}

	slog.Info("self-update candidate detected",
		"current_version", currentDisplay,
		"latest_version", latestDisplay,
		"check_only", opts.CheckOnly,
	)
	meter.Report(1, 3, localize.T("selfupdate.progress.resolving_asset", localize.Data{
		"Version": latestDisplay,
	}))

	release, found, err := m.updater.DetectVersion(ctx, m.repository, latest.tag)
	if err != nil {
		return Result{}, m.mapDetectError(latestDisplay, err)
	}
	if !found {
		return Result{}, assetUnavailableError(latestDisplay, m.os, m.arch)
	}

	if opts.CheckOnly {
		meter.Done(localize.T("selfupdate.progress.available"))
		return Result{
			CurrentVersion: currentDisplay,
			LatestVersion:  latestDisplay,
			Checked:        true,
			Output: localize.T("selfupdate.status.available", localize.Data{
				"Current": currentDisplay,
				"Latest":  latestDisplay,
			}),
		}, nil
	}

	cmdPath, err := m.executablePath()
	if err != nil {
		return Result{}, fmt.Errorf("%s: %w", localize.T("error.selfupdate.executable_path"), err)
	}

	meter.Report(2, 3, localize.T("selfupdate.progress.applying", localize.Data{
		"Version": latestDisplay,
	}))
	if err := m.applyRelease(ctx, release, cmdPath); err != nil {
		return Result{}, m.mapApplyError(latestDisplay, err)
	}

	meter.Done(localize.T("selfupdate.progress.updated"))
	return Result{
		CurrentVersion: currentDisplay,
		LatestVersion:  latestDisplay,
		Updated:        true,
		Output: localize.T("selfupdate.status.updated", localize.Data{
			"Current": currentDisplay,
			"Latest":  latestDisplay,
		}),
	}, nil
}

func (m *Manager) latestStableRelease(ctx context.Context) (*releaseCandidate, error) {
	releases, err := m.source.ListReleases(ctx, m.repository)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", localize.T("error.selfupdate.list_releases"), err)
	}

	var latest *releaseCandidate
	for _, release := range releases {
		if release == nil || release.GetDraft() || release.GetPrerelease() {
			continue
		}

		version, ok := parseTagVersion(release.GetTagName())
		if !ok {
			continue
		}
		if latest == nil || version.GreaterThan(latest.version) {
			latest = &releaseCandidate{
				tag:     release.GetTagName(),
				version: version,
			}
		}
	}

	return latest, nil
}

func parseCurrentVersion(raw string) (*semver.Version, error) {
	trimmed := strings.TrimSpace(raw)
	lower := strings.ToLower(trimmed)
	if trimmed == "" || lower == "dev" || strings.HasPrefix(lower, "dev-") {
		return nil, ErrInvalidCurrentVersion
	}

	version, ok := parseTagVersion(trimmed)
	if !ok {
		return nil, ErrInvalidCurrentVersion
	}
	return version, nil
}

func parseTagVersion(raw string) (*semver.Version, bool) {
	trimmed := strings.TrimSpace(raw)
	indices := versionPattern.FindStringIndex(trimmed)
	if indices == nil {
		return nil, false
	}

	version, err := semver.NewVersion(trimmed[indices[0]:])
	if err != nil {
		return nil, false
	}
	return version, true
}

func displayVersion(version *semver.Version) string {
	if version == nil {
		return ""
	}
	return "v" + version.String()
}

func invalidCurrentVersionError(raw string) error {
	version := strings.TrimSpace(raw)
	if version == "" {
		version = "<empty>"
	}
	return fmt.Errorf("%s: %w", localize.T("error.selfupdate.invalid_current_version", localize.Data{
		"Version": version,
	}), ErrInvalidCurrentVersion)
}

func noStableReleaseError() error {
	return fmt.Errorf("%s: %w", localize.T("error.selfupdate.no_stable_release"), ErrNoStableRelease)
}

func assetUnavailableError(version string, osName string, archName string) error {
	return fmt.Errorf("%s: %w", localize.T("error.selfupdate.asset_unavailable", localize.Data{
		"Version": version,
		"OS":      osName,
		"Arch":    archName,
	}), upstream.ErrAssetNotFound)
}

func (m *Manager) mapDetectError(version string, err error) error {
	if errors.Is(err, upstream.ErrValidationAssetNotFound) {
		return fmt.Errorf("%s: %w", localize.T("error.selfupdate.missing_checksum", localize.Data{
			"Version": version,
		}), err)
	}

	return fmt.Errorf("%s: %w", localize.T("error.selfupdate.detect_release", localize.Data{
		"Version": version,
	}), err)
}

func (m *Manager) mapApplyError(version string, err error) error {
	if rollbackErr := m.rollbackError(err); rollbackErr != nil {
		return fmt.Errorf("%s: %w", localize.T("error.selfupdate.rollback", localize.Data{
			"Error": rollbackErr.Error(),
		}), err)
	}
	if errors.Is(err, upstream.ErrChecksumValidationFailed) ||
		errors.Is(err, upstream.ErrIncorrectChecksumFile) ||
		errors.Is(err, upstream.ErrHashNotFound) {
		return fmt.Errorf("%s: %w", localize.T("error.selfupdate.checksum", localize.Data{
			"Version": version,
		}), err)
	}

	return fmt.Errorf("%s: %w", localize.T("error.selfupdate.apply", localize.Data{
		"Version": version,
	}), err)
}
