package selfupdate

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/borro/ragcli/internal/localize"
	"github.com/borro/ragcli/internal/verbose"
	upstream "github.com/creativeprojects/go-selfupdate"
)

type fakeSource struct {
	releases  []upstream.SourceRelease
	downloads []int64
	assets    map[int64][]byte
	err       error
}

func (s *fakeSource) ListReleases(context.Context, upstream.Repository) ([]upstream.SourceRelease, error) {
	if s.err != nil {
		return nil, s.err
	}
	return s.releases, nil
}

func (s *fakeSource) DownloadReleaseAsset(_ context.Context, _ *upstream.Release, assetID int64) (io.ReadCloser, error) {
	s.downloads = append(s.downloads, assetID)
	data, ok := s.assets[assetID]
	if !ok {
		return nil, fmt.Errorf("asset %d not found", assetID)
	}
	return io.NopCloser(bytes.NewReader(data)), nil
}

type fakeRelease struct {
	id         int64
	tag        string
	name       string
	url        string
	draft      bool
	prerelease bool
	published  time.Time
	assets     []upstream.SourceAsset
}

func (r fakeRelease) GetID() int64                      { return r.id }
func (r fakeRelease) GetTagName() string                { return r.tag }
func (r fakeRelease) GetDraft() bool                    { return r.draft }
func (r fakeRelease) GetPrerelease() bool               { return r.prerelease }
func (r fakeRelease) GetPublishedAt() time.Time         { return r.published }
func (r fakeRelease) GetReleaseNotes() string           { return "" }
func (r fakeRelease) GetName() string                   { return r.name }
func (r fakeRelease) GetURL() string                    { return r.url }
func (r fakeRelease) GetAssets() []upstream.SourceAsset { return r.assets }

type fakeAsset struct {
	id   int64
	name string
	url  string
	size int
}

func (a fakeAsset) GetID() int64                  { return a.id }
func (a fakeAsset) GetName() string               { return a.name }
func (a fakeAsset) GetSize() int                  { return a.size }
func (a fakeAsset) GetBrowserDownloadURL() string { return a.url }

func TestRunRejectsInvalidCurrentVersion(t *testing.T) {
	setEnglishLocale(t)

	manager, err := New(Config{
		Source: &fakeSource{},
		OS:     "linux",
		Arch:   "amd64",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	for _, current := range []string{"", "dev", "not-a-version"} {
		_, err := manager.Run(context.Background(), current, Options{CheckOnly: true}, noopMeter())
		if !errors.Is(err, ErrInvalidCurrentVersion) {
			t.Fatalf("Run(%q) error = %v, want ErrInvalidCurrentVersion", current, err)
		}
	}
}

func TestRunSelectsLatestStableAssetForPlatform(t *testing.T) {
	setEnglishLocale(t)

	const (
		linuxAssetID     = 101
		checksumsAssetID = 102
	)
	archive := tarGzExecutable(t, "ragcli", []byte("new linux binary"))
	checksums := checksumFile("ragcli_linux_amd64.tar.gz", archive)

	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:         1,
				tag:        "v1.2.0-rc1",
				prerelease: true,
				published:  time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/rc.tar.gz", size: 1},
				},
			},
			fakeRelease{
				id:        2,
				tag:       "v1.2.0",
				published: time.Unix(2, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: linuxAssetID, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: len(archive)},
					fakeAsset{id: 201, name: "ragcli_darwin_arm64.tar.gz", url: "https://example.test/darwin.tar.gz", size: 1},
					fakeAsset{id: checksumsAssetID, name: checksumAssetName, url: "https://example.test/checksums.txt", size: len(checksums)},
				},
			},
		},
		assets: map[int64][]byte{
			linuxAssetID:     archive,
			checksumsAssetID: checksums,
		},
	}

	path := writeExecutable(t, []byte("old"))
	var appliedAsset string
	manager, err := New(Config{
		Source:         source,
		OS:             "linux",
		Arch:           "amd64",
		ExecutablePath: func() (string, error) { return path, nil },
		ApplyRelease: func(_ context.Context, rel *upstream.Release, _ string) error {
			appliedAsset = rel.AssetName
			return nil
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := manager.Run(context.Background(), "v1.0.0", Options{}, noopMeter())
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if appliedAsset != "ragcli_linux_amd64.tar.gz" {
		t.Fatalf("appliedAsset = %q, want ragcli_linux_amd64.tar.gz", appliedAsset)
	}
	if !result.Updated {
		t.Fatalf("Updated = %v, want true", result.Updated)
	}
}

func TestRunFailsWhenAssetIsUnavailableForPlatform(t *testing.T) {
	setEnglishLocale(t)

	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_darwin_arm64.tar.gz", url: "https://example.test/darwin.tar.gz", size: 1},
				},
			},
		},
	}

	manager, err := New(Config{
		Source: source,
		OS:     "linux",
		Arch:   "amd64",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = manager.Run(context.Background(), "v1.0.0", Options{CheckOnly: true}, noopMeter())
	if !errors.Is(err, upstream.ErrAssetNotFound) {
		t.Fatalf("Run() error = %v, want ErrAssetNotFound", err)
	}
}

func TestRunFailsWhenChecksumsAssetIsMissing(t *testing.T) {
	setEnglishLocale(t)

	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: 1},
				},
			},
		},
	}

	manager, err := New(Config{
		Source: source,
		OS:     "linux",
		Arch:   "amd64",
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = manager.Run(context.Background(), "v1.0.0", Options{CheckOnly: true}, noopMeter())
	if !errors.Is(err, upstream.ErrValidationAssetNotFound) {
		t.Fatalf("Run() error = %v, want ErrValidationAssetNotFound", err)
	}
}

func TestRunCheckDoesNotWriteOrDownloadAssets(t *testing.T) {
	setEnglishLocale(t)

	archive := tarGzExecutable(t, "ragcli", []byte("new binary"))
	checksums := checksumFile("ragcli_linux_amd64.tar.gz", archive)
	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: len(archive)},
					fakeAsset{id: 2, name: checksumAssetName, url: "https://example.test/checksums.txt", size: len(checksums)},
				},
			},
		},
		assets: map[int64][]byte{
			1: archive,
			2: checksums,
		},
	}

	path := writeExecutable(t, []byte("old binary"))
	manager, err := New(Config{
		Source:         source,
		OS:             "linux",
		Arch:           "amd64",
		ExecutablePath: func() (string, error) { return path, nil },
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := manager.Run(context.Background(), "v1.0.0", Options{CheckOnly: true}, noopMeter())
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := readFile(t, path); got != "old binary" {
		t.Fatalf("file = %q, want old binary", got)
	}
	if len(source.downloads) != 0 {
		t.Fatalf("downloads = %v, want none", source.downloads)
	}
	if !result.Checked {
		t.Fatalf("Checked = %v, want true", result.Checked)
	}
}

func TestRunUpdateReplacesExecutable(t *testing.T) {
	setEnglishLocale(t)

	archive := tarGzExecutable(t, "ragcli", []byte("new binary"))
	checksums := checksumFile("ragcli_linux_amd64.tar.gz", archive)
	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: len(archive)},
					fakeAsset{id: 2, name: checksumAssetName, url: "https://example.test/checksums.txt", size: len(checksums)},
				},
			},
		},
		assets: map[int64][]byte{
			1: archive,
			2: checksums,
		},
	}

	path := writeExecutable(t, []byte("old binary"))
	manager, err := New(Config{
		Source:         source,
		OS:             "linux",
		Arch:           "amd64",
		ExecutablePath: func() (string, error) { return path, nil },
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	result, err := manager.Run(context.Background(), "v1.0.0", Options{}, noopMeter())
	if err != nil {
		t.Fatalf("Run() error = %v", err)
	}
	if got := readFile(t, path); got != "new binary" {
		t.Fatalf("file = %q, want new binary", got)
	}
	if !result.Updated {
		t.Fatalf("Updated = %v, want true", result.Updated)
	}
}

func TestRunChecksumMismatchDoesNotModifyExecutable(t *testing.T) {
	setEnglishLocale(t)

	archive := tarGzExecutable(t, "ragcli", []byte("new binary"))
	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: len(archive)},
					fakeAsset{id: 2, name: checksumAssetName, url: "https://example.test/checksums.txt", size: len(checksumFile("ragcli_linux_amd64.tar.gz", []byte("wrong")))},
				},
			},
		},
		assets: map[int64][]byte{
			1: archive,
			2: checksumFile("ragcli_linux_amd64.tar.gz", []byte("wrong")),
		},
	}

	path := writeExecutable(t, []byte("old binary"))
	manager, err := New(Config{
		Source:         source,
		OS:             "linux",
		Arch:           "amd64",
		ExecutablePath: func() (string, error) { return path, nil },
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = manager.Run(context.Background(), "v1.0.0", Options{}, noopMeter())
	if !errors.Is(err, upstream.ErrChecksumValidationFailed) {
		t.Fatalf("Run() error = %v, want ErrChecksumValidationFailed", err)
	}
	if got := readFile(t, path); got != "old binary" {
		t.Fatalf("file = %q, want old binary", got)
	}
}

func TestRunMapsRollbackFailure(t *testing.T) {
	setEnglishLocale(t)

	source := &fakeSource{
		releases: []upstream.SourceRelease{
			fakeRelease{
				id:        1,
				tag:       "v1.1.0",
				published: time.Unix(1, 0),
				assets: []upstream.SourceAsset{
					fakeAsset{id: 1, name: "ragcli_linux_amd64.tar.gz", url: "https://example.test/linux.tar.gz", size: 1},
					fakeAsset{id: 2, name: checksumAssetName, url: "https://example.test/checksums.txt", size: 1},
				},
			},
		},
	}

	manager, err := New(Config{
		Source:         source,
		OS:             "linux",
		Arch:           "amd64",
		ExecutablePath: func() (string, error) { return writeExecutable(t, []byte("old")), nil },
		ApplyRelease: func(context.Context, *upstream.Release, string) error {
			return errors.New("apply failed")
		},
		RollbackError: func(error) error {
			return errors.New("rollback failed")
		},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	_, err = manager.Run(context.Background(), "v1.0.0", Options{}, noopMeter())
	if err == nil || !strings.Contains(err.Error(), "rollback failed") {
		t.Fatalf("Run() error = %v, want rollback message", err)
	}
}

func noopMeter() verbose.Meter {
	return verbose.Meter{}
}

func setEnglishLocale(t *testing.T) {
	t.Helper()
	if err := localize.SetCurrent(localize.EN); err != nil {
		t.Fatalf("SetCurrent() error = %v", err)
	}
}

func writeExecutable(t *testing.T, content []byte) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "ragcli")
	if err := os.WriteFile(path, content, 0o755); err != nil {
		t.Fatalf("WriteFile() error = %v", err)
	}
	return path
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error = %v", err)
	}
	return string(data)
}

func checksumFile(assetName string, content []byte) []byte {
	sum := sha256.Sum256(content)
	return []byte(fmt.Sprintf("%x  %s\n", sum, assetName))
}

func tarGzExecutable(t *testing.T, name string, content []byte) []byte {
	t.Helper()

	var buf bytes.Buffer
	gz := gzip.NewWriter(&buf)
	tw := tar.NewWriter(gz)
	if err := tw.WriteHeader(&tar.Header{
		Name: name,
		Mode: 0o755,
		Size: int64(len(content)),
	}); err != nil {
		t.Fatalf("WriteHeader() error = %v", err)
	}
	if _, err := tw.Write(content); err != nil {
		t.Fatalf("Write() error = %v", err)
	}
	if err := tw.Close(); err != nil {
		t.Fatalf("tar.Close() error = %v", err)
	}
	if err := gz.Close(); err != nil {
		t.Fatalf("gzip.Close() error = %v", err)
	}
	return buf.Bytes()
}
