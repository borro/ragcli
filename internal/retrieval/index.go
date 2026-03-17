package retrieval

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

var (
	ErrIndexSchemaMismatch = errors.New("index schema mismatch")
	ErrIndexTTLExpired     = errors.New("index ttl expired")
	ErrIndexCorrupted      = errors.New("index corrupted")
	ErrIndexAlreadyExists  = errors.New("index already exists")
)

type Manifest struct {
	SchemaVersion  int       `json:"schema_version"`
	CreatedAt      time.Time `json:"created_at"`
	InputHash      string    `json:"input_hash"`
	SourcePath     string    `json:"source_path"`
	EmbeddingModel string    `json:"embedding_model"`
	ChunkSize      int       `json:"chunk_size"`
	ChunkOverlap   int       `json:"chunk_overlap"`
	ItemCount      int       `json:"item_count"`
}

type Filenames struct {
	Manifest   string
	Items      string
	Embeddings string
}

func WrapIndexError(scope string, err error, corruptedDetail string) error {
	switch {
	case errors.Is(err, ErrIndexSchemaMismatch):
		return fmt.Errorf("%s %w", scope, ErrIndexSchemaMismatch)
	case errors.Is(err, ErrIndexTTLExpired):
		return fmt.Errorf("%s %w", scope, ErrIndexTTLExpired)
	case errors.Is(err, ErrIndexCorrupted):
		if corruptedDetail == "" {
			return fmt.Errorf("%s %w", scope, ErrIndexCorrupted)
		}
		return fmt.Errorf("%s %w: %s", scope, ErrIndexCorrupted, corruptedDetail)
	default:
		return err
	}
}

func CleanupExpiredIndexes(baseDir string, ttl time.Duration) {
	entries, err := os.ReadDir(baseDir)
	if err != nil {
		return
	}
	now := time.Now()
	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		info, err := entry.Info()
		if err != nil {
			continue
		}
		if now.Sub(info.ModTime()) <= ttl {
			continue
		}
		_ = os.RemoveAll(filepath.Join(baseDir, entry.Name()))
	}
}

func PersistIndex[T any](indexDir string, manifest Manifest, items []T, embeddings [][]float32, filenames Filenames) error {
	parentDir := filepath.Dir(indexDir)
	if err := os.MkdirAll(parentDir, 0o700); err != nil {
		return fmt.Errorf("failed to create parent index dir: %w", err)
	}

	tempDir, err := os.MkdirTemp(parentDir, "."+filepath.Base(indexDir)+".tmp-*")
	if err != nil {
		return fmt.Errorf("failed to create temp index dir: %w", err)
	}
	defer func() {
		_ = os.RemoveAll(tempDir)
	}()

	if err := os.Chmod(tempDir, 0o700); err != nil {
		return fmt.Errorf("failed to secure temp index dir: %w", err)
	}

	if _, err := os.Stat(indexDir); err == nil {
		return ErrIndexAlreadyExists
	} else if !errors.Is(err, os.ErrNotExist) {
		return fmt.Errorf("failed to stat index dir: %w", err)
	}

	if err := writeJSONLines(filepath.Join(tempDir, filenames.Items), items, "items"); err != nil {
		return err
	}
	if err := writeJSONLines(filepath.Join(tempDir, filenames.Embeddings), embeddings, "embeddings"); err != nil {
		return err
	}
	if err := writeJSONFile(filepath.Join(tempDir, filenames.Manifest), manifest, "manifest"); err != nil {
		return err
	}

	if err := os.Rename(tempDir, indexDir); err != nil {
		if errors.Is(err, os.ErrExist) {
			return ErrIndexAlreadyExists
		}
		return fmt.Errorf("failed to publish index dir: %w", err)
	}
	return nil
}

func LoadIndex[T any](indexDir string, ttl time.Duration, expectedSchema int, filenames Filenames) (Manifest, []T, [][]float32, error) {
	var zero Manifest

	manifest, err := readManifest(filepath.Join(indexDir, filenames.Manifest))
	if err != nil {
		return zero, nil, nil, err
	}
	if manifest.SchemaVersion != expectedSchema {
		return zero, nil, nil, ErrIndexSchemaMismatch
	}
	if time.Since(manifest.CreatedAt) > ttl {
		return zero, nil, nil, ErrIndexTTLExpired
	}

	items, err := readJSONLines[T](filepath.Join(indexDir, filenames.Items), 2*1024*1024, "items")
	if err != nil {
		return zero, nil, nil, err
	}
	embeddings, err := readJSONLines[[]float32](filepath.Join(indexDir, filenames.Embeddings), 8*1024*1024, "embeddings")
	if err != nil {
		return zero, nil, nil, err
	}
	if len(items) != len(embeddings) {
		return zero, nil, nil, fmt.Errorf("%w: items and embeddings count mismatch", ErrIndexCorrupted)
	}
	return manifest, items, embeddings, nil
}

func readManifest(path string) (Manifest, error) {
	var manifest Manifest

	manifestData, err := os.ReadFile(path)
	if err != nil {
		return manifest, fmt.Errorf("failed to read manifest file: %w", err)
	}
	if err := json.Unmarshal(manifestData, &manifest); err != nil {
		return manifest, fmt.Errorf("failed to decode manifest: %w", err)
	}
	return manifest, nil
}

func writeJSONFile(path string, value any, label string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("failed to create %s file: %w", label, err)
	}
	if err := json.NewEncoder(file).Encode(value); err != nil {
		_ = file.Close()
		return fmt.Errorf("failed to write %s: %w", label, err)
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("failed to close %s file: %w", label, err)
	}
	return nil
}

func writeJSONLines[T any](path string, values []T, label string) error {
	file, err := os.OpenFile(path, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0o600)
	if err != nil {
		return fmt.Errorf("failed to create %s file: %w", label, err)
	}

	encoder := json.NewEncoder(file)
	for _, value := range values {
		if err := encoder.Encode(value); err != nil {
			_ = file.Close()
			return fmt.Errorf("failed to write %s: %w", label, err)
		}
	}
	if err := file.Close(); err != nil {
		return fmt.Errorf("failed to close %s file: %w", label, err)
	}
	return nil
}

func readJSONLines[T any](path string, maxTokenSize int, label string) ([]T, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open %s file: %w", label, err)
	}
	defer func() {
		_ = file.Close()
	}()

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), maxTokenSize)
	values := make([]T, 0, 32)
	for scanner.Scan() {
		var value T
		if err := json.Unmarshal(scanner.Bytes(), &value); err != nil {
			return nil, fmt.Errorf("failed to decode %s entry: %w", label, err)
		}
		values = append(values, value)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("failed to scan %s file: %w", label, err)
	}
	return values, nil
}
