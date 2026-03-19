package testutil

import (
	"flag"
	"io"
	"sync"
	"testing"
	"time"
)

func TestApplyMutationRapidDefaults_NoMutationEnvLeavesDefaults(t *testing.T) {
	fs := newRapidFlagSet(t)
	var once sync.Once
	var applyErr error

	if err := applyMutationRapidDefaultsWithState(fs, envLookup(map[string]string{}), &once, &applyErr); err != nil {
		t.Fatalf("applyMutationRapidDefaultsWithState() error = %v", err)
	}

	assertFlagValue(t, fs, "rapid.checks", "100")
	assertFlagValue(t, fs, "rapid.seed", "0")
	assertFlagValue(t, fs, "rapid.shrinktime", "30s")
	assertFlagValue(t, fs, "rapid.nofailfile", "false")
}

func TestConfigureMutationRapidDefaults_AppliesMutationDefaults(t *testing.T) {
	fs := newRapidFlagSet(t)

	if err := configureMutationRapidDefaults(fs, envLookup(map[string]string{})); err != nil {
		t.Fatalf("configureMutationRapidDefaults() error = %v", err)
	}

	assertFlagValue(t, fs, "rapid.checks", defaultRapidChecks)
	assertFlagValue(t, fs, "rapid.seed", defaultRapidSeed)
	assertFlagValue(t, fs, "rapid.shrinktime", defaultRapidShrinkTime)
	assertFlagValue(t, fs, "rapid.nofailfile", defaultRapidNoFailFile)
}

func TestConfigureMutationRapidDefaults_ExplicitEnvOverridesDefaults(t *testing.T) {
	fs := newRapidFlagSet(t)

	if err := configureMutationRapidDefaults(fs, envLookup(map[string]string{
		rapidChecksEnvName:     "7",
		rapidSeedEnvName:       "42",
		rapidShrinkTimeEnvName: "2s",
		rapidNoFailFileEnvName: "false",
	})); err != nil {
		t.Fatalf("configureMutationRapidDefaults() error = %v", err)
	}

	assertFlagValue(t, fs, "rapid.checks", "7")
	assertFlagValue(t, fs, "rapid.seed", "42")
	assertFlagValue(t, fs, "rapid.shrinktime", "2s")
	assertFlagValue(t, fs, "rapid.nofailfile", "false")
}

func TestConfigureMutationRapidDefaults_ExplicitFlagsWinOverEnvAndDefaults(t *testing.T) {
	fs := newRapidFlagSet(t)
	if err := fs.Set("rapid.checks", "9"); err != nil {
		t.Fatalf("fs.Set(rapid.checks) error = %v", err)
	}
	if err := fs.Set("rapid.seed", "11"); err != nil {
		t.Fatalf("fs.Set(rapid.seed) error = %v", err)
	}
	if err := fs.Set("rapid.shrinktime", "3s"); err != nil {
		t.Fatalf("fs.Set(rapid.shrinktime) error = %v", err)
	}
	if err := fs.Set("rapid.nofailfile", "false"); err != nil {
		t.Fatalf("fs.Set(rapid.nofailfile) error = %v", err)
	}

	if err := configureMutationRapidDefaults(fs, envLookup(map[string]string{
		rapidChecksEnvName:     "7",
		rapidSeedEnvName:       "42",
		rapidShrinkTimeEnvName: "2s",
		rapidNoFailFileEnvName: "true",
	})); err != nil {
		t.Fatalf("configureMutationRapidDefaults() error = %v", err)
	}

	assertFlagValue(t, fs, "rapid.checks", "9")
	assertFlagValue(t, fs, "rapid.seed", "11")
	assertFlagValue(t, fs, "rapid.shrinktime", "3s")
	assertFlagValue(t, fs, "rapid.nofailfile", "false")
}

func TestMutationEnabled_InvalidValueErrors(t *testing.T) {
	enabled, err := mutationEnabled(envLookup(map[string]string{
		mutationEnvName: "maybe",
	}))
	if err == nil {
		t.Fatal("mutationEnabled() error = nil, want parse error")
	}
	if enabled {
		t.Fatal("mutationEnabled() = true, want false on parse error")
	}
}

func newRapidFlagSet(t *testing.T) *flag.FlagSet {
	t.Helper()

	fs := flag.NewFlagSet("rapid-test", flag.ContinueOnError)
	fs.SetOutput(io.Discard)
	fs.Int("rapid.checks", 100, "")
	fs.Uint64("rapid.seed", 0, "")
	fs.Duration("rapid.shrinktime", 30*time.Second, "")
	fs.Bool("rapid.nofailfile", false, "")
	return fs
}

func envLookup(values map[string]string) func(string) (string, bool) {
	return func(name string) (string, bool) {
		value, ok := values[name]
		return value, ok
	}
}

func assertFlagValue(t *testing.T, fs *flag.FlagSet, name string, want string) {
	t.Helper()

	gotFlag := fs.Lookup(name)
	if gotFlag == nil {
		t.Fatalf("flag %q is not registered", name)
	}
	if gotFlag.Value.String() != want {
		t.Fatalf("%s = %q, want %q", name, gotFlag.Value.String(), want)
	}
}
