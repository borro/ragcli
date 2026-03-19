package testutil

import (
	"flag"
	"fmt"
	"os"
	"strconv"
	"sync"
	"testing"
	"time"

	"pgregory.net/rapid"
)

const (
	mutationEnvName        = "RAGCLI_MUTATION"
	rapidChecksEnvName     = "RAGCLI_MUTATION_RAPID_CHECKS"
	rapidSeedEnvName       = "RAGCLI_MUTATION_RAPID_SEED"
	rapidShrinkTimeEnvName = "RAGCLI_MUTATION_RAPID_SHRINKTIME"
	rapidNoFailFileEnvName = "RAGCLI_MUTATION_RAPID_NOFAILFILE"
	defaultRapidChecks     = "20"
	defaultRapidSeed       = "1337"
	defaultRapidShrinkTime = "1s"
	defaultRapidNoFailFile = "true"
)

type rapidFlagSpec struct {
	flagName     string
	envName      string
	defaultValue string
}

var rapidFlagSpecs = []rapidFlagSpec{
	{flagName: "rapid.checks", envName: rapidChecksEnvName, defaultValue: defaultRapidChecks},
	{flagName: "rapid.seed", envName: rapidSeedEnvName, defaultValue: defaultRapidSeed},
	{flagName: "rapid.shrinktime", envName: rapidShrinkTimeEnvName, defaultValue: defaultRapidShrinkTime},
	{flagName: "rapid.nofailfile", envName: rapidNoFailFileEnvName, defaultValue: defaultRapidNoFailFile},
}

var (
	mutationRapidDefaultsOnce sync.Once
	errMutationRapidDefaults  error
)

// RapidCheck applies mutation-mode defaults for rapid once and then runs rapid.Check.
func RapidCheck(t *testing.T, prop func(*rapid.T)) {
	t.Helper()

	if err := applyMutationRapidDefaults(flag.CommandLine, os.LookupEnv); err != nil {
		t.Fatalf("apply mutation rapid defaults: %v", err)
	}

	rapid.Check(t, prop)
}

func applyMutationRapidDefaults(fs *flag.FlagSet, lookupEnv func(string) (string, bool)) error {
	return applyMutationRapidDefaultsWithState(fs, lookupEnv, &mutationRapidDefaultsOnce, &errMutationRapidDefaults)
}

func applyMutationRapidDefaultsWithState(fs *flag.FlagSet, lookupEnv func(string) (string, bool), once *sync.Once, applyErr *error) error {
	enabled, err := mutationEnabled(lookupEnv)
	if err != nil || !enabled {
		return err
	}

	once.Do(func() {
		*applyErr = configureMutationRapidDefaults(fs, lookupEnv)
	})

	return *applyErr
}

func configureMutationRapidDefaults(fs *flag.FlagSet, lookupEnv func(string) (string, bool)) error {
	for _, spec := range rapidFlagSpecs {
		if fs.Lookup(spec.flagName) == nil {
			return fmt.Errorf("flag %q is not registered", spec.flagName)
		}
		if flagWasExplicitlySet(fs, spec.flagName) {
			continue
		}

		value := spec.defaultValue
		if envValue, ok := lookupEnv(spec.envName); ok {
			value = envValue
		}

		if err := validateRapidFlagValue(spec.flagName, value); err != nil {
			return err
		}
		if err := fs.Set(spec.flagName, value); err != nil {
			return fmt.Errorf("set %s=%q: %w", spec.flagName, value, err)
		}
	}

	return nil
}

func mutationEnabled(lookupEnv func(string) (string, bool)) (bool, error) {
	raw, ok := lookupEnv(mutationEnvName)
	if !ok || raw == "" {
		return false, nil
	}

	enabled, err := strconv.ParseBool(raw)
	if err != nil {
		return false, fmt.Errorf("parse %s=%q: %w", mutationEnvName, raw, err)
	}

	return enabled, nil
}

func flagWasExplicitlySet(fs *flag.FlagSet, name string) bool {
	explicit := false
	fs.Visit(func(f *flag.Flag) {
		if f.Name == name {
			explicit = true
		}
	})
	return explicit
}

func validateRapidFlagValue(flagName string, value string) error {
	switch flagName {
	case "rapid.checks":
		parsed, err := strconv.Atoi(value)
		if err != nil {
			return fmt.Errorf("parse %s=%q: %w", flagName, value, err)
		}
		if parsed < 1 {
			return fmt.Errorf("%s must be >= 1, got %d", flagName, parsed)
		}
	case "rapid.seed":
		if _, err := strconv.ParseUint(value, 10, 64); err != nil {
			return fmt.Errorf("parse %s=%q: %w", flagName, value, err)
		}
	case "rapid.shrinktime":
		if _, err := time.ParseDuration(value); err != nil {
			return fmt.Errorf("parse %s=%q: %w", flagName, value, err)
		}
	case "rapid.nofailfile":
		if _, err := strconv.ParseBool(value); err != nil {
			return fmt.Errorf("parse %s=%q: %w", flagName, value, err)
		}
	default:
		return fmt.Errorf("unsupported rapid flag %q", flagName)
	}

	return nil
}
