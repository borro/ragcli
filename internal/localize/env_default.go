package localize

import "os"

func syscallLookupEnv(key string) (string, bool) {
	return os.LookupEnv(key)
}
