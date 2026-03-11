package app

type CommonOptions struct {
	InputPath string
	Prompt    string
	Raw       bool
	Debug     bool
	Verbose   bool
}

type LLMOptions struct {
	APIURL         string
	Model          string
	EmbeddingModel string
	APIKey         string
	RetryCount     int
	ProxyURL       string
	NoProxy        bool
}

type commandInvocation struct {
	spec    *commandSpec
	Common  CommonOptions
	LLM     LLMOptions
	payload any
}

func (inv commandInvocation) Name() string {
	if inv.spec == nil {
		return ""
	}

	return inv.spec.name
}
