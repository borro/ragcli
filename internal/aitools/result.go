package aitools

import (
	"encoding/json"
	"fmt"
)

const HintSuggestedNextOffset = "suggested_next_offset"

type EvidenceKind string

const (
	EvidenceRetrieved EvidenceKind = "retrieved"
	EvidenceVerified  EvidenceKind = "verified"
)

type Evidence struct {
	Kind       EvidenceKind
	SourcePath string
	StartLine  int
	EndLine    int
}

type ToolResult struct {
	Payload      any
	Summary      map[string]any
	ProgressKeys []string
	Hints        map[string]any
	Evidence     []Evidence
}

type Execution struct {
	Result    ToolResult
	Cached    bool
	Duplicate bool
}

type ToolResponseMeta struct {
	Cached              bool   `json:"cached,omitempty"`
	Duplicate           bool   `json:"duplicate,omitempty"`
	AlreadySeen         bool   `json:"already_seen,omitempty"`
	NovelLines          int    `json:"novel_lines"`
	ProgressHint        string `json:"progress_hint,omitempty"`
	SuggestedNextOffset int    `json:"suggested_next_offset,omitempty"`
}

type toolMessageEnvelope struct {
	Result any               `json:"result,omitempty"`
	Meta   *ToolResponseMeta `json:"meta,omitempty"`
	Error  *ToolError        `json:"error,omitempty"`
}

func MarshalJSON(v any) (string, error) {
	data, err := json.Marshal(v)
	if err != nil {
		return "", fmt.Errorf("failed to marshal tool result: %w", err)
	}
	return string(data), nil
}

func EncodeToolResult(execution Execution, meta ToolResponseMeta) (string, error) {
	meta.Cached = meta.Cached || execution.Cached
	meta.Duplicate = meta.Duplicate || execution.Duplicate
	return MarshalJSON(toolMessageEnvelope{
		Result: execution.Result.Payload,
		Meta:   &meta,
	})
}

func EncodeToolError(err error) (string, error) {
	toolErr := AsToolError(err)
	return MarshalJSON(toolMessageEnvelope{
		Error: &toolErr,
	})
}

func DecodeToolResult(raw string, out any) (ToolResponseMeta, error) {
	var envelope struct {
		Result json.RawMessage  `json:"result"`
		Meta   ToolResponseMeta `json:"meta"`
		Error  *ToolError       `json:"error"`
	}
	if err := json.Unmarshal([]byte(raw), &envelope); err != nil {
		return ToolResponseMeta{}, fmt.Errorf("failed to decode tool envelope: %w", err)
	}
	if envelope.Error != nil {
		return ToolResponseMeta{}, *envelope.Error
	}
	if out == nil {
		return envelope.Meta, nil
	}
	if len(envelope.Result) == 0 {
		return ToolResponseMeta{}, fmt.Errorf("tool envelope does not contain result payload")
	}
	if err := json.Unmarshal(envelope.Result, out); err != nil {
		return ToolResponseMeta{}, fmt.Errorf("failed to decode tool result payload: %w", err)
	}
	return envelope.Meta, nil
}

func CloneExecution(execution Execution) Execution {
	return Execution{
		Result:    CloneToolResult(execution.Result),
		Cached:    execution.Cached,
		Duplicate: execution.Duplicate,
	}
}

func CloneToolResult(result ToolResult) ToolResult {
	return ToolResult{
		Payload:      result.Payload,
		Summary:      CloneSummary(result.Summary),
		ProgressKeys: CloneStrings(result.ProgressKeys),
		Hints:        CloneHints(result.Hints),
		Evidence:     CloneEvidence(result.Evidence),
	}
}

func CloneEvidence(items []Evidence) []Evidence {
	if items == nil {
		return nil
	}
	return append([]Evidence(nil), items...)
}
