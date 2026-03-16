package aitools

func CloneTools(tools []Tool) []Tool {
	if len(tools) == 0 {
		return nil
	}
	cloned := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		if tool == nil {
			continue
		}
		if cloneable, ok := tool.(CloneableTool); ok {
			cloned = append(cloned, cloneable.CloneTool())
			continue
		}
		cloned = append(cloned, tool)
	}
	return cloned
}
