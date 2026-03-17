package selfupdate

type Options struct {
	CheckOnly bool
}

type Result struct {
	CurrentVersion string
	LatestVersion  string
	Checked        bool
	Updated        bool
	Output         string
}
