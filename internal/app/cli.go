package app

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/borro/ragcli/internal/localize"
	"github.com/urfave/cli/v3"
)

type commandExecutor func(context.Context, commandInvocation) error

type cliConfig struct {
	stdout  io.Writer
	stderr  io.Writer
	version string
	execute commandExecutor
}

func newCLI(cfg cliConfig) *cli.Command {
	cli.VersionPrinter = printVersion
	cli.ShowCommandHelp = showCommandHelp
	cli.VersionFlag = &cli.BoolFlag{
		Name:        "version",
		Usage:       localize.T("cli.version.flag.usage"),
		HideDefault: true,
		Local:       true,
	}

	return &cli.Command{
		Name:                  "ragcli",
		Usage:                 localize.T("cli.root.usage"),
		Version:               strings.TrimSpace(cfg.version),
		Flags:                 globalFlags(),
		Action:                showRootHelp,
		OnUsageError:          showUsageHelp,
		Before:                validateGlobalLocale,
		Commands:              buildCLICommands(cfg),
		Writer:                cfg.stdout,
		ErrWriter:             cfg.stderr,
		ExitErrHandler:        suppressExitError,
		HideHelp:              false,
		EnableShellCompletion: true,
	}
}

func buildCLICommands(cfg cliConfig) []*cli.Command {
	commands := make([]*cli.Command, 0, len(commandSpecs())+1)
	for _, spec := range commandSpecs() {
		commands = append(commands, newCLICommand(spec, buildFlags(spec.flagSpecs), cfg.execute))
	}
	commands = append(commands, newVersionCLICommand())

	for _, cmd := range commands {
		cmd.OnUsageError = showUsageHelp
	}

	return commands
}

func newCLICommand(spec *commandSpec, flags []cli.Flag, execute commandExecutor) *cli.Command {
	var arguments []cli.Argument
	if !spec.noPrompt {
		arguments = promptArguments()
	}

	return &cli.Command{
		Name:        spec.name,
		Usage:       spec.usage,
		Description: spec.description,
		Arguments:   arguments,
		Flags:       flags,
		Action:      bindAndExecute(spec, execute),
	}
}

func promptArguments() []cli.Argument {
	return []cli.Argument{
		&cli.StringArgs{Name: "prompt", Min: 0, Max: -1, UsageText: "<prompt>"},
	}
}

func bindAndExecute(spec *commandSpec, execute commandExecutor) cli.ActionFunc {
	return func(ctx context.Context, cmd *cli.Command) error {
		bound, err := spec.bind(cmd, spec)
		if err != nil {
			_ = cli.ShowSubcommandHelp(cmd)
			return err
		}
		return execute(ctx, bound)
	}
}

func showRootHelp(_ context.Context, cmd *cli.Command) error {
	return cli.ShowRootCommandHelp(cmd)
}

func newVersionCLICommand() *cli.Command {
	return &cli.Command{
		Name:  "version",
		Usage: localize.T("cli.command.version.usage"),
		Action: func(_ context.Context, cmd *cli.Command) error {
			cli.ShowVersion(cmd.Root())
			return nil
		},
	}
}

func printVersion(cmd *cli.Command) {
	_, _ = fmt.Fprintln(cmd.Root().Writer, strings.TrimSpace(cmd.Version))
}

// Workaround for urfave/cli/v3: built-in "help <command>" treats leaf commands
// as subcommands because of the hidden help command and drops GLOBAL OPTIONS.
func showCommandHelp(ctx context.Context, cmd *cli.Command, commandName string) error {
	for _, subCmd := range cmd.Commands {
		if !subCmd.HasName(commandName) {
			continue
		}

		tmpl := subCmd.CustomHelpTemplate
		if tmpl == "" {
			if len(subCmd.VisibleCommands()) == 0 {
				tmpl = cli.CommandHelpTemplate
			} else {
				tmpl = cli.SubcommandHelpTemplate
			}
		}

		cli.HelpPrinter(cmd.Root().Writer, tmpl, subCmd)
		return nil
	}

	return cli.DefaultShowCommandHelp(ctx, cmd, commandName)
}

func globalFlags() []cli.Flag {
	return buildFlags(globalFlagSpecs())
}

func showUsageHelp(_ context.Context, cmd *cli.Command, err error, _ bool) error {
	if cmd.Root() == cmd {
		_ = cli.ShowRootCommandHelp(cmd)
		return err
	}

	_ = cli.ShowSubcommandHelp(cmd)
	return err
}

func suppressExitError(context.Context, *cli.Command, error) {}

func validateGlobalLocale(ctx context.Context, cmd *cli.Command) (context.Context, error) {
	if !cmd.IsSet("lang") {
		return ctx, nil
	}
	if _, valid := localize.Normalize(cmd.String("lang")); valid {
		return ctx, nil
	}
	return ctx, fmt.Errorf("%s", localize.T("error.locale.unsupported", localize.Data{"Value": cmd.String("lang")}))
}
