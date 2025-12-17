"""Main CLI entry point."""

import click
from rich.console import Console

from .commands import (
    compress,
    tokens,
    enhance,
    analyze,
    generate,
    recommend,
    list_templates,
    list_styles,
)

console = Console()


@click.group()
@click.version_option(version="1.0.0", prog_name="promptmanager")
def cli():
    """PromptManager - Production-ready prompt management.

    Control, Enhance, Compress, and Generate LLM prompts.

    \b
    Examples:
        pm compress "Your long prompt..." --ratio 0.5
        pm enhance "messy prompt" --level moderate
        pm generate "Write a Python sort function"
        pm analyze "Check this prompt quality"

    Use --help on any command for more details.
    """
    pass


# Compression commands
cli.add_command(compress)
cli.add_command(tokens)

# Enhancement commands
cli.add_command(enhance)
cli.add_command(analyze)

# Generation commands
cli.add_command(generate)
cli.add_command(recommend)
cli.add_command(list_templates)
cli.add_command(list_styles)


@cli.command()
@click.argument("prompt", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read from file")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Write to file")
@click.option("--compress/--no-compress", default=True, help="Enable compression")
@click.option("--enhance/--no-enhance", "do_enhance", default=True, help="Enable enhancement")
@click.option("--validate/--no-validate", default=True, help="Enable validation")
@click.option("-r", "--ratio", default=0.7, type=float, help="Compression ratio")
@click.option(
    "-l", "--level",
    type=click.Choice(["minimal", "light", "moderate", "aggressive"]),
    default="moderate",
    help="Enhancement level"
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def process(prompt, input_file, output_file, compress, do_enhance, validate, ratio, level, verbose):
    """Process a prompt through the full pipeline.

    Runs enhancement, compression, and validation.

    Examples:

        pm process "Your prompt here..."

        pm process -f input.txt -o output.txt --no-compress

        pm process "..." --ratio 0.5 --level aggressive -v
    """
    import asyncio
    from ..pipeline import process_prompt
    from ..enhancement import EnhancementLevel

    # Get prompt
    if input_file:
        with open(input_file) as f:
            prompt = f.read()
    elif not prompt:
        prompt = click.get_text_stream("stdin").read()

    if not prompt or not prompt.strip():
        console.print("[red]Error:[/red] No prompt provided")
        raise click.Abort()

    async def run():
        return await process_prompt(
            prompt=prompt,
            compress=compress,
            enhance=do_enhance,
            validate=validate,
            compression_ratio=ratio,
            enhancement_level=EnhancementLevel(level)
        )

    with console.status("[bold green]Processing..."):
        result = asyncio.run(run())

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(result.output_text)
        console.print(f"[green]Saved to:[/green] {output_file}")
    else:
        click.echo(result.output_text)

    if verbose:
        console.print(f"\n[bold]Pipeline Results:[/bold]")
        console.print(f"  Success: {'[green]Yes[/green]' if result.success else '[red]No[/red]'}")
        console.print(f"  Total Time: {result.total_duration_ms:.1f}ms")

        if result.step_results:
            console.print("\n[bold]Steps:[/bold]")
            for step in result.step_results:
                status = "[green]✓[/green]" if step.success else "[red]✗[/red]"
                console.print(f"  {status} {step.step_name} ({step.duration_ms:.1f}ms)")
                if step.error:
                    console.print(f"      [red]Error:[/red] {step.error}")


@cli.command()
@click.argument("prompt")
def validate(prompt):
    """Validate a prompt for security and quality issues.

    Example:

        pm validate "Your prompt here..."
    """
    from ..control import PromptValidator
    from rich.table import Table

    validator = PromptValidator()
    result = validator.validate(prompt)

    # Overall result
    status = "[green]Valid[/green]" if result.is_valid else "[red]Invalid[/red]"
    console.print(f"\n[bold]Validation Result:[/bold] {status}")
    console.print(f"[bold]Score:[/bold] {result.score:.1%}")

    if result.issues:
        table = Table(title="Issues Found")
        table.add_column("Severity", style="bold")
        table.add_column("Category")
        table.add_column("Rule")
        table.add_column("Message")

        for issue in result.issues:
            severity_color = {
                "error": "red",
                "warning": "yellow",
                "info": "blue"
            }.get(issue.severity.value, "white")

            table.add_row(
                f"[{severity_color}]{issue.severity.value}[/{severity_color}]",
                issue.category.value,
                issue.rule_name,
                issue.message
            )

        console.print(table)

        # Suggestions
        suggestions = [i.suggestion for i in result.issues if i.suggestion]
        if suggestions:
            console.print("\n[bold]Suggestions:[/bold]")
            for suggestion in suggestions:
                console.print(f"  - {suggestion}")
    else:
        console.print("[green]No issues found![/green]")


@cli.command()
@click.option("-h", "--host", default="0.0.0.0", help="Host to bind to")
@click.option("-p", "--port", default=8000, type=int, help="Port to listen on")
@click.option("--reload", is_flag=True, help="Enable auto-reload")
@click.option("-w", "--workers", default=1, type=int, help="Number of workers")
def serve(host, port, reload, workers):
    """Start the REST API server.

    Example:

        pm serve --port 8080 --reload
    """
    console.print(f"[bold]Starting PromptManager API server...[/bold]")
    console.print(f"  Host: {host}")
    console.print(f"  Port: {port}")
    console.print(f"  Workers: {workers}")
    console.print(f"  Reload: {'Yes' if reload else 'No'}")
    console.print(f"\n[dim]API docs available at http://{host}:{port}/docs[/dim]\n")

    from ..api import run_server
    run_server(host=host, port=port, reload=reload, workers=workers)


@cli.command()
def info():
    """Show information about PromptManager."""
    from rich.panel import Panel

    info_text = """[bold]PromptManager[/bold] - Production-ready LLM Prompt Management

[bold]Features:[/bold]
  • [cyan]Compression[/cyan]: Reduce token count while preserving meaning
  • [cyan]Enhancement[/cyan]: Improve prompt clarity and effectiveness
  • [cyan]Generation[/cyan]: Create optimized prompts from task descriptions
  • [cyan]Validation[/cyan]: Check for security and quality issues
  • [cyan]Control[/cyan]: Version management and template library

[bold]Deployment:[/bold]
  • Python SDK: import promptmanager
  • REST API: pm serve
  • CLI: pm <command>

[bold]Documentation:[/bold]
  API Docs: http://localhost:8000/docs (when server running)
  CLI Help: pm --help
  Command Help: pm <command> --help"""

    console.print(Panel(info_text, title="PromptManager v1.0.0", border_style="green"))


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
