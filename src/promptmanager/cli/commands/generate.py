"""Generation CLI commands."""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command()
@click.argument("task")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Write result to file")
@click.option(
    "-s", "--style",
    type=click.Choice(["zero_shot", "few_shot", "chain_of_thought", "code_generation"]),
    default=None,
    help="Prompt style (auto-selected if not specified)"
)
@click.option("-t", "--template", default=None, help="Specific template to use")
@click.option("-l", "--language", default=None, help="Programming language for code tasks")
@click.option("-c", "--context", default=None, help="Additional context")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def generate(task, output_file, style, template, language, context, verbose):
    """Generate an optimized prompt from a task description.

    Examples:

        pm generate "Write a Python function to sort a list"

        pm generate "Explain quantum computing" --style chain_of_thought

        pm generate "Build a REST API" --language Python -v
    """
    from ...generation import PromptGenerator, PromptStyle

    async def run_generation():
        generator = PromptGenerator()

        kwargs = {}
        if language:
            kwargs["language"] = language
        if context:
            kwargs["context"] = context

        return await generator.generate(
            task=task,
            style=PromptStyle(style) if style else None,
            template=template,
            **kwargs
        )

    with console.status("[bold green]Generating..."):
        result = asyncio.run(run_generation())

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(result.prompt)
        console.print(f"[green]Saved to:[/green] {output_file}")
    else:
        if verbose:
            console.print(Panel(result.prompt, title="Generated Prompt"))
        else:
            click.echo(result.prompt)

    if verbose:
        table = Table(title="Generation Details")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Style Used", result.style_used.value)
        table.add_row("Template", result.template_used)

        if result.style_recommendation:
            table.add_row("Style Confidence", f"{result.style_recommendation.confidence:.1%}")
            if result.style_recommendation.alternative_styles:
                alts = ", ".join(s.value for s in result.style_recommendation.alternative_styles)
                table.add_row("Alternatives", alts)

        table.add_row("Processing Time", f"{result.processing_time_ms:.1f}ms")

        console.print(table)

        if result.warnings:
            console.print("\n[yellow]Warnings:[/yellow]")
            for warning in result.warnings:
                console.print(f"  - {warning}")


@click.command()
@click.argument("task")
@click.option("--examples/--no-examples", default=False, help="Whether you have examples")
@click.option("--complexity", default=0.5, type=float, help="Task complexity (0-1)")
def recommend(task, examples, complexity):
    """Get a style recommendation for a task.

    Example:

        pm recommend "Analyze sentiment of customer reviews" --examples
    """
    from ...generation import PromptGenerator

    generator = PromptGenerator()
    rec = generator.recommend_style(
        task=task,
        has_examples=examples,
        complexity=complexity
    )

    console.print(f"\n[bold]Recommended Style:[/bold] [green]{rec.style.value}[/green]")
    console.print(f"[bold]Template:[/bold] {rec.template_name}")
    console.print(f"[bold]Confidence:[/bold] {rec.confidence:.1%}")
    console.print(f"[bold]Reasoning:[/bold] {rec.reasoning}")

    if rec.alternative_styles:
        console.print(f"\n[dim]Alternatives: {', '.join(s.value for s in rec.alternative_styles)}[/dim]")


@click.command("list-templates")
def list_templates():
    """List all available prompt templates."""
    from ...generation import PromptGenerator

    generator = PromptGenerator()
    templates = generator.list_templates()

    table = Table(title="Available Templates")
    table.add_column("Name", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Description")
    table.add_column("Required Variables")

    for t in templates:
        table.add_row(
            t["name"],
            t["type"],
            t.get("description", ""),
            ", ".join(t.get("required_variables", []))
        )

    console.print(table)


@click.command("list-styles")
def list_styles():
    """List all available prompt styles."""
    from ...generation import PromptGenerator

    generator = PromptGenerator()
    styles = generator.list_styles()

    table = Table(title="Available Styles")
    table.add_column("Style", style="cyan")
    table.add_column("Name", style="green")
    table.add_column("Description")
    table.add_column("Best For")

    for s in styles:
        table.add_row(
            s["style"],
            s["name"],
            s["description"],
            ", ".join(s["best_for"][:3]) + ("..." if len(s["best_for"]) > 3 else "")
        )

    console.print(table)
