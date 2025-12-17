"""Enhancement CLI commands."""

import asyncio
import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


@click.command()
@click.argument("prompt", required=False)
@click.option("-f", "--file", "input_file", type=click.Path(exists=True), help="Read prompt from file")
@click.option("-o", "--output", "output_file", type=click.Path(), help="Write result to file")
@click.option(
    "-m", "--mode",
    type=click.Choice(["rules_only", "llm_only", "hybrid", "adaptive"]),
    default="rules_only",
    help="Enhancement mode"
)
@click.option(
    "-l", "--level",
    type=click.Choice(["minimal", "light", "moderate", "aggressive"]),
    default="moderate",
    help="Enhancement level"
)
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def enhance(prompt, input_file, output_file, mode, level, verbose):
    """Enhance a prompt for clarity and effectiveness.

    Examples:

        pm enhance "messy prompt here"

        pm enhance -f prompt.txt --level aggressive

        pm enhance "..." --mode rules_only -v
    """
    # Get prompt
    if input_file:
        with open(input_file) as f:
            prompt = f.read()
    elif not prompt:
        prompt = click.get_text_stream("stdin").read()

    if not prompt or not prompt.strip():
        console.print("[red]Error:[/red] No prompt provided")
        raise click.Abort()

    from ...enhancement import PromptEnhancer, EnhancementMode, EnhancementLevel

    async def run_enhancement():
        enhancer = PromptEnhancer()
        return await enhancer.enhance(
            prompt,
            mode=EnhancementMode(mode),
            level=EnhancementLevel(level)
        )

    with console.status("[bold green]Enhancing..."):
        result = asyncio.run(run_enhancement())

    # Output
    if output_file:
        with open(output_file, "w") as f:
            f.write(result.enhanced_prompt)
        console.print(f"[green]Saved to:[/green] {output_file}")
    else:
        if verbose:
            console.print(Panel(result.enhanced_prompt, title="Enhanced Prompt"))
        else:
            click.echo(result.enhanced_prompt)

    if verbose:
        table = Table(title="Enhancement Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        if result.detected_intent:
            table.add_row("Detected Intent", result.detected_intent.value)
            table.add_row("Intent Confidence", f"{result.intent_confidence:.1%}")

        if result.final_quality:
            table.add_row("Quality Score", f"{result.final_quality.overall_score:.1%}")
        table.add_row("Quality Improvement", f"{result.quality_improvement:+.1%}")
        table.add_row("Rules Applied", ", ".join(result.applied_rules) or "None")
        table.add_row("LLM Enhanced", "Yes" if result.llm_enhanced else "No")

        console.print(table)

        if result.suggestions:
            console.print("\n[bold]Suggestions:[/bold]")
            for suggestion in result.suggestions:
                console.print(f"  - {suggestion}")


@click.command()
@click.argument("prompt")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed scores")
def analyze(prompt, verbose):
    """Analyze a prompt without modifying it.

    Shows intent detection and quality scores.

    Example:

        pm analyze "Your prompt here..."
    """
    from ...enhancement import PromptEnhancer

    async def run_analysis():
        enhancer = PromptEnhancer()
        return await enhancer.analyze(prompt)

    with console.status("[bold green]Analyzing..."):
        result = asyncio.run(run_analysis())

    # Intent
    console.print("\n[bold]Intent Detection:[/bold]")
    console.print(f"  Primary: [green]{result['intent']['primary']}[/green]")
    console.print(f"  Confidence: [cyan]{result['intent']['confidence']:.1%}[/cyan]")

    if verbose and result['intent']['all_scores']:
        console.print("  All scores:")
        for intent, score in sorted(result['intent']['all_scores'].items(), key=lambda x: -x[1])[:5]:
            console.print(f"    - {intent}: {score:.1%}")

    # Quality
    console.print("\n[bold]Quality Scores:[/bold]")
    quality = result['quality']
    table = Table(show_header=False, box=None)
    table.add_column("Dimension", style="cyan", width=15)
    table.add_column("Score", style="green", width=10)

    table.add_row("Overall", f"{quality['overall_score']:.1%}")
    table.add_row("Clarity", f"{quality['clarity']:.1%}")
    table.add_row("Structure", f"{quality['structure']:.1%}")
    table.add_row("Completeness", f"{quality['completeness']:.1%}")
    table.add_row("Specificity", f"{quality['specificity']:.1%}")
    table.add_row("Grammar", f"{quality['grammar']:.1%}")

    console.print(table)

    # Issues and suggestions
    if quality['issues']:
        console.print("\n[bold yellow]Issues:[/bold yellow]")
        for issue in quality['issues']:
            console.print(f"  - {issue}")

    if quality['suggestions']:
        console.print("\n[bold]Suggestions:[/bold]")
        for suggestion in quality['suggestions']:
            console.print(f"  - {suggestion}")

    # Stats
    if verbose:
        console.print("\n[bold]Statistics:[/bold]")
        stats = result['statistics']
        console.print(f"  Length: {stats['length']} chars")
        console.print(f"  Words: {stats['word_count']}")
        console.print(f"  Sentences: {stats['sentence_count']}")
