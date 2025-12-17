"""Compression CLI commands."""

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
@click.option("-r", "--ratio", default=0.7, type=float, help="Target compression ratio (0.1-1.0)")
@click.option(
    "-s", "--strategy",
    type=click.Choice(["lexical", "statistical", "code", "hybrid"]),
    default="hybrid",
    help="Compression strategy"
)
@click.option("-m", "--model", default="gpt-4", help="Model for tokenization")
@click.option("--preserve-code/--no-preserve-code", default=True, help="Preserve code blocks")
@click.option("-v", "--verbose", is_flag=True, help="Show detailed output")
def compress(prompt, input_file, output_file, ratio, strategy, model, preserve_code, verbose):
    """Compress a prompt to reduce token count.

    Provide prompt as argument or use -f to read from file.

    Examples:

        pm compress "Your long prompt here..."

        pm compress -f prompt.txt -o compressed.txt

        pm compress "..." --ratio 0.5 --strategy lexical
    """
    # Get prompt from argument or file
    if input_file:
        with open(input_file) as f:
            prompt = f.read()
    elif not prompt:
        prompt = click.get_text_stream("stdin").read()

    if not prompt or not prompt.strip():
        console.print("[red]Error:[/red] No prompt provided")
        raise click.Abort()

    # Import here to avoid slow startup
    from ...compression import PromptCompressor, StrategyType

    async def run_compression():
        compressor = PromptCompressor(model=model)
        return await compressor.compress(
            prompt,
            target_ratio=ratio,
            strategy=StrategyType(strategy),
            preserve_code=preserve_code
        )

    with console.status("[bold green]Compressing..."):
        result = asyncio.run(run_compression())

    # Output result
    if output_file:
        with open(output_file, "w") as f:
            f.write(result.compressed_text)
        console.print(f"[green]Saved to:[/green] {output_file}")
    else:
        if verbose:
            console.print(Panel(result.compressed_text, title="Compressed Prompt"))
        else:
            click.echo(result.compressed_text)

    if verbose:
        table = Table(title="Compression Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Original Tokens", str(result.original_tokens))
        table.add_row("Compressed Tokens", str(result.compressed_tokens))
        table.add_row("Compression Ratio", f"{result.compression_ratio:.1%}")
        table.add_row("Strategy Used", result.strategy_used)
        table.add_row("Original Length", f"{len(prompt)} chars")
        table.add_row("Compressed Length", f"{len(result.compressed_text)} chars")

        console.print(table)


@click.command()
@click.argument("prompt")
@click.option("-m", "--model", default="gpt-4", help="Model for tokenization")
def tokens(prompt, model):
    """Count tokens in a prompt.

    Example:

        pm tokens "Your prompt here..."
    """
    from ...compression import PromptCompressor

    compressor = PromptCompressor(model=model)
    count = compressor.count_tokens(prompt)

    console.print(f"[bold]Token count:[/bold] [green]{count}[/green]")
    console.print(f"[dim]Model: {model}[/dim]")
