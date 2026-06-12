"""
CLI entry point.

Usage:
    minecraft-ai build            # scrape wiki + embed + store in ChromaDB
    minecraft-ai build --test     # quick smoke-test run (50 pages only)
    minecraft-ai serve            # start FastAPI sidecar on localhost:8765
    minecraft-ai query "..."      # one-shot test query
"""

import asyncio

import click
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

console = Console()


def _make_progress() -> Progress:
    """Shared progress-bar style used across all build phases."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description:<35}"),
        BarColumn(bar_width=36),
        MofNCompleteColumn(),
        TextColumn("[progress.percentage]{task.percentage:>4.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--test", is_flag=True, help="Limit to 50 new pages for a quick smoke-test.")
@click.option(
    "--categories",
    default=None,
    help="Comma-separated category names to scrape (default: all built-in categories).",
)
def build(test: bool, categories: str | None) -> None:
    """Scrape the Minecraft Wiki, embed chunks, and store in ChromaDB."""
    asyncio.run(_build(test, categories))


async def _build(test: bool, categories: str | None) -> None:
    from minecraft_ai_helper.pipeline.chunker import chunk_page
    from minecraft_ai_helper.pipeline.ingestor import get_existing_page_titles, ingest_chunks
    from minecraft_ai_helper.pipeline.scraper import SCRAPE_CATEGORIES, collect_all_titles, fetch_pages

    cats = [c.strip() for c in categories.split(",")] if categories else SCRAPE_CATEGORIES
    max_new = 50 if test else None

    console.print(Panel("[bold green]Minecraft AI Helper — Knowledge Pipeline[/bold green]"))
    if test:
        console.print("[yellow]Test mode: capped at 50 new pages.[/yellow]")

    # ── Step 0: what's already embedded? ──────────────────────────────────────
    console.print("[dim]Checking existing knowledge base…[/dim]")
    existing_titles = get_existing_page_titles()
    if existing_titles:
        console.print(
            f"[green]  {len(existing_titles):,} pages already in DB — will be skipped.[/green]"
        )
    else:
        console.print("[dim]  DB is empty — full build.[/dim]")

    with _make_progress() as progress:

        # ── Step 1: discover category membership (fast API calls) ──────────────
        collect_task = progress.add_task("Collecting categories", total=len(cats))
        cat_map: dict[str, list[str]] = {}

        def _on_cat_done(_cat: str, _count: int) -> None:
            progress.advance(collect_task)

        cat_map = await collect_all_titles(cats, on_category_done=_on_cat_done)

        # Flatten, dedupe, exclude already-embedded, apply test cap
        seen: set[str] = set()
        titles_to_fetch: list[str] = []
        for cat_titles in cat_map.values():
            for t in cat_titles:
                if t in seen or t in existing_titles:
                    continue
                seen.add(t)
                titles_to_fetch.append(t)
                if max_new and len(titles_to_fetch) >= max_new:
                    break
            if max_new and len(titles_to_fetch) >= max_new:
                break

        if not titles_to_fetch:
            console.print("[bold green]Nothing new to embed — knowledge base is up to date![/bold green]")
            return

        console.print(
            f"[dim]  {len(titles_to_fetch):,} new pages to fetch "
            f"({len(existing_titles):,} skipped — already in DB)[/dim]"
        )

        # ── Step 2: fetch HTML for new pages ──────────────────────────────────
        fetch_task = progress.add_task("Fetching pages", total=len(titles_to_fetch))
        pages = []

        def _on_page_done(page) -> None:
            if page is not None:
                pages.append(page)
            progress.advance(fetch_task)

        await fetch_pages(titles_to_fetch, on_page_done=_on_page_done)

        if not pages:
            console.print("[yellow]No pages could be fetched.[/yellow]")
            return

        # ── Step 3: chunk ──────────────────────────────────────────────────────
        chunk_task = progress.add_task("Chunking pages", total=len(pages))
        all_chunks = []
        for page in pages:
            all_chunks.extend(chunk_page(page.title, page.url, page.html))
            progress.advance(chunk_task)

        # ── Step 4: embed + store ──────────────────────────────────────────────
        embed_task = progress.add_task("Embedding chunks", total=len(all_chunks))
        ingest_chunks(all_chunks, progress=progress, task_id=embed_task)

    # ── Summary ────────────────────────────────────────────────────────────────
    from minecraft_ai_helper.pipeline.ingestor import _get_collection
    total_in_db = _get_collection().count()
    console.print(
        Panel(
            f"[bold green]Build complete![/bold green]\n\n"
            f"  Pages fetched   : {len(pages):>6,}\n"
            f"  Chunks embedded : {len(all_chunks):>6,}\n"
            f"  Total in DB     : {total_in_db:>6,}",
            title="Knowledge Base",
        )
    )


@cli.command()
def serve() -> None:
    """Start the FastAPI sidecar server on localhost:8765."""
    from minecraft_ai_helper.server.app import serve as _serve

    console.print(Panel("[bold green]Minecraft AI Helper — Server starting…[/bold green]"))
    _serve()


@cli.command()
@click.argument("question")
def query(question: str) -> None:
    """Send a one-shot query to the pipeline and print the response."""
    from minecraft_ai_helper.agents import intent_classifier, orchestrator

    async def _run():
        intent_result = await intent_classifier.classify_intent(question)
        console.print(
            f"[cyan]Intent:[/cyan] {intent_result.intent}  "
            f"[cyan]Agents:[/cyan] {intent_result.agents_to_invoke}"
        )
        response = await orchestrator.run(question, intent_result)
        console.print(Panel(response.full_answer, title="Full Answer"))
        console.print(Panel(response.hud_answer, title="HUD Answer"))
        if response.follow_up_hints:
            console.print("[cyan]Follow-up hints:[/cyan]")
            for hint in response.follow_up_hints:
                console.print(f"  • {hint}")

    asyncio.run(_run())


if __name__ == "__main__":
    cli()
