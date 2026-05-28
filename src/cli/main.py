"""Typer CLI entry point.

Commands implemented in Phase 0:

* `pka chat`     — interactive REPL against the multi-agent crew.
* `pka version`  — print the package version.

Phase 1 will add `ingest`, `list-sessions`, `export`, etc.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from src import __version__
from src.agent.conversation_manager import ConversationManager
from src.agent.orchestrator import KnowledgeAgent
from src.agent.tools import build_default_tools
from src.config import get_settings
from src.utils import configure_logging, get_logger

app = typer.Typer(
    name="pka",
    help="Personal Knowledge Agent — multi-turn conversational RAG over your documents.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
logger = get_logger(__name__)


@app.command()
def version() -> None:
    """Print the package version."""
    console.print(f"pka {__version__}")


@app.command()
def chat(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show CrewAI's per-step reasoning."
    ),
    show_tokens: bool = typer.Option(
        True, "--show-tokens/--no-show-tokens", help="Show token usage per turn."
    ),
) -> None:
    """Start an interactive multi-turn chat session.

    Type `/exit` (or Ctrl-D) to quit, `/new` to start a fresh session.
    """
    configure_logging()
    settings = get_settings()

    if not settings.has_gemini:
        console.print(
            "[red]GOOGLE_API_KEY is not set.[/red] Copy `env.example` to "
            "`.env` and add your key. See README.md."
        )
        raise typer.Exit(code=1)

    settings.ensure_storage_dirs()

    # Build the crew once and reuse it across turns. Constructing the LLM
    # is cheap but not free, so this matters for snappy interactive use.
    tools = build_default_tools()
    agent = KnowledgeAgent(tools, settings=settings, verbose=verbose)
    manager = ConversationManager(agent=agent, settings=settings)

    _print_welcome(manager.session.id)

    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            return

        cmd = user_input.strip()
        if not cmd:
            continue
        if cmd in {"/exit", "/quit"}:
            console.print("[dim]Goodbye.[/dim]")
            return
        if cmd == "/new":
            manager = ConversationManager(agent=agent, settings=settings)
            console.print(f"[dim]Started new session {manager.session.id}.[/dim]\n")
            continue
        if cmd == "/help":
            _print_help()
            continue

        # Real turn.
        with console.status("[dim]Thinking...[/dim]", spinner="dots"):
            try:
                turn = manager.process_turn(cmd)
            except Exception as e:  # noqa: BLE001 — never let the REPL die
                console.print(f"[red]Unexpected error:[/red] {e}")
                logger.exception("Unhandled error in chat loop")
                continue

        console.print()
        console.print(
            Panel(
                Markdown(turn.agent_response),
                title="Agent",
                title_align="left",
                border_style="green",
            )
        )
        if show_tokens:
            console.print(
                f"[dim]turn {turn.turn_number} · "
                f"{turn.token_usage.total_tokens} tokens · "
                f"{turn.metadata.get('latency_ms', 0):.0f}ms[/dim]\n"
            )


def _print_welcome(session_id: str) -> None:
    console.print(
        Panel.fit(
            (
                "[bold]Personal Knowledge Agent[/bold]\n"
                f"session: [cyan]{session_id}[/cyan]\n"
                "Type [bold]/help[/bold] for commands, [bold]/exit[/bold] to quit."
            ),
            border_style="cyan",
        )
    )


def _print_help() -> None:
    console.print(
        "[bold]Commands[/bold]\n"
        "  /new   — start a new session (clears history)\n"
        "  /help  — show this help\n"
        "  /exit  — quit (also Ctrl-D)\n"
    )


def main() -> None:
    """Console-script entry point (`uv run pka`)."""
    try:
        app()
    except SystemExit:
        raise
    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
