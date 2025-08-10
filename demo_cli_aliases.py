#!/usr/bin/env python3
"""
Demo CLI using the new auto-generator to show the alias functionality.
"""
import json
import sys
import tempfile
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer

from opensloth.cli.autogen import cli_from_pydantic
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments

app = typer.Typer(name="demo-cli", help="Demo CLI with auto-generated aliases")

@app.command()
@cli_from_pydantic(OpenSlothConfig, TrainingArguments)
def train(
    dataset: Path = typer.Argument(..., help="Path to dataset"),
    dry_run: bool = typer.Option(False, help="Show config without training"),
):
    """
    Demo training command with auto-generated CLI aliases.
    """
    # The config will be available through the decorator-injected kwargs
    pass

if __name__ == "__main__":
    app()
