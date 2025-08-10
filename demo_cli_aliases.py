#!/usr/bin/env python3
"""
Demo CLI using the new auto-generator to show the alias functionality.
"""
import sys
from pathlib import Path
import json
import tempfile

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import typer
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
from opensloth.cli.autogen import cli_from_pydantic

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
