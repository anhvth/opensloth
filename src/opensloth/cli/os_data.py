# opensloth/cli/os_data.py
"""
CLI for preparing datasets for OpenSloth training (SFT, DPO, GRPO).
"""
import json
from pathlib import Path

import typer
import typer as _typer_internal

from opensloth.api import run_prepare_data
from opensloth.config.builder import TRAINING_PRESETS, PrepConfigBuilder

app = typer.Typer(name="os-data", help="Prepare datasets for OpenSloth training.", add_completion=True)

# Chat template configurations based on common formats
CHAT_TEMPLATES = {
    "chatml": {
        "description": "ChatML format used by many chat models (Qwen, OpenHermes, etc.)",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "train_on_target_only": True,
    },
}

@app.command()
def main(
    model: str = typer.Option(..., "--model", "-m", help="Base model/tokenizer to use for tokenization."),
    # Accept either a local file path OR a remote dataset name (do NOT enforce exists=True)
    input_file: str = typer.Argument(..., help="Path to an input .jsonl file OR a HuggingFace dataset name."),
    method: str = typer.Option("sft", "--method", help="Preparation method (sft, dpo, grpo). Affects required data format."),
    chat_template: str = typer.Option("chatml", "--chat-template", "-t", help="Chat template to apply (e.g., 'chatml')."),
    output: Path = typer.Option(None, "--output", "-o", help="Output directory for the processed dataset."),
    target_only: bool = typer.Option(True, "--target-only/--full-conversation", help="For SFT, mask out prompts and train only on responses."),
    samples: int = typer.Option(-1, "--samples", "-n", help="Number of samples to process (-1 for all)."),
    max_seq_length: int = typer.Option(4096, "--max-seq-length", help="Maximum sequence length."),
    gpus: int = typer.Option(1, "--gpus", "-g", help="Number of GPU shards to create for the dataset."),
    workers: int = typer.Option(8, "--workers", "-w", help="Number of parallel processing workers."),
    force: bool = typer.Option(False, "--force", "-f", help="Force overwrite of existing output directory."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the configuration and exit without running."),
):
    """
    Prepares a raw dataset for OpenSloth training by tokenizing, formatting,
    and sharding it for single or multi-GPU use.
    """

    # Defensive fallback if OptionInfo objects passed directly (misconfigured entrypoint)
    if isinstance(method, _typer_internal.models.OptionInfo):  # type: ignore
        return app()
    method_l = method.lower()
    if method_l not in {"sft", "dpo", "grpo"}:
        typer.secho(f"Error: Invalid method '{method}'. Must be 'sft', 'dpo', or 'grpo'.", fg="red")
        raise typer.Exit(1)

    # Build configuration from CLI args. We pass the raw string (can be file path or HF dataset)
    cfg_builder = PrepConfigBuilder(method=method_l).with_base(
        tok_name=model,
        dataset_name=input_file,
        num_samples=samples,
        num_proc=workers,
        max_seq_length=max_seq_length,
        gpus=gpus,
        train_on_target_only=target_only if method_l == "sft" else False,
    )

    if chat_template:
        tpl = CHAT_TEMPLATES.get(chat_template)
        if not tpl:
            typer.secho(f"Error: Unknown chat template '{chat_template}'.", fg="red")
            raise typer.Exit(1)
        cfg_builder.with_base(
            chat_template=chat_template,
            instruction_part=tpl["instruction_part"],
            response_part=tpl["response_part"],
            train_on_target_only=tpl["train_on_target_only"],
        )

    config = cfg_builder.build()

    # Determine output directory
    if not output:
        # Derive a friendly base name whether it's a path or remote dataset identifier
        base_name = Path(input_file).stem if Path(input_file).suffix else input_file.replace('/', '_')
        output_dir = Path(f"data/{method_l}_{base_name}_n{samples if samples > 0 else 'all'}")
    else:
        output_dir = output
    config["output_dir"] = str(output_dir)

    if output_dir.exists() and not force:
        typer.secho(f"Error: Output directory '{output_dir}' already exists. Use --force to overwrite.", fg="red")
        raise typer.Exit(1)

    if dry_run:
        typer.echo("DRY RUN: Configuration that would be used:")
        typer.echo(json.dumps(config, indent=2))
        return

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    typer.secho(f"🚀 Starting data preparation for '{method.upper()}'...", fg="green")
    final_output_dir = run_prepare_data(config)
    typer.secho(f"✅ Data preparation complete. Processed dataset saved to: {final_output_dir}", fg="green")

if __name__ == "__main__":
    app()
