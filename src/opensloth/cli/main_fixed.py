"""Thin Typer CLI entrypoint delegating to builder & API layers."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from opensloth.config.builder import TrainingConfigBuilder, PrepConfigBuilder, TRAINING_PRESETS
from opensloth.api import run_training, run_prepare_data

app = typer.Typer(name="os", add_completion=True, help="OpenSloth CLI")

def _generate_output_dir(model_name: str, dataset_path: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model_name.split("/")[-1].replace("-", "_").lower()
    dataset_short = dataset_path.split("/")[-1].replace("-", "_").lower()
    return f"outputs/{model_short}_{dataset_short}_{timestamp}"

# Chat template configurations based on common formats
CHAT_TEMPLATES = {
    "chatml": {
        "description": "ChatML format used by many chat models (Qwen, OpenHermes, etc.)",
        "instruction_part": "<|im_start|>user\n",
        "response_part": "<|im_start|>assistant\n",
        "train_on_target_only": True,
    },
}

@app.command("prepare-data")
def prepare_data(
    model: str = typer.Option(..., "--model", "-m"),
    method: str = typer.Option("sft", "--method", "-M"),
    input_file: Optional[str] = typer.Argument(None),
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d"),
    chat_template: Optional[str] = typer.Option(None, "--chat-template", "-t"),
    target_only: bool = typer.Option(False, "--target-only/--full-conversation"),
    split: str = typer.Option("train"),
    samples: int = typer.Option(1000, "--samples", "-n"),
    max_seq_length: int = typer.Option(4096, "--max-seq-length"),
    workers: int = typer.Option(4, "--workers", "-w"),
    gpus: int = typer.Option(1, "--gpus", "-g"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    debug: int = typer.Option(0, "--debug"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    force: bool = typer.Option(False, "--force", "-f"),
):
    method_l = method.lower()
    if method_l not in {"sft", "dpo", "grpo"}:
        raise typer.Exit(1)
    actual_dataset = input_file or dataset
    if method_l in {"sft", "grpo"} and not actual_dataset:
        raise typer.Exit(1)
    cfg_builder = PrepConfigBuilder(method=method_l).with_base(
        tok_name=model,
        dataset_name=actual_dataset,
        split=split,
        num_samples=samples,
        num_proc=workers,
        max_seq_length=max_seq_length,
        debug=debug,
        gpus=gpus,
        train_on_target_only=bool(target_only) if method_l == "sft" else False,
    )
    if chat_template:
        tpl = CHAT_TEMPLATES.get(chat_template)
        if not tpl:
            raise typer.Exit(1)
        cfg_builder.with_base(chat_template=chat_template, instruction_part=tpl["instruction_part"], response_part=tpl["response_part"], train_on_target_only=tpl["train_on_target_only"])
    config = cfg_builder.build()
    if not output:
        base = (actual_dataset or method_l).split("/")[-1]
        output = f"data/{method_l}_{base}_n{samples if samples>0 else 'all'}"
    config["output_dir"] = output
    if Path(output).exists() and not force:
        raise typer.Exit(1)
    if dry_run:
        typer.echo(json.dumps(config, indent=2))
        return
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    out_dir = run_prepare_data(config)
    typer.echo(out_dir)

@app.command("train")
def train(
    dataset: str = typer.Argument(help="Path to processed dataset directory"),
    method: str = typer.Option("sft", "--method", "-M"),
    model: Optional[str] = typer.Option(None, "--model", "-m"),
    output: Optional[str] = typer.Option(None, "--output", "-o"),
    epochs: Optional[int] = typer.Option(None, "--epochs"),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    grad_accum: Optional[int] = typer.Option(None, "--grad-accum"),
    learning_rate: Optional[float] = typer.Option(None, "--lr"),
    max_seq_length: Optional[int] = typer.Option(None, "--max-seq-length"),
    load_4bit: bool = typer.Option(True, "--4bit/--no-4bit"),
    full_finetune: bool = typer.Option(False, "--full-finetune"),
    lora_r: Optional[int] = typer.Option(None, "--lora-r"),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha"),
    preset: Optional[str] = typer.Option(None, "--preset"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    use_tmux: bool = typer.Option(False, "--use-tmux/--no-tmux"),
    tmux_session: Optional[str] = typer.Option(None, "--tmux-session"),
    tmux_auto_kill: bool = typer.Option(False, "--tmux-auto-kill/--no-tmux-auto-kill"),
):
    if not Path(dataset).exists():
        raise typer.Exit(1)
    cli_args = dict(
        model=model,
        output=output,
        epochs=epochs,
        max_steps=max_steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        max_seq_length=max_seq_length,
        load_4bit=load_4bit,
        full_finetune=full_finetune,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )
    builder = (
        TrainingConfigBuilder(dataset_path=dataset, method=method)
        .with_preset(preset)
        .with_cli_args(**cli_args)
        .infer_from_dataset()
        .finalise()
    )
    opensloth_cfg, train_args = builder.build()
    if not output:
        train_args.output_dir = _generate_output_dir(opensloth_cfg.fast_model_args.model_name, dataset)
    summary = {"opensloth_config": opensloth_cfg.model_dump(), "training_args": train_args.model_dump()}
    if dry_run:
        typer.echo(json.dumps(summary, indent=2))
        return
    run_training(opensloth_cfg, train_args, use_tmux=use_tmux, tmux_session=tmux_session, tmux_auto_kill=tmux_auto_kill)

@app.command("list-presets")
def list_presets():
    for name, info in TRAINING_PRESETS.items():
        typer.echo(f"{name}: {info['description']}")

@app.command("list-templates")
def list_templates():
    """📋 List all available chat templates"""
    typer.echo("Available Chat Templates:")
    for name, info in CHAT_TEMPLATES.items():
        typer.echo(f"  {name}: {info['description']}")

@app.command("list-datasets")
def list_datasets():
    """📂 List available processed datasets"""
    data_dir = Path("data")
    
    if not data_dir.exists():
        typer.echo("📂 No data directory found")
        return
    
    datasets = []
    for item in data_dir.iterdir():
        if not item.is_dir():
            continue
            
        # Check if it looks like a processed dataset
        dataset_indicators = ["dataset_info.json", "state.json"]
        has_indicator = any((item / f).exists() for f in dataset_indicators)
        has_arrow = any(f.suffix == ".arrow" for f in item.glob("*.arrow"))
        
        if has_indicator or has_arrow:
            datasets.append(str(item))
    
    if not datasets:
        typer.echo("📂 No processed datasets found in the data directory")
        return
    
    typer.echo("Available Processed Datasets:")
    for dataset in sorted(datasets):
        typer.echo(f"  📁 {dataset}")
    
    typer.echo(f"\n💡 Use any dataset path with: opensloth train --dataset <path>")

@app.command("info")  
def dataset_info(
    path: str = typer.Argument(help="Path to processed dataset directory")
):
    """📖 Show information about a processed dataset"""
    dataset_path = Path(path)
    
    if not dataset_path.exists():
        typer.echo(f"❌ Dataset directory not found: {path}", err=True)
        raise typer.Exit(1)
    
    # Load dataset info
    info = {}
    
    config_file = dataset_path / "dataset_config.json"
    if config_file.exists():
        with open(config_file) as f:
            info.update(json.load(f))
    
    dataset_info_file = dataset_path / "dataset_info.json"
    if dataset_info_file.exists():
        with open(dataset_info_file) as f:
            dataset_info_data = json.load(f)
            info.update(dataset_info_data)
    
    if not info:
        typer.echo(f"❌ No dataset information found in {path}", err=True)
        raise typer.Exit(1)
    
    # Display information
    typer.echo(f"📊 Dataset Information for: {dataset_path}")
    typer.echo(f"📁 Source Dataset: {info.get('dataset_name', 'Unknown')}")
    typer.echo(f"🤖 Model/Tokenizer: {info.get('tok_name', 'Unknown')}")
    typer.echo(f"🔢 Samples: {info.get('num_samples', 'Unknown')}")
    typer.echo(f"📂 Split: {info.get('split', 'Unknown')}")
    typer.echo(f"🎯 Target Only: {info.get('train_on_target_only', 'Unknown')}")
    
    if info.get('chat_template'):
        typer.echo(f"💬 Chat Template: {info['chat_template']}")

@app.command("debug")
def debug_dataset(
    dataset: str = typer.Option(..., "--dataset", "-d", help="📁 Dataset directory to debug"),
    samples: int = typer.Option(5, "--samples", "-n", help="🔢 Number of samples to analyze"),
):
    """
    🔍 Debug and analyze your training dataset
    
    This command helps you understand how your data will be processed during training.
    """
    
    try:
        from datasets import load_from_disk
        from transformers import AutoTokenizer
        
        typer.echo(f"🔍 Analyzing Dataset: {dataset}")
        
        # Validate dataset exists
        dataset_path = Path(dataset)
        if not dataset_path.exists():
            typer.echo(f"❌ Dataset not found: {dataset}", err=True)
            raise typer.Exit(1)
        
        # Load dataset configuration
        config_file = dataset_path / "dataset_config.json"
        dataset_config = {}
        if config_file.exists():
            with open(config_file) as f:
                dataset_config = json.load(f)
        
        typer.echo(f"📂 Loading dataset from {dataset_path}...")
        dataset_ds = load_from_disk(str(dataset_path))
        
        typer.echo(f"✅ Dataset loaded: {len(dataset_ds)} samples")
        
        # Show basic info
        typer.echo(f"📋 Features: {', '.join(dataset_ds.column_names)}")
        
        # Show a few samples
        num_to_show = min(samples, len(dataset_ds))
        typer.echo(f"\n🔍 Showing {num_to_show} sample(s):")
        
        for i in range(num_to_show):
            sample = dataset_ds[i]
            typer.echo(f"\n📝 Sample {i+1}:")
            
            # Show key information
            if "input_ids" in sample:
                typer.echo(f"  📏 Length: {len(sample['input_ids'])} tokens")
            
            # Show text if available
            if "text" in sample:
                text = str(sample["text"])[:200] + ("..." if len(str(sample["text"])) > 200 else "")
                typer.echo(f"  📝 Text preview: {text}")
                
        typer.echo(f"\n🎯 Debug complete!")
        
    except ImportError as e:
        typer.echo(f"❌ Missing dependencies: {e}", err=True)
        typer.echo("💡 Please install required packages: datasets, transformers")
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"❌ Error during debug: {e}", err=True)
        raise typer.Exit(1)

@app.callback()
def root(version: bool = typer.Option(False, "--version")):
    if version:
        typer.echo("OpenSloth CLI v1 (refactored)")
        raise typer.Exit()

def main():
    app()

if __name__ == "__main__":
    main()
