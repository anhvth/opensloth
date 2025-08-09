@app.command("debug")
def debug_dataset(
    dataset: Annotated[str, typer.Argument(help="📊 Path to processed dataset directory")],
    samples_per_page: Annotated[int, typer.Option("--samples", "-n", help="🔢 Number of samples per page")] = 3,
    show_tokens: Annotated[bool, typer.Option("--show-tokens", help="🔍 Show individual tokens with colors")] = True,
    batch_size: Annotated[int, typer.Option("--batch-size", help="📦 Batch size for debugging")] = 1,
):
    """
    🐛 Debug dataset with color-coded token visualization
    
    Shows how the dataset looks after tokenization with color coding:
    - Green: Context/instruction tokens (not trained on)
    - Yellow: Target/response tokens (trained on)
    - Red: Padding tokens
    
    **Interactive Controls:**
    - Press ENTER: Show next random samples
    - Press ESC or Ctrl+C: Exit
    
    **Examples:**
    
    • **Basic debugging:**
      [cyan]os debug data/my_dataset[/cyan]
    
    • **More samples per page:**
      [cyan]os debug data/my_dataset --samples 5[/cyan]
    
    • **Token-level inspection:**
      [cyan]os debug data/my_dataset --show-tokens --samples 2[/cyan]
    """
    
    try:
        if not os.path.exists(dataset):
            _fail(f"Dataset not found: {dataset}")
        
        # Load dataset config to get model info
        dataset_config = _load_dataset_config(dataset)
        if not dataset_config:
            console.print("⚠️  No dataset config found, using basic debugging", style="yellow")
        
        _print_header(f"🐛 [bold]Debug Dataset: {dataset}[/bold]")
        
        # Import required modules
        try:
            from transformers import AutoTokenizer
            from datasets import load_from_disk
            import random
            import torch
        except ImportError as e:
            _fail(f"Required modules not available: {e}")
        
        # Load the tokenizer based on dataset config
        model_name = dataset_config.get("tok_name", "unsloth/Qwen2.5-0.5B-Instruct")
        console.print(f"🤖 Loading tokenizer: [cyan]{model_name}[/cyan]")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            console.print(f"⚠️  Failed to load tokenizer: {e}", style="yellow")
            console.print("🔄 Falling back to default tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("unsloth/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
        
        # Add padding token if missing
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load dataset
        console.print(f"📊 Loading dataset from: [cyan]{dataset}[/cyan]")
        try:
            dataset_obj = load_from_disk(dataset)
            if hasattr(dataset_obj, 'train'):
                dataset_obj = dataset_obj['train']
        except Exception as e:
            _fail(f"Failed to load dataset: {e}")
        
        total_samples = len(dataset_obj)
        console.print(f"📈 Total samples in dataset: [cyan]{total_samples}[/cyan]")
        
        # Show dataset info
        if dataset_config:
            console.print(f"\n📋 [bold]Dataset Info:[/bold]")
            info_items = [
                ("🤖 Model:", dataset_config.get("tok_name", "Unknown")),
                ("🔢 Total Samples:", str(total_samples)),
                ("📏 Max Length:", str(dataset_config.get("max_seq_length", "Unknown"))),
                ("💬 Chat Template:", dataset_config.get("chat_template", "None")),
                ("🎯 Target Only:", "✅" if dataset_config.get("train_on_target_only") else "❌"),
            ]
            _print_kv(info_items)
        
        # Interactive debugging loop
        console.print("\n🎨 [bold]Color-coded Token Visualization:[/bold]")
        console.print("=" * 70)
        console.print("🟢 [green]Context (not trained)[/green] | 🟡 [yellow]Target (trained)[/yellow] | 🔴 [red]Padding[/red]")
        console.print("=" * 70)
        console.print("💡 [dim]Press ENTER for next samples, ESC or Ctrl+C to exit[/dim]\n")
        
        def _display_samples():
            """Display a batch of random samples with color coding."""
            # Get random sample indices
            sample_indices = random.sample(range(total_samples), min(samples_per_page, total_samples))
            
            for i, idx in enumerate(sample_indices, 1):
                sample = dataset_obj[idx]
                console.print(f"\n📝 [bold cyan]Sample {i}/{samples_per_page} (index: {idx})[/bold cyan]")
                console.print("-" * 50)
                
                # Get the tokenized data
                input_ids = sample.get('input_ids', [])
                labels = sample.get('labels', [])
                attention_mask = sample.get('attention_mask', [])
                
                if not input_ids:
                    console.print("⚠️  No input_ids found in sample", style="yellow")
                    continue
                
                # Convert to lists for safe processing
                try:
                    input_ids_list = input_ids if isinstance(input_ids, list) else input_ids.tolist()
                    labels_list = labels if isinstance(labels, list) else labels.tolist()
                    attention_mask_list = attention_mask if isinstance(attention_mask, list) else attention_mask.tolist()
                except Exception as e:
                    console.print(f"⚠️  Error processing tensors: {e}", style="yellow")
                    continue
                
                # Handle empty or mismatched arrays
                if not input_ids_list:
                    console.print("⚠️  Empty input_ids", style="yellow")
                    continue
                
                max_len = len(input_ids_list)
                
                # Pad labels and attention_mask if needed
                while len(labels_list) < max_len:
                    labels_list.append(-100)
                while len(attention_mask_list) < max_len:
                    attention_mask_list.append(1)
                
                # Decode and display tokens with color coding
                if show_tokens:
                    console.print("🔍 [bold]Token-level view:[/bold]")
                    
                    for j, (token_id, label_id, attention) in enumerate(zip(input_ids_list, labels_list, attention_mask_list)):
                        # Skip padding tokens for cleaner display
                        if attention == 0:
                            continue
                            
                        try:
                            token = tokenizer.decode([token_id], skip_special_tokens=False)
                        except Exception:
                            token = f"<UNK:{token_id}>"
                        
                        # Color coding based on labels
                        if label_id == -100:
                            # Context token (not trained on)
                            color = "green"
                            marker = "C"
                        elif token_id == tokenizer.pad_token_id:
                            # Padding token
                            color = "red"
                            marker = "P"
                        else:
                            # Target token (trained on)
                            color = "yellow"
                            marker = "T"
                        
                        # Clean up token for display
                        token_display = repr(token)[1:-1]  # Remove outer quotes
                        console.print(f"[{color}]{marker}:{token_display}[/{color}]", end=" ")
                    
                    console.print("\n")
                
                # Show full text reconstruction
                console.print("📖 [bold]Full text:[/bold]")
                try:
                    full_text = tokenizer.decode(input_ids_list, skip_special_tokens=True)
                    console.print(full_text)
                except Exception as e:
                    console.print(f"<Could not decode: {e}>")
                
                # Split into context and target parts based on labels
                context_tokens = []
                target_tokens = []
                
                for token_id, label_id in zip(input_ids_list, labels_list):
                    if label_id == -100:
                        context_tokens.append(token_id)
                    else:
                        target_tokens.append(token_id)
                
                if context_tokens:
                    try:
                        context_text = tokenizer.decode(context_tokens, skip_special_tokens=True)
                        console.print(f"[green]Context: {context_text}[/green]")
                    except Exception:
                        console.print("[green]Context: <decode error>[/green]")
                
                if target_tokens:
                    try:
                        target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
                        console.print(f"[yellow]Target: {target_text}[/yellow]")
                    except Exception:
                        console.print("[yellow]Target: <decode error>[/yellow]")
                
                # Show statistics
                total_tokens = len(input_ids_list)
                target_count = sum(1 for l in labels_list if l != -100)
                console.print(f"\n📊 [dim]Stats: {total_tokens} tokens, {target_count} target tokens[/dim]")
        
        # Interactive loop
        try:
            while True:
                _display_samples()
                
                console.print(f"\n{'='*70}")
                console.print("💡 Press [bold green]ENTER[/bold green] for more samples, [bold red]ESC/Ctrl+C[/bold red] to exit")
                
                # Wait for user input
                try:
                    user_input = input().strip()
                    if user_input.lower() in ['q', 'quit', 'exit']:
                        break
                    # Continue with ENTER or any other input
                except (KeyboardInterrupt, EOFError):
                    break
                
                console.print()  # Add some spacing
                
        except KeyboardInterrupt:
            pass
        
        console.print(f"\n✅ [bold green]Debug session completed![/bold green]")
        
    except typer.Exit:
        raise
    except KeyboardInterrupt:
        console.print(f"\n⏹️  Debug interrupted by user")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n❌ [bold red]Debug error: {e}[/bold red]")
        raise typer.Exit(1)
