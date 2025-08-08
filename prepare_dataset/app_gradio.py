import os
import sys
import json
import traceback
import subprocess
import signal
import html
import re
import datetime
from typing import Any

import gradio as gr
import commentjson

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


# Lazy import functions - import heavy modules only when needed
def _get_dataset_prep_config():
    from prepare_dataset.config_schema import DatasetPrepConfig
    return DatasetPrepConfig


# State for running/canceling background jobs
RUN_STATE = {
    'proc': None,
}

PRESETS_DIR = os.path.join('prepare_dataset', 'presets')
PRESETS_DATA_DIR = os.path.join(PRESETS_DIR, 'data')
PRESETS_TRAIN_DIR = os.path.join(PRESETS_DIR, 'train')

DEFAULT_DATA_PREP_JSON = """{
  // HuggingFace model identifier or local path - determines tokenizer and chat template
  "tok_name": "unsloth/Qwen2.5-0.5B-Instruct",
  
  // Template for formatting conversations (auto-detected from model, can override)
  "chat_template": "qwen-2.5",
  
  // HuggingFace dataset or local file path
  "dataset_name": "mlabonne/FineTome-100k",
  
  // Dataset split to use (train/validation/test)
  "split": "train",
  
  // Number of samples to process (-1 for all, use small number for testing)
  "num_samples": 1000,
  
  // Number of parallel workers for processing (adjust based on CPU cores)
  "num_proc": 4,
  
  // Only train on assistant responses (highly recommended for chat models)
  "train_on_target_only": true,
  
  // Text that starts user messages (auto-detected from chat template)
  "instruction_part": "<|im_start|>user\\n",
  
  // Text that starts assistant responses (auto-detected from chat template)  
  "response_part": "<|im_start|>assistant\\n",
  
  // Number of samples to preview in logs (0 = disabled, useful for debugging)
  "debug": 0,
  
  // Auto-generated based on model name and date - do not modify
  "output_dir": null,
  
  // Required for accessing gated models/datasets (leave null if not needed)
  "hf_token": null
}"""

DEFAULT_TRAINING_JSON = """{
  // OpenSloth Configuration
  "opensloth_config": {
    // Path to processed dataset (auto-filled from Data Preparation tab)
    "data_cache_path": "data/...",
    
    // List of GPU indices to use (e.g. [0] for single GPU, [0,1] for multi-GPU)
    "devices": [0],
    
    // Use sequence packing for faster training (recommended)
    "sequence_packing": true,

    // Model Configuration
    "fast_model_args": {
      // HuggingFace model identifier or local path
      "model_name": "unsloth/Qwen2.5-0.5B-Instruct",
      
      // Maximum input length (higher = more VRAM usage)
      "max_seq_length": 2048,
      
      // Use 4-bit quantization to reduce memory usage (recommended)
      "load_in_4bit": true,
      "load_in_8bit": false,
      
      // Train all parameters vs LoRA (full_finetuning requires much more VRAM)
      "full_finetuning": false
    },

    // LoRA Configuration (ignored if full_finetuning is true)
    "lora_args": {
      // LoRA rank (higher = more parameters, more VRAM)
      "r": 8,
      "lora_alpha": 16,
      "lora_dropout": 0.0,
      
      // Which layers to apply LoRA to
      "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
      ],
      
      // Use Rank-stabilized LoRA (experimental)
      "use_rslora": false
    }
  },

  // HuggingFace Trainer Arguments
  "training_args": {
    // Where to save the trained model
    "output_dir": "outputs/exps/test_run",
    
    // Batch size per GPU (reduce if you get OOM errors)
    "per_device_train_batch_size": 1,
    
    // Effective batch size = batch_size * num_gpus * accumulation_steps
    "gradient_accumulation_steps": 4,
    
    // Model learning rate (typically 1e-4 to 5e-4 for LoRA)
    "learning_rate": 0.0001,
    
    // Log training progress every N steps
    "logging_steps": 1,
    
    // For quick tests, set max_steps. For full training, use num_train_epochs
    "max_steps": 50,
    
    // Number of complete passes through the dataset  
    "num_train_epochs": 1,
    
    // Learning rate schedule (linear is most common)
    "lr_scheduler_type": "linear",
    
    // Number of warmup steps at the beginning
    "warmup_steps": 10,
    
    // Maximum number of checkpoints to keep (saves disk space)
    "save_total_limit": 1,
    
    // L2 regularization parameter
    "weight_decay": 0.01,
    
    // Optimizer (adamw_8bit is memory efficient)
    "optim": "adamw_8bit",
    
    // Random seed for reproducible results
    "seed": 3407,
    
    // Logging backend: 'none', 'tensorboard', or 'wandb'
    "report_to": "none"
  }
}"""


def _list_cached_datasets(base_dir: str = 'data') -> list[str]:
    out = []
    if not os.path.isdir(base_dir):
        return out
    for root, _, files in os.walk(base_dir):
        if 'dataset_info.json' in files or any(
            f.endswith('.arrow') for f in files
        ):
            out.append(root)
    out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return out


def _safe_kill_running():
    proc = RUN_STATE.get('proc')
    if proc and proc.poll() is None:
        try:
            proc.send_signal(signal.SIGINT)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    RUN_STATE['proc'] = None


def _ensure_preset_dirs():
    os.makedirs(PRESETS_DATA_DIR, exist_ok=True)
    os.makedirs(PRESETS_TRAIN_DIR, exist_ok=True)


def _list_json_files(dir_path: str) -> list[str]:
    if not os.path.isdir(dir_path):
        return []
    names = [f for f in os.listdir(dir_path) if f.endswith('.json')]
    names.sort()
    return names


def _save_json(path: str, data: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _generate_dataset_name(model_name: str, dataset_name: str, num_samples: int) -> str:
    """Auto-generate dataset directory name based on model and config"""
    today = datetime.datetime.now().strftime("%m%d")
    
    # Extract model family from model name
    model_family = "unknown"
    if "qwen" in model_name.lower():
        model_family = "qwen"
    elif "gemma" in model_name.lower():
        model_family = "gemma" 
    elif "llama" in model_name.lower():
        model_family = "llama"
    elif "mistral" in model_name.lower():
        model_family = "mistral"
    
    # Extract dataset name (remove path/organization)
    dataset_short = dataset_name.split('/')[-1].replace('-', '_').lower()
    
    # Format: model-dataset-samples-mmdd
    name = f"{model_family}_{dataset_short}_n{num_samples}_{today}"
    return name


def dict_to_json(data: dict[str, Any]) -> str:
    return commentjson.dumps(
        data, indent=2, ensure_ascii=False
    )


def json_file_to_json_str(path: str) -> str:
    data = _load_json(path)
    return dict_to_json(data)


def _get_config_choices():
    presets = []
    for filename in _list_json_files(PRESETS_DATA_DIR):
        try:
            preset_data = _load_json(os.path.join(PRESETS_DATA_DIR, filename))
            description = preset_data.get('description', '')
            name = filename.replace('.json', '').replace('_', ' ').title()
            presets.append(f'{name}: {description}' if description else name)
        except Exception:
            presets.append(filename.replace('.json', ''))
    return presets


def _extract_preset_filename(choice_with_desc: str) -> str | None:
    if not choice_with_desc:
        return None
    name_part = choice_with_desc.split(':')[0].strip()
    filename = name_part.lower().replace(' ', '_') + '.json'
    return (
        filename
        if filename in _list_json_files(PRESETS_DATA_DIR)
        else None
    )


def _ansi_to_text(s: str) -> str:
    return re.sub(r'\x1b\[[0-9;]*m', '', s)


def _to_log_html(text: str, elem_id: str) -> str:
    escaped = html.escape(_ansi_to_text(text))
    toolbar = (
        '<div class="small" style="display:flex;gap:12px;align-items:center;'
        'margin:6px 0;">'
        '<strong>Live logs</strong>'
        f'<button onclick="navigator.clipboard.writeText('
        f'document.getElementById(\'{elem_id}\').innerText)" '
        'style="margin-left:auto">Copy</button>'
        '</div>'
    )
    return (
        f'{toolbar}<div id="{elem_id}" class="logbox">{escaped}</div>'
        f'<script>var el=document.getElementById(\'{elem_id}\'); '
        'if(el) el.scrollTop=el.scrollHeight;</script>'
    )


def run_dataset_prep_gui(config_json: str):
    try:
        cfg = commentjson.loads(config_json)
    except commentjson.JSONLibraryException as e:
        error_msg = f'Invalid JSON format: {e}'
        return gr.update(), gr.update(value=error_msg, visible=True), gr.update()

    # Auto-generate dataset name and set output directory
    if not cfg.get('output_dir'):
        model_name = cfg.get('tok_name', 'unknown')
        dataset_name = cfg.get('dataset_name', 'unknown')
        num_samples = cfg.get('num_samples', -1)
        generated_name = _generate_dataset_name(model_name, dataset_name, num_samples)
        cfg['output_dir'] = f'data/{generated_name}'

    if cfg.get('train_on_target_only') and (
        not cfg.get('instruction_part') or not cfg.get('response_part')
    ):
        error_msg = (
            'instruction_part and response_part are required when '
            'train_on_target_only is enabled.'
        )
        return (
            gr.update(),
            gr.update(value=error_msg, visible=True),
            gr.update(value=''),
        )

    try:
        _safe_kill_running()
        proc = subprocess.Popen(
            ['python', 'prepare_dataset/run_prep_job.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        RUN_STATE['proc'] = proc
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(cfg))
        proc.stdin.close()
        # Stream logs to UI
        out_lines: list[str] = []
        out_dir = ''
        for line in proc.stdout:
            out_lines.append(line.rstrip('\n'))
            if line.strip().startswith('[JOB] Completed. Saved at:'):
                out_dir = line.split(':', 1)[1].strip()
            log_html = _to_log_html('\n'.join(out_lines), elem_id='prep-log')
            yield (
                gr.update(value=out_dir),
                gr.update(value=log_html, visible=True),
                gr.update(value=''),
            )
        code = proc.wait()
        RUN_STATE['proc'] = None

        debug_html_content = ''
        if cfg.get('debug') and os.path.exists('.log/dataloader_examples.html'):
            try:
                with open(
                    '.log/dataloader_examples.html', 'r', encoding='utf-8'
                ) as hf:
                    debug_html_content = hf.read()
            except Exception:
                pass
        full_log_html = _to_log_html('\n'.join(out_lines), elem_id='prep-log')
        if code != 0:
            return (
                gr.update(),
                gr.update(value=full_log_html, visible=True),
                gr.update(value=debug_html_content),
            )
        
        # Save configuration to the dataset folder for reference
        if out_dir and os.path.exists(out_dir):
            try:
                config_path = os.path.join(out_dir, 'preparation_config.json')
                with open(config_path, 'w', encoding='utf-8') as f:
                    commentjson.dump(cfg, f, indent=2, ensure_ascii=False)
            except Exception:
                pass  # Don't fail if we can't save config
            
        return (
            out_dir,
            gr.update(value=full_log_html, visible=True),
            gr.update(value=debug_html_content),
        )
    except Exception:
        _safe_kill_running()
        return (
            gr.update(),
            gr.update(value=traceback.format_exc(), visible=True),
            gr.update(value=''),
        )


def launch_training_gui(config_json: str):
    try:
        cfg = commentjson.loads(config_json)
        data_cache_path = cfg.get('opensloth_config', {}).get('data_cache_path')
        model_name = (
            cfg.get('opensloth_config', {})
            .get('fast_model_args', {})
            .get('model_name')
        )
    except commentjson.JSONLibraryException as e:
        return gr.update(value=f'❌ Invalid JSON format: {e}', visible=True)

    if not data_cache_path or not os.path.exists(data_cache_path):
        msg = '❌ Error: Dataset path is required and must exist in the YAML.'
        return gr.update(value=msg, visible=True)

    if not model_name:
        msg = '❌ Error: Model name is required in the YAML config.'
        return gr.update(value=msg, visible=True)

    # Check for tokenizer compatibility warning
    try:
        dataset_config_path = os.path.join(data_cache_path, 'dataset_config.json')
        if os.path.exists(dataset_config_path):
            with open(dataset_config_path, 'r') as f:
                dataset_config = json.load(f)
            dataset_tokenizer = dataset_config.get('tok_name', '')
            if dataset_tokenizer and dataset_tokenizer != model_name:
                warning_msg = (
                    f"⚠️ Warning: Dataset was prepared with tokenizer "
                    f"'{dataset_tokenizer}' but training with '{model_name}'. "
                    f"This may cause issues."
                )
                yield gr.update(value=warning_msg, visible=True)
    except Exception:
        pass  # Continue if we can't check

    try:
        _safe_kill_running()
        proc = subprocess.Popen(
            ['python', 'prepare_dataset/run_train_job.py'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        RUN_STATE['proc'] = proc
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(cfg))
        proc.stdin.close()

        out_lines: list[str] = []
        yield gr.update(value='🚀 Starting training process...', visible=True)

        for line in proc.stdout:
            out_lines.append(line.rstrip('\n'))
            log_html = _to_log_html('\n'.join(out_lines), elem_id='train-log')
            yield gr.update(value=log_html, visible=True)

        code = proc.wait()
        RUN_STATE['proc'] = None
        final_html = _to_log_html(
            '\n'.join(out_lines) or ('Exit code: ' + str(code)),
            elem_id='train-log',
        )
        output_dir = cfg.get('training_args', {}).get('output_dir')

        if code == 0:
            success_msg = (
                f'✅ Training completed successfully! '
                f'Model saved to: {output_dir}'
            )
            final_html = success_msg + '\n\n' + final_html
        else:
            error_msg = f'❌ Training failed with exit code: {code}'
            final_html = error_msg + '\n\n' + final_html

        return gr.update(value=final_html, visible=True)

    except Exception as e:
        _safe_kill_running()
        error_msg = f'❌ Training error: {str(e)}\n\n{traceback.format_exc()}'
        return gr.update(value=error_msg, visible=True)


def build_ui() -> gr.Blocks:
    _ensure_preset_dirs()

    with gr.Blocks(title='OpenSloth Studio') as demo:
        gr.Markdown(
            """
		# 🦥 OpenSloth Studio
		**Simple SFT Workflow:** Choose preset → Edit config → Process data → Train model

		*Optimized for Supervised Fine-Tuning (SFT) with small, efficient models.*
		"""
        )

        gr.HTML(
            """
		<style>
		:root { --log-bg: #0b0e14; --log-fg: #f0f3f6; --log-border: #1f2937; }
		@media (prefers-color-scheme: light) {
            :root {
                --log-bg: #f8fafc;
                --log-fg: #111827;
                --log-border: #e5e7eb;
            }
        }
		.logbox { height: 340px; overflow-y: auto; background: var(--log-bg);
        color: var(--log-fg); padding: 10px; border-radius: 8px;
        border: 1px solid var(--log-border); white-space: pre-wrap;
        word-break: break-word; font: 12px/1.5 ui-monospace, SFMono-Regular,
        Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05) inset; }
		.small { font-size: 12px; color: #6b7280; }
		.step-header { font-size: 18px; font-weight: 600; color: #1f2937;
        margin-bottom: 12px; }
		.step-description { font-size: 14px; color: #6b7280;
        margin-bottom: 16px; }
		</style>
		"""
        )

        with gr.Tab('📊 Data Preparation'):
            with gr.Group():
                gr.HTML('<div class="step-header">Step 1: Choose a Config</div>')
                with gr.Row():
                    ds_preset = gr.Dropdown(
                        label='Config Templates',
                        choices=_get_config_choices(),
                        value=None,
                        info='Pre-configured settings for common use cases',
                    )
                    load_config_ds = gr.Dropdown(
                        label='Load Custom Config',
                        choices=_list_json_files(PRESETS_DATA_DIR),
                        info='Load a previously saved config',
                    )
                    refresh_presets_btn = gr.Button('🔄 Refresh', size='sm')

            with gr.Group():
                gr.HTML('<div class="step-header">Step 2: Edit Config</div>')
                data_prep_json = gr.Code(
                    label='Configuration (JSON with Comments)',
                    value=DEFAULT_DATA_PREP_JSON.strip(),
                    language='javascript',  # JavaScript highlighting works well for JSON
                    
                    # lines=15,
                    # max_lines=30,
                )

            with gr.Group():
                gr.HTML('<div class="step-header">Step 3: Process Dataset</div>')
                with gr.Row():
                    run_prep = gr.Button('🚀 Process', variant='primary', size='lg')
                    cancel_prep = gr.Button('❌ Cancel', size='sm')
                    clear_prep = gr.Button('🧹 Clear Logs', size='sm')

            out_dir_box = gr.Textbox(
                label='✅ Processed Dataset Path', interactive=False
            )
            prep_status = gr.HTML(visible=False)
            debug_html = gr.HTML(label='📋 Debug Preview', visible=False)

            with gr.Accordion('💾 Save Custom Config', open=False):
                with gr.Row():
                    config_name_ds = gr.Textbox(
                        label='Config Name',
                        placeholder='my_qwen_config',
                        info='Save current YAML settings as a reusable config',
                    )
                    save_config_ds = gr.Button('💾 Save')

            def apply_starter_config(preset_choice: str) -> str:
                filename = _extract_preset_filename(preset_choice)
                if not filename:
                    return DEFAULT_DATA_PREP_JSON.strip()
                try:
                    path = os.path.join(PRESETS_DATA_DIR, filename)
                    DatasetPrepConfig = _get_dataset_prep_config()
                    cfg = DatasetPrepConfig.model_validate(_load_json(path))
                    return dict_to_json(cfg.model_dump(exclude_none=True))
                except Exception:
                    return DEFAULT_DATA_PREP_JSON.strip()

            def save_ds_config(name: str, json_str: str):
                if not name:
                    return gr.update(), gr.update()
                try:
                    data = commentjson.loads(json_str)
                    filename = f'{name.strip().replace(" ", "_")}.json'
                    _save_json(os.path.join(PRESETS_DATA_DIR, filename), data)
                    config_choices = _list_json_files(PRESETS_DATA_DIR)
                    starter_choices = _get_config_choices()
                    return (
                        gr.update(value='✅ Saved!'),
                        gr.update(choices=config_choices, value=filename),
                        gr.update(choices=starter_choices),
                    )
                except Exception as e:
                    return gr.update(value=f'❌ Error: {e}'), gr.update(), gr.update()

            def refresh_presets():
                return gr.update(
                    choices=_get_config_choices()
                ), gr.update(choices=_list_json_files(PRESETS_DATA_DIR))

            ds_preset.change(
                apply_starter_config,
                inputs=[ds_preset],
                outputs=[data_prep_json],
            )
            load_config_ds.change(
                lambda fname: (
                    json_file_to_json_str(os.path.join(PRESETS_DATA_DIR, fname))
                    if fname
                    else gr.update()
                ),
                inputs=[load_config_ds],
                outputs=[data_prep_json],
            )
            run_prep.click(
                run_dataset_prep_gui,
                inputs=[data_prep_json],
                outputs=[out_dir_box, prep_status, debug_html],
            )
            save_config_ds.click(
                save_ds_config,
                inputs=[config_name_ds, data_prep_json],
                outputs=[prep_status, load_config_ds, ds_preset],
            )
            refresh_presets_btn.click(
                refresh_presets, outputs=[ds_preset, load_config_ds]
            )
            cancel_prep.click(
                lambda: _safe_kill_running() or gr.update(value='Cancelled.'),
                outputs=[prep_status],
            )
            clear_prep.click(
                lambda: (gr.update(value='', visible=False), gr.update(value='')),
                outputs=[prep_status, debug_html],
            )

        with gr.Tab('🚀 Training'):
            with gr.Group():
                gr.HTML('<div class="step-header">Step 1: Select Resources</div>')
                with gr.Row():
                    ds_pick = gr.Dropdown(
                        label='Select Processed Dataset',
                        choices=_list_cached_datasets(),
                        info='Choose a dataset to automatically fill the path',
                    )
                    load_preset_tr = gr.Dropdown(
                        label='Load Training Config',
                        choices=_list_json_files(PRESETS_TRAIN_DIR),
                        info='Load a saved training configuration',
                    )
                    refresh_train_btn = gr.Button('🔄 Refresh', size='sm')

            with gr.Group():
                gr.HTML('<div class="step-header">Step 2: Edit Config</div>')
                training_json = gr.Code(
                    label='Configuration (JSON with Comments)',
                    value=DEFAULT_TRAINING_JSON.strip(),
                    language='javascript',  # JavaScript highlighting works well for JSON
                    # lines=20,
                    # max_lines=40,
                )

            with gr.Group():
                gr.HTML('<div class="step-header">Step 3: Start Training</div>')
                with gr.Row():
                    train_btn = gr.Button(
                        '🚀 Start Training', variant='primary', size='lg'
                    )
                    cancel_train = gr.Button('❌ Cancel', size='sm')
                    clear_train = gr.Button('🧹 Clear Logs', size='sm')

            train_status = gr.HTML(visible=False)

            with gr.Accordion('💾 Save Custom Config', open=False):
                with gr.Row():
                    preset_name_tr = gr.Textbox(
                        label='Config Name', placeholder='my_lora_config'
                    )
                    save_preset_tr = gr.Button('💾 Save')

            def update_json_with_path(json_str: str, path: str | None) -> str:
                if not path:
                    return gr.update()
                try:
                    data = commentjson.loads(json_str)
                    data['opensloth_config']['data_cache_path'] = path
                    return dict_to_json(data)
                except (commentjson.JSONLibraryException, KeyError):
                    return gr.update()

            def save_train_config(name: str, json_str: str):
                if not name:
                    return gr.update(), gr.update()
                try:
                    data = commentjson.loads(json_str)
                    filename = f'{name.strip().replace(" ", "_")}.json'
                    _save_json(os.path.join(PRESETS_TRAIN_DIR, filename), data)
                    choices = _list_json_files(PRESETS_TRAIN_DIR)
                    return (
                        gr.update(value='✅ Saved!'),
                        gr.update(choices=choices, value=filename),
                    )
                except Exception as e:
                    return gr.update(value=f'❌ Error: {e}'), gr.update()

            def refresh_training_resources():
                return gr.update(
                    choices=_list_cached_datasets()
                ), gr.update(choices=_list_json_files(PRESETS_TRAIN_DIR))

            train_btn.click(
                launch_training_gui, inputs=[training_json], outputs=[train_status]
            )
            ds_pick.change(
                update_json_with_path,
                inputs=[training_json, ds_pick],
                outputs=[training_json],
            )
            load_preset_tr.change(
                lambda fname: (
                    json_file_to_json_str(os.path.join(PRESETS_TRAIN_DIR, fname))
                    if fname
                    else gr.update()
                ),
                inputs=[load_preset_tr],
                outputs=[training_json],
            )
            save_preset_tr.click(
                save_train_config,
                inputs=[preset_name_tr, training_json],
                outputs=[train_status, load_preset_tr],
            )
            refresh_train_btn.click(
                refresh_training_resources, outputs=[ds_pick, load_preset_tr]
            )
            cancel_train.click(
                lambda: _safe_kill_running()
                or gr.update(value='Cancelled.', visible=True),
                outputs=[train_status],
            )
            clear_train.click(
                lambda: gr.update(value='', visible=False), outputs=[train_status]
            )

    return demo


# Build the demo at module level so Gradio can find it
demo = build_ui()

if __name__ == '__main__':
    # Set no_proxy for local development if needed
    os.environ['no_proxy'] = 'localhost,127.0.0.1'
    demo.launch()
