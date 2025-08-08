import os
import sys
import json
import traceback
import subprocess
import signal

import gradio as gr
import re
import html

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Lazy import functions - import heavy modules only when needed
def _get_dataset_prep_config():
    try:
        from prepare_dataset.config_schema import DatasetPrepConfig
    except ImportError:
        # Try absolute import from project root
        import sys
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from prepare_dataset.config_schema import DatasetPrepConfig
    return DatasetPrepConfig

def _get_dataset_preparers():
    # Return a simple dict without actually importing the heavy modules
    # We don't need the actual classes since we use subprocess for dataset prep
    return {"Qwen": "QwenDatasetPreparer", "Gemma": "GemmaDatasetPreparer"}


# State for running/canceling background jobs
RUN_STATE = {
	"proc": None,
}


def _list_cached_datasets(base_dir: str = "data") -> list[str]:
	out = []
	if not os.path.isdir(base_dir):
		return out
	# collect folders that look like saved datasets (arrow shards or dataset_info.json)
	for root, dirs, files in os.walk(base_dir):
		if "dataset_info.json" in files or any(f.endswith(".arrow") for f in files):
			out.append(root)
	# latest first by mtime
	out.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return out


def _safe_kill_running():
	proc = RUN_STATE.get("proc")
	if proc and proc.poll() is None:
		try:
			proc.send_signal(signal.SIGINT)
		except Exception:
			try:
				proc.terminate()
			except Exception:
				pass
	RUN_STATE["proc"] = None


PRESETS_DIR = os.path.join("prepare_dataset", "presets")
PRESETS_DATA_DIR = os.path.join(PRESETS_DIR, "data")
PRESETS_TRAIN_DIR = os.path.join(PRESETS_DIR, "train")


def _ensure_preset_dirs():
	os.makedirs(PRESETS_DATA_DIR, exist_ok=True)
	os.makedirs(PRESETS_TRAIN_DIR, exist_ok=True)


def _list_json_files(dir_path: str) -> list[str]:
	if not os.path.isdir(dir_path):
		return []
	names = [f for f in os.listdir(dir_path) if f.endswith(".json")]
	names.sort()
	return names


def _save_json(path: str, data: dict) -> None:
	with open(path, "w", encoding="utf-8") as f:
		json.dump(data, f, ensure_ascii=False, indent=2)


def _load_json(path: str) -> dict:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


MODEL_FAMILIES = None  # Will be loaded lazily

def _get_model_families():
    # Return the model families without loading heavy imports
    return {"Qwen": "QwenDatasetPreparer", "Gemma": "GemmaDatasetPreparer"}


def _default_dataset_prep(model_family: str):
    DatasetPrepConfig = _get_dataset_prep_config()
    if model_family == "Qwen":
        return DatasetPrepConfig(
            tok_name="unsloth/Qwen3-0.6B-Instruct",
            chat_template="qwen-3",
            dataset_name="mlabonne/FineTome-100k",
            split="train",
            train_on_target_only=True,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>assistant\n",
        )
    else:  # Gemma
        return DatasetPrepConfig(
            tok_name="unsloth/gemma-3-1b-it",
            chat_template="gemma-3",
            dataset_name="mlabonne/FineTome-100k",
            split="train",
            train_on_target_only=True,
            instruction_part="<start_of_turn>user\n",
            response_part="<start_of_turn>model\n",
        )


def _ansi_to_text(s: str) -> str:
	"""Remove ANSI color codes for clean HTML rendering."""
	return re.sub(r"\x1b\[[0-9;]*m", "", s)


def _to_log_html(text: str, elem_id: str) -> str:
	escaped = html.escape(_ansi_to_text(text))
	toolbar = (
		f"<div class='small' style='display:flex;gap:12px;align-items:center;margin:6px 0;'>"
		f"<strong>Live logs</strong>"
		f"<button onclick=\"navigator.clipboard.writeText(document.getElementById('{elem_id}').innerText)\" style='margin-left:auto'>Copy</button>"
		f"</div>"
	)
	return (
		toolbar
		+ f"<div id='{elem_id}' class='logbox'>{escaped}</div>"
		+ f"<script>var el=document.getElementById('{elem_id}'); if(el) el.scrollTop=el.scrollHeight;</script>"
	)


def run_dataset_prep_gui(
	model_family: str,
	tok_name: str,
	chat_template: str,
	dataset_name: str,
	split: str,
	num_samples: int,
	num_proc: int,
	train_on_target_only: bool,
	instruction_part: str,
	response_part: str,
	debug: int,
	output_dir: str,
	overrides_json: str,
):
	# run prep as a subprocess to stream logs & allow cancel
	if train_on_target_only and (not instruction_part or not response_part):
		return (
			gr.update(),
			gr.update(
				value="instruction_part and response_part are required when train_on_target_only is enabled.",
				visible=True,
			),
			gr.update(value=""),
		)

	cfg: dict = {
		"model_family": model_family,
		"tok_name": tok_name,
		"chat_template": chat_template,
		"dataset_name": dataset_name,
		"split": split,
		"num_samples": num_samples,
		"num_proc": num_proc,
		"train_on_target_only": train_on_target_only,
		"instruction_part": instruction_part,
		"response_part": response_part,
		"debug": debug,
		"output_dir": output_dir or None,
	}

	# Optional JSON overrides
	if overrides_json and overrides_json.strip():
		try:
			overrides = json.loads(overrides_json)
			if isinstance(overrides, dict):
				cfg.update(overrides)
		except Exception:
			pass

	try:
		_safe_kill_running()
		RUN_STATE["proc"] = subprocess.Popen(
			["python", "prepare_dataset/run_prep_job.py"],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)
		assert RUN_STATE["proc"].stdin is not None
		RUN_STATE["proc"].stdin.write(json.dumps(cfg))
		RUN_STATE["proc"].stdin.close()
		# stream logs to UI
		out_lines: list[str] = []
		out_dir = ""
		for line in RUN_STATE["proc"].stdout:  # type: ignore[arg-type]
			out_lines.append(line.rstrip("\n"))
			if line.strip().startswith("[JOB] Completed. Saved at:"):
				out_dir = line.split(":", 1)[1].strip()
			# stream with ANSI cleaned + autoscroll
			log_html = _to_log_html("\n".join(out_lines), elem_id="prep-log")
			yield (
				gr.update(value=out_dir),
				gr.update(value=log_html, visible=True),
				gr.update(value=""),
			)
		code = RUN_STATE["proc"].wait()
		RUN_STATE["proc"] = None
		# try load debug HTML if any
		debug_html_content = ""
		if debug and os.path.exists(".log/dataloader_examples.html"):
			try:
				with open(".log/dataloader_examples.html", "r", encoding="utf-8") as hf:
					debug_html_content = hf.read()
			except Exception:
				pass
		full_log_html = _to_log_html("\n".join(out_lines), elem_id="prep-log")
		if code != 0:
			return gr.update(), gr.update(value=full_log_html, visible=True), gr.update(value=debug_html_content)
		return out_dir, gr.update(value=full_log_html, visible=True), gr.update(value=debug_html_content)
	except Exception:
		_safe_kill_running()
		return gr.update(), gr.update(value=traceback.format_exc(), visible=True), gr.update(value="")


def launch_training_gui(
	data_cache_path: str,
	devices_csv: str,
	model_name: str,
	max_seq_length: int,
	load_in_4bit: bool,
	load_in_8bit: bool,
	full_finetuning: bool,
	r: int,
	lora_alpha: int,
	lora_dropout: float,
	target_modules_csv: str,
	use_rslora: bool,
	output_dir: str,
	per_device_train_batch_size: int,
	gradient_accumulation_steps: int,
	learning_rate: float,
	logging_steps: int,
	num_train_epochs: int,
	lr_scheduler_type: str,
	warmup_steps: int,
	save_total_limit: int,
	weight_decay: float,
	optim: str,
	seed: int,
	report_to: str,
	sequence_packing: bool,
):
	# run training via subprocess to stream logs & allow cancel
	try:
		devices = [int(x.strip()) for x in devices_csv.split(",") if x.strip()]
		target_modules = [x.strip() for x in target_modules_csv.split(",") if x.strip()] or [
			"q_proj",
			"k_proj",
			"v_proj",
			"o_proj",
			"gate_proj",
			"up_proj",
			"down_proj",
		]

		cfg = {
			"opensloth_config": {
				"data_cache_path": data_cache_path,
				"devices": devices,
				"fast_model_args": {
					"model_name": model_name,
					"max_seq_length": int(max_seq_length),
					"load_in_4bit": bool(load_in_4bit),
					"load_in_8bit": bool(load_in_8bit),
					"full_finetuning": bool(full_finetuning),
				},
				"lora_args": {
					"r": int(r),
					"lora_alpha": int(lora_alpha),
					"lora_dropout": float(lora_dropout),
					"target_modules": target_modules,
					"use_rslora": bool(use_rslora),
				},
				"sequence_packing": bool(sequence_packing),
			},
			"training_args": {
				"output_dir": output_dir,
				"per_device_train_batch_size": int(per_device_train_batch_size),
				"gradient_accumulation_steps": int(gradient_accumulation_steps),
				"learning_rate": float(learning_rate),
				"logging_steps": int(logging_steps),
				"num_train_epochs": int(num_train_epochs),
				"lr_scheduler_type": lr_scheduler_type,
				"warmup_steps": int(warmup_steps),
				"save_total_limit": int(save_total_limit),
				"weight_decay": float(weight_decay),
				"optim": optim,
				"seed": int(seed),
				"report_to": report_to,
			},
		}

		_safe_kill_running()
		RUN_STATE["proc"] = subprocess.Popen(
			["python", "prepare_dataset/run_train_job.py"],
			stdin=subprocess.PIPE,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			text=True,
			bufsize=1,
		)
		assert RUN_STATE["proc"].stdin is not None
		RUN_STATE["proc"].stdin.write(json.dumps(cfg))
		RUN_STATE["proc"].stdin.close()

		out_lines: list[str] = []
		for line in RUN_STATE["proc"].stdout:  # type: ignore[arg-type]
			out_lines.append(line.rstrip("\n"))
			log_html = _to_log_html("\n".join(out_lines), elem_id="train-log")
			yield gr.update(value=log_html, visible=True)
		code = RUN_STATE["proc"].wait()
		RUN_STATE["proc"] = None
		final_html = _to_log_html("\n".join(out_lines) or ("Exit code: " + str(code)), elem_id="train-log")
		return gr.update(value=final_html, visible=True)
	except Exception:
		_safe_kill_running()
		return gr.update(value=traceback.format_exc(), visible=True)


def build_ui() -> gr.Blocks:
	with gr.Blocks(title="OpenSloth Studio") as demo:
		gr.Markdown("""
		# OpenSloth Studio
		Prepare chat datasets and fine-tune models with a simple UI.
		""")

		# Inject styles for log boxes
		gr.HTML("""
		<style>
		:root { --log-bg: #0b0e14; --log-fg: #f0f3f6; --log-border: #1f2937; }
		@media (prefers-color-scheme: light) { :root { --log-bg: #f8fafc; --log-fg: #111827; --log-border: #e5e7eb; } }
		.logbox { height: 340px; overflow-y: auto; background: var(--log-bg); color: var(--log-fg); padding: 10px; border-radius: 8px; border: 1px solid var(--log-border); white-space: pre-wrap; word-break: break-word; font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace; box-shadow: 0 1px 2px rgba(0,0,0,0.05) inset; }
		.small { font-size: 12px; color: #6b7280; }
		button.copy-btn { font-size: 12px; padding: 2px 8px; border: 1px solid var(--log-border); background: transparent; border-radius: 4px; color: var(--log-fg); cursor: pointer; }
		button.copy-btn:hover { background: rgba(0,0,0,0.04); }
		</style>
		""")

		# Data Processing Tab
		with gr.Tab("Data Processing"):
			with gr.Row():
				model_family = gr.Dropdown(
					label="Model Family",
					choices=list(_get_model_families().keys()),
					value="Qwen",
				)
				ds_preset = gr.Dropdown(
					label="Starter config",
					choices=["Qwen default", "Gemma default"] + _list_json_files(PRESETS_DATA_DIR),
					value="Qwen default",
				)
				refresh_datasets_btn = gr.Button("Refresh datasets")

			# Basic controls for new users
			with gr.Row():
				tok_name = gr.Textbox(label="Tokenizer / Model", value="unsloth/Qwen3-0.6B-Instruct")
				dataset_name = gr.Textbox(
					label="Dataset (HF repo or local file)",
					value="mlabonne/FineTome-100k",
				)

			with gr.Row():
				split = gr.Textbox(label="Split", value="train")
				num_samples = gr.Number(label="Num Samples (-1 for all)", value=-1, precision=0)
				debug = gr.Number(label="Debug (dump N samples)", value=0, precision=0)

			# Advanced options tucked away
			with gr.Accordion("Advanced options", open=False):
				with gr.Row():
					chat_template = gr.Textbox(label="Chat Template", value="qwen-3")
					num_proc = gr.Number(label="Workers", value=8, precision=0)
				with gr.Row():
					train_on_target_only = gr.Checkbox(label="Train on target only", value=True)
					instruction_part = gr.Textbox(label="Instruction marker", value="<start_of_turn>user\n")
					response_part = gr.Textbox(label="Response marker", value="<start_of_turn>assistant\n")

			with gr.Row():
				output_dir = gr.Textbox(label="Output dir (optional)")
				run_prep = gr.Button("Prepare Dataset")
				cancel_prep = gr.Button("Cancel")
				clear_prep = gr.Button("Clear logs")

			overrides_ds = gr.Textbox(label="Dataset Prep overrides (JSON)", lines=4, value="")

			out_dir_box = gr.Textbox(label="Prepared dataset path", interactive=False)
			prep_status = gr.HTML(visible=False)
			debug_html = gr.HTML(label="Debug preview (HTML)")

			# list cached datasets
			dataset_list = gr.Dropdown(
				label="Available prepared datasets",
				choices=["Auto (match current config)"] + _list_cached_datasets(),
				value="Auto (match current config)",
			)

			def _apply_starter_config(preset):
				# Determine which family to use from the preset choice
				family = "Qwen"
				if preset == "Qwen default":
					family = "Qwen"
					cfg = _default_dataset_prep("Qwen")
				elif preset == "Gemma default":
					family = "Gemma"
					cfg = _default_dataset_prep("Gemma")
				else:
					# This is a JSON preset file
					try:
						DatasetPrepConfig = _get_dataset_prep_config()
						preset_data = _load_json(os.path.join(PRESETS_DATA_DIR, preset))
						family = preset_data.get("model_family", family)
						cfg = DatasetPrepConfig(**preset_data)
					except Exception:
						# Fallback to Qwen default if preset loading fails
						family = "Qwen"
						cfg = _default_dataset_prep("Qwen")
				return (
					family,
					cfg.tok_name,
					cfg.chat_template,
					cfg.dataset_name,
					cfg.split,
					cfg.num_samples,
					cfg.num_proc,
					cfg.debug,
					cfg.train_on_target_only,
					cfg.instruction_part,
					cfg.response_part,
				)

			ds_preset.change(
				_apply_starter_config,
				inputs=[ds_preset],
				outputs=[
					model_family,
					tok_name,
					chat_template,
					dataset_name,
					split,
					num_samples,
					num_proc,
					debug,
					train_on_target_only,
					instruction_part,
					response_part,
				],
			)

			run_prep.click(
				run_dataset_prep_gui,
				inputs=[
					model_family,
					tok_name,
					chat_template,
					dataset_name,
					split,
					num_samples,
					num_proc,
					train_on_target_only,
					instruction_part,
					response_part,
					debug,
					output_dir,
					overrides_ds,
				],
				outputs=[out_dir_box, prep_status, debug_html],
			)

			def _refresh_ds():
				lst = _list_cached_datasets()
				starter_choices = ["Qwen default", "Gemma default"] + _list_json_files(PRESETS_DATA_DIR)
				return gr.update(choices=lst, value=(lst[0] if lst else None)), gr.update(choices=starter_choices)

			refresh_datasets_btn.click(_refresh_ds, outputs=[dataset_list, ds_preset])

			def _cancel_job():
				_safe_kill_running()
				return gr.update(value="Cancelled.", visible=True)

			cancel_prep.click(_cancel_job, outputs=[prep_status])
			clear_prep.click(lambda: gr.update(value="", visible=True), outputs=[prep_status])

			# Dataset presets inside Data tab
			_ensure_preset_dirs()
			with gr.Row():
				preset_name_ds = gr.Textbox(label="Preset name (dataset)")
				save_preset_ds = gr.Button("Save preset")
				load_preset_ds = gr.Dropdown(label="Load preset", choices=_list_json_files(PRESETS_DATA_DIR))

			def _collect_ds_cfg(*vals):
				(
					model_family_v,
					tok_name_v,
					chat_template_v,
					dataset_name_v,
					split_v,
					num_samples_v,
					num_proc_v,
					train_on_target_only_v,
					instruction_part_v,
					response_part_v,
					debug_v,
					output_dir_v,
				) = vals
				return {
					"model_family": model_family_v,
					"tok_name": tok_name_v,
					"chat_template": chat_template_v,
					"dataset_name": dataset_name_v,
					"split": split_v,
					"num_samples": num_samples_v,
					"num_proc": num_proc_v,
					"train_on_target_only": train_on_target_only_v,
					"instruction_part": instruction_part_v,
					"response_part": response_part_v,
					"debug": debug_v,
					"output_dir": output_dir_v,
				}

			def _save_ds_preset(name, *vals):
				if not name:
					return gr.update(), gr.update(choices=_list_json_files(PRESETS_DATA_DIR)), gr.update()
				data = _collect_ds_cfg(*vals)
				_ensure_preset_dirs()
				_save_json(os.path.join(PRESETS_DATA_DIR, f"{name}.json"), data)
				preset_choices = _list_json_files(PRESETS_DATA_DIR)
				starter_choices = ["Qwen default", "Gemma default"] + preset_choices
				return (
					gr.update(value="Saved preset."), 
					gr.update(choices=preset_choices),
					gr.update(choices=starter_choices)
				)

			def _apply_preset_file(fname):
				if not fname:
					return [gr.update() for _ in range(12)]
				try:
					DatasetPrepConfig = _get_dataset_prep_config()
					path = os.path.join(PRESETS_DATA_DIR, fname)
					preset_data = _load_json(path)
					cfg = DatasetPrepConfig(**preset_data)
					return [
						preset_data.get("model_family", "Qwen"),
						cfg.tok_name,
						cfg.chat_template,
						cfg.dataset_name,
						cfg.split,
						cfg.num_samples,
						cfg.num_proc,
						cfg.train_on_target_only,
						cfg.instruction_part,
						cfg.response_part,
						cfg.debug,
						cfg.output_dir or "",
					]
				except Exception as e:
					print(f"Error loading preset {fname}: {e}")
					return [gr.update() for _ in range(12)]

			save_preset_ds.click(
				_save_ds_preset,
				inputs=[
					preset_name_ds,
					model_family,
					tok_name,
					chat_template,
					dataset_name,
					split,
					num_samples,
					num_proc,
					train_on_target_only,
					instruction_part,
					response_part,
					debug,
					output_dir,
				],
				outputs=[prep_status, load_preset_ds, ds_preset],
			)

			load_preset_ds.change(
				_apply_preset_file,
				inputs=[load_preset_ds],
				outputs=[
					model_family,
					tok_name,
					chat_template,
					dataset_name,
					split,
					num_samples,
					num_proc,
					train_on_target_only,
					instruction_part,
					response_part,
					debug,
					output_dir,
				],
			)

		# Training Tab
		with gr.Tab("Training"):
			gr.Markdown("Quick config for training; advanced options are tucked away.")
			# Tip to avoid hot-reload issues if running with `gradio` CLI
			if not os.getenv("GRADIO_WATCH_DIRS"):
				gr.Markdown(
					"""
					Note: for stable reloads when using the `gradio` CLI, restrict watch dirs to this folder to avoid reloads from model caches:
					
					- export GRADIO_WATCH_DIRS="$(pwd)/prepare_dataset"
					
					Otherwise, changes in other folders (e.g. unsloth_compiled_cache) may trigger reload during long jobs.
					"""
				)
			# Basics first
			with gr.Row():
				data_cache_path = gr.Textbox(label="Dataset path", value="")
				ds_pick = gr.Dropdown(label="Pick prepared dataset", choices=_list_cached_datasets(), value=None)
				devices_csv = gr.Textbox(label="CUDA devices", value="0,1")
				sequence_packing = gr.Checkbox(label="Sequence packing", value=True)

			with gr.Row():
				model_name = gr.Textbox(label="Model name/path", value="unsloth/gemma-3-1b-it-unsloth-bnb-4bit")
				per_device_train_batch_size = gr.Number(label="Per-device batch size", value=2, precision=0)
				num_train_epochs = gr.Number(label="Num epochs", value=1, precision=0)
				learning_rate = gr.Number(label="Learning rate", value=1e-5)
				output_dir = gr.Textbox(label="Output dir", value="outputs/exps/exp1")

			# Advanced training
			with gr.Accordion("Advanced training options", open=False):
				with gr.Row():
					max_seq_length = gr.Number(label="Max seq length", value=16000, precision=0)
					load_in_4bit = gr.Checkbox(label="Load in 4-bit", value=True)
					load_in_8bit = gr.Checkbox(label="Load in 8-bit", value=False)
					full_finetuning = gr.Checkbox(label="Full finetuning", value=False)
				with gr.Row():
					r = gr.Number(label="LoRA r", value=8, precision=0)
					lora_alpha = gr.Number(label="LoRA alpha", value=16, precision=0)
					lora_dropout = gr.Number(label="LoRA dropout", value=0.0)
					target_modules_csv = gr.Textbox(
						label="LoRA target modules (comma)",
						value="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
					)
					use_rslora = gr.Checkbox(label="Use rslora", value=False)
				with gr.Row():
					gradient_accumulation_steps = gr.Number(label="Grad accumulation", value=16, precision=0)
					logging_steps = gr.Number(label="Logging steps", value=1, precision=0)
					lr_scheduler_type = gr.Textbox(label="LR scheduler", value="linear")
					warmup_steps = gr.Number(label="Warmup steps", value=5, precision=0)
					save_total_limit = gr.Number(label="Save total limit", value=1, precision=0)
					weight_decay = gr.Number(label="Weight decay", value=0.01)
					optim = gr.Textbox(label="Optimizer", value="adamw_8bit")
					seed = gr.Number(label="Seed", value=3407, precision=0)
					report_to = gr.Dropdown(label="Report to", choices=["none", "tensorboard", "wandb"], value="none")

			train_btn = gr.Button("Run Training")
			cancel_train = gr.Button("Cancel")
			clear_train = gr.Button("Clear logs")
			train_status = gr.HTML(visible=False)

			# Presets for training (auto apply)
			_ensure_preset_dirs()
			with gr.Row():
				preset_name_tr = gr.Textbox(label="Preset name (training)")
				save_preset_tr = gr.Button("Save preset")
				load_preset_tr = gr.Dropdown(label="Load preset", choices=_list_json_files(PRESETS_TRAIN_DIR))

			train_btn.click(
				launch_training_gui,
				inputs=[
					data_cache_path,
					devices_csv,
					model_name,
					max_seq_length,
					load_in_4bit,
					load_in_8bit,
					full_finetuning,
					r,
					lora_alpha,
					lora_dropout,
					target_modules_csv,
					use_rslora,
					output_dir,
					per_device_train_batch_size,
					gradient_accumulation_steps,
					learning_rate,
					logging_steps,
					num_train_epochs,
					lr_scheduler_type,
					warmup_steps,
					save_total_limit,
					weight_decay,
					optim,
					seed,
					report_to,
					sequence_packing,
				],
				outputs=[train_status],
			)

			ds_pick.change(lambda v: gr.update(value=v or ""), inputs=[ds_pick], outputs=[data_cache_path])
			cancel_train.click(lambda: _cancel_job(), outputs=[train_status])
			clear_train.click(lambda: gr.update(value="", visible=True), outputs=[train_status])

			# Save/load training presets
			def _collect_train_cfg(
				data_cache_path, devices_csv, model_name, max_seq_length, load_in_4bit, load_in_8bit, full_finetuning,
				r, lora_alpha, lora_dropout, target_modules_csv, use_rslora, output_dir,
				per_device_train_batch_size, gradient_accumulation_steps, learning_rate, logging_steps,
				num_train_epochs, lr_scheduler_type, warmup_steps, save_total_limit, weight_decay, optim, seed, report_to,
				sequence_packing,
			):
				return {
					"data_cache_path": data_cache_path,
					"devices_csv": devices_csv,
					"model_name": model_name,
					"max_seq_length": max_seq_length,
					"load_in_4bit": load_in_4bit,
					"load_in_8bit": load_in_8bit,
					"full_finetuning": full_finetuning,
					"r": r,
					"lora_alpha": lora_alpha,
					"lora_dropout": lora_dropout,
					"target_modules_csv": target_modules_csv,
					"use_rslora": use_rslora,
					"output_dir": output_dir,
					"per_device_train_batch_size": per_device_train_batch_size,
					"gradient_accumulation_steps": gradient_accumulation_steps,
					"learning_rate": learning_rate,
					"logging_steps": logging_steps,
					"num_train_epochs": num_train_epochs,
					"lr_scheduler_type": lr_scheduler_type,
					"warmup_steps": warmup_steps,
					"save_total_limit": save_total_limit,
					"weight_decay": weight_decay,
					"optim": optim,
					"seed": seed,
					"report_to": report_to,
					"sequence_packing": sequence_packing,
				}

			def _save_train_preset(name, *vals):
				if not name:
					return gr.update(), gr.update(choices=_list_json_files(PRESETS_TRAIN_DIR))
				data = _collect_train_cfg(*vals)
				_ensure_preset_dirs()
				_save_json(os.path.join(PRESETS_TRAIN_DIR, f"{name}.json"), data)
				return gr.update(value="Saved preset."), gr.update(choices=_list_json_files(PRESETS_TRAIN_DIR))

			def _apply_train_preset(fname):
				if not fname:
					return [gr.update() for _ in range(26)]
				path = os.path.join(PRESETS_TRAIN_DIR, fname)
				cfg = _load_json(path)
				# Return values in the exact order of outputs specified in load_preset_tr.change
				return [
					cfg.get("data_cache_path", ""),                               # data_cache_path
					cfg.get("devices_csv", "0,1"),                                 # devices_csv
					cfg.get("model_name", ""),                                    # model_name
					cfg.get("per_device_train_batch_size", 2),                      # per_device_train_batch_size
					cfg.get("learning_rate", 1e-5),                                 # learning_rate
					cfg.get("num_train_epochs", 1),                                 # num_train_epochs
					cfg.get("sequence_packing", True),                              # sequence_packing
					cfg.get("output_dir", "outputs/exps/exp1"),                    # output_dir
					cfg.get("max_seq_length", 16000),                               # max_seq_length
					cfg.get("load_in_4bit", True),                                  # load_in_4bit
					cfg.get("load_in_8bit", False),                                 # load_in_8bit
					cfg.get("full_finetuning", False),                              # full_finetuning
					cfg.get("r", 8),                                                # r
					cfg.get("lora_alpha", 16),                                      # lora_alpha
					cfg.get("lora_dropout", 0.0),                                   # lora_dropout
					cfg.get("target_modules_csv", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"),  # target_modules_csv
					cfg.get("use_rslora", False),                                   # use_rslora
					cfg.get("gradient_accumulation_steps", 16),                      # gradient_accumulation_steps
					cfg.get("logging_steps", 1),                                    # logging_steps
					cfg.get("lr_scheduler_type", "linear"),                         # lr_scheduler_type
					cfg.get("warmup_steps", 5),                                     # warmup_steps
					cfg.get("save_total_limit", 1),                                 # save_total_limit
					cfg.get("weight_decay", 0.01),                                  # weight_decay
					cfg.get("optim", "adamw_8bit"),                                # optim
					cfg.get("seed", 3407),                                          # seed
					cfg.get("report_to", "none"),                                  # report_to
				]

			save_preset_tr.click(
				_save_train_preset,
				inputs=[
					preset_name_tr,
					data_cache_path,
					devices_csv,
					model_name,
					max_seq_length,
					load_in_4bit,
					load_in_8bit,
					full_finetuning,
					r,
					lora_alpha,
					lora_dropout,
					target_modules_csv,
					use_rslora,
					output_dir,
					per_device_train_batch_size,
					gradient_accumulation_steps,
					learning_rate,
					logging_steps,
					num_train_epochs,
					lr_scheduler_type,
					warmup_steps,
					save_total_limit,
					weight_decay,
					optim,
					seed,
					report_to,
					sequence_packing,
				],
				outputs=[train_status, load_preset_tr],
			)

			load_preset_tr.change(
				_apply_train_preset,
				inputs=[load_preset_tr],
				outputs=[
					data_cache_path,
					devices_csv,
					model_name,
					per_device_train_batch_size,
					learning_rate,
					num_train_epochs,
					sequence_packing,
					output_dir,
					max_seq_length,
					load_in_4bit,
					load_in_8bit,
					full_finetuning,
					r,
					lora_alpha,
					lora_dropout,
					target_modules_csv,
					use_rslora,
					gradient_accumulation_steps,
					logging_steps,
					lr_scheduler_type,
					warmup_steps,
					save_total_limit,
					weight_decay,
					optim,
					seed,
					report_to,
				],
			)

	return demo


demo = build_ui()

if __name__ == "__main__":
	import os
	os.environ["no_proxy"] = "localhost, 127.0.0.1"
	demo.launch(theme=gr.themes.Monochrome())
