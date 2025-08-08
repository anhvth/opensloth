# How to Prepare and Store a Trainer Dataset

Follow these steps to extract and save a dataset from an Unsloth notebook:

1. Visit the [Unsloth Notebooks Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks).
2. Select the notebook for your target model.
3. Export the notebook to a Python script.
4. Copy all code up to (but not including) `trainer.train()`.
5. Run the code to initialize the trainer.
6. Save the trainer's dataset:
7. [Optional] if modify the dataset to your internal use case

   ```python
   trainer.train_dataset.save_to_disk("data/cache_qwen3_dataset")
   ```
NOTE: this task is about saving the dataset, not training the model.
This will store the processed dataset for later use.

## OpenSloth Studio (GUI)

You can now prepare datasets and launch training from a simple UI:

1. Start the app:
   - Programmatic: `python -m prepare_dataset.app_gradio`
2. Use the "Data Processing" tab to generate a chat dataset (HF or local file):
   - Choose model family (Qwen/Gemma)
   - Fill tokenizer, chat template, dataset path/repo, and markers
   - Click "Prepare Dataset"; output path is shown on success
3. Use the "Training" tab to configure and run fine-tuning:
   - Maps to `train_scripts/train_*.py` parameters (OpenSlothConfig + TrainingArguments)
   - Click "Run Training" to start. Logs stream to the console.

Programmatic API: every `prepare_*.py` preparer exposes `run_with_config(dict)` which returns the output directory.