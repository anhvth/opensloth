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