"""
OpenSloth GRPO Wrapper
Simple wrapper around Unsloth + TRL GRPOTrainer for multi-GPU support
"""

import os
import numpy as np
from typing import Dict, List

from opensloth.logging_config import OpenslothLogger
from opensloth.opensloth_config import OpenSlothConfig, TrainingArguments
from opensloth.grpo_rewards import (
    get_reward_functions, 
    create_reward_preset, 
    get_chat_template_for_task
)


def setup_grpo_training(
    opensloth_config: OpenSlothConfig,
    hf_train_args: TrainingArguments,
    logger: OpenslothLogger,
    gpu: int,
    unsloth_modules: Dict
):
    """
    Setup GRPO training using native Unsloth + TRL GRPOTrainer.
    This is a thin wrapper that follows the Unsloth tutorial pattern exactly.
    """
    # Get modules from the passed dictionary
    FastLanguageModel = unsloth_modules["FastLanguageModel"]
    GRPOConfig = unsloth_modules["GRPOConfig"]
    GRPOTrainer = unsloth_modules["GRPOTrainer"]
    
    from datasets import load_from_disk
    from vllm import SamplingParams
    
    grpo_args = opensloth_config.grpo_args
    if grpo_args is None:
        raise ValueError("grpo_args must be configured for GRPO training")
    
    # Load model and tokenizer using Unsloth's FastLanguageModel
    logger.start_timing("model_loading")
    model, tokenizer = FastLanguageModel.from_pretrained(
        **opensloth_config.fast_model_args.model_dump()
    )
    logger.finish_timing("model_loading")
    
    # Apply LoRA
    if not opensloth_config.fast_model_args.full_finetuning and opensloth_config.lora_args:
        logger.start_timing("lora_setup")
        
        lora_dict = opensloth_config.lora_args.model_dump()
        supported_lora_args = {
            'r', 'lora_alpha', 'lora_dropout', 'bias', 'target_modules', 
            'use_rslora', 'random_state'
        }
        filtered_lora_args = {k: v for k, v in lora_dict.items() if k in supported_lora_args}
        
        model = FastLanguageModel.get_peft_model(model, **filtered_lora_args)
        logger.finish_timing("lora_setup")
    
    # Setup chat template
    if grpo_args.use_custom_chat_template:
        template = get_chat_template_for_task(grpo_args.task_type, tokenizer.eos_token)
        tokenizer.chat_template = template
        logger.info(f"Applied {grpo_args.task_type} chat template")
    
    # Load dataset
    logger.start_timing("dataset_loading")
    train_dataset = load_from_disk(opensloth_config.data_cache_path)
    logger.info(f"Loaded dataset with {len(train_dataset)} samples")
    
    # Validate dataset format
    logger.info("Validating dataset format...")
    try:
        if len(train_dataset) == 0:
            raise ValueError("Dataset is empty")
        
        required_columns = ["prompt", "answer"]
        dataset_columns = set(train_dataset.column_names)
        missing_columns = [col for col in required_columns if col not in dataset_columns]
        
        if missing_columns:
            raise ValueError(f"Dataset missing required columns: {missing_columns}. Available columns: {list(dataset_columns)}")
        
        # Check first sample
        sample = train_dataset[0]
        if not isinstance(sample["prompt"], str) or not sample["prompt"].strip():
            raise ValueError(f"First sample has invalid prompt: {repr(sample['prompt'])}")
        if not isinstance(sample["answer"], str) or not sample["answer"].strip():
            raise ValueError(f"First sample has invalid answer: {repr(sample['answer'])}")
        
        logger.info("Dataset format validation passed")
        logger.info(f"Dataset columns: {list(dataset_columns)}")
        logger.info(f"Sample prompt length: {len(sample['prompt'])} chars")
        logger.info(f"Sample answer: '{sample['answer']}'")
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        raise ValueError(f"Invalid dataset format: {e}")
    
    logger.finish_timing("dataset_loading")
    
    # Filter dataset by prompt length (following tutorial)
    logger.start_timing("prompt_length_filtering")
    
    def tokenize_prompts_with_error_handling(examples):
        """Tokenize prompts with proper batch handling and error tracing."""
        try:
            if isinstance(examples["prompt"], list):
                # Batched input - handle each prompt individually
                tokens_list = []
                for i, prompt in enumerate(examples["prompt"]):
                    try:
                        # The prompt is already formatted, just tokenize directly
                        tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                        tokens_list.append(tokens)
                    except Exception as e:
                        logger.error(f"Failed to tokenize prompt {i}: {str(e)}")
                        logger.error(f"Problematic prompt: {prompt[:200]}...")
                        raise ValueError(f"Tokenization failed for prompt {i}: {str(e)}")
                return {"tokens": tokens_list}
            else:
                # Single input
                try:
                    # The prompt is already formatted, just tokenize directly
                    tokens = tokenizer(examples["prompt"], add_special_tokens=False)["input_ids"]
                    return {"tokens": tokens}
                except Exception as e:
                    logger.error(f"Failed to tokenize single prompt: {str(e)}")
                    logger.error(f"Problematic prompt: {examples['prompt'][:200]}...")
                    raise ValueError(f"Tokenization failed for single prompt: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error in tokenize_prompts_with_error_handling: {str(e)}")
            logger.error(f"Examples type: {type(examples)}")
            logger.error(f"Examples keys: {list(examples.keys()) if isinstance(examples, dict) else 'Not a dict'}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    logger.info("Starting tokenization of prompts...")
    tokenized = train_dataset.map(
        tokenize_prompts_with_error_handling,
        batched=True,
    )
    logger.info("Tokenization completed, calculating lengths...")
    
    def calculate_lengths_with_error_handling(examples):
        """Calculate token lengths with proper batch handling and error tracing."""
        try:
            if isinstance(examples["tokens"], list) and len(examples["tokens"]) > 0:
                # Check if it's a batch of token lists
                if isinstance(examples["tokens"][0], list):
                    # Batched input with list of token lists
                    return {"L": [len(tokens) for tokens in examples["tokens"]]}
                else:
                    # Single token list wrapped in examples
                    return {"L": len(examples["tokens"])}
            else:
                logger.error(f"Unexpected tokens format: {type(examples['tokens'])}")
                raise ValueError(f"Unexpected tokens format: {type(examples['tokens'])}")
        except Exception as e:
            logger.error(f"Error calculating lengths: {str(e)}")
            logger.error(f"Tokens type: {type(examples.get('tokens', 'Missing'))}")
            if 'tokens' in examples:
                logger.error(f"Tokens sample: {str(examples['tokens'])[:200]}...")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    tokenized = tokenized.map(calculate_lengths_with_error_handling, batched=True)
    maximum_length = int(np.quantile(tokenized["L"], grpo_args.prompt_length_percentile))
    maximum_length = min(maximum_length, grpo_args.max_prompt_length)
    
    logger.info(f"Max prompt length ({grpo_args.prompt_length_percentile*100:.0f}th percentile): {maximum_length}")
    
    keep_idx = np.where(np.array(tokenized["L"]) <= maximum_length)[0]
    train_dataset = train_dataset.select(keep_idx)
    del tokenized
    
    max_prompt_length = maximum_length + 1
    
    # For GRPO, we want to limit completions to max_new_tokens, not the full remaining sequence
    # This ensures consistency between vLLM generation and GRPO trainer expectations
    grpo_max_completion = min(
        grpo_args.max_new_tokens,  # User-specified limit for new tokens
        opensloth_config.fast_model_args.max_seq_length - max_prompt_length  # Technical limit
    )
    
    logger.info(f"Filtered dataset: {len(train_dataset)} samples")
    logger.info(f"Prompt limits: max_prompt_length={max_prompt_length}")
    logger.info(f"Completion limits: grpo_max_completion={grpo_max_completion} (from max_new_tokens={grpo_args.max_new_tokens})")
    logger.info(f"Total sequence limit: {max_prompt_length + grpo_max_completion} / {opensloth_config.fast_model_args.max_seq_length}")
    logger.finish_timing("prompt_length_filtering")
    
    # Setup reward functions
    if grpo_args.reward_functions:
        reward_names = grpo_args.reward_functions
    else:
        reward_names = create_reward_preset(grpo_args.task_type)
        logger.info(f"Auto-selected reward functions for '{grpo_args.task_type}': {reward_names}")
    
    reward_functions = get_reward_functions(reward_names)
    logger.info(f"Using reward functions: {[rf.name for rf in reward_functions]}")
    
    # Convert our reward functions to TRL format
    def create_trl_reward_func(reward_func):
        def trl_reward(prompts, completions, **kwargs):
            return reward_func(prompts, completions, **kwargs)
        return trl_reward
    
    trl_reward_funcs = [create_trl_reward_func(rf) for rf in reward_functions]
    
    # Setup vLLM sampling (exactly like tutorial)
    vllm_sampling_params = SamplingParams(
        min_p=grpo_args.min_p,
        top_p=grpo_args.top_p,
        top_k=grpo_args.top_k if grpo_args.top_k else -1,
        temperature=grpo_args.temperature,
        max_tokens=grpo_args.max_new_tokens,
        seed=42,
        stop=[tokenizer.eos_token] + grpo_args.stop_sequences,
        include_stop_str_in_output=grpo_args.include_stop_str_in_output,
    )
    
    # Create GRPOConfig (following tutorial exactly)
    logger.start_timing("grpo_config_creation")
    logger.info(f"vLLM Sampling Parameters:")
    logger.info(f"  - max_tokens: {vllm_sampling_params.max_tokens} (this limits NEW tokens per generation)")
    logger.info(f"  - temperature: {vllm_sampling_params.temperature}")
    logger.info(f"  - top_p: {vllm_sampling_params.top_p}, top_k: {vllm_sampling_params.top_k}, min_p: {vllm_sampling_params.min_p}")
    logger.info(f"  - stop: {vllm_sampling_params.stop}")
    logger.info(f"Max prompt length: {max_prompt_length}, Max completion length: {grpo_max_completion}")
    logger.info(f"Total sequence length limit: {max_prompt_length + grpo_max_completion}")
    logger.info(f"NOTE: vLLM max_tokens={vllm_sampling_params.max_tokens} vs GRPO max_completion_length={grpo_max_completion}")
    
    # Verify token limits are consistent
    if vllm_sampling_params.max_tokens > grpo_max_completion:
        logger.warning(f"⚠️  vLLM max_tokens ({vllm_sampling_params.max_tokens}) > max_completion_length ({grpo_max_completion})")
        logger.warning(f"    This may cause truncation during training. Consider adjusting max_new_tokens or max_seq_length.")
    training_args = GRPOConfig(
        vllm_sampling_params=vllm_sampling_params,
        temperature=grpo_args.temperature,
        learning_rate=hf_train_args.learning_rate,
        weight_decay=hf_train_args.weight_decay,
        warmup_ratio=getattr(hf_train_args, 'warmup_ratio', 0.1),
        lr_scheduler_type=hf_train_args.lr_scheduler_type,
        optim=hf_train_args.optim,
        logging_steps=hf_train_args.logging_steps,
        per_device_train_batch_size=hf_train_args.per_device_train_batch_size,
        gradient_accumulation_steps=hf_train_args.gradient_accumulation_steps,
        num_generations=grpo_args.group_size,
        max_prompt_length=max_prompt_length,
        max_completion_length=grpo_max_completion,
        max_steps=getattr(hf_train_args, 'max_steps', 100),
        save_steps=getattr(hf_train_args, 'save_steps', 100),
        report_to="none",  # Avoid JSON serialization issues
        output_dir=hf_train_args.output_dir,
    )
    logger.finish_timing("grpo_config_creation")
    
    # Create GRPOTrainer (exactly like tutorial)
    logger.start_timing("grpo_trainer_creation")
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=trl_reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
    )
    logger.finish_timing("grpo_trainer_creation")
    
    return trainer, model, tokenizer


def run_grpo_training(trainer, model, tokenizer, logger, gpu, opensloth_config):
    """Run GRPO training and save results."""
    
    # Setup multi-GPU gradient sync if needed
    if len(opensloth_config.devices) > 1:
        from opensloth.nccl_grad_sync import get_callback_and_setup_method
        nccl_grad_sync_callback, setup_nccl_for_opensloth = get_callback_and_setup_method()
        
        # Setup NCCL
        setup_nccl_for_opensloth(
            rank=int(os.environ["OPENSLOTH_LOCAL_RANK"]),
            gpus=opensloth_config.devices,
        )
        
        grad_sync_cb = nccl_grad_sync_callback(
            model=trainer.model,
            gpu=gpu,
            gpus=opensloth_config.devices,
        )
        logger.info(f"Using gradient sync callback for GPU {gpu}")
        trainer.add_callback(grad_sync_cb)
    else:
        logger.info("Single GPU GRPO training detected")
    
    # Start training
    logger.start_timing("actual_training")
    try:
        logger.info("Starting GRPO training...")
        trainer.train()
        logger.info("GRPO training completed successfully!")
    except Exception as e:
        logger.error(f"GRPO training failed: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        
        # Add comprehensive error details
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"Full traceback:\n{full_traceback}")
        
        # Add context about the trainer state
        try:
            logger.error(f"Trainer state: {trainer.state}")
            logger.error(f"Training dataset length: {len(trainer.train_dataset) if hasattr(trainer, 'train_dataset') else 'Unknown'}")
            logger.error(f"Model device: {next(trainer.model.parameters()).device if hasattr(trainer, 'model') else 'Unknown'}")
        except Exception as context_error:
            logger.error(f"Could not get trainer context: {context_error}")
        
        raise
    logger.finish_timing("actual_training")
    
    # Save model from rank 0 only
    if gpu == opensloth_config.devices[0]:
        logger.start_timing("model_saving")
        output_dir = trainer.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Save LoRA adapter (following tutorial)
        lora_save_path = os.path.join(output_dir, "grpo_lora_adapter")
        
        # Check if model has save_lora method, otherwise use save_pretrained_merged
        if hasattr(model, 'save_lora'):
            model.save_lora(lora_save_path)
            logger.info(f"Saved GRPO LoRA adapter to {lora_save_path}")
        elif hasattr(model, 'save_pretrained'):
            # For LoRA models, save the adapter
            if hasattr(model, 'peft_config'):
                model.save_pretrained(lora_save_path)
                logger.info(f"Saved GRPO LoRA adapter to {lora_save_path}")
            else:
                # Full model save
                model.save_pretrained(output_dir)
                logger.info(f"Saved GRPO model to {output_dir}")
        else:
            logger.warning("Model doesn't have save_lora or save_pretrained method")
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Saved tokenizer to {output_dir}")
        logger.finish_timing("model_saving")
    
    logger.log_training_summary()
