# OpenSloth GRPO Tutorial: Building a Math Reasoning Model

This tutorial demonstrates a powerful two-stage technique to create a high-quality reasoning model using OpenSloth. We will first use Supervised Fine-Tuning (SFT) to teach a base model a specific format for mathematical proofs, and then use Group Relative Policy Optimization (GRPO) to improve its reasoning capabilities.

This workflow is based on the popular [Unsloth Qwen3-GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb).

## The Two-Stage Workflow

1.  **Stage 1: Supervised Fine-Tuning (SFT)**
    -   **Goal:** Teach the model a specific output format for reasoning. This "pre-finetuning" step makes the subsequent GRPO training more stable and efficient, as the model already knows how to structure its responses.
    -   **Dataset:** We use a small, curated set of examples (`unsloth/OpenMathReasoning-mini`) formatted with custom tags like `<start_working_out>` and `<SOLUTION>`.

2.  **Stage 2: Group Relative Policy Optimization (GRPO)**
    -   **Goal:** Improve the model's mathematical reasoning and correctness.
    -   **Dataset:** We use a larger preference dataset (`open-r1/DAPO-Math-17k-Processed`) containing math problems.
    -   **Process:** The model generates multiple responses for each problem, and OpenSloth's built-in reward functions score them based on correctness and formatting. The model is then updated to prefer higher-scoring responses. We use the LoRA adapter from the SFT stage as the starting point for this training.

---

## Prerequisites

Ensure you have `opensloth` and its dependencies installed.

```bash
pip install "openslothai[cli] @ git+https://github.com/anhvth/opensloth.git"
```

## How to Run the Tutorial

The entire pipeline is automated with just **2 simple scripts** using OpenSloth's new unified `os-train` CLI. Each script handles both data preparation and training in one command.

```bash
# Make sure you are in the root directory of the opensloth project
cd /path/to/opensloth

# Grant execution permissions to the scripts
chmod +x grpo_tutorial/scripts/*.sh

# Run the 2-step workflow
bash grpo_tutorial/scripts/01_sft_end_to_end.sh      # SFT: Data prep + training
bash grpo_tutorial/scripts/02_grpo_end_to_end.sh     # GRPO: Data prep + training
```

### Previous 4-Script Workflow (Still Available)

If you prefer the step-by-step approach, the original 4 scripts are still available:

```bash
bash grpo_tutorial/scripts/00_prepare_sft_dataset.sh
bash grpo_tutorial/scripts/01_train_sft.sh
bash grpo_tutorial/scripts/02_prepare_grpo_dataset.sh
bash grpo_tutorial/scripts/03_train_grpo.sh
```

---

## Step-by-Step Breakdown

### New Unified Workflow (Recommended)

**Two Scripts, Complete Pipeline:**

#### Step 1: End-to-End SFT Training

**Script:** `grpo_tutorial/scripts/01_sft_end_to_end.sh`

This script uses OpenSloth's new unified `os-train sft` command to:
1. Prepare raw SFT data (downloads and formats `unsloth/OpenMathReasoning-mini`)
2. Automatically tokenize and shard the dataset
3. Run supervised fine-tuning in one seamless command

-   **Input:** Raw `unsloth/OpenMathReasoning-mini` dataset (downloaded automatically)
-   **Output:** A LoRA adapter trained for reasoning format at `outputs/sft_qwen_reasoning_model`

#### Step 2: End-to-End GRPO Training

**Script:** `grpo_tutorial/scripts/02_grpo_end_to_end.sh`

This script uses OpenSloth's new unified `os-train grpo` command to:
1. Automatically download and process `open-r1/DAPO-Math-17k-Processed`
2. Tokenize and prepare the preference dataset
3. Run GRPO training using the SFT model as the base

-   **Input:** 
    -   Raw `open-r1/DAPO-Math-17k-Processed` dataset (downloaded automatically)
    -   SFT model from Step 1 (`outputs/sft_qwen_reasoning_model`)
-   **Output:** Final reasoning model at `outputs/grpo_final_model`

### Traditional Step-by-Step Workflow (Still Available)

The original 4-script approach remains available for users who prefer granular control:

### Step 1: Prepare the SFT Dataset

**Script:** `grpo_tutorial/scripts/00_prepare_sft_dataset.sh`

This script first runs a Python helper (`prepare_sft_data.py`) to download and format the `unsloth/OpenMathReasoning-mini` dataset into the custom structure required. It then uses the `os-data` CLI to tokenize and prepare this data for SFT.

-   **Input:** Raw `unsloth/OpenMathReasoning-mini` dataset from Hugging Face.
-   **Output:** A tokenized, sharded dataset ready for SFT, located at `data/sft_openmath_prepared`.

### Step 2: Run Supervised Fine-Tuning (SFT)

**Script:** `grpo_tutorial/scripts/01_train_sft.sh`

This script takes the prepared SFT dataset and runs a short training session using `os-sft`. The goal is not to make the model a math expert yet, but simply to teach it the formatting.

-   **Input:** The prepared dataset from `data/sft_openmath_prepared`.
-   **Output:** A LoRA adapter trained to produce the correct format, saved to `outputs/sft_qwen_reasoning_model`.

### Step 3: Prepare the GRPO Dataset

**Script:** `grpo_tutorial/scripts/02_prepare_grpo_dataset.sh`

Now we prepare the main dataset for preference tuning. This script uses `os-data` with `--method grpo` to process the `open-r1/DAPO-Math-17k-Processed` dataset. This creates a dataset with `prompt` and `answer` columns suitable for the `GRPOTrainer`.

-   **Input:** Raw `open-r1/DAPO-Math-17k-Processed` dataset from Hugging Face.
-   **Output:** A dataset ready for GRPO, located at `data/grpo_dapo_prepared`.

### Step 4: Run Group Relative Policy Optimization (GRPO)

**Script:** `grpo_tutorial/scripts/03_train_grpo.sh`

This is the final and most important step. We use `os-grpo` to train the model. Crucially, we use the `--model` flag to point to the LoRA adapter we created in the SFT step. This continues the training from our format-aware checkpoint.

OpenSloth's GRPO trainer automatically uses a suite of math-focused reward functions to score the generated answers for correctness and format. The tutorial script now explicitly passes `--task math`, which maps to the preset reward bundle:

`math_format`, `math_answer`, `math_number`, and a lightweight `demo_reward` printer.

-   **Input:**
    -   The prepared GRPO dataset from `data/grpo_dapo_prepared`.
    -   The SFT LoRA adapter from `outputs/sft_qwen_reasoning_model`.
    -   Reward preset selected via `--task math`.
-   **Output:** The final, reasoning-capable LoRA adapter, saved to `outputs/grpo_final_model`.

### Step 5: Inference with the Final Model

After the final script completes, your powerful reasoning model is ready at `outputs/grpo_final_model`. You can now use it for inference.

> **Note:** The OpenSloth CLI is focused on training. For inference, you can use the `unsloth` library directly.

Here's a sample Python snippet to load your model and run inference:

```python
from unsloth import FastLanguageModel
import torch

# Define the custom chat template parts
system_prompt = "You are given a problem.\nThink about the problem and provide your working out.\nPlace it between <start_working_out> and <end_working_out>.\nThen, provide your solution between <SOLUTION></SOLUTION>"
reasoning_start = "<start_working_out>"

chat_template_str = (
    "{% if messages[0]['role'] == 'system' %}"
        "{{ messages[0]['content'] + eos_token }}"
    "{% else %}"
        "{{ '" + system_prompt + "' + eos_token }}"
    "{% endif %}"
    "{% for message in messages[1:] %}"
        "{% if message['role'] == 'user' %}"
            "{{ message['content'] }}"
        "{% elif message['role'] == 'assistant' %}"
            "{{ message['content'] + eos_token }}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '" + reasoning_start + "' }}{% endif %}"
)


# Load the base model and apply the final GRPO LoRA adapter
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/grpo_final_model", # Load the final LoRA
    load_in_4bit=True,
)
tokenizer.chat_template = chat_template_str

# Prepare the prompt
messages = [
    {"role": "user", "content": "What is the square root of 101?"},
]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

# Generate the response
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer=text_streamer, max_new_tokens=512)
```

Congratulations! You have successfully built a specialized reasoning model using OpenSloth.

---

## Understanding the Workflow

### Why Two Stages?

The two-stage approach is crucial for optimal results:

1. **Format Learning (SFT):** The model first learns the specific formatting conventions (`<start_working_out>`, `<SOLUTION>`, etc.) through traditional supervised fine-tuning. This creates a stable foundation.

2. **Reasoning Improvement (GRPO):** With the format already learned, the model can focus on improving the quality and correctness of its reasoning during preference optimization.

### Dataset Choices

- **SFT Dataset (`unsloth/OpenMathReasoning-mini`):** A small, high-quality dataset focused on teaching the format.
- **GRPO Dataset (`open-r1/DAPO-Math-17k-Processed`):** A larger dataset with diverse math problems for comprehensive reasoning improvement.

### Key Benefits of OpenSloth's Approach

- **Unified CLI (`os-train`):** Combines data preparation and training into single commands for streamlined workflows
- **Automated Data Processing:** Automatically downloads, tokenizes, and shards datasets for optimal multi-GPU performance  
- **Automated Reward Functions:** OpenSloth includes built-in reward functions specifically designed for mathematical reasoning
- **Efficient Multi-GPU Training:** The CLI tools automatically handle distributed training across multiple GPUs
- **Memory Optimization:** Uses advanced techniques like gradient checkpointing and mixed precision to handle large models efficiently
- **GPU Consistency:** Ensures data sharding and training use the same number of GPUs to prevent configuration errors

---

## Troubleshooting

### Common Issues

1. **GPU Memory Errors:** Reduce batch size or use more aggressive quantization (e.g., `--load_in_4bit`).
2. **Dataset Loading Errors:** Ensure you have sufficient disk space and stable internet connection for dataset downloads.
3. **NCCL Errors:** Make sure all GPUs are properly configured and visible to CUDA.

### Monitoring Training

You can monitor training progress through:
- **TensorBoard:** OpenSloth automatically logs metrics that can be viewed with TensorBoard.
- **Console Output:** Each script provides detailed progress information.
- **Checkpoint Files:** Intermediate checkpoints are saved automatically.

---

## Next Steps

After completing this tutorial, you can:

1. **Experiment with Different Models:** Try the workflow with other base models like Llama, Mistral, or newer Qwen variants.
2. **Custom Datasets:** Adapt the data preparation scripts to work with your own datasets.
3. **Hyperparameter Tuning:** Modify the training scripts to experiment with different learning rates, batch sizes, and other parameters.
4. **Production Deployment:** Use the final model in production systems or integrate it into larger applications.

---

## Additional Resources

- [OpenSloth Documentation](https://github.com/cognitivecomputations/opensloth)
- [Unsloth GRPO Notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen3_(4B)-GRPO.ipynb)
- [GRPO Paper](https://arxiv.org/abs/2402.03300)
- [Mathematical Reasoning Benchmarks](https://paperswithcode.com/task/mathematical-reasoning)
