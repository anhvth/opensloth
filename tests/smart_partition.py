import torch
from speedy_utils import *


def smart_partition_batches(all_batches: list, num_devices: int) -> dict:
    """
    Partition batches across devices by sorting sequences by length.

    Args:
        all_batches: List of batch dictionaries with input_ids, labels, attention_mask
        num_devices: Number of devices to partition across

    Returns:
        Dictionary mapping device indices to lists of batches
    """
    perdevice_batch_size = all_batches[0]["input_ids"].shape[0] // num_devices

    # Flatten all sequences
    input_ids = [
        batch["input_ids"][i]
        for batch in all_batches
        for i in range(len(batch["input_ids"]))
    ]
    labels = [
        batch["labels"][i] for batch in all_batches for i in range(len(batch["labels"]))
    ]
    attention_mask = [
        batch["attention_mask"][i]
        for batch in all_batches
        for i in range(len(batch["attention_mask"]))
    ]

    # Sort by sequence length
    num_tokens = [mask.sum().item() for mask in attention_mask]
    sorted_ids = sorted(range(len(num_tokens)), key=lambda i: num_tokens[i])

    new_input_ids = [input_ids[i] for i in sorted_ids]
    new_labels = [labels[i] for i in sorted_ids]
    new_attention_mask = [attention_mask[i] for i in sorted_ids]
    lens = [num_tokens[i] for i in sorted_ids]

    def item_to_perdevice_batch(items, counts, padding_value) -> torch.Tensor:
        max_len = max(counts)
        output_tensor = torch.full(
            (len(items), max_len),
            padding_value,
            dtype=items[0].dtype,
            device=items[0].device,
        )
        for i, item in enumerate(items):
            output_tensor[i, : counts[i]] = item[: counts[i]]
        return output_tensor

    # Partition into device batches
    by_gpu_batches = {gpu_idx: [] for gpu_idx in range(num_devices)}
    gpu_idx = 0

    for i in range(0, len(new_input_ids), perdevice_batch_size):
        input_ids_tensor = item_to_perdevice_batch(
            new_input_ids[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            0,
        )
        labels_tensor = item_to_perdevice_batch(
            new_labels[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            -100,
        )
        attention_mask_tensor = item_to_perdevice_batch(
            new_attention_mask[i : i + perdevice_batch_size],
            lens[i : i + perdevice_batch_size],
            0,
        )

        by_gpu_batches[gpu_idx].append(
            {
                "input_ids": input_ids_tensor,
                "labels": labels_tensor,
                "attention_mask": attention_mask_tensor,
            }
        )
        gpu_idx = (gpu_idx + 1) % num_devices

    return by_gpu_batches


# Load and process batches
all_batches = load_by_ext("./assets/all_batches.pkl")

# Count tokens before processing
total_train_tokens_before = 0
total_num_tokens_before = 0
for batch in all_batches:
    total_train_tokens_before += batch["attention_mask"].sum().item()
    total_num_tokens_before += batch["input_ids"].numel()

print(
    f"Before Total train tokens: {total_train_tokens_before}, "
    f"total num tokens: {total_num_tokens_before}, "
    f"train percentage: {total_train_tokens_before / total_num_tokens_before:.2%}"
)

# Partition batches
num_devices = 2
by_gpu_batches = smart_partition_batches(all_batches, num_devices)

# Count tokens after processing
total_train_tokens = 0
total_num_tokens = 0
for gpu_batches in by_gpu_batches.values():
    for batch in gpu_batches:
        total_train_tokens += batch["attention_mask"].sum().item()
        total_num_tokens += batch["input_ids"].numel()

print(
    f"After Total train tokens: {total_train_tokens}, "
    f"total num tokens: {total_num_tokens}, "
    f"train percentage: {total_train_tokens / total_num_tokens:.2%}"
)
