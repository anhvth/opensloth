#!/usr/bin/env python3
"""
TensorBoard visualization script for GRPO training metrics.
Loads result files from worker/done/, extracts metrics, and creates TensorBoard logs.

Usage:
    python show_board.py ./outputs/tmp_vis_dir/
"""

import os
import sys
from typing import Dict, Any, Tuple

from utils.atomic_file_ops import AtomicFileOps

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Error: tensorboard not found. Install with: pip install tensorboard")
    sys.exit(1)


def parse_filename_timestamp(filename: str) -> float:
    """Extract timestamp from filename like '000123_000_collected_consumed_000123_000'"""
    try:
        # Extract the timestamp part - try the step number first (6 digits)
        basename = os.path.basename(filename)
        parts = basename.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            return float(parts[0])  # Use step number as timestamp for ordering
        
        # Fallback: look for timestamp-like numbers
        for part in parts:
            if part.isdigit() and len(part) >= 8:  # Looks like a timestamp
                return float(part)
        
        # Final fallback to file modification time
        return os.path.getmtime(filename)
    except (ValueError, IndexError):
        # Fallback to file modification time
        return os.path.getmtime(filename)


def load_result_file(filepath: str) -> Tuple[Dict, str]:
    """Load a result file and return (data, job_id)"""
    try:
        data = AtomicFileOps.read_torch(filepath, map_location='cpu', weights_only=False)
        if data is None:
            return None, None
        # Extract job_id from filename or data - new format: {step}_{job_id}_collected_consumed_{step}_{job_id}
        basename = os.path.basename(filepath)
        parts = basename.split('_')
        if len(parts) >= 2 and parts[0].isdigit():
            job_id = f"{parts[0]}_{parts[1]}"  # step_index format
        else:
            job_id = data.get('job_id', 'unknown')
        return data, job_id
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def extract_metrics(data: Dict) -> Dict[str, Any]:
    """Extract relevant metrics from result data"""
    metrics = {}
    
    # Basic job info
    metrics['job_id'] = data.get('job_id', 'unknown')
    metrics['processed_at_step'] = data.get('processed_at_step', 0)
    metrics['worker_device'] = data.get('worker_device', -1)
    metrics['job_created_step'] = data.get('job_created_step', 0)
    
    # Timing metrics
    metrics['job_age'] = data.get('job_age', 0.0)
    metrics['queue_time'] = data.get('queue_time', 0.0)
    metrics['processing_time'] = data.get('processing_time', 0.0)
    
    # Training metrics from nested structure
    if 'metrics' in data and 'train' in data['metrics']:
        train_metrics = data['metrics']['train']
        
        # Flatten the metrics - each value is a list with single element
        for key, value_list in train_metrics.items():
            if value_list and len(value_list) > 0:
                metrics[f'train/{key}'] = value_list[0]
    
    # Result-level metrics
    if 'result' in data:
        result = data['result']
        if 'advantages' in result:
            advantages = result['advantages']
            if hasattr(advantages, 'cpu'):
                advantages = advantages.cpu()
            metrics['advantages/mean'] = float(advantages.mean())
            metrics['advantages/std'] = float(advantages.std())
            metrics['advantages/min'] = float(advantages.min())
            metrics['advantages/max'] = float(advantages.max())
        
        # Token counts
        if 'prompt_ids' in result:
            prompt_ids = result['prompt_ids']
            if hasattr(prompt_ids, 'shape'):
                metrics['prompt_ids/batch_size'] = prompt_ids.shape[0]
                metrics['prompt_ids/seq_length'] = prompt_ids.shape[1] if len(prompt_ids.shape) > 1 else 0
        
        if 'completion_ids' in result:
            completion_ids = result['completion_ids']
            if hasattr(completion_ids, 'shape'):
                metrics['completion_ids/batch_size'] = completion_ids.shape[0]
                metrics['completion_ids/seq_length'] = completion_ids.shape[1] if len(completion_ids.shape) > 1 else 0
    
    return metrics


def write_metrics_to_tensorboard(writer: SummaryWriter, metrics: Dict[str, Any], step: int, timestamp: float):
    """Write metrics to TensorBoard"""
    
    # Organize metrics by category
    categories = {
        'timing': ['job_age', 'queue_time', 'processing_time'],
        'advantages': ['advantages/mean', 'advantages/std', 'advantages/min', 'advantages/max'],
        'tokens': ['prompt_ids/batch_size', 'prompt_ids/seq_length', 'completion_ids/batch_size', 'completion_ids/seq_length'],
        'job_info': ['worker_device', 'job_created_step', 'processed_at_step'],
    }
    
    # Write basic metrics
    for category, metric_names in categories.items():
        for metric_name in metric_names:
            if metric_name in metrics:
                value = metrics[metric_name]
                if isinstance(value, (int, float)) and not (isinstance(value, bool)):
                    writer.add_scalar(f'{category}/{metric_name.split("/")[-1]}', value, step)
    
    # Write training metrics
    for key, value in metrics.items():
        if key.startswith('train/') and isinstance(value, (int, float)) and not isinstance(value, bool):
            # Clean up the key for better visualization
            clean_key = key.replace('train/', '').replace('/', '_')
            writer.add_scalar(f'training/{clean_key}', value, step)
    
    # Write timestamp as a reference
    writer.add_scalar('system/timestamp', timestamp, step)


def main():
    if len(sys.argv) != 2:
        print("Usage: python show_board.py <output_tensorboard_dir>")
        print("Example: python show_board.py ./outputs/tmp_vis_dir/")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    # Find all result files
    all_files = AtomicFileOps.list_files("worker/done/", "*_collected_consumed_*.pt", sort_by_mtime=True)
    # Filter out lock files
    result_files = [f for f in all_files if not f.endswith('.lock')]
    
    if not result_files:
        print("No result files found in worker/done/")
        print("Looking for files matching pattern: worker/done/*_collected_consumed_*.pt (excluding .lock files)")
        sys.exit(1)
    
    print(f"Found {len(result_files)} result files")
    
    # Sort by timestamp (extracted from filename)
    result_files.sort(key=parse_filename_timestamp)
    
    # Create TensorBoard writer
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(output_dir)
    
    print(f"Processing files and writing to TensorBoard: {output_dir}")
    
    processed_count = 0
    failed_count = 0
    
    for i, filepath in enumerate(result_files):
        data, job_id = load_result_file(filepath)
        
        if data is None:
            failed_count += 1
            continue
        
        # Extract metrics
        metrics = extract_metrics(data)
        
        # Use file timestamp for step ordering
        timestamp = parse_filename_timestamp(filepath)
        
        # Use processed_at_step if available, otherwise use index
        step = metrics.get('processed_at_step', i)
        
        # Write to TensorBoard
        write_metrics_to_tensorboard(writer, metrics, step, timestamp)
        
        processed_count += 1
        
        if processed_count % 10 == 0:
            print(f"Processed {processed_count}/{len(result_files)} files...")
    
    writer.close()
    
    print("\nCompleted!")
    print(f"Successfully processed: {processed_count} files")
    print(f"Failed to process: {failed_count} files")
    print(f"TensorBoard logs written to: {output_dir}")
    print("\nTo view TensorBoard:")
    print(f"tensorboard --logdir {output_dir}")
    print("Then open http://localhost:6006 in your browser")
    
    # Print some sample metrics from the last file for verification
    if processed_count > 0:
        print(f"\nSample metrics from last file ({result_files[-1]}):")
        last_data, _ = load_result_file(result_files[-1])
        if last_data:
            last_metrics = extract_metrics(last_data)
            for key, value in list(last_metrics.items())[:10]:  # Show first 10 metrics
                print(f"  {key}: {value}")
            if len(last_metrics) > 10:
                print(f"  ... and {len(last_metrics) - 10} more metrics")


if __name__ == "__main__":
    main()