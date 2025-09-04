#!/usr/bin/env python3
"""
Performance monitoring and metrics collection for async-grpo system.
"""
import time
import threading
from collections import defaultdict, deque
from typing import Dict
from utils.atomic_file_ops import AtomicFileOps

class PerformanceMonitor:
    """Monitor and collect performance metrics for the async-grpo system."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.metrics = defaultdict(deque)
        self.counters = defaultdict(int)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
    def record_job_processing(self, job_id: str, processing_time: float, 
                            job_age: float, queue_time: float, worker_device: int):
        """Record job processing metrics."""
        with self.lock:
            self.metrics['processing_times'].append(processing_time)
            self.metrics['job_ages'].append(job_age) 
            self.metrics['queue_times'].append(queue_time)
            self.counters[f'jobs_processed_device_{worker_device}'] += 1
            self.counters['total_jobs_processed'] += 1
            
            # Keep only recent data
            for key in self.metrics:
                if len(self.metrics[key]) > self.window_size:
                    self.metrics[key].popleft()
    
    def record_weight_sync(self, sync_time: float, worker_device: int):
        """Record weight synchronization metrics."""
        with self.lock:
            self.metrics['weight_sync_times'].append(sync_time)
            self.counters[f'weight_syncs_device_{worker_device}'] += 1
            self.counters['total_weight_syncs'] += 1
            
            if len(self.metrics['weight_sync_times']) > self.window_size:
                self.metrics['weight_sync_times'].popleft()
    
    def get_stats(self) -> Dict:
        """Get current performance statistics."""
        with self.lock:
            stats = {
                'uptime': time.time() - self.start_time,
                'counters': dict(self.counters),
                'averages': {}
            }
            
            for key, values in self.metrics.items():
                if values:
                    stats['averages'][key] = {
                        'mean': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'count': len(values)
                    }
            
            # Calculate throughput
            if stats['uptime'] > 0:
                stats['throughput'] = {
                    'jobs_per_second': self.counters['total_jobs_processed'] / stats['uptime'],
                    'syncs_per_second': self.counters['total_weight_syncs'] / stats['uptime']
                }
            
            return stats
    
    def save_stats(self, filepath: str):
        """Save current stats to file."""
        stats = self.get_stats()
        AtomicFileOps.write_json(filepath, stats)
    
    def print_summary(self):
        """Print a summary of current performance."""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"Performance Summary (uptime: {stats['uptime']:.1f}s)")
        print(f"{'='*50}")
        
        if 'total_jobs_processed' in stats['counters']:
            print(f"Jobs processed: {stats['counters']['total_jobs_processed']}")
            
        if 'throughput' in stats:
            print(f"Throughput: {stats['throughput']['jobs_per_second']:.2f} jobs/s")
            
        if 'processing_times' in stats['averages']:
            pt = stats['averages']['processing_times']
            print(f"Processing time: {pt['mean']:.2f}s (min: {pt['min']:.2f}s, max: {pt['max']:.2f}s)")
            
        if 'job_ages' in stats['averages']:
            ja = stats['averages']['job_ages'] 
            print(f"Job age: {ja['mean']:.2f}s (min: {ja['min']:.2f}s, max: {ja['max']:.2f}s)")
            
        if 'weight_sync_times' in stats['averages']:
            ws = stats['averages']['weight_sync_times']
            print(f"Weight sync: {ws['mean']:.3f}s (min: {ws['min']:.3f}s, max: {ws['max']:.3f}s)")

# Global monitor instance
_monitor = None

def get_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PerformanceMonitor()
    return _monitor

def record_job_metrics(job_id: str, processing_time: float, job_age: float, 
                      queue_time: float, worker_device: int):
    """Convenience function to record job metrics."""
    get_monitor().record_job_processing(job_id, processing_time, job_age, queue_time, worker_device)

def record_sync_metrics(sync_time: float, worker_device: int):
    """Convenience function to record sync metrics.""" 
    get_monitor().record_weight_sync(sync_time, worker_device)

def print_performance_summary():
    """Convenience function to print performance summary."""
    get_monitor().print_summary()