"""Deprecated placeholder for removed batch sampling/packing patch.

This module formerly implemented custom per-rank batch slicing and sequence
packing. After introducing pre-sharded datasets (one shard per GPU), runtime
batch slicing became unnecessary and standard Trainer batching is used.

Any import of this module should be treated as legacy and updated to rely on
pre-sharded dataset loading instead.
"""

def __getattr__(name):  # pragma: no cover
    raise AttributeError(
        "get_batch_samples deprecated: dataset pre-sharding removed need for this patch."
    )
