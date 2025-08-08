import os

import torch

from src.opensloth._debug_dataloader import debug_chat_dataloader_for_training_markdown


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
    def decode(self, ids, skip_special_tokens=False):
        # Just join ints for demo
        return " ".join(str(i) for i in ids)

def make_dummy_dataloader(batch_size=1, seq_len=8, n_batches=3):
    for _ in range(n_batches):
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        # Alternate trainable/context: even idx trainable, odd context
        labels = torch.tensor([[i if i % 2 == 0 else -100 for i in range(seq_len)] for _ in range(batch_size)])
        yield {"input_ids": input_ids, "labels": labels}

def test_debug_markdown():
    dataloader = make_dummy_dataloader()
    tokenizer = DummyTokenizer()
    debug_chat_dataloader_for_training_markdown(dataloader, tokenizer, n_example=2)
    assert os.path.exists(".log/dataloader_examples.md"), "Markdown file not created!"
    with open(".log/dataloader_examples.md") as f:
        content = f.read()
        assert "<span style=\"color:green\"" in content, "Green span missing!"
        assert "---" in content, "Separator missing!"
    print("Test passed: Markdown debug output created and formatted.")

if __name__ == "__main__":
    test_debug_markdown()
