#!/usr/bin/env python3
"""
Generate JSON Schema for Hugging Face TrainingArguments
so VS Code (or other editors) can provide IntelliSense + validation.
"""

import json
from transformers import TrainingArguments
from pydantic.dataclasses import dataclass
from pydantic import TypeAdapter


# Wrap Hugging Face TrainingArguments with Pydantic
@dataclass
class TrainingArgsSchema(TrainingArguments):
    pass


def main():
    # Export JSON Schema using TypeAdapter
    adapter = TypeAdapter(TrainingArgsSchema)
    schema = adapter.json_schema()

    out_file = "training_args.schema.json"
    with open(out_file, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"✅ JSON Schema written to {out_file}")


if __name__ == "__main__":
    main()
