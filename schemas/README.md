# JSON Schema for OpenSloth Configuration

This directory contains auto-generated JSON Schema files that provide IntelliSense support for OpenSloth configuration files in VS Code.

## Files

- `training_config.schema.json` - Complete schema for training configuration files
  - Generated from Pydantic models in `src/opensloth/opensloth_config.py`
  - Provides auto-completion and validation for training configs

## How It Works

1. **Pydantic Models** in `src/opensloth/opensloth_config.py` define the configuration structure
2. **Schema Generation** via `scripts/generate_schema.py` converts Pydantic models to JSON Schema
3. **VS Code Integration** via `.vscode/settings.json` associates schemas with config files
4. **IntelliSense** automatically provides suggestions when editing JSON config files

## Regenerating Schemas

When you modify the Pydantic configuration models, regenerate the schema:

```bash
# Python script
uv run scripts/generate_schema.py

# Fish script wrapper
./scripts/regenerate_schema.fish
```

## VS Code Features

With the schema setup, you get:

- **Auto-completion** for configuration keys
- **Type validation** (string, number, boolean, etc.)
- **Enum suggestions** for fields with limited values
- **Documentation** on hover for each configuration option
- **Error highlighting** for invalid values or structure

## Example Usage

Create a `training_config.json` file and VS Code will automatically provide:

```json
{
  "$schema": "./schemas/training_config.schema.json",
  "opensloth_config": {
    "data_cache_path": "...",  // ← Auto-completion appears here
    "fast_model_args": {
      "model_name": "...",     // ← Suggestions for model names
      "optim": "..."          // ← Enum values like "adamw_8bit"
    }
  }
}
```

## Files Using This Schema

- `**/training_config.json` - Generated training configurations
- `**/config.json` - General configuration files
- `**/*training*.json` - Any file with "training" in the name
- `**/*config*.json` - Any file with "config" in the name

The schema association is configured in `.vscode/settings.json`.