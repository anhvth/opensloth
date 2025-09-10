#!/usr/bin/env fish
# Regenerate JSON Schema for OpenSloth training configurations
# This ensures the schema stays in sync with the Pydantic models

echo "🔄 Regenerating OpenSloth JSON Schema..."

# Run the schema generation script
uv run scripts/generate_schema.py

if test $status -eq 0
    echo "✅ Schema regenerated successfully!"
    echo "📁 Updated: schemas/training_config.schema.json"
    echo "💡 VS Code IntelliSense is now up to date"
else
    echo "❌ Schema generation failed!"
    exit 1
end