#!/usr/bin/env bash
# Test script to verify os-tmux-worker entry point

echo "Testing os-tmux-worker entry point..."

# Test if the command exists
if uv run os-tmux-worker --help > /dev/null 2>&1; then
    echo "✅ os-tmux-worker command is available"
else
    echo "❌ os-tmux-worker command not found"
    exit 1
fi

echo "✅ All tests passed!"