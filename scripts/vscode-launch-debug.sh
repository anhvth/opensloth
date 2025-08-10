#!/bin/bash

# VSCode Debug Launch Script for OpenSloth
# Usage: ./scripts/vscode-launch-debug.sh <original_command>
# Example: ./scripts/vscode-launch-debug.sh "os-grpo data/grpo_dapo_prepared --model outputs/sft_model --tmux"

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default debug settings
DEFAULT_DEBUG_PORT_START=5678
DEFAULT_WAIT_FOR_CLIENT="True"
TEMP_DIR="/tmp/opensloth_debug_$(date +%s)_$$"
LAUNCH_JSON_PATH=".vscode/launch.json"

echo -e "${BLUE}🐛 OpenSloth VSCode Debug Launcher${NC}"
echo "=================================="

# Check if command is provided
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: No command provided${NC}"
    echo ""
    echo "Usage: $0 <original_command>"
    echo ""
    echo "Examples:"
    echo "  $0 'os-grpo data/grpo_dapo_prepared --model outputs/sft_model'"
    echo "  $0 'os-grpo data/grpo_dapo_prepared --model outputs/sft_model --tmux'"
    echo "  $0 'os-sft data/sft_prepared --model unsloth/Qwen2.5-7B-Instruct'"
    exit 1
fi

ORIGINAL_CMD="$1"
echo -e "${GREEN}📝 Original command:${NC} $ORIGINAL_CMD"

# Parse the original command to extract key information
CMD_PARTS=($ORIGINAL_CMD)
CLI_SCRIPT="${CMD_PARTS[0]}"
REMAINING_ARGS="${CMD_PARTS[@]:1}"

# Check if it's a known OpenSloth CLI command
case "$CLI_SCRIPT" in
    "os-grpo"|"os-sft"|"os-dpo")
        echo -e "${GREEN}✅ Detected OpenSloth CLI command: $CLI_SCRIPT${NC}"
        ;;
    *)
        echo -e "${YELLOW}⚠️  Warning: Unknown command '$CLI_SCRIPT'. Proceeding anyway...${NC}"
        ;;
esac

# Check if tmux mode is requested
USE_TMUX=false
if [[ "$ORIGINAL_CMD" == *"--tmux"* ]]; then
    USE_TMUX=true
    echo -e "${BLUE}🖥️  Tmux mode detected - will create multi-process debug configuration${NC}"
fi

# Create temporary directory for debug configs
mkdir -p "$TEMP_DIR"
echo -e "${BLUE}📁 Using temp directory: $TEMP_DIR${NC}"

# Function to generate a debug config file for a single process
generate_debug_config() {
    local rank=$1
    local world_size=$2
    local port=$3
    local config_file="$TEMP_DIR/debug_config_rank_${rank}.py"
    
    # Create a temporary config that simulates the original command but with debug setup
    cat > "$config_file" << EOF
#!/usr/bin/env python3
"""
Auto-generated debug configuration for rank $rank
Original command: $ORIGINAL_CMD
"""

import os
import sys
import debugpy

# Configure debugpy
debugpy.listen(('localhost', $port))
print(f"🐛 [Rank $rank] Debugpy listening on port $port")

if $DEFAULT_WAIT_FOR_CLIENT:
    print(f"🐛 [Rank $rank] Waiting for debugger to attach...")
    debugpy.wait_for_client()
    print(f"🐛 [Rank $rank] Debugger attached!")

# Set environment variables for this rank
os.environ['OPENSLOTH_LOCAL_RANK'] = '$rank'
os.environ['OPENSLOTH_WORLD_SIZE'] = '$world_size'
os.environ['USE_TMUX'] = '0'  # Disable tmux since we're debugging

# Import and run the original CLI
try:
    if '$CLI_SCRIPT' == 'os-grpo':
        from opensloth.cli.os_grpo import app
        # Parse args and run
        import sys
        sys.argv = ['$CLI_SCRIPT'] + '''$REMAINING_ARGS'''.split()
        # Remove --tmux if present since we're handling multi-process differently
        sys.argv = [arg for arg in sys.argv if arg != '--tmux']
        app()
    elif '$CLI_SCRIPT' == 'os-sft':
        from opensloth.cli.os_sft import app
        import sys
        sys.argv = ['$CLI_SCRIPT'] + '''$REMAINING_ARGS'''.split()
        sys.argv = [arg for arg in sys.argv if arg != '--tmux']
        app()
    elif '$CLI_SCRIPT' == 'os-dpo':
        from opensloth.cli.os_dpo import app
        import sys
        sys.argv = ['$CLI_SCRIPT'] + '''$REMAINING_ARGS'''.split()
        sys.argv = [arg for arg in sys.argv if arg != '--tmux']
        app()
    else:
        print(f"❌ Unknown CLI script: $CLI_SCRIPT")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ [Rank $rank] Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
    
    chmod +x "$config_file"
    echo "$config_file"
}

# Function to detect number of GPUs to use
detect_gpu_count() {
    # Try to parse from --devices argument first
    if [[ "$ORIGINAL_CMD" == *"--devices"* ]]; then
        local devices_arg=$(echo "$ORIGINAL_CMD" | grep -o -- '--devices [^[:space:]]*' | cut -d' ' -f2)
        if [[ -n "$devices_arg" ]]; then
            local gpu_count=$(echo "$devices_arg" | tr ',' '\n' | wc -l)
            echo "$gpu_count"
            return
        fi
    fi
    
    # Default to 2 GPUs for multi-process debugging
    if [ "$USE_TMUX" = true ]; then
        echo "2"
    else
        echo "1"
    fi
}

GPU_COUNT=$(detect_gpu_count)
echo -e "${BLUE}🖥️  Will create debug configs for $GPU_COUNT GPU(s)${NC}"

# Generate debug configurations and launch.json entries
LAUNCH_CONFIGS=()
DEBUG_CONFIG_FILES=()

for ((rank=0; rank<GPU_COUNT; rank++)); do
    port=$((DEFAULT_DEBUG_PORT_START + rank))
    config_file=$(generate_debug_config $rank $GPU_COUNT $port)
    DEBUG_CONFIG_FILES+=("$config_file")
    
    # Create launch.json configuration
    config_name="OpenSloth Debug (rank $rank)"
    LAUNCH_CONFIGS+=("\"$config_name\"")
    
    echo -e "${GREEN}✅ Generated debug config for rank $rank (port $port)${NC}"
done

# Update launch.json
echo -e "${BLUE}📝 Updating $LAUNCH_JSON_PATH...${NC}"

# Backup existing launch.json
if [ -f "$LAUNCH_JSON_PATH" ]; then
    cp "$LAUNCH_JSON_PATH" "$LAUNCH_JSON_PATH.backup.$(date +%s)"
    echo -e "${YELLOW}💾 Backed up existing launch.json${NC}"
fi

# Create .vscode directory if it doesn't exist
mkdir -p .vscode

# Generate new launch.json with Python helper to avoid shell escaping issues
python3 << EOF
import json
import os

gpu_count = $GPU_COUNT
debug_config_files = [$(printf '"%s",' "${DEBUG_CONFIG_FILES[@]}" | sed 's/,$//')]
default_debug_port_start = $DEFAULT_DEBUG_PORT_START

# Create configurations
configurations = []
compound_configs = []

for rank in range(gpu_count):
    port = default_debug_port_start + rank
    config_file = debug_config_files[rank]
    config_name = f"OpenSloth Debug (rank {rank})"
    compound_configs.append(config_name)
    
    config = {
        "name": config_name,
        "type": "debugpy", 
        "request": "launch",
        "program": config_file,
        "console": "integratedTerminal",
        "justMyCode": False,
        "cwd": "\${workspaceFolder}",
        "env": {
            "PYTHONPATH": "\${workspaceFolder}",
            "OPENSLOTH_LOCAL_RANK": str(rank),
            "OPENSLOTH_WORLD_SIZE": str(gpu_count),
            "USE_TMUX": "0"
        },
        "args": []
    }
    configurations.append(config)

# Create launch.json structure
launch_json = {
    "version": "0.2.0",
    "configurations": configurations
}

# Add compounds for multi-GPU
if gpu_count > 1:
    launch_json["compounds"] = [{
        "name": "OpenSloth Multi-GPU Debug (All Ranks)",
        "configurations": compound_configs,
        "stopAll": True,
        "presentation": {
            "hidden": False,
            "group": "opensloth-debug", 
            "order": 1
        }
    }]

# Write launch.json
with open("$LAUNCH_JSON_PATH", "w") as f:
    json.dump(launch_json, f, indent=4)

print("✅ Generated launch.json successfully")
EOF

echo -e "${GREEN}✅ Updated $LAUNCH_JSON_PATH with debug configurations${NC}"

# Print instructions
echo ""
echo -e "${GREEN}🎉 Debug setup complete!${NC}"
echo "========================"
echo ""
echo -e "${BLUE}📋 Next steps:${NC}"
echo ""
if [ $GPU_COUNT -gt 1 ]; then
    echo "1. Open VSCode: code ."
    echo "2. Go to Run and Debug (Ctrl+Shift+D)"
    echo "3. Select 'OpenSloth Multi-GPU Debug (All Ranks)' from the dropdown"
    echo "4. Press F5 or click the green play button"
    echo ""
    echo -e "${YELLOW}💡 This will start all $GPU_COUNT processes simultaneously${NC}"
else
    echo "1. Open VSCode: code ."
    echo "2. Go to Run and Debug (Ctrl+Shift+D)"
    echo "3. Select 'OpenSloth Debug (rank 0)' from the dropdown"
    echo "4. Press F5 or click the green play button"
fi
echo ""
echo -e "${BLUE}🐛 Debug Info:${NC}"
for ((rank=0; rank<GPU_COUNT; rank++)); do
    port=$((DEFAULT_DEBUG_PORT_START + rank))
    echo "  - Rank $rank: localhost:$port"
done
echo ""
echo -e "${BLUE}📁 Debug files created in: $TEMP_DIR${NC}"
echo ""
echo -e "${YELLOW}🧹 Cleanup:${NC}"
echo "  Run: rm -rf $TEMP_DIR"
echo "  To restore original launch.json: mv $LAUNCH_JSON_PATH.backup.* $LAUNCH_JSON_PATH"
echo ""
echo -e "${GREEN}Happy debugging! 🐛✨${NC}"
