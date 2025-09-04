#!/bin/bash

# Start parameter server first and let it initialize
# echo "Starting parameter server..."
# CUDA_VISIBLE_DEVICES=0 python ps.py &
# PS_PID=$!

# Wait a bit for PS to initialize and export IPC handle
# echo "Waiting for parameter server to initialize..."
# sleep 5

# # Check if PS is still running
# if ! kill -0 $PS_PID 2>/dev/null; then
#     echo "Parameter server failed to start!"
#     exit 1
# fi

echo "Starting workers..."
# Start workers with staggered delays to reduce contention
python worker.py -d 1 &
sleep 1
python worker.py -d 2 &
sleep 1  
python worker.py -d 3 &
# Wait for all background processes
wait