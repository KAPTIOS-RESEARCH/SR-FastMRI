#!/bin/bash

LOG_FILE="./output.log"
nohup python main.py --config_path ./tasks/SR/config.yaml > "$LOG_FILE" 2>&1 &
echo "Script started in the background"
echo "Logs are being saved to output.log"