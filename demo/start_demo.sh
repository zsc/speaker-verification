#!/bin/bash
echo "Installing dependencies..."
pip install -r demo/requirements.txt

echo "Starting Demo Server..."
python demo/server.py
