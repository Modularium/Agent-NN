#!/usr/bin/env bash
set -e
# System packages
sudo apt-get update
sudo apt-get install -y python3.11 python3.11-venv python3.11-distutils curl docker.io docker-compose nodejs npm
# Poetry
curl -sSL https://install.python-poetry.org | python3 - --version 1.8.2
# Node tools
sudo npm install -g pnpm
# Python dev tools
python3.11 -m pip install --upgrade pip
python3.11 -m pip install poetry ruff black pytest httpx fastapi uvicorn mypy bandit tox coverage pydantic slowapi prometheus-client aiofiles loguru python-json-logger aiohttp structlog
# Frontend testing tools
sudo npm install -g playwright jest
# Optional diagram tool
sudo npm install -g @mermaid-js/mermaid-cli
