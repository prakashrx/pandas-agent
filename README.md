# Agents

Experiments with AI agents.

## Project Structure

```
.
├── data/                 # Dataset storage
│   └── countries of the world.csv
├── src/                  # Source code
│   ├── simple/           # Simple agents
├── .env.example          # Example environment variables
├── .python-version       # Python runtime version
├── pandas_agent.py       # A pandas agent for data exploration
└── pyproject.toml        # Python project metadata
```

## Setup

1. Create virtual environment:

```bash
uv venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install project dependencies
uv pip install -e .
```

## Environment Variables Setup

Copy `.env.example` to `.env.local` and set your OpenAI API key.
