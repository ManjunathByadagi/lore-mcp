# Contributing to Lore

Thanks for your interest in contributing.

## Development Setup

```bash
git clone https://github.com/davidgut1982/lore-mcp.git
cd lore-mcp
python -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Running Tests

```bash
make check         # lint + import check
pytest tests/ -v   # full test suite
```

## Code Style

Lore uses [Ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
make lint    # check
make format  # fix
```

## Submitting Changes

1. Fork the repo and create a feature branch
2. Make your changes with tests
3. Run `make check` — must pass clean
4. Submit a pull request with a clear description

## Reporting Issues

Use [GitHub Issues](https://github.com/davidgut1982/lore-mcp/issues). Include:
- What you expected vs what happened
- Your backend (SQLite / PostgreSQL / Supabase)
- Python version and OS
