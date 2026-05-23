.PHONY: lint format check fix

lint:
	venv/bin/ruff check src/

format:
	venv/bin/ruff format src/

fix:
	venv/bin/ruff check src/ --fix
	venv/bin/ruff format src/

check: lint
	venv/bin/python -c "from knowledge_mcp.server import app; print(\"import OK\")"
