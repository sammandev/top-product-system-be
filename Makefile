.PHONY: dev start lint format check test install migrate upgrade downgrade migrate-history

install:
	uv pip install -e .

dev:
	uv run python -m fastapi dev src/app/main.py --port 8001

start:
	uv run python -m uvicorn app.main:app --host 0.0.0.0 --port 8001

lint:
	uv run ruff check .

format:
	uv run ruff format .

check:
	uv run ruff check . && uv run ruff format --check .

fix:
	uv run ruff check --fix .

# Database migration commands
migrate:
	@if [ -z "$(msg)" ]; then \
		echo "Error: msg parameter is required. Usage: make migrate msg=\"Migration Success\""; \
		exit 1; \
	fi
	uv run python -m alembic revision --autogenerate -m "$(msg)"

upgrade:
	uv run python -m alembic upgrade head

downgrade:
	uv run python -m alembic downgrade -1

migrate-history:
	uv run python -m alembic history --verbose

migrate-current:
	uv run python -m alembic current

benchmark:
	uv run python -m scripts\benchmark_dut.py --bearer-token "<access_token>" --iterations 2 --scenario-file temp\benchmark_scenarios.json  

test:
	uv run python -m pytest -vv

test-cov:
	uv run python -m pytest --cov=app --cov-report=html tests/