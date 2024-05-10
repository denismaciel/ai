
setup-dev:
	uv venv --prompt venv
	uv pip install -r requirements.txt -r requirements-dev.txt --editable .

compile-dependencies:
	uv pip compile --output-file requirements.txt pyproject.toml
	uv pip compile --extra dev --output-file requirements-dev.txt pyproject.toml
