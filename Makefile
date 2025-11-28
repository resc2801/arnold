# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║     █████╗ ██████╗ ███╗   ██╗ ██████╗ ██╗     ██████╗                        ║
# ║    ██╔══██╗██╔══██╗████╗  ██║██╔═══██╗██║     ██╔══██╗                       ║
# ║    ███████║██████╔╝██╔██╗ ██║██║   ██║██║     ██║  ██║                       ║
# ║    ██╔══██║██╔══██╗██║╚██╗██║██║   ██║██║     ██║  ██║                       ║
# ║    ██║  ██║██║  ██║██║ ╚████║╚██████╔╝███████╗██████╔╝                       ║
# ║    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚══════╝╚═════╝                        ║
# ║                                                                              ║
# ║    Kolmogorov-Arnold Networks for Keras                                      ║
# ║    Development Makefile                                                      ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

.PHONY: help env dev install test lint format docs clean build publish

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Configuration                                                                │
# └──────────────────────────────────────────────────────────────────────────────┘

ENV_NAME     := arnold-dev
PYTHON_VER   := 3.11
CONDA        := conda
PYTHON       := $(CONDA) run -n $(ENV_NAME) --no-capture-output python
PIP          := $(CONDA) run -n $(ENV_NAME) --no-capture-output pip

# Colors & Formatting
BOLD         := \033[1m
DIM          := \033[2m
RESET        := \033[0m
RED          := \033[31m
GREEN        := \033[32m
YELLOW       := \033[33m
BLUE         := \033[34m
MAGENTA      := \033[35m
CYAN         := \033[36m
WHITE        := \033[37m

# Icons (requires Unicode-capable terminal)
ICON_OK      := ✓
ICON_FAIL    := ✗
ICON_ARROW   := ➜
ICON_PACKAGE := 📦
ICON_TEST    := 🧪
ICON_DOCS    := 📚
ICON_CLEAN   := 🧹
ICON_ROCKET  := 🚀
ICON_GEAR    := ⚙️
ICON_PYTHON  := 🐍
ICON_LINT    := 🔍
ICON_FORMAT  := ✨

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Help                                                                         │
# └──────────────────────────────────────────────────────────────────────────────┘

help:
	@echo ""
	@echo "$(BOLD)$(MAGENTA)╔══════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BOLD)$(MAGENTA)║$(RESET)  $(BOLD)ARNOLD$(RESET) — Development Commands                               $(BOLD)$(MAGENTA)║$(RESET)"
	@echo "$(BOLD)$(MAGENTA)╚══════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_PYTHON) Environment$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make env$(RESET)           Create conda environment ($(ENV_NAME))"
	@echo "  $(GREEN)make dev$(RESET)           Install in development mode"
	@echo "  $(GREEN)make install$(RESET)       Install production package"
	@echo "  $(GREEN)make env-export$(RESET)    Export environment.yml"
	@echo "  $(GREEN)make env-remove$(RESET)    Remove conda environment"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_TEST) Testing$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make test$(RESET)          Run all tests"
	@echo "  $(GREEN)make test-cov$(RESET)      Run tests with coverage report"
	@echo "  $(GREEN)make test-fast$(RESET)     Run fast tests only (skip slow)"
	@echo "  $(GREEN)make test-gpu$(RESET)      Run GPU-specific tests"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_LINT) Code Quality$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make lint$(RESET)          Check code style (ruff)"
	@echo "  $(GREEN)make format$(RESET)        Auto-format code"
	@echo "  $(GREEN)make typecheck$(RESET)     Run type checker (mypy)"
	@echo "  $(GREEN)make check$(RESET)         Run all quality checks"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_DOCS) Documentation$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make docs$(RESET)          Build Sphinx documentation"
	@echo "  $(GREEN)make docs-serve$(RESET)    Build and serve docs locally"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_PACKAGE) Build & Release$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make build$(RESET)         Build wheel and sdist"
	@echo "  $(GREEN)make build-check$(RESET)   Build and validate package"
	@echo "  $(GREEN)make publish-test$(RESET)  Upload to TestPyPI"
	@echo "  $(GREEN)make publish$(RESET)       Upload to PyPI $(DIM)(production!)$(RESET)"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_GEAR) Versioning$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make version$(RESET)       Show current version"
	@echo "  $(GREEN)make version-patch$(RESET) Bump patch (0.1.0 → 0.1.1)"
	@echo "  $(GREEN)make version-minor$(RESET) Bump minor (0.1.0 → 0.2.0)"
	@echo "  $(GREEN)make version-major$(RESET) Bump major (0.1.0 → 1.0.0)"
	@echo ""
	@echo "$(BOLD)$(CYAN)  $(ICON_CLEAN) Maintenance$(RESET)"
	@echo "  $(DIM)─────────────────────────────────────────────────────────$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)         Remove build artifacts"
	@echo "  $(GREEN)make clean-all$(RESET)     Deep clean (caches, coverage)"
	@echo "  $(GREEN)make update-deps$(RESET)   Update dependencies"
	@echo ""
	@echo "$(DIM)  Tip: Run $(RESET)$(YELLOW)make <target> -n$(RESET)$(DIM) to preview commands$(RESET)"
	@echo ""

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Environment Management                                                       │
# └──────────────────────────────────────────────────────────────────────────────┘

env:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_GEAR) Creating conda environment...$(RESET)"
	@echo ""
	@$(CONDA) create -n $(ENV_NAME) python=$(PYTHON_VER) -y
	@$(PIP) install --upgrade pip setuptools wheel
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Environment $(BOLD)$(ENV_NAME)$(RESET)$(GREEN) created successfully!$(RESET)"
	@echo ""
	@echo "$(YELLOW)$(ICON_ARROW) Next steps:$(RESET)"
	@echo "   $(DIM)1.$(RESET) conda activate $(ENV_NAME)"
	@echo "   $(DIM)2.$(RESET) make dev"
	@echo ""

dev:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_PACKAGE) Installing development dependencies...$(RESET)"
	@echo ""
	@$(PIP) install -e ".[dev,test,docs]"
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Development environment ready!$(RESET)"
	@echo ""

install:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_PACKAGE) Installing ARNOLD...$(RESET)"
	@echo ""
	@$(PIP) install .
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Installation complete!$(RESET)"
	@echo ""

env-export:
	@echo "$(BLUE)$(ICON_GEAR) Exporting environment...$(RESET)"
	@$(CONDA) env export -n $(ENV_NAME) > environment.yml
	@echo "$(GREEN)$(ICON_OK) Saved to environment.yml$(RESET)"

env-remove:
	@echo "$(YELLOW)$(ICON_CLEAN) Removing environment $(ENV_NAME)...$(RESET)"
	@$(CONDA) env remove -n $(ENV_NAME) -y
	@echo "$(GREEN)$(ICON_OK) Environment removed$(RESET)"

bench:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_GEAR) Running polynomial benchmarks...$(RESET)"
	@echo ""
	@$(PYTHON) benchmarks/poly_bench.py --degree 32 --repeats 20 --input-dim 16 --batch 256
	@echo ""

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Testing                                                                      │
# └──────────────────────────────────────────────────────────────────────────────┘

test:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_TEST) Running tests...$(RESET)"
	@echo ""
	@$(PYTHON) -m pytest tests/ -v --tb=short \
		&& echo "" && echo "$(GREEN)$(ICON_OK) All tests passed!$(RESET)" \
		|| (echo "" && echo "$(RED)$(ICON_FAIL) Tests failed$(RESET)" && exit 1)
	@echo ""

test-cov:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_TEST) Running tests with coverage...$(RESET)"
	@echo ""
	@$(PYTHON) -m pytest tests/ -v \
		--cov=arnold \
		--cov-report=term-missing \
		--cov-report=html:htmlcov \
		--cov-fail-under=50
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Coverage report: $(BOLD)htmlcov/index.html$(RESET)"
	@echo ""

test-fast:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_TEST) Running fast tests...$(RESET)"
	@echo ""
	@$(PYTHON) -m pytest tests/ -v -m "not slow" --tb=short
	@echo ""

test-gpu:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_TEST) Running GPU tests...$(RESET)"
	@echo ""
	@$(PYTHON) -m pytest tests/ -v -m "gpu" --tb=short
	@echo ""

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Code Quality                                                                 │
# └──────────────────────────────────────────────────────────────────────────────┘

lint:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_LINT) Checking code style...$(RESET)"
	@echo ""
	@$(PYTHON) -m ruff check src/ tests/ \
		&& echo "$(GREEN)$(ICON_OK) No issues found!$(RESET)" \
		|| (echo "$(RED)$(ICON_FAIL) Issues found$(RESET)" && exit 1)
	@echo ""

format:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_FORMAT) Formatting code...$(RESET)"
	@echo ""
	@$(PYTHON) -m ruff format src/ tests/
	@$(PYTHON) -m ruff check --fix src/ tests/ || true
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Code formatted!$(RESET)"
	@echo ""

typecheck:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_LINT) Running type checker...$(RESET)"
	@echo ""
	@$(PYTHON) -m mypy src/arnold --ignore-missing-imports
	@echo ""

check: lint typecheck
	@echo ""
	@echo "$(GREEN)$(ICON_OK) All quality checks passed!$(RESET)"
	@echo ""

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Documentation                                                                │
# └──────────────────────────────────────────────────────────────────────────────┘

docs:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_DOCS) Building documentation...$(RESET)"
	@echo ""
	@$(CONDA) run -n $(ENV_NAME) $(MAKE) -C docs html
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Docs built: $(BOLD)docs/_build/html/index.html$(RESET)"
	@echo ""

docs-serve: docs
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_DOCS) Serving documentation at $(CYAN)http://localhost:8000$(RESET)"
	@echo "$(DIM)    Press Ctrl+C to stop$(RESET)"
	@echo ""
	@$(PYTHON) -m http.server 8000 --directory docs/_build/html

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Build & Publish                                                              │
# └──────────────────────────────────────────────────────────────────────────────┘

build: clean
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_PACKAGE) Building package...$(RESET)"
	@echo ""
	@$(PYTHON) -m build
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Build complete!$(RESET)"
	@echo ""
	@ls -lh dist/
	@echo ""

build-check: build
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_LINT) Validating wheel contents...$(RESET)"
	@echo ""
	@$(PYTHON) -m check_wheel_contents dist/*.whl
	@$(PYTHON) -m twine check dist/*
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Package validation passed!$(RESET)"
	@echo ""

publish-test: build-check
	@echo ""
	@echo "$(BOLD)$(YELLOW)$(ICON_ROCKET) Publishing to TestPyPI...$(RESET)"
	@echo "$(DIM)    https://test.pypi.org/project/arnold-kan/$(RESET)"
	@echo ""
	@$(PYTHON) -m twine upload \
		--repository testpypi \
		--verbose \
		dist/*
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Published to TestPyPI!$(RESET)"
	@echo ""
	@echo "$(CYAN)Test installation with:$(RESET)"
	@echo "    pip install --index-url https://test.pypi.org/simple/ arnold-kan"
	@echo ""

publish: build-check
	@echo ""
	@echo "$(BOLD)$(RED)════════════════════════════════════════════════════════$(RESET)"
	@echo "$(BOLD)$(RED)  $(ICON_ROCKET) PUBLISHING TO PyPI (PRODUCTION)$(RESET)"
	@echo "$(BOLD)$(RED)════════════════════════════════════════════════════════$(RESET)"
	@echo ""
	@echo "$(YELLOW)  Package: arnold-kan$(RESET)"
	@echo "$(YELLOW)  Version: $(shell grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)$(RESET)"
	@echo ""
	@read -p "  $(BOLD)Continue? [y/N]$(RESET) " confirm && [ "$$confirm" = "y" ]
	@echo ""
	@$(PYTHON) -m twine upload \
		--verbose \
		dist/*
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Published to PyPI!$(RESET)"
	@echo ""
	@echo "$(CYAN)Install with:$(RESET)"
	@echo "    pip install arnold-kan"
	@echo ""

# ────────────────────────────────────────────────────────────────────────────────
# Version Management
# ────────────────────────────────────────────────────────────────────────────────

.PHONY: version version-patch version-minor version-major

version:
	@echo ""
	@echo "$(BOLD)$(CYAN)Current version:$(RESET) $(shell grep 'version = ' pyproject.toml | head -1 | cut -d'"' -f2)"
	@echo ""

version-patch:
	@echo "$(BLUE)$(ICON_GEAR) Bumping patch version...$(RESET)"
	@$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		new_version = f'{version.group(1)}.{version.group(2)}.{int(version.group(3))+1}'; \
		content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(content); \
		print(f'Version bumped to {new_version}')"

version-minor:
	@echo "$(BLUE)$(ICON_GEAR) Bumping minor version...$(RESET)"
	@$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		new_version = f'{version.group(1)}.{int(version.group(2))+1}.0'; \
		content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(content); \
		print(f'Version bumped to {new_version}')"

version-major:
	@echo "$(BLUE)$(ICON_GEAR) Bumping major version...$(RESET)"
	@$(PYTHON) -c "import re; \
		content = open('pyproject.toml').read(); \
		version = re.search(r'version = \"(\d+)\.(\d+)\.(\d+)\"', content); \
		new_version = f'{int(version.group(1))+1}.0.0'; \
		content = re.sub(r'version = \"\d+\.\d+\.\d+\"', f'version = \"{new_version}\"', content); \
		open('pyproject.toml', 'w').write(content); \
		print(f'Version bumped to {new_version}')"

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Maintenance                                                                  │
# └──────────────────────────────────────────────────────────────────────────────┘

clean:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_CLEAN) Cleaning build artifacts...$(RESET)"
	@rm -rf dist/ build/ *.egg-info src/*.egg-info
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)$(ICON_OK) Clean!$(RESET)"
	@echo ""

clean-all: clean
	@echo "$(BOLD)$(BLUE)$(ICON_CLEAN) Deep cleaning...$(RESET)"
	@rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	@echo "$(GREEN)$(ICON_OK) All caches removed!$(RESET)"
	@echo ""

update-deps:
	@echo ""
	@echo "$(BOLD)$(BLUE)$(ICON_GEAR) Updating dependencies...$(RESET)"
	@echo ""
	@$(PIP) install --upgrade pip setuptools wheel
	@$(PIP) install -e ".[dev,test,docs]" --upgrade
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Dependencies updated!$(RESET)"
	@echo ""

# ┌──────────────────────────────────────────────────────────────────────────────┐
# │ Shortcuts                                                                    │
# └──────────────────────────────────────────────────────────────────────────────┘

# Quick development cycle
.PHONY: q quick
q quick: format lint test-fast
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Quick check complete!$(RESET)"
	@echo ""

# Pre-commit hook equivalent  
.PHONY: pre-commit
pre-commit: format lint typecheck test
	@echo ""
	@echo "$(GREEN)$(ICON_OK) Ready to commit!$(RESET)"
	@echo ""

# Full CI simulation
.PHONY: ci
ci: clean lint typecheck test-cov docs build
	@echo ""
	@echo "$(GREEN)$(BOLD)══════════════════════════════════════════════════════════$(RESET)"
	@echo "$(GREEN)$(BOLD)  $(ICON_OK) CI Pipeline Complete!$(RESET)"
	@echo "$(GREEN)$(BOLD)══════════════════════════════════════════════════════════$(RESET)"
	@echo ""
