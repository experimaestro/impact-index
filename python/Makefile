SPHINX_DEPS = --with sphinx --with sphinx-autoapi --with sphinx-codeautolink --with furo

.PHONY: docs docs-clean

docs:
	uv run $(SPHINX_DEPS) sphinx-build -b html docs docs/_build

docs-clean:
	rm -rf docs/_build
