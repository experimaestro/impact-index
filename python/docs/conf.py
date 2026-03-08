# Configuration file for the Sphinx documentation builder.

project = "impact-index"
copyright = "2024, Benjamin Piwowarski"
author = "Benjamin Piwowarski"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "autoapi.extension",
    "sphinx_codeautolink",
]

# -- AutoAPI configuration (reads .pyi stubs without importing the module) --
autoapi_type = "python"
autoapi_dirs = [".."]  # parent (python/), where impact_index.pyi lives
autoapi_file_patterns = ["*.pyi"]
autoapi_generate_api_docs = True
autoapi_options = [
    "members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_python_class_content = "both"  # show both class docstring and __init__
autoapi_member_order = "groupwise"

# Napoleon settings (Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True

# Intersphinx (cross-reference numpy, python stdlib)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- General configuration --
templates_path = ["_templates"]
exclude_patterns = ["_build"]

# -- HTML output --
html_theme = "furo"
html_title = "impact-index"
html_static_path = []
