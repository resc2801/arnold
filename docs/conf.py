## Copyright (c) 2025 René Schubotz. All rights reserved.
# Licensed under the terms specified in the LICENSE file in the project root.
import sys, os
sys.path.insert(0, os.path.abspath('../src'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ARNOLD'
copyright = '2025, René Schubotz'
author = 'René Schubotz'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc.typehints",
    "sphinx.ext.mathjax",
    "sphinx_rtd_theme",
    # "sphinx_proof",  # Temporarily disabled due to compatibility issues
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
typehints_fully_qualified = True

# Suppress warnings for unreferenced citations in docstrings (informational references)
suppress_warnings = ["ref.citation"]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- MathJax Configuration ---------------------------------------------------
# Use MathJax 3 for rendering LaTeX math in docstrings
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    },
}
