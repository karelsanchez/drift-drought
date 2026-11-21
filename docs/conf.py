import os
import sys

sys.path.insert(0, os.path.abspath("../src"))
project = "DRIFT"
author = "Karel A. Sanchez H."
copyright = "2026, " + author
release = "0.1.1"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.mathjax", "sphinx.ext.viewcode",
              "sphinx.ext.autosummary"]
autodoc_member_order = "bysource"
autodoc_mock_imports = ["higra"]
napoleon_numpy_docstring = True
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "DRIFT drought events"
