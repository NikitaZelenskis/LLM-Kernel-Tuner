import os
import sys

os.environ["SPHINX_BUILD"] = "1"
sys.path.insert(0, os.path.abspath("../.."))
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


project = 'llm_kernel_tuner'
copyright = '2025, Nikita Zelenskis'
author = 'Nikita Zelenskis'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
    "sphinxcontrib.spelling",
]

spelling_lang = 'en_US'
spelling_word_list_filename = ['spelling_wordlist.txt', 'spelling_namelist.txt']

# autodoc_mock_imports = [
#     "kernel_tuner",
#     "python-dotenv",
#     "langchain-core",
#     "langchain-openai",
#     "langgraph",
#     "libclang",
#     "pycuda",
#     "numpy",
# ]

autodoc_typehints = "both" 
autoclass_member_order = 'bysource'
# autodoc_class_signature = "separated"


bibtex_bibfiles = ['bibliography.bib']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


pygments_style = "sphinx"

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

html_context = {
    "display_github": True,  # Integrate GitHub
    "github_user": "NikitaZelenskis",  # Username
    "github_repo": "LLM-Kernel-Tuner",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/",  # Path in the checkout to the docs root
}

html_static_path = ['_static']

numfig = True

