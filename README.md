# Introduction

This repo is regarding time series in LLM.

We will begin by experimenting with the following three approaches using *numeric and univariate* time series data.
- prompts only
- function calling
- code generation

## Setup

1. Install required packages: `pip install -e .`
2. [Optional] Install dev packages (pre-commit hook and ruff): `pip install -e .[dev]`


## Folder structure

```bash
├── data
├── docs # for all documents like markdown files, experimentation results
├── notebooks # for jupyter notebooks
├── src # for .py code
├── pyproject.toml # for required python packages and tool settings
├── .pre-commit-config.yaml # configuration for pre-commit hook
├── .gitignore
└── etc.
```
