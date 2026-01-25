---
tags:
    - info
hide:
    - footer
---

# Installation

`SymbolicDSGE` can be installed via PyPI using the following command for use in a python environment.

```bash
pip install SymbolicDSGE
```

For people that want to modify or contribute to the project, forking the [repository](https://github.com/GongJr0/SymbolicDSGE) is advisable compared to a `pip install`. In the repository, you will find a `uv.lock` file. To interact with all components of the project using `uv sync --all-extras` is the quickest and easiest route. This will replicate the environment used at development and will include the `dev` dependency group for linting, auto formatting, etc.
