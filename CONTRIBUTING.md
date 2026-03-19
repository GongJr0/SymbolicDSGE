# Contributing to SymbolicDSGE

> __Everybody is welcome!__
>
> Any contribution to the project is gladly accepted. Whether it's a bug report, a feature request, a code change, or just a suggestion for the direction of the project, all input is valuable and appreciated.

On this page, we will outline the contribution process and go through what a contribution can look like in several different forms. In short, the contribution process usually follows these steps:

1. Open an issue (or find an existing one)
2. Branch from `main`
3. Set up your environment
4. Install pre-commit hooks
5. Make your changes and add tests if necessary
6. Run tests locally
7. Write documentation if necessary
8. Open a pull request and link the issue

## Reporting a bug

Bug report submissions end up as issues in the [GitHub repository](https://github.com/GongJr0/SymbolicDSGE).
You can use the "Bug Report" issue template within the repository, or use the bug report card in the [Contact](./contact.md) page to redirect you to the issue template.

Even if you're unsure whether you found a bug in the project, it's always better to report it so we can investigate.

Familiarity with the project code is not essential to filling an informative bug report, but at the very least it's very helpful to include the exact method call that produced the error in a reproducible way, and the error message that was produced.

## Feature Requests

Similar to bug reports, feature requests also end up as issues in the repository and templates are available both directly in the repository and through the "Contact" page.

Feature requests are a very broad category and can include a simple method that you think is useful to an entirely new module.
It's generally preferable to fill out a feature request form as soon as you have a concrete idea of the feature; we can then discuss, improve, and adapt the feature to the project's structure and direction.

Of course, you can also make a fresh branch and submit your feature as a pull request. However, we can't guarantee that the feature will be merged.
Therefore, a discussion upfront is always recommended even if you want to handle the implementation completely on your own.

## Code Contributions

Code contributions are also welcome, especially compared to what you might see in larger OSS projects.
However, contributing code does mean that some conventions of the repository should be followed.

First and foremost, all code contributions should be made through pull requests.
The `main` branch is the base branch for contributions and should be used as the starting point for your branch.

> __Code Style__
>
> We do not follow a strict repository-wide style guideline, but we do use `black` to format our code and `mypy` to type check it.
> The only "requirement" that isn't automatically applied is the use of type hints.
> SymbolicDSGE makes extensive use of type hints and generally favors explicitness over highly dynamic Python patterns.
>
> If you're contributing to an existing module, you can follow the style of the surrounding code. Otherwise, you can follow your style as long as the code remains readable in context.

### Setting up your environment

The project uses `uv` to manage the environment and dependencies. A `uv.lock` file and `pyproject.toml` are included in the repository for easy setup.

You want to make sure that you install the `dev` dependency group, which bundles the testing, linting, and formatting dependencies alongside some Jupyter notebook dependencies.

If you don't have `uv` installed, run the following commands in the terminal to install it:

```bash
pip install pipx  # `uv` suggests using `pipx` for installation, but you can also use `pip` if you prefer.
pipx install uv
```

To quickly set up your environment, you can use one of the following commands in the terminal:

```bash
uv sync --all-extras --all-groups # installs all optional dependencies and groups, including the `dev` group
uv sync --group dev # installs only the `dev` group
```

These commands will create a virtual environment, install the correct Python version, and resolve all dependencies.

If your IDE/editor does not automatically detect the environment, you can run:

```bash
source .venv/bin/activate # Linux/macOS
.venv\Scripts\Activate.ps1 # Windows PowerShell
```

to point your shell towards the correct environment.

### Pre-commit checks

We ship a pre-commit stack that focuses on formatting, linting, and protecting the developer's secrets.
It is strongly recommended to make use of the hooks by running the following command in the terminal:

```bash
uv run pre-commit install
```

The command installs a hook sequence that will:

- Format your code with `black`
- Run a type check with `mypy`
- Fix EOF trailing newlines
- Fix whitespace issues
- Check for any staged `env` files with populated secrets and block the commit if any are found

This sequence will run on every commit and commits will fail if any of the checks fail.
`black`, EOF, and whitespace issues will be automatically fixed after the initial failure.

### Keeping secrets in your local environment

SymbolicDSGE has a `FRED` module that allows users to pull data from St. Louis Fed's FRED API.

You will need a free API key to use this module. You can place that key in a local `.env` file and tell `git` to ignore local changes to that file.
You will find an empty `.env` file in the repository for this purpose.

With an empty `.env` file in your local repository, run the following command before adding your API key:

```bash
git update-index --skip-worktree <your-env-file-path>
```

This command will make git ignore any changes to the file without you having to push a global `.gitignore` change.

### Tests

We ship a testing suite with `pytest` located in the `./tests` directory.
The complete test suite is run on every commit and pull request, but you can also locally ensure that the suite is passing.

When you make changes that cause a test failure, you can make a comment in your pull request to explain the behavior change and why the test is failing.
As you add new code, some pre-existing tests may naturally become outdated or obsolete. In such cases, mention it in the pull request and we can decide whether the tests should be updated as part of the PR.

If you're adding uncovered code, you should also add tests alongside the code contribution.
We use the `pytest-cov` plugin to track test coverage and you can run the test stack with the following command to see how much of your code is covered:

```bash
uv run pytest <optional:your-test-file> --cov
```

### Performance critical code

SymbolicDSGE, being a dynamic modeling library, has multiple hot paths that require careful planning and optimization.
We generally rely on `numba`'s no-python JIT compilation to speed things up and avoid the platform-specific nature of Cython or its C/C++ counterparts.

If you're contributing to a hot path, you should inspect the surrounding code to understand how `numba` is being utilized.
As a general rule of thumb, sticking to `numpy` types and avoiding Python objects in compiled functions will ensure your code can compile, but it doesn't ensure it will be fast.

However, if you're not confident in your knowledge of `numba` and its many quirks, you can write the logic and ask for help and/or someone to refactor it into compilable code.
The above suggestion also applies to new features. You are welcome to implement a feature in plain Python and we can iterate on it together to make it suit the current codebase before merging the PR.
