# Contributing Guide

Interested in helping build xcollection? Have code from your work that
you believe others will find useful? Have a few minutes to tackle an issue?

Contributions are highly welcomed and appreciated. Every little help counts,
so do not hesitate!

The following sections cover some general guidelines
regarding development in xcollection for maintainers and contributors.
Nothing here is set in stone and can't be changed.
Feel free to suggest improvements or changes in the workflow.

## Feature requests and feedback

We'd also like to hear about your propositions and suggestions. Feel free to
submit them as issues on [xcollection's GitHub issue tracker](https://github.com/ncar-xdev/xcollection) and:

- Explain in detail how they should work.
- Keep the scope as narrow as possible. This will make it easier to implement.

## Report bugs

Report bugs for xcollection in the [issue tracker](https://github.com/ncar-xdev/xcollection).

If you are reporting a bug, please include:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting,
  specifically the Python interpreter version, installed libraries, and xcollection
  version.
- Detailed steps to reproduce the bug.

If you can write a demonstration test that currently fails but should pass
(xfail), that is a very useful commit to make as well, even if you cannot
fix the bug itself.

## Fix bugs

Look through the [GitHub issues for bugs](https://github.com/ncar-xdev/xcollection/labels/type:%20bug).

Talk to developers to find out how you can fix specific bugs.

## Write documentation

xcollection could always use more documentation. What exactly is needed?

- More complementary documentation. Have you perhaps found something unclear?
- Docstrings. There can never be too many of them.
- Blog posts, articles and such -- they're all very appreciated.

You can also edit documentation files directly in the GitHub web interface,
without using a local copy. This can be convenient for small fixes.

Build the documentation locally with the following command:

```bash
$ make docs
```

## Preparing Pull Requests

1. Fork the [xcollection GitHub repository](https://github.com/ncar-xdev/xcollection).

2. Clone your fork locally using [git](https://git-scm.com/), connect your repository
   to the upstream (main project), and create a branch::

   ```bash
   $ git clone git@github.com:YOUR_GITHUB_USERNAME/xcollection.git
   $ cd xcollection
   $ git remote add upstream git@github.com:ncar-xdev/xcollection.git
   ```

   now, to fix a bug or add feature create your own branch off "master":

   ```bash
   $ git checkout -b your-bugfix-feature-branch-name master
   ```

   If you need some help with Git, follow this quick start
   guide: https://git.wiki.kernel.org/index.php/QuickStart

3. Install dependencies into a new conda environment::

   ```bash
   $ conda env update -f ci/environment.yml
   $ conda activate xcollection-dev
   ```

4. Make an editable install of xcollection by running::

   ```bash
   $ python -m pip install -e .
   ```

5. Install `pre-commit <https://pre-commit.com>`\_ hooks on the xcollection repo::

   ```bash
   $ pre-commit install
   ```

   Afterwards `pre-commit` will run whenever you commit.

   [pre-commit](https://pre-commit.com) is a framework for managing and maintaining multi-language pre-commit hooks to ensure code-style and code formatting is consistent.

   Now you have an environment called `xcollection-dev` that you can work in.
   You’ll need to make sure to activate that environment next time you want
   to use it after closing the terminal or your system.

6. (Optional) Run all the tests

   Now running tests is as simple as issuing this command::

   ```bash
   $ pytest --cov=./
   ```

   This command will run tests via the `pytest` tool.

7. Commit and push once your tests pass and you are happy with your change(s)::

   When committing, `pre-commit` will re-format the files if necessary.

   ```bash
   $ git commit -a -m "<commit message>"
   $ git push -u
   ```

8. Finally, submit a pull request through the GitHub website using this data::

   ```console
   head-fork: YOUR_GITHUB_USERNAME/xcollection
   compare: your-branch-name

   base-fork: ncar-xdev/xcollection
   base: master # if it's a bugfix or feature
   ```
