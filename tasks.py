"""Common project tasks."""
from invoke import task


@task
def clean(cline):
    """Clean project artifacts such as python cache."""
    print("Deleting __pycache__ directories.")
    cline.run("find . -iname '__pycache__' | xargs rm -rf")


@task
def style(cline, write=False):
    """Style and optionally re-write the source files."""
    check_arg = "" if write else "--check"
    print("Checking code style with Black.")
    cline.run(f"python3 -m black {check_arg} .")


@task
def lint(cline):
    """Check for linting errors in the source code."""
    print("Linting with pylint.")
    cline.run(r"find . -iname '*.py' -not -path '*/\.*' | xargs python3 -m pylint -j 0")
    print("Type checking with mypy.")
    cline.run("python3 -m mypy .")


@task
def test(cline):
    """Discover and run test cases."""
    print("Running unit tests.")
    cline.run("TF_CPP_MIN_LOG_LEVEL=3 python3 -m unittest")


@task(style, lint, test)
def build(cline):
    """Build project code."""
    print("Loading summary.")
    cline.run("wc -l **.py")
    print("Build complete.")
