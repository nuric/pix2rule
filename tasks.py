"""Common project tasks."""
from invoke import task


@task
def test(c):
    """Discover and run test cases."""
    c.run("python3 -m unittest")
