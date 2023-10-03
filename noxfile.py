import nox


@nox.session(python=["3.11"])
def tests(session):
    session.install("poetry")
    session.run("poetry", "install", external=True)
    session.run("poetry", "run", "pytest", "tests", external=True)
