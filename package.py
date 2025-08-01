import os
import sys

argv = sys.argv[1:]

root_dir = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(root_dir, "dist")
dist_pypi_dir = os.path.join(dist_dir, "pypi")

def _build():
    import shutil
    source_dir = os.path.join(root_dir, "source")
    shutil.rmtree(os.path.join(source_dir, "__pycache__"), ignore_errors=True)
    shutil.rmtree(dist_pypi_dir, ignore_errors=True)
    shutil.copytree(source_dir, os.path.join(dist_pypi_dir, "netron"))
    shutil.copyfile(
        os.path.join(root_dir, "pyproject.toml"),
        os.path.join(dist_pypi_dir, "pyproject.toml"))
    os.remove(os.path.join(dist_pypi_dir, "netron", "desktop.mjs"))
    os.remove(os.path.join(dist_pypi_dir, "netron", "app.js"))

def _install():
    import pip._internal.cli.main
    pip._internal.cli.main.main(["install", dist_pypi_dir])

def _version():
    import json
    import re
    path = os.path.join(root_dir, "package.json")
    with open(path, encoding="utf-8") as file:
        package = json.load(file)
    version = package["version"]
    date = package["date"]
    entries = [
        ("pyproject.toml", '(version\\s*=\\s*")(.*)(")', version),
        ("netron/server.py", '(__version__\\s=\\s")(.*)(")', version),
        ("netron/index.html", '(<meta name="version" content=")(.*)(">)', version),
        ("netron/index.html", '(<meta name="date" content=")(.*)(">)', date)
    ]
    for path, regex, value in entries:
        path = os.path.join(dist_pypi_dir, path)
        with open(path, encoding="utf-8") as file:
            content = file.read()
        content, count = re.subn(regex, rf"\g<1>{value}\g<3>", content)
        if count == 0:
            raise ValueError(f"Failed to update '{path}' with '{value}'.")
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)

def _start():
    """ Start server """
    sys.path.insert(0, os.path.join(root_dir, "dist", "pypi"))
    __import__("netron").main()
    argv.clear()

def main():
    table = {
        "build": _build,
        "install": _install,
        "version": _version,
        "start": _start
    }
    while len(argv) > 0:
        command = argv.pop(0)
        del sys.argv[1]
        table[command]()

if __name__ == "__main__":
    main()
