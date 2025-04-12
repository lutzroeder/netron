""" Python Server publish script """

import json
import os
import re
import shutil
import subprocess
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
dist_dir = os.path.join(root_dir, "dist")
dist_pypi_dir = os.path.join(dist_dir, "pypi")
source_dir = os.path.join(root_dir, "source")
publish_dir = os.path.join(root_dir, "publish")

def _read(path):
    with open(path, encoding="utf-8") as file:
        return file.read()

def _write(path, content):
    with open(path, "w", encoding="utf-8") as file:
        file.write(content)

def _update(path, regex, value):
    content = _read(path)
    def repl(match):
        return f"{match.group(1)}{value}{match.group(3)}"
    content = re.sub(regex, repl, content)
    _write(path, content)
    if content.find(value) == -1:
        raise ValueError(f"Failed to update '{path}' with '{value}'.")

def _build():
    """ Build dist/pypi """
    shutil.rmtree(os.path.join(source_dir, "__pycache__"), ignore_errors=True)
    shutil.rmtree(dist_pypi_dir, ignore_errors=True)
    shutil.copytree(source_dir, os.path.join(dist_pypi_dir, "netron"))
    shutil.copyfile(
        os.path.join(root_dir, "pyproject.toml"),
        os.path.join(dist_pypi_dir, "pyproject.toml"))
    os.remove(os.path.join(dist_pypi_dir, "netron", "desktop.mjs"))
    os.remove(os.path.join(dist_pypi_dir, "netron", "app.js"))

def _install():
    """ Install dist/pypi """
    args = [ "python", "-m", "pip", "install", dist_pypi_dir ]
    try:
        subprocess.run(args, check=False)
    except (KeyboardInterrupt, SystemExit):
        pass

def _version():
    """ Update version """
    package = json.loads(_read("./package.json"))
    _update("./dist/pypi/pyproject.toml",
        '(version\\s*=\\s*")(.*)(")',
        package["version"])
    _update("./dist/pypi/netron/server.py",
        '(__version__ = ")(.*)(")',
        package["version"])
    _update("./dist/pypi/netron/index.html",
        '(<meta name="version" content=")(.*)(">)',
        package["version"])
    _update("./dist/pypi/netron/index.html",
        '(<meta name="date" content=")(.*)(">)',
        package["date"])

def _start():
    """ Start server """
    sys.path.insert(0, os.path.join(root_dir, "dist", "pypi"))
    __import__("netron").main()
    sys.args = []
    del sys.argv[1:]

def main():
    table = {
        "build": _build,
        "install": _install,
        "version": _version,
        "start": _start
    }
    sys.args = sys.argv[1:]
    while len(sys.args) > 0:
        command = sys.args.pop(0)
        del sys.argv[1]
        table[command]()

if __name__ == "__main__":
    main()
