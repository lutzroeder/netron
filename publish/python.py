''' Python Server publish script '''

import json
import os
import re
import sys
import shutil
import subprocess

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dist_dir = os.path.join(root_dir, 'dist')
dist_pypi_dir = os.path.join(dist_dir, 'pypi')
source_dir = os.path.join(root_dir, 'source')
publish_dir = os.path.join(root_dir, 'publish')

def _read(path):
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def _write(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def _update(path, regex, value):
    content = _read(path)
    def repl(match):
        return match.group(1) + value + match.group(3)
    content = re.sub(regex, repl, content)
    _write(path, content)

def _build():
    ''' Build dist/pypi '''
    shutil.rmtree(os.path.join(source_dir, '__pycache__'), ignore_errors=True)
    shutil.rmtree(dist_pypi_dir, ignore_errors=True)
    shutil.copytree(source_dir, os.path.join(dist_pypi_dir, 'netron'))
    shutil.copyfile(os.path.join(publish_dir, 'setup.py'), os.path.join(dist_pypi_dir, 'setup.py'))
    os.remove(os.path.join(dist_pypi_dir, 'netron', 'electron.js'))
    os.remove(os.path.join(dist_pypi_dir, 'netron', 'app.js'))

def _install():
    ''' Install dist/pypi '''
    args = [ 'python', '-m', 'pip', 'install', dist_pypi_dir ]
    try:
        subprocess.run(args, check=False)
    except (KeyboardInterrupt, SystemExit):
        pass

def _version():
    ''' Update version '''
    package = json.loads(_read('./package.json'))
    _update('./dist/pypi/setup.py', '(    version=")(.*)(",)', package['version'])
    _update('./dist/pypi/netron/server.py',
        "(__version__ = ')(.*)(')",
        package['version'])
    _update('./dist/pypi/netron/index.html',
        '(<meta name="version" content=")(.*)(">)',
        package['version'])
    _update('./dist/pypi/netron/index.html',
        '(<meta name="date" content=")(.*)(">)',
        package['date'])

def _start():
    ''' Start server '''
    # args = [ sys.executable, '-c', 'import netron; netron.main();' ] + sys.args
    # try:
    #     subprocess.run(args, env={ 'PYTHONPATH': './dist/pypi' }, check=False)
    # except (KeyboardInterrupt, SystemExit):
    #     pass
    sys.path.insert(0, os.path.join(root_dir, 'dist', 'pypi'))
    __import__('netron').main()
    sys.args = []
    del sys.argv[1:]

def main(): # pylint: disable=missing-function-docstring
    table = { 'build': _build, 'install': _install, 'version': _version, 'start': _start }
    sys.args = sys.argv[1:]
    while len(sys.args) > 0:
        command = sys.args.pop(0)
        del sys.argv[1]
        table[command]()

if __name__ == '__main__':
    main()
