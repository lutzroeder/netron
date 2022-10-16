''' Python Server publish script '''

import json
import os
import re
import sys
import shutil

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
    shutil.rmtree('./source/__pycache__', ignore_errors=True)
    shutil.rmtree('./dist/pypi', ignore_errors=True)
    shutil.copytree('./source/', './dist/pypi/netron/')
    shutil.copyfile('./publish/setup.py', './dist/pypi/setup.py')
    os.remove('./dist/pypi/netron/electron.js')
    os.remove('./dist/pypi/netron/app.js')

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
    sys.path.insert(0, 'dist/pypi')
    __import__('netron').main()
    sys.args = []
    del sys.argv[1:]

def main(): # pylint: disable=missing-function-docstring
    table = { 'build': _build, 'version': _version, 'start': _start }
    sys.args = sys.argv[1:]
    while len(sys.args) > 0:
        command = sys.args.pop(0)
        del sys.argv[1]
        table[command]()

if __name__ == '__main__':
    main()
