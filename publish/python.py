''' Python Server publish script '''

import json
import os
import re
import sys
import shutil
import subprocess

def read(path):
    ''' Read file content '''
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()

def write(path, content):
    ''' Write file content '''
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def update(path, regex, value):
    ''' Update regex patter in file content with value '''
    content = read(path)
    def repl(match):
        return match.group(1) + value + match.group(3)
    content = re.sub(regex, repl, content)
    write(path, content)

def build():
    ''' Build dist/pypi '''
    shutil.rmtree('./source/__pycache__', ignore_errors=True)
    shutil.rmtree('./dist/pypi', ignore_errors=True)
    shutil.copytree('./source/', './dist/pypi/netron/')
    shutil.copyfile('./publish/setup.py', './dist/pypi/setup.py')
    os.remove('./dist/pypi/netron/electron.html')
    os.remove('./dist/pypi/netron/electron.js')
    os.remove('./dist/pypi/netron/app.js')

def version():
    ''' Update version '''
    package = json.loads(read('./package.json'))
    update('./dist/pypi/setup.py', '(    version=")(.*)(",)', package['version'])
    update('./dist/pypi/netron/server.py',
        "(__version__ = ')(.*)(')",
        package['version'])
    update('./dist/pypi/netron/index.html',
        '(<meta name="version" content=")(.*)(">)',
        package['version'])
    update('./dist/pypi/netron/index.html',
        '(<meta name="date" content=")(.*)(">)',
        package['date'])

def start():
    ''' Start server '''
    sys.path.insert(0, './dist/pypi')
    args = [ sys.executable, '-c', 'import netron; netron.main()' ] + sys.args
    sys.args = []
    subprocess.run(args, env={ 'PYTHONPATH': './dist/pypi' }, check=False)

def main(): # pylint: disable=missing-function-docstring
    command_table = { 'build': build, 'version': version, 'start': start }
    sys.args = sys.argv[1:]
    while len(sys.args) > 0:
        command = sys.args.pop(0)
        command_table[command]()

if __name__ == '__main__':
    main()
