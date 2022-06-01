
import json

with open('./package.json') as file:
    package = json.load(file)
    version = package['version']

def replace(path, old, new):
    with open(path, 'r') as file:
        content = file.read()
    content = content.replace(old, new)
    with open(path, 'w') as file:
        file.write(content)

replace('./dist/pypi/setup.py', '0.0.0', version)
replace('./dist/pypi/netron/__version__.py', '0.0.0', version)
replace('./dist/pypi/netron/index.html', '0.0.0', version)
