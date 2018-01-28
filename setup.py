#!/usr/bin/python

import distutils
import os
import setuptools
import setuptools.command
import setuptools.command.build_py
import json

TOP_DIR = os.path.realpath(os.path.dirname(__file__))
with open(os.path.join(TOP_DIR, 'package.json')) as package_file:
    package_manifest = json.load(package_file)
    package_version = package_manifest['version']

node_dependencies = [ 
    ( 'netron', [
        'node_modules/d3/build/d3.min.js',
        'node_modules/dagre/dist/dagre.min.js',
        'node_modules/handlebars/dist/handlebars.min.js',
        'node_modules/marked/marked.min.js',
        'node_modules/protobufjs/dist/protobuf.min.js',
        'node_modules/flatbuffers/js/flatbuffers.js',
        'node_modules/npm-font-open-sans/open-sans.css' ]),
    ( 'netron/fonts/Regular', [
        'node_modules/npm-font-open-sans/fonts/Regular/OpenSans-Regular.eot',
        'node_modules/npm-font-open-sans/fonts/Regular/OpenSans-Regular.svg',
        'node_modules/npm-font-open-sans/fonts/Regular/OpenSans-Regular.ttf',
        'node_modules/npm-font-open-sans/fonts/Regular/OpenSans-Regular.woff',
        'node_modules/npm-font-open-sans/fonts/Regular/OpenSans-Regular.woff2' ]),
    ( 'netron/fonts/Semibold', [
        'node_modules/npm-font-open-sans/fonts/Semibold/OpenSans-Semibold.eot',
        'node_modules/npm-font-open-sans/fonts/Semibold/OpenSans-Semibold.svg',
        'node_modules/npm-font-open-sans/fonts/Semibold/OpenSans-Semibold.ttf',
        'node_modules/npm-font-open-sans/fonts/Semibold/OpenSans-Semibold.woff',
        'node_modules/npm-font-open-sans/fonts/Semibold/OpenSans-Semibold.woff2' ]),
    ( 'netron/fonts/Bold', [
        'node_modules/npm-font-open-sans/fonts/Bold/OpenSans-Bold.eot',
        'node_modules/npm-font-open-sans/fonts/Bold/OpenSans-Bold.svg',
        'node_modules/npm-font-open-sans/fonts/Bold/OpenSans-Bold.ttf',
        'node_modules/npm-font-open-sans/fonts/Bold/OpenSans-Bold.woff',
        'node_modules/npm-font-open-sans/fonts/Bold/OpenSans-Bold.woff2' ])
]

class build_py(setuptools.command.build_py.build_py):
    def run(self):
        result = setuptools.command.build_py.build_py.run(self)
        for target, files in node_dependencies:
            target = os.path.join(self.build_lib, target)
            if not os.path.exists(target):
                os.makedirs(target)
            for file in files:
                self.copy_file(file, target)
        return result

setuptools.setup(
    name="netron",
    version=package_version,
    description="Viewer for neural network models",
    license="MIT",
    cmdclass={
        'build_py': build_py
    },
    package_dir={
        'netron': 'src'
    },
    packages=[
        'netron'
    ],
    package_data={
        'netron': [ 
            'netron', 'netron.py',
            'logo.svg', 'favicon.ico',
            'onnx-model.js', 'onnx.js', 'onnx-operator.json',
            'tf-model.js', 'tf.js', 'tf-operator.pb',
            'tflite-model.js', 'tflite.js', 'tflite-operator.json',
            'keras-model.js', 'keras-operator.json', 'hdf5.js',
            'view-browser.html', 'view-browser.js',
            'view.js', 'view.css', 'view-render.css', 'view-render.js', 'view-template.js'
        ]
    },
    install_requires=[],
    author='Lutz Roeder',
    author_email='lutzroeder@users.noreply.github.com',
    url='https://github.com/lutzroeder/Netron',
    scripts=[
        'src/netron'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)