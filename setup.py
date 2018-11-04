#!/usr/bin/env python

import distutils
import io
import json
import os
import setuptools
import setuptools.command.build_py

node_dependencies = [ 
    ( 'netron', [
        'node_modules/d3/dist/d3.min.js',
        'node_modules/dagre/dist/dagre.min.js',
        'node_modules/handlebars/dist/handlebars.min.js',
        'node_modules/marked/marked.min.js',
        'node_modules/pako/dist/pako.min.js',
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
    def build_module(self, module, module_file, package):
        setuptools.command.build_py.build_py.build_module(self, module, module_file, package)
        if module == '__version__':
            package = package.split('.')
            outfile = self.get_module_outfile(self.build_lib, package, module)
            with open(outfile, 'w+') as f:
                f.write("__version__ = '" + package_version() + "'\n")

def package_version():
    folder = os.path.realpath(os.path.dirname(__file__))
    with open(os.path.join(folder, 'package.json')) as package_file:
        package_manifest = json.load(package_file)
        return package_manifest['version']

setuptools.setup(
    name="netron",
    version=package_version(),
    description="Viewer for neural network, deep learning and machine learning models",
    long_description='Netron is a viewer for neural network, deep learning and machine learning models.\n\n' +
                     'Netron supports **ONNX** (`.onnx`, `.pb`), **Keras** (`.h5`, `.keras`), **CoreML** (`.mlmodel`) and **TensorFlow Lite** (`.tflite`). Netron has experimental support for **Caffe** (`.caffemodel`), **Caffe2** (`predict_net.pb`), **MXNet** (`-symbol.json`), **TensorFlow.js** (`model.json`, `.pb`) and **TensorFlow** (`.pb`, `.meta`).',
    keywords=[
        'onnx', 'keras', 'tensorflow', 'coreml', 'mxnet', 'caffe', 'caffe2',
        'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'visualizer', 'viewer'
    ],
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
            'favicon.ico', 'icon.png',
            'onnx.js', 'onnx.js', 'onnx-metadata.json',
            'tf-model.js', 'tf-metadata.json', 'tf-proto.js', 
            'tflite-model.js', 'tflite-metadata.json', 'tflite-schema.js', 
            'keras.js', 'keras-metadata.json', 'hdf5.js',
            'coreml.js', 'coreml-metadata.json', 'coreml-proto.js',
            'caffe.js', 'caffe-metadata.json', 'caffe-proto.js',
            'caffe2.js', 'caffe2-metadata.json', 'caffe2-proto.js',
            'mxnet.js', 'mxnet-metadata.json',
            'cntk.js', 'cntk-metadata.json', 'cntk-proto.js',
            'pytorch.js', 'pytorch-metadata.json', 'pickle.js',
            'sklearn.js', 'sklearn-metadata.json',
            'numpy.js', 'zip.js', 'tar.js', 'gzip.js',
            'view-browser.html', 'view-browser.js',
            'view-grapher.css', 'view-grapher.js',
            'view-sidebar.css', 'view-sidebar.js',
            'view.js', 'view.css',
            'server.py'
        ]
    },
    install_requires=[],
    author='Lutz Roeder',
    author_email='lutzroeder@users.noreply.github.com',
    url='https://github.com/lutzroeder/netron',
    entry_points={
        'console_scripts': [ 'netron = netron:main' ]
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Visualization'
    ]
)