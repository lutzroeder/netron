#!/usr/bin/env python

import distutils
import io
import json
import os
import setuptools
import setuptools.command.build_py
import distutils.command.build

node_dependencies = [ 
    ( 'netron', [
        'node_modules/d3/dist/d3.min.js',
        'node_modules/dagre/dist/dagre.min.js',
        'node_modules/handlebars/dist/handlebars.min.js',
        'node_modules/marked/marked.min.js',
        'node_modules/pako/dist/pako.min.js',
        'node_modules/long/dist/long.js',
        'node_modules/protobufjs/dist/protobuf.min.js',
        'node_modules/protobufjs/ext/prototxt/prototxt.js',
        'node_modules/flatbuffers/js/flatbuffers.js' ] )
]

class build(distutils.command.build.build):
    user_options = distutils.command.build.build.user_options + [ ('version', None, 'version' ) ]
    def initialize_options(self):
        distutils.command.build.build.initialize_options(self)
        self.version = None
    def finalize_options(self):
        distutils.command.build.build.finalize_options(self)
    def run(self):
        build_py.version = bool(self.version)
        return distutils.command.build.build.run(self)

class build_py(setuptools.command.build_py.build_py):
    user_options = setuptools.command.build_py.build_py.user_options + [ ('version', None, 'version' ) ]
    def initialize_options(self):
        setuptools.command.build_py.build_py.initialize_options(self)
        self.version = None
    def finalize_options(self):
        setuptools.command.build_py.build_py.finalize_options(self)
    def run(self):
        setuptools.command.build_py.build_py.run(self)
        for target, files in node_dependencies:
            target = os.path.join(self.build_lib, target)
            if not os.path.exists(target):
                os.makedirs(target)
            for file in files:
                self.copy_file(file, target)
    def build_module(self, module, module_file, package):
        setuptools.command.build_py.build_py.build_module(self, module, module_file, package)
        if build_py.version and module == '__version__':
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
                     'Netron supports **ONNX** (`.onnx`, `.pb`), **Keras** (`.h5`, `.keras`), **Core ML** (`.mlmodel`), **Caffe** (`.caffemodel`, `.prototxt`), **Caffe2** (`predict_net.pb`), **MXNet** (`.model`, `-symbol.json`), **TorchScript** (`.pt`, `.pth`), NCNN (`.param`) and **TensorFlow Lite** (`.tflite`). Netron has experimental support for **PyTorch** (`.pt`, `.pth`), **Torch** (`.t7`), **CNTK** (`.model`, `.cntk`), **Deeplearning4j** (`.zip`), **PaddlePaddle** (`__model__`), **Darknet** (`.cfg`), **scikit-learn** (`.pkl`), **ML.NET** (`.zip`), **TensorFlow.js** (`model.json`, `.pb`) and **TensorFlow** (`.pb`, `.meta`, `.pbtxt`).',
    keywords=[
        'onnx', 'keras', 'tensorflow', 'coreml', 'mxnet', 'caffe', 'caffe2', 'torchscript', 'pytorch', 'ncnn', 'openvino', 'darknet',
        'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'visualizer', 'viewer'
    ],
    license="MIT",
    cmdclass={
        'build': build,
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
            'numpy.js', 'base.js', 'zip.js', 'tar.js', 'gzip.js',
            'onnx.js', 'onnx-metadata.json', 'onnx-proto.js',
            'caffe.js', 'caffe-metadata.json', 'caffe-proto.js',
            'caffe2.js', 'caffe2-metadata.json', 'caffe2-proto.js',
            'cntk.js', 'cntk-metadata.json', 'cntk-proto.js',
            'coreml.js', 'coreml-metadata.json', 'coreml-proto.js',
            'darknet.js', 'darknet-metadata.json',
            'dl4j.js', 'dl4j-metadata.json',
            'flux.js', 'flux-metadata.json', 'bson.js',
            'keras.js', 'keras-metadata.json', 'hdf5.js',
            'mlnet.js', 'mlnet-metadata.json',
            'mxnet.js', 'mxnet-metadata.json',
            'ncnn.js', 'ncnn-metadata.json',
            'mnn.js', 'mnn-metadata.json', 'mnn-schema.js',
            'openvino.js', 'openvino-metadata.json', 'openvino-parser.js',
            'paddle.js', 'paddle-metadata.json', 'paddle-proto.js',
            'pytorch.js', 'pytorch-metadata.json', 'pickle.js',
            'sklearn.js', 'sklearn-metadata.json',
            'tf.js', 'tf-metadata.json', 'tf-proto.js', 
            'tflite.js', 'tflite-metadata.json', 'tflite-schema.js', 
            'torch.js', 'torch-metadata.json',
            'torchscript.js', 'torchscript-metadata.json', 'python.js',
            'index.html', 'index.js',
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