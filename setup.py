#!/usr/bin/env python

import distutils
import io
import json
import os
import re
import setuptools
import setuptools.command.build_py
import distutils.command.build

node_dependencies = [ 
    ( 'netron', [
        'node_modules/d3/dist/d3.min.js',
        'node_modules/dagre/dist/dagre.min.js',
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
        if build_py.version:
            for package, src_dir, build_dir, filenames in self.data_files:
                for filename in filenames:
                    if filename == 'index.html':
                        filepath = os.path.join(build_dir, filename)
                        with open(filepath, 'r') as file :
                            content = file.read()
                        content = re.sub(r'(<meta name="version" content=")\d+.\d+.\d+(">)', r'\g<1>' + package_version() + r'\g<2>', content)
                        with open(filepath, 'w') as file:
                            file.write(content)
    def build_module(self, module, module_file, package):
        setuptools.command.build_py.build_py.build_module(self, module, module_file, package)
        if build_py.version and module == '__version__':
            outfile = self.get_module_outfile(self.build_lib, package.split('.'), module)
            with open(outfile, 'w+') as file:
                file.write("__version__ = '" + package_version() + "'\n")

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
                     'Netron supports **ONNX** (`.onnx`, `.pb`), **Keras** (`.h5`, `.keras`), **Core ML** (`.mlmodel`), **Caffe** (`.caffemodel`, `.prototxt`), **Caffe2** (`predict_net.pb`), **Darknet** (`.cfg`), **MXNet** (`.model`, `-symbol.json`), ncnn (`.param`) and **TensorFlow Lite** (`.tflite`). Netron has experimental support for **TorchScript** (`.pt`, `.pth`), **PyTorch** (`.pt`, `.pth`), **Torch** (`.t7`), **ArmNN** (`.armnn`), **Barracuda** (`.nn`), **BigDL** (`.bigdl`, `.model`), **Chainer** (`.npz`, `.h5`), **CNTK** (`.model`, `.cntk`), **Deeplearning4j** (`.zip`), **PaddlePaddle** (`__model__`), **MediaPipe** (`.pbtxt`), **ML.NET** (`.zip`), MNN (`.mnn`), **OpenVINO** (`.xml`), **scikit-learn** (`.pkl`), **Tengine** (`.tmfile`), **TensorFlow.js** (`model.json`, `.pb`) and **TensorFlow** (`.pb`, `.meta`, `.pbtxt`, `.ckpt`, `.index`).',
    keywords=[
        'onnx', 'keras', 'tensorflow', 'tflite', 'coreml', 'mxnet', 'caffe', 'caffe2', 'torchscript', 'pytorch', 'ncnn', 'mnn', 'openvino', 'darknet', 'paddlepaddle', 'chainer',
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
            'base.js', 
            'numpy.js', 'pickle.js', 'hdf5.js', 'bson.js',
            'zip.js', 'tar.js', 'gzip.js',
            'armnn.js', 'armnn-metadata.json', 'armnn-schema.js',
            'bigdl.js', 'bigdl-metadata.json', 'bigdl-proto.js',
            'barracuda.js',
            'caffe.js', 'caffe-metadata.json', 'caffe-proto.js',
            'caffe2.js', 'caffe2-metadata.json', 'caffe2-proto.js',
            'chainer.js',
            'cntk.js', 'cntk-metadata.json', 'cntk-proto.js',
            'coreml.js', 'coreml-metadata.json', 'coreml-proto.js',
            'darknet.js', 'darknet-metadata.json',
            'dl4j.js', 'dl4j-metadata.json',
            'flux.js', 'flux-metadata.json',
            'keras.js', 'keras-metadata.json',
            'mediapipe.js',
            'mlnet.js', 'mlnet-metadata.json',
            'mnn.js', 'mnn-metadata.json', 'mnn-schema.js',
            'mxnet.js', 'mxnet-metadata.json',
            'ncnn.js', 'ncnn-metadata.json',
            'tnn.js', 'tnn-metadata.json',
            'onnx.js', 'onnx-metadata.json', 'onnx-proto.js',
            'openvino.js', 'openvino-metadata.json', 'openvino-parser.js',
            'paddle.js', 'paddle-metadata.json', 'paddle-proto.js',
            'pytorch.js', 'pytorch-metadata.json', 'python.js',
            'sklearn.js', 'sklearn-metadata.json',
            'tengine.js', 'tengine-metadata.json', 
            'uff.js', 'uff-metadata.json', 'uff-proto.js',
            'tf.js', 'tf-metadata.json', 'tf-proto.js', 
            'tflite.js', 'tflite-metadata.json', 'tflite-schema.js',
            'torch.js', 'torch-metadata.json',
            'index.html', 'index.js',
            'view-grapher.css', 'view-grapher.js',
            'view-sidebar.css', 'view-sidebar.js',
            'view.js',
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