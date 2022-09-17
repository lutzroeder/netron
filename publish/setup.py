''' Python Server setup script '''

import setuptools

setuptools.setup(
    name="netron",
    version="0.0.0",
    description="Viewer for neural network, deep learning, and machine learning models",
    long_description='Netron is a viewer for '
                     'neural network, deep learning, and machine learning models.\n\n'
                     'Netron supports ONNX, TensorFlow Lite, Keras, Caffe, Darknet, ncnn, MNN, '
                     'PaddlePaddle, Core ML, MXNet, RKNN, MindSpore Lite, TNN, Barracuda, '
                     'Tengine, TensorFlow.js, Caffe2 and UFF. '
                     'Netron has experimental support for '
                     'PyTorch, TensorFlow, TorchScript, OpenVINO, Torch, Vitis AI, Arm NN, '
                     'BigDL, Chainer, CNTK, Deeplearning4j, MediaPipe, MegEngine, '
                     'ML.NET and scikit-learn.',
    keywords=[
        'onnx', 'keras', 'tensorflow', 'tflite', 'coreml', 'mxnet', 'caffe', 'caffe2',
        'torchscript', 'pytorch', 'ncnn', 'mnn', 'openvino', 'darknet', 'paddlepaddle', 'chainer',
        'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
        'visualizer', 'viewer'
    ],
    license="MIT",
    package_dir={ 'netron': 'netron' },
    packages=[ 'netron' ],
    package_data={ 'netron': [ '*.*' ] },
    exclude_package_data={ 'netron': [ 'app.js', 'electron.*' ] },
    install_requires=[],
    author='Lutz Roeder',
    author_email='lutzroeder@users.noreply.github.com',
    url='https://github.com/lutzroeder/netron',
    entry_points={ 'console_scripts': [ 'netron = netron:main' ] },
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
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
