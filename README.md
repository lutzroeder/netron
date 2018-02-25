
<p align='center'><img width='400' src='media/logo.png'/></p>

Netron is a viewer for neural network and machine learning models. 

Netron supports **[ONNX](http://onnx.ai)** models (`.onnx`, `.pb`), **Keras** models (`.keras`, `.h5`) and **TensorFlow Lite** models (`.tflite`).

Netron has experimental support for **CoreML** models (`.mlmodel`) and **TensorFlow** models (`.pb`, `.meta`).

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='media/screenshot.png' width='800'></a></p>

## Install

**macOS**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.dmg` file or with [Homebrew](https://caskroom.github.io) run `brew cask install netron`

**Linux**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.AppImage` or `.deb` file. 

**Windows**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.exe` installer.

**Browser**

[**Start**](https://www.lutzroeder.com/ai/netron) the browser version.


**Python Server**

Run `pip install netron` and `netron [MODEL_FILE]`.  
Serve a model in Python using `import netron; netron.serve_file('my_model.onnx')`.

## Download Models

Sample model files you can download and open:

**ONNX Models**

* [Inception v1](https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz)
* [Inception v2](https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz)
* [ResNet-50](https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz)
* [SqueezeNet](https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz)

**Keras Models**

* [resnet](https://github.com/Hyperparticle/one-pixel-attack-keras/raw/master/networks/models/resnet.h5)
* [densenet](https://github.com/Hyperparticle/one-pixel-attack-keras/raw/master/networks/models/densenet.h5)
* [tiny-yolo-voc](https://github.com/hollance/YOLO-CoreML-MPSNNGraph/raw/master/Convert/yad2k/model_data/tiny-yolo-voc.h5)

**TensorFlow Lite Models**

* [Smart Reply 1.0 ](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip)
* [Mobilenet 1.0 224 Float](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_float_2017_11_08.zip)
* [Inception v3 2016](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip)

**CoreML Models**

* [MobileNet](https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel)
* [Places205-GoogLeNet](https://docs-assets.developer.apple.com/coreml/models/GoogLeNetPlaces.mlmodel)
* [Inception v3](https://docs-assets.developer.apple.com/coreml/models/Inceptionv3.mlmodel)

**TensorFlow models**

* [Inception v3](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)
* [Inception v4](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz)
* [Inception 5h](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)
