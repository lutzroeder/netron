
<p align='center'><img width='400' src='media/logo.png'/></p>

Netron is a viewer for neural network models.

Netron loads **[ONNX](http://onnx.ai)** models (`.onnx` or `.pb`), **Keras** models (`.keras`, `.h5` or `.json`) and **TensorFlow Lite** models (`.tflite`) and has experimental support for **TensorFlow** models (`.pb` and `.meta`).

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='media/screenshot.png' width='800'></a></p>

## Install

**macOS**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.dmg` file or with [Homebrew](https://caskroom.github.io) run `brew cask install netron`

**Linux**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.AppImage` or `.deb` file. The `.AppImage` needs to be made [executable](http://discourse.appimage.org/t/how-to-make-an-appimage-executable/80) after download.

**Windows**

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.exe` file.

## Download Models

Below are a few model files you can download and open:

**ONNX Models**

* [Inception v1](https://github.com/onnx/models/blob/master/inception_v1)
* [Inception v2](https://github.com/onnx/models/blob/master/inception_v2)
* [ResNet-50](https://github.com/onnx/models/blob/master/resnet50)
* [ShuffleNet](https://github.com/onnx/models/blob/master/shufflenet)
* [SqueezeNet](https://github.com/onnx/models/blob/master/squeezenet)
* [VGG-19](https://github.com/onnx/models/blob/master/vgg19)
* [BVLC AlexNet](https://github.com/onnx/models/blob/master/bvlc_alexnet)

**TensorFlow Lite Models**

* [Smart Reply 1.0 ](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip)
* [Mobilenet 1.0 224 Float](https://storage.googleapis.com/download.tensorflow.org/models/tflite/mobilenet_v1_1.0_224_float_2017_11_08.zip)
* [Inception v3 2015](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_2015_2017_11_10.zip)
* [Inception v3 2016 Slim](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip)

**TensorFlow models**

* [Inception v3](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)
* [Inception v4](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz)
* [Inception 5h](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)

## Install Python Model Server 

To run Netron in a web browser, install the Python web server using pip: 
```
pip install netron
```

Launch the model server:

```
netron my_model.onnx
```

To serve a model from Python code:
```
import netron

netron.serve_file('my_model.onnx')
```
