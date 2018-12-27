
<p align='center'><a href='https://github.com/lutzroeder/netron'><img width='400' src='media/logo.png'/></a></p>

Netron is a viewer for neural network, deep learning and machine learning models. 

Netron supports **[ONNX](http://onnx.ai)** (`.onnx`, `.pb`, `.pbtxt`), **Keras** (`.h5`, `.keras`), **CoreML** (`.mlmodel`), **Caffe2** (`predict_net.pb`, `predict_net.pbtxt`), **MXNet** (`.model`, `-symbol.json`) and **TensorFlow Lite** (`.tflite`). Netron has experimental support for **Caffe** (`.caffemodel`, `.prototxt`), **PyTorch** (`.pth`), **Torch** (`.t7`), **CNTK** (`.model`, `.cntk`), **PaddlePaddle** (`__model__`), **Darknet** (`.cfg`), **scikit-learn** (`.pkl`), **TensorFlow.js** (`model.json`, `.pb`) and **TensorFlow** (`.pb`, `.meta`, `.pbtxt`).

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='media/screenshot.png' width='800'></a></p>

## Install

**macOS**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.dmg` file or run `brew cask install netron`

**Linux**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.AppImage` or `.deb` file. 

**Windows**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.exe` installer.

**Browser**: [**Start**](https://www.lutzroeder.com/ai/netron) the browser version.

**Python Server**: Run `pip install netron` and `netron -b [MODEL_FILE]`. In Python run `import netron` and `netron.start('model.onnx')`.

## Download Models

Sample model files you can download and open:

**ONNX Models**: [Inception v1](https://s3.amazonaws.com/download.onnx/models/inception_v1.tar.gz), [Inception v2](https://s3.amazonaws.com/download.onnx/models/inception_v2.tar.gz), [ResNet-50](https://s3.amazonaws.com/download.onnx/models/resnet50.tar.gz), [SqueezeNet](https://s3.amazonaws.com/download.onnx/models/squeezenet.tar.gz)

**Keras Models**: [resnet](https://github.com/Hyperparticle/one-pixel-attack-keras/raw/master/networks/models/resnet.h5), [tiny-yolo-voc](https://github.com/hollance/YOLO-CoreML-MPSNNGraph/raw/master/Convert/yad2k/model_data/tiny-yolo-voc.h5)

**CoreML Models**: [MobileNet](https://docs-assets.developer.apple.com/coreml/models/MobileNet.mlmodel), [Places205-GoogLeNet](https://docs-assets.developer.apple.com/coreml/models/GoogLeNetPlaces.mlmodel), [Inception v3](https://docs-assets.developer.apple.com/coreml/models/Inceptionv3.mlmodel)

**TensorFlow Lite Models**: [Smart Reply 1.0 ](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip), [Inception v3 2016](https://storage.googleapis.com/download.tensorflow.org/models/tflite/inception_v3_slim_2016_android_2017_11_10.zip)

**Caffe Models**: [BVLC AlexNet](http://dl.caffe.berkeleyvision.org/bvlc_alexnet.caffemodel), [BVLC CaffeNet](http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel), [BVLC GoogleNet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)

**Caffe2 Models**: [BVLC GoogleNet](https://github.com/caffe2/models/raw/master/bvlc_googlenet/predict_net.pb), [Inception v2](https://github.com/caffe2/models/raw/master/inception_v2/predict_net.pb)

**MXNet Models**: [CaffeNet](http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.1-symbol.json), [SqueezeNet v1.1](https://mxnet.incubator.apache.org/model_zoo/index.html)

**TensorFlow models**: [Inception v3](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz), [Inception v4](https://storage.googleapis.com/download.tensorflow.org/models/inception_v4_2016_09_09_frozen.pb.tar.gz), [Inception 5h](https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip)
