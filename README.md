
<p align='center'><a href='https://github.com/lutzroeder/netron'><img width='400' src='.github/logo.png'/></a></p>

Netron is a viewer for neural network, deep learning and machine learning models. 

Netron supports **ONNX** (`.onnx`, `.pb`, `.pbtxt`), **Keras** (`.h5`, `.keras`), **Core ML** (`.mlmodel`), **Caffe** (`.caffemodel`, `.prototxt`), **Caffe2** (`predict_net.pb`, `predict_net.pbtxt`), **Darknet** (`.cfg`), **MXNet** (`.model`, `-symbol.json`), **ncnn** (`.param`) and **TensorFlow Lite** (`.tflite`).

Netron has experimental support for **TorchScript** (`.pt`, `.pth`), **PyTorch** (`.pt`, `.pth`), **Torch** (`.t7`), **Arm NN** (`.armnn`), **BigDL** (`.bigdl`, `.model`), **Chainer** (`.npz`, `.h5`), **CNTK** (`.model`, `.cntk`), **Deeplearning4j** (`.zip`), **ML.NET** (`.zip`), **MNN** (`.mnn`), **OpenVINO** (`.xml`), **PaddlePaddle** (`.zip`, `__model__`), **scikit-learn** (`.pkl`), **TensorFlow.js** (`model.json`, `.pb`) and **TensorFlow** (`.pb`, `.meta`, `.pbtxt`, `.ckpt`, `.index`).

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='.github/screenshot.png' width='800'></a></p>

## Install

**macOS**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.dmg` file or run `brew cask install netron`

**Linux**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.AppImage` file or run `snap install netron`

**Windows**: [**Download**](https://github.com/lutzroeder/netron/releases/latest) the `.exe` installer.

**Browser**: [**Start**](https://www.lutzroeder.com/ai/netron) the browser version.

**Python Server**: Run `pip install netron` and `netron [FILE]` or `import netron; netron.start('[FILE]')`.

## Download Models

Sample model files to download and open:

 * **ONNX**: [resnet-18](https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx)
 * **Keras**: [tiny-yolo-voc](https://github.com/hollance/YOLO-CoreML-MPSNNGraph/raw/master/Convert/yad2k/model_data/tiny-yolo-voc.h5)
 * **CoreML**: [faces_model](https://github.com/NovaTecConsulting/FaceRecognition-in-ARKit/files/1526806/faces_model.mlmodel.zip) 
 * **TensorFlow Lite**: [smartreply](https://storage.googleapis.com/download.tensorflow.org/models/tflite/smartreply_1.0_2017_11_01.zip)
 * **MXNet**: [inception_v1](https://s3.amazonaws.com/model-server/models/onnx-inception_v1/inception_v1.model)
 * **Caffe**: [mobilenet_v2](https://raw.githubusercontent.com/shicai/MobileNet-Caffe/master/mobilenet_v2.caffemodel)
 * **TensorFlow**: [inception_v3](https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz)
