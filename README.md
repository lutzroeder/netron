
<p align='center'><img width='400' src='media/logo.png'/></p>

Netron is a viewer for [ONNX](http://onnx.ai) neural network models.

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='media/screenshot.png' width='800'></a></p>

## Getting Started

Download and install the Netron app for Windows, macOS or Linux from [here](https://github.com/lutzroeder/Netron/releases).

Download example ONNX models [here](https://github.com/onnx/models).

## Python Model Server 

To run Netron in a web browser, install the Python web server using pip: 
```
pip install netron
```

Launch the model server and open web browser:

```
netron --browse my_model.onnx
```

To serve a model from Python code:
```
import netron

netron.serve_file('my_model.onnx', browse=True)
```
