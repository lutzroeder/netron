
<p align='center'><img width='400' src='media/logo.png'/></p>

Netron is a viewer for [ONNX](http://onnx.ai) neural network models.

<p align='center'><a href='https://www.lutzroeder.com/ai'><img src='media/screenshot.png' width='800'></a></p>

## Install

### macOS

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.dmg` file or with [Homebrew](https://caskroom.github.io) run `brew cask install netron`

### Linux

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.AppImage` or `.deb` file. The `.AppImage` needs to be made [executable](http://discourse.appimage.org/t/how-to-make-an-appimage-executable/80) after download.

### Windows

[**Download**](https://github.com/lutzroeder/Netron/releases/latest) the `.exe` file.

## Models

Download sample ONNX model files [here](https://github.com/onnx/models).

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
