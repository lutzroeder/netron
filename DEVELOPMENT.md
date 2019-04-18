# How to Develop Netron

Netron can run as both an [Electron](https://electronjs.org) app or a Python web server.

## Develop the Electron app

To start the Electron app, install [Node.js](https://nodejs.org) and run: 

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
npm install
npx electron ./
```

To debug the Electron app use [Visual Studio Code](https://code.visualstudio.com) and install the [Debugger for Chrome](https://marketplace.visualstudio.com/items?itemName=msjsdiag.debugger-for-chrome) extension. Open the `./netron` root folder and press `F5`. To attach the debugger to a render process select the `Debug` tab and `Debug Renderer Process` before launching.

To build Electron release binaries to the `./build/electron` folder run:

```bash
npx electron-builder --mac --linux --win
```

## Develop the Python server

To build and launch the Python server run:

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
npm install
python setup.py build
PYTHONPATH=build/python/lib python -c "import netron; netron.main()"
```
