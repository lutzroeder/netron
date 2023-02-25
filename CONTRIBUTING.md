# How to Develop Netron

## Debugging

Netron can run as both an [Electron](https://electronjs.org) app or a Python web server.

To start the Electron app, install [Node.js](https://nodejs.org) and run: 

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
npm install
npx electron .
```

To debug the Electron app, open the folder in [Visual Studio Code](https://code.visualstudio.com) and press <kbd>F5</kbd>. To attach the debugger to a render process select the `Debug` tab and `Electron View` before launching.

To build and launch the Python server run:

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
python publish/python.py build start --browse
```

## Validation

To validate changes run:

```bash
npm run -s lint
npm run -s test <format> # e.g. npm run -s test onnx
```
