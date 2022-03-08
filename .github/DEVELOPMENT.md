# How to Develop Netron

Netron can run as both an [Electron](https://electronjs.org) app or a Python web server.

## Electron app

To start the Electron app, install [Node.js](https://nodejs.org) and run: 

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
npm install
npx electron .
```

To debug the Electron app, open the folder in [Visual Studio Code](https://code.visualstudio.com) and press <kbd>F5</kbd>. To attach the debugger to a render process select the `Debug` tab and `Debug Renderer Process` before launching.

## Python server

To build and launch the Python server run:

```bash
git clone https://github.com/lutzroeder/netron.git
cd netron
python publish/setup.py build
export PYTHONPATH=dist/lib:${PYTHONPATH}
python -c "import netron; netron.start()"
```
