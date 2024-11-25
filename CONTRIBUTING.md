# How to Develop Netron

## Debugging

Netron can run as both an [Electron](https://electronjs.org) app or a web app.

To start the Electron app, install [Node.js](https://nodejs.org) and run: 

```bash
npm install
npm start
```

To debug the Electron app, open the folder in [Visual Studio Code](https://code.visualstudio.com) and press <kbd>F5</kbd>. To attach the debugger to the render process select the `Debug` tab and pick `Electron View` before launching.

To build and launch the web app, pick `Browser` in the `Debug` tab or run this command:

```bash
python package.py build start --browse
```

## Validation

To validate changes run:

```bash
npm run lint
npm test [format] # e.g. npm test onnx
```
