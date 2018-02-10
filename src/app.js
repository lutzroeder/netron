/*jshint esversion: 6 */

const electron = require('electron');
const updater = require('electron-updater');
const fs = require('fs');
const os = require('os');
const path = require('path');
const process = require('process');
const url = require('url');

class Application {

    constructor() {
        this._views = [];
        this._openFileQueue = [];
        this._configuration = null;

        electron.app.setAppUserModelId('com.lutzroeder.netron');

        var application = this;
        if (electron.app.makeSingleInstance(() => { application.restoreWindow(); })) {
            electron.app.quit();
        }

        electron.ipcMain.on('open-file-dialog', (e, data) => {
            this.openFileDialog();
        });

        electron.ipcMain.on('drop-file', (e, data) => {
            application.dropFile(data.file, data.windowId);
        });

        electron.ipcMain.on('update-window', (e, data) => {
            application.updateWindow(data.file, data.windowId);
        });

        electron.app.on('will-finish-launching', () => {
            electron.app.on('open-file', (e, path) => {
                application.openFile(path);
            });
        });

        electron.app.on('ready', () => {
            application.ready();
        });

        electron.app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                electron.app.quit();
            }
        });

        electron.app.on('will-quit', () => {
            application.saveConfiguration();
        });

        if (process.platform == 'win32' && process.argv.length > 1) {
            process.argv.slice(1).forEach((arg) => {
                if (!arg.startsWith('-') && arg.split('.').pop() != 'js') {
                    application.openFile(arg);
                }
            });
        }

        this.update();
    }

    ready() {
        this.loadConfiguration();
        this.updateMenu();
        if (this._openFileQueue) {
            var openFileQueue = this._openFileQueue;
            this._openFileQueue = null;
            while (openFileQueue.length > 0) {
                var file = openFileQueue.shift();
                this.openFile(file);
            }
        }
        if (this._views.length == 0) {
            this.openView();
        }
    }

    openFileDialog() {
        var showOpenDialogOptions = { 
            properties: [ 'openFile' ], 
            filters: [
                { name: 'ONNX Model', extensions: [ 'onnx', 'pb' ] },
                { name: 'TensorFlow Saved Model', extensions: [ 'saved_model.pb' ] },
                { name: 'TensorFlow Graph', extensions: [ 'pb', 'meta' ] },
                { name: 'TensorFlow Lite Model', extensions: [ 'tflite' ] },
                { name: 'Keras Model', extension: [ 'json', 'keras', 'h5' ] }
            ]
        };
        electron.dialog.showOpenDialog(showOpenDialogOptions, (selectedFiles) => {
            if (selectedFiles) {
                selectedFiles.forEach((selectedFile) => {
                    this.openFile(selectedFile);
                });
            }
        });
    }

    openFile(file) {
        if (this._openFileQueue) {
            this._openFileQueue.push(file);
            return;
        }
        if (file && file.length > 0 && fs.existsSync(file))
        {
            // find existing view for this file
            var view = this._views.find(view => view.path && view.path == file);
            // find empty welcome window
            if (view == null) {
                view = this._views.find(view => !view.path || view.path.length == 0);
            }
            // create new window
            if (view == null) {
                view = this.openView();
            }
            this.loadFile(file, view);
        }
    }

    loadFile(file, view) {
        this._configuration.recents = this._configuration.recents.filter(recent => file != recent.path);
        var window = view.window;
        if (view.ready) {
            window.webContents.send("open-file", { file: file });
        }
        else {
            window.webContents.on('dom-ready', () => {
                window.webContents.send("open-file", { file: file });
            });
            var location = url.format({
                pathname: path.join(__dirname, 'view-electron.html'),
                protocol: 'file:',
                slashes: true
            });
            window.loadURL(location);
        }
        this._configuration.recents.unshift({ path: file });
        if (this._configuration.recents.length > 10) {
            this._configuration.recents.splice(10);
        }
        this.updateMenu();
    }

    dropFile(file, windowId) {
        if (file) { 
            var view = null;
            if (windowId) {
                var window = electron.BrowserWindow.fromId(windowId);
                if (window) {
                    view = this._views.find(view => view.window == window);
                }
            }
            if (view) {
                this.loadFile(file, view);
            }
            else {
                this.openFile(file);
            }
        }
    }

    updateWindow(file, windowId) {
        var window = electron.BrowserWindow.fromId(windowId);
        var view = this._views.find(view => view.window == window);
        if (view) {
            view.path = file;
            var title = Application.minimizePath(file);
            if (process.platform !== 'darwin') {
                title = file + ' - ' + electron.app.getName();
            }
            window.setTitle(title);
        }
    }

    openView() {
        const title = electron.app.getName();
        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        if (size.width > 1024) {
            size.width = 1024;
        }
        if (size.height > 768) {
            size.height = 768;
        }
        var window = new electron.BrowserWindow({ 
            title: title,
            backgroundColor: '#eeeeee',
            minWidth: 600,
            minHeight: 400,
            width: size.width,
            height: size.height,
            icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png'))
        });
        var application = this;
        window.on('closed', function () {
            application.closeWindow(this);
        });
        var view = { 
            window: window,
            ready: false
        };
        window.webContents.on('dom-ready', function() {
            view.ready = true;
        });
        var location = url.format({ pathname: path.join(__dirname, 'view-electron.html'), protocol: 'file:', slashes: true });
        window.loadURL(location);
        this._views.push(view);
        return view;
    }

    closeWindow(window) {
        for (var i = this._views.length - 1; i >= 0; i--) {
            if (this._views[i].window == window) {
                this._views.splice(i, 1);
            }   
        }
    }

    restoreWindow() {
        if (this._views.length > 0) {
            var view = this._views[0];
            if (view && view.window) { 
                if (view.window.isMinimized()) {
                    view.window.restore();
                }
                view.window.show();
            }
        }
    }

    update() {
        var isDev = ('ELECTRON_IS_DEV' in process.env) ?
            (parseInt(process.env.ELECTRON_IS_DEV, 10) === 1) :
            (process.defaultApp || /node_modules[\\/]electron[\\/]/.test(process.execPath));
        if (!isDev) {
            updater.autoUpdater.checkForUpdatesAndNotify();
        }
    }

    loadConfiguration() {
        var dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            var file = path.join(dir, 'configuration.json'); 
            if (fs.existsSync(file)) {
                var data = fs.readFileSync(file);
                if (data) {
                    this._configuration = JSON.parse(data);
                }
            }
        }
        if (!this._configuration) {
            this._configuration = {
                'recents': []
            };
        }
    }

    saveConfiguration() {
        if (this._configuration) {
            var data = JSON.stringify(this._configuration);
            if (data) {
                var dir = electron.app.getPath('userData');
                if (dir && dir.length > 0) {
                    var file = path.join(dir, 'configuration.json'); 
                    fs.writeFileSync(file, data);          
                }
            }
        }
    }

    updateMenu() {

        var menuRecentsTemplate = [];
        if (this._configuration && this._configuration.recents) {
            this._configuration.recents = this._configuration.recents.filter(recent => fs.existsSync(recent.path));
            this._configuration.recents.forEach((recent) => {
                var file = recent.path;
                menuRecentsTemplate.push({ 
                    label: Application.minimizePath(recent.path),
                    click: () => { this.openFile(file); }
                });
            });
        }

        var menuTemplate = [];
        
        if (process.platform === 'darwin') {
            menuTemplate.unshift({
                label: electron.app.getName(),
                submenu: [
                    { role: "about" },
                    { type: 'separator' },
                    { role: 'hide' },
                    { role: 'hideothers' },
                    { role: 'unhide' },
                    { type: 'separator' },
                    { role: "quit" }
                ]
            });
        }
        
        menuTemplate.push({
            label: '&File',
            submenu: [
                {
                    label: '&Open...',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => { this.openFileDialog(); }
                },
                {
                    label: 'Open &Recent',
                    submenu: menuRecentsTemplate
                },
                { type: 'separator' },
                { role: 'close' },
            ]
        });
        
        if (process.platform !== 'darwin') {
            menuTemplate.slice(-1)[0].submenu.push(
                { type: 'separator' },
                { role: 'quit' }
            );
        }
        
        if (process.platform == 'darwin') {
            electron.systemPreferences.setUserDefault('NSDisabledDictationMenuItem', 'boolean', true);
            electron.systemPreferences.setUserDefault('NSDisabledCharacterPaletteMenuItem', 'boolean', true);
        }

        menuTemplate.push({
            label: '&Edit',
            submenu: [
                { role: 'copy' }
            ]
        });

        if (process.platform === 'darwin') {
            menuTemplate.push({
                role: 'window',
                submenu: [
                    { role: 'minimize' },
                    { role: 'zoom' },
                    { type: 'separator' },
                    { role: 'front'}
                ]
            });
        }    
        
        menuTemplate.push({
            role: 'help',
            submenu: [
                {
                    label: '&Search Feature Requests',
                    click: () => { electron.shell.openExternal('https://www.github.com/lutzroeder/Netron/issues'); }
                },
                {
                    label: 'Report &Issues',
                    click: () => { electron.shell.openExternal('https://www.github.com/lutzroeder/Netron/issues/new'); }
                }
            ]
        });

        var menu = electron.Menu.buildFromTemplate(menuTemplate);
        electron.Menu.setApplicationMenu(menu);
    }

    static minimizePath(file) {
        var home = os.homedir();
        if (file.startsWith(home))
        {
            return '~' + file.substring(home.length);
        }
        return file;
    }

}

var application = new Application();