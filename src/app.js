/*jshint esversion: 6 */

const electron = require('electron');
const updater = require('electron-updater');
const fs = require('fs');
const os = require('os');
const path = require('path');
const process = require('process');
const url = require('url');

electron.app.setAppUserModelId('com.lutzroeder.netron');

var views = [];

const quit = electron.app.makeSingleInstance(() => {
    if (views.length > 0) {
        var view = views[0];
        if (view && view.window) { 
            if (view.window.isMinimized()) {
                view.window.restore();
            }
            view.window.show();
        }
    }
});

if (quit) {
    electron.app.quit();
}

function openFileDialog() {
    var showOpenDialogOptions = { 
        properties: [ 'openFile' ], 
        filters: [
            { name: 'ONNX Model', extensions: [ 'onnx', 'pb' ] },
            { name: 'TensorFlow Saved Model', extensions: [ 'saved_model.pb' ] },
            { name: 'TensorFlow Graph', extensions: [ 'pb' ] },
            { name: 'TensorFlow Lite Model', extensions: [ 'tflite' ]}
        ]
    };
    electron.dialog.showOpenDialog(showOpenDialogOptions, (selectedFiles) => {
        if (selectedFiles) {
            selectedFiles.forEach((selectedFile) => {
                openFile(selectedFile);
            });
        }
    });
}

function openFile(file) {
    if (file && file.length > 0 && fs.existsSync(file))
    {
        // find existing view for this file
        var view = views.find(view => view.path && view.path == file);
        // find empty welcome window
        if (view == null) {
            view = views.find(view => !view.path || view.path.length == 0);
        }
        // create new window
        if (view == null) {
            view = openView();
        }
        loadFile(file, view);
    }
}

function loadFile(file, view) {
    configuration.recents = configuration.recents.filter(recent => file != recent.path);
    var title = minimizePath(file);
    if (process.platform !== 'darwin') {
        title = file + ' - ' + electron.app.getName();
    }
    var window = view.window;
    window.setTitle(title);
    view.path = file;
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
    configuration.recents.unshift({ path: file });
    if (configuration.recents.length > 10) {
        configuration.recents.splice(10);
    }
    updateMenu();
}

electron.ipcMain.on('open-file-dialog', (e, data) => {
    openFileDialog();
});

electron.ipcMain.on('open-file', (e, data) => {
    var file = data.file;
    if (file) { 
        var view = null;
        if (data.window) {
            var window = electron.BrowserWindow.fromId(data.window);
            if (window) {
                view = views.find(view => view.window == window);
            }
        }
        if (view) {
            loadFile(file, view);
        }
        else {
            openFile(file);
        }
    }
});

Array.prototype.remove = function(obj) {
    var index = this.length;
    while (index--) {
        if (this[index] == obj) {
            this.splice(index, 1);
        }
    }
};

function openView() {
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
        // backgroundColor: '#f0fcfe',
        backgroundColor: '#eeeeee',
        minWidth: 600,
        minHeight: 400,
        width: size.width,
        height: size.height,
        icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png'))
    });
    
    window.on('closed', function () {
        for (var i = views.length - 1; i >= 0; i--) {
            if (views[i].window == window) {
                views.splice(i, 1);
            }   
        }
    });
    var view = { 
        window: window,
        ready: false
    };
    window.webContents.on('dom-ready', function() {
        view.ready = true;
    });        
    window.loadURL(url.format({
        pathname: path.join(__dirname, 'view-electron.html'),
        protocol: 'file:',
        slashes: true
    }));
    views.push(view);
    return view;
}

update();

electron.app.on('will-finish-launching', () => {
    electron.app.on('open-file', (e, path) => {
        if (openFileQueue) {
            openFileQueue.push(path);
        }
        else {
            openFile(path);
        }
    });
});

var openFileQueue = [];

electron.app.on('ready', () => {
    loadConfiguration();
    updateMenu();
    
    while (openFileQueue.length > 0) {
        var file = openFileQueue.shift();
        openFile(file);
    }
    openFileQueue = null;

    if (views.length == 0) {
        openView();
    }
});

electron.app.on('window-all-closed', () => {
    if (process.platform !== 'darwin') {
        electron.app.quit();
    }
});

electron.app.on('will-quit', () => {
    saveConfiguration();
});

function update() {
    var isDev = ('ELECTRON_IS_DEV' in process.env) ?
        (parseInt(process.env.ELECTRON_IS_DEV, 10) === 1) :
        (process.defaultApp || /node_modules[\\/]electron[\\/]/.test(process.execPath));
    if (!isDev) {
        updater.autoUpdater.checkForUpdatesAndNotify();
    }
}

var configuration = null;

function loadConfiguration() {
    var dir = electron.app.getPath('userData');
    if (dir && dir.length > 0) {
        var file = path.join(dir, 'configuration.json'); 
        if (fs.existsSync(file)) {
            var data = fs.readFileSync(file);
            if (data) {
                configuration = JSON.parse(data);
            }
        }
    }
    if (!configuration) {
        configuration = {
            'recents': []
        };
    }
}

function saveConfiguration() {
    if (configuration) {
        var data = JSON.stringify(configuration);
        if (data) {
            var dir = electron.app.getPath('userData');
            if (dir && dir.length > 0) {
                var file = path.join(dir, 'configuration.json'); 
                fs.writeFileSync(file, data);          
            }
        }
    }
}

function updateMenu() {

    var menuRecentsTemplate = [];
    if (configuration && configuration.recents) {
        configuration.recents = configuration.recents.filter(recent => fs.existsSync(recent.path));
        configuration.recents.forEach((recent) => {
            var file = recent.path;
            menuRecentsTemplate.push({ 
                label: minimizePath(recent.path),
                click: () => { openFile(file); }
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
                click: () => { openFileDialog(); }
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

function minimizePath(file) {
    var home = os.homedir();
    if (file.startsWith(home))
    {
        return '~' + file.substring(home.length);
    }
    return file;
}
