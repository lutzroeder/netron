
const electron = require('electron')
const updater = require('electron-updater')
const fs = require('fs')
const os = require('os')
const path = require('path')
const process = require('process')
const url = require('url')

electron.app.setAppUserModelId('com.lutzroeder.netron');

var views = []

const quit = electron.app.makeSingleInstance(function() {
    if (views.length > 0) {
        var view = views[0];
        if (view) { 
            if (view.isMinimized()) {
                view.restore();
            }
            view.show();
        }
    }
});

if (quit) {
    electron.app.quit();
}

function openFile() {
    var showOpenDialogOptions = { 
        properties: [ 'openFile'], 
        filters: [ { name: 'ONNX Model', extensions: ['pb', 'onnx'] } ]
    };
    electron.dialog.showOpenDialog(showOpenDialogOptions, function(selectedFiles) {
        if (selectedFiles) {
            selectedFiles.forEach(function(selectedFile) {
                openFileLocation(selectedFile);
            });
        }
    });
};

function openFileLocation(file) {
    configuration['recents'] = configuration['recents'].filter(recent => file != recent['path']);
    if (file && file.length > 0 && fs.existsSync(file))
    {
        var appName = electron.app.getName();
        var view = null;
        views.forEach(function (item) {
            if (item.getTitle() == appName) {
                view = item;
            }
        });
        if (view == null) {
            view = openView();
        }
        var title = minimizePath(file);
        if (process.platform !== 'darwin') {
            title = file + ' - ' + appName;
        }
        view.setTitle(title);
        if (view.__isReady) {
            view.webContents.send("open-file", { file: file });
        }
        else {
            view.webContents.on('dom-ready', function() {
                view.webContents.send("open-file", { file: file });
            });
            var location = url.format({
                pathname: path.join(__dirname, 'view.html'),
                protocol: 'file:',
                slashes: true
            });
            view.loadURL(location);
        }
        configuration['recents'].unshift({ 'path': file });
        if (configuration['recents'].length > 10) {
            configuration['recents'].slice(0, 10);
        }
    }
    updateMenu();
}

electron.ipcMain.on('open-file', function(e, data) {
    openFile();
});

Array.prototype.remove = function(obj) {
    var index = this.length;
    while (index--) {
        if (this[index] == obj) {
            this.splice(index, 1);
        }
    }
}

function openView() {
    const title = electron.app.getName();
    const size = electron.screen.getPrimaryDisplay().workAreaSize;
    if (size.width > 1024) {
        size.width = 1024;
    }
    if (size.height > 768) {
        size.height = 768;
    }
    var view = new electron.BrowserWindow({ 
        title: title,
        // backgroundColor: '#f0fcfe',
        backgroundColor: '#eeeeee',
        minWidth: 600,
        minHeight: 400,
        width: size.width,
        height: size.height,
        icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png'))
    });
    
    view.on('closed', function () {
        views.remove(view);
    });
    view.webContents.on('dom-ready', function() {
        view.__isReady = true;
    });        
    view.loadURL(url.format({
        pathname: path.join(__dirname, 'view.html'),
        protocol: 'file:',
        slashes: true
    }));
    views.push(view);
    return view;
}

update();

electron.app.on('will-finish-launching', function() {
    electron.app.on('open-file', function(e, path) {
        if (openFileQueue) {
            openFileQueue.push(path);
        }
        else {
            openFileLocation(path);
        }
    });
});

var openFileQueue = [];

electron.app.on('ready', function () {
    updateMenu();
    loadConfiguration();

    while (openFileQueue.length > 0) {
        var file = openFileQueue.shift();
        openFileLocation(file);
    }
    openFileQueue = null;

    if (views.length == 0) {
        openView();
    }
});

electron.app.on('window-all-closed', function () {
    if (process.platform !== 'darwin') {
        electron.app.quit();
    }
});

electron.app.on('will-quit', function() {
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
    if (configuration && configuration['recents']) {
        configuration['recents'].forEach(function(recent) {
            var file = recent.path;
            menuRecentsTemplate.push({ 
                label: minimizePath(recent.path),
                click: function() { openFileLocation(file) }
            });
        })
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
                click: function() { openFile(); }
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
        menuTemplate.slice(-1)[0]['submenu'].push(
            { type: 'separator' },
            { role: 'quit' }
        );
    }
    
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
                click() { electron.shell.openExternal('https://www.github.com/lutzroeder/Netron/issues') }
            },
            {
                label: 'Report &Issues',
                click() { electron.shell.openExternal('https://www.github.com/lutzroeder/Netron/issues/new') }
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
