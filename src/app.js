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
        this._views = new ViewCollection();
        this._configuration = new ConfigurationService();
        this._openFileQueue = [];

        electron.app.setAppUserModelId('com.lutzroeder.netron');

        if (this.makeSingleInstance()) {
            electron.app.quit();
        }

        electron.ipcMain.on('open-file-dialog', (e, data) => {
            this.openFileDialog();
        });

        electron.ipcMain.on('drop-files', (e, data) => {
            this.dropFiles(e.sender, data.files);
        });

        electron.app.on('will-finish-launching', () => {
            electron.app.on('open-file', (e, path) => {
                this.openFile(path);
            });
        });

        electron.app.on('ready', () => {
            this.ready();
        });

        electron.app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                electron.app.quit();
            }
        });

        electron.app.on('will-quit', () => {
            this._configuration.save();
        });

        this.parseCommandLine(process.argv);
        this.checkForUpdates();
    }

    makeSingleInstance() {
        return electron.app.makeSingleInstance((argv, workingDirectory) => { 
            var currentDirectory = process.cwd();
            process.chdir(workingDirectory);
            var open = this.parseCommandLine(argv);
            process.chdir(currentDirectory);
            if (!open) {
                if (this._views.count > 0) {
                    var view = this._views.item(0);
                    if (view) {
                        view.restore();
                    }
                }
            }
        });
    }

    parseCommandLine(argv) {
        var open = false;
        if (process.platform == 'win32' && argv.length > 1) {
            argv.slice(1).forEach((arg) => {
                if (!arg.startsWith('-')) {
                    var extension = arg.split('.').pop();
                    if (extension != '' && extension != 'js' && fs.existsSync(arg) && fs.statSync(arg).isFile()) {
                        this.openFile(arg);
                        open = true;
                    }
                }
            });
        }
        return open;
    }

    ready() {
        this._configuration.load();
        if (this._openFileQueue) {
            var openFileQueue = this._openFileQueue;
            this._openFileQueue = null;
            while (openFileQueue.length > 0) {
                var file = openFileQueue.shift();
                this.openFile(file);
            }
        }
        if (this._views.count == 0) {
            this._views.openView();
        }
        this.resetMenu();
        this._views.on('active-view-changed', (e) => {
            this.resetMenu();
        });
        this._views.on('active-view-updated', (e) => {
            this.resetMenu();
        });
    }

    openFileDialog() {
        var showOpenDialogOptions = { 
            properties: [ 'openFile' ], 
            filters: [
                { name: 'ONNX Model', extension: [ 'onnx', 'pb' ] },
                { name: 'Keras Model', extension: [ 'json', 'keras', 'h5' ] },
                { name: 'CoreML Model', extension: [ 'mlmodel' ] },
                { name: 'Caffe Model', extension: [ 'caffemodel' ] },
                { name: 'Caffe2 Model', extension: [ 'pb' ] },
                { name: 'MXNet Model', extension: [ 'json' ] },
                { name: 'TensorFlow Graph', extension: [ 'pb', 'meta' ] },
                { name: 'TensorFlow Saved Model', extension: [ 'saved_model.pb' ] },
                { name: 'TensorFlow Lite Model', extension: [ 'tflite' ] }
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
            var view = this._views.find(file);
            // find empty welcome window
            if (view == null) {
                view = this._views.find(null);
            }
            // create new window
            if (view == null) {
                view = this._views.openView();
            }
            this.loadFile(file, view);
        }
    }

    loadFile(file, view) {
        var recents = this._configuration.get('recents');
        recents = recents.filter(recent => file != recent.path);
        view.open(file);
        recents.unshift({ path: file });
        if (recents.length > 9) {
            recents.splice(9);
        }
        this._configuration.set('recents', recents);
        this.resetMenu();
    }

    dropFiles(sender, files) {
        var view = this._views.from(sender);
        files.forEach((file) => {
            if (view) {
                this.loadFile(file, view);
                view = null;
            }
            else {
                this.openFile(file);
            }
        });
    }


    export() {
        var view = this._views.activeView;
        if (view && view.path) {
            var defaultPath = 'Untitled';
            var file = view.path;
            var lastIndex = file.lastIndexOf('.');
            if (lastIndex != -1) {
                defaultPath = file.substring(0, lastIndex);
            }
            var owner = electron.BrowserWindow.getFocusedWindow();
            var showSaveDialogOptions = {
                title: 'Export',
                defaultPath: defaultPath,
                buttonLabel: 'Export',
                filters: [
                    { name: 'PNG', extensions: [ 'png' ] },
                    { name: 'SVG', extensions: [ 'svg' ] }
                ]
            };
            electron.dialog.showSaveDialog(owner, showSaveDialogOptions, (filename) => {
                if (filename) {
                    view.execute('export', { 'file': filename });
                }
            });
        }
    }

    execute(command, data) {
        var view = this._views.activeView;
        if (view) {
            view.execute(command, data || {});
        }
    }

    reload() {
        var view = this._views.activeView;
        if (view && view.path) {
            this.loadFile(view.path, view);
        }
    }

    toggleDevTools() {
        if (this.isDev()) {
            var window = electron.BrowserWindow.getFocusedWindow();
            if (window) {
                window.toggleDevTools();
            }
        }
    }

    checkForUpdates() {
        if (this.isDev()) {
            return;
        }
        var autoUpdater = updater.autoUpdater;
        autoUpdater.autoDownload = false;
        autoUpdater.on('update-available', (info) => {
            var owner = electron.BrowserWindow.getFocusedWindow();
            var messageBoxOptions = {
                icon: path.join(__dirname, 'icon.png'),
                title: ' ',
                message: 'A new version of ' + electron.app.getName() + ' is available.',
                detail: 'Click \'Download and Install\' to download the update and automatically install it on exit.',
                buttons: ['Download and Install', 'Remind Me Later'],
                defaultId: 0,
                cancelId: 1
            };
            var result = electron.dialog.showMessageBox(owner, messageBoxOptions);
            if (result == 0) {
                autoUpdater.autoDownload = true;
                autoUpdater.checkForUpdatesAndNotify();
            }
        });
        autoUpdater.checkForUpdates();
    }

    get package() { 
        if (!this._package) {
            var appPath = electron.app.getAppPath();
            var file = appPath + '/package.json'; 
            var data = fs.readFileSync(file);
            this._package = JSON.parse(data);
            this._package.date = new Date(fs.statSync(file).mtime);
        }
        return this._package;
    }

    about() {
        var owner = electron.BrowserWindow.getFocusedWindow();
        var author = this.package.author;
        var date = this.package.date;
        var details = [];
        details.push('Version ' + electron.app.getVersion());
        if (author && author.name && date) {
            details.push('');
            details.push('Copyright \u00A9 ' + date.getFullYear().toString() + ' ' + author.name);
        }
        var aboutDialogOptions = {
            icon: path.join(__dirname, 'icon.png'),
            title: ' ',
            message: electron.app.getName(),
            detail: details.join('\n')
        };
        electron.dialog.showMessageBox(owner, aboutDialogOptions);
    }

    isDev() {
        return ('ELECTRON_IS_DEV' in process.env) ?
            (parseInt(process.env.ELECTRON_IS_DEV, 10) === 1) :
            (process.defaultApp || /node_modules[\\/]electron[\\/]/.test(process.execPath));
    }

    resetMenu() {

        var view = this._views.activeView;

        var menuRecentsTemplate = [];
        if (this._configuration.has('recents')) {
            var recents = this._configuration.get('recents');
            recents = recents.filter(recent => fs.existsSync(recent.path) && fs.statSync(recent.path).isFile());
            if (recents.length > 9) {
                recents.splice(9);
            }
            this._configuration.set('recents', recents);
            recents.forEach((recent, index) => {
                var file = recent.path;
                menuRecentsTemplate.push({
                    label: Application.minimizePath(recent.path),
                    accelerator: ((process.platform === 'darwin') ? 'Cmd+' : 'Ctrl+') + (index + 1).toString(),
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
                { 
                    label: '&Export...',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => this.export(),
                    enabled: view && view.path ? true : false
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
                { 
                    label: '&Copy',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+C' : 'Ctrl+C',
                    click: () => this.execute('copy', null),
                    enabled: view && view.path ? true : false
                },
                /* { type: 'separator' },
                {
                    label: '&Find...',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+F' : 'Ctrl+F',
                    click: () => this.execute('find', null),
                    enabled: view && view.path ? true : false
                } */
            ]
        });
    
        var viewTemplate = {
            label: '&View',
            submenu: [
                {
                    label: !view || !view.get('show-details') ?  'Show &Details' : 'Hide &Details',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+D' : 'Ctrl+D',
                    click: () => this.execute('toggle-details', null),
                    enabled: view && view.path ? true : false
                },
                {
                    label: !view || !view.get('show-names') ?  'Show &Names' : 'Hide &Names',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+U' : 'Ctrl+U',
                    click: () => this.execute('toggle-names', null),
                    enabled: view && view.path ? true : false
                },
                { type: 'separator' },
                {
                    label: '&Reload',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+R' : 'F5',
                    click: () => this.reload(),
                    enabled: view && view.path ? true : false
                },
                { type: 'separator' },
                {
                    label: 'Actual &Size',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+Backspace' : 'Ctrl+Backspace',
                    click: () => this.execute('reset-zoom', null),
                    enabled: view && view.path ? true : false
                },
                {
                    label: 'Zoom &In',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+Up' : 'Ctrl+Down',
                    click: () => this.execute('zoom-in', null),
                    enabled: view && view.path ? true : false
                },
                {
                    label: 'Zoom &Out',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+Down' : 'Ctrl+Down',
                    click: () => this.execute('zoom-out', null),
                    enabled: view && view.path ? true : false
                },
                { type: 'separator' },
                {
                    label: '&Properties...',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+Enter' : 'Ctrl+Enter',
                    click: () => this.execute('show-properties', null),
                    enabled: view && view.path ? true : false
                }        
            ]
        };
        if (this.isDev()) {
            viewTemplate.submenu.push({ type: 'separator' });
            viewTemplate.submenu.push({ role: 'toggledevtools' });
        }
        menuTemplate.push(viewTemplate);

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

        var helpSubmenu = [
            {
                label: '&Search Feature Requests',
                click: () => { electron.shell.openExternal('https://www.github.com/' + this.package.repository + '/issues'); }
            },
            {
                label: 'Report &Issues',
                click: () => { electron.shell.openExternal('https://www.github.com/' + this.package.repository + '/issues/new'); }
            }
        ];

        if (process.platform != 'darwin') {
            helpSubmenu.push({ type: 'separator' });
            helpSubmenu.push({
                role: 'about',
                click: () => this.about()
            });
        }

        menuTemplate.push({
            role: 'help',
            submenu: helpSubmenu
        });

        var menu = electron.Menu.buildFromTemplate(menuTemplate);
        electron.Menu.setApplicationMenu(menu);
    }

    static minimizePath(file) {
        if (process.platform != 'win32') {
            var home = os.homedir();
            if (file.startsWith(home))
            {
                return '~' + file.substring(home.length);
            }
        }
        return file;
    }

}

class View {

    constructor(owner) {
        this._owner = owner;
        this._ready = false;
        this._path = null;
        this._properties = {};

        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        var options = {};
        options.title = electron.app.getName(); 
        options.backgroundColor = '#eeeeee';
        options.icon = electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png'));
        options.minWidth = 600;
        options.minHeight = 400;
        options.width = size.width;
        options.height = size.height;
        if (options.width > 1024) {
            options.width = 1024;
        }
        if (options.height > 768) {
            options.height = 768;
        }
        if (this._owner.count > 0 && View._position && View._position.length == 2) {
            options.x = View._position[0] + 30;
            options.y = View._position[1] + 30;
            if (options.x + options.width > size.width) {
                options.x = 0;
            }
            if (options.y + options.height > size.height) {
                options.y = 0;
            }
        }
        this._window = new electron.BrowserWindow(options);
        View._position = this._window.getPosition();
        this._updateCallback = (e, data) => { 
            if (e.sender == this._window.webContents) {
                this.update(data.name, data.value); 
                this.raise('updated');
            }
        };
        electron.ipcMain.on('update', this._updateCallback);
        this._window.on('closed', () => {
            electron.ipcMain.removeListener('update', this._updateCallback);
            this._owner.closeView(this);
        });
        this._window.on('focus', (e) => {
            this.raise('activated');
        });
        this._window.on('blur', (e) => {
            this.raise('deactivated');
        });
        this._window.webContents.on('dom-ready', () => {
            this._ready = true;
        });
        var location = url.format({
            pathname: path.join(__dirname, 'view-electron.html'),
            protocol: 'file:',
            slashes: true
        });
        this._window.loadURL(location);
    }

    get window() {
        return this._window;
    }

    get path() {
        return this._path;
    }

    open(file) {
        this._openPath = file;
        if (this._ready) {
            this._window.webContents.send("open", { file: file });
        }
        else {
            this._window.webContents.on('dom-ready', () => {
                this._window.webContents.send("open", { file: file });
            });
            var location = url.format({
                pathname: path.join(__dirname, 'view-electron.html'),
                protocol: 'file:',
                slashes: true
            });
            this._window.loadURL(location);
        }
    }

    restore() {
        if (this._window) { 
            if (this._window.isMinimized()) {
                this._window.restore();
            }
            this._window.show();
        }
    }

    match(path) {
        if (this._openPath) {
            if (path == null) {
                return false;
            }
            if (path == this._openPath) {
                return true;
            }
        }
        return (this._path == path);
    }

    execute(command, data) {
        if (this._window && this._window.webContents) {
            this._window.webContents.send(command, data);
        }
    }

    update(name, value) {
        switch (name) {
            case 'path':
                if (value) {
                    this._path = value;
                    var title = Application.minimizePath(this._path);
                    if (process.platform !== 'darwin') {
                        title = title + ' - ' + electron.app.getName();
                    }
                    this._window.setTitle(title);
                    this._window.focus();
                }
                this._openPath = null;
                break;
            default:
                this._properties[name] = value;
                break;
        }
    }

    get(name) {
        return this._properties[name];
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    raise(event, data) {
        if (this._events && this._events[event]) {
            this._events[event].forEach((callback) => {
                callback(this, data);
            });
        }
    }
}

class ViewCollection {
    constructor() {
        this._views = [];
    }

    get count() {
        return this._views.length;
    }

    item(index) {
        return this._views[index];
    }

    openView() {
        var view = new View(this);
        view.on('activated', (sender) => {
            this._activeView = sender;
            this.raise('active-view-changed', { activeView: this._activeView });
        });
        view.on('updated', (sender) => {
            this.raise('active-view-updated', { activeView: this._activeView });            
        });
        view.on('deactivated', (sender) => {
            this._activeView = null;
            this.raise('active-view-changed', { activeView: this._activeView });
        });
        this._views.push(view);
        this.updateActiveView();
        return view;
    }

    closeView(view) {
        for (var i = this._views.length - 1; i >= 0; i--) {
            if (this._views[i] == view) {
                this._views.splice(i, 1);
            }
        }
        this.updateActiveView();
    }

    find(path) {
        return this._views.find(view => view.match(path));
    }

    from(contents) {
        return this._views.find(view => view && view.window && view.window.webContents && view.window.webContents == contents);
    }

    updateActiveView() {
        var window = electron.BrowserWindow.getFocusedWindow();
        var view = this._views.find(view => view.window == window) || null;
        if (view != this._activeView) {
            this._activeView = view;
            this.raise('active-view-changed', { activeView: this._activeView });        
        }
    }

    get activeView() {
        return this._activeView;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    raise(event, data) {
        if (this._events && this._events[event]) {
            this._events[event].forEach((callback) => {
                callback(this, data);
            });
        }
    }
}

class ConfigurationService {

    load() {
        var dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            var file = path.join(dir, 'configuration.json'); 
            if (fs.existsSync(file)) {
                var data = fs.readFileSync(file);
                if (data) {
                    this._data = JSON.parse(data);
                }
            }
        }
        if (!this._data) {
            this._data = {
                'recents': []
            };
        }
    }

    save() {
        if (this._data) {
            var data = JSON.stringify(this._data);
            if (data) {
                var dir = electron.app.getPath('userData');
                if (dir && dir.length > 0) {
                    var file = path.join(dir, 'configuration.json'); 
                    fs.writeFileSync(file, data);          
                }
            }
        }
    }

    has(name) {
        return this._data && this._data.hasOwnProperty(name);
    }

    set(name, value) {
        this._data[name] = value;
    }

    get(name) {
        return this._data[name];
    }

}

var application = new Application();
