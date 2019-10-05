/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

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
        this._menu = new MenuService();
        this._openFileQueue = [];

        electron.app.setAppUserModelId('com.lutzroeder.netron');

        if (!electron.app.requestSingleInstanceLock()) {
            electron.app.quit();
        }

        electron.app.on('second-instance', (event, commandLine, workingDirectory) => {
            const currentDirectory = process.cwd();
            process.chdir(workingDirectory);
            const open = this._parseCommandLine(commandLine);
            process.chdir(currentDirectory);
            if (!open) {
                if (this._views.count > 0) {
                    const view = this._views.item(0);
                    if (view) {
                        view.restore();
                    }
                }
            }
        });

        electron.ipcMain.on('open-file-dialog', () => {
            this._openFileDialog();
        });

        electron.ipcMain.on('drop-files', (e, data) => {
            const files = data.files.filter((file) => fs.statSync(file).isFile());
            this._dropFiles(e.sender, files);
        });

        electron.app.on('will-finish-launching', () => {
            electron.app.on('open-file', (e, path) => {
                this._openFile(path);
            });
        });

        electron.app.on('ready', () => {
            this._ready();
        });

        electron.app.on('window-all-closed', () => {
            if (process.platform !== 'darwin') {
                electron.app.quit();
            }
        });

        electron.app.on('will-quit', () => {
            this._configuration.save();
        });

        this._parseCommandLine(process.argv);
        this._checkForUpdates();
    }

    _parseCommandLine(argv) {
        let open = false;
        if (argv.length > 1) {
            for (let arg of argv.slice(1)) {
                if (!arg.startsWith('-')) {
                    const extension = arg.split('.').pop().toLowerCase();
                    if (extension != '' && extension != 'js' && fs.existsSync(arg) && fs.statSync(arg).isFile()) {
                        this._openFile(arg);
                        open = true;
                    }
                }
            }
        }
        return open;
    }

    _ready() {
        this._configuration.load();
        if (!this._configuration.has('userId')) {
            this._configuration.set('userId', require('uuid').v4());
        }
        global.userId = this._configuration.get('userId');
        if (this._openFileQueue) {
            let openFileQueue = this._openFileQueue;
            this._openFileQueue = null;
            while (openFileQueue.length > 0) {
                const file = openFileQueue.shift();
                this._openFile(file);
            }
        }
        if (this._views.count == 0) {
            this._views.openView();
        }
        this._resetMenu();
        this._views.on('active-view-changed', () => {
            this._updateMenu();
        });
        this._views.on('active-view-updated', () => {
            this._updateMenu();
        });
    }

    _openFileDialog() {
        const showOpenDialogOptions = { 
            properties: [ 'openFile' ], 
            filters: [
                { name: 'All Model Files',  extensions: [ 
                    'onnx', 'pb',
                    'h5', 'hd5', 'hdf5', 'json', 'keras',
                    'mlmodel',
                    'caffemodel',
                    'model', 'dnn', 'cmf',
                    'mar', 'params',
                    'meta',
                    'tflite', 'lite', 'tfl', 'bin',
                    'param', 'ncnn',
                    'pt', 'pth', 't7',
                    'pkl', 'joblib',
                    'pbtxt', 'prototxt',
                    'cfg',
                    'xml',
                    'mnn' ] }
            ]
        };
        electron.dialog.showOpenDialog(showOpenDialogOptions, (selectedFiles) => {
            if (selectedFiles) {
                for (let selectedFile of selectedFiles) {
                    this._openFile(selectedFile);
                }
            }
        });
    }

    _openFile(file) {
        if (this._openFileQueue) {
            this._openFileQueue.push(file);
            return;
        }
        if (file && file.length > 0 && fs.existsSync(file) && fs.statSync(file).isFile()) {
            // find existing view for this file
            let view = this._views.find(file);
            // find empty welcome window
            if (view == null) {
                view = this._views.find(null);
            }
            // create new window
            if (view == null) {
                view = this._views.openView();
            }
            this._loadFile(file, view);
        }
    }

    _loadFile(file, view) {
        const recents = this._configuration.get('recents').filter(recent => file != recent.path);
        view.open(file);
        recents.unshift({ path: file });
        if (recents.length > 9) {
            recents.splice(9);
        }
        this._configuration.set('recents', recents);
        this._resetMenu();
    }

    _dropFiles(sender, files) {
        let view = this._views.from(sender);
        for (let file of files) {
            if (view) {
                this._loadFile(file, view);
                view = null;
            }
            else {
                this._openFile(file);
            }
        }
    }

    _export() {
        let view = this._views.activeView;
        if (view && view.path) {
            let defaultPath = 'Untitled';
            const file = view.path;
            const lastIndex = file.lastIndexOf('.');
            if (lastIndex != -1) {
                defaultPath = file.substring(0, lastIndex);
            }
            const owner = electron.BrowserWindow.getFocusedWindow();
            const showSaveDialogOptions = {
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
        const view = this._views.activeView;
        if (view) {
            view.execute(command, data || {});
        }
        this._updateMenu();
    }

    _reload() {
        const view = this._views.activeView;
        if (view && view.path) {
            this._loadFile(view.path, view);
        }
    }

    _checkForUpdates() {
        if (!electron.app.isPackaged) {
            return;
        }
        const autoUpdater = updater.autoUpdater;
        const promise = autoUpdater.checkForUpdates();
        if (promise) {
            promise.catch((error) => {
                console.log(error.message);
            });
        }
    }

    get package() { 
        if (!this._package) {
            const appPath = electron.app.getAppPath();
            const file = appPath + '/package.json'; 
            const data = fs.readFileSync(file);
            this._package = JSON.parse(data);
            this._package.date = new Date(fs.statSync(file).mtime);
        }
        return this._package;
    }

    _about() {
        const owner = electron.BrowserWindow.getFocusedWindow();
        const author = this.package.author;
        const date = this.package.date;
        let details = [];
        details.push('Version ' + electron.app.getVersion());
        if (author && author.name && date) {
            details.push('');
            details.push('Copyright \u00A9 ' + date.getFullYear().toString() + ' ' + author.name);
        }
        let aboutDialogOptions = {
            icon: path.join(__dirname, 'icon.png'),
            title: ' ',
            message: electron.app.getName(),
            detail: details.join('\n')
        };
        electron.dialog.showMessageBoxSync(owner, aboutDialogOptions);
    }

    _updateMenu() {
        const window = electron.BrowserWindow.getFocusedWindow();
        this._menu.update({
            window: window,
            webContents: window ? window.webContents : null,
            view: this._views.activeView
        });
    }

    _resetMenu() {
        let menuRecentsTemplate = [];
        if (this._configuration.has('recents')) {
            let recents = this._configuration.get('recents');
            recents = recents.filter(recent => fs.existsSync(recent.path) && fs.statSync(recent.path).isFile());
            if (recents.length > 9) {
                recents.splice(9);
            }
            this._configuration.set('recents', recents);
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                menuRecentsTemplate.push({
                    file: recent.path,
                    label: Application.minimizePath(recent.path),
                    accelerator: ((process.platform === 'darwin') ? 'Cmd+' : 'Ctrl+') + (i + 1).toString(),
                    click: (item) => { this._openFile(item.file); }
                });
            }
        }

        let menuTemplate = [];
        
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
                    click: () => { this._openFileDialog(); }
                },
                {
                    label: 'Open &Recent',
                    submenu: menuRecentsTemplate
                },
                { type: 'separator' },
                { 
                    id: 'file.export',
                    label: '&Export...',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => this._export(),
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
                    id: 'edit.cut',
                    label: 'Cu&t',
                    accelerator: 'CmdOrCtrl+X',
                    click: () => this.execute('cut', null),
                },
                {
                    id: 'edit.copy',
                    label: '&Copy',
                    accelerator: 'CmdOrCtrl+C',
                    click: () => this.execute('copy', null),
                },
                {
                    id: 'edit.paste',
                    label: '&Paste',
                    accelerator: 'CmdOrCtrl+V',
                    click: () => this.execute('paste', null),
                },
                {
                    id: 'edit.select-all',
                    label: 'Select &All',
                    accelerator: 'CmdOrCtrl+A',
                    click: () => this.execute('selectall', null),
                },
                { type: 'separator' },
                {
                    id: 'edit.find',
                    label: '&Find...',
                    accelerator: 'CmdOrCtrl+F',
                    click: () => this.execute('find', null),
                }
            ]
        });
    
        const viewTemplate = {
            label: '&View',
            submenu: [
                {
                    id: 'view.show-attributes',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => this.execute('toggle-attributes', null),
                },
                {
                    id: 'view.show-initializers',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => this.execute('toggle-initializers', null),
                },
                {
                    id: 'view.show-names',
                    accelerator: 'CmdOrCtrl+U',
                    click: () => this.execute('toggle-names', null),
                },
                { type: 'separator' },
                {
                    id: 'view.reload',
                    label: '&Reload',
                    accelerator: (process.platform === 'darwin') ? 'Cmd+R' : 'F5',
                    click: () => this._reload(),
                },
                { type: 'separator' },
                {
                    id: 'view.reset-zoom',
                    label: 'Actual &Size',
                    accelerator: 'Shift+Backspace',
                    click: () => this.execute('reset-zoom', null),
                },
                {
                    id: 'view.zoom-in',
                    label: 'Zoom &In',
                    accelerator: 'Shift+Up',
                    click: () => this.execute('zoom-in', null),
                },
                {
                    id: 'view.zoom-out',
                    label: 'Zoom &Out',
                    accelerator: 'Shift+Down',
                    click: () => this.execute('zoom-out', null),
                },
                { type: 'separator' },
                {
                    id: 'view.show-properties',
                    label: '&Properties...',
                    accelerator: 'CmdOrCtrl+Enter',
                    click: () => this.execute('show-properties', null),
                }
            ]
        };
        if (!electron.app.isPackaged) {
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

        const helpSubmenu = [
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
                click: () => this._about()
            });
        }

        menuTemplate.push({
            role: 'help',
            submenu: helpSubmenu
        });

        const commandTable = new Map();
        commandTable.set('file.export', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.cut', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.copy', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.paste', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.select-all', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('edit.find', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.show-attributes', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-attributes') ? 'Show &Attributes' : 'Hide &Attributes'; }
        });
        commandTable.set('view.show-initializers', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-initializers') ? 'Show &Initializers' : 'Hide &Initializers'; }
        });
        commandTable.set('view.show-names', {
            enabled: (context) => { return context.view && context.view.path ? true : false; },
            label: (context) => { return !context.view || !context.view.get('show-names') ? 'Show &Names' : 'Hide &Names'; }
        });
        commandTable.set('view.reload', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.reset-zoom', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.zoom-in', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.zoom-out', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });
        commandTable.set('view.show-properties', {
            enabled: (context) => { return context.view && context.view.path ? true : false; }
        });

        this._menu.build(menuTemplate, commandTable);
        this._updateMenu();
    }

    static minimizePath(file) {
        if (process.platform != 'win32') {
            const homeDir = os.homedir();
            if (file.startsWith(homeDir)) {
                return '~' + file.substring(homeDir.length);
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
        let options = {};
        options.title = electron.app.getName(); 
        options.backgroundColor = electron.systemPreferences.isDarkMode() ? '#1d1d1d' : '#e6e6e6';
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
        options.webPreferences = { nodeIntegration: true };
        this._window = new electron.BrowserWindow(options);
        View._position = this._window.getPosition();
        this._updateCallback = (e, data) => { 
            if (e.sender == this._window.webContents) {
                this.update(data.name, data.value); 
                this._raise('updated');
            }
        };
        electron.ipcMain.on('update', this._updateCallback);
        this._window.on('closed', () => {
            electron.ipcMain.removeListener('update', this._updateCallback);
            this._owner.closeView(this);
        });
        this._window.on('focus', () => {
            this._raise('activated');
        });
        this._window.on('blur', () => {
            this._raise('deactivated');
        });
        this._window.webContents.on('dom-ready', () => {
            this._ready = true;
        });
        this._window.webContents.on('new-window', (event, url) => {
            if (url.startsWith('http://') || url.startsWith('https://')) {
                event.preventDefault();
                electron.shell.openExternal(url);
            }
        });
        const location = url.format({
            pathname: path.join(__dirname, 'electron.html'),
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
            const location = url.format({
                pathname: path.join(__dirname, 'electron.html'),
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
        if (name === 'path') {
            if (value) {
                this._path = value;
                let title = Application.minimizePath(this._path);
                if (process.platform !== 'darwin') {
                    title = title + ' - ' + electron.app.getName();
                }
                this._window.setTitle(title);
                this._window.focus();
            }
            this._openPath = null;
            return;
        }
        this._properties[name] = value;
    }

    get(name) {
        return this._properties[name];
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (let callback of this._events[event]) {
                callback(this, data);
            }
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
        const view = new View(this);
        view.on('activated', (sender) => {
            this._activeView = sender;
            this._raise('active-view-changed', { activeView: this._activeView });
        });
        view.on('updated', () => {
            this._raise('active-view-updated', { activeView: this._activeView });
        });
        view.on('deactivated', () => {
            this._activeView = null;
            this._raise('active-view-changed', { activeView: this._activeView });
        });
        this._views.push(view);
        this._updateActiveView();
        return view;
    }

    closeView(view) {
        for (let i = this._views.length - 1; i >= 0; i--) {
            if (this._views[i] == view) {
                this._views.splice(i, 1);
            }
        }
        this._updateActiveView();
    }

    find(path) {
        return this._views.find(view => view.match(path));
    }

    from(contents) {
        return this._views.find(view => view && view.window && view.window.webContents && view.window.webContents == contents);
    }

    get activeView() {
        return this._activeView;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (let callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    _updateActiveView() {
        let window = electron.BrowserWindow.getFocusedWindow();
        let view = this._views.find(view => view.window == window) || null;
        if (view != this._activeView) {
            this._activeView = view;
            this._raise('active-view-changed', { activeView: this._activeView });
        }
    }
}

class ConfigurationService {

    load() {
        this._data = { 'recents': [] };
        const dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            let file = path.join(dir, 'configuration.json'); 
            if (fs.existsSync(file)) {
                let data = fs.readFileSync(file);
                if (data) {
                    try {
                        this._data = JSON.parse(data);
                    }
                    catch (error) {
                        // continue regardless of error
                    }
                }
            }
        }
    }

    save() {
        if (this._data) {
            const data = JSON.stringify(this._data);
            if (data) {
                let dir = electron.app.getPath('userData');
                if (dir && dir.length > 0) {
                    let file = path.join(dir, 'configuration.json'); 
                    fs.writeFileSync(file, data);
                }
            }
        }
    }

    has(name) {
        return this._data && Object.prototype.hasOwnProperty.call(this._data, name);
    }

    set(name, value) {
        this._data[name] = value;
    }

    get(name) {
        return this._data[name];
    }

}

class MenuService {

    build(menuTemplate, commandTable) {
        this._menuTemplate = menuTemplate;
        this._commandTable = commandTable;
        this._itemTable = new Map();
        for (let menu of menuTemplate) {
            for (let item of menu.submenu) {
                if (item.id) {
                    if (!item.label) {
                        item.label = '';
                    }
                    this._itemTable.set(item.id, item);
                }
            }
        }
        this._rebuild();
    }

    update(context) {
        if (!this._menu && !this._commandTable) {
            return;
        }
        if (this._updateLabel(context)) {
            this._rebuild();
        }
        this._updateEnabled(context);
    }

    _rebuild() {
        this._menu = electron.Menu.buildFromTemplate(this._menuTemplate);
        electron.Menu.setApplicationMenu(this._menu);
    }

    _updateLabel(context) {
        let rebuild = false;
        for (let entry of this._commandTable.entries()) {
            const menuItem = this._menu.getMenuItemById(entry[0]);
            const command = entry[1];
            if (command && command.label) {
                let label = command.label(context);
                if (label != menuItem.label) {
                    if (this._itemTable.has(entry[0])) {
                        this._itemTable.get(entry[0]).label = label;
                        rebuild = true;
                    }
                }
            }
        }
        return rebuild;
    }

    _updateEnabled(context) {
        for (let entry of this._commandTable.entries()) {
            const menuItem = this._menu.getMenuItemById(entry[0]);
            if (menuItem) {
                let command = entry[1];
                if (command.enabled) {
                    menuItem.enabled = command.enabled(context);
                }
            }
        }
    }
}

global.application = new Application();
