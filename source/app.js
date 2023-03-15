
const electron = require('electron');
const updater = require('electron-updater');
const fs = require('fs');
const os = require('os');
const path = require('path');
const process = require('process');
const url = require('url');
const base = require('./base');

var app = {};

app.Application = class {

    constructor() {

        this._views = new app.ViewCollection(this);
        this._configuration = new app.ConfigurationService(this._views);
        this._menu = new app.MenuService(this._views);
        this._openQueue = [];

        const packageFile = path.join(path.dirname(__dirname), 'package.json');
        const packageContent = fs.readFileSync(packageFile, 'utf-8');
        this._package = JSON.parse(packageContent);

        electron.app.setAppUserModelId('com.lutzroeder.netron');
        electron.app.allowRendererProcessReuse = true;

        if (!electron.app.requestSingleInstanceLock()) {
            electron.app.quit();
            return;
        }

        electron.app.on('second-instance', (event, commandLine, workingDirectory) => {
            const currentDirectory = process.cwd();
            process.chdir(workingDirectory);
            const open = this._parseCommandLine(commandLine);
            process.chdir(currentDirectory);
            if (!open && !this._views.empty) {
                const view = this._views.first();
                if (view) {
                    view.restore();
                }
            }
        });
        electron.ipcMain.on('get-environment', (event) => {
            event.returnValue = this.environment;
        });
        electron.ipcMain.on('get-configuration', (event, obj) => {
            event.returnValue = this._configuration.has(obj.name) ? this._configuration.get(obj.name) : undefined;
        });
        electron.ipcMain.on('set-configuration', (event, obj) => {
            this._configuration.set(obj.name, obj.value);
            this._configuration.save();
            event.returnValue = null;
        });
        electron.ipcMain.on('drop-paths', (event, data) => {
            const paths = data.paths.filter((path) => {
                if (fs.existsSync(path)) {
                    const stat = fs.statSync(path);
                    return stat.isFile() || stat.isDirectory();
                }
                return false;
            });
            this._dropPaths(event.sender, paths);
            event.returnValue = null;
        });
        electron.ipcMain.on('show-message-box', (event, options) => {
            const owner = event.sender.getOwnerBrowserWindow();
            event.returnValue = electron.dialog.showMessageBoxSync(owner, options);
        });
        electron.ipcMain.on('show-save-dialog', (event, options) => {
            const owner = event.sender.getOwnerBrowserWindow();
            event.returnValue = electron.dialog.showSaveDialogSync(owner, options);
        });
        electron.ipcMain.on('execute', (event, data) => {
            const owner = event.sender.getOwnerBrowserWindow();
            this.execute(data.name, data.value || null, owner);
            event.returnValue = null;
        });

        electron.app.on('will-finish-launching', () => {
            electron.app.on('open-file', (event, path) => {
                this._openPath(path);
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

    get environment() {
        this._environment = this._environment || {
            packaged: electron.app.isPackaged,
            version: this._package.version,
            date: this._package.date,
            repository: 'https://github.com' + this._package.repository,
            platform: process.platform,
            separator: path.sep,
            homedir: os.homedir(),
            titlebar: true // process.platform === 'darwin'
        };
        return this._environment;
    }

    _parseCommandLine(argv) {
        let open = false;
        if (argv.length > 1) {
            for (const arg of argv.slice(1)) {
                if (!arg.startsWith('-') && arg !== path.dirname(__dirname)) {
                    const extension = path.extname(arg).toLowerCase();
                    if (extension !== '' && extension !== '.js' && fs.existsSync(arg)) {
                        const stat = fs.statSync(arg);
                        if (stat.isFile() || stat.isDirectory()) {
                            this._openPath(arg);
                            open = true;
                        }
                    }
                }
            }
        }
        return open;
    }

    _ready() {
        this._configuration.load();
        if (this._openQueue) {
            const queue = this._openQueue;
            this._openQueue = null;
            while (queue.length > 0) {
                const file = queue.shift();
                this._openPath(file);
            }
        }
        if (this._views.empty) {
            this._views.openView();
        }
        this._updateRecents();
        this._views.on('active-view-changed', () => {
            this._menu.update();
        });
        this._views.on('active-view-updated', () => {
            this._menu.update();
        });
    }

    _open(path) {
        let paths = path ? [ path ] : [];
        if (paths.length === 0) {
            const extensions = new base.Metadata().extensions;
            const showOpenDialogOptions = {
                properties: [ 'openFile' ],
                filters: [ { name: 'All Model Files', extensions: extensions } ]
            };
            paths = electron.dialog.showOpenDialogSync(showOpenDialogOptions);
        }
        if (Array.isArray(paths) && paths.length > 0) {
            for (const path of paths) {
                this._openPath(path);
            }
        }
    }

    _openPath(path) {
        if (this._openQueue) {
            this._openQueue.push(path);
            return;
        }
        if (path && path.length > 0) {
            const exists = fs.existsSync(path);
            if (exists) {
                const stat = fs.statSync(path);
                if (stat.isFile() || stat.isDirectory()) {
                    const views = Array.from(this._views.views);
                    // find existing view for this file
                    let view = views.find(view => view.match(path));
                    // find empty welcome window
                    if (view == null) {
                        view = views.find(view => view.match(null));
                    }
                    // create new window
                    if (view == null) {
                        view = this._views.openView();
                    }
                    view.open(path);
                }
            }
            this._updateRecents(exists ? path : undefined);
        }
    }

    _dropPaths(sender, paths) {
        const window = sender.getOwnerBrowserWindow();
        let view = this._views.get(window);
        for (const path of paths) {
            if (view) {
                view.open(path);
                this._updateRecents(path);
                view = null;
            } else {
                this._openPath(path);
            }
        }
    }

    _export() {
        const view = this._views.activeView;
        if (view && view.path) {
            let defaultPath = 'Untitled';
            const file = view.path;
            const lastIndex = file.lastIndexOf('.');
            if (lastIndex !== -1) {
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
            const selectedFile = electron.dialog.showSaveDialogSync(owner, showSaveDialogOptions);
            if (selectedFile) {
                view.execute('export', { 'file': selectedFile });
            }
        }
    }

    execute(command, value, window) {
        switch (command) {
            case 'open': this._open(value); break;
            case 'export': this._export(); break;
            case 'close': window.close(); break;
            case 'quit': electron.app.quit(); break;
            case 'reload': this._reload(); break;
            case 'report-issue': electron.shell.openExternal('https://github.com/' + this._package.repository + '/issues/new'); break;
            case 'about': this._about(); break;
            default: {
                const view = this._views.get(window) || this._views.activeView;
                if (view) {
                    view.execute(command, value || {});
                }
                this._menu.update();
            }
        }
    }

    _reload() {
        const view = this._views.activeView;
        if (view && view.path) {
            view.open(view.path);
            this._updateRecents(view.path);
        }
    }

    _checkForUpdates() {
        if (!electron.app.isPackaged) {
            return;
        }
        const autoUpdater = updater.autoUpdater;
        if (autoUpdater.app && autoUpdater.app.appUpdateConfigPath && !fs.existsSync(autoUpdater.app.appUpdateConfigPath)) {
            return;
        }
        const promise = autoUpdater.checkForUpdates();
        if (promise) {
            promise.catch((error) => {
                /* eslint-disable no-console */
                console.log(error.message);
                /* eslint-enable no-console */
            });
        }
    }

    _about() {
        let view = this._views.activeView;
        if (view == null) {
            view = this._views.openView();
        }
        view.execute('about');
    }

    _updateRecents(path) {
        let updated = false;
        let recents = this._configuration.has('recents') ? this._configuration.get('recents') : [];
        if (path && (recents.length === 0 || recents[0] !== path)) {
            recents = recents.filter((recent) => path !== recent.path);
            recents.unshift({ path: path });
            updated = true;
        }
        const value = [];
        for (const recent of recents) {
            if (value.length >= 9) {
                updated = true;
                break;
            }
            const path = recent.path;
            if (!fs.existsSync(path)) {
                updated = true;
                continue;
            }
            const stat = fs.statSync(path);
            if (!stat.isFile() && !stat.isDirectory()) {
                updated = true;
                continue;
            }
            value.push(recent);
        }
        if (updated) {
            this._configuration.set('recents', value);
        }
        this._resetMenu();
    }

    _resetMenu() {
        const menuRecentsTemplate = [];
        if (this._configuration.has('recents')) {
            const recents = this._configuration.get('recents');
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                menuRecentsTemplate.push({
                    path: recent.path,
                    label: app.Application.minimizePath(recent.path),
                    accelerator: ((process.platform === 'darwin') ? 'Cmd+' : 'Ctrl+') + (i + 1).toString(),
                    click: (item) => this._openPath(item.path)
                });
            }
        }

        const menuTemplate = [];

        if (process.platform === 'darwin') {
            menuTemplate.unshift({
                label: electron.app.name,
                submenu: [
                    {
                        label: 'About ' + electron.app.name,
                        click: () => /* this.execute('about', null) */ this._about()
                    },
                    { type: 'separator' },
                    { role: 'hide' },
                    { role: 'hideothers' },
                    { role: 'unhide' },
                    { type: 'separator' },
                    { role: 'quit' }
                ]
            });
        }

        menuTemplate.push({
            label: '&File',
            submenu: [
                {
                    label: '&Open...',
                    accelerator: 'CmdOrCtrl+O',
                    click: () => this._open(null)
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
                    click: () => this.execute('export', null)
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
                    id: 'view.toggle-attributes',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => this.execute('toggle', 'attributes'),
                },
                {
                    id: 'view.toggle-weights',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => this.execute('toggle', 'weights'),
                },
                {
                    id: 'view.toggle-names',
                    accelerator: 'CmdOrCtrl+U',
                    click: () => this.execute('toggle', 'names'),
                },
                {
                    id: 'view.toggle-direction',
                    accelerator: 'CmdOrCtrl+K',
                    click: () => this.execute('toggle', 'direction')
                },
                {
                    id: 'view.toggle-mousewheel',
                    accelerator: 'CmdOrCtrl+M',
                    click: () => this.execute('toggle', 'mousewheel'),
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
                    { role: 'front' }
                ]
            });
        }

        const helpSubmenu = [
            {
                label: 'Report &Issue',
                click: () => this.execute('report-issue', null)
            }
        ];

        if (process.platform !== 'darwin') {
            helpSubmenu.push({ type: 'separator' });
            helpSubmenu.push({
                label: '&About ' + electron.app.name,
                click: () => this.execute('about', null)
            });
        }

        menuTemplate.push({
            role: 'help',
            submenu: helpSubmenu
        });

        const commandTable = new Map();
        commandTable.set('file.export', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('edit.cut', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('edit.copy', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('edit.paste', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('edit.select-all', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('edit.find', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('view.toggle-attributes', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('attributes') ? 'Hide &Attributes' : 'Show &Attributes'
        });
        commandTable.set('view.toggle-weights', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('weights') ? 'Hide &Weights' : 'Show &Weights'
        });
        commandTable.set('view.toggle-names', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('names') ? 'Hide &Names' : 'Show &Names'
        });
        commandTable.set('view.toggle-direction', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('direction') === 'vertical' ? 'Show &Horizontal' : 'Show &Vertical'
        });
        commandTable.set('view.toggle-mousewheel', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('mousewheel') === 'scroll' ? '&Mouse Wheel: Zoom' : '&Mouse Wheel: Scroll'
        });
        commandTable.set('view.reload', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('view.reset-zoom', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('view.zoom-in', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('view.zoom-out', {
            enabled: (view) => view && view.path ? true : false
        });
        commandTable.set('view.show-properties', {
            enabled: (view) => view && view.path ? true : false
        });

        this._menu.build(menuTemplate, commandTable);
        this._menu.update();
    }

    static minimizePath(file) {
        if (process.platform !== 'win32') {
            const homeDir = os.homedir();
            if (file.startsWith(homeDir)) {
                return '~' + file.substring(homeDir.length);
            }
        }
        return file;
    }
};

app.View = class {

    constructor(owner) {
        this._owner = owner;
        this._ready = false;
        this._path = null;
        this._properties = new Map();
        this._dispatch = [];
        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        const options = {
            show: false,
            title: electron.app.name,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#1d1d1d' : '#e6e6e6',
            icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png')),
            minWidth: 600,
            minHeight: 600,
            width: size.width > 1024 ? 1024 : size.width,
            height: size.height > 768 ? 768 : size.height,
            webPreferences: {
                preload: path.join(__dirname, 'electron.js'),
                nodeIntegration: true
            }
        };
        if (owner.application.environment.titlebar) {
            options.frame = false;
            options.thickFrame = true;
            options.titleBarStyle = 'hiddenInset';
        }
        if (!this._owner.empty && app.View._position && app.View._position.length == 2) {
            options.x = app.View._position[0] + 30;
            options.y = app.View._position[1] + 30;
            if (options.x + options.width > size.width) {
                options.x = 0;
            }
            if (options.y + options.height > size.height) {
                options.y = 0;
            }
        }
        this._window = new electron.BrowserWindow(options);
        app.View._position = this._window.getPosition();
        this._window.on('close', () => this._owner.closeView(this));
        this._window.on('focus', () => this.emit('activated'));
        this._window.on('blur', () => this.emit('deactivated'));
        this._window.on('minimize', () => this.state());
        this._window.on('restore', () => this.state());
        this._window.on('maximize', () => this.state());
        this._window.on('unmaximize', () => this.state());
        this._window.on('enter-full-screen', () => this.state());
        this._window.on('leave-full-screen', () => this.state());
        this._window.webContents.on('did-finish-load', () => {
            this._didFinishLoad = true;
        });
        this._window.webContents.setWindowOpenHandler((detail) => {
            const url = detail.url;
            if (url.startsWith('http://') || url.startsWith('https://')) {
                electron.shell.openExternal(url);
            }
            return { action: 'deny' };
        });
        this._window.once('ready-to-show', () => {
            this._window.show();
        });
        this._loadURL();
    }

    get window() {
        return this._window;
    }

    get path() {
        return this._path;
    }

    open(path) {
        this._openPath = path;
        if (this._didFinishLoad) {
            this._window.webContents.send('open', { path: path });
        } else {
            this._window.webContents.on('did-finish-load', () => {
                this._window.webContents.send('open', { path: path });
            });
            this._loadURL();
        }
    }

    _loadURL() {
        const pathname = path.join(__dirname, 'index.html');
        let content = fs.readFileSync(pathname, 'utf-8');
        content = content.replace(/<\s*script[^>]*>[\s\S]*?(<\s*\/script[^>]*>|$)/ig, '');
        const data = 'data:text/html;charset=utf-8,' + encodeURIComponent(content);
        const options = {
            baseURLForDataURL: url.pathToFileURL(pathname).toString()
        };
        this._window.loadURL(data, options);
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
            return this._openPath === path;
        }
        return this._path === path;
    }

    execute(command, data) {
        if (this._dispatch) {
            this._dispatch.push({ command: command, data: data });
        } else if (this._window && this._window.webContents) {
            this._window.webContents.send(command, data);
        }
    }

    update(data) {
        for (const entry of Object.entries(data)) {
            const name = entry[0];
            const value = entry[1];
            switch (name) {
                case 'path': {
                    if (value) {
                        this._path = value;
                        const path = app.Application.minimizePath(this._path);
                        const title = process.platform !== 'darwin' ? path + ' - ' + electron.app.name : path;
                        this._window.setTitle(title);
                        this._window.focus();
                    }
                    delete this._openPath;
                    break;
                }
                default: {
                    this._properties.set(name, value);
                }
            }
        }
        this.emit('updated');
    }

    get(name) {
        return this._properties.get(name);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    state() {
        this.execute('window-state', {
            minimized: this._window.isMinimized(),
            maximized: this._window.isMaximized(),
            fullscreen: this._window.isFullScreen()
        });
        if (this._dispatch) {
            const dispatch = this._dispatch;
            delete this._dispatch;
            for (const obj of dispatch) {
                this.execute(obj.command, obj.data);
            }
        }
    }
};

app.ViewCollection = class {

    constructor(application) {
        this._application = application;
        this._views = new Map();
        electron.ipcMain.on('window-close', (event) => {
            const window = event.sender.getOwnerBrowserWindow();
            window.close();
            event.returnValue = null;
        });
        electron.ipcMain.on('window-toggle', (event) => {
            const window = event.sender.getOwnerBrowserWindow();
            if (window.isFullScreen()) {
                window.setFullScreen(false);
            } else if (window.isMaximized()) {
                window.unmaximize();
            } else {
                window.maximize();
            }
            event.returnValue = null;
        });
        electron.ipcMain.on('window-minimize', (event) => {
            const window = event.sender.getOwnerBrowserWindow();
            window.minimize();
            event.returnValue = null;
        });
        electron.ipcMain.on('window-update', (event, data) => {
            const window = event.sender.getOwnerBrowserWindow();
            if (this._views.has(window)) {
                this._views.get(window).update(data);
            }
            event.returnValue = null;
        });
        electron.ipcMain.on('update-window-state', (event) => {
            const window = event.sender.getOwnerBrowserWindow();
            if (this._views.has(window)) {
                this._views.get(window).state();
            }
            event.returnValue = null;
        });
    }

    get application() {
        return this._application;
    }

    get views() {
        return this._views.values();
    }

    get empty() {
        return this._views.size === 0;
    }

    get(window) {
        return this._views.get(window);
    }

    openView() {
        const view = new app.View(this);
        view.on('activated', (view) => {
            this._activeView = view;
            this.emit('active-view-changed', { activeView: this._activeView });
        });
        view.on('updated', () => {
            this.emit('active-view-updated', { activeView: this._activeView });
        });
        view.on('deactivated', () => {
            this._activeView = null;
            this.emit('active-view-changed', { activeView: this._activeView });
        });
        this._views.set(view.window, view);
        this._updateActiveView();
        return view;
    }

    closeView(view) {
        this._views.delete(view.window);
        this._updateActiveView();
    }

    first() {
        return this.empty ? null : this._views.values().next().value;
    }

    get activeView() {
        return this._activeView;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    _updateActiveView() {
        const window = electron.BrowserWindow.getFocusedWindow();
        const view = window && this._views.has(window) ? this._views.get(window) : null;
        if (view !== this._activeView) {
            this._activeView = view;
            this.emit('active-view-changed', { activeView: this._activeView });
        }
    }
};

app.ConfigurationService = class {

    constructor(views) {
        this._views = views;
        const dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            this._file = path.join(dir, 'configuration.json');
        }
    }

    load() {
        this._data = { 'recents': [] };
        if (this._file && fs.existsSync(this._file)) {
            const data = fs.readFileSync(this._file, 'utf-8');
            if (data) {
                try {
                    this._data = JSON.parse(data);
                } catch (error) {
                    // continue regardless of error
                }
            }
        }
    }

    save() {
        if (this._data && this._file) {
            const data = JSON.stringify(this._data, null, 2);
            fs.writeFileSync(this._file, data);
        }
    }

    has(name) {
        return this._data && Object.prototype.hasOwnProperty.call(this._data, name);
    }

    set(name, value) {
        this._data[name] = value;
        for (const view of this._views.views) {
            view.execute('update-configuration', { name: name, value: value });
        }
    }

    get(name) {
        return this._data[name];
    }
};

app.MenuService = class {

    constructor(views) {
        this._views = views;
    }

    build(menuTemplate, commandTable) {
        this._menuTemplate = menuTemplate;
        this._commandTable = commandTable;
        this._itemTable = new Map();
        for (const menu of menuTemplate) {
            for (const item of menu.submenu) {
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

    update() {
        if (!this._menu && !this._commandTable) {
            return;
        }
        const view = this._views.activeView;
        if (this._updateLabel(view)) {
            this._rebuild();
        }
        this._updateEnabled(view);
    }

    _rebuild() {
        if (process.platform === 'darwin') {
            this._menu = electron.Menu.buildFromTemplate(this._menuTemplate);
            electron.Menu.setApplicationMenu(this._menu);
        } else if (!this._views.application.environment.titlebar) {
            this._menu = electron.Menu.buildFromTemplate(this._menuTemplate);
            for (const view of this._views.views) {
                view.window.setMenu(this._menu);
            }
        }
    }

    _updateLabel(view) {
        let rebuild = false;
        for (const entry of this._commandTable.entries()) {
            if (this._menu) {
                const menuItem = this._menu.getMenuItemById(entry[0]);
                const command = entry[1];
                if (command && command.label) {
                    const label = command.label(view);
                    if (label !== menuItem.label) {
                        if (this._itemTable.has(entry[0])) {
                            this._itemTable.get(entry[0]).label = label;
                            rebuild = true;
                        }
                    }
                }
            }
        }
        return rebuild;
    }

    _updateEnabled(view) {
        for (const entry of this._commandTable.entries()) {
            if (this._menu) {
                const menuItem = this._menu.getMenuItemById(entry[0]);
                const command = entry[1];
                if (menuItem && command.enabled) {
                    menuItem.enabled = command.enabled(view);
                }
            }
        }
    }
};

global.application = new app.Application();
