
const electron = require('electron');
const updater = require('electron-updater');
const fs = require('fs');
const os = require('os');
const path = require('path');
const process = require('process');
const url = require('url');
const base = require('./base');

class Application {

    constructor() {

        this._views = new ViewCollection(this);
        this._configuration = new ConfigurationService();
        this._menu = new MenuService(this._views);
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
        electron.ipcMain.on('open-file-dialog', (event) => {
            this._openFileDialog();
            event.returnValue = null;
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
            repository: this._package.repository,
            platform: process.platform,
            separator: path.sep,
            homedir: os.homedir(),
            titlebar: process.platform === 'darwin'
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
        this._resetMenu();
        this._views.on('active-view-changed', () => {
            this._menu.update();
        });
        this._views.on('active-view-updated', () => {
            this._menu.update();
        });
    }

    _openFileDialog() {
        const extensions = new base.Metadata().extensions;
        const showOpenDialogOptions = {
            properties: [ 'openFile' ],
            filters: [ { name: 'All Model Files', extensions: extensions } ]
        };
        const selectedFiles = electron.dialog.showOpenDialogSync(showOpenDialogOptions);
        if (selectedFiles) {
            for (const file of selectedFiles) {
                this._openPath(file);
            }
        }
    }

    _openPath(path) {
        if (this._openQueue) {
            this._openQueue.push(path);
            return;
        }
        if (path && path.length > 0 && fs.existsSync(path)) {
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
                this._loadPath(path, view);
            }
        }
    }

    _loadPath(path, view) {
        const recents = this._configuration.get('recents').filter((recent) => path !== recent.path);
        view.open(path);
        recents.unshift({ path: path });
        if (recents.length > 9) {
            recents.splice(9);
        }
        this._configuration.set('recents', recents);
        this._resetMenu();
    }

    _dropPaths(sender, paths) {
        const window = sender.getOwnerBrowserWindow();
        let view = this._views.get(window);
        for (const path of paths) {
            if (view) {
                this._loadPath(path, view);
                view = null;
            }
            else {
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

    service(name) {
        if (name == 'configuration') {
            return this._configuration;
        }
        return undefined;
    }

    execute(command, data) {
        const view = this._views.activeView;
        if (view) {
            view.execute(command, data || {});
        }
        this._menu.update();
    }

    _reload() {
        const view = this._views.activeView;
        if (view && view.path) {
            this._loadPath(view.path, view);
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
        let dialog = null;
        const options = {
            show: false,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#2d2d2d' : '#e6e6e6',
            width: 400,
            height: 280,
            center: true,
            minimizable: false,
            maximizable: false,
            useContentSize: true,
            resizable: true,
            fullscreenable: false,
            webPreferences: {
                nodeIntegration: true,
            }
        };
        if (process.platform === 'darwin') {
            options.title = '';
            dialog = Application._aboutDialog;
        }
        else {
            options.title = 'About ' + electron.app.name;
            options.parent = electron.BrowserWindow.getFocusedWindow();
            options.modal = true;
            options.showInTaskbar = false;
        }
        if (process.platform === 'win32') {
            options.type = 'toolbar';
        }
        if (!dialog) {
            dialog = new electron.BrowserWindow(options);
            if (process.platform === 'darwin') {
                Application._aboutDialog = dialog;
            }
            dialog.removeMenu();
            dialog.excludedFromShownWindowsMenu = true;
            dialog.webContents.setWindowOpenHandler((detail) => {
                const url = detail.url;
                if (url.startsWith('http://') || url.startsWith('https://')) {
                    electron.shell.openExternal(url);
                }
                return { action: 'deny' };
            });
            const pathname = path.join(__dirname, 'index.html');
            let content = fs.readFileSync(pathname, 'utf-8');
            content = content.replace(/<\s*script[^>]*>[\s\S]*?(<\s*\/script[^>]*>|$)/ig, '');
            content = content.replace('{version}', this._package.version);
            content = content.replace('<title>Netron</title>', '');
            content = content.replace('<body class="welcome spinner">', '<body class="default about desktop">');
            content = content.replace(/<link.*>/gi, '');
            dialog.once('ready-to-show', () => {
                dialog.resizable = false;
                dialog.show();
            });
            dialog.on('close', function() {
                electron.globalShortcut.unregister('Escape');
                Application._aboutDialog = null;
            });
            dialog.loadURL('data:text/html;charset=utf-8,' + encodeURIComponent(content));
            electron.globalShortcut.register('Escape', function() {
                dialog.close();
            });
        }
        else {
            dialog.show();
        }
    }

    _resetMenu() {
        const menuRecentsTemplate = [];
        if (this._configuration.has('recents')) {
            let recents = this._configuration.get('recents');
            recents = recents.filter((recent) => {
                const path = recent.path;
                if (fs.existsSync(path)) {
                    const stat = fs.statSync(path);
                    if (stat.isFile() || stat.isDirectory()) {
                        return true;
                    }
                }
                return false;
            });
            if (recents.length > 9) {
                recents.splice(9);
            }
            this._configuration.set('recents', recents);
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                menuRecentsTemplate.push({
                    path: recent.path,
                    label: Application.minimizePath(recent.path),
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
                    click: () => this._openFileDialog()
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
                    id: 'view.toggle-attributes',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => this.execute('toggle', 'attributes'),
                },
                {
                    id: 'view.toggle-initializers',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => this.execute('toggle', 'initializers'),
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
                label: '&Search Feature Requests',
                click: () => electron.shell.openExternal('https://www.github.com/' + this._package.repository + '/issues?q=is:open+is:issue+label:feature')
            },
            {
                label: 'Report &Issues',
                click: () => electron.shell.openExternal('https://www.github.com/' + this._package.repository + '/issues/new')
            }
        ];

        if (process.platform !== 'darwin') {
            helpSubmenu.push({ type: 'separator' });
            helpSubmenu.push({
                label: 'About ' + electron.app.name,
                click: () => this._about()
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
        commandTable.set('view.toggle-initializers', {
            enabled: (view) => view && view.path ? true : false,
            label: (view) => !view || view.get('initializers') ? 'Hide &Initializers' : 'Show &Initializers'
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

}

class View {

    constructor(owner) {
        this._owner = owner;
        this._ready = false;
        this._path = null;
        this._properties = new Map();
        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        const options = {
            show: false,
            title: electron.app.name,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#1d1d1d' : '#e6e6e6',
            icon: electron.nativeImage.createFromPath(path.join(__dirname, 'icon.png')),
            minWidth: 600,
            minHeight: 400,
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
        if (!this._owner.empty && View._position && View._position.length == 2) {
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
        }
        else {
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
        if (this._window && this._window.webContents) {
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
                        const path = Application.minimizePath(this._path);
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
    }
}

class ViewCollection {

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
            }
            else if (window.isMaximized()) {
                window.unmaximize();
            }
            else {
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
        const view = new View(this);
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
        const view = window ? this._views.get(window) : null;
        if (view !== this._activeView) {
            this._activeView = view;
            this.emit('active-view-changed', { activeView: this._activeView });
        }
    }
}

class ConfigurationService {

    constructor() {
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
                }
                catch (error) {
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
    }

    get(name) {
        return this._data[name];
    }

}

class MenuService {

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
        }
        else if (!this._views.application.environment.titlebar) {
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
}

global.application = new Application();
