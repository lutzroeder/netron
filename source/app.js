
import * as base from './base.js';
import * as electron from 'electron';
import * as fs from 'fs';
import * as os from 'os';
import * as path from 'path';
import * as process from 'process';
import * as updater from 'electron-updater';
import * as url from 'url';

const app = {};

app.Application = class {

    constructor() {
        this._views = new app.ViewCollection(this);
        this._configuration = new app.ConfigurationService();
        this._menu = new app.MenuService(this._views);
        this._openQueue = [];
        this._package = {};
    }

    async start() {
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        const packageFile = path.join(path.dirname(dirname), 'package.json');
        const packageContent =  fs.readFileSync(packageFile, 'utf-8');
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
        electron.ipcMain.on('delete-configuration', (event, obj) => {
            this._configuration.delete(obj.name);
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
        electron.ipcMain.on('update-recents', (event, data) => {
            this._updateRecents(data.path);
            event.returnValue = null;
        });
        electron.ipcMain.on('show-save-dialog', async (event, options) => {
            const owner = event.sender.getOwnerBrowserWindow();
            const argument = {};
            try {
                const { filePath, canceled } = await electron.dialog.showSaveDialog(owner, options);
                argument.filePath = filePath;
                argument.canceled = canceled;
            } catch (error) {
                argument.error = error.message;
            }
            event.sender.send('show-save-dialog-complete', argument);
        });
        electron.ipcMain.on('execute', async (event, data) => {
            const owner = event.sender.getOwnerBrowserWindow();
            const argument = {};
            try {
                argument.value = await this.execute(data.name, data.value || null, owner);
            } catch (error) {
                argument.error = error.message;
            }
            event.sender.send('execute-complete', argument);
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
        await this._checkForUpdates();
    }

    get environment() {
        this._environment = this._environment || {
            packaged: electron.app.isPackaged,
            name: this._package.productName,
            version: this._package.version,
            date: this._package.date,
            repository: `https://github.com/${this._package.repository}`,
            platform: process.platform,
            separator: path.sep,
            titlebar: true // process.platform === 'darwin'
        };
        return this._environment;
    }

    _parseCommandLine(argv) {
        let open = false;
        if (argv.length > 1) {
            for (const arg of argv.slice(1)) {
                const dirname = path.dirname(url.fileURLToPath(import.meta.url));
                if (!arg.startsWith('-') && arg !== path.dirname(dirname)) {
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
        this._configuration.open();
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
        this._updateMenu();
        this._views.on('active-view-changed', () => {
            this._menu.update();
        });
        this._views.on('active-view-updated', () => {
            this._menu.update();
        });
    }

    _open(path) {
        let paths = path ? [path] : [];
        if (paths.length === 0) {
            const extensions = new base.Metadata().extensions;
            const options = {
                properties: ['openFile'],
                filters: [{ name: 'All Model Files', extensions }]
            };
            const owner = electron.BrowserWindow.getFocusedWindow();
            paths = electron.dialog.showOpenDialogSync(owner, options);
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
                    let view = views.find((view) => view.match(path));
                    // find empty welcome window
                    if (!view) {
                        view = views.find((view) => view.match(null));
                    }
                    // create new window
                    if (!view) {
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
                view = null;
            } else {
                this._openPath(path);
            }
        }
    }

    async _export() {
        const view = this._views.activeView;
        if (view && view.path) {
            let defaultPath = 'Untitled';
            const file = view.path;
            const lastIndex = file.lastIndexOf('.');
            if (lastIndex !== -1) {
                defaultPath = file.substring(0, lastIndex);
            }
            const owner = electron.BrowserWindow.getFocusedWindow();
            const options = {
                title: 'Export',
                defaultPath,
                buttonLabel: 'Export',
                filters: [
                    { name: 'PNG', extensions: ['png'] },
                    { name: 'SVG', extensions: ['svg'] }
                ]
            };
            const { filePath, canceled } = await electron.dialog.showSaveDialog(owner, options);
            if (filePath && !canceled) {
                view.execute('export', { 'file': filePath });
            }
        }
    }

    async execute(command, value, window) {
        switch (command) {
            case 'open': this._open(value); break;
            case 'export': await this._export(); break;
            case 'close': window.close(); break;
            case 'quit': electron.app.quit(); break;
            case 'reload': this._reload(); break;
            case 'report-issue': electron.shell.openExternal(`https://github.com/${this._package.repository}/issues/new`); break;
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
        }
    }

    async _checkForUpdates() {
        if (!electron.app.isPackaged) {
            return;
        }
        const autoUpdater = updater.default.autoUpdater;
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
        if (!view) {
            view = this._views.openView();
        }
        view.execute('about');
    }

    _updateRecents(path) {
        let updated = false;
        let recents = this._configuration.has('recents') ? this._configuration.get('recents') : [];
        if (path && (recents.length === 0 || recents[0] !== path)) {
            recents = recents.filter((recent) => path !== recent);
            recents.unshift(path);
            updated = true;
        }
        const value = [];
        for (const recent of recents) {
            if (value.length >= 9) {
                updated = true;
                break;
            }
            if (!fs.existsSync(recent)) {
                updated = true;
                continue;
            }
            const stat = fs.statSync(recent);
            if (!stat.isFile() && !stat.isDirectory()) {
                updated = true;
                continue;
            }
            value.push(recent);
        }
        if (updated) {
            this._configuration.set('recents', value);
            this._updateMenu();
        }
    }

    _updateMenu() {

        let recents = [];
        if (this._configuration.has('recents')) {
            const value = this._configuration.get('recents');
            recents = value.map((recent) => app.Application.location(recent));
        }

        if (this.environment.titlebar && recents.length > 0) {
            for (const view of this._views.views) {
                view.execute('recents', recents);
            }
        }

        const darwin = process.platform === 'darwin';
        if (!this.environment.titlebar || darwin) {
            const menuRecentsTemplate = [];
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                menuRecentsTemplate.push({
                    path: recent.path,
                    label: recent.label,
                    accelerator: (darwin ? 'Cmd+' : 'Ctrl+') + (i + 1).toString(),
                    click: (item) => this._openPath(item.path)
                });
            }

            const menuTemplate = [];

            if (darwin) {
                menuTemplate.unshift({
                    label: electron.app.name,
                    submenu: [
                        {
                            label: `About ${electron.app.name}`,
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

            const fileSubmenu = [
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
                    click: async () => await this.execute('export', null)
                },
                { type: 'separator' },
                { role: 'close' },
            ];

            if (!darwin) {
                fileSubmenu.push(
                    { type: 'separator' },
                    { role: 'quit' }
                );
            }

            menuTemplate.push({
                label: '&File',
                submenu: fileSubmenu
            });

            if (darwin) {
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
                        click: async () => await this.execute('cut', null),
                    },
                    {
                        id: 'edit.copy',
                        label: '&Copy',
                        accelerator: 'CmdOrCtrl+C',
                        click: async () => await this.execute('copy', null),
                    },
                    {
                        id: 'edit.paste',
                        label: '&Paste',
                        accelerator: 'CmdOrCtrl+V',
                        click: async () => await this.execute('paste', null),
                    },
                    {
                        id: 'edit.select-all',
                        label: 'Select &All',
                        accelerator: 'CmdOrCtrl+A',
                        click: async () => await this.execute('selectall', null),
                    },
                    { type: 'separator' },
                    {
                        id: 'edit.find',
                        label: '&Find...',
                        accelerator: 'CmdOrCtrl+F',
                        click: async () => await this.execute('find', null),
                    }
                ]
            });

            const viewTemplate = {
                label: '&View',
                submenu: [
                    {
                        id: 'view.toggle-attributes',
                        accelerator: 'CmdOrCtrl+D',
                        click: async () => await this.execute('toggle', 'attributes'),
                    },
                    {
                        id: 'view.toggle-weights',
                        accelerator: 'CmdOrCtrl+I',
                        click: async () => await this.execute('toggle', 'weights'),
                    },
                    {
                        id: 'view.toggle-names',
                        accelerator: 'CmdOrCtrl+U',
                        click: async () => await this.execute('toggle', 'names'),
                    },
                    {
                        id: 'view.toggle-direction',
                        accelerator: 'CmdOrCtrl+K',
                        click: async () => await this.execute('toggle', 'direction')
                    },
                    {
                        id: 'view.toggle-mousewheel',
                        accelerator: 'CmdOrCtrl+M',
                        click: async () => await this.execute('toggle', 'mousewheel'),
                    },
                    { type: 'separator' },
                    {
                        id: 'view.reload',
                        label: '&Reload',
                        accelerator: darwin ? 'Cmd+R' : 'F5',
                        click: async () => await this._reload(),
                    },
                    { type: 'separator' },
                    {
                        id: 'view.reset-zoom',
                        label: 'Actual &Size',
                        accelerator: 'Shift+Backspace',
                        click: async () => await this.execute('reset-zoom', null),
                    },
                    {
                        id: 'view.zoom-in',
                        label: 'Zoom &In',
                        accelerator: 'Shift+Up',
                        click: async () => await this.execute('zoom-in', null),
                    },
                    {
                        id: 'view.zoom-out',
                        label: 'Zoom &Out',
                        accelerator: 'Shift+Down',
                        click: async () => await this.execute('zoom-out', null),
                    },
                    { type: 'separator' },
                    {
                        id: 'view.show-properties',
                        label: '&Properties...',
                        accelerator: 'CmdOrCtrl+Enter',
                        click: async () => await this.execute('show-properties', null),
                    }
                ]
            };
            if (!electron.app.isPackaged) {
                viewTemplate.submenu.push({ type: 'separator' });
                viewTemplate.submenu.push({ role: 'toggledevtools' });
            }
            menuTemplate.push(viewTemplate);

            if (darwin) {
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
                    click: async () => await this.execute('report-issue', null)
                }
            ];

            if (!darwin) {
                helpSubmenu.push({ type: 'separator' });
                helpSubmenu.push({
                    label: `&About ${electron.app.name}`,
                    click: async () => await this.execute('about', null)
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
                enabled: (view) => view && (view.path || view.get('can-copy')) ? true : false
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
    }

    static location(path) {
        if (process.platform !== 'win32') {
            const homeDir = os.homedir();
            if (path.startsWith(homeDir)) {
                return { path, label: `~${path.substring(homeDir.length)}` };
            }
        }
        return { path, label: path };
    }
};

app.View = class {

    constructor(owner) {
        this._owner = owner;
        this._ready = false;
        this._path = null;
        this._properties = new Map();
        this._dispatch = [];
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        const size = electron.screen.getPrimaryDisplay().workAreaSize;
        const options = {
            show: false,
            title: electron.app.name,
            backgroundColor: electron.nativeTheme.shouldUseDarkColors ? '#1e1e1e' : '#ececec',
            icon: electron.nativeImage.createFromPath(path.join(dirname, 'icon.png')),
            minWidth: 600,
            minHeight: 600,
            width: size.width > 1024 ? 1024 : size.width,
            height: size.height > 768 ? 768 : size.height,
            webPreferences: {
                preload: path.join(dirname, 'desktop.mjs'),
                nodeIntegration: true,
                enableDeprecatedPaste: true
            }
        };
        if (owner.application.environment.titlebar) {
            options.frame = false;
            options.thickFrame = true;
            options.titleBarStyle = 'hiddenInset';
        }
        if (!this._owner.empty && app.View._position && app.View._position.length === 2) {
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
        this._window.on('enter-full-screen', () => this.state('enter-full-screen'));
        this._window.on('leave-full-screen', () => this.state('leave-full-screen'));
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
        if (owner.application.environment.titlebar && process.platform !== 'darwin') {
            this._window.removeMenu();
        }
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
        const location = app.Application.location(path);
        if (this._didFinishLoad) {
            this._window.webContents.send('open', location);
        } else {
            this._window.webContents.on('did-finish-load', () => {
                this._window.webContents.send('open', location);
            });
            this._loadURL();
        }
    }

    _loadURL() {
        const dirname = path.dirname(url.fileURLToPath(import.meta.url));
        const pathname = path.join(dirname, 'index.html');
        let content = fs.readFileSync(pathname, 'utf-8');
        content = content.replace(/<\s*script[^>]*>[\s\S]*?(<\s*\/script[^>]*>|$)/ig, '');
        const data = `data:text/html;charset=utf-8,${encodeURIComponent(content)}`;
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
            this._dispatch.push({ command, data });
        } else if (this._window && this._window.webContents) {
            const window = this._window;
            const contents = window.webContents;
            switch (command) {
                case 'toggle-developer-tools':
                    if (contents.isDevToolsOpened()) {
                        contents.closeDevTools();
                    } else {
                        contents.openDevTools();
                    }
                    break;
                case 'fullscreen':
                    window.setFullScreen(!window.isFullScreen());
                    break;
                default:
                    contents.send(command, data);
                    break;
            }
        }
    }

    update(data) {
        for (const [name, value] of Object.entries(data)) {
            switch (name) {
                case 'path': {
                    if (value) {
                        this._path = value;
                        const location = app.Application.location(this._path);
                        const title = process.platform === 'darwin' ? location.label : `${location.label} - ${electron.app.name}`;
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

    state(event) {
        let fullscreen = false;
        switch (event) {
            case 'enter-full-screen': fullscreen = true; break;
            case 'leave-full-screen': fullscreen = false; break;
            default: fullscreen = this._window.isFullScreen(); break;
        }
        this.execute('window-state', {
            minimized: this._window.isMinimized(),
            maximized: this._window.isMaximized(),
            fullscreen
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
                const view = this._views.get(window);
                view.update(data);
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

    constructor() {
        const dir = electron.app.getPath('userData');
        if (dir && dir.length > 0) {
            this._file = path.join(dir, 'configuration.json');
        }
    }

    open() {
        this._content = { 'recents': [] };
        if (this._file && fs.existsSync(this._file)) {
            const data = fs.readFileSync(this._file, 'utf-8');
            if (data) {
                try {
                    this._content = JSON.parse(data);
                    if (Array.isArray(this._content.recents)) {
                        this._content.recents = this._content.recents.map((recent) => typeof recent !== 'string' && recent && recent.path ? recent.path : recent);
                    }
                } catch {
                    // continue regardless of error
                }
            }
        }
    }

    save() {
        if (this._content && this._file) {
            const data = JSON.stringify(this._content, null, 2);
            fs.writeFileSync(this._file, data);
        }
    }

    has(name) {
        return this._content && Object.prototype.hasOwnProperty.call(this._content, name);
    }

    set(name, value) {
        this._content[name] = value;
    }

    get(name) {
        return this._content[name];
    }

    delete(name) {
        delete this._content[name];
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
        for (const [name, command] of this._commandTable.entries()) {
            if (this._menu) {
                const item = this._menu.getMenuItemById(name);
                if (command && command.label) {
                    const label = command.label(view);
                    if (label !== item.label) {
                        if (this._itemTable.has(name)) {
                            this._itemTable.get(name).label = label;
                            rebuild = true;
                        }
                    }
                }
            }
        }
        return rebuild;
    }

    _updateEnabled(view) {
        for (const [name, command] of this._commandTable.entries()) {
            if (this._menu) {
                const item = this._menu.getMenuItemById(name);
                if (item && command.enabled) {
                    item.enabled = command.enabled(view);
                }
            }
        }
    }
};

try {
    global.application = new app.Application();
    await global.application.start();
} catch (error) {
    /* eslint-disable no-console */
    console.error(error.message);
    /* eslint-enable no-console */
}
