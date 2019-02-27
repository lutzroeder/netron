/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var host = host || {};

const electron = require('electron');
const fs = require('fs');
const process = require('process');
const path = require('path');
const protobuf = require('protobufjs'); // eslint-disable-line no-unused-vars
const view = require('./view');

host.ElectronHost = class {

    constructor() {
        if (electron.remote.app.isPackaged) {
            this._telemetry = require('universal-analytics')('UA-54146-13');
        }

        this._name = electron.remote.app.getName();
        this._version = electron.remote.app.getVersion();

        process.on('uncaughtException', (err) => {
            this.exception(err, true);
        });
        window.eval = global.eval = () => {
            throw new Error('window.eval() not supported.');
        };

        this._updateTheme();
        if (electron.remote.systemPreferences.subscribeNotification) {
            electron.remote.systemPreferences.subscribeNotification('AppleInterfaceThemeChangedNotification', () => this._updateTheme());
        }
        document.body.style.opacity = 1;
    }

    get document() {
        return window.document;
    }

    _updateTheme() {
        if (electron.remote.systemPreferences.isDarkMode &&
            electron.remote.systemPreferences.isDarkMode()) {
            document.body.classList.add('dark-mode');
        }
        else {
            document.body.classList.remove('dark-mode');
        }
    }

    get name() {
        return this._name;
    }

    get version() {
        return this._version;
    }

    get type() {
        return 'Electron';
    }

    initialize(view) {
        this._view = view;
        this._view.show('Welcome');

        electron.ipcRenderer.on('open', (_, data) => {
            this._openFile(data.file);
        });
        electron.ipcRenderer.on('export', (_, data) => {
            this._view.export(data.file);
        });
        electron.ipcRenderer.on('cut', () => {
            this._view.cut();
        });
        electron.ipcRenderer.on('copy', () => {
            this._view.copy();
        });
        electron.ipcRenderer.on('paste', () => {
            this._view.paste();
        });
        electron.ipcRenderer.on('selectall', () => {
            this._view.selectAll();
        });
        electron.ipcRenderer.on('toggle-attributes', () => {
            this._view.toggleAttributes();
            this._update('show-attributes', this._view.showAttributes);
        });
        electron.ipcRenderer.on('toggle-initializers', () => {
            this._view.toggleInitializers();
            this._update('show-initializers', this._view.showInitializers);
        });
        electron.ipcRenderer.on('toggle-names', () => {
            this._view.toggleNames();
            this._update('show-names', this._view.showNames);
        });
        electron.ipcRenderer.on('zoom-in', () => {
            document.getElementById('zoom-in-button').click();
        });
        electron.ipcRenderer.on('zoom-out', () => {
            document.getElementById('zoom-out-button').click();
        });
        electron.ipcRenderer.on('reset-zoom', () => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', () => {
            document.getElementById('model-properties-button').click();
        });
        electron.ipcRenderer.on('find', () => {
            this._view.find();
        });

        var openFileButton = document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.opacity = 1;
            openFileButton.addEventListener('click', () => {
                electron.ipcRenderer.send('open-file-dialog', {});
            });
        }

        document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        document.addEventListener('drop', (e) => {
            e.preventDefault();
        });
        document.body.addEventListener('drop', (e) => { 
            e.preventDefault();
            var files = [];
            for (var i = 0; i < e.dataTransfer.files.length; i++) {
                var file = e.dataTransfer.files[i].path;
                if (this._view.accept(file)) {
                    files.push(e.dataTransfer.files[i].path);
                }
            }
            if (files.length > 0) {
                electron.ipcRenderer.send('drop-files', { files: files });
            }
            return false;
        });
    }

    environment(name) {
        if (name == 'zoom') {
            return 'd3';
        }
        return null;
    }

    error(message, detail) {
        var owner = electron.remote.getCurrentWindow();
        var options = {
            type: 'error',
            message: message,
            detail: detail,
        };
        electron.remote.dialog.showMessageBox(owner, options);
    }

    confirm(message, detail) {
        var owner = electron.remote.getCurrentWindow();
        var options = {
            type: 'question',
            message: message,
            detail: detail,
            buttons: ['Yes', 'No'],
            defaultId: 0,
            cancelId: 1
        };
        var result = electron.remote.dialog.showMessageBox(owner, options);
        return result == 0;
    }

    require(id, callback) {
        try {
            callback(null, require(id));
        }
        catch (err) {
            callback(err, null);
        }
    }

    save(name, extension, defaultPath, callback) {
        var owner = electron.remote.BrowserWindow.getFocusedWindow();
        var showSaveDialogOptions = {
            title: 'Export Tensor',
            defaultPath: defaultPath,
            buttonLabel: 'Export',
            filters: [ { name: name, extensions: [ extension ] } ]
        };
        electron.remote.dialog.showSaveDialog(owner, showSaveDialogOptions, (filename) => {
            if (filename) {
                callback(filename);
            }
        });
    }

    export(file, blob) {
        var reader = new FileReader();
        reader.onload = (e) => {
            var data = new Uint8Array(e.target.result);
            var encoding = null;
            fs.writeFile(file, data, encoding, (err) => {
                if (err) {
                    this.exception(err, false);
                    this.error('Export write failure.', err);
                }
            });
        };
        reader.readAsArrayBuffer(blob);
    }

    request(base, file, encoding, callback) {
        var pathname = path.join(base || __dirname, file);
        fs.exists(pathname, (exists) => {
            if (!exists) {
                callback(new Error('File not found.'), null);
            }
            else {
                fs.readFile(pathname, encoding, (err, data) => {
                    if (err) {
                        callback(err, null);
                    }
                    else {
                        callback(null, data);
                    }
                });
            }
        });
    }

    openURL(url) {
        electron.shell.openExternal(url);
    }

    exception(err, fatal) {
        if (this._telemetry) {
            try {
                var description = [];
                description.push((err && err.name ? (err.name + ': ') : '') + (err && err.message ? err.message : '(null)'));
                if (err.stack) {
                    var match = err.stack.match(/\n {4}at (.*)\((.*)\)/);
                    if (match) {
                        description.push(match[1] + '(' + match[2].split('/').pop().split('\\').pop() + ')');
                    }
                }
    
                var params = { 
                    applicationName: this.type,
                    applicationVersion: this.version,
                    userAgentOverride: navigator.userAgent
                };
                this._telemetry.exception(description.join(' @ '), fatal, params, () => { });
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    screen(name) {
        if (this._telemetry) {
            try {
                var params = {
                    userAgentOverride: navigator.userAgent
                };
                this._telemetry.screenview(name, this.type, this.version, null, null, params, () => { });
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    event(category, action, label, value) {
        if (this._telemetry) {
            try {
                var params = { 
                    applicationName: this.type,
                    applicationVersion: this.version,
                    userAgentOverride: navigator.userAgent
                };
                this._telemetry.event(category, action, label, value, params, () => { });
            }
            catch (e) {
                // continue regardless of error
            }
        }
    }

    _openFile(file) {
        if (file) {
            this._view.show('Spinner');
            this._readFile(file, (err, buffer) => {
                if (err) {
                    this.exception(err, false);
                    this._view.show(null);
                    this.error('Error while reading file.', err.message);
                    this._update('path', null);
                    return;
                }
                var context = new ElectonContext(this, path.dirname(file), path.basename(file), buffer);
                this._view.openContext(context, (err, model) => {
                    this._view.show(null);
                    if (err) {
                        this.exception(err, false);
                        this.error(err.name, err.message);
                        this._update('path', null);
                    }
                    if (model) {
                        this._update('path', file);
                    }
                    this._update('show-attributes', this._view.showAttributes);
                    this._update('show-initializers', this._view.showInitializers);
                    this._update('show-names', this._view.showNames);
                });
            });
        }
    }

    _readFile(file, callback) {
        fs.exists(file, (exists) => {
            if (!exists) {
                callback(new Error('The file \'' + file + '\' does not exist.'), null);
                return;
            }
            fs.stat(file, (err, stats) => {
                if (err) {
                    callback(err, null);
                    return;
                }
                fs.open(file, 'r', (err, fd) => {
                    if (err) {
                        callback(err, null);
                        return;
                    }
                    var size = stats.size;
                    var buffer = new Uint8Array(size);
                    fs.read(fd, buffer, 0, size, 0, (err, bytesRead, buffer) => {
                        if (err) {
                            callback(err, null);
                            return;
                        }
                        fs.close(fd, (err) => {
                            if (err) {
                                callback(err, null);
                                return;
                            }
                            callback(null, buffer);
                        });
                    });
                });
            });
        });
    }

    _update(name, value) {
        electron.ipcRenderer.send('update', { name: name, value: value });
    }
};

class ElectonContext {

    constructor(host, folder, identifier, buffer) {
        this._host = host;
        this._folder = folder;
        this._identifier = identifier;
        this._buffer = buffer;
    }

    request(file, encoding, callback) {
        this._host.request(this._folder, file, encoding, (err, buffer) => {
            callback(err, buffer);
        });
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }
}

window.__view__ = new view.View(new host.ElectronHost());
