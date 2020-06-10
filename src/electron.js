/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var host = host || {};

const electron = require('electron');
const fs = require('fs');
const http = require('http');
const https = require('https');
const process = require('process');
const path = require('path');
const view = require('./view');

global.protobuf = require('protobufjs');

host.ElectronHost = class {

    constructor() {
        process.on('uncaughtException', (err) => {
            this.exception(err, true);
        });
        window.eval = global.eval = () => {
            throw new Error('window.eval() not supported.');
        };
        this._version = electron.remote.app.getVersion();
    }

    get document() {
        return window.document;
    }

    get version() {
        return this._version;
    }

    get type() {
        return 'Electron';
    }

    initialize(view) {
        this._view = view;
        return new Promise((resolve /*, reject */) => {
            const accept = () => {
                if (electron.remote.app.isPackaged) {
                    this._telemetry = require('universal-analytics')('UA-54146-13', this._getConfiguration('userId'));
                    this._telemetry.set('anonymizeIp', 1);
                }
                resolve();
            };
            const request = () => {
                this._view.show('welcome consent');
                const acceptButton = this.document.getElementById('consent-accept-button');
                if (acceptButton) {
                    acceptButton.addEventListener('click', () => {
                        this._setConfiguration('consent', Date.now());
                        accept();
                    });
                }
            };
            const time = this._getConfiguration('consent');
            if (time && (Date.now() - time) < 30 * 24 * 60 * 60 * 1000) {
                accept();
            }
            else {
                this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 'utf-8', 2000).then((text) => {
                    try {
                        const json = JSON.parse(text);
                        const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                        if (json && json.country && !countries.indexOf(json.country) !== -1) {
                            this._setConfiguration('consent', Date.now());
                            accept();
                        }
                        else {
                            request();
                        }
                    }
                    catch (err) {
                        request();
                    }
                }).catch(() => {
                    request();
                });
            }
        });
    }

    start() {
        this._view.show('welcome');

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
            this.document.getElementById('zoom-in-button').click();
        });
        electron.ipcRenderer.on('zoom-out', () => {
            this.document.getElementById('zoom-out-button').click();
        });
        electron.ipcRenderer.on('reset-zoom', () => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', () => {
            this.document.getElementById('menu-button').click();
        });
        electron.ipcRenderer.on('find', () => {
            this._view.find();
        });
        this.document.getElementById('menu-button').addEventListener('click', () => {
            this._view.showModelProperties();
        });

        const openFileButton = this.document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.opacity = 1;
            openFileButton.addEventListener('click', () => {
                electron.ipcRenderer.send('open-file-dialog', {});
            });
        }

        this.document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        this.document.addEventListener('drop', (e) => {
            e.preventDefault();
        });
        this.document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            const files = [];
            for (let i = 0; i < e.dataTransfer.files.length; i++) {
                const file = e.dataTransfer.files[i].path;
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
        const owner = electron.remote.getCurrentWindow();
        const options = {
            type: 'error',
            message: message,
            detail: detail,
        };
        electron.remote.dialog.showMessageBoxSync(owner, options);
    }

    confirm(message, detail) {
        const owner = electron.remote.getCurrentWindow();
        const options = {
            type: 'question',
            message: message,
            detail: detail,
            buttons: ['Yes', 'No'],
            defaultId: 0,
            cancelId: 1
        };
        const result = electron.remote.dialog.showMessageBoxSync(owner, options);
        return result == 0;
    }

    require(id) {
        try {
            return Promise.resolve(require(id));
        }
        catch (error) {
            return Promise.reject(error);
        }
    }

    save(name, extension, defaultPath, callback) {
        const owner = electron.remote.BrowserWindow.getFocusedWindow();
        const showSaveDialogOptions = {
            title: 'Export Tensor',
            defaultPath: defaultPath,
            buttonLabel: 'Export',
            filters: [ { name: name, extensions: [ extension ] } ]
        };
        const selectedFile = electron.remote.dialog.showSaveDialogSync(owner, showSaveDialogOptions);
        if (selectedFile) {
            callback(selectedFile);
        }
    }

    export(file, blob) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const data = new Uint8Array(e.target.result);
            fs.writeFile(file, data, null, (err) => {
                if (err) {
                    this.exception(err, false);
                    this.error('Error writing file.', err.message);
                }
            });
        };

        let err = null;
        if (!blob) {
            err = new Error("Export blob is '" + JSON.stringify(blob) + "'.");
        }
        else if (!(blob instanceof Blob)) {
            err = new Error("Export blob type is '" + (typeof blob) + "'.");
        }

        if (err) {
            this.exception(err, false);
            this.error('Error exporting image.', err.message);
        }
        else {
            reader.readAsArrayBuffer(blob);
        }
    }

    request(base, file, encoding) {
        return new Promise((resolve, reject) => {
            const pathname = path.join(base || __dirname, file);
            fs.exists(pathname, (exists) => {
                if (!exists) {
                    reject(new Error("File not found '" + file + "'."));
                }
                else {
                    fs.readFile(pathname, encoding, (err, data) => {
                        if (err) {
                            reject(err);
                        }
                        else {
                            resolve(data);
                        }
                    });
                }
            });
        });
    }

    openURL(url) {
        electron.shell.openExternal(url);
    }

    exception(error, fatal) {
        if (this._telemetry && error && error.telemetry !== false) {
            try {
                const description = [];
                description.push((error && error.name ? (error.name + ': ') : '') + (error && error.message ? error.message : '(null)'));
                if (error.stack) {
                    const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                    if (match) {
                        description.push(match[1] + '(' + match[2].split('/').pop().split('\\').pop() + ')');
                    }
                }

                const params = {
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
                const params = {
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
                const params = {
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
            this._view.show('welcome spinner');
            this._readFile(file).then((buffer) => {
                const context = new ElectonContext(this, path.dirname(file), path.basename(file), buffer);
                this._view.open(context).then((model) => {
                    this._view.show(null);
                    if (model) {
                        this._update('path', file);
                    }
                    this._update('show-attributes', this._view.showAttributes);
                    this._update('show-initializers', this._view.showInitializers);
                    this._update('show-names', this._view.showNames);
                }).catch((error) => {
                    if (error) {
                        this._view.show(null);
                        this.exception(error, false);
                        this.error(error.name, error.message);
                        this._update('path', null);
                    }
                    this._update('show-attributes', this._view.showAttributes);
                    this._update('show-initializers', this._view.showInitializers);
                    this._update('show-names', this._view.showNames);
                });
            }).catch((error) => {
                this.exception(error, false);
                this._view.show(null);
                this.error('Error while reading file.', error.message);
                this._update('path', null);
            });
        }
    }

    _readFile(file) {
        return new Promise((resolve, reject) => {
            fs.exists(file, (exists) => {
                if (!exists) {
                    reject(new Error('The file \'' + file + '\' does not exist.'));
                }
                else {
                    fs.readFile(file, null, (err, buffer) => {
                        if (err) {
                            reject(err);
                        }
                        else {
                            resolve(buffer);
                        }
                    });
                }
            });
        });
    }

    _request(url, headers, encoding, timeout) {
        return new Promise((resolve, reject) => {
            const httpModule = url.split(':').shift() === 'https' ? https : http;
            const options = {
                headers: headers
            };
            const request = httpModule.get(url, options, (response) => {
                if (response.statusCode !== 200) {
                    const err = new Error("The web request failed with status code " + response.statusCode + " at '" + url + "'.");
                    err.type = 'error';
                    err.url = url;
                    err.status = response.statusCode;
                    reject(err);
                }
                else {
                    let data = '';
                    response.on('data', (chunk) => {
                        data += chunk;
                    });
                    response.on('err', (err) => {
                        reject(err);
                    });
                    response.on('end', () => {
                        resolve(data);
                    });
                }
            }).on("error", (err) => {
                reject(err);
            });
            if (timeout) {
                request.setTimeout(timeout, () => {
                    request.abort();
                    const err = new Error("The web request timed out at '" + url + "'.");
                    err.type = 'timeout';
                    err.url = url;
                    reject(err);
                });
            }
        });
    }

    _getConfiguration(name) {
        const configuration = electron.remote.getGlobal('global').application.service('configuration');
        return configuration && configuration.has(name) ? configuration.get(name) : undefined;
    }

    _setConfiguration(name, value) {
        const configuration = electron.remote.getGlobal('global').application.service('configuration');
        if (configuration) {
            configuration.set(name, value);
        }
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

    request(file, encoding) {
        return this._host.request(this._folder, file, encoding);
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }
}

window.__view__ = new view.View(new host.ElectronHost());
