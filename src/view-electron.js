/*jshint esversion: 6 */

var electron = require('electron');
var fs = require('fs');
var path = require('path');

class ElectronHost {

    constructor() {
    }

    get name() {
        return electron.remote.app.getName();
    }

    initialize(view) {
        this._view = view;
        this._view.show('welcome');

        electron.ipcRenderer.on('open', (event, data) => {
            this._openFile(data.file);
        });
        electron.ipcRenderer.on('export', (event, data) => {
            this._view.export(data.file);
        });
        electron.ipcRenderer.on('cut', (event, data) => {
            this._view.cut();
        });
        electron.ipcRenderer.on('copy', (event, data) => {
            this._view.copy();
        });
        electron.ipcRenderer.on('paste', (event, data) => {
            this._view.paste();
        });
        electron.ipcRenderer.on('selectall', (event, data) => {
            this._view.selectAll();
        });
        electron.ipcRenderer.on('toggle-details', (event, data) => {
            this._view.toggleDetails();
            this._update('show-details', this._view.showDetails);
        });
        electron.ipcRenderer.on('toggle-names', (event, data) => {
            this._view.toggleNames();
            this._update('show-names', this._view.showNames);
        });
        electron.ipcRenderer.on('zoom-in', (event, data) => {
            document.getElementById('zoom-in-button').click();
        });
        electron.ipcRenderer.on('zoom-out', (event, data) => {
            document.getElementById('zoom-out-button').click();
        });
        electron.ipcRenderer.on('reset-zoom', (event, data) => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', (event, data) => {
            document.getElementById('model-properties-button').click();
        });
        electron.ipcRenderer.on('find', (event, data) => {
            this._view.find();
        });

        var openFileButton = document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.opacity = 1;
            openFileButton.addEventListener('click', (e) => {
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
                files.push(e.dataTransfer.files[i].path);
            }
            electron.ipcRenderer.send('drop-files', { files: files });
            return false;
        });
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

    import(file, callback) {
        var pathname = path.join(__dirname, file);
        for (var i = 0; i < document.scripts.length; i++) {
            if (pathname == document.scripts[i]) {
                callback(null);
                return;
            }
        }
        var script = document.createElement('script');
        script.onload = () => {
            callback(null);
        };
        script.onerror = (e) => {
            callback(new Error('The script \'' + e.target.src + '\' failed to load.'));
        };
        script.setAttribute('type', 'text/javascript');
        script.setAttribute('src', pathname);
        document.head.appendChild(script);
    }

    export(file, data, mimeType) {
        var encoding = 'utf-8';
        if (mimeType == 'image/png') {
            try
            {
                var nativeImage = electron.nativeImage.createFromDataURL(data);
                data = nativeImage.toPNG();
                encoding = 'binary';
            }
            catch (e)
            {
                this.error('Export failure.', e);
                return;
            }    
        }
        fs.writeFile(file, data, encoding, (err) => {
            if (err) {
                this.error('Export write failure.', err);
            }
        });
    }

    request(file, callback) {
        var pathname = path.join(__dirname, file);
        fs.exists(pathname, (exists) => {
            if (!exists) {
                callback('File not found.', null);
            }
            else {
                fs.readFile(pathname, (err, data) => {
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

    _openFile(file) {
        if (file) {
            this._view.show('spinner');
            this._readFile(file, (err, buffer) => {
                if (err) {
                    this._view.show(null);
                    this.error('Error while reading file.', err.message);
                    this._update('path', null);
                    return;
                }
                this._view.openBuffer(buffer, path.basename(file), (err, model) => {
                    this._view.show(null);
                    if (err) {
                        this.error(err.name, err.message);
                        this._update('path', null);
                    }
                    if (model) {
                        this._update('path', file);
                    }
                    this._update('show-details', this._view.showDetails);
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
}

window.host = new ElectronHost();
