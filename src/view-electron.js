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
            var file = data.file;
            if (file) {
                this._view.show('spinner');
                this.openFile(file, (err) => {
                    if (err) {
                        this.showError(err.toString());
                        this._view.show(null);
                        this.update('path', null);
                        this.update('show-details', this._view.showDetails);
                        this.update('show-names', this._view.showNames);
                        return;
                    }
                    this.update('path', file);
                    this.update('show-details', this._view.showDetails);
                    this.update('show-names', this._view.showNames);
                });
            }
        });

        electron.ipcRenderer.on('export', (event, data) => {
            this._view.export(data.file);
        });
        electron.ipcRenderer.on('copy', (event, data) => {
            this._view.copy();
        });
        electron.ipcRenderer.on('toggle-details', (event, data) => {
            this._view.toggleDetails();
            this.update('show-details', this._view.showDetails);
        });
        electron.ipcRenderer.on('toggle-names', (event, data) => {
            this._view.toggleNames();
            this.update('show-names', this._view.showNames);
        });
        electron.ipcRenderer.on('zoom-in', (event, data) => {
            this._view.zoomIn();
        });
        electron.ipcRenderer.on('zoom-out', (event, data) => {
            this._view.zoomOut();
        });
        electron.ipcRenderer.on('reset-zoom', (event, data) => {
            this._view.resetZoom();
        });
        electron.ipcRenderer.on('show-properties', (event, data) => {
            this._view.showModelProperties();
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
            this.dropFiles(files);
            return false;
        });
    }

    update(name, value) {
        electron.ipcRenderer.send('update', { name: name, value: value });
    }

    dropFiles(files) {
        electron.ipcRenderer.send('drop-files', { files: files });
    }

    showError(message) {
        if (message) {
            electron.remote.dialog.showErrorBox(electron.remote.app.getName(), message);        
        }
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
                this.showError(e);
                return;
            }    
        }
        fs.writeFile(file, data, encoding, (err) => {
            if (err) {
                this.showError(err);
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

    openFile(file, callback) {
        fs.exists(file, (exists) => {
            if (!exists) {
                this._view.showError('File not found.');
            }
            else {
                fs.stat(file, (err, stats) => {
                    if (err) {
                        this._view.showError(err);
                    }
                    else {
                        var size = stats.size;
                        var buffer = new Uint8Array(size);
                        fs.open(file, 'r', (err, fd) => {
                            if (err) {
                                this._view.showError(err);
                            }
                            else {
                                fs.read(fd, buffer, 0, size, 0, (err, bytesRead, buffer) => {
                                    if (err) {
                                        this._view.showError(err);
                                    }
                                    else {
                                        fs.close(fd, (err) => {
                                            if (err) {
                                                this._view.showError(err);
                                            }
                                            else {
                                                this._view.openBuffer(buffer, path.basename(file), (err) => {
                                                    callback(err);
                                                });
                                            }
                                        });
                                    }
                                });
                            }
                        });
                    }
                });
            }
        });
    }
}

window.host = new ElectronHost();
