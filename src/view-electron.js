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
        
        electron.ipcRenderer.on('open-file', (event, data) => {
            var file = data.file;
            if (file) {
                this._view.show('spinner');
                this.openFile(file, (err) => {
                    if (err) {
                        this.showError(err.toString());
                        this._view.show(null);
                        return;
                    }
                    data.windowId = electron.remote.getCurrentWindow().id;
                    electron.ipcRenderer.send('update-window', data);
                });
            }
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
            var files = e.dataTransfer.files;
            for (var i = 0; i < files.length; i++) {
                this.dropFile(files[i].path, i == 0);
            }
            return false;
        });

        document.addEventListener('keydown', function(e) {
            if (e.which == 123) {
                electron.remote.getCurrentWindow().toggleDevTools();
            }
        });
    }

    dropFile(file, drop) {
        var data = { file: file };
        if (drop) {
            data.windowId = electron.remote.getCurrentWindow().id;
        }
        electron.ipcRenderer.send('drop-file', data);
    }

    showError(message) {
        electron.remote.dialog.showErrorBox(electron.remote.app.getName(), message);        
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
