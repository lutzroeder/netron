/*jshint esversion: 6 */

var electron = require('electron');
var fs = require('fs');
var path = require('path');

class ElectronHost {

    constructor() {
    }

    initialize(view) {
        this._view = view;
        this._view.show('welcome');
        
        electron.ipcRenderer.on('open-file', (event, data) => {
            var file = data.file;
            if (file) {
                this._view.show('spinner');
                this.openBuffer(file);
            }
        });
    
        var openFileButton = document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.opacity = 1;
            openFileButton.addEventListener('click', (e) => {
                openFileButton.style.opacity = 0;
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

    openBuffer(file) {
        fs.exists(file, (exists) => {
            if (!exists) {
                this._view.openBuffer('File not found.', null, null);
            }
            else {
                fs.stat(file, (err, stats) => {
                    if (err) {
                        this._view.openBuffer(err, null, null);
                    }
                    else {
                        var size = stats.size;
                        var buffer = new Uint8Array(size);
                        fs.open(file, 'r', (err, fd) => {
                            if (err) {
                                this._view.openBuffer(err, null, null);
                            }
                            else {
                                fs.read(fd, buffer, 0, size, 0, (err, bytesRead, buffer) => {
                                    if (err) {
                                        this._view.openBuffer(err, null, null);
                                    }
                                    else {
                                        fs.close(fd, (err) => {
                                            if (err) {
                                                this._view.openBuffer(err, null);
                                            }
                                            else {
                                                this._view.openBuffer(null, buffer, path.basename(file));
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
