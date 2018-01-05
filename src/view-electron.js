/*jshint esversion: 6 */

var electron = require('electron');
var fs = require('fs');
var path = require('path');

class ElectronHostService {

    constructor() {
    }

    initialize(callback) {
        this.callback = callback;
        
        updateView('welcome');
        
        electron.ipcRenderer.on('open-file', (event, data) => {
            var file = data.file;
            if (file) {
                updateView('spinner');
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
                this.openFile(files[i].path, i == 0);
            }
            return false;
        });  
        
        document.addEventListener('keydown', function(e) {
            if (e.which == 123) {
                electron.remote.getCurrentWindow().toggleDevTools();
            }
        });
    }

    openFile(file, drop) {
        var data = { file: file };
        if (drop) {
            data.window = electron.remote.getCurrentWindow().id;
        } 
        electron.ipcRenderer.send('open-file', data);
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
                this.callback('File not found.', null, null);
            }
            else {
                fs.stat(file, (err, stats) => {
                    if (err) {
                        this.callback(err, null, null);
                    }
                    else {
                        var size = stats.size;
                        var buffer = new Uint8Array(size);
                        fs.open(file, 'r', (err, fd) => {
                            if (err) {
                                this.callback(err, null, null);
                            }
                            else {
                                fs.read(fd, buffer, 0, size, 0, (err, bytesRead, buffer) => {
                                    if (err) {
                                        this.callback(err, null, null);
                                    }
                                    else {
                                        fs.close(fd, (err) => {
                                            if (err) {
                                                this.callback(err, null);
                                            }
                                            else {
                                                this.callback(null, buffer, path.basename(file));
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

var hostService = new ElectronHostService();
