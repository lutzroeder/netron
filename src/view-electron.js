
const electron = require('electron');
const fs = require('fs');
const path = require('path');

var hostService = new ElectronHostService();

function ElectronHostService()
{
    var self = this;

    electron.ipcRenderer.on('open-file', function(event, data) {
        openFile(data['file']);
    });

    window.addEventListener('load', function(e) {
        var openFileButton = document.getElementById('open-file-button');
        if (openFileButton) {
            openFileButton.style.display = 'block';
            openFileButton.addEventListener('click', function(e) {
                self.openFileDialog();
            });
        }
        document.addEventListener('dragover', function(e) {
            e.preventDefault();
        });
        document.addEventListener('drop', function(e) {
            e.preventDefault();
        });
        document.body.addEventListener('drop', function(e) { 
            e.preventDefault();
            var files = e.dataTransfer.files;
            for (var i = 0; i < files.length; i++) {
                self.openFile(files[i].path, i == 0);
            }
            return false;
        });
    });

    const contextMenu = new electron.remote.Menu();
    contextMenu.append(new electron.remote.MenuItem({
        label: 'Properties...', 
        click: function() { showProperties(); }
    }));
    
    window.addEventListener('contextmenu', function(e) {
        e.preventDefault();
        if (contextMenu) {
            contextMenu.popup(electron.remote.getCurrentWindow(), { async: true });
        }
    }, false);
}

ElectronHostService.prototype.openFile = function(file, drop) {
    var data = {};
    data['file'] = file;
    if (drop) {
        data['window'] = electron.remote.getCurrentWindow().id;
    } 
    electron.ipcRenderer.send('open-file', data);
}

ElectronHostService.prototype.openFileDialog = function() {
    electron.ipcRenderer.send('open-file-dialog', {});
}

ElectronHostService.prototype.showError = function(message) {
    electron.remote.dialog.showErrorBox(electron.remote.app.getName(), message);
}

ElectronHostService.prototype.getResource = function(file, callback) {
    var file = path.join(__dirname, file);
    if (fs.existsSync(file)) {
        var data = fs.readFileSync(file);
        if (data) {
            callback(data);
            return;
        }
    }
    callback(null);
}

ElectronHostService.prototype.openBuffer = function(file, callback) {
    fs.exists(file, function(exists) {
        if (exists) {
            fs.stat(file, function(err, stats) {
                if (err) {
                    callback(err, null);
                }
                else {
                    var size = stats.size;
                    var buffer = new Uint8Array(size);
                    fs.open(file, 'r', function(err, fd) {
                        if (err) {
                            callback(err, null);
                        }
                        else {
                            fs.read(fd, buffer, 0, size, 0, function(err, bytesRead, buffer) {
                                if (err) {
                                    callback(err, null);
                                }
                                else {
                                    fs.close(fd, function(err) {
                                        if (err) {
                                            callback(err, null);
                                        }
                                        else {
                                            callback(null, buffer);
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
