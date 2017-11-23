
var hostService = new BrowserHostService();

function BrowserHostService()
{
    var openFileButton = document.getElementById('open-file-button');
    if (openFileButton) {
        openFileButton.style.display = 'none';
    }

    /*
    electron.ipcRenderer.on('open-file', function(event, data) {
        openFile(data['file']);
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
    */

    var self = this;

    window.addEventListener('load', function(e) {
        var request = new XMLHttpRequest();
        updateView('clock');
        request.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
                var base64 = this.responseText;
                var raw = window.atob(base64);
                var length = raw.length;
                var buffer = new Uint8Array(new ArrayBuffer(length));
                for(var i = 0; i < length; i++) {
                    buffer[i] = raw.charCodeAt(i);
                }
                self.callback(null, buffer);
            }
        };
        request.open("GET", "model", true);
        request.send();
    });

    // TODO error
}

BrowserHostService.prototype.openFile = function(file, drop) {
    /*
    var data = {};
    data['file'] = file;
    if (drop) {
        data['window'] = electron.remote.getCurrentWindow().id;
    } 
    electron.ipcRenderer.send('open-file', data);
    */
}

BrowserHostService.prototype.openFileDialog = function() {
    /*
    electron.ipcRenderer.send('open-file-dialog', {});
    */
}

BrowserHostService.prototype.showError = function(message) {
    /*
    electron.remote.dialog.showErrorBox(electron.remote.app.getName(), message);
    */
}

BrowserHostService.prototype.getResource = function(file, callback) {

    var request = new XMLHttpRequest();
    request.onreadystatechange = function() {
        if (this.readyState == 4 && this.status == 200) {
            callback(null, this.responseText);
        }
    }
    request.open("GET", file, true);
    request.send();

    // TODO error
}

BrowserHostService.prototype.registerCallback = function(callback) {
    this.callback = callback;
}

BrowserHostService.prototype.openBuffer = function(file) {
    /*
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
    */
}
