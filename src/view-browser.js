
var hostService = new BrowserHostService();

function BrowserHostService()
{
    var openFileButton = document.getElementById('open-file-button');
    if (openFileButton) {
        openFileButton.style.display = 'none';
    }

    /*
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
        updateView('clock');
        var request = new XMLHttpRequest();
        request.responseType = 'arraybuffer';
        request.onload = function () {
            self.callback(null, new Uint8Array(request.response));
        }
        request.onerror = function () {
            self.callback(request.status, null);
        }
        request.open("GET", "model");
        request.send();
    });
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

BrowserHostService.prototype.showError = function(message) {
    alert(message);
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
