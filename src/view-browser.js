/*jshint esversion: 6 */

var hostService = new BrowserHostService();

function BrowserHostService() {
}

BrowserHostService.prototype.showError = function(message) {
    alert(message);
};

BrowserHostService.prototype.request = function(file, callback) {
    var request = new XMLHttpRequest();
    request.onload = function() {
        if (request.status == 200) {
            callback(null, this.responseText);
        }
        else {
            callback(request.status, null);
        }
    };
    request.onerror = function () {
        callback(request.status, null);
    };
    request.open('GET', file, true);
    request.send();
};

BrowserHostService.prototype.initialize = function(callback) {
    this.callback = callback;

    updateView('spinner');
    
    var request = new XMLHttpRequest();
    request.responseType = 'arraybuffer';
    request.onload = () => {
        if (request.status == 200) {
            this.callback(null, new Uint8Array(request.response), document.title);
        }
        else {
            this.callback(request.status, null);
        }
    };
    request.onerror = () => {
        this.callback(request.status, null);
    };
    request.open('GET', '/data', true);
    request.send();
};
