
var hostService = new BrowserHostService();

function BrowserHostService()
{
    var self = this;

    window.addEventListener('load', function(e) {

        var propertiesButton = document.getElementById('properties-button');
        if (propertiesButton) {
            propertiesButton.addEventListener('click', function(e) {
                showModelProperties(modelService.activeModel);
            });
        }
    
        updateView('clock');
        var request = new XMLHttpRequest();
        request.responseType = 'arraybuffer';
        request.onload = function () {
            if (request.status == 200) {
                self.callback(null, new Uint8Array(request.response), 'TODO');
            }
            else {
                self.callback(request.status, null);
            }
        }
        request.onerror = function () {
            self.callback(request.status, null);
        }
        request.open('GET', 'model');
        request.send();
    });
}

BrowserHostService.prototype.showError = function(message) {
    alert(message);
}

BrowserHostService.prototype.getResource = function(file, callback) {
    var request = new XMLHttpRequest();
    request.onload = function() {
        callback(null, this.responseText);
    }
    request.open('GET', file, true);
    request.send();

    // TODO error
}

BrowserHostService.prototype.registerCallback = function(callback) {
    this.callback = callback;
}
