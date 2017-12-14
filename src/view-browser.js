/*jshint esversion: 6 */

class BrowserHostService {

    constructor() {
    }

    initialize(callback) {
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
    }
    
    showError(message) {
        alert(message);
    }
    
    request(file, callback) {
        var request = new XMLHttpRequest();
        request.onload = () => {
            if (request.status == 200) {
                callback(null, request.responseText);
            }
            else {
                callback(request.status, null);
            }
        };
        request.onerror = () => {
            callback(request.status, null);
        };
        request.open('GET', file, true);
        request.send();
    }
}

var hostService = new BrowserHostService();