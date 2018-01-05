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
        if (file.endsWith('.pb')) {
            request.responseType = 'arraybuffer';

        }
        request.onload = () => {
            if (request.status == 200) {
                if (request.responseType == 'arraybuffer') {
                    callback(null, new Uint8Array(request.response));
                }
                else {
                    callback(null, request.responseText);
                }
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

    openURL(url) {
        window.open(url, '_target');
    }
}

var hostService = new BrowserHostService();