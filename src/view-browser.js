/*jshint esversion: 6 */

class BrowserHostService {

    constructor() {
    }

    initialize(callback) {
        this.callback = callback;

        var fileElement = Array.from(document.getElementsByTagName('meta')).filter(e => e.name == 'file').shift();
        if (fileElement) {
            updateView('spinner');
            var file = fileElement.content;
            var request = new XMLHttpRequest();
            request.responseType = 'arraybuffer';
            request.onload = () => {
                if (request.status == 200) {
                    document.title = file;
                    this.callback(null, new Uint8Array(request.response), file);
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
            return;
        }

        updateView('welcome');

        var openFileButton = document.getElementById('open-file-button');
        var openFileDialog = document.getElementById('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.style.opacity = 1;
            openFileDialog.addEventListener('change', (e) => {
                if (e.target && e.target.files && e.target.files.length == 1) {
                    openFileButton.style.opacity = 0;
                    this.openFile(e.target.files[0]);
                }
            });
            openFileButton.addEventListener('click', (e) => {
                openFileDialog.click();
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
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length == 1) {
                this.openFile(e.dataTransfer.files[0]);
            }
            return false;
        });
    }
    
    showError(message) {
        alert(message);
    }
    
    request(file, callback) {
        var url = file;
        if (window && window.location && window.location.href) {
            var location = window.location.href;
            if (location.endsWith('/')) {
                location = location.slice(0, -1);
            }
            url = location + file;
        }
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
        request.open('GET', url, true);
        request.send();
    }

    openURL(url) {
        window.open(url, '_target');
    }

    openFile(file, callback) {
        updateView('spinner');
        var size = file.size;
        var reader = new FileReader();
        reader.onloadend = () => {
            if (reader.error) {
                this.callback(reader.error, null, null);
            }
            else {
                var buffer = new Uint8Array(reader.result);
                this.callback(null, buffer, file.name);
                document.title = file.name;
            }
        };
        reader.readAsArrayBuffer(file);
    }
}

var hostService = new BrowserHostService();