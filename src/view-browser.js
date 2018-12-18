/*jshint esversion: 6 */

var host = host || {};

host.BrowserHost = class {

    constructor() {
        if (!window.ga) {
            window.GoogleAnalyticsObject = 'ga';
            window.ga = window.ga || function() {
                window.ga.q = window.ga.q || [];
                window.ga.q.push(arguments);
            };
            window.ga.l = 1 * new Date();
        }
        window.ga('create', 'UA-54146-13', 'auto');

        window.addEventListener('error', (e) => {
            this.exception(e.error, true);
        });
        window.eval = () => {
            throw new Error('window.eval() not supported.');
        };
    }

    get document() {
        return window.document;
    }

    get name() {
        return 'Netron';
    }

    get version() {
        return this._version;
    }

    get type() {
        return this._type;
    }

    initialize(view) {
        this._view = view;

        window.addEventListener('keydown', (e) => {
            this._keyHandler(e);
        });

        var meta = {};
        Array.from(document.getElementsByTagName('meta')).forEach((element) => {
            if (element.content) {
                meta[element.name] = meta[element.name] || [];
                meta[element.name].push(element.content);
            }
        });

        this._version = meta.version ? meta.version[0] : null;
        this._type = meta.type ? meta.type[0] : 'Browser';

        this._zoom = this._getQueryParameter('zoom') || 'd3';

        if (meta.file) {
            this._openModel(meta.file[0], null);
            return;
        }

        var urlParam = this._getQueryParameter('url');
        if (urlParam) {
            this._openModel(urlParam, this._getQueryParameter('identifier') || null);
            return;
        }

        var gistParam = this._getQueryParameter('gist');
        if (gistParam) {
            this._openGist(gistParam);
            return;
        }

        this._view.show('Welcome');
        var openFileButton = document.getElementById('open-file-button');
        var openFileDialog = document.getElementById('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.addEventListener('click', (e) => {
                openFileDialog.value = '';
                openFileDialog.click();
            });
            openFileDialog.addEventListener('change', (e) => {
                if (e.target && e.target.files && e.target.files.length == 1) {
                    this._openFile(e.target.files[0]);
                }
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
                var file = e.dataTransfer.files[0];
                if (file.name.split('.').length > 1) {
                    this._openFile(file);
                }
            }
        });
    }

    environment(name) {
        if (name == 'zoom') {
            return this._zoom;
        }
        return null;
    }

    error(message, detail) {
        alert((message == 'Error' ? '' : message + ' ') + detail);
    }

    confirm(message, detail) {
        return confirm(message + ' ' + detail);
    }

    require(id, callback) {
        var url = this._url(id + '.js');
        window.__modules__ = window.__modules__ || {};
        if (window.__modules__[url]) {
            callback(null, window.__exports__[url]);
            return;
        }
        window.module = { exports: {} };
        var script = document.createElement('script');
        script.setAttribute('id', id);
        script.setAttribute('type', 'text/javascript');
        script.setAttribute('src', url);
        script.onload = () => {
            var exports = window.module.exports;
            delete window.module;
            window.__modules__[id] = exports;
            callback(null, exports);
        };
        script.onerror = (e) => {
            delete window.module;
            callback(new Error('The script \'' + e.target.src + '\' failed to load.'), null);
        };
        document.head.appendChild(script);
    }

    save(name, extension, defaultPath, callback) {
        callback(defaultPath + '.' + extension);
    }

    export(file, blob) {
        var element = document.createElement('a');
        element.download = file;
        element.href = URL.createObjectURL(blob);
        document.body.appendChild(element);
        element.click();
        document.body.removeChild(element);
    }

    request(base, file, encoding, callback) {
        var url = base ? (base + '/' + file) : this._url(file);
        var request = new XMLHttpRequest();
        if (encoding == null) {
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

    exception(err, fatal) {
        if (window.ga && this.version) {
            var description = [];
            description.push((err && err.name ? (err.name + ': ') : '') + (err && err.message ? err.message : '(null)'));
            if (err.stack) {
                var match = err.stack.match(/\n    at (.*)\((.*)\)/);
                if (match) {
                    description.push(match[1] + '(' + match[2].split('/').pop() + ')');
                }
                else {
                    description.push(err.stack.split('\n').shift());
                }
            }
            window.ga('send', 'exception', {
                exDescription: description.join(' @ '),
                exFatal: fatal,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    screen(name) {
        if (window.ga && this.version) {
            window.ga('send', 'screenview', {
                screenName: name,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    event(category, action, label, value) {
        if (window.ga && this.version) {
            window.ga('send', 'event', {
                eventCategory: category,
                eventAction: action,
                eventLabel: label,
                eventValue: value,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    _url(file) {
        var url = file;
        if (window && window.location && window.location.href) {
            var location = window.location.href.split('?').shift();
            if (location.endsWith('.html')) {
                location = location.split('/').slice(0, -1).join('/');
            }
            if (location.endsWith('/')) {
                location = location.slice(0, -1);
            }
            url = location + '/' + file;
        }
        return url;
    }

    _getQueryParameter(name) {
        var url = window.location.href;
        name = name.replace(/[\[\]]/g, "\\$&");
        var regex = new RegExp("[?&]" + name + "(=([^&#]*)|&|#|$)");
        var results = regex.exec(url);
        if (!results) {
            return null;
        }
        if (!results[2]) {
            return '';
        }
        return decodeURIComponent(results[2].replace(/\+/g, " "));
    }

    _openModel(url, identifier) {
        this._view.show('Spinner');
        var request = new XMLHttpRequest();
        request.responseType = 'arraybuffer';
        request.onload = () => {
            if (request.status == 200) {
                var buffer = new Uint8Array(request.response);
                var context = new BrowserContext(this, url, identifier, buffer);
                this._view.openContext(context, (err, model) => {
                    if (err) {
                        this.exception(err, false);
                        this.error(err.name, err.message);
                    }
                    if (model) {
                        document.title = identifier || url.split('/').pop();
                    }
                });
            }
            else {
                this.error('Model load request failed.', request.status);
            }
        };
        request.onerror = () => {
            this.error('Error while requesting model.', request.status);
        };
        request.open('GET', url + ((/\?/).test(url) ? "&" : "?") + (new Date()).getTime(), true);
        request.send();
    }

    _openFile(file) {
        this._view.show('Spinner');
        this._openBuffer(file, (err, model) => {
            this._view.show(null);
            if (err) {
                this.exception(err, false);
                this.error(err.name, err.message);
            }
            if (model) {
                document.title = file.name;
            }
        });
    }

    _openGist(gist) {
        this._view.show('Spinner');
        var url = 'https://api.github.com/gists/' + gist;
        var request = new XMLHttpRequest();
        request.onload = () => {
            var identifier = null;
            var buffer = null;
            var json = JSON.parse(request.response);
            if (json.message) {
                this.error('Error while loading Gist.', json.message);
                return;
            }
            if (json.files) {
                Object.keys(json.files).forEach((key) => {
                    var file = json.files[key];
                    identifier = file.filename;
                    var extension = identifier.split('.').pop().toLowerCase();
                    if (extension == 'json' || extension == 'pbtxt' || extension == 'prototxt') {
                        var encoder = new TextEncoder();
                        buffer = encoder.encode(file.content);
                    }
                });
            }
            if (buffer == null || identifier == null) {
                this.error('Error while loading Gist.', 'Gist does not contain model file.');
                return;
            }
            var context = new BrowserContext(this, '', identifier, buffer);
            this._view.openContext(context, (err, model) => {
                if (err) {
                    this.exception(err, false);
                    this.error(err.name, err.message);
                }
                if (model) {
                    document.title = identifier;
                }
            });
        };
        request.onerror = () => {
            this.error('Error while requesting Gist.', request.status);
        };
        request.open('GET', url, true);
        request.send();
    }

    _openBuffer(file, callback) {
        var size = file.size;
        var reader = new FileReader();
        reader.onload = (e) => {
            var buffer = new Uint8Array(e.target.result);
            var context = new BrowserContext(this, '', file.name, buffer);
            this._view.openContext(context, (err, model) => {
                callback(err, model);
            });
        };
        reader.onerror = (e) => {
            e = e || window.event;
            var message = '';
            switch(e.target.error.code) {
                case e.target.error.NOT_FOUND_ERR:
                    message = 'File not found.';          
                    break;
                case e.target.error.NOT_READABLE_ERR:
                    message = 'File not readable.';
                    break;
                case e.target.error.SECURITY_ERR:
                    message = 'File access denied.';
                    break;
                default:
                    message = "File read error '" + e.target.error.code.toString() + "'.";
                    break;
            }
            callback(new Error(message), null);
        };
        reader.readAsArrayBuffer(file);
    }

    _keyHandler(e) {
        if (!e.altKey && !e.shiftKey && (e.ctrlKey || e.metaKey)) {
            switch (e.keyCode) {
                case 70: // F
                    this._view.find();
                    e.preventDefault();
                    break;
                case 68: // D
                    this._view.toggleAttributes();
                    e.preventDefault();
                    break;
                case 73: // I
                    this._view.toggleInitializers();
                    e.preventDefault();
                    break;
                case 85: // U
                    this._view.toggleNames();
                    e.preventDefault();
                    break;
                case 13: // Return
                    document.getElementById('model-properties-button').click();
                    e.preventDefault();
                    break;
                case 8: // Backspace
                    this._view.resetZoom();
                    e.preventDefault();
                    break;
                case 38: // Up
                    document.getElementById('zoom-in-button').click();
                    e.preventDefault();
                    break;
                case 40: // Down
                    document.getElementById('zoom-out-button').click();
                    e.preventDefault();
                    break;
            }
        }
        if (e.shiftKey && (e.ctrlKey || e.metaKey)) {
            switch (e.keyCode) {
                case 69: // E
                    if (e.altKey) {
                        this._view.export(document.title + '.svg');
                    }
                    else {
                        this._view.export(document.title + '.png');
                    }
                    e.preventDefault();
                    break;
            }
        }
    }
};

if (typeof TextDecoder === "undefined") {
    TextDecoder = function TextDecoder(encoding) {
        this._encoding = encoding;
    };
    TextDecoder.prototype.decode = function decode(buffer) {
        var result = '';
        var length = buffer.length;
        var i = 0;
        switch (this._encoding) {
            case 'utf-8':
                while (i < length) {
                    var c = buffer[i++];
                    switch(c >> 4)
                    { 
                        case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
                            result += String.fromCharCode(c);
                            break;
                        case 12: case 13:
                            c2 = buffer[i++];
                            result += String.fromCharCode(((c & 0x1F) << 6) | (c2 & 0x3F));
                            break;
                        case 14:
                            var c2 = buffer[i++];
                            var c3 = buffer[i++];
                            result += String.fromCharCode(((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | ((c3 & 0x3F) << 0));
                            break;
                    }
                }
                break;
            case 'ascii':
                while (i < length) {
                    result += String.fromCharCode(buffer[i++]);
                }
                break;
        }
        return result;
    };
}

if (typeof TextEncoder === "undefined") {
    TextEncoder = function TextEncoder() {
    };
    TextEncoder.prototype.encode = function encode(str) {
        "use strict";
        var length = str.length, resPos = -1;
        var resArr = typeof Uint8Array === "undefined" ? new Array(length * 2) : new Uint8Array(length * 3);
        for (var point = 0, nextcode = 0, i = 0; i !== length; ) {
            point = str.charCodeAt(i);
            i += 1;
            if (point >= 0xD800 && point <= 0xDBFF) {
                if (i === length) {
                    resArr[resPos += 1] = 0xef; resArr[resPos += 1] = 0xbf;
                    resArr[resPos += 1] = 0xbd; break;
                }
                nextcode = str.charCodeAt(i);
                if (nextcode >= 0xDC00 && nextcode <= 0xDFFF) {
                    point = (point - 0xD800) * 0x400 + nextcode - 0xDC00 + 0x10000;
                    i += 1;
                    if (point > 0xffff) {
                        resArr[resPos += 1] = (0x1e<<3) | (point>>>18);
                        resArr[resPos += 1] = (0x2<<6) | ((point>>>12)&0x3f);
                        resArr[resPos += 1] = (0x2<<6) | ((point>>>6)&0x3f);
                        resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
                        continue;
                    }
                } else {
                    resArr[resPos += 1] = 0xef; resArr[resPos += 1] = 0xbf;
                    resArr[resPos += 1] = 0xbd; continue;
                }
            }
            if (point <= 0x007f) {
                resArr[resPos += 1] = (0x0<<7) | point;
            } else if (point <= 0x07ff) {
                resArr[resPos += 1] = (0x6<<5) | (point>>>6);
                resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
            } else {
                resArr[resPos += 1] = (0xe<<4) | (point>>>12);
                resArr[resPos += 1] = (0x2<<6) | ((point>>>6)&0x3f);
                resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
            }
        }
        if (typeof Uint8Array!=="undefined") {
            return new Uint8Array(resArr.buffer.slice(0, resPos+1));
        }
        else {
            return resArr.length === resPos + 1 ? resArr : resArr.slice(0, resPos + 1);
        }
    };
    TextEncoder.prototype.toString = function() { 
        return "[object TextEncoder]"
    };
    try {
        Object.defineProperty(TextEncoder.prototype,"encoding", {
            get:function() {
                if (TextEncoder.prototype.isPrototypeOf(this)) {
                    return"utf-8";
                }
                else {
                    throw TypeError("Illegal invocation");
                }
            }
        });
    }
    catch (e) {
        TextEncoder.prototype.encoding = "utf-8";
    }
    if (typeof Symbol !== "undefined") {
        TextEncoder.prototype[Symbol.toStringTag] = "TextEncoder";
    }
}

if (!HTMLCanvasElement.prototype.toBlob) {
    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
        var canvas = this;
        setTimeout(function() {
            var data = atob(canvas.toDataURL(type, quality).split(',')[1]);
            var length = data.length;
            var buffer = new Uint8Array(length);
            for (var i = 0; i < length; i++) {
                buffer[i] = data.charCodeAt(i);
            }
            callback(new Blob([ buffer ], { type: type || 'image/png' }));
        });
    };
}

class BrowserContext {

    constructor(host, url, identifier, buffer) {
        this._tags = {};
        this._host = host;
        this._buffer = buffer;
        if (identifier) {
            this._identifier = identifier;
            this._base = url;
            if (this._base.endsWith('/')) {
                this._base.substring(0, this._base.length - 1);
            }
        }
        else {
            url = url.split('/');
            this._identifier = url.pop();
            this._base = url.join('/');
        }
    }

    request(file, encoding, callback) {
        this._host.request(this._base, file, encoding, (err, buffer) => {
            callback(err, buffer);
        });
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }

    get text() {
        if (!this._text) {
            var decoder = new TextDecoder('utf-8');
            this._text = decoder.decode(this._buffer);
        }
        return this._text;
    }

    tags(extension) {
        var tags = this._tags[extension];
        if (!tags) {
            tags = {};
            try {
                var reader = null;
                switch (extension) {
                    case 'pbtxt':
                        reader = protobuf.TextReader.create(this.text);
                        reader.start(false);
                        while (!reader.end(false)) {
                            var tag = reader.tag();
                            tags[tag] = true;
                            reader.skip();
                        }
                        break;
                    case 'pb':
                        reader = new protobuf.Reader.create(this.buffer);
                        while (tags != null && reader.pos < reader.len) {
                            var tagType = reader.uint32();
                            tags[tagType >>> 3] = tagType & 7;
                            switch (tagType & 7) {
                                case 0: reader.int64(); break;
                                case 1: reader.fixed64(); break;
                                case 2: reader.bytes(); break;
                                default: tags = {}; break;
                            }
                        }
                        break;
                }
            }
            catch (error) {
                tags = {};
            }
            this._tags[extension] = tags;
        }
        return tags;
    }
}

window.__view__ = new view.View(new host.BrowserHost());
