/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */
/* eslint "no-global-assign": ["error", {"exceptions": [ "TextDecoder", "TextEncoder", "URLSearchParams" ] } ] */
/* global view */

var host = {};

host.BrowserHost = class {

    constructor() {
        window.eval = () => {
            throw new Error('window.eval() not supported.');
        };
        this._document = window.document;
        this._meta = {};
        for (const element of Array.from(this._document.getElementsByTagName('meta'))) {
            if (element.content) {
                this._meta[element.name] = this._meta[element.name] || [];
                this._meta[element.name].push(element.content);
            }
        }
        this._type = this._meta.type ? this._meta.type[0] : 'Browser';
        this._version = this._meta.version ? this._meta.version[0] : null;
        this._telemetry = this._version && this._version !== '0.0.0';
    }

    get document() {
        return this._document;
    }

    get version() {
        return this._version;
    }

    get type() {
        return this._type;
    }

    initialize(view) {
        this._view = view;
        return new Promise((resolve /*, reject */) => {
            const accept = () => {
                if (this._telemetry) {
                    const script = this.document.createElement('script');
                    script.setAttribute('type', 'text/javascript');
                    script.setAttribute('src', 'https://www.google-analytics.com/analytics.js');
                    script.onload = () => {
                        if (window.ga) {
                            window.ga.l = 1 * new Date();
                            window.ga('create', 'UA-54146-13', 'auto');
                            window.ga('set', 'anonymizeIp', true);
                        }
                        resolve();
                    };
                    script.onerror = () => {
                        resolve();
                    };
                    this.document.body.appendChild(script);
                }
                else {
                    resolve();
                }
            };
            const request = () => {
                this._view.show('welcome consent');
                const acceptButton = this.document.getElementById('consent-accept-button');
                if (acceptButton) {
                    acceptButton.addEventListener('click', () => {
                        this._setCookie('consent', 'yes', 30);
                        accept();
                    });
                }
            };
            if (this._getCookie('consent')) {
                accept();
            }
            else {
                this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 'utf-8', 2000).then((text) => {
                    try {
                        const json = JSON.parse(text);
                        const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                        if (json && json.country && !countries.indexOf(json.country) !== -1) {
                            this._setCookie('consent', Date.now(), 30);
                            accept();
                        }
                        else {
                            request();
                        }
                    }
                    catch (err) {
                        request();
                    }
                }).catch(() => {
                    request();
                });
            }
        });
    }

    start() {
        window.addEventListener('error', (e) => {
            this.exception(e.error, true);
        });

        const params = new URLSearchParams(window.location.search);

        this._zoom = params.get('zoom') || 'd3';

        this._menu = new host.Dropdown(this.document, 'menu-button', 'menu-dropdown');
        this._menu.add({
            label: 'Properties...',
            accelerator: 'CmdOrCtrl+Enter',
            click: () => this._view.showModelProperties()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Find...',
            accelerator: 'CmdOrCtrl+F',
            click: () => this._view.find()
        });
        this._menu.add({});
        this._menu.add({
            label: () => this._view.showAttributes ? 'Hide Attributes' : 'Show Attributes',
            accelerator: 'CmdOrCtrl+D',
            click: () => this._view.toggleAttributes()
        });
        this._menu.add({
            label: () => this._view.showInitializers ? 'Hide Initializers' : 'Show Initializers',
            accelerator: 'CmdOrCtrl+I',
            click: () => this._view.toggleInitializers()
        });
        this._menu.add({
            label: () => this._view.showNames ? 'Hide Names' : 'Show Names',
            accelerator: 'CmdOrCtrl+U',
            click: () => this._view.toggleNames()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Zoom In',
            accelerator: 'Shift+Up',
            click: () => this.document.getElementById('zoom-in-button').click()
        });
        this._menu.add({
            label: 'Zoom Out',
            accelerator: 'Shift+Down',
            click: () => this.document.getElementById('zoom-out-button').click()
        });
        this._menu.add({
            label: 'Actual Size',
            accelerator: 'Shift+Backspace',
            click: () => this._view.resetZoom()
        });
        this._menu.add({});
        this._menu.add({
            label: 'Export as PNG',
            accelerator: 'CmdOrCtrl+Shift+E',
            click: () => this._view.export(document.title + '.png')
        });
        this._menu.add({
            label: 'Export as SVG',
            accelerator: 'CmdOrCtrl+Alt+E',
            click: () => this._view.export(document.title + '.svg')
        });
        this.document.getElementById('menu-button').addEventListener('click', (e) => {
            this._menu.toggle();
            e.preventDefault();
        });
        this._menu.add({});
        this._menu.add({
            label: 'About ' + this.document.title,
            click: () => this._about()
        });

        this.document.getElementById('version').innerText = this.version;

        if (this._meta.file) {
            this._openModel(this._meta.file[0], null);
            return;
        }

        const url = params.get('url');
        if (url) {
            const identifier = params.get('identifier') || null;
            const location = url.replace(new RegExp('^https://github.com/([\\w]*/[\\w]*)/blob/([\\w/_.]*)(\\?raw=true)?$'), 'https://raw.githubusercontent.com/$1/$2');
            this._openModel(location, identifier);
            return;
        }

        const gist = params.get('gist');
        if (gist) {
            this._openGist(gist);
            return;
        }

        this._view.show('welcome');
        const openFileButton = this.document.getElementById('open-file-button');
        const openFileDialog = this.document.getElementById('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.addEventListener('click', () => {
                openFileDialog.value = '';
                openFileDialog.click();
            });
            openFileDialog.addEventListener('change', (e) => {
                if (e.target && e.target.files && e.target.files.length > 0) {
                    const files = Array.from(e.target.files);
                    const file = files.find((file) => this._view.accept(file.name));
                    if (file) {
                        this._open(file, files);
                    }
                }
            });
        }
        const downloadButton = this.document.getElementById('download-button');
        const downloadLink = this.document.getElementById('logo-github');
        if (downloadButton && downloadLink) {
            downloadButton.style.opacity = 1;
            downloadButton.addEventListener('click', () => {
                this.openURL(downloadLink.href);
            });
        }
        this.document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });
        this.document.addEventListener('drop', (e) => {
            e.preventDefault();
        });
        this.document.body.addEventListener('drop', (e) => {
            e.preventDefault();
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                const files = Array.from(e.dataTransfer.files);
                const file = files.find((file) => this._view.accept(file.name));
                if (file) {
                    this._open(file, files);
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

    require(id) {
        const url = this._url(id + '.js');
        window.__modules__ = window.__modules__ || {};
        if (window.__modules__[url]) {
            return Promise.resolve(window.__exports__[url]);
        }
        return new Promise((resolve, reject) => {
            window.module = { exports: {} };
            let script = document.createElement('script');
            script.setAttribute('id', id);
            script.setAttribute('type', 'text/javascript');
            script.setAttribute('src', url);
            script.onload = () => {
                const exports = window.module.exports;
                delete window.module;
                window.__modules__[id] = exports;
                resolve(exports);
            };
            script.onerror = (e) => {
                delete window.module;
                reject(new Error('The script \'' + e.target.src + '\' failed to load.'));
            };
            this.document.head.appendChild(script);
        });
    }

    save(name, extension, defaultPath, callback) {
        callback(defaultPath + '.' + extension);
    }

    export(file, blob) {
        let element = this.document.createElement('a');
        element.download = file;
        element.href = URL.createObjectURL(blob);
        this.document.body.appendChild(element);
        element.click();
        this.document.body.removeChild(element);
    }

    request(base, file, encoding) {
        const url = base ? (base + '/' + file) : this._url(file);
        return this._request(url, null, encoding);
    }

    openURL(url) {
        window.open(url, '_target');
    }

    exception(error, fatal) {
        if (this._telemetry && window.ga) {
            let description = [];
            description.push((error && error.name ? (error.name + ': ') : '') + (error && error.message ? error.message : '(null)'));
            if (error.stack) {
                const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                if (match) {
                    description.push(match[1] + '(' + match[2].split('/').pop() + ')');
                }
                else {
                    description.push(error.stack.split('\n').shift());
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
        if (this._telemetry && window.ga) {
            window.ga('send', 'screenview', {
                screenName: name,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    event(category, action, label, value) {
        if (this._telemetry && window.ga) {
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

    _request(url, headers, encoding, timeout) {
        return new Promise((resolve, reject) => {
            const request = new XMLHttpRequest();
            if (!encoding) {
                request.responseType = 'arraybuffer';
            }
            if (timeout) {
                request.timeout = timeout;
            }
            const error = (status) => {
                const err = new Error("The web request failed with status code " + status + " at '" + url + "'.");
                err.type = 'error';
                err.url = url;
                return err;
            };
            request.onload = () => {
                if (request.status == 200) {
                    if (request.responseType == 'arraybuffer') {
                        resolve(new Uint8Array(request.response));
                    }
                    else {
                        resolve(request.responseText);
                    }
                }
                else {
                    reject(error(request.status));
                }
            };
            request.onerror = (e) => {
                const err = error(request.status);
                err.type = e.type;
                reject(err);
            };
            request.ontimeout = () => {
                request.abort();
                const err = new Error("The web request timed out in '" + url + "'.");
                err.type = 'timeout';
                err.url = url;
                reject(err);
            };
            request.open('GET', url, true);
            if (headers) {
                for (const name of Object.keys(headers)) {
                    request.setRequestHeader(name, headers[name]);
                }
            }
            request.send();
        });
    }

    _url(file) {
        let url = file;
        if (window && window.location && window.location.href) {
            let location = window.location.href.split('?').shift();
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

    _openModel(url, identifier) {
        url = url + ((/\?/).test(url) ? "&" : "?") + (new Date()).getTime();
        this._view.show('welcome spinner');
        this._request(url).then((buffer) => {
            const context = new BrowserContext(this, url, identifier, buffer);
            this._view.open(context).then(() => {
                this.document.title = identifier || context.identifier;
            }).catch((err) => {
                if (err) {
                    this.exception(err, false);
                    this.error(err.name, err.message);
                    this._view.show('welcome');
                }
            });
        }).catch((err) => {
            this.error('Model load request failed.', err.message);
            this._view.show('welcome');
        });
    }

    _open(file, files) {
        this._view.show('welcome spinner');
        const context = new BrowserFileContext(file, files);
        context.open().then(() => {
            return this._view.open(context).then((model) => {
                this._view.show(null);
                this.document.title = files[0].name;
                return model;
            });
        }).catch((error) => {
            this._view.show(null);
            this.exception(error, false);
            this.error(error.name, error.message);
        });
    }

    _openGist(gist) {
        this._view.show('welcome spinner');
        const url = 'https://api.github.com/gists/' + gist;
        this._request(url, 'utf-8').then((text) => {
            let identifier = null;
            let buffer = null;
            const json = JSON.parse(text);
            if (json.message) {
                this.error('Error while loading Gist.', json.message);
                return;
            }
            if (json.files) {
                for (const key of Object.keys(json.files)) {
                    const file = json.files[key];
                    identifier = file.filename;
                    const extension = identifier.split('.').pop().toLowerCase();
                    if (extension == 'json' || extension == 'pbtxt' || extension == 'prototxt') {
                        const encoder = new TextEncoder();
                        buffer = encoder.encode(file.content);
                    }
                }
            }
            if (buffer == null || identifier == null) {
                this.error('Error while loading Gist.', 'Gist does not contain model file.');
                return;
            }
            const context = new BrowserContext(this, '', identifier, buffer);
            this._view.open(context).then(() => {
                this.document.title = identifier;
            }).catch((error) => {
                if (error) {
                    this.exception(error, false);
                    this.error(error.name, error.message);
                }
            });
        }).catch((err) => {
            this.error('Model load request failed.', err.message);
            this._view.show('welcome');
        });
    }

    _setCookie(name, value, days) {
        const date = new Date();
        date.setTime(date.getTime() + ((typeof days !== "number" ? 365 : days) * 24 * 60 * 60 * 1000));
        document.cookie = name + "=" + value + ";path=/;expires=" + date.toUTCString();
    }

    _getCookie(name) {
        const cookie = '; ' + document.cookie;
        const parts = cookie.split('; ' + name + '=');
        return parts.length < 2 ? undefined : parts.pop().split(';').shift();
    }

    _about() {
        const self = this;
        const eventHandler = () => {
            window.removeEventListener('keydown', eventHandler);
            self.document.body.removeEventListener('click', eventHandler);
            self._view.show('default');
        };
        window.addEventListener('keydown', eventHandler);
        this.document.body.addEventListener('click', eventHandler);
        this._view.show('about');
    }
};

if (typeof TextDecoder === "undefined") {
    TextDecoder = function TextDecoder(encoding) {
        this._encoding = encoding;
    };
    TextDecoder.prototype.decode = function decode(buffer) {
        let result = '';
        const length = buffer.length;
        let i = 0;
        switch (this._encoding) {
            case 'utf-8':
                while (i < length) {
                    const c = buffer[i++];
                    switch(c >> 4) {
                        case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7: {
                            result += String.fromCharCode(c);
                            break;
                        }
                        case 12: case 13: {
                            const c2 = buffer[i++];
                            result += String.fromCharCode(((c & 0x1F) << 6) | (c2 & 0x3F));
                            break;
                        }
                        case 14: {
                            const c2 = buffer[i++];
                            const c3 = buffer[i++];
                            result += String.fromCharCode(((c & 0x0F) << 12) | ((c2 & 0x3F) << 6) | ((c3 & 0x3F) << 0));
                            break;
                        }
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

if (typeof TextEncoder === 'undefined') {
    TextEncoder = function TextEncoder() {
    };
    TextEncoder.prototype.encode = function encode(str) {
        "use strict";
        const length = str.length;
        let resPos = -1;
        const resArr = typeof Uint8Array === "undefined" ? new Array(length * 2) : new Uint8Array(length * 3);
        for (let point = 0, nextcode = 0, i = 0; i !== length; ) {
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
                }
                else {
                    resArr[resPos += 1] = 0xef; resArr[resPos += 1] = 0xbf;
                    resArr[resPos += 1] = 0xbd; continue;
                }
            }
            if (point <= 0x007f) {
                resArr[resPos += 1] = (0x0<<7) | point;
            }
            else if (point <= 0x07ff) {
                resArr[resPos += 1] = (0x6<<5) | (point>>>6);
                resArr[resPos += 1] = (0x2<<6) | (point&0x3f);
            }
            else {
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
        return "[object TextEncoder]";
    };
    try {
        Object.defineProperty(TextEncoder.prototype,"encoding", {
            get:function() {
                if (Object.prototype.isPrototypeOf.call(TextEncoder.prototype, this)) {
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

if (typeof URLSearchParams === 'undefined') {
    URLSearchParams = function URLSearchParams(search) {
        const decode = (str) => {
            return str.replace(/[ +]/g, '%20').replace(/(%[a-f0-9]{2})+/ig, (match) => { return decodeURIComponent(match); });
        };
        this._dict = {};
        if (typeof search === 'string') {
            search = search.indexOf('?') === 0 ? search.substring(1) : search;
            const properties = search.split('&');
            for (const property of properties) {
                const index = property.indexOf('=');
                const name = (index > -1) ? decode(property.substring(0, index)) : decode(property);
                const value = (index > -1) ? decode(property.substring(index + 1)) : '';
                if (!Object.prototype.hasOwnProperty.call(this._dict, name)) {
                    this._dict[name] = [];
                }
                this._dict[name].push(value);
            }
        }
    };
    URLSearchParams.prototype.get = function(name) {
        return Object.prototype.hasOwnProperty.call(this._dict, name) ? this._dict[name][0] : null;
    };
}

if (!HTMLCanvasElement.prototype.toBlob) {
    HTMLCanvasElement.prototype.toBlob = function(callback, type, quality) {
        const canvas = this;
        setTimeout(function() {
            const data = atob(canvas.toDataURL(type, quality).split(',')[1]);
            const length = data.length;
            const buffer = new Uint8Array(length);
            for (let i = 0; i < length; i++) {
                buffer[i] = data.charCodeAt(i);
            }
            callback(new Blob([ buffer ], { type: type || 'image/png' }));
        });
    };
}

host.Dropdown = class {

    constructor(document, button, dropdown) {
        this._document = document;
        this._dropdown = document.getElementById(dropdown);
        this._button = document.getElementById(button);
        this._items = [];
        this._apple = /(Mac|iPhone|iPod|iPad)/i.test(navigator.platform);
        this._acceleratorMap = {};
        window.addEventListener('keydown', (e) => {
            let code = e.keyCode;
            code |= ((e.ctrlKey && !this._apple) || (e.metaKey && this._apple)) ? 0x0400 : 0;
            code |= e.altKey ? 0x0200 : 0;
            code |= e.shiftKey ? 0x0100 : 0;
            if (code == 0x001b) { // Escape
                this.close();
                return;
            }
            const item = this._acceleratorMap[code.toString()];
            if (item) {
                item.click();
                e.preventDefault();
            }
        });
        this._document.body.addEventListener('click', (e) => {
            if (!this._button.contains(e.target)) {
                this.close();
            }
        });
    }

    add(item) {
        const accelerator = item.accelerator;
        if (accelerator) {
            let cmdOrCtrl = false;
            let alt = false;
            let shift = false;
            let key = '';
            for (const part of item.accelerator.split('+')) {
                switch (part) {
                    case 'CmdOrCtrl': cmdOrCtrl = true; break;
                    case 'Alt': alt = true; break;
                    case 'Shift': shift = true; break;
                    default: key = part; break;
                }
            }
            if (key !== '') {
                item.accelerator = {};
                item.accelerator.text = '';
                if (this._apple) {
                    item.accelerator.text += alt ? '&#x2325;' : '';
                    item.accelerator.text += shift ? '&#x21e7;' : '';
                    item.accelerator.text += cmdOrCtrl ? '&#x2318;' : '';
                    const keyTable = { 'Enter': '&#x23ce;', 'Up': '&#x2191;', 'Down': '&#x2193;', 'Backspace': '&#x232B;' };
                    item.accelerator.text += keyTable[key] ? keyTable[key] : key;
                }
                else {
                    let list = [];
                    if (cmdOrCtrl) {
                        list.push('Ctrl');
                    }
                    if (alt) {
                        list.push('Alt');
                    }
                    if (shift) {
                        list.push('Shift');
                    }
                    list.push(key);
                    item.accelerator.text = list.join('+');
                }
                let code = 0;
                switch (key) {
                    case 'Backspace': code = 0x08; break;
                    case 'Enter': code = 0x0D; break;
                    case 'Up': code = 0x26; break;
                    case 'Down': code = 0x28; break;
                    default: code = key.charCodeAt(0); break;
                }
                code |= cmdOrCtrl ? 0x0400 : 0;
                code |= alt ? 0x0200 : 0;
                code |= shift ? 0x0100 : 0;
                this._acceleratorMap[code.toString()] = item;
            }
        }
        this._items.push(item);
    }

    toggle() {

        if (this._dropdown.style.display === 'block') {
            this.close();
            return;
        }

        while (this._dropdown.lastChild) {
            this._dropdown.removeChild(this._dropdown.lastChild);
        }

        for (const item of this._items) {
            if (Object.keys(item).length > 0) {
                let button = this._document.createElement('button');
                button.innerText = (typeof item.label == 'function') ? item.label() : item.label;
                button.addEventListener('click', () => {
                    this.close();
                    setTimeout(() => {
                        item.click();
                    }, 10);
                });
                this._dropdown.appendChild(button);
                if (item.accelerator) {
                    let accelerator = this._document.createElement('span');
                    accelerator.style.float = 'right';
                    accelerator.innerHTML = item.accelerator.text;
                    button.appendChild(accelerator);
                }
            }
            else {
                let separator = this._document.createElement('div');
                separator.setAttribute('class', 'separator');
                this._dropdown.appendChild(separator);
            }
        }

        this._dropdown.style.display = 'block';
    }

    close() {
        this._dropdown.style.display = 'none';
    }
};


class BrowserFileContext {

    constructor(file, blobs) {
        this._file = file;
        this._blobs = {};
        for (const blob of blobs) {
            this._blobs[blob.name] = blob;
        }
    }

    get identifier() {
        return this._file.name;
    }

    get buffer() {
        return this._buffer;
    }

    open() {
        return this.request(this._file.name, null).then((data) => {
            this._buffer = data;
        });
    }

    request(file, encoding) {
        const blob = this._blobs[file];
        if (!blob) {
            return Promise.reject(new Error("File not found '" + file + "'."));
        }
        return new Promise((resolve, reject) => {
            let reader = new FileReader();
            reader.onload = (e) => {
                resolve(encoding ? e.target.result : new Uint8Array(e.target.result));
            };
            reader.onerror = (e) => {
                e = e || window.event;
                let message = '';
                switch(e.target.error.code) {
                    case e.target.error.NOT_FOUND_ERR:
                        message = "File not found '" + file + "'.";
                        break;
                    case e.target.error.NOT_READABLE_ERR:
                        message = "File not readable '" + file + "'.";
                        break;
                    case e.target.error.SECURITY_ERR:
                        message = "File access denied '" + file + "'.";
                        break;
                    default:
                        message = "File read '" + e.target.error.code.toString() + "' error '" + file + "'.";
                        break;
                }
                reject(new Error(message));
            };
            if (encoding === 'utf-8') {
                reader.readAsText(blob, encoding);
            }
            else {
                reader.readAsArrayBuffer(blob);
            }
        });
    }
}

class BrowserContext {

    constructor(host, url, identifier, buffer) {
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
            const parts = url.split('?')[0].split('/');
            this._identifier = parts.pop();
            this._base = parts.join('/');
        }
    }

    request(file, encoding) {
        return this._host.request(this._base, file, encoding);
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }
}

window.__view__ = new view.View(new host.BrowserHost());
