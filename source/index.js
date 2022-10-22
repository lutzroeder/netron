
var host = {};

host.BrowserHost = class {

    constructor() {
        this._document = window.document;
        this._window = window;
        this._navigator = navigator;
        if (this._window.location.hostname.endsWith('.github.io')) {
            this._window.location.replace('https://netron.app');
        }
        this._window.eval = () => {
            throw new Error('window.eval() not supported.');
        };
        this._meta = {};
        for (const element of Array.from(this._document.getElementsByTagName('meta'))) {
            if (element.name !== undefined && element.content !== undefined) {
                this._meta[element.name] = this._meta[element.name] || [];
                this._meta[element.name].push(element.content);
            }
        }
        this._environment = {
            'type': this._meta.type ? this._meta.type[0] : 'Browser',
            'version': this._meta.version ? this._meta.version[0] : null,
            'date': this._meta.date ? new Date(this._meta.date[0].split(' ').join('T') + 'Z') : new Date(),
        };
        this._telemetry = this._environment.version && this._environment.version !== '0.0.0';
        this.window.require = (id) => {
            const name = id.startsWith('./') ? id.substring(2) : id;
            const value = this.window[name];
            if (value) {
                return value;
            }
            throw new Error("Module '" + id + "' not found.");
        };
        const require = (ids) => {
            return Promise.all(ids.map((id) => this.require(id)));
        };
        require([ 'base', 'text', 'flatbuffers', 'flexbuffers', 'zip',  'tar', 'python', 'dagre' ]).then(() => {
            return require([ 'json', 'xml', 'protobuf', 'hdf5', 'grapher', 'dialog' ]).then(() => {
                return require([ 'view' ]).then(() => {
                    this.window.__view__ = new this.window.view.View(this);
                });
            });
        }).catch((error) => {
            this._message(error.message);
        });
    }

    get window() {
        return this._window;
    }

    get document() {
        return this._document;
    }

    get version() {
        return this._environment.version;
    }

    get type() {
        return this._environment.type;
    }

    get agent() {
        const userAgent = this._navigator.userAgent.toLowerCase();
        if (userAgent.indexOf('safari') !== -1 && userAgent.indexOf('chrome') === -1) {
            return 'safari';
        }
        return 'any';
    }

    initialize(view) {
        this._view = view;
        return new Promise((resolve /*, reject */) => {
            const age = (new Date() - new Date(this._environment.date)) / ( 24 * 60 * 60 * 1000);
            if (age > 180) {
                this._message('Please update to the newest version.', 'Download', () => {
                    const link = this.document.getElementById('logo-github').href;
                    this.openURL(link);
                }, true);
            }
            else {
                const features = () => {
                    const features = [ 'TextDecoder', 'TextEncoder', 'fetch', 'URLSearchParams', 'HTMLCanvasElement.prototype.toBlob' ];
                    const supported = features.filter((feature) => {
                        const path = feature.split('.').reverse();
                        let item = this.window[path.pop()];
                        while (item && path.length > 0) {
                            item = item[path.pop()];
                        }
                        return !item;
                    });
                    if (supported.length > 0) {
                        for (const feature of features) {
                            this.event('Host', 'Browser', feature, 1);
                        }
                        this._message('Your browser is not supported.');
                    }
                    else {
                        resolve();
                    }
                };
                const telemetry = () => {
                    if (this._telemetry) {
                        const script = this.document.createElement('script');
                        script.setAttribute('type', 'text/javascript');
                        script.setAttribute('src', 'https://www.google-analytics.com/analytics.js');
                        script.onload = () => {
                            if (this.window.ga) {
                                this.window.ga.l = 1 * new Date();
                                this.window.ga('create', 'UA-54146-13', 'auto');
                                this.window.ga('set', 'anonymizeIp', true);
                            }
                            features();
                        };
                        script.onerror = () => {
                            features();
                        };
                        this.document.body.appendChild(script);
                    }
                    else {
                        features();
                    }
                };
                const consent = () => {
                    this._message('This app uses cookies to report errors and anonymous usage information.', 'Accept', () => {
                        this._setCookie('consent', 'yes', 30);
                        telemetry();
                    });
                };
                if (this._getCookie('consent')) {
                    telemetry();
                }
                else {
                    this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 'utf-8', null, 2000).then((text) => {
                        try {
                            const json = JSON.parse(text);
                            const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                            if (json && json.country && countries.indexOf(json.country) >= 0) {
                                consent();
                            }
                            else {
                                this._setCookie('consent', Date.now(), 30);
                                telemetry();
                            }
                        }
                        catch (err) {
                            consent();
                        }
                    }).catch(() => {
                        consent();
                    });
                }
            }
        });
    }

    start() {
        this.window.addEventListener('error', (e) => {
            this.exception(new Error(e ? e.message : JSON.stringify(e)), true);
        });

        const params = new URLSearchParams(this.window.location.search);

        const versionLabel = this.document.getElementById('version');
        if (versionLabel) {
            versionLabel.innerText = this.version;
        }

        this._menu = new host.Dropdown(this, 'menu-button', 'menu-dropdown');
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
            label: () => this._view.options.attributes ? 'Hide Attributes' : 'Show Attributes',
            accelerator: 'CmdOrCtrl+D',
            click: () => this._view.toggle('attributes')
        });
        this._menu.add({
            label: () => this._view.options.initializers ? 'Hide Initializers' : 'Show Initializers',
            accelerator: 'CmdOrCtrl+I',
            click: () => this._view.toggle('initializers')
        });
        this._menu.add({
            label: () => this._view.options.names ? 'Hide Names' : 'Show Names',
            accelerator: 'CmdOrCtrl+U',
            click: () => this._view.toggle('names')
        });
        this._menu.add({
            label: () => this._view.options.direction === 'vertical' ? 'Show Horizontal' : 'Show Vertical',
            accelerator: 'CmdOrCtrl+K',
            click: () => this._view.toggle('direction')
        });
        this._menu.add({
            label: () => this._view.options.mousewheel === 'scroll' ? 'Mouse Wheel: Zoom' : 'Mouse Wheel: Scroll',
            accelerator: 'CmdOrCtrl+M',
            click: () => this._view.toggle('mousewheel')
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

        if (this._meta.file) {
            const url = this._meta.file[0];
            if (this._view.accept(url)) {
                this._openModel(this._url(url), null);
                return;
            }
        }

        const url = params.get('url');
        if (url) {
            const identifier = params.get('identifier') || null;
            const location = url
                .replace(new RegExp('^https://github.com/([\\w]*/[\\w]*)/blob/([\\w/_.]*)(\\?raw=true)?$'), 'https://raw.githubusercontent.com/$1/$2')
                .replace(new RegExp('^https://huggingface.co/(.*)/blob/(.*)$'), 'https://huggingface.co/$1/resolve/$2');
            if (this._view.accept(identifier || location)) {
                this._openModel(location, identifier).then((identifier) => {
                    this.document.title = identifier;
                });
                return;
            }
        }

        const gist = params.get('gist');
        if (gist) {
            this._openGist(gist);
            return;
        }

        const openFileButton = this.document.getElementById('open-file-button');
        const openFileDialog = this.document.getElementById('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.addEventListener('click', () => {
                openFileDialog.value = '';
                openFileDialog.click();
            });
            const extensions = new this.window.base.Metadata().extensions.map((extension) => '.' + extension);
            openFileDialog.setAttribute('accept', extensions.join(', '));
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
        const githubButton = this.document.getElementById('github-button');
        const githubLink = this.document.getElementById('logo-github');
        if (githubButton && githubLink) {
            githubButton.style.opacity = 1;
            githubButton.addEventListener('click', () => {
                this.openURL(githubLink.href);
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

        this._view.show('welcome');
    }

    environment(name) {
        return this._environment[name];
    }

    error(message, detail) {
        alert((message == 'Error' ? '' : message + ' ') + detail);
    }

    confirm(message, detail) {
        return confirm(message + ' ' + detail);
    }

    require(id) {
        const name = id.startsWith('./') ? id.substring(2) : id;
        const value = this.window[name];
        if (value) {
            return Promise.resolve(value);
        }
        return new Promise((resolve, reject) => {
            this.window.module = { exports: {} };
            const url = this._url(id + '.js');
            const script = document.createElement('script');
            script.setAttribute('id', id);
            script.setAttribute('type', 'text/javascript');
            script.setAttribute('src', url);
            script.onload = () => {
                if (!this.window[name]) {
                    this.window[name] = this.window.module.exports;
                    delete this.window.module;
                }
                resolve(this.window[name]);
            };
            script.onerror = (e) => {
                delete this.window.module;
                reject(new Error('The script \'' + e.target.src + '\' failed to load.'));
            };
            this.document.head.appendChild(script);
        });
    }

    save(name, extension, defaultPath, callback) {
        callback(defaultPath + '.' + extension);
    }

    export(file, blob) {
        const element = this.document.createElement('a');
        element.download = file;
        element.href = URL.createObjectURL(blob);
        this.document.body.appendChild(element);
        element.click();
        this.document.body.removeChild(element);
    }

    request(file, encoding, base) {
        const url = base ? (base + '/' + file) : this._url(file);
        return this._request(url, null, encoding);
    }

    openURL(url) {
        this.window.location = url;
    }

    exception(error, fatal) {
        if (this._telemetry && this.window.ga && error && error.telemetry !== false) {
            const name = error.name ? error.name + ': ' : '';
            const message = error.message ? error.message : JSON.stringify(error);
            const description = [ name + message ];
            if (error.stack) {
                const format = (file, line, column) => {
                    return file.split('\\').join('/').split('/').pop() + ':' + line + ':' + column;
                };
                const match = error.stack.match(/\n {4}at (.*) \((.*):(\d*):(\d*)\)/);
                if (match) {
                    description.push(match[1] + ' (' + format(match[2], match[3], match[4]) + ')');
                }
                else {
                    const match = error.stack.match(/\n {4}at (.*):(\d*):(\d*)/);
                    if (match) {
                        description.push('(' + format(match[1], match[2], match[3]) + ')');
                    }
                    else {
                        const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                        if (match) {
                            description.push('(' + format(match[1], match[2], match[3]) + ')');
                        }
                        else {
                            const match = error.stack.match(/\s*@\s*(.*):(.*):(.*)/);
                            if (match) {
                                description.push('(' + format(match[1], match[2], match[3]) + ')');
                            }
                            else {
                                const match = error.stack.match(/.*\n\s*(.*)\s*/);
                                description.push(match ? match[1] : error.stack.split('\n').shift());
                            }
                        }
                    }
                }
            }
            this.window.ga('send', 'exception', {
                exDescription: description.join(' @ '),
                exFatal: fatal,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    screen(name) {
        if (this._telemetry && this.window.ga) {
            this.window.ga('send', 'screenview', {
                screenName: name,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    event(category, action, label, value) {
        if (this._telemetry && this.window.ga) {
            this.window.ga('send', 'event', {
                eventCategory: category,
                eventAction: action,
                eventLabel: label,
                eventValue: value,
                appName: this.type,
                appVersion: this.version
            });
        }
    }

    _request(url, headers, encoding, callback, timeout) {
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
            const progress = (value) => {
                if (callback) {
                    callback(value);
                }
            };
            request.onload = () => {
                progress(0);
                if (request.status == 200) {
                    if (request.responseType == 'arraybuffer') {
                        resolve(new host.BrowserHost.BinaryStream(new Uint8Array(request.response)));
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
                progress(0);
                const err = error(request.status);
                err.type = e.type;
                reject(err);
            };
            request.ontimeout = () => {
                progress(0);
                request.abort();
                const err = new Error("The web request timed out in '" + url + "'.");
                err.type = 'timeout';
                err.url = url;
                reject(err);
            };
            request.onprogress = (e) => {
                if (e && e.lengthComputable) {
                    progress(e.loaded / e.total * 100);
                }
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
        file = file.startsWith('./') ? file.substring(2) : file.startsWith('/') ? file.substring(1) : file;
        const location = this.window.location;
        const pathname = location.pathname.endsWith('/') ?
            location.pathname :
            location.pathname.split('/').slice(0, -1).join('/') + '/';
        return location.protocol + '//' + location.host + pathname + file;
    }

    _openModel(url, identifier) {
        url = url + ((/\?/).test(url) ? '&' : '?') + 'cb=' + (new Date()).getTime();
        this._view.show('welcome spinner');
        const progress = (value) => {
            this._view.progress(value);
        };
        return this._request(url, null, null, progress).then((stream) => {
            const context = new host.BrowserHost.BrowserContext(this, url, identifier, stream);
            return this._view.open(context).then(() => {
                return identifier || context.identifier;
            }).catch((err) => {
                if (err) {
                    this._view.error(err, null, 'welcome');
                }
            });
        }).catch((err) => {
            this.error('Model load request failed.', err.message);
            this._view.show('welcome');
        });
    }

    _open(file, files) {
        this._view.show('welcome spinner');
        const context = new host.BrowserHost.BrowserFileContext(this, file, files);
        context.open().then(() => {
            return this._view.open(context).then((model) => {
                this._view.show(null);
                this.document.title = files[0].name;
                return model;
            });
        }).catch((error) => {
            this._view.error(error, null, null);
        });
    }

    _openGist(gist) {
        this._view.show('welcome spinner');
        const url = 'https://api.github.com/gists/' + gist;
        this._request(url, { 'Content-Type': 'application/json' }, 'utf-8').then((text) => {
            const json = JSON.parse(text);
            if (json.message) {
                this.error('Error while loading Gist.', json.message);
                return;
            }
            const key = Object.keys(json.files).find((key) => this._view.accept(json.files[key].filename));
            if (!key) {
                this.error('Error while loading Gist.', 'Gist does not contain a model file.');
                return;
            }
            const file = json.files[key];
            const identifier = file.filename;
            const encoder = new TextEncoder();
            const buffer = encoder.encode(file.content);
            const stream = new host.BrowserHost.BinaryStream(buffer);
            const context = new host.BrowserHost.BrowserContext(this, '', identifier, stream);
            this._view.open(context).then(() => {
                this.document.title = identifier;
            }).catch((error) => {
                if (error) {
                    this._view.error(error, error.name, 'welcome');
                }
            });
        }).catch((err) => {
            this._view.error(err, 'Model load request failed.', 'welcome');
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

    _message(message, button, callback) {
        const messageText = this.document.getElementById('message');
        if (messageText) {
            messageText.innerText = message;
        }
        const messageButton = this.document.getElementById('message-button');
        if (messageButton) {
            if (button && callback) {
                messageButton.style.removeProperty('display');
                messageButton.innerText = button;
                messageButton.onclick = () => {
                    messageButton.onclick = null;
                    callback();
                };
            }
            else {
                messageButton.style.display = 'none';
                messageButton.onclick = null;
            }
        }
        const page = 'welcome message';
        if (this._view) {
            this._view.show(page);
        }
        else {
            this._document.body.setAttribute('class', page);
        }
    }

    _about() {
        const self = this;
        const eventHandler = () => {
            this.window.removeEventListener('keydown', eventHandler);
            self.document.body.removeEventListener('click', eventHandler);
            self._view.show('default');
        };
        this.window.addEventListener('keydown', eventHandler);
        this.document.body.addEventListener('click', eventHandler);
        this._view.show('about');
    }
};

host.Dropdown = class {

    constructor(host, button, dropdown) {
        this._host = host;
        this._dropdown = this._host.document.getElementById(dropdown);
        this._button = this._host.document.getElementById(button);
        this._items = [];
        this._apple = /(Mac|iPhone|iPod|iPad)/i.test(navigator.platform);
        this._acceleratorMap = {};
        this._host.window.addEventListener('keydown', (e) => {
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
        this._host.document.body.addEventListener('click', (e) => {
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
                    const list = [];
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
                const button = this._host.document.createElement('button');
                button.innerText = (typeof item.label == 'function') ? item.label() : item.label;
                button.addEventListener('click', () => {
                    this.close();
                    setTimeout(() => {
                        item.click();
                    }, 10);
                });
                this._dropdown.appendChild(button);
                if (item.accelerator) {
                    const accelerator = this._host.document.createElement('span');
                    accelerator.style.float = 'right';
                    accelerator.innerHTML = item.accelerator.text;
                    button.appendChild(accelerator);
                }
            }
            else {
                const separator = this._host.document.createElement('div');
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

host.BrowserHost.BinaryStream = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const buffer = this.read(length);
        return new host.BrowserHost.BinaryStream(buffer.slice(0));
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
    }

    peek(length) {
        if (this._position === 0 && length === undefined) {
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        const end = this._position;
        this.seek(position);
        return this._buffer.subarray(position, end);
    }

    read(length) {
        if (this._position === 0 && length === undefined) {
            this._position = this._length;
            return this._buffer;
        }
        const position = this._position;
        this.skip(length !== undefined ? length : this._length - this._position);
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._buffer[position];
    }
};

host.BrowserHost.BrowserFileContext = class {

    constructor(host, file, blobs) {
        this._host = host;
        this._file = file;
        this._blobs = {};
        for (const blob of blobs) {
            this._blobs[blob.name] = blob;
        }
    }

    get identifier() {
        return this._file.name;
    }

    get stream() {
        return this._stream;
    }

    request(file, encoding, base) {
        if (base !== undefined) {
            return this._host.request(file, encoding, base);
        }
        const blob = this._blobs[file];
        if (!blob) {
            return Promise.reject(new Error("File not found '" + file + "'."));
        }
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                resolve(encoding ? e.target.result : new host.BrowserHost.BinaryStream(new Uint8Array(e.target.result)));
            };
            reader.onerror = (e) => {
                e = e || this.window.event;
                let message = '';
                const error = e.target.error;
                switch(error.code) {
                    case error.NOT_FOUND_ERR:
                        message = "File not found '" + file + "'.";
                        break;
                    case error.NOT_READABLE_ERR:
                        message = "File not readable '" + file + "'.";
                        break;
                    case error.SECURITY_ERR:
                        message = "File access denied '" + file + "'.";
                        break;
                    default:
                        message = error.message ? error.message : "File read '" + error.code.toString() + "' error '" + file + "'.";
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

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }

    open() {
        return this.request(this._file.name, null).then((stream) => {
            this._stream = stream;
        });
    }
};

host.BrowserHost.BrowserContext = class {

    constructor(host, url, identifier, stream) {
        this._host = host;
        this._stream = stream;
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

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    request(file, encoding, base) {
        return this._host.request(file, encoding, base === undefined ? this._base : base);
    }

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }
};

if (!('scrollBehavior' in window.document.documentElement.style)) {
    const __scrollTo__ = Element.prototype.scrollTo;
    Element.prototype.scrollTo = function(options) {
        if (options === undefined) {
            return;
        }
        if (options === null || typeof options !== 'object' || options.behavior === undefined || arguments[0].behavior === 'auto' || options.behavior === 'instant') {
            if (__scrollTo__) {
                __scrollTo__.apply(this, arguments);
            }
            return;
        }
        const now = () => {
            return window.performance && window.performance.now ? window.performance.now() : Date.now();
        };
        const ease = (k) => {
            return 0.5 * (1 - Math.cos(Math.PI * k));
        };
        const step = (context) => {
            const value = ease(Math.min((now() - context.startTime) / 468, 1));
            const x = context.startX + (context.x - context.startX) * value;
            const y = context.startY + (context.y - context.startY) * value;
            context.element.scrollLeft = x;
            context.element.scrollTop = y;
            if (x !== context.x || y !== context.y) {
                window.requestAnimationFrame(step.bind(window, context));
            }
        };
        const context = {
            element: this,
            x: typeof options.left === 'undefined' ? this.scrollLeft : ~~options.left,
            y: typeof options.top === 'undefined' ? this.scrollTop : ~~options.top,
            startX: this.scrollLeft,
            startY: this.scrollTop,
            startTime: now()
        };
        step(context);
    };
}

window.addEventListener('load', () => {
    window.__host__ = new host.BrowserHost();
});
