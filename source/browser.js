
var host = {};

host.BrowserHost = class {

    constructor() {
        this._window = window;
        this._navigator = window.navigator;
        this._document = window.document;
        const base = require('./base');
        this._telemetry = new base.Telemetry(this._window);
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
            name: this._document.title,
            type: this._meta.type ? this._meta.type[0] : 'Browser',
            version: this._meta.version ? this._meta.version[0] : null,
            date: Array.isArray(this._meta.date) && this._meta.date.length > 0 && this._meta.date[0] ? new Date(this._meta.date[0].split(' ').join('T') + 'Z') : new Date(),
            packaged: this._meta.version && this._meta.version[0] !== '0.0.0',
            platform: /(Mac|iPhone|iPod|iPad)/i.test(this._navigator.platform) ? 'darwin' : undefined,
            agent: this._navigator.userAgent.toLowerCase().indexOf('safari') !== -1 && this._navigator.userAgent.toLowerCase().indexOf('chrome') === -1 ? 'safari' : '',
            repository: this._element('logo-github').getAttribute('href'),
            menu: true
        };
        if (!/^\d\.\d\.\d$/.test(this.version)) {
            throw new Error('Invalid version.');
        }
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

    async view(view) {
        this._view = view;
        const age = async () => {
            const days = (new Date() - new Date(this._environment.date)) / (24 * 60 * 60 * 1000);
            if (days > 180) {
                this.document.body.classList.remove('spinner');
                this.window.terminate('Please update to the newest version.', 'Download', () => {
                    const link = this._element('logo-github').href;
                    this.openURL(link);
                });
                return new Promise(() => {});
            }
            return Promise.resolve();
        };
        const consent = async () => {
            if (this._getCookie('consent') || this._getCookie('_ga')) {
                return;
            }
            let consent = true;
            try {
                const text = await this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 'utf-8', null, 2000);
                const json = JSON.parse(text);
                const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                if (json && json.country && countries.indexOf(json.country) === -1) {
                    consent = false;
                }
            } catch (error) {
                // continue regardless of error
            }
            if (consent) {
                this.document.body.classList.remove('spinner');
                await this._message('This app uses cookies to report errors and anonymous usage information.', 'Accept');
            }
            this._setCookie('consent', Date.now().toString(), 30);
        };
        const telemetry = async () => {
            if (this._environment.packaged) {
                this._window.addEventListener('error', (event) => {
                    const error = event instanceof ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
                    this.exception(error, true);
                });
                const ga4 = async () => {
                    const measurement_id = '848W2NVWVH';
                    const user = this._getCookie('_ga').replace(/^(GA1\.\d\.)*/, '');
                    const session = this._getCookie('_ga' + measurement_id);
                    await this._telemetry.start('G-' + measurement_id, user, session);
                    this._telemetry.set('page_location', this._document.location && this._document.location.href ? this._document.location.href : null);
                    this._telemetry.set('page_title', this._document.title ? this._document.title : null);
                    this._telemetry.set('page_referrer', this._document.referrer ? this._document.referrer : null);
                    this._telemetry.send('page_view', {
                        app_name: this.type,
                        app_version: this.version,
                    });
                    this._telemetry.send('scroll', {
                        percent_scrolled: 90,
                        app_name: this.type,
                        app_version: this.version
                    });
                    this._setCookie('_ga', 'GA1.2.' + this._telemetry.get('client_id'), 1200);
                    this._setCookie('_ga' + measurement_id, 'GS1.1.' + this._telemetry.session, 1200);
                };
                const ua = async () => {
                    return new Promise((resolve) => {
                        this._telemetry_ua = true;
                        const script = this.document.createElement('script');
                        script.setAttribute('type', 'text/javascript');
                        script.setAttribute('src', 'https://www.google-analytics.com/analytics.js');
                        script.onload = () => {
                            if (this.window.ga) {
                                this.window.ga.l = 1 * new Date();
                                this.window.ga('create', 'UA-54146-13', 'auto');
                                this.window.ga('set', 'anonymizeIp', true);
                            }
                            resolve();
                        };
                        script.onerror = () => {
                            resolve();
                        };
                        this.document.body.appendChild(script);
                    });
                };
                await ga4();
                await ua();
            }
        };
        const capabilities = async () => {
            const filter = (list) => {
                return list.filter((capability) => {
                    const path = capability.split('.').reverse();
                    let obj = this.window[path.pop()];
                    while (obj && path.length > 0) {
                        obj = obj[path.pop()];
                    }
                    return obj;
                });
            };
            const required = [
                'TextDecoder', 'TextEncoder',
                'URLSearchParams',
                'HTMLCanvasElement.prototype.toBlob',
                'Promise', 'Symbol.asyncIterator'
            ];
            const optional = [
                'fetch',
                'DataView.prototype.getBigInt64',
                'Worker',
            ];
            const available = filter(required);
            const capabilities = available.concat(filter(optional));
            this.event('browser_open', {
                browser_capabilities: capabilities.map((capability) => capability.split('.').pop()).join(',')
            });
            if (required.length > available.length) {
                this.window.terminate('Your browser is not supported.');
                return new Promise(() => {});
            }
            return Promise.resolve();
        };
        await age();
        await consent();
        await telemetry();
        await capabilities();
    }

    async start() {
        const hash = this.window.location.hash ? this.window.location.hash.replace(/^#/, '') : '';
        const search = this.window.location.search;
        const params = new URLSearchParams(search + (hash ? '&' + hash : ''));
        if (this._meta.file && this._meta.identifier) {
            const url = this._meta.file[0];
            if (this._view.accept(url)) {
                this._openModel(this._url(url), null);
                this._document.title = this._meta.identifier;
                return;
            }
        }
        const url = params.get('url');
        if (url) {
            const identifier = params.get('identifier') || null;
            const location = url
                .replace(/^https:\/\/github\.com\/([\w-]*\/[\w-]*)\/blob\/([\w/\-_.]*)(\?raw=true)?$/, 'https://raw.githubusercontent.com/$1/$2')
                .replace(/^https:\/\/github\.com\/([\w-]*\/[\w-]*)\/raw\/([\w/\-_.]*)$/, 'https://raw.githubusercontent.com/$1/$2')
                .replace(/^https:\/\/huggingface.co\/(.*)\/blob\/(.*)$/, 'https://huggingface.co/$1/resolve/$2');
            if (this._view.accept(identifier || location)) {
                const title = await this._openModel(location, identifier);
                if (title) {
                    this.document.title = title;
                    return;
                }
            }
        }
        const gist = params.get('gist');
        if (gist) {
            this._openGist(gist);
            return;
        }
        const openFileButton = this._element('open-file-button');
        const openFileDialog = this._element('open-file-dialog');
        if (openFileButton && openFileDialog) {
            openFileButton.addEventListener('click', () => {
                this.execute('open');
            });
            const mobileSafari = this.environment('platform') === 'darwin' && navigator.maxTouchPoints && navigator.maxTouchPoints > 1;
            if (!mobileSafari) {
                const base = require('./base');
                const extensions = new base.Metadata().extensions.map((extension) => '.' + extension);
                openFileDialog.setAttribute('accept', extensions.join(', '));
            }
            openFileDialog.addEventListener('change', (e) => {
                if (e.target && e.target.files && e.target.files.length > 0) {
                    const files = Array.from(e.target.files);
                    const file = files.find((file) => this._view.accept(file.name, file.size));
                    if (file) {
                        this._open(file, files);
                    }
                }
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
                const file = files.find((file) => this._view.accept(file.name, file.size));
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

    async error(message, detail) {
        alert((message == 'Error' ? '' : message + ' ') + detail);
        return 0;
    }

    confirm(message, detail) {
        return confirm(message + ' ' + detail);
    }

    require(id) {
        return new Promise((resolve, reject) => {
            this.window.require(id, (module) => resolve(module), (error) => reject(error));
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

    execute(name /*, value */) {
        switch (name) {
            case 'open': {
                const openFileDialog = this._element('open-file-dialog');
                if (openFileDialog) {
                    openFileDialog.value = '';
                    openFileDialog.click();
                }
                break;
            }
            case 'report-issue': {
                this.openURL(this.environment('repository') + '/issues/new');
                break;
            }
            case 'about': {
                this._view.about();
                break;
            }
            default: {
                break;
            }
        }
    }

    request(file, encoding, base) {
        const url = base ? (base + '/' + file) : this._url(file);
        return this._request(url, null, encoding);
    }

    openURL(url) {
        this.window.location = url;
    }

    exception(error, fatal) {
        if ((this._telemetry_ua || this._telemetry) && error) {
            const name = error.name ? error.name + ': ' : '';
            const message = error.message ? error.message : JSON.stringify(error);
            const description = name + message;
            let context = '';
            let stack = '';
            if (error.stack) {
                const format = (file, line, column) => {
                    return file.split('\\').join('/').split('/').pop() + ':' + line + ':' + column;
                };
                const match = error.stack.match(/\n {4}at (.*) \((.*):(\d*):(\d*)\)/);
                if (match) {
                    stack = match[1] + ' (' + format(match[2], match[3], match[4]) + ')';
                } else {
                    const match = error.stack.match(/\n {4}at (.*):(\d*):(\d*)/);
                    if (match) {
                        stack = '(' + format(match[1], match[2], match[3]) + ')';
                    } else {
                        const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                        if (match) {
                            stack = '(' + format(match[1], match[2], match[3]) + ')';
                        } else {
                            const match = error.stack.match(/\s*@\s*(.*):(.*):(.*)/);
                            if (match) {
                                stack = '(' + format(match[1], match[2], match[3]) + ')';
                            } else {
                                const match = error.stack.match(/.*\n\s*(.*)\s*/);
                                if (match) {
                                    stack = match[1];
                                }
                            }
                        }
                    }
                }
            }
            if (error.context) {
                context = typeof error.context === 'string' ? error.context : JSON.stringify(error.context);
            }
            if (this._telemetry_ua && this.window.ga) {
                this.window.ga('send', 'exception', {
                    exDescription: stack ? description + ' @ ' + stack : description,
                    exFatal: fatal,
                    appName: this.type,
                    appVersion: this.version
                });
            }
            this._telemetry.send('exception', {
                app_name: this.type,
                app_version: this.version,
                error_name: name,
                error_message: message,
                error_context: context,
                error_stack: stack,
                error_fatal: fatal ? true : false
            });
        }
    }

    event_ua(category, action, label, value) {
        if (this._telemetry_ua && this.window.ga && category && action && label) {
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

    event(name, params) {
        if (name && params) {
            params.app_name = this.type;
            params.app_version = this.version;
            this._telemetry.send(name, params);
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
                        const base = require('./base');
                        const buffer = new Uint8Array(request.response);
                        const stream = new base.BinaryStream(buffer);
                        resolve(stream);
                    } else {
                        resolve(request.responseText);
                    }
                } else {
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

    async _openModel(url, identifier) {
        url = url.startsWith('data:') ? url : url + ((/\?/).test(url) ? '&' : '?') + 'cb=' + (new Date()).getTime();
        this._view.show('welcome spinner');
        let context = null;
        try {
            const progress = (value) => {
                this._view.progress(value);
            };
            let stream = await this._request(url, null, null, progress);
            if (url.startsWith('https://raw.githubusercontent.com/') && stream.length < 150) {
                const buffer = stream.peek();
                const content = Array.from(buffer).map((c) => String.fromCodePoint(c)).join('');
                if (content.split('\n')[0] === 'version https://git-lfs.github.com/spec/v1') {
                    url = url.replace('https://raw.githubusercontent.com/', 'https://media.githubusercontent.com/media/');
                    stream = await this._request(url, null, null, progress);
                }
            }
            context = new host.BrowserHost.Context(this, url, identifier, stream);
            this._telemetry.set('session_engaged', 1);
        } catch (error) {
            await this.error('Model load request failed.', error.message);
            this._view.show('welcome');
            return null;
        }
        try {
            await this._view.open(context);
            return identifier || context.identifier;
        } catch (err) {
            if (err) {
                this._view.error(err, null, 'welcome');
            }
            return null;
        }
    }

    async _open(file, files) {
        this._view.show('welcome spinner');
        const context = new host.BrowserHost.BrowserFileContext(this, file, files);
        try {
            await context.open();
            this._telemetry.set('session_engaged', 1);
            await this._view.open(context);
            this._view.show(null);
            this.document.title = files[0].name;
        } catch (error) {
            this._view.error(error, null, null);
        }
    }

    async _openGist(gist) {
        this._view.show('welcome spinner');
        const url = 'https://api.github.com/gists/' + gist;
        try {
            const text = await this._request(url, { 'Content-Type': 'application/json' }, 'utf-8');
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
            const base = require('./base');
            const file = json.files[key];
            const identifier = file.filename;
            const encoder = new TextEncoder();
            const buffer = encoder.encode(file.content);
            const stream = new base.BinaryStream(buffer);
            const context = new host.BrowserHost.Context(this, '', identifier, stream);
            this._telemetry.set('session_engaged', 1);
            try {
                await this._view.open(context);
                this.document.title = identifier;
            } catch (error) {
                if (error) {
                    this._view.error(error, error.name, 'welcome');
                }
            }
        } catch (error) {
            this._view.error(error, 'Model load request failed.', 'welcome');
        }
    }

    _setCookie(name, value, days) {
        this.document.cookie = name + '=; Max-Age=0';
        const location = this.window.location;
        const domain = location && location.hostname && location.hostname.indexOf('.') !== -1 ? ';domain=.' + location.hostname.split('.').slice(-2).join('.') : '';
        const date = new Date();
        date.setTime(date.getTime() + (days * 24 * 60 * 60 * 1000));
        this.document.cookie = name + "=" + value + domain + ";path=/;expires=" + date.toUTCString();
    }

    _getCookie(name) {
        for (const cookie of this.document.cookie.split(';')) {
            const entry = cookie.split('=');
            if (entry[0].trim() === name) {
                return entry[1].trim();
            }
        }
        return '';
    }

    get(context, name) {
        if (context === 'configuration' && typeof localStorage !== 'undefined') {
            try {
                const content = localStorage.getItem(name);
                return JSON.parse(content);
            } catch (error) {
                // continue regardless of error
            }
        }
        return undefined;
    }

    set(context, name, value) {
        if (context === 'configuration' && typeof localStorage !== 'undefined') {
            try {
                localStorage.setItem(name, JSON.stringify(value));
            } catch (error) {
                // continue regardless of error
            }
        }
    }

    delete(context, name) {
        if (context === 'configuration' && typeof localStorage !== 'undefined') {
            try {
                localStorage.removeItem(name);
            } catch (error) {
                // continue regardless of error
            }
        }
    }

    _element(id) {
        return this.document.getElementById(id);
    }

    _message(message, action) {
        return new Promise((resolve) => {
            this._element('message-text').innerText = message;
            const button = this._element('message-button');
            if (action) {
                button.style.removeProperty('display');
                button.innerText = action;
                button.onclick = () => {
                    button.onclick = null;
                    this._document.body.classList.remove('message');
                    resolve(0);
                };
                button.focus();
            } else {
                button.style.display = 'none';
                button.onclick = null;
            }
            this._document.body.classList.add('message');
        });
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

    async request(file, encoding, basename) {
        if (basename !== undefined) {
            return this._host.request(file, encoding, basename);
        }
        const blob = this._blobs[file];
        if (!blob) {
            throw new Error("File not found '" + file + "'.");
        }
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            const size = 1024 * 1024 * 1024;
            let position = 0;
            const chunks = [];
            reader.onload = (e) => {
                if (encoding) {
                    resolve(e.target.result);
                } else {
                    const buffer = new Uint8Array(e.target.result);
                    if (position === 0 && buffer.length === blob.size) {
                        const base = require('./base');
                        const stream = new base.BinaryStream(buffer);
                        resolve(stream);
                    } else {
                        chunks.push(buffer);
                        position += buffer.length;
                        if (position < blob.size) {
                            const slice = blob.slice(position, Math.min(position + size, blob.size));
                            reader.readAsArrayBuffer(slice);
                        } else {
                            const stream = new host.BrowserHost.FileStream(chunks, size, 0, position);
                            resolve(stream);
                        }
                    }
                }
            };
            reader.onerror = (event) => {
                event = event || this._host.window.event;
                let message = '';
                const error = event.target.error;
                switch (error.code) {
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
            } else {
                const slice = blob.slice(position, Math.min(position + size, blob.size));
                reader.readAsArrayBuffer(slice);
            }
        });
    }

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }

    async open() {
        this._stream = await this.request(this._file.name, null);
    }
};

host.BrowserHost.FileStream = class {

    constructor(chunks, size, start, length) {
        this._chunks = chunks;
        this._size = size;
        this._start = start;
        this._length = length;
        this._position = 0;
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    stream(length) {
        const file = new host.BrowserHost.FileStream(this._chunks, this._size, this._position, length);
        this.skip(length);
        return file;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    peek(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            this._position -= length;
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        this.seek(position);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    read(length) {
        length = length !== undefined ? length : this._length - this._position;
        if (length < 0x10000000) {
            const position = this._fill(length);
            return this._buffer.subarray(position, position + length);
        }
        const position = this._position;
        this.skip(length);
        const buffer = new Uint8Array(length);
        this._read(buffer, position);
        return buffer;
    }

    byte() {
        const position = this._fill(1);
        return this._buffer[position];
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error('Expected ' + (this._position + length - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._buffer = new Uint8Array(Math.min(0x10000000, this._size, this._length - this._offset));
            this._read(this._buffer, this._offset);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }

    _read(buffer, offset) {
        const index = Math.floor(offset / this._size);
        offset = offset - (index * this._size);
        const length = Math.min(this._chunks[index].length - offset, buffer.length);
        buffer.set(this._chunks[index].subarray(offset, offset + length), 0);
        if (length !== buffer.length) {
            buffer.set(this._chunks[index + 1].subarray(0, buffer.length - length), length);
        }
    }
};

host.BrowserHost.Context = class {

    constructor(host, url, identifier, stream) {
        this._host = host;
        this._stream = stream;
        if (identifier) {
            this._identifier = identifier;
            this._base = url;
            if (this._base.endsWith('/')) {
                this._base.substring(0, this._base.length - 1);
            }
        } else {
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
        if (options !== undefined) {
            if (options === null || typeof options !== 'object' || options.behavior === undefined || arguments[0].behavior === 'auto' || options.behavior === 'instant') {
                if (__scrollTo__) {
                    __scrollTo__.apply(this, arguments);
                }
            } else {
                const now = () =>  window.performance && window.performance.now ? window.performance.now() : Date.now();
                const ease = (k) => 0.5 * (1 - Math.cos(Math.PI * k));
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
            }
        }
    };
}
