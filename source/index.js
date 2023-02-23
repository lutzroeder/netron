
var host = {};

host.BrowserHost = class {

    constructor() {
        this._window = window;
        this._navigator = window.navigator;
        this._document = window.document;
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
            'date': Array.isArray(this._meta.date) && this._meta.date.length > 0 && this._meta.date[0] ? new Date(this._meta.date[0].split(' ').join('T') + 'Z') : new Date(),
            'platform': /(Mac|iPhone|iPod|iPad)/i.test(this._navigator.platform) ? 'darwin' : undefined,
            'repository': this._document.getElementById('logo-github').getAttribute('href'),
            'menu': true
        };
        if (!/^\d\.\d\.\d$/.test(this.version)) {
            throw new Error('Invalid version.');
        }
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
            return require([ 'json', 'xml', 'protobuf', 'hdf5', 'grapher' ]).then(() => {
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

    view(view) {
        this._view = view;
        return this._age().then(() => this._consent()).then(() => this._telemetry()).then(() => this._capabilities());
    }

    _age() {
        const age = (new Date() - new Date(this._environment.date)) / (24 * 60 * 60 * 1000);
        if (age <= 180) {
            return Promise.resolve();
        }
        const callback = () => {
            const link = this.document.getElementById('logo-github').href;
            this.openURL(link);
        };
        this.document.body.classList.remove('spinner');
        this._message('Please update to the newest version.', 'Download', callback, true);
        return new Promise(() => {});
    }

    _consent() {
        if (this._getCookie('consent') || this._getCookie('_ga')) {
            return Promise.resolve();
        }
        const consent = () => {
            return new Promise((resolve) => {
                this.document.body.classList.remove('spinner');
                this._message('This app uses cookies to report errors and anonymous usage information.', 'Accept', () => {
                    this._setCookie('consent', Date.now().toString(), 30);
                    resolve();
                });
            });
        };
        return this._request('https://ipinfo.io/json', { 'Content-Type': 'application/json' }, 'utf-8', null, 2000).then((text) => {
            try {
                const json = JSON.parse(text);
                const countries = ['AT', 'BE', 'BG', 'HR', 'CZ', 'CY', 'DK', 'EE', 'FI', 'FR', 'DE', 'EL', 'HU', 'IE', 'IT', 'LV', 'LT', 'LU', 'MT', 'NL', 'NO', 'PL', 'PT', 'SK', 'ES', 'SE', 'GB', 'UK', 'GR', 'EU', 'RO'];
                if (json && json.country && countries.indexOf(json.country) === -1) {
                    this._setCookie('consent', Date.now().toString(), 30);
                    return Promise.resolve();
                }
                return consent();
            }
            catch (err) {
                return consent();
            }
        }).catch(() => {
            return consent();
        });
    }

    _telemetry() {
        if (this._environment.version && this._environment.version !== '0.0.0') {
            const ga4 = () => {
                const base = this.window.base;
                const measurement_id = '848W2NVWVH';
                const user = this._getCookie('_ga').replace(/^(GA1\.\d\.)*/, '');
                const session = this._getCookie('_ga' + measurement_id);
                this._telemetry_ga4 = new base.Telemetry(this._window, 'G-' + measurement_id, user, session);
                return this._telemetry_ga4.start().then(() => {
                    this._telemetry_ga4.set('page_location', this._document.location && this._document.location.href ? this._document.location.href : null);
                    this._telemetry_ga4.set('page_title', this._document.title ? this._document.title : null);
                    this._telemetry_ga4.set('page_referrer', this._document.referrer ? this._document.referrer : null);
                    this._telemetry_ga4.send('page_view', {
                        app_name: this.type,
                        app_version: this.version,
                    });
                    this._telemetry_ga4.send('scroll', {
                        percent_scrolled: 90,
                        app_name: this.type,
                        app_version: this.version
                    });
                    this._setCookie('_ga', 'GA1.2.' + this._telemetry_ga4.get('client_id'), 1200);
                    this._setCookie('_ga' + measurement_id, 'GS1.1.' + this._telemetry_ga4.session, 1200);
                });
            };
            const ua = () => {
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
            return ga4().then(() => ua());
        }
        return Promise.resolve();
    }

    _capabilities() {
        const list = [
            'TextDecoder', 'TextEncoder',
            'fetch', 'URLSearchParams',
            'HTMLCanvasElement.prototype.toBlob'
        ];
        const capabilities = list.filter((capability) => {
            const path = capability.split('.').reverse();
            let obj = this.window[path.pop()];
            while (obj && path.length > 0) {
                obj = obj[path.pop()];
            }
            return obj;
        });
        this.event('browser_open', {
            browser_capabilities: capabilities.map((capability) => capability.split('.').pop()).join(',')
        });
        if (capabilities.length < list.length) {
            this._message('Your browser is not supported.');
            return new Promise(() => {});
        }
        return Promise.resolve();
    }

    start() {
        this.window.addEventListener('error', (event) => {
            const error = event instanceof ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
            this.exception(error, true);
        });

        const hash = this.window.location.hash ? this.window.location.hash.replace(/^#/, '') : '';
        const search = this.window.location.search;
        const params = new URLSearchParams(search + (hash ? '&' + hash : ''));

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
                this.execute('open');
            });
            const extensions = new this.window.base.Metadata().extensions.map((extension) => '.' + extension);
            openFileDialog.setAttribute('accept', extensions.join(', '));
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

    error(message, detail, url) {
        alert((message == 'Error' ? '' : message + ' ') + detail);
        if (url) {
            this.openURL(url);
        }
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
        this.window.module = { exports: {} };
        const url = this._url(id + '.js');
        const script = document.createElement('script');
        script.setAttribute('id', id);
        script.setAttribute('type', 'text/javascript');
        script.setAttribute('src', url);
        return new Promise((resolve, reject) => {
            script.onload = () => {
                let module = this.window[name];
                if (!module) {
                    module = this.window.module.exports;
                    this.window[name] = module;
                }
                delete this.window.module;
                resolve(module);
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

    execute(name /*, value */) {
        switch (name) {
            case 'open': {
                const openFileDialog = this.document.getElementById('open-file-dialog');
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
                this.document.getElementById('version').innerText = this.version;
                const handler = () => {
                    this.window.removeEventListener('keydown', handler);
                    this.document.body.removeEventListener('click', handler);
                    this.document.body.classList.remove('about');
                };
                this.window.addEventListener('keydown', handler);
                this.document.body.addEventListener('click', handler);
                this.document.body.classList.add('about');
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
        if ((this._telemetry_ua || this._telemetry_ga4) && error) {
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
                }
                else {
                    const match = error.stack.match(/\n {4}at (.*):(\d*):(\d*)/);
                    if (match) {
                        stack = '(' + format(match[1], match[2], match[3]) + ')';
                    }
                    else {
                        const match = error.stack.match(/\n {4}at (.*)\((.*)\)/);
                        if (match) {
                            stack = '(' + format(match[1], match[2], match[3]) + ')';
                        }
                        else {
                            const match = error.stack.match(/\s*@\s*(.*):(.*):(.*)/);
                            if (match) {
                                stack = '(' + format(match[1], match[2], match[3]) + ')';
                            }
                            else {
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
            if (this._telemetry_ga4) {
                this._telemetry_ga4.send('exception', {
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
        if (this._telemetry_ga4 && name && params) {
            params.app_name = this.type,
            params.app_version = this.version,
            this._telemetry_ga4.send(name, params);
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
        url = url.startsWith('data:') ? url : url + ((/\?/).test(url) ? '&' : '?') + 'cb=' + (new Date()).getTime();
        this._view.show('welcome spinner');
        const progress = (value) => {
            this._view.progress(value);
        };
        return this._request(url, null, null, progress).then((stream) => {
            const context = new host.BrowserHost.Context(this, url, identifier, stream);
            if (this._telemetry_ga4) {
                this._telemetry_ga4.set('session_engaged', 1);
            }
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
            if (this._telemetry_ga4) {
                this._telemetry_ga4.set('session_engaged', 1);
            }
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
            const context = new host.BrowserHost.Context(this, '', identifier, stream);
            if (this._telemetry_ga4) {
                this._telemetry_ga4.set('session_engaged', 1);
            }
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

    _message(message, action, callback, modal) {
        const text = this.document.getElementById('message-text');
        if (text) {
            text.innerText = message;
        }
        const button = this.document.getElementById('message-button');
        if (button) {
            if (action && callback) {
                button.style.removeProperty('display');
                button.innerText = action;
                button.onclick = () => {
                    if (!modal) {
                        this._document.body.classList.remove('message');
                        button.onclick = null;
                    }
                    callback();
                };
            }
            else {
                button.style.display = 'none';
                button.onclick = null;
            }
        }
        this._document.body.classList.add('message');
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
