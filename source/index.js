window.exports = {};

window.exports.require = function(id, callback) {
    if (!/^[a-zA-Z0-9_-]+$/.test(id)) {
        throw new Error("Invalid module '" + id + "'.");
    }
    let base = window.location.href || '';
    base = base.split('?')[0].split('#')[0];
    const index = base.lastIndexOf('/');
    base = index > 0 ? base.substring(0, index + 1) : base;
    base = base.lastIndexOf('/') === base.length - 1 ? base : base + '/';
    var url = base + id + '.js';
    var scripts = document.head.getElementsByTagName('script');
    for (var i = 0; i < scripts.length; i++) {
        if (url === scripts[i].getAttribute('src')) {
            throw new Error("Duplicate import of '" + url + "'.");
        }
    }
    var script = document.createElement('script');
    script.setAttribute('id', id);
    script.setAttribute('type', 'module');
    /* eslint-disable no-use-before-define */
    var loadHandler = function() {
        script.removeEventListener('load', loadHandler);
        script.removeEventListener('error', errorHandler);
        callback();
    };
    var errorHandler = function(e) {
        script.removeEventListener('load', loadHandler);
        script.removeEventListener('error', errorHandler);
        callback(null, new Error("The script '" + e.target.src + "' failed to load."));
    };
    /* eslint-enable no-use-before-define */
    script.addEventListener('load', loadHandler, false);
    script.addEventListener('error', errorHandler, false);
    script.setAttribute('src', url);
    document.head.appendChild(script);
};

window.exports.preload = function(callback) {
    var modules = [
        ['view'],
        ['json', 'xml', 'protobuf', 'hdf5', 'grapher', 'browser'],
        ['base', 'text', 'flatbuffers', 'flexbuffers', 'zip',  'tar', 'python']
    ];
    var next = function() {
        if (modules.length === 0) {
            callback();
        } else {
            var ids = modules.pop();
            /* eslint-disable no-loop-func */
            var resolved = ids.length;
            for (var i = 0; i < ids.length; i++) {
                window.exports.require(ids[i], function(module, error) {
                    if (error) {
                        callback(null, error);
                    } else {
                        resolved--;
                        if (resolved === 0) {
                            next();
                        }
                    }
                });
            }
            /* eslint-enable no-loop-func */
        }
    };
    next();
};

window.exports.terminate = function(message) {
    document.getElementById('message-text').innerText = message;
    var button = document.getElementById('message-button');
    button.style.display = 'none';
    button.onclick = null;
    document.body.setAttribute('class', 'welcome message');
    if (window.__view__) {
        /* eslint-disable no-unused-vars */
        try {
            window.__view__.show('welcome message');
        } catch (error) {
            // continue regardless of error
        }
        /* eslint-enable no-unused-vars */
    }
};

window.addEventListener('error', function (event) {
    var error = event instanceof ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
    window.exports.terminate(error.message);
});

window.addEventListener('load', function() {
    if (typeof Symbol !== 'function' || typeof Symbol.asyncIterator !== 'symbol' ||
        typeof BigInt !== 'function' || typeof BigInt.asIntN !== 'function' || typeof BigInt.asUintN !== 'function' || typeof DataView.prototype.getBigInt64 !== 'function') {
        throw new Error('Please update your browser to use this application.');
    }
    var ua = window.navigator.userAgent;
    var chrome = ua.match(/Chrom(e|ium)\/([0-9]+)\./);
    var safari = ua.match(/Version\/(\d+)\.(\d+).*Safari/);
    var firefox = ua.match(/Firefox\/([0-9]+)\./);
    if ((Array.isArray(chrome) && parseInt(chrome[2], 10) < 80) ||
        (Array.isArray(safari) && (parseInt(safari[1], 10) < 16 || (parseInt(safari[1], 10) === 16 && parseInt(safari[2], 10) < 4))) ||
        (Array.isArray(firefox) && parseInt(firefox[1], 10) < 114)) {
        throw new Error('Please update your browser to use this application.');
    }
    window.exports.preload(function(value, error) {
        if (error) {
            window.exports.terminate(error.message);
        } else {
            var host = new window.exports.browser.Host();
            window.__view__ = new window.exports.view.View(host);
            window.__view__.start();
        }
    });
});
