
/* eslint-env es2015 */
/* eslint-disable no-var */
/* eslint-disable prefer-arrow-callback */
/* eslint-disable prefer-template */
/* eslint-disable prefer-destructuring */

window.exports = {};

window.exports.require = function(id, callback) {
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
        ['base', 'text', 'flatbuffers', 'flexbuffers', 'zip',  'tar', 'python', 'dagre']
    ];
    var next = function() {
        if (modules.length === 0) {
            callback();
            return;
        }
        var ids = modules.pop();
        var resolved = ids.length;
        for (var i = 0; i < ids.length; i++) {
            window.exports.require(ids[i], function(module, error) {
                if (error) {
                    callback(null, error);
                    return;
                }
                resolved--;
                if (resolved === 0) {
                    next();
                }
            }, true);
        }
    };
    next();
};

window.exports.terminate = function(message, action, callback) {
    document.getElementById('message-text').innerText = message;
    var button = document.getElementById('message-button');
    if (action) {
        button.style.removeProperty('display');
        button.innerText = action;
        button.onclick = function() {
            callback();
        };
        button.focus();
    } else {
        button.style.display = 'none';
        button.onclick = null;
    }
    if (window.__view__) {
        /* eslint-disable no-unused-vars */
        try {
            window.__view__.show('welcome message');
        } catch (error) {
            // continue regardless of error
        }
        /* eslint-enable no-unused-vars */
    }
    document.body.setAttribute('class', 'welcome message');
};

window.addEventListener('error', function (event) {
    var error = event instanceof ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
    window.exports.terminate(error.message);
});

window.addEventListener('load', function() {
    if (typeof Symbol !== 'function' || typeof Symbol.asyncIterator !== 'symbol' ||
        typeof BigInt !== 'function' || typeof BigInt.asIntN !== 'function' || typeof BigInt.asUintN !== 'function' || typeof DataView.prototype.getBigInt64 !== 'function') {
        throw new Error('Your browser is not supported.');
    }
    window.exports.preload(function(value, error) {
        if (error) {
            window.exports.terminate(error.message);
        } else {
            var host = new window.exports.browser.BrowserHost();
            window.__view__ = new window.exports.view.View(host);
            window.__view__.start();
        }
    });
});
