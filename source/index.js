
/* eslint-env es2015 */

if (window.location.hostname.endsWith('.github.io')) {
    window.location.replace('https://netron.app');
}

window.require = function(id, callback, preload) {
    var name = id.startsWith('./') ? id.substring(2) : id;
    var key = name === 'browser' ? 'host' : name;
    var value = window[key];
    if (callback) {
        if (value) {
            return callback(value);
        }
        window.module = { exports: {} };
        var url = new URL(id + '.js', window.location.href).href;
        var script = document.createElement('script');
        script.setAttribute('id', 'script-' + id);
        script.setAttribute('type', 'text/javascript');
        /* eslint-disable no-use-before-define */
        var loadHandler = function() {
            script.removeEventListener('load', loadHandler);
            script.removeEventListener('error', errorHandler);
            var module = window[key];
            if (!module) {
                if (preload) {
                    callback(null, new Error('The script \'' + id + '\' failed to load.'));
                    return;
                }
                module = window.module.exports;
                window[key] = module;
            }
            delete window.module;
            callback(module);
        };
        var errorHandler = function(e) {
            script.removeEventListener('load', loadHandler);
            script.removeEventListener('error', errorHandler);
            document.head.removeChild(script);
            delete window.module;
            callback(null, new Error('The script \'' + e.target.src + '\' failed to load.'));
        };
        /* eslint-enable no-use-before-define */
        script.addEventListener('load', loadHandler, false);
        script.addEventListener('error', errorHandler, false);
        script.setAttribute('src', url);
        document.head.appendChild(script);
        return null;
    }
    if (!value) {
        throw new Error("Module '" + id + "' not found.");
    }
    return value;
};

window.preload = function(callback) {
    var modules = [
        [ './view' ],
        [ './json', './xml', './protobuf', './hdf5', './grapher', './browser' ],
        [ './base', './text', './flatbuffers', './flexbuffers', './zip',  './tar', './python', './dagre' ]
    ];
    var next = function() {
        if (modules.length === 0) {
            callback();
            return;
        }
        var ids = modules.pop();
        var resolved = ids.length;
        for (var i = 0; i < ids.length; i++) {
            window.require(ids[i], function(module, error) {
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

window.terminate = function(message, action, callback) {
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
        try {
            window.__view__.show('welcome message');
        } catch (error) {
            // continue regardless of error
        }
    }
    document.body.setAttribute('class', 'welcome message');
};

window.addEventListener('error', (event) => {
    var error = event instanceof ErrorEvent && event.error && event.error instanceof Error ? event.error : new Error(event && event.message ? event.message : JSON.stringify(event));
    window.terminate(error.message);
});

window.addEventListener('load', function() {
    if (!Symbol || !Symbol.asyncIterator) {
        throw new Error('Your browser is not supported.');
    }
    window.preload(function(value, error) {
        if (error) {
            window.terminate(error.message);
        } else {
            var host = new window.host.BrowserHost();
            var view = require('./view');
            window.__view__ = new view.View(host);
            window.__view__.start();
        }
    });
});
