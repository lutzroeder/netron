/*jshint esversion: 6 */

const fs = require('fs');
const path = require('path');
const process = require('process');
const vm = require('vm');
const http = require('http');
const https = require('https');
const url = require('url');
const protobuf = require('protobufjs');
const view = require('../src/view.js');
const zip = require('../src/zip');
const gzip = require('../src/gzip');
const tar = require('../src/tar');

global.TextDecoder = require('util').TextDecoder;

var models = JSON.parse(fs.readFileSync(__dirname + '/models.json', 'utf-8'));
var testRootFolder = __dirname + '/data';

class TestHost {

    require(id, callback) {
        var filename = path.join(path.join(__dirname, '../src'), id + '.js');
        var data = fs.readFileSync(filename, 'utf-8');
        eval(data);
        callback(null);
    }

    request(base, file, encoding, callback) {
        var pathname = path.join(path.join(__dirname, '../src'), file);
        fs.exists(pathname, (exists) => {
            if (!exists) {
                callback(new Error('File not found.'), null);
            }
            else {
                fs.readFile(pathname, encoding, (err, data) => {
                    if (err) {
                        callback(err, null);
                    }
                    else {
                        callback(null, data);
                    }
                });
            }
        });
    }

    exception(err, fatal) {
        console.log("ERROR: " + err.toString());
    }
}

class TestContext {

    constructor(host, folder, identifier, buffer) {
        this._host = host;
        this._folder = folder;
        this._identifier = identifier;
        this._buffer = buffer;
    }

    request(file, encoding, callback) {
        this._host.request(this._folder, file, encoding, (err, buffer) => {
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

    get tags() {
        if (!this._tags) {
            this._tags = {};
            try {
                var reader = protobuf.TextReader.create(this.text);
                reader.start(false);
                while (!reader.end(false)) {
                    var tag = reader.tag();
                    this._tags[tag] = true;
                    reader.skip();
                }
            }
            catch (error) {
            }
        }
        return this._tags;
    }
}

function makeDir(dir) {
    if (!fs.existsSync(dir)){
        makeDir(path.dirname(dir));
        fs.mkdirSync(dir);
    }
}

function loadModel(target, item, callback) {
    var host = new TestHost();
    var folder = path.dirname(target);
    var identifier = path.basename(target);
    var size = fs.statSync(target).size;
    var buffer = new Uint8Array(size);
    var fd = fs.openSync(target, 'r');
    fs.readSync(fd, buffer, 0, size, 0);
    fs.closeSync(fd);
    var context = new TestContext(host, folder, identifier, buffer);
    var modelFactoryService = new view.ModelFactoryService(host);
    modelFactoryService.create(context, (err, model) => {
        if (err) {
            callback(err, null);
            return;
        }
        if (!model.format || (item.format && model.format != item.format)) {
            callback(new Error("ERROR: Invalid model format '" + model.format + "'."), null);
            return;
        }
        else {
            try {
                model.graphs.forEach((graph) => {
                    graph.inputs.forEach((input) => {
                    });
                    graph.outputs.forEach((output) => {
                    });
                    graph.nodes.forEach((node) => {
                        node.attributes.forEach((attribute) => {
                        });
                        node.inputs.forEach((input) => {
                            input.connections.forEach((connection) => {
                                if (connection.initializer) {
                                    var value = connection.initializer.toString();
                                }
                            });
                        });
                        node.outputs.forEach((output) => {
                            output.connections.forEach((connection) => {
                            });
                        });
                    });
                });
            }
            catch (error) {
                callback(error, null);
                return;
            }
            callback(null, model);
        }
    });
}

function decompress(buffer, identifier) {
    var archive = null;
    extension = identifier.split('.').pop();
    if (extension == 'gz' || extension == 'tgz') {
        archive = new gzip.Archive(buffer);
        if (archive.entries.length == 1) {
            entry = archive.entries[0];
            if (entry.name) {
                identifier = entry.name;
            }
            else {
                identifier = identifier.substring(0, identifier.lastIndexOf('.'));
                if (extension == 'tgz') {
                    identifier += '.tar';
                }
            }
            buffer = entry.data;
            archive = null;
        }
    }

    switch (identifier.split('.').pop()) {
        case 'tar':
            archive = new tar.Archive(buffer);
            break;
        case 'zip':
            archive = new zip.Archive(buffer);
            break;
    }
    return archive;
}

function request(location, callback) {
    var data = [];
    var position = 0;
    var protocol = url.parse(location).protocol;
    var httpModules = { 'http:': http, 'https:': https };
    var httpModule = httpModules[protocol];

    var httpRequest = httpModule.request(location);
    httpRequest.on('response', (response) => {
        if (response.statusCode == 301 || response.statusCode == 302) {
            if (url.parse(response.headers.location).hostname) {
                location = response.headers.location;
            }
            else {
                location = url.parse(location).protocol + '//' + url.parse(location).hostname + response.headers.location;
            }
            request(location, callback);
        }
        else {
            var length = response.headers['content-length'] ? Number(response.headers['content-length']) : -1;
            response.on("data", (chunk) => {
                position += chunk.length;
                if (length >= 0) {
                    var label = location.length > 70 ? location.substring(0, 67) + '...' : location; 
                    process.stdout.write('  (' + ('  ' + Math.floor(100 * (position / length))).slice(-3) + '%) ' + label + '\r');
                }
                else {
                    process.stdout.write(position + '\r');
                }
                data.push(chunk);
            });
            response.on("end", () => {
                callback(null, Buffer.concat(data));
            });
            response.on("error", (err) => {
                callback(err, null);
            });
        }
    });
    httpRequest.on('error', (err) => {
        callback(err, null);
    });
    httpRequest.end();
}

function next() {
    if (models.length > 0) {
        var item = models.shift();
        if (item.status && item.status == 'fail') {
            next();
            return;
        }
        // if (item.target != 'coreml/GestureAI.mlmodel') { next(); return; }
        // if (!item.target.startsWith('onnx/')) { next(); return; }
        var targets = item.target.split(',');
        var source = item.source;
        var files = [];
        var index = source.indexOf('[');
        if (index != -1) {
            var contents = source.substring(index);
            if (contents.startsWith('[') && contents.endsWith(']')) {
                files = contents.substring(1, contents.length - 1).split(',').map((file) => file.trim());
            }
            source = source.substring(0, index);
        }
        if (process.stdout.clearLine) {
            process.stdout.clearLine();
        }
        process.stdout.write(targets[0] + '\n');
        if (!targets.every((target) => fs.existsSync(testRootFolder + '/' + target))) {
            targets.forEach((target) => makeDir(path.dirname(testRootFolder + '/' + target)));
            request(source, (err, data) => {
                if (err) {
                    console.log("ERROR: " + err.toString());
                    return;
                }

                if (files.length > 0) {
                    if (process.stdout.clearLine) {
                        process.stdout.clearLine();
                    }
                    process.stdout.write('  decompress...\r');
                    var archive = decompress(data, source.split('/').pop());
                    // console.log(archive);
                    files.forEach((file, index) => {
                        if (process.stdout.clearLine) {
                            process.stdout.clearLine();
                        }
                        process.stdout.write('  write ' + file + '\n');
                        var entry = archive.entries.filter((entry) => entry.name == file)[0];
                        if (!entry) {
                            console.log("ERROR: Entry not found '" + file + '. Archive contains entries: ' + JSON.stringify(archive.entries.map((entry) => entry.name)) + " .");
                        }
                        fs.writeFileSync(testRootFolder + '/' + targets[index], entry.data, null);
                    });
                }
                else {
                    if (process.stdout.clearLine) {
                        process.stdout.clearLine();
                    }
                    process.stdout.write('  write ' + targets[0] + '\r');
                    fs.writeFileSync(testRootFolder + '/' + targets[0], data, null);
                }
                if (process.stdout.clearLine) {
                    process.stdout.clearLine();
                }
                loadModel(testRootFolder + '/' + targets[0], item, (err, model) => {
                    if (err) {
                        console.log(err);
                        return;
                    }
                    next();
                });
            });
        }
        else {
            loadModel(testRootFolder + '/' + targets[0], item, (err, model) => {
                if (err) {
                    console.log(err);
                    return;
                }
                next();
            });
        }
    }
}

next();
