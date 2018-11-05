/*jshint esversion: 6 */

const fs = require('fs');
const path = require('path');
const process = require('process');
const vm = require('vm');
const http = require('http');
const https = require('https');
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

    inflateRaw(data) {
        return require('zlib').inflateRawSync(data);
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

function loadFile(target, next) {
    var host = new TestHost();
    var folder = path.dirname(target);
    var identifier = path.basename(target);
    var buffer = fs.readFileSync(folder + '/' + identifier, null);
    var context = new TestContext(host, folder, identifier, buffer);
    var modelFactoryService = new view.ModelFactoryService(host);
    modelFactoryService.create(context, (err, model) => {
        if (err) {
            console.log('ERROR: ' + err.toString());
            return;
        }
        else if (!model.format) {
            console.log('ERROR: No model format.');
        }
        else {
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

            next();
        }
    });
}

function decompressSync(buffer, identifier) {
    var archive = null;
    extension = identifier.split('.').pop();
    if (extension == 'gz' || extension == 'tgz') {
        archive = new gzip.Archive(buffer, require('zlib').inflateRawSync);
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
            archive = new zip.Archive(buffer, require('zlib').inflateRawSync);
            break;
    }
    return archive;
}

function next() {
    if (models.length > 0) {
        var model = models.shift();
        // if (model.target != 'coreml/GestureAI.mlmodel') { next(); return; }
        // if (!model.target.startsWith('onnx/')) { next(); return; }
        var targets = model.target.split(',');
        var source = model.source;
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
            var data = [];
            var position = 0;
            var web = null;
            if (source.startsWith('http://')) {
                web = http;
            }
            else if (source.startsWith('https://')) {
                web = https;
            }
            web.get(source, (response) => {
                var length = response.headers['content-length'] ? Number(response.headers['content-length']) : -1;
                response.on("data", (chunk) => {
                    position += chunk.length;
                    if (length >= 0) {
                        process.stdout.write('  (' + ('  ' + Math.floor(100 * (position / length))).slice(-3) + '%) ' + source + '\r');
                    }
                    else {
                        process.stdout.write(position + '\r');
                    }
                    data.push(chunk);
                });
                response.on("end", () => {
                    var buffer = Buffer.concat(data);
                    if (files.length > 0) {
                        if (process.stdout.clearLine) {
                            process.stdout.clearLine();
                        }
                        process.stdout.write('  decompress...\r');
                        var archive = decompressSync(buffer, source.split('/').pop());
                        // console.log(archive);
                        files.forEach((file, index) => {
                            if (process.stdout.clearLine) {
                                process.stdout.clearLine();
                            }
                            process.stdout.write('  write ' + file + '\n');
                            var entry = archive.entries.filter((entry) => entry.name == file)[0];
                            fs.writeFileSync(testRootFolder + '/' + targets[index], entry.data, null);
                        });
                    }
                    else {
                        if (process.stdout.clearLine) {
                            process.stdout.clearLine();
                        }
                        process.stdout.write('  write ' + targets[0] + '\r');
                        fs.writeFileSync(testRootFolder + '/' + targets[0], buffer, null);
                    }
                    if (process.stdout.clearLine) {
                        process.stdout.clearLine();
                    }
                    loadFile(testRootFolder + '/' + targets[0], next);
                });
                response.on("error", (err) => {
                    console.log("ERROR: " + err.toString());
                });
            });
        }
        else {
            loadFile(testRootFolder + '/' + targets[0], next);
        }
    }
}

next();
