/*jshint esversion: 6 */

const fs = require('fs');
const path = require('path');
const process = require('process');
const child_process = require('child_process');
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
global.protobuf = protobuf;

var type = process.argv.length > 2 ? process.argv[2] : null;

var models = JSON.parse(fs.readFileSync(__dirname + '/models.json', 'utf-8'));
var dataFolder = __dirname + '/data';

class TestHost {

    constructor() {
        this._exceptions = [];
    }

    require(id, callback) {
        try {
            var file = path.join(path.join(__dirname, '../src'), id + '.js');
            callback(null, require(file));
        }
        catch (err) {
            callback(err, null);
        }
    }

    request(base, file, encoding, callback) {
        var pathname = path.join(base || path.join(__dirname, '../src'), file);
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
        this._exceptions.push(err);
    }

    get exceptions() {
        return this._exceptions;
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
        if (item.producer && model.producer != item.producer) {
            callback(new Error("ERROR: Invalid producer '" + model.producer + "'."), null);
            return;
        }
        try {
            model.graphs.forEach((graph) => {
                graph.inputs.forEach((input) => {
                    input.connections.forEach((connection) => {
                        if (connection.type) {
                            connection.type.toString();
                        }
                    });
                });
                graph.outputs.forEach((output) => {
                    output.connections.forEach((connection) => {
                        if (connection.type) {
                            connection.type.toString();
                        }
                    });
                });
                graph.nodes.forEach((node) => {
                    node.attributes.forEach((attribute) => {
                        var value = view.View.formatAttributeValue(attribute.value, attribute.type)
                        if (value && value.length > 1000) {
                            value = value.substring(0, 1000) + '...';
                        }
                        value = value.split('<');
                    });
                    node.inputs.forEach((input) => {
                        input.connections.forEach((connection) => {
                            if (connection.type) {
                                connection.type.toString();
                            }
                            if (connection.initializer) {
                                var value = connection.initializer.toString();
                            }
                        });
                    });
                    node.outputs.forEach((output) => {
                        output.connections.forEach((connection) => {
                            if (connection.type) {
                                connection.type.toString();
                            }
                        });
                    });
                });
            });
        }
        catch (error) {
            callback(error, null);
            return;
        }
        if (host.exceptions.length > 0) {
            callback(host.exceptions[0], null);
            return;
        }
        callback(null, model);
    });
}

function decompress(buffer, identifier) {
    var archive = null;
    extension = identifier.split('.').pop().toLowerCase();
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

    switch (identifier.split('.').pop().toLowerCase()) {
        case 'tar':
            archive = new tar.Archive(buffer);
            break;
        case 'zip':
            archive = new zip.Archive(buffer);
            break;
    }
    return archive;
}

function request(location, cookie, callback) {
    var data = [];
    var position = 0;
    var protocol = url.parse(location).protocol;
    var httpModules = { 'http:': http, 'https:': https };
    var httpModule = httpModules[protocol];
    var httpRequest = httpModule.request(location, {
        rejectUnauthorized: false
    });
    if (cookie.length > 0) {
        httpRequest.setHeader('Cookie', cookie);
    }
    httpRequest.on('response', (response) => {
        if (response.statusCode == 200 && url.parse(location).hostname == 'drive.google.com' && 
            response.headers['set-cookie'].some((cookie) => cookie.startsWith('download_warning_'))) {
            cookie = response.headers['set-cookie'];
            var download = cookie.filter((cookie) => cookie.startsWith('download_warning_')).shift();
            var confirmToken = download.split(';').shift().split('=').pop();
            location = location + '&confirm=' + confirmToken;
            request(location, cookie, callback);
            return;
        }
        if (response.statusCode == 301 || response.statusCode == 302) {
            if (url.parse(response.headers.location).hostname) {
                location = response.headers.location;
            }
            else {
                location = url.parse(location).protocol + '//' + url.parse(location).hostname + response.headers.location;
            }
            request(location, cookie, callback);
            return;
        }
        if (response.statusCode != 200) {
            callback(new Error(response.statusCode.toString() + ' ' + location), null);
            return;
        }
        var length = response.headers['content-length'] ? Number(response.headers['content-length']) : -1;
        response.on("data", (chunk) => {
            position += chunk.length;
            if (length >= 0) {
                var label = location.length > 70 ? location.substring(0, 66) + '...' : location; 
                process.stdout.write('  (' + ('  ' + Math.floor(100 * (position / length))).slice(-3) + '%) ' + label + '\r');
            }
            else {
                process.stdout.write('  ' + position + ' bytes\r');
            }
            data.push(chunk);
        });
        response.on("end", () => {
            callback(null, Buffer.concat(data));
        });
        response.on("error", (err) => {
            callback(err, null);
        });
    });
    httpRequest.on('error', (err) => {
        callback(err, null);
    });
    httpRequest.end();
}

function download(folder, targets, sources, completed, callback) {
    if (targets.every((file) => fs.existsSync(folder + '/' + file))) {
        targets.forEach((target) => completed.push(target));
        callback(null, completed);
        return;
    }
    if (!sources) {
        callback(new Error('Download source not specified.'), null);
        return;
    }
    var source = '';
    var sourceFiles = [];
    var startIndex = sources.indexOf('[');
    var endIndex = sources.indexOf(']');
    if (startIndex != -1 && endIndex != -1 && endIndex > startIndex) {
        sourceFiles = sources.substring(startIndex + 1, endIndex).split(',').map((sourceFile) => sourceFile.trim());
        source = sources.substring(0, startIndex);
        sources = sources.substring(endIndex + 1);
        if (sources.startsWith(',')) {
            sources = sources.substring(1);
        }
    }
    else {
        var commaIndex = sources.indexOf(',');
        if (commaIndex != -1) {
            source = sources.substring(0, commaIndex);
            sources = sources.substring(commaIndex + 1);
        }
        else {
            source = sources;
            sources = '';
        }
    }
    targets.forEach((target) => {
        makeDir(path.dirname(folder + '/' + target));
    });
    request(source, [], (err, data) => {
        if (err) {
            callback(err, null);
            return;
        }
        if (sourceFiles.length > 0) {
            if (process.stdout.clearLine) {
                process.stdout.clearLine();
            }
            process.stdout.write('  decompress...\r');
            var archive = decompress(data, source.split('/').pop());
            // console.log(archive);
            sourceFiles.forEach((file, index) => {
                if (process.stdout.clearLine) {
                    process.stdout.clearLine();
                }
                process.stdout.write('  write ' + file + '\n');
                var entry = archive.entries.filter((entry) => entry.name == file)[0];
                if (!entry) {
                    callback(new Error("Entry not found '" + file + '. Archive contains entries: ' + JSON.stringify(archive.entries.map((entry) => entry.name)) + " ."), null);
                }
                var target = targets.shift();
                fs.writeFileSync(folder + '/' + target, entry.data, null);
                completed.push(target);
            });
        }
        else {
            var target = targets.shift();
            if (process.stdout.clearLine) {
                process.stdout.clearLine();
            }
            process.stdout.write('  write ' + target + '\r');
            fs.writeFileSync(folder + '/' + target, data, null);
            completed.push(target);
        }
        if (process.stdout.clearLine) {
            process.stdout.clearLine();
        }
        if (sources.length > 0) {
            download(folder, targets, sources, completed, callback);
            return;
        }
        callback(null, completed);
    });
}

function next() {
    if (models.length == 0) {
        return;
    }
    var item = models.shift();
    if (type && item.type != type) {
        next();
        return;
    }
    var targets = item.target.split(',');
    if (process.stdout.clearLine) {
        process.stdout.clearLine();
    }
    var folder = dataFolder + '/' + item.type;
    process.stdout.write(item.type + '/' + targets[0] + '\n');
    var sources = item.source;
    download(folder, targets, sources, [], (err, completed) => {
        if (err) {
            if (item.script) {
                try {
                    var root = path.dirname(__dirname);
                    var command = item.script[0].replace('${root}', root);
                    var args = item.script[1].replace('${root}', root);
                    console.log('  ' + command + ' ' + args);
                    child_process.execSync(command + ' ' + args, { stdio: [ 0, 1 , 2] });
                    completed = targets;
                }
                catch (err) {
                    console.log(err);
                    return;
                }
            }
            else {
                console.log(err);
                return;
            }
        }
        loadModel(folder + '/' + completed[0], item, (err, model) => {
            if (err) {
                if (!item.error && item.error != err.message) {
                    console.log(err);
                    return;
                }
            }
            next();
        });
    });
}

next();
