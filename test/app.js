/*jshint esversion: 6 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const https = require('https');
const protobuf = require('protobufjs');
const view = require('../src/view.js');

global.TextDecoder = require('util').TextDecoder;

class TestHost {

    require(id, callback) {
        var filename = '../src/' + id + '.js';
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

makeDir = function(dir) {
    if (!fs.existsSync(dir)){
        makeDir(path.dirname(dir))
        fs.mkdirSync(dir);
    }
}

loadFile = function (target) {
    console.log(target);
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
        console.log('Format: ' + model.format);
    });
}

var models = JSON.parse(fs.readFileSync('models.json', 'utf-8'));

var testRootFolder = __dirname + '/data';
makeDir(testRootFolder);

models.forEach((model) => {
//    if (model.target != 'keras/tiny-yolo-voc.h5') { return; }
//    if (!model.target.startsWith('tflite/')) { return; }
    var target = testRootFolder + '/' + model.target;
    if (!fs.existsSync(target)) {
        console.log(model.source);
        var directory = path.dirname(target);
        makeDir(directory);
        var file = fs.createWriteStream(target);
        https.get(model.source, (response) => {
            response.on("end", function() {
                loadFile(target);
            });
            response.pipe(file);
        });
    }
    else {
        loadFile(target);
    }
});
