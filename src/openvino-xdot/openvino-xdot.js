/*jshint esversion: 6 */

var openvinoXdot = openvinoXdot || {};

openvinoXdot.ModelFactory = class {
    match(context) {
        return context.identifier.endsWith('.dot');
    }

    open(context, host, callback) {
        host.require('openvino-ir-xdot/openvino-ir-xdot', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new openvinoXdot.Error('File format is not OpenVINO IR Xdot compliant.'), null);
                return;
            }

            try {
                var parsed_xml = openvinoXdot.Parser.parse(xml_content);
            } catch (error) {
                callback(new openvinoXdot.Error('Unable to parse OpenVINO IR Xdot file.'), null);
                return;
            }

            try {
                var model = new openvinoXdot.Model(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new openvinoXdot.Error(error.message), null);
                return;
            }

            openvinoXdot.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

openvinoXdot.Model = class {
    constructor(netDef, init) {
        var graph = new openvinoXdot.Graph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR Xdot';
    }

    get graphs() {
        return this._graphs;
    }
}

openvinoXdot.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR Xdot model.';
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvinoXdot.ModelFactory;
}
