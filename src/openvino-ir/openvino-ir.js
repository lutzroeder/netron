/*jshint esversion: 6 */

var openvinoIR = openvinoIR || {};

if (window.require) {
    openvinoIR.Graph = openvinoIR.Graph || require('./openvino-ir-graph').Graph;
    openvinoIR.OperatorMetadata = openvinoIR.OperatorMetadata || require('./openvino-ir-metadata').OperatorMetadata;
}

openvinoIR.ModelFactory = class {
    match(context) {
        return context.identifier.endsWith('.xml');
    }

    open(context, host, callback) {
        host.require('./openvino-ir/openvino-ir-parser', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new openvinoIR.Error('File format is not OpenVINO IR compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINOIRParser.parse(xml_content);
            } catch (error) {
                callback(new openvinoIR.Error('Unable to parse OpenVINO IR file.'), null);
                return;
            }

            try {
                var model = new openvinoIR.Model(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new openvinoIR.Error(error.message), null);
                return;
            }

            openvinoIR.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

openvinoIR.Model = class {
    constructor(netDef, init) {
        var graph = new openvinoIR.Graph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }
}

openvinoIR.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR model.';
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvinoIR.ModelFactory;
}
