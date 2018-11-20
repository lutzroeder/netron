/*jshint esversion: 6 */

var openvinoDot = openvinoDot || {};

if (window.require) {
    openvinoDot.Graph = openvinoDot.Graph || require('./openvino-dot-graph').Graph;
    openvinoDot.OperatorMetadata = openvinoDot.OperatorMetadata || require('./openvino-dot-metadata').OperatorMetadata;
}

openvinoDot.ModelFactory = class {
    match(context) {
        return context.identifier.endsWith('.dot');
    }

    open(context, host, callback) {
        host.require('./openvino-dot/openvino-dot-parser', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new openvinoDot.Error('File format is not OpenVINO IR Dot compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINODotParser.parse(xml_content);
            } catch (error) {
                callback(new openvinoDot.Error('Unable to parse OpenVINO IR Dot file.'), null);
                return;
            }

            try {
                var model = new openvinoDot.Model(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new openvinoDot.Error(error.message), null);
                return;
            }

            openvinoDot.OperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

openvinoDot.Model = class {
    constructor(netDef, init) {
        var graph = new openvinoDot.Graph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR Dot';
    }

    get graphs() {
        return this._graphs;
    }
}

openvinoDot.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR Dot model.';
    }
}

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvinoDot.ModelFactory;
}
