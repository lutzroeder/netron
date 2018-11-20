/*jshint esversion: 6 */

class OpenVINOIRModelFactory {
    match(context) {
        return context.identifier.endsWith('.xml');
    }

    open(context, host, callback) {
        host.require('openvino-ir/openvino-ir', (err) => {
            if (err) {
                callback(err, null);
                return;
            }

            try {
                var xml_content = new TextDecoder("utf-8").decode(context.buffer);
            } catch (error) {
                callback(new OpenVINOIRError('File format is not OpenVINO IR compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINOIRParser.parse(xml_content);
            } catch (error) {
                callback(new OpenVINOIRError('Unable to parse OpenVINO IR file.'), null);
                return;
            }

            try {
                var model = new OpenVINOIRModel(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new OpenVINOIRError(error.message), null);
                return;
            }

            OpenVINOIROperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

class OpenVINOIRModel {
    constructor(netDef, init) {
        var graph = new OpenVINOIRGraph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }
}

class OpenVINOIRError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR model.';
    }
}
