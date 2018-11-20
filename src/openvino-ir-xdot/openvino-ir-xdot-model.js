/*jshint esversion: 6 */

class OpenVINOIRXdotModelFactory {
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
                callback(new OpenVINOIRXdotError('File format is not OpenVINO IR Xdot compliant.'), null);
                return;
            }

            try {
                var parsed_xml = OpenVINOIRXdotParser.parse(xml_content);
            } catch (error) {
                callback(new OpenVINOIRXdotError('Unable to parse OpenVINO IR Xdot file.'), null);
                return;
            }

            try {
                var model = new OpenVINOIRXDotModel(parsed_xml);
            } catch (error) {
                host.exception(error, false);
                callback(new OpenVINOIRXdotError(error.message), null);
                return;
            }

            OpenVINOIRXdotOperatorMetadata.open(host, (err, metadata) => {
                callback(null, model);
            });
        });
    }
}

class OpenVINOIRXDotModel {
    constructor(netDef, init) {
        var graph = new OpenVINOIRXdotGraph(netDef, init);
        this._graphs = [graph];
    }

    get format() {
        return 'OpenVINO IR Xdot';
    }

    get graphs() {
        return this._graphs;
    }
}

class OpenVINOIRXdotError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO IR Xdot model.';
    }
}
