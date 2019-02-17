/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var openvino = openvino || {};
var marked = marked || require('marked');

openvino.Utils = class {
    static countDirectChildrenByName(element, childName) {
        return openvino.Utils.findDirectChildrenByName(element, childName).length;
    }

    static findDirectChildrenByName(element, childName) {
        let whereToSearch;
        try {
            whereToSearch = Array.from(element.childNodes.values());
        } catch (e) {
            // tests use NodeJS implementation of DOM parser - it does not have the values() function
            whereToSearch = Object.values(element.childNodes);
        }
        return whereToSearch.reduce((acc, el) => {
            if (el.nodeName === childName) {
                acc.push(el);
            }
            return acc;
        }, []);
    }
}

openvino.ModelFactory = class {

    match(context) {
        var extension = context.identifier.split('.').pop().toLowerCase();
        if (extension === 'xml') {
            if (context.text.includes('<net')) {
                return true;
            }
        }
        return false;
    }

    open(context, host, callback) {
        openvino.Metadata.open(host, (err, metadata) => {
            try {
                var errors = false;
                var parser = new DOMParser({ errorHandler: () => { errors = true; } });
                var xml = parser.parseFromString(context.text, 'text/xml');
                if (errors || xml.documentElement == null || xml.getElementsByTagName('parsererror').length > 0) {
                    callback(new openvino.Error('File format is not OpenVINO XML.'), null);
                    return;
                }
                var net$ = xml.documentElement;
                if (!this.isIRCompliant(net$)) {
                    callback(new openvino.Error('File format is not OpenVINO IR.'), null);
                    return;
                }
                var model = new openvino.Model(metadata, net$);
                model.validate();
                callback(null, model);
                return;
            } catch (error) {
                host.exception(error, false);
                callback(new openvino.Error(error.message), null);
                return;
            }
        });
    }

    isIRCompliant(netElement) {
        if (!netElement || netElement.nodeName != 'net') {
            return false;
        }

        const countTopLevelLayers = openvino.Utils.countDirectChildrenByName(netElement, 'layers');
        const countTopLevelEdges = openvino.Utils.countDirectChildrenByName(netElement, 'edges');

        return countTopLevelLayers === 1 && countTopLevelEdges === 1;
    }
};

openvino.Model = class {

    constructor(metadata, net$) {
        var graph = new openvino.Graph(metadata, net$);
        this._graphs = [ graph ];
    }

    get format() {
        return 'OpenVINO IR';
    }

    get graphs() {
        return this._graphs;
    }

    validate() {
        this._graphs.forEach((graph) => {
            const pseudoInputsCount = graph.getPseudoInputs().length;
            // if (pseudoInputsCount !== 0){
            //     throw Error('Graph contains more than one connected component. Unable to show.');
            // }
        });
    }
};


openvino.Graph = class {

    constructor(metadata, net$) {
        this._metadata = metadata;
        this._name = net$.getAttribute('name') || '';
        this._batch = net$.getAttribute('batch') || '';
        this._version = net$.getAttribute('version') || '';

        this._nodes = [];
        this._operators = {};
        this._inputs = [];
        this._outputs = [];

        this._connections = {};

        var layersElement$ = openvino.Utils.findDirectChildrenByName(net$, 'layers')[0];
        var edgesElement$ = openvino.Utils.findDirectChildrenByName(net$, 'edges')[0];

        var layers = openvino.Utils.findDirectChildrenByName(layersElement$, 'layer');
        var edges = openvino.Utils.findDirectChildrenByName(edgesElement$, 'edge');

        var edgeMap = this.collectEdges(edges);

        layers.forEach((layer$) => {
            var operator = layer$.getAttribute('type');
            this.registerOperator(operator);

            switch (operator) {
                case 'Input':
                    var connections = [];
                    var precision = layer$.getAttribute('precision');
                    var name = layer$.getAttribute('name') || '';
                    var id = layer$.getAttribute('id');
                    var outputElements = openvino.Utils.findDirectChildrenByName(layer$, 'output');
                    outputElements.forEach((outputElement$) => {
                        var portElements = openvino.Utils.findDirectChildrenByName(outputElement$, 'port');
                        portElements.forEach((portElement$) => {
                            connections.push(this._connection(id, precision, portElement$, null));
                        });
                    });
                    this._inputs.push(new openvino.Argument(name, connections));
                    break;
                default:
                    this._nodes.push(new openvino.Node(this, this._metadata, layer$, edgeMap));
                    break;
            }
        });

        this.replaceTensorIteratorWithSubgraph(layers, edges);

        delete this._connections;
    }

    get name() {
        return this._name;
    }

    get inputs() {
        return this._inputs;
    }

    get outputs() {
        return this._outputs;
    }

    get nodes() {
        return this._nodes;
    }

    get operators() {
        return this._operators;
    }

    _connection(layer, precision, port, map) {
        var id = layer + ':' + port.getAttribute('id');
        if (map) {
            id = map[id];
        }
        var connection = this._connections[id];
        if (!connection) {
            var dimensions = [];
            Array.prototype.slice.call(port.getElementsByTagName('dim')).forEach((dimElement) => {
                dimensions.push(parseInt(dimElement.textContent.trim()));
            });
            var shape = (dimensions.length == 0) ? null : new openvino.TensorShape(dimensions);
            connection = new openvino.Connection(id, new openvino.TensorType(precision, shape), null);
        }
        return connection;
    }

    registerOperator(operator) {
        this._operators[operator] = (this._operators[operator] || 0) + 1;
    }

    addNewNode(node) {
        this._nodes.push(node);
    }

    getPseudoInputs() {
        // all graph elements are split between inputs and nodes
        // by definition IR is a graph can have inputs of two types: "Input" and "Const"
        // "Input" layers are already moved to inputs when we parse a graph
        // if there are any layers that do not have input connections and they are no Const ones
        // this means that this graph was not properly processed by the graph building logic
        const allNodesOutputs = this._nodes.reduce((acc, node) => {
           const nodesRes = this.collectConnectionsIds(node._outputs);
           acc = acc.concat(nodesRes);
           return acc;
        }, []);
        const allInputsOutputs = this.collectConnectionsIds(this._inputs);

        const outputSet = new Set([...allNodesOutputs, ...allInputsOutputs]);
        const nodesWithNonExistentInputs = this._nodes.reduce((acc, node) => {
            const nodesInputs = this.collectConnectionsIds(node._inputs);
            const diff = nodesInputs.filter((value) => !outputSet.has(value));
            if (diff.length > 0) {
                acc.push(node);
            }
            return acc;
        }, []);
        return nodesWithNonExistentInputs;
    }

    replaceTensorIteratorWithSubgraph(layers, edges) {
        const tiNodes = layers.filter((node$) => node$.getAttribute('type') === 'TensorIterator');

        tiNodes.forEach((singleTensorIteratorNode$) => {
            const singleTensorIteratorNodeId = singleTensorIteratorNode$.getAttribute("id");
            const body$ = openvino.Utils.findDirectChildrenByName(singleTensorIteratorNode$, 'body')[0];
            const layersContainer$ = openvino.Utils.findDirectChildrenByName(body$, 'layers')[0];
            const edgesContainer$ = openvino.Utils.findDirectChildrenByName(body$, 'edges')[0];
            const iteratorLayers = openvino.Utils.findDirectChildrenByName(layersContainer$, 'layer');
            const iteratorEdges = openvino.Utils.findDirectChildrenByName(edgesContainer$, 'edge');
            const iteratorEdgeMap = this.collectEdges(iteratorEdges);
            const iteratorBackEdgesContainer$ = openvino.Utils.findDirectChildrenByName(singleTensorIteratorNode$, 'back_edges')[0];
            const iteratorBackEdges = openvino.Utils.findDirectChildrenByName(iteratorBackEdgesContainer$, 'edge')
            const iteratorBackEdgesMap = this.collectEdges(iteratorBackEdges);
            const iteratorAllEdges = Object.assign({}, iteratorEdgeMap, iteratorBackEdgesMap);

            const mappingForNestedIR = this.parseMappingBlock(singleTensorIteratorNode$);

            iteratorLayers.forEach((nestedLayer$) => {
                const nestedNode = new openvino.Node(this, this._metadata, nestedLayer$, iteratorAllEdges);
                nestedNode._id = `${singleTensorIteratorNodeId}_${nestedLayer$.getAttribute('id')}`;
                nestedNode._inputs.forEach((input) => {
                    input.connections.forEach((connection) => {
                        // we had a connection with id: 0:1  - meaning from layer "0" and its port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. internal layer with id "0" and TI with id "14" results in internal layer to get id "14_0"
                        if (!connection._id){
                            return;
                        }
                        connection._id = `${singleTensorIteratorNodeId}_${connection._id}`;
                    });
                });

                nestedNode._outputs.forEach((output) => {
                    output.connections.forEach((connection) => {
                        // we had a connection with id: 1:1  - meaning from me with id "1" and my port "1"
                        // now as we rename all internal nodes to have an id of the TI included
                        // e.g. my layer with id "1" and TI with id "14" results in internal layer to get id "14_1"
                        if (!connection._id){
                            return;
                        }
                        connection._id = `${singleTensorIteratorNodeId}_${connection._id}`;
                    });
                });
                
                const operator = nestedLayer$.getAttribute('type');
                this.registerOperator(operator);
                this.addNewNode(nestedNode);
            });

            // We know for sure that edges that appeared in the nested IR are not
            // aware of the external context
            mappingForNestedIR.input.forEach((nestedInput) => {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedInput.internal_layer_id}`);
                const candidate_edges = edges.filter((edge$) => {
                    return edge$.getAttribute('to-layer') === singleTensorIteratorNodeId && 
                            edge$.getAttribute('to-port') === nestedInput.external_port_id;
                });
                if (!candidate_edges.length){
                    return;
                }
                candidate_edges.forEach((candidate_edge$) => {
                    const parentLayerID = candidate_edge$.getAttribute('from-layer');
                    const parentPortID = candidate_edge$.getAttribute('from-port');
                    
                    if (!nestedNode._inputs){
                        throw Error(`Tensor Iterator node with name ${nestedNode._name} does not have inputs. Unable to process it.`);
                    }
                    const newId = `${parentLayerID}:${parentPortID}`;
                    const inputWithoutId = nestedNode._inputs.find((input) => {
                        return Boolean(input._connections.find((connection)=> !Boolean(connection._id)));
                    });
                    if (inputWithoutId) {
                        const connectionWithoutId = inputWithoutId._connections.find((connection)=> !Boolean(connection._id));
                        if (connectionWithoutId){
                            connectionWithoutId._id = newId;;
                        } 
                    } else {
                        // TODO: no tensor information in the new connection - passed as null for now
                        nestedNode._inputs.push(new openvino.Argument((nestedNode._inputs.length+1).toString(), [
                            new openvino.Connection(newId, null, null)
                        ]));
                    }
                });
            });

            mappingForNestedIR.output.forEach((nestedOutput) => {
                const nestedNode = this._nodes.find((n) => n._id === `${singleTensorIteratorNodeId}_${nestedOutput.internal_layer_id}`);

                const candidate_edges = edges.filter((edge$) => {
                    return edge$.getAttribute('from-layer') === singleTensorIteratorNodeId && 
                            edge$.getAttribute('from-port') === nestedOutput.external_port_id;
                });
                if (candidate_edges.length === 0){
                    return;
                }
                candidate_edges.forEach((candidate_edge$) => {
                    const childLayerID = candidate_edge$.getAttribute('to-layer');
                    const child = this._nodes.find((layer) => layer._id === childLayerID);
                    
                    if (!child._inputs || (child._inputs && child._inputs.length === 0)){
                        return;
                    }

                    child._inputs.forEach((input) => {
                        input._connections.forEach((connection) => {
                            if (!connection._id || (connection._id && connection._id.split(':')[0] !== singleTensorIteratorNodeId)) {
                                return;
                            }
                            const myPort = nestedNode._outputs[0]._connections[0]._id.split(':')[1];
                            connection._id = `${nestedNode.id}:${myPort}`;
                        });
                    });
                });
            });
            this._nodes = this._nodes.filter((node) => node._type !== 'TensorIterator');
        });
    }

    collectEdges(edges){
        let edgeMap = {};
        edges.forEach((edge$) => {
            var fromLayer = edge$.getAttribute('from-layer');
            var fromPort = edge$.getAttribute('from-port');
            var toLayer = edge$.getAttribute('to-layer');
            var toPort = edge$.getAttribute('to-port');
            edgeMap[toLayer + ':' + toPort] = fromLayer + ':' + fromPort;
        });
        return edgeMap;
    }

    collectPortsInformation(ports){
        return ports.reduce((acc, port$) => {
            acc.push({
                axis: port$.getAttribute("axis"),
                external_port_id: port$.getAttribute("external_port_id"),
                internal_layer_id: port$.getAttribute("internal_layer_id"),
                internal_port_id: port$.getAttribute("internal_port_id")
            });
            return acc;
        }, []);
    }

    parseMappingBlock(singleTensorIteratorNode$) {
        const portMap$ = openvino.Utils.findDirectChildrenByName(singleTensorIteratorNode$, 'port_map')[0];
        const inputs = openvino.Utils.findDirectChildrenByName(portMap$, 'input');
        const outputs = openvino.Utils.findDirectChildrenByName(portMap$, 'output');

        return {
            input: this.collectPortsInformation(inputs),
            output: this.collectPortsInformation(outputs)
        };
    }

    collectConnectionsIds(where) {
        return where.reduce((accOutput, output) => {
            const res = output._connections.reduce((accConn, connection) => {
                accConn.push(connection._id);
                return accConn;
            }, []);
            accOutput = accOutput.concat(res);
            return accOutput;
        }, []);
    }
};

openvino.Node = class {

    constructor(graph, metadata, layer$, edgeMap) {
        this._metadata = metadata;
        this._type = layer$.getAttribute('type');
        this._name = layer$.getAttribute('name') || '';
        this._id = layer$.getAttribute('id');
        this._inputs = [];
        this._outputs = [];
        this._initializers = [];
        this._attributes = [];

        var precision = layer$.getAttribute('precision');

        var inputIndex = 0;
        const input$ = openvino.Utils.findDirectChildrenByName(layer$, 'input')[0];
        if (input$) {
            const inputPorts = openvino.Utils.findDirectChildrenByName(input$, 'port');
            inputPorts.forEach((portElement$) => {
                var inputName = (inputIndex == 0) ? 'input' : inputIndex.toString(); 
                this._inputs.push(new openvino.Argument(inputName, [
                    graph._connection(this._id, precision, portElement$, edgeMap)
                ]));
                inputIndex++;
            });
        }

        var outputIndex = 0;
        const output$ = openvino.Utils.findDirectChildrenByName(layer$, 'output')[0];
        if (output$) {
            const outputPorts = openvino.Utils.findDirectChildrenByName(output$, 'port');
            outputPorts.forEach((portElement$) => {
                var outputName = (outputIndex == 0) ? 'output' : outputIndex.toString(); 
                this._outputs.push(new openvino.Argument(outputName, [
                    graph._connection(this._id, precision, portElement$, null)
                ]));
                outputIndex++;
            });
        }

        const data$ = openvino.Utils.findDirectChildrenByName(layer$, 'data')[0];
        if (data$ && data$.attributes) {
            for (var i = 0; i < data$.attributes.length; i++) {
                var key = data$.attributes[i].name;
                var value = data$.attributes[i].value;
                this._attributes.push(new openvino.Attribute(metadata, this, key, value));
            }
        }

        const blobsContainer$ = openvino.Utils.findDirectChildrenByName(layer$, 'blobs')[0];
        if (blobsContainer$){
            Array.from(blobsContainer$.childNodes)
                .filter((node) => node.nodeName != '#text')
                .forEach((blobElement$) => {
                    var name = blobElement$.nodeName;
                    var offset = parseInt(blobElement$.getAttribute('offset'));
                    var size = parseInt(blobElement$.getAttribute('size'));
                    this._initializers.push(new openvino.Argument(name, [
                        new openvino.Connection('', null, new openvino.Tensor(precision, null, offset, size))
                    ]));
                });
        }
    }

    get id() {
        return this._id;
    }

    get name() {
        return this._name;
    }

    get device() {
        return this._device || '';
    }

    get operator() {
        return this._type;
    }

    get category() {
        var schema = this._metadata.getSchema(this._type);
        return (schema && schema.category) ? schema.category : null;
    }

    get documentation() {
        var schema = this._metadata.getSchema(this._type);
        if (schema) {
            schema = JSON.parse(JSON.stringify(schema));
            schema.name = this._type;
            if (schema.description) {
                schema.description = marked(schema.description);
            }
            if (schema.attributes) {
                schema.attributes.forEach((attribute) => {
                    if (attribute.description) {
                        attribute.description = marked(attribute.description);
                    }
                });
            }
            if (schema.inputs) {
                schema.inputs.forEach((input) => {
                    if (input.description) {
                        input.description = marked(input.description);
                    }
                });
            }
            if (schema.outputs) {
                schema.outputs.forEach((output) => {
                    if (output.description) {
                        output.description = marked(output.description);
                    }
                });
            }
            if (schema.references) {
                schema.references.forEach((reference) => {
                    if (reference) {
                        reference.description = marked(reference.description);
                    }
                });
            }
            return schema;
        }
        return null;
    }

    get attributes() {
        return this._attributes;
    }

    get inputs() {
        return this._inputs.concat(this._initializers);
    }

    get outputs() {
        return this._outputs;
    }
};

openvino.Argument = class {

    constructor(name, connections) {
        this._name = name;
        this._connections = connections;
    }

    get name() {
        return this._name;
    }

    get visible() {
        return true;
    }

    get connections() {
        return this._connections;
    }
};

openvino.Connection = class {

    constructor(id, type, initializer) {
        this._id = id;
        this._type = type || null;
        this._initializer = initializer || null;
    }

    get id() {
        return this._id;
    }

    get type() {
        if (this._initializer) {
            return this._initializer.type;
        }
        return this._type;
    }

    get initializer() {
        return this._initializer;
    }
};

openvino.Attribute = class {

    constructor(metadata, node, name, value) {
        this._node = node;
        this._name = name;
        this._value = value;

        // TODO remove once schema.default fitltering is implemented
        // this._visible = false;
        
        var schema = metadata.getAttributeSchema(node.operator, name);
        if (schema) {
            if (schema.hasOwnProperty('type')) {
                switch (schema.type) {
                    case 'boolean':
                        switch (value) {
                            case 'true':
                                this._value = true;
                                break;
                            case 'false':
                                this._value = false;
                                break;
                        }
                        break;
                    case 'int32':
                        var intValue = Number.parseInt(this._value, 10);
                        this._value = Number.isNaN(this._value - intValue) ? value : intValue;
                        break;
                    case 'float32':
                    case 'float64':
                        var floatValue = Number.parseFloat(this._value);
                        this._value = Number.isNaN(this._value - floatValue) ? value : floatValue;
                        break;
                    case 'int32[]':
                        if (this._value.length > 2) {
                            var array = [];
                            this._value.split(',').map((item) => {
                                item = item.trim();
                                var intValue = Number.parseInt(item, 10);
                                if (Number.isNaN(item - intValue)) {
                                    array = null;
                                }
                                else if (array != null) {
                                    array.push(intValue);
                                }
                            });
                            if (array != null) {
                                this._value = array;
                            }
                        }
                        break;
                }
            }
            if (schema.hasOwnProperty('default')) {
                if (value == schema.default) {
                    this._visible = false;
                }
            }
        }
    }

    get name() {
        return this._name;
    }

    get value() {
        return this._value;
    }

    get visible() {
        return this._visible == false ? false : true;
    }
};

openvino.Tensor = class {

    constructor(precision, shape, offset, size) {
        this._data = null;
        this._reference = '{ offset: ' + offset.toString() + ', size: ' + size.toString() + ' }';
        this._shape = shape;
        this._type = new openvino.TensorType(precision, this._shape);
    }

    get type() {
        return this._type;
    }

    get kind() {
        return 'Blob';
    }

    get reference() {
        return this._reference;
    }

    get state() {
        return this._context().state;
    }

    get value() {
        var context = this._context();
        if (context.state) {
            return null;
        }
        context.limit = Number.MAX_SAFE_INTEGER;
        return this._decode(context, 0);
    }

    toString() {
        var context = this._context();
        if (context.state) {
            return '';
        }
        context.limit = 10000;
        var value = this._decode(context, 0);
        return JSON.stringify(value, null, 4);
    }

    _context() {
        var context = {};
        context.state = null;
        context.index = 0;
        context.count = 0;
        context.data = this._data;
        if (!this._data) {
            context.state = 'Tensor data is empty.';
            return context;
        }
        context.state = this._data.toString();
        return context;
    }

    _decode(context, dimension) {
        var results = [];
        var size = this._shape[dimension];
        if (dimension == this._shape.length - 1) {
            for (var i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(context.data[context.index]);
                context.index++;
                context.count++;
            }
        }
        else {
            for (var j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decode(context, dimension + 1));
            }
        }
        return results;
    }
};

openvino.TensorType = class {

    constructor(precision, shape) {
        switch (precision) {
            case 'FP32':
                this._dataType = 'float32';
                break;
            default:
                this._dataType = precision;
                break;
        }
        this._shape = shape;
    }

    get dataType() {
        return this._dataType;
    }

    get shape() {
        return this._shape;
    }

    toString() {
        if (this._shape == null) {
            return this.dataType + '[?]';
        }
        return this.dataType + this._shape.toString();
    }
};

openvino.TensorShape = class {

    constructor(dimensions) {
        this._dimensions = dimensions;
    }

    get dimensions() {
        return this._dimensions;
    }

    toString() {
        if (!this._dimensions || this._dimensions.length == 0) {
            return '';
        }
        return '[' + this._dimensions.join(',') + ']';
    }
};

openvino.Metadata = class {

    static open(host, callback) {
        if (openvino.Metadata._metadata) {
            callback(null, openvino.Metadata._metadata);
        }
        else {
            host.request(null, 'openvino-metadata.json', 'utf-8', (err, data) => {
                openvino.Metadata._metadata = new openvino.Metadata(data);
                callback(null, openvino.Metadata._metadata);
            });
        }
    }

    constructor(data) {
        this._map = {};
        this._attributeCache = {};
        if (data) {
            var items = JSON.parse(data);
            if (items) {
                items.forEach((item) => {
                    if (item.name && item.schema) {
                        var name = item.name;
                        var schema = item.schema;
                        this._map[name] = schema;
                    }
                });
            }
        }
    }

    getSchema(operator) {
        return this._map[operator];
    }

    getAttributeSchema(operator, name) {
        var map = this._attributeCache[operator];
        if (!map) {
            map = {};
            var schema = this.getSchema(operator);
            if (schema && schema.attributes && schema.attributes.length > 0) {
                schema.attributes.forEach((attribute) => {
                    map[attribute.name] = attribute;
                });
            }
            this._attributeCache[operator] = map;
        }
        return map[name] || null;
    }
};

openvino.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading OpenVINO model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = openvino.ModelFactory;
}
