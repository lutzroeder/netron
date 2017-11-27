
const onnx = protobuf.roots.onnx.onnx;

debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

hostService.registerCallback(openBuffer);
var modelService = new OnnxModelService(hostService);

window.addEventListener('load', function(e) {
    updateSize();
});

window.addEventListener('resize', function() {
    updateSize();
});

function updateSize() {
    var graphElement = document.getElementById('graph');
    if (graphElement) {
        graphElement.setAttribute('width', window.innerWidth.toString());
        graphElement.setAttribute('height', window.innerHeight.toString());
    }
}

function updateView(page) {
    updateSize();

    var welcomeElement = document.getElementById('welcome');
    var clockElement = document.getElementById('clock');

    if (page == 'welcome') {
        welcomeElement.style.display = 'block';
        clockElement.style.display = 'none';
        document.body.style.cursor = 'default';
        
    }

    if (page == 'clock') {
        welcomeElement.style.display = 'none';
        clockElement.style.display = 'block';
        document.body.style.cursor = 'wait';
    }

    if (page == 'graph') {
        welcomeElement.style.display = 'none';
        clockElement.style.display = 'none';
        document.body.style.cursor = 'default';
    }
}

function openBuffer(err, buffer) {
    if (err) {
        hostService.showError(err.toString());
        updateView('welcome');
    }
    else {
        try {
            modelService.openBuffer(buffer);
        }
        catch (err) {
            hostService.showError('Decoding failure: ' + err);
            updateView('welcome');
            return;
        }

        renderModel(modelService.model);
    }
}

function renderModel(model) {

    var svgElement = document.getElementById('graph');
    while (svgElement.lastChild) {
        svgElement.removeChild(svgElement.lastChild);
    }
    var svg = dagreD3.d3.select(svgElement);

    var g = new dagreD3.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(() => { return {}; });

    var nodeId = 0;
    var edgeMap = {};

    var graph = modelService.model.graph;
    
    var initializerMap = {};
    graph.initializer.forEach(function (tensor) {
        initializerMap[tensor.name] = tensor;
    });
    
    graph.node.forEach(function (node) {

        var operator = node.opType;

        var formatter = new NodeFormatter();

        formatter.addItem(operator, 'node-item-operator', null, function() { showDocumentation(operator) });

        node.input.forEach(function (edge, index)
        {
            var name = modelService.getOperatorService().getInputName(operator, index);
            if (!name) {
                name = '(' + index.toString() + ')';
            }
    
            var initializer = initializerMap[edge];
            if (initializer) {
                var type = formatTensorType(initializer);
                formatter.addItem(name, 'node-item-input', type, function() { showInitializer(initializer); });
            }
            else {
                // TODO is there a way to infer the type of the input?
                formatter.addItem(name, null, null, null);

                var tuple = edgeMap[edge];
                if (!tuple) {
                    tuple = { from: null, to: [] };
                    edgeMap[edge] = tuple;
                }
                tuple.to.push({ 
                    node: nodeId, 
                    name: name
                });
            }
        });

        node.output.forEach(function (edge, index)
        {
            var tuple = edgeMap[edge];
            if (!tuple) {
                tuple = { from: null, to: [] };
                edgeMap[edge] = tuple;
            }
            tuple.from = { 
                node: nodeId,
                name: modelService.getOperatorService().getOutputName(operator, index) 
            };
        });

        if (node.name || node.docString || node.domain) {
            if (node.name) {
                formatter.addProperty('name', node.name);
            }
            if (node.docString) {
                var doc = node.docString;
                if (doc.length > 50) {
                    doc = doc.substring(0, 25) + '...';
                }
                formatter.addProperty('doc', doc);
            }
            if (node.domain) {
                formatter.addProperty('domain', node.domain);
            }
            formatter.setPropertyHandler(function() { showNodeProperties(node) });
        }

        if (node.attribute && node.attribute.length > 0) {
            node.attribute.forEach(function (attribute) {
                var attributeValue = formatAttributeValue(attribute);
                if (attributeValue.length > 25)
                {
                    attributeValue = attributeValue.substring(0, 25) + '...';
                }
                var attributeType = formatAttributeType(attribute);
                formatter.addAttribute(attribute.name, attributeValue, attributeType);
            });

            formatter.setAttributeHandler(function() { showNodeAttributes(node.attribute); });
        }

        g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 });
    });

    graph.input.forEach(function (valueInfo) {
        if (!initializerMap[valueInfo.name]) {
            var tuple = edgeMap[valueInfo.name];
            if (!tuple) {
                tuple = { from: null, to: [] };
                edgeMap[valueInfo.name] = tuple;
            }
            tuple.from = { 
                node: nodeId,
                // name: valueInfo.name
            };
    
            var type = formatType(valueInfo.type);

            var formatter = new NodeFormatter();
            formatter.addItem(valueInfo.name, null, type, null);
            g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 } ); 
        }
    });

    graph.output.forEach(function (valueInfo) {
        
        var tuple = edgeMap[valueInfo.name];
        if (!tuple) {
            tuple = { from: null, to: [] };
            edgeMap[valueInfo.name] = tuple;
        }
        tuple.to.push({ 
            node: nodeId,
            // name: valueInfo.name
        });

        var formatter = new NodeFormatter();
        formatter.addItem(valueInfo.name, null, null, null);
        g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 } ); 
    });

    Object.keys(edgeMap).forEach(function (edge) {
        var tuple = edgeMap[edge];
        if (tuple.from != null) {
            tuple.to.forEach(function (to) {
                var text = ''
                if (tuple.from.name && to.name) {
                    text = tuple.from.name + ' => ' + to.name;                    
                }
                else if (tuple.from.name) {
                    text = tuple.from.name;
                }
                else {
                    text = to.name;
                }

                g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee' });
            })
        }
        else {
            console.log('?');
        }

        if (tuple.from == null || tuple.to.length == 0) {
            console.log(edge);
        }
    });

    var inner = svg.append('g');

    // Set up zoom support
    var zoom = dagreD3.d3.behavior.zoom().scaleExtent([0.2, 2]).on('zoom', function() {
        inner.attr('transform', 'translate(' + dagreD3.d3.event.translate + ')' + 'scale(' + dagreD3.d3.event.scale + ')');
    });
    svg.call(zoom);
    
    var render = new dagreD3.render();
    render(dagreD3.d3.select('svg g'), g);

    var inputElements = svgElement.getElementsByClassName('input');
    if (inputElements && inputElements.length > 0) {
        // Center view based on input elements
        var x = 0;
        var y = 0;
        for (var i = 0; i < inputElements.length; i++) {
            var inputTransform = dagreD3.d3.transform(dagreD3.d3.select(inputElements[i]).attr('transform'));
            x += inputTransform.translate[0];
            y += inputTransform.translate[1];
        }
        x = x / inputElements.length;
        y = y / inputElements.length;
        zoom.translate([ 
            (svg.attr('width') / 2) - x,
            (svg.attr('height') / 4) - y ]).event(svg);
    }
    else {
        zoom.translate([ (svg.attr('width') - g.graph().width) / 2, 40 ]).event(svg);
    }    

    updateView('graph');
}

function formatType(type) {
    if (type.value == 'tensorType') {
        var tensorType = type.tensorType;
        var text = formatElementType(tensorType.elemType); 
        if (tensorType.shape && tensorType.shape.dim) {
            text += '[' + tensorType.shape.dim.map(dimension => dimension.dimValue.toString()).join(',') + ']';
        }
        return text;
    }
    if (type.value == 'mapType') {
        var mapType = type.mapType;
        return '<' + formatElementType(mapType.keyType) + ', ' + formatType(mapType.valueType) + '>';
    }
    debugger;
    return '[UNKNOWN]';
}

function formatTensorType(type) {
    var text = formatElementType(type.dataType);
    if (type.dims) {
        text += '[' + type.dims.map(dimension => dimension.toString()).join(',') + ']';        
    }
    return text;
}

const elementTypeMap = { };
elementTypeMap[onnx.TensorProto.DataType.UNDEFINED] = 'UNDEFINED';
elementTypeMap[onnx.TensorProto.DataType.FLOAT] = 'float';
elementTypeMap[onnx.TensorProto.DataType.UINT8] = 'uint8';
elementTypeMap[onnx.TensorProto.DataType.INT8] = 'int8';
elementTypeMap[onnx.TensorProto.DataType.UINT16] = 'uint16';
elementTypeMap[onnx.TensorProto.DataType.INT16] = 'int16';
elementTypeMap[onnx.TensorProto.DataType.INT32] = 'int32';
elementTypeMap[onnx.TensorProto.DataType.INT64] = 'int64';
elementTypeMap[onnx.TensorProto.DataType.STRING] = 'string';
elementTypeMap[onnx.TensorProto.DataType.BOOL] = 'bool';
elementTypeMap[onnx.TensorProto.DataType.FLOAT16] = 'float16';
elementTypeMap[onnx.TensorProto.DataType.DOUBLE] = 'double';
elementTypeMap[onnx.TensorProto.DataType.UINT32] = 'uint32';
elementTypeMap[onnx.TensorProto.DataType.UINT64] = 'uint64';
elementTypeMap[onnx.TensorProto.DataType.COMPLEX64] = 'complex64';
elementTypeMap[onnx.TensorProto.DataType.COMPLEX128] = 'complex128';

function formatElementType(elementType) {
    var name = elementTypeMap[elementType];
    if (name) {
        return name;
    }
    debugger;
    return elementTypeMap[onnx.TensorProto.DataType.UNDEFINED];
}

function formatAttributeType(attribute) {
    if (attribute.type) { 
        var elementType = formatElementType(attribute.type);
        if ((attribute.ints && attribute.ints.length > 0) ||
            (attribute.floats && attribute.floats.length > 0) ||
            (attribute.strings && attribute.strings.length > 0))
        {
            return elementType + '[]';
        }
        return elementType;
    }
    return null;
}

function formatAttributeValue(attribute) {
    if (attribute.ints && attribute.ints.length > 0) {
        return attribute.ints.map(v => v.toString()).join(', ');
    }
    if (attribute.floats && attribute.floats.length > 0) {
        return attribute.floats.map(v => v.toString()).join(', ');
    }
    if (attribute.strings && attribute.strings.length > 0) {
        return attribute.strings.map(function(s) {
            if (s.filter(c => c <= 32 && c >= 128).length == 0) {
                return '"' + String.fromCharCode.apply(null, s) + '"';
            }
            return s.map(v => v.toString()).join(', ');    
        }).join(', ');
    }
    if (attribute.s && attribute.s.length > 0) {
        if (attribute.s.filter(c => c <= 32 && c >= 128).length == 0) {
            return '"' + String.fromCharCode.apply(null, attribute.s) + '"';
        }
        return attribute.s.map(v => v.toString()).join(', ');
    }
    if (attribute.f && attribute.f != 0 || attribute.f == 0 || isNaN(attribute.f)) {
        return attribute.f.toString();
    }
    if (attribute.i && (attribute.i != 0 || attribute.i == 0)) {
        return attribute.i.toString();
    }
    debugger;
    return '?';
}

function showDocumentation(operator) {
    var documentation = modelService.getOperatorService().getOperatorDocumentation(operator);
    if (documentation) {
        openSidebar(documentation, 'Documentation');
    }
}

function showModelProperties() {
    var view = modelService.getModelProperties();
    if (view) {
        var template = Handlebars.compile(modelPropertiesTemplate, 'utf-8');
        var data = template(view);
        openSidebar(data, 'Model Properties');
    }
}

function showInitializer(initializer) {
    var view = { 'items': [] };
    view['items'].push({
        'name': initializer.name,
        'type': formatTensorType(initializer),
        'value': formatTensorValue(initializer)
    });

    var template = Handlebars.compile(itemsTemplate, 'utf-8');
    var data = template(view);
    openSidebar(data, 'Initializer');
}

function formatTensorValue(tensor) {

    // var formatter = new TensorFormatter(tensor);
    // return formatter.format();

    return '// TODO ';
}

function TensorFormatter(tensor) {
    this.tensor = tensor;
    this.elementType = tensor.dataType;
    this.dimensions = tensor.dims;
    if (this.elementType == onnx.TensorProto.DataType.FLOAT && this.dimensions && tensor.floatData) {
        this.data = tensor.floatData;
    }
    if (this.data) {
        this.index = 0;
        this.output = this.read(0);
    }
    else {
        debugger;
    }
}

TensorFormatter.prototype.read = function(dimension) {
    var size = this.dimensions[dimension];
    var result = [];
    if (dimension == this.dimensions.length - 1) {
        for (var i = 0; i < size; i++) {
            result.push(this.data[this.index++]);
        }
    }
    else {
        for (var i = 0; i < size; i++) {
            result.push(this.read(dimension + 1));
        }
    }
    return result;
};

TensorFormatter.prototype.format = function() { 
    if (this.output) {
        return JSON.stringify(this.output);
    }
    debugger;
    return '?';
};

function showNodeProperties(node) {
    if (node.name || node.docString || node.domain) {
        
        var view = { 'items': [] };        
        if (node.name) {
            view['items'].push({ 'name': 'Name', 'value': node.name });
        }
        if (node.docString) {
            view['items'].push({ 'name': 'Documentation', 'value': node.docString });
        }
        if (node.domain) {
            view['items'].push({ 'name': 'Domain', 'value': node.domain });
        }

        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template(view);
        openSidebar(data, 'Node Properties');
    }
}

function showNodeAttributes(attributes) {
    if (attributes && attributes.length > 0) {

        var view = { 'items': [] };        

        if (attributes && attributes.length > 0) {
            attributes.forEach(function (attribute) { 
                var item = {};
                item['name'] = attribute.name;
                item['value'] = formatAttributeValue(attribute);
                item['type'] = formatAttributeType(attribute);
                if (attribute.docString) {
                    item['doc'] = attribute.docString;
                }
                view['items'].push(item);
            });
        }

        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template(view);
        openSidebar(data, 'Node Attributes');
    }
}

var closeSidebarHandler = closeSidebar;
var closeSidebarKeyDownHandler = closeSidebarKeyDown; 

function openSidebar(content, title) {
    var sidebarElement = document.getElementById('sidebar');
    var titleElement = document.getElementById('sidebar-title');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement && titleElement) {
        titleElement.innerHTML = title ? title.toUpperCase() : '';
        closeButtonElement.addEventListener('click', closeSidebarHandler);
        closeButtonElement.style.color = '#818181';
        document.addEventListener('keydown', closeSidebarKeyDownHandler);
        contentElement.style.height = window.innerHeight - 60;
        sidebarElement.style.height = window.innerHeight;
        contentElement.innerHTML = content
        contentElement.style.width = '460px';
        sidebarElement.style.width = '500px';
    }
}

function closeSidebarKeyDown(e) {
    if (e.keyCode == 27) {
        e.preventDefault()
        closeSidebar();
    }
}

function closeSidebar() {
    var sidebarElement = document.getElementById('sidebar');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement) {
        closeButtonElement.removeEventListener('click', closeSidebarHandler);
        closeButtonElement.style.color = '#f8f8f8';
        document.removeEventListener('keydown', closeSidebarKeyDownHandler);
        sidebarElement.style.width = '0';
    }
}
