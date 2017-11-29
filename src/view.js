
const onnx = protobuf.roots.onnx.onnx;

debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

hostService.registerCallback(openBuffer);
var modelService = null;

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
    var propertiesElement = document.getElementById('properties-button');

    if (page == 'welcome') {
        welcomeElement.style.display = 'block';
        clockElement.style.display = 'none';
        document.body.style.cursor = 'default';
        propertiesElement.style.display = 'none';
    }

    if (page == 'clock') {
        welcomeElement.style.display = 'none';
        clockElement.style.display = 'block';
        document.body.style.cursor = 'wait';
        propertiesElement.style.display = 'none';
    }

    if (page == 'graph') {
        welcomeElement.style.display = 'none';
        clockElement.style.display = 'none';
        document.body.style.cursor = 'default';
        propertiesElement.style.display = 'block';
    }
}

function openBuffer(err, buffer) {
    if (err) {
        hostService.showError(err.toString());
        updateView('welcome');
    }
    else {
        modelService = new OnnxModelService(hostService);
        var err = modelService.openBuffer(buffer);        
        if (err) {
            hostService.showError('Decoding failure: ' + err);
            updateView('welcome');
            return;
        }

        renderModel();
    }
}

function renderModel() {

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
                var result = modelService.formatTensor(initializer);
                formatter.addItem(name, 'node-item-input', result['type'], function() { 
                    showInitializer(initializer);
                });
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

        var properties = modelService.formatNodeProperties(node);
        if (properties) {
            formatter.setPropertyHandler(function() { showNodeProperties(node) });
            properties.forEach(function (property) {
                formatter.addProperty(property['name'], property['value_short']());
            });
        }

        var attributes = modelService.formatNodeAttributes(node);
        if (attributes) {
            formatter.setAttributeHandler(function() { showNodeAttributes(node); });
            attributes.forEach(function (attribute) {
                formatter.addAttribute(attribute['name'], attribute['value_short'](), attribute['type']);
            });
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
            g.setNode(nodeId++, { label: formatter.format(svg).node(), class: 'graph-input', labelType: 'svg', padding: 0 } ); 
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

    // Workaround for Safari background drag/zoom issue:
    // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
    svg.insert('rect', ':first-child').attr('width', '100%').attr('height', '100%').attr('fill', 'none').attr('pointer-events', 'all');

    var inputElements = svgElement.getElementsByClassName('graph-input');
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
        var text = modelService.formatElementType(tensorType.elemType); 
        if (tensorType.shape && tensorType.shape.dim) {
            text += '[' + tensorType.shape.dim.map(dimension => dimension.dimValue.toString()).join(',') + ']';
        }
        return text;
    }
    if (type.value == 'mapType') {
        var mapType = type.mapType;
        return '<' + modelService.formatElementType(mapType.keyType) + ', ' + formatType(mapType.valueType) + '>';
    }
    debugger;
    return '[UNKNOWN]';
}

function showDocumentation(operator) {
    var documentation = modelService.getOperatorService().getOperatorDocumentation(operator);
    if (documentation) {
        sidebar.open(documentation, 'Documentation');
    }
}

function showModelProperties() {
    var view = modelService.formatModelProperties();
    if (view) {
        var template = Handlebars.compile(modelPropertiesTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, 'Model Properties');
    }
}

function showInitializer(initializer) {
    var initializer = modelService.formatTensor(initializer);
    if (initializer) {
        var view = { 'items': [ initializer ] };
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, 'Initializer');
    }
}

function showNodeProperties(node) {
    var properties = modelService.formatNodeProperties(node);
    if (properties) {
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template({ 'items': properties });
        sidebar.open(data, 'Node Properties');
    }
}

function showNodeAttributes(node) {
    var attributes = modelService.formatNodeAttributes(node);
    if (attributes) {
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template({ 'items': attributes });
        sidebar.open(data, 'Node Attributes');
    }
}

function Sidebar() {
    var self = this;
    this.closeSidebarHandler = function (e) {
        self.close();
    };
    this.closeSidebarKeyDownHandler = function (e) {
        if (e.keyCode == 27) {
            e.preventDefault()
            self.close();
        }
    }
}

Sidebar.prototype.open = function(content, title) {
    var sidebarElement = document.getElementById('sidebar');
    var titleElement = document.getElementById('sidebar-title');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement && titleElement) {
        titleElement.innerHTML = title ? title.toUpperCase() : '';
        closeButtonElement.addEventListener('click', this.closeSidebarHandler);
        closeButtonElement.style.color = '#818181';
        document.addEventListener('keydown', this.closeSidebarKeyDownHandler);
        contentElement.style.height = window.innerHeight - 60;
        sidebarElement.style.height = window.innerHeight;
        contentElement.innerHTML = content
        contentElement.style.width = '460px';
        sidebarElement.style.width = '500px';
    }
}

Sidebar.prototype.close = function() {
    var sidebarElement = document.getElementById('sidebar');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement) {
        closeButtonElement.removeEventListener('click', this.closeSidebarHandler);
        closeButtonElement.style.color = '#f8f8f8';
        document.removeEventListener('keydown', this.closeSidebarKeyDownHandler);
        sidebarElement.style.width = '0';
    }
}

var sidebar = new Sidebar();