/*jshint esversion: 6 */

hostService.initialize(openBuffer);

document.documentElement.style.overflow = 'hidden';
document.body.scroll = 'no';

var navigationButton = document.getElementById('navigation-button');
if (navigationButton) {
    navigationButton.addEventListener('click', (e) => {
        showModelSummary(modelService.activeModel);
    });
}

function updateView(page) {

    var welcomeElement = document.getElementById('welcome');
    var openFileButton = document.getElementById('open-file-button');
    var spinnerElement = document.getElementById('spinner');
    var graphElement = document.getElementById('graph');
    var navigationElement = document.getElementById('navigation-button');

    if (page == 'welcome') {
        document.body.style.cursor = 'default';
        welcomeElement.style.display = 'block';
        welcomeElement.offsetHeight;
        openFileButton.style.display = 'block';
        spinnerElement.style.display = 'none';
        graphElement.style.display = 'none';
        graphElement.style.opacity = 0;
        navigationElement.style.display = 'none';
    }

    if (page == 'spinner') {
        document.body.style.cursor = 'wait';
        welcomeElement.style.display = 'block';
        openFileButton.style.display = 'none';
        spinnerElement.style.display = 'block';
        spinnerElement.offsetHeight;
        graphElement.style.display = 'block';
        graphElement.style.opacity = 0;
        navigationElement.style.display = 'none';
    }

    if (page == 'graph') {
        welcomeElement.style.display = 'none';
        openFileButton.style.display = 'none';
        spinnerElement.style.display = 'none';
        graphElement.style.display = 'block';
        graphElement.style.opacity = 1;
        navigationElement.style.display = 'block';
        document.body.style.cursor = 'default';
    }
}

function openBuffer(err, buffer, identifier) {
    sidebar.close();
    if (err) {
        hostService.showError(err.toString());
        updateView('welcome');
    }
    else {
        setTimeout(function () {
            modelService.openBuffer(buffer, identifier, function(err, model) {
                if (err) {
                    hostService.showError(err);
                    updateView('welcome');
                    return;
                }
                setTimeout(function () {
                    updateGraph(model);
                }, 20);   
            });    
        }, 20);
    }
}

function updateActiveGraph(name) {
    sidebar.close();
    var model = modelService.activeModel;
    if (model) {
        model.updateActiveGraph(name);
        updateView('spinner');
        setTimeout(function () {
            updateGraph(model);
        }, 250);
    }
}

function updateGraph(model) {

    var graph = model.activeGraph;
    if (!graph) {
        this.updateView('graph');
        return;
    }

    var svgElement = document.getElementById('graph');
    while (svgElement.lastChild) {
        svgElement.removeChild(svgElement.lastChild);
    }

    var g = new dagre.graphlib.Graph();
    g.setGraph({ });
    // g.setGraph({ align: 'DR' });
    // g.setGraph({ ranker: 'network-simplex' });
    // g.setGraph({ ranker: 'tight-tree' });
    // g.setGraph({ ranker: 'longest-path' });
    // g.setGraph({ acyclicer: 'greedy' });
    g.setDefaultEdgeLabel(function() { return {}; });

    var nodeId = 0;
    var edgeMap = {};

    graph.nodes.forEach((node) => {
        var formatter = new NodeFormatter();
        var style = node.constant ? 'node-item-constant' : 'node-item-operator';
        var primitive = node.primitive;
        formatter.addItem(primitive ? primitive : node.operator, style, node.name, function() { 
            showNodeOperatorDocumentation(node);
        });

        node.inputs.forEach((input) => {
            var hasInitializers = false;
            input.connections.forEach((connection) => {
                if (connection.initializer) {
                    hasInitializers = true;
                }
            });
            if (hasInitializers) {
                input.connections.forEach((connection) => {
                    if (connection.initializer) {
                        formatter.addItem(input.name, 'node-item-constant', connection.type, function() { 
                            showTensor(model, connection.initializer);
                        });
                    }
                    else {
                        formatter.addItem(input.name, null, input.type, null);
                        var tuple = edgeMap[connection.id];
                        if (!tuple) {
                            tuple = { from: null, to: [] };
                            edgeMap[connection.id] = tuple;
                        }
                        tuple.to.push({ 
                            node: nodeId, 
                            name: input.name,
                            control: connection.control
                        });
                    }
                });
            }
            else {
                formatter.addItem(input.name, null, input.type, null);
                input.connections.forEach((connection) => {
                    var tuple = edgeMap[connection.id];
                    if (!tuple) {
                        tuple = { from: null, to: [] };
                        edgeMap[connection.id] = tuple;
                    }
                    tuple.to.push({ 
                        node: nodeId, 
                        name: input.name
                    });
                });
            }
        });

        node.outputs.forEach((output) => {
            var outputIds = Array.isArray(output.id) ? output.id : [ output.id ];
            outputIds.forEach((outputId) => {
                var tuple = edgeMap[outputId];
                if (!tuple) {
                    tuple = { from: null, to: [] };
                    edgeMap[outputId] = tuple;
                }
                tuple.from = { 
                    node: nodeId,
                    name: output.name
                };    
            });
        });

        node.dependencies.forEach((dependency) => {
            var tuple = edgeMap[dependency];
            if (!tuple) {
                tuple = { from: null, to: [] };
                edgeMap[dependency] = tuple;
            }
            tuple.to.push({ 
                node: nodeId, 
                dependency: true
            });
        });

        if (node.attributes && !primitive) {
            formatter.setAttributeHandler(() => { 
                showNodeDetails(node);
            });
            node.attributes.forEach((attribute) => {
                if (attribute.hidden) {
                }
                else {
                    var attributeValue = '';
                    if (attribute.tensor) {
                        attributeValue = '[...]';
                    }
                    else {
                        attributeValue = attribute.value;
                        if (attributeValue.length > 25) {
                            attributeValue = attributeValue.substring(0, 25) + '...';
                        }
                    }
                    formatter.addAttribute(attribute.name, attributeValue, attribute.type);
                }
            });
        }

        g.setNode(nodeId++, { label: formatter.format(svgElement) });
    });

    graph.inputs.forEach((input) => {
        var tuple = edgeMap[input.id];
        if (!tuple) {
            tuple = { from: null, to: [] };
            edgeMap[input.id] = tuple;
        }
        tuple.from = { 
            node: nodeId,
            // name: valueInfo.name
        };

        var formatter = new NodeFormatter();
        formatter.addItem(input.name, null, input.type, null);
        g.setNode(nodeId++, { label: formatter.format(svgElement), class: 'graph-input' } ); 
    });

    graph.outputs.forEach((output) => {
        var outputId = output.id;
        var outputName = output.name;
        var tuple = edgeMap[outputId];
        if (!tuple) {
            tuple = { from: null, to: [] };
            edgeMap[outputId] = tuple;
        }
        tuple.to.push({ 
            node: nodeId,
            // name: valueInfo.name
        });

        var formatter = new NodeFormatter();
        formatter.addItem(output.name, null, output.type, null);
        g.setNode(nodeId++, { label: formatter.format(svgElement) } ); 
    });

    Object.keys(edgeMap).forEach((edge) => {
        var tuple = edgeMap[edge];
        if (tuple.from != null) {
            tuple.to.forEach(function (to) {
                var text = '';
                if (tuple.from.name && to.name) {
                    text = tuple.from.name + ' => ' + to.name;
                }
                else if (tuple.from.name) {
                    text = tuple.from.name;
                }
                else {
                    text = to.name;
                }

                if (to.dependency) { 
                    g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee', curve: d3.curveBasis, class: 'edge-path-control' } );
                }
                else {
                    g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee', curve: d3.curveBasis } );
                }
            });
        }
        else {
            console.log('?');
        }

        if (tuple.from == null || tuple.to.length == 0) {
            console.log(edge);
        }
    });

    // Workaround for Safari background drag/zoom issue:
    // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
    var rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttribute('width', '100%');
    rect.setAttribute('height', '100%');
    rect.setAttribute('fill', 'none');
    rect.setAttribute('pointer-events', 'all');
    svgElement.appendChild(rect);

    var outputGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    svgElement.appendChild(outputGroup);

    // Set up zoom support
    var zoom = d3.zoom();
    zoom.scaleExtent([0.1, 2]);
    zoom.on('zoom', function(e) {
        outputGroup.setAttribute('transform', d3.event.transform.toString());
    });
    var svg = d3.select(svgElement);
    // zoom.transform(svg, d3.zoomIdentity);
    zoom(svg);

    setTimeout(function () {

        var graphRenderer = new GraphRenderer(outputGroup);
        graphRenderer.render(g);

        var svgSize = svgElement.getBoundingClientRect();

        var inputElements = svgElement.getElementsByClassName('graph-input');
        if (inputElements && inputElements.length > 0) {
            // Center view based on input elements
            var x = 0;
            var y = 0;
            for (var i = 0; i < inputElements.length; i++) {
                var inputTransform = inputElements[i].transform.baseVal.consolidate().matrix;
                x += inputTransform.e;
                y += inputTransform.f;
            }
            x = x / inputElements.length;
            y = y / inputElements.length;

            zoom.translateBy(svg, (svgSize.width / 2) - x, (svgSize.height / 4) - y);
        }
        else {
            zoom.translateBy(svg, (svgSize.width - g.graph().width) / 2, (svgSize.height - g.graph().height) / 2);
        }    
    
        updateView('graph');
    }, 20);
}

function showModelSummary(model) {
    var view = model.format();
    if (view) {
        var template = Handlebars.compile(summaryTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, 'Summary', '100%');
    }
}

function showTensor(model, tensor) {
    if (tensor) {
        var view = { 'items': [ tensor ] };
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, tensor.title ? tensor.title : 'Tensor');
    }
}

function showNodeOperatorDocumentation(node) {
    var documentation = node.documentation;
    if (documentation) {
        sidebar.open(documentation, 'Documentation');
    }
}

function showNodeDetails(node) {
    if (node) {
        var template = Handlebars.compile(nodeTemplate, 'utf-8');
        var data = template(node);
        sidebar.open(data, 'Node Details');
    }
}

function showNodeProperties(node) {
    if (node.properties) {
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template({ 'items': node.properties });
        sidebar.open(data, 'Node Properties');
    }
}

function showNodeAttributes(node) {
    if (node.attributes) {
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template({ 'items': node.attributes });
        sidebar.open(data, 'Node Attributes');
    }
}

class Sidebar {

    constructor() {
        this._closeSidebarHandler = (e) => {
            this.close();
        };
        this._closeSidebarKeyDownHandler = (e) => {
            if (e.keyCode == 27) {
                e.preventDefault();
                this.close();
            }
        };
        this._resizeSidebarHandler = (e) => {
            var contentElement = document.getElementById('sidebar-content');
            if (contentElement) {
                contentElement.style.height = window.innerHeight - 60;
            }
        };
    
    }

    open(content, title, width) {
        var sidebarElement = document.getElementById('sidebar');
        var titleElement = document.getElementById('sidebar-title');
        var contentElement = document.getElementById('sidebar-content');
        var closeButtonElement = document.getElementById('sidebar-closebutton');
        if (sidebarElement && contentElement && closeButtonElement && titleElement) {
            titleElement.innerHTML = title ? title.toUpperCase() : '';
            window.addEventListener('resize', this._resizeSidebarHandler);
            document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
            closeButtonElement.addEventListener('click', this._closeSidebarHandler);
            closeButtonElement.style.color = '#818181';
            contentElement.style.height = window.innerHeight - 60;
            contentElement.innerHTML = content;
            sidebarElement.style.width = width ? width : '500px';    
            if (width && width.endsWith('%')) {
                contentElement.style.width = '100%';
            }
            else {
                contentElement.style.width = 'calc(' + sidebarElement.style.width + ' - 40px)';
            }
        }
    }
    
    close() {
        var sidebarElement = document.getElementById('sidebar');
        var contentElement = document.getElementById('sidebar-content');
        var closeButtonElement = document.getElementById('sidebar-closebutton');
        if (sidebarElement && contentElement && closeButtonElement) {
            document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
            sidebarElement.removeEventListener('resize', this._resizeSidebarHandler);
            closeButtonElement.removeEventListener('click', this._closeSidebarHandler);
            closeButtonElement.style.color = '#f8f8f8';
            sidebarElement.style.width = '0';
        }
    }
}

var sidebar = new Sidebar();

class ModelService {

    constructor(hostService) {
        this.hostService = hostService;
    }

    openBuffer(buffer, identifier, callback) {
        var model = null;
        var err = null;
    
        var extension = identifier.split('.').pop();
    
        if (extension == 'tflite') {
            TensorFlowLiteModel.open(buffer, identifier, hostService, (err, model) => {
                this._activeModel = model;
                callback(err, model);
           });
        }
        else if (identifier == 'saved_model.pb') {
            TensorFlowModel.open(buffer, identifier, hostService, (err, model) => {
                this._activeModel = model;
                callback(err, model);
            });
        }
        else if (extension == 'onnx') {
            OnnxModel.open(buffer, identifier, hostService, (err, model) => {
                this._activeModel = model;
                callback(err, model);
            });
        }
        else if (extension == 'pb') {
            OnnxModel.open(buffer, identifier, hostService, (err, model) => {
                if (!err) {
                    this._activeModel = model;
                    callback(err, model);    
                }
                else {
                    TensorFlowModel.open(buffer, identifier, hostService, (err, model) => {
                        this._activeModel = model;
                        callback(err, model);
                    });
                }
            });
        }
    }

    get activeModel() {
        return this._activeModel;
    }
}

var modelService = new ModelService(hostService);

class Int64 {

    constructor(buffer) {
        this._buffer = buffer;
    }

    toString(radix) {
        var high = this.readInt32(4);
        var low = this.readInt32(0);
        var str = '';
        var sign = high & 0x80000000;
        if (sign) {
            high = ~high;
            low = 0x100000000 - low;
        }
        radix = radix || 10;
        while (true) {
            var mod = (high % radix) * 0x100000000 + low;
            high = Math.floor(high / radix);
            low = Math.floor(mod / radix);
            str = (mod % radix).toString(radix) + str;
            if (!high && !low) 
            {
                break;
            }
        }
        if (sign) {
            str = "-" + str;
        }
        return str;
    }

    readInt32(offset) {
      return (this._buffer[offset + 3] * 0x1000000) + (this._buffer[offset + 2] << 16) + (this._buffer[offset + 1] << 8) + this._buffer[offset + 0];
    }
}

class Uint64 {

    constructor(buffer) {
        this._buffer = buffer;
    }

    toString(radix) {
        var high = this.readInt32(4);
        var low = this.readInt32(0);
        var str = '';
        radix = radix || 10;
        while (true) {
            var mod = (high % radix) * 0x100000000 + low;
            high = Math.floor(high / radix);
            low = Math.floor(mod / radix);
            str = (mod % radix).toString(radix) + str;
            if (!high && !low) 
            {
                break;
            }
        }
        return str;
    }

    readInt32(offset) {
      return (this._buffer[offset + 3] * 0x1000000) + (this._buffer[offset + 2] << 16) + (this._buffer[offset + 1] << 8) + this._buffer[offset + 0];
    }
}
