/*jshint esversion: 6 */

debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

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
    var svg = dagreD3.d3.select(svgElement);

    var g = new dagreD3.graphlib.Graph();
    g.setGraph({});
    g.setDefaultEdgeLabel(function() { return {}; });

    var nodeId = 0;
    var edgeMap = {};

    var initializerMap = {};
    graph.initializers.forEach((initializer) => {
        var id = initializer.id;
        initializerMap[id] = initializer;
    });

    graph.nodes.forEach((node) => {
        var formatter = new NodeFormatter();
        var style = node.constant ? 'node-item-constant' : 'node-item-operator';
        var primitive = node.primitive;
        formatter.addItem(primitive ? primitive : node.operator, style, node.name, function() { 
            showNodeOperatorDocumentation(node);
        });

        var hasInitializerInputs = false;
        node.inputs.forEach((input) => {
            if (initializerMap[input.id]) {
                hasInitializerInputs = true;
            }
        });

        node.inputs.forEach((input) => {
            var inputId = input.id;
            var initializer = initializerMap[inputId];
            if (initializer) {
                if (!primitive || hasInitializerInputs) {
                    formatter.addItem(input.name, 'node-item-constant', initializer.type, function() { 
                        showTensor(model, initializer);
                    });
                }
            }
            else {
                // TODO is there a way to infer the type of the6 input?
                if (!primitive || hasInitializerInputs) {
                    formatter.addItem(input.name, null, input.type, null);
                }
                var tuple = edgeMap[inputId];
                if (!tuple) {
                    tuple = { from: null, to: [] };
                    edgeMap[inputId] = tuple;
                }
                tuple.to.push({ 
                    node: nodeId, 
                    name: input.name
                });
            }
        });

        node.outputs.forEach((output) =>
        {
            var outputId = output.id;
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

        if (node.attributes && !primitive) {
            formatter.setAttributeHandler(() => { 
                showNodeDetails(node);
            });
            node.attributes.forEach((attribute) => {
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
            });
        }

        g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 });
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
        g.setNode(nodeId++, { label: formatter.format(svg).node(), class: 'graph-input', labelType: 'svg', padding: 0 } ); 
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
        g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 } ); 
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

                g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee' });
            });
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
    var zoom = dagreD3.d3.behavior.zoom().scaleExtent([0.1, 2]).on('zoom', function() {
        inner.attr('transform', 'translate(' + dagreD3.d3.event.translate + ')' + 'scale(' + dagreD3.d3.event.scale + ')');
    });
    svg.call(zoom);

    setTimeout(function () {

        var render = new dagreD3.render();
        render(dagreD3.d3.select('svg g'), g);
    
        // Workaround for Safari background drag/zoom issue:
        // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
        svg.insert('rect', ':first-child').attr('width', '100%').attr('height', '100%').attr('fill', 'none').attr('pointer-events', 'all');
    
        var svgSize = svgElement.getBoundingClientRect();

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
                (svgSize.width / 2) - x,
                (svgSize.height / 4) - y ]).event(svg);
        }
        else {
        //    zoom.translate([ (svgSize.width - g.graph().width) / 2, 40 ]).event(svg);
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
    
        if (identifier != null && extension == 'tflite')
        {
            model = new TensorFlowLiteModel(hostService); 
            err = model.openBuffer(buffer, identifier);
        }
        else if (identifier != null && identifier == 'saved_model.pb') {
            model = new TensorFlowModel(hostService);
            err = model.openBuffer(buffer, identifier);
        }
        else if (extension == 'onnx') {
            model = new OnnxModel(hostService);
            err = model.openBuffer(buffer, identifier);
        }
        else if (extension == 'pb') {
            model = new OnnxModel(hostService);
            err = model.openBuffer(buffer, identifier);
            if (err) {
                model = new TensorFlowModel(hostService);
                err = model.openBuffer(buffer, identifier);
            }
        }
    
        if (err) {
            callback(err, null);
        }
        else {
            this._activeModel = model;
            callback(null, model);
        }
    }

    get activeModel() {
        return this._activeModel;
    }
}

var modelService = new ModelService(hostService);

class Int64 {
    constructor(data) {
        this._data = data;
    }

    toString() {
        return this._data;
    }
}

class Uint64 {
    constructor(data) {
        this._data = data;
    }

    toString() {
        return this._data;
    }
}
