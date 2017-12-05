
debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

hostService.initialize(openBuffer);
var modelService = new ModelService(hostService);

document.documentElement.style.overflow = 'hidden';
document.body.scroll = 'no';
updateSize();

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
    var openFileButton = document.getElementById('open-file-button');
    var spinnerElement = document.getElementById('spinner');
    var propertiesElement = document.getElementById('properties-button');

    if (page == 'welcome') {
        document.body.style.cursor = 'default';
        welcomeElement.style.display = 'block';
        welcomeElement.offsetHeight;
        openFileButton.style.display = 'block';
        spinnerElement.style.display = 'none';
        propertiesElement.style.display = 'none';
    }

    if (page == 'clock') {
        document.body.style.cursor = 'wait';
        welcomeElement.style.display = 'block';
        openFileButton.style.display = 'none';
        spinnerElement.style.display = 'block';
        spinnerElement.offsetHeight;
        propertiesElement.style.display = 'none';
    }

    if (page == 'graph') {
        welcomeElement.style.display = 'none';
        openFileButton.style.display = 'none';
        spinnerElement.style.display = 'none';
        propertiesElement.style.display = 'block';
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
                    renderModel(model);
                }, 20);   
            });    
        }, 20);
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

    var graph = model.getGraph(0);
    if (graph) {
        var initializerMap = {};
        model.getGraphInitializers(graph).forEach(function (initializer) {
            var id = initializer['id'];
            initializerMap[id] = initializer;
        });

        model.getNodes(graph).forEach(function (node) {
            var operator = model.getNodeOperator(node);
            var formatter = new NodeFormatter();
            var style = (operator != 'Constant' && operator != 'Const') ? 'node-item-operator' : 'node-item-constant';
            formatter.addItem(operator, style, null, function() { 
                showNodeOperatorDocumentation(model, graph, node)
            });
    
            model.getNodeInputs(graph, node).forEach(function (input)
            {
                var inputId = input['id'];
                var initializer = initializerMap[inputId];
                if (initializer) {
                    formatter.addItem(input['name'], 'node-item-constant', initializer['type'], function() { 
                        showTensor(model, initializerMap[input['id']]);
                    });
                }
                else {
                    // TODO is there a way to infer the type of the6 input?
                    formatter.addItem(input['name'], null, input['type'], null);
                    var tuple = edgeMap[inputId];
                    if (!tuple) {
                        tuple = { from: null, to: [] };
                        edgeMap[inputId] = tuple;
                    }
                    tuple.to.push({ 
                        node: nodeId, 
                        name: input['name']
                    });
                }
            });
    
            model.getNodeOutputs(graph, node).forEach(function (output)
            {
                var outputId = output['id'];
                var tuple = edgeMap[outputId];
                if (!tuple) {
                    tuple = { from: null, to: [] };
                    edgeMap[outputId] = tuple;
                }
                tuple.from = { 
                    node: nodeId,
                    name: output['name']
                };
            });
    
            var properties = model.formatNodeProperties(node);
            if (properties) {
                formatter.setPropertyHandler(function() {
                    showNodeProperties(model, node);
                });
                properties.forEach(function (property) {
                    formatter.addProperty(property['name'], property['value_short']());
                });
            }
    
            var attributes = model.formatNodeAttributes(node);
            if (attributes) {
                formatter.setAttributeHandler(function() { 
                    showNodeAttributes(model, node);
                });
                attributes.forEach(function (attribute) {
                    formatter.addAttribute(attribute['name'], attribute['value_short'](), attribute['type']);
                });
            }
    
            g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 });
        });
    
        model.getGraphInputs(graph).forEach(function (input) {
            var inputId = input['id'];
            var inputName = input['name'];
            if (!initializerMap[inputId]) {
                var tuple = edgeMap[inputId];
                if (!tuple) {
                    tuple = { from: null, to: [] };
                    edgeMap[inputId] = tuple;
                }
                tuple.from = { 
                    node: nodeId,
                    // name: valueInfo.name
                };
        
                var formatter = new NodeFormatter();
                formatter.addItem(input['name'], null, input['type'], null);
                g.setNode(nodeId++, { label: formatter.format(svg).node(), class: 'graph-input', labelType: 'svg', padding: 0 } ); 
            }
        });
    
        model.getGraphOutputs(graph).forEach(function (output) {
            var outputId = output['id'];
            var outputName = output['name'];
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
            formatter.addItem(output['name'], null, output['type'], null);
            g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 } ); 
        });
    
        Object.keys(edgeMap).forEach(function (edge) {
            var tuple = edgeMap[edge];
            if (tuple.from != null) {
                tuple.to.forEach(function (to) {
                    var text = ''
                    if (tuple.from['name'] && to['name']) {
                        text = tuple.from['name'] + ' => ' + to['name'];
                    }
                    else if (tuple.from['name']) {
                        text = tuple.from['name'];
                    }
                    else {
                        text = to['name'];
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
    
        setTimeout(function () {
    
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
        }, 20);
    }
    else {
        updateView('graph');
    }
}

function showNodeOperatorDocumentation(model, graph, node) {
    var documentation = model.getNodeOperatorDocumentation(graph, node);
    if (documentation) {
        sidebar.open(documentation, 'Documentation');
    }
}

function showModelProperties(model) {
    var view = model.formatModelProperties();
    if (view) {
        var template = Handlebars.compile(modelPropertiesTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, 'Model Properties');
    }
}

function showTensor(model, tensor) {
    if (tensor) {
        var view = { 'items': [ tensor ] };
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template(view);
        sidebar.open(data, 'Initializer');
    }
}

function showNodeProperties(model, node) {
    var properties = model.formatNodeProperties(node);
    if (properties) {
        var template = Handlebars.compile(itemsTemplate, 'utf-8');
        var data = template({ 'items': properties });
        sidebar.open(data, 'Node Properties');
    }
}

function showNodeAttributes(model, node) {
    var attributes = model.formatNodeAttributes(node);
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

function ModelService(hostService) {
    this.hostService = hostService;
}

ModelService.prototype.openBuffer = function(buffer, identifier, callback) {
    if (identifier != null && identifier.split('.').pop() == 'tflite')
    {
        var model = new TensorFlowLiteModel(hostService); 
        var err = model.openBuffer(buffer, identifier);
        if (err) {
            callback(err, null);
        }
        else {
            this.activeModel = model;
            callback(null, model);
        }
    }
    else if (identifier != null && identifier == 'saved_model.pb') {
        var model = new TensorFlowModel(hostService);
        var err = model.openBuffer(buffer, identifier);
        if (err) {
            callback(err, null);
        }
        else {
            this.activeModel = model;
            callback(null, model);
        }
    }
    else {
        var model = new OnnxModel(hostService);
        var err = model.openBuffer(buffer, identifier);
        if (err) {
            callback(err, null);
        }
        else {
            this.activeModel = model;
            callback(null, model);
        }
    }
}

function Int64(data) {
    this.data = data;
}

Int64.prototype.toString = function() {
    return this.data;
}

function Uint64(data) {
    this.data = data;
}

Uint64.prototype.toString = function() {
    return this.data;
}