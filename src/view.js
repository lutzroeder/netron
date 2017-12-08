
debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

hostService.initialize(openBuffer);
var modelService = new ModelService(hostService);

document.documentElement.style.overflow = 'hidden';
document.body.scroll = 'no';

var navigationButton = document.getElementById('navigation-button');
if (navigationButton) {
    navigationButton.addEventListener('click', function(e) {
        showModelSummary(modelService.getActiveModel());
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
    var model = modelService.getActiveModel();
    if (model) {
        model.updateActiveGraph(name);
        updateView('spinner');
        setTimeout(function () {
            updateGraph(model);
        }, 250);
    }
}

function updateGraph(model) {

    var graph = model.getActiveGraph();
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
    model.getGraphInitializers(graph).forEach(function (initializer) {
        var id = initializer.id;
        initializerMap[id] = initializer;
    });

    model.getNodes(graph).forEach(function (node) {
        var operator = model.getNodeOperator(node);
        var formatter = new NodeFormatter();
        var style = (operator != 'Constant' && operator != 'Const') ? 'node-item-operator' : 'node-item-constant';
        formatter.addItem(operator, style, null, function() { 
            showNodeOperatorDocumentation(model, graph, node);
        });

        model.getNodeInputs(graph, node).forEach(function (input)
        {
            var inputId = input.id;
            var initializer = initializerMap[inputId];
            if (initializer) {
                formatter.addItem(input.name, 'node-item-constant', initializer.type, function() { 
                    showTensor(model, initializer);
                });
            }
            else {
                // TODO is there a way to infer the type of the6 input?
                formatter.addItem(input.name, null, input.type, null);
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

        model.getNodeOutputs(graph, node).forEach(function (output)
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

        var properties = model.formatNodeProperties(node);
        if (properties) {
            formatter.setPropertyHandler(function() {
                showNodeProperties(model, node);
            });
            properties.forEach(function (property) {
                formatter.addProperty(property.name, property.value_short());
            });
        }

        var attributes = model.formatNodeAttributes(node);
        if (attributes) {
            formatter.setAttributeHandler(function() { 
                showNodeAttributes(model, node);
            });
            attributes.forEach(function (attribute) {
                formatter.addAttribute(attribute.name, attribute.value_short(), attribute.type);
            });
        }

        g.setNode(nodeId++, { label: formatter.format(svg).node(), labelType: 'svg', padding: 0 });
    });

    model.getGraphInputs(graph).forEach(function (input) {
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

    model.getGraphOutputs(graph).forEach(function (output) {
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

    Object.keys(edgeMap).forEach(function (edge) {
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
            zoom.translate([ (svgSize.width - g.graph().width) / 2, 40 ]).event(svg);
        }    
    
        updateView('graph');
    }, 20);
}

function showNodeOperatorDocumentation(model, graph, node) {
    var documentation = model.getNodeOperatorDocumentation(graph, node);
    if (documentation) {
        sidebar.open(documentation, 'Documentation');
    }
}

function showModelSummary(model) {
    var view = model.formatModelSummary();
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
            e.preventDefault();
            self.close();
        }
    };
    this.resizeSidebarHandler = function (e) {
        var contentElement = document.getElementById('sidebar-content');
        if (contentElement) {
            contentElement.style.height = window.innerHeight - 60;
        }
    };
 }

Sidebar.prototype.open = function(content, title, width, margin) {
    var sidebarElement = document.getElementById('sidebar');
    var titleElement = document.getElementById('sidebar-title');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement && titleElement) {
        window.addEventListener('resize', this.resizeSidebarHandler);
        document.addEventListener('keydown', this.closeSidebarKeyDownHandler);
        titleElement.innerHTML = title ? title.toUpperCase() : '';
        closeButtonElement.addEventListener('click', this.closeSidebarHandler);
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
};

Sidebar.prototype.close = function() {
    var sidebarElement = document.getElementById('sidebar');
    var contentElement = document.getElementById('sidebar-content');
    var closeButtonElement = document.getElementById('sidebar-closebutton');
    if (sidebarElement && contentElement && closeButtonElement) {
        document.removeEventListener('keydown', this.closeSidebarKeyDownHandler);
        sidebarElement.removeEventListener('resize', this.resizeSidebarHandler);
        closeButtonElement.removeEventListener('click', this.closeSidebarHandler);
        closeButtonElement.style.color = '#f8f8f8';
        sidebarElement.style.width = '0';
    }
};

var sidebar = new Sidebar();

function ModelService(hostService) {
    this.hostService = hostService;
}

ModelService.prototype.openBuffer = function(buffer, identifier, callback) {
    var model = null;
    var err = null;

    if (identifier != null && identifier.split('.').pop() == 'tflite')
    {
        model = new TensorFlowLiteModel(hostService); 
        err = model.openBuffer(buffer, identifier);
    }
    else if (identifier != null && identifier == 'saved_model.pb') {
        model = new TensorFlowModel(hostService);
        err = model.openBuffer(buffer, identifier);
    }
    else {
        model = new OnnxModel(hostService);
        err = model.openBuffer(buffer, identifier);
    }

    if (err) {
        callback(err, null);
    }
    else {
        this.activeModel = model;
        callback(null, model);
    }
};

ModelService.prototype.getActiveModel = function() {
    return this.activeModel;
};

function Int64(data) {
    this.data = data;
}

Int64.prototype.toString = function() {
    return this.data;
};

function Uint64(data) {
    this.data = data;
}

Uint64.prototype.toString = function() {
    return this.data;
};