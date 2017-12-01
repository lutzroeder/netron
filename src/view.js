
const onnx = protobuf.roots.onnx.onnx;

debugger;
// electron.remote.getCurrentWindow().webContents.openDevTools();

hostService.registerCallback(openBuffer);
var model = null;
var modelService = new ModelService(hostService);

document.documentElement.style.overflow = 'hidden';

window.addEventListener('load', function(e) {
    document.body.scroll = 'no';
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

function openBuffer(err, buffer) {
    if (err) {
        hostService.showError(err.toString());
        updateView('welcome');
    }
    else {
        setTimeout(function () {
            modelService.openBuffer(buffer, function(err, model) {
                if (err) {
                    hostService.showError('Decoding failure: ' + err);
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

    var graph = model.model.graph;
    
    var initializerMap = {};
    graph.initializer.forEach(function (tensor) {
        initializerMap[tensor.name] = tensor;
    });
    
    graph.node.forEach(function (node) {

        var operator = node.opType;

        var formatter = new NodeFormatter();

        formatter.addItem(operator, 'node-item-operator', null, function() { 
            showOperatorDocumentation(model, operator)
        });

        node.input.forEach(function (edge, index)
        {
            var name = model.getOperatorService().getInputName(operator, index);
    
            var initializer = initializerMap[edge];
            if (initializer) {
                var result = model.formatTensor(initializer);
                formatter.addItem(name, 'node-item-input', result['type'], function() { 
                    showInitializer(model, initializer);
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
                name: model.getOperatorService().getOutputName(operator, index) 
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
    
            var type = model.formatType(valueInfo.type);

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

function showOperatorDocumentation(model, operator) {
    var documentation = model.getOperatorService().getOperatorDocumentation(operator);
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

function showInitializer(model, initializer) {
    var initializer = model.formatTensor(initializer);
    if (initializer) {
        var view = { 'items': [ initializer ] };
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

ModelService.prototype.openBuffer = function(buffer, callback) {
    var model = new OnnxModel(hostService);
    var err = model.openBuffer(buffer);
    if (err) {
        callback(err, null);
    }
    else {
        this.activeModel = model;
        callback(null, model);
    }
}