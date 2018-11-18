/*jshint esversion: 6 */

var view = view || {};

var base = base || require('./base');
var zip = zip || require('./zip');
var gzip = gzip || require('./gzip');
var tar = tar || require('./tar');

var caffe = caffe || require('./caffe');
var caffe2 = caffe2 || require('./caffe2');
var cntk = cntk || require('./cntk');
var coreml = coreml || require('./coreml');
var keras = keras || require('./keras');
var mxnet = mxnet || require('./mxnet');
var onnx = onnx || require('./onnx');
var pytorch = pytorch || require('./pytorch');
var sklearn = sklearn || require('./sklearn');
var tf = tf || require('./tf');
var tflite = tflite || require('./tflite');

var d3 = d3 || require('d3');
var dagre = dagre || require('dagre');

view.View = class {

    constructor(host) {
        this._host = host;
        this._model = null;
        this._selection = [];
        this._sidebar = new Sidebar();
        this._host.initialize(this);
        this._showDetails = true;
        this._showNames = false;
        this._searchText = '';
        document.documentElement.style.overflow = 'hidden';
        document.body.scroll = 'no';        
        document.getElementById('model-properties-button').addEventListener('click', (e) => {
            this.showModelProperties();
        });
        document.getElementById('zoom-in-button').addEventListener('click', (e) => {
            this.zoomIn();
        });
        document.getElementById('zoom-out-button').addEventListener('click', (e) => {
            this.zoomOut();
        });
        document.getElementById('toolbar').addEventListener('mousewheel', (e) => {
            this.preventZoom(e);
        });
        document.getElementById('sidebar').addEventListener('mousewheel', (e) => {
            this.preventZoom(e);
        });
        document.addEventListener('keydown', (e) => {
            this.clearSelection();
        });
    }
    
    show(page) {

        if (!page) {
            page = (!this._model && !this._activeGraph) ? 'Welcome' : 'Graph';
        }

        this._host.screen(page);

        this._sidebar.close();

        var welcomeElement = document.getElementById('welcome');
        var openFileButton = document.getElementById('open-file-button');
        var spinnerElement = document.getElementById('spinner');
        var graphElement = document.getElementById('graph');
        var toolbarElement = document.getElementById('toolbar');
    
        if (page == 'Welcome') {
            document.body.style.cursor = 'default';
            welcomeElement.style.display = 'block';
            openFileButton.style.display = 'block';
            openFileButton.style.opacity = 1;
            spinnerElement.style.display = 'none';
            graphElement.style.display = 'none';
            graphElement.style.opacity = 0;
            toolbarElement.style.display = 'none';
        }

        if (page == 'Spinner') {
            document.body.style.cursor = 'wait';
            welcomeElement.style.display = 'block';
            spinnerElement.style.display = 'block';
            openFileButton.style.display = 'block';
            graphElement.style.display = 'block';
            graphElement.style.opacity = 0;
            toolbarElement.style.display = 'none';
        }

        if (page == 'Graph') {
            welcomeElement.style.display = 'none';
            openFileButton.style.display = 'none';
            openFileButton.style.opacity = 0;
            spinnerElement.style.display = 'none';
            graphElement.style.display = 'block';
            graphElement.style.opacity = 1;
            toolbarElement.style.display = 'block';
            document.body.style.cursor = 'default';
        }
    }

    cut() {
        document.execCommand('cut');
    }

    copy() {
        document.execCommand('copy');
    }

    paste() {
        document.execCommand('paste');
    }

    selectAll() {
        document.execCommand('selectall');
    }

    find() {
        if (this._activeGraph) {
            this.clearSelection();
            var graphElement = document.getElementById('graph');
            var view = new FindSidebar(graphElement, this._activeGraph);
            view.on('search-text-changed', (sender, text) => {
                this._searchText = text;
            });
            view.on('select', (sender, selection) => {
                this._sidebar.close();
                this.select(selection);
            });
            this._sidebar.open(view.content, 'Find');  
            view.focus(this._searchText);  
        }
    }

    toggleDetails() {
        this._showDetails = !this._showDetails;
        this.show('Spinner');
        this.updateGraph(this._model, this._activeGraph, (err) => {
            if (err) {
                this.error('Graph update failed.', err);
            }
        });
    }

    get showDetails() {
        return this._showDetails;
    }

    toggleNames() {
        this._showNames = !this._showNames;
        this.show('Spinner');
        this.updateGraph(this._model, this._activeGraph, (err) => {
            if (err) {
                this.error('Graph update failed.', err);
            }
        });
    }

    get showNames() {
        return this._showNames;
    }

    zoomIn() {
        if (this._zoom) {
            this._zoom.scaleBy(d3.select(document.getElementById('graph')), 1.2);
        }
    }

    zoomOut() {
        if (this._zoom) {
            this._zoom.scaleBy(d3.select(document.getElementById('graph')), 0.8);
        }
    }

    resetZoom() { 
        if (this._zoom) {
            this._zoom.scaleTo(d3.select(document.getElementById('graph')), 1);
        }
    }

    preventZoom(e) {
        if (e.shiftKey || e.ctrlKey) {
            e.preventDefault();
        }
    }

    select(selection) {
        this.clearSelection();
        if (selection && selection.length > 0) {
            var graphElement = document.getElementById('graph');
            var graphRect = graphElement.getBoundingClientRect();
            var x = 0;
            var y = 0;
            selection.forEach((element) => {
                element.classList.add('select');
                this._selection.push(element);
                var box = element.getBBox();
                var ex = box.x + (box.width / 2);
                var ey = box.y + (box.height / 2);
                var transform = element.transform.baseVal.consolidate();
                if (transform) {
                    ex = transform.matrix.e;
                    ey = transform.matrix.f;
                }
                x += ex;
                y += ey;
            });
            x = x / selection.length;
            y = y / selection.length;
            this._zoom.transform(d3.select(graphElement), d3.zoomIdentity.translate((graphRect.width / 2) - x, (graphRect.height / 2) - y));        
        }
    }

    clearSelection() {
        while (this._selection.length > 0) {
            var element = this._selection.pop();
            element.classList.remove('select');
        }
    }

    loadContext(context, callback) {
        var modelFactoryService = new view.ModelFactoryService(this._host);
        modelFactoryService.create(context, callback);
    }

    error(message, err) {
        this._sidebar.close();
        this._host.exception(err, false);
        this._host.error(message, err.toString());
        this.show('Welcome');
    }

    openContext(context, callback) {
        this._host.event('Model', 'Open', 'Size', context.buffer.length);
        this._sidebar.close();
        setTimeout(() => {
            this.loadContext(context, (err, model) => {
                if (err) {
                    callback(err);
                }
                else {
                    var format = model.format;
                    if (format) {
                        format = format + (model.producer ? ' (' + model.producer + ')' : '');
                        this._host.event('Model', 'Format', format);
                    }

                    setTimeout(() => {
                        var graph = model.graphs.length > 0 ? model.graphs[0] : null;
                        this.updateGraph(model, graph, (err, model) => {
                            callback(err, model);
                        });
                    }, 20);   
                }
            });    
        }, 2);
    }

    updateActiveGraph(name) {
        this._sidebar.close();
        if (this._model) {
            var model = this._model;
            var graph = model.graphs.filter(graph => name == graph.name).shift();
            if (graph) {
                this.show('Spinner');
                setTimeout(() => {
                    this.updateGraph(model, graph, (err, model) => {
                        if (err) {
                            this.error('Graph update failed.', err);
                        }
                    });
                }, 200);
            }
        }
    }

    updateGraph(model, graph, callback) {
        setTimeout(() => {
            if (graph && graph != this._activeGraph) {
                var nodes = graph.nodes;
                if (nodes.length > 1500) {
                    if (!this._host.confirm('Large model detected.', 'This graph contains a large number of nodes and might take a long time to render. Do you want to continue?')) {
                        this._host.event('Graph', 'Render', 'Skip', nodes.length);
                        this.show(null);
                        callback(null, null);
                        return;
                    }  
                }
            }

            this.renderGraph(graph, (err) => {
                if (err) {
                    this.renderGraph(this._activeGraph, (nestedError) => {
                        if (nestedError) {
                            this._model = null;
                            this._activeGraph = null;
                            this.show('Welcome');
                        }
                        else {
                            this.show('Graph');
                        }
                        callback(err, this._model);
                    });
                }
                else {
                    this._model = model;
                    this._activeGraph = graph;
                    this.show('Graph');
                    callback(null, this._model);
                }
            });
        }, 100);
    }

    renderGraph(graph, callback) {
        try {
            if (!graph) {
                callback(null);
            }
            else {
                var graphElement = document.getElementById('graph');
                while (graphElement.lastChild) {
                    graphElement.removeChild(graphElement.lastChild);
                }
    
                this._zoom = null;
    
                var groups = graph.groups;
    
                var graphOptions = {};
                graphOptions.nodesep = 25;
                graphOptions.ranksep = 25;
    
                var g = new dagre.graphlib.Graph({ compound: groups });
                g.setGraph(graphOptions);
                g.setDefaultEdgeLabel(() => { return {}; });
            
                var nodeId = 0;
                var edgeMap = {};
            
                var clusterMap = {};
                var clusterParentMap = {};
    
                var id = new Date().getTime();
                var nodes = graph.nodes;
        
                this._host.event('Graph', 'Render', 'Size', nodes.length);

                if (groups) {
                    nodes.forEach((node) => {
                        if (node.group) {
                            var path = node.group.split('/');
                            while (path.length > 0) {
                                var name = path.join('/');
                                path.pop();
                                clusterParentMap[name] = path.join('/');
                            }
                        }
                    });
                }
    
                nodes.forEach((node) => {
    
                    var formatter = new grapher.NodeElement();

                    if (node.function) {
                        formatter.addItem('+', null, [ 'node-item-function' ], null, () => { 
                            debugger;
                        });
                    }

                    function addOperator(view, formatter, node) {
                        if (node) {
                            var styles = [ 'node-item-operator' ];
                            var category = node.category;
                            if (category) {
                                styles.push('node-item-operator-' + category.toLowerCase());
                            }
                            var text = view.showNames && node.name ? node.name : (node.primitive ? node.primitive : node.operator);
                            var title = view.showNames && node.name ? node.operator : node.name;
                            formatter.addItem(text, null, styles, title, () => { 
                                view.showNodeProperties(node, null);
                            });
                        }
                    }
    
                    addOperator(this, formatter, node);
                    addOperator(this, formatter, node.inner);
            
                    var primitive = node.primitive;
            
                    var hiddenInputs = false;
                    var hiddenInitializers = false;
            
                    node.inputs.forEach((input) => {
                        // TODO what about mixed input & initializer
                        if (input.connections.length > 0) {
                            var initializers = input.connections.filter(connection => connection.initializer);
                            var inputId = null;
                            var inputClass = 'node-item-input';
                            if (initializers.length == 0) {
                                inputClass = 'node-item-input';
                                if (!input.visible) {
                                    hiddenInputs = true;
                                }
                            }
                            else {
                                if (initializers.length == 1) {
                                    inputId = 'initializer-' + initializers[0].initializer.name;
                                }
                                if (initializers.length == input.connections.length) {
                                    inputClass = 'node-item-constant';
                                    if (!input.visible) {
                                        hiddenInitializers = true;
                                    }
                                }
                                else {
                                    inputClass = 'node-item-constant';
                                    if (!input.visible) {
                                        hiddenInputs = true;
                                    }
                                }
                            }
            
                            if (this._showDetails) {
                                if (input.visible) {
                                    var types = input.connections.map(connection => connection.type || '').join('\n');
                                    formatter.addItem(input.name, inputId, [ inputClass ], types, () => {
                                        this.showNodeProperties(node, input);
                                    });    
                                }
                            }
            
                            input.connections.forEach((connection) => {
                                if (!connection.initializer) {
                                    var tuple = edgeMap[connection.id];
                                    if (!tuple) {
                                        tuple = { from: null, to: [] };
                                        edgeMap[connection.id] = tuple;
                                    }
                                    tuple.to.push({ 
                                        node: nodeId, 
                                        name: input.name
                                    });
                                }
                            });    
                        }
                    });
            
                    if (this._showDetails) {
                        if (hiddenInputs) {
                            formatter.addItem('...', null, [ 'node-item-input' ], '', () => {
                                this.showNodeProperties(node, null);
                            });    
                        }
                        if (hiddenInitializers) {
                            formatter.addItem('...', null, [ 'node-item-constant' ], '', () => {
                                this.showNodeProperties(node, null);
                            });    
                        }
                    }
            
                    node.outputs.forEach((output) => {
                        output.connections.forEach((connection) => {
                            var tuple = edgeMap[connection.id];
                            if (!tuple) {
                                tuple = { from: null, to: [] };
                                edgeMap[connection.id] = tuple;
                            }
                            tuple.from = { 
                                node: nodeId,
                                name: output.name,
                                type: connection.type
                            };
                        });
                    });
            
                    var dependencies = node.dependencies;
                    if (dependencies && dependencies.length > 0) {
                        formatter.setControlDependencies();
                    }
            
                    if (this._showDetails) {
                        var attributes = node.attributes; 
                        if (attributes && !primitive) {
                            formatter.setAttributeHandler(() => { 
                                this.showNodeProperties(node, null);
                            });
                            attributes.forEach((attribute) => {
                                if (attribute.visible) {
                                    var attributeValue = '';
                                    if (attribute.tensor) {
                                        attributeValue = '[...]';
                                    }
                                    else {
                                        attributeValue = view.View.formatAttributeValue(attribute.value, attribute.type);
                                    }
                                    if (attributeValue && attributeValue.length > 25) {
                                        attributeValue = attributeValue.substring(0, 25) + '...';
                                    }
                                    formatter.addAttribute(attribute.name, attributeValue, attribute.type);
                                }
                            });
                        }
                    }
    
                    var name = node.name;
                    if (name) {
                        g.setNode(nodeId, { label: formatter.format(graphElement), id: 'node-' + name });
                    }
                    else {
                        g.setNode(nodeId, { label: formatter.format(graphElement), id: 'node-' + id.toString() });
                        id++;
                    }
            
                    function createCluster(name) {
                        if (!clusterMap[name]) {
                            g.setNode(name, { rx: 5, ry: 5});
                            clusterMap[name] = true;
                            var parent = clusterParentMap[name];
                            if (parent) {
                                createCluster(parent);
                                g.setParent(name, parent);
                            }
                        }
                    }
    
                    if (groups) {
                        var groupName = node.group;
                        if (groupName && groupName.length > 0) {
                            if (!clusterParentMap.hasOwnProperty(groupName)) {
                                var lastIndex = groupName.lastIndexOf('/');
                                if (lastIndex != -1) {
                                    groupName = groupName.substring(0, lastIndex);
                                    if (!clusterParentMap.hasOwnProperty(groupName)) {
                                        groupName = null;
                                    }
                                }
                                else {
                                    groupName = null;
                                }
                            }
                            if (groupName) {
                                createCluster(groupName);
                                g.setParent(nodeId, groupName);
                            }
                        }
                    }
                
                    nodeId++;
                });
            
                graph.inputs.forEach((input) => {
                    input.connections.forEach((connection) => {
                        var tuple = edgeMap[connection.id];
                        if (!tuple) {
                            tuple = { from: null, to: [] };
                            edgeMap[connection.id] = tuple;
                        }
                        tuple.from = { 
                            node: nodeId,
                            type: connection.type
                        };    
                    });
                    var types = input.connections.map(connection => connection.type || '').join('\n');
    
                    var formatter = new grapher.NodeElement();
                    formatter.addItem(input.name, null, [ 'graph-item-input' ], types, () => {
                        this.showModelProperties();
                    });
                    g.setNode(nodeId++, { label: formatter.format(graphElement), class: 'graph-input' } ); 
                });
            
                graph.outputs.forEach((output) => {
                    output.connections.forEach((connection) => {
                        var tuple = edgeMap[connection.id];
                        if (!tuple) {
                            tuple = { from: null, to: [] };
                            edgeMap[connection.id] = tuple;
                        }
                        tuple.to.push({ node: nodeId });
                    });
                    var types = output.connections.map(connection => connection.type || '').join('\n');
            
                    var formatter = new grapher.NodeElement();
                    formatter.addItem(output.name, null, [ 'graph-item-output' ], types, () => {
                        this.showModelProperties();
                    });
                    g.setNode(nodeId++, { label: formatter.format(graphElement) } ); 
                });
            
                Object.keys(edgeMap).forEach((edge) => {
                    var tuple = edgeMap[edge];
                    if (tuple.from != null) {
                        tuple.to.forEach((to) => {
                            var text = '';
                            var type = tuple.from.type;
                            if (type && type.shape && type.shape.dimensions && type.shape.dimensions.length > 0) {
                                text = type.shape.dimensions.join('\u00D7');
                            }
                            else if (tuple.from.name && to.name) {
                                text = tuple.from.name + ' \u21E8 ' + to.name;
                            }
                            else if (tuple.from.name) {
                                text = tuple.from.name;
                            }
                            else {
                                text = to.name;
                            }
            
                            if (this._showNames) {
                                text = edge.split('\n').shift(); // custom connection id
                            }
                            if (!this._showDetails) {
                                text = '';
                            }
    
                            if (to.dependency) { 
                                g.setEdge(tuple.from.node, to.node, { label: text, id: 'edge-' + edge, arrowhead: 'vee', class: 'edge-path-control' } );
                            }
                            else {
                                g.setEdge(tuple.from.node, to.node, { label: text, id: 'edge-' + edge, arrowhead: 'vee' } );
                            }
                        });
                    }
                });
            
                // Workaround for Safari background drag/zoom issue:
                // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
                var backgroundElement = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                backgroundElement.setAttribute('id', 'background');
                backgroundElement.setAttribute('width', '100%');
                backgroundElement.setAttribute('height', '100%');
                backgroundElement.setAttribute('fill', 'none');
                backgroundElement.setAttribute('pointer-events', 'all');
                graphElement.appendChild(backgroundElement);
            
                var originElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                originElement.setAttribute('id', 'origin');
                graphElement.appendChild(originElement);
            
                // Set up zoom support
                var svg = d3.select(graphElement);
                this._zoom = d3.zoom();
                this._zoom(svg);
                this._zoom.scaleExtent([0.1, 2]);
                this._zoom.on('zoom', (e) => {
                    originElement.setAttribute('transform', d3.event.transform.toString());
                });
                this._zoom.transform(svg, d3.zoomIdentity);

                setTimeout(() => {
                    try {
                        var graphRenderer = new grapher.Renderer(originElement);
                        graphRenderer.render(g);
            
                        var svgSize = graphElement.getBoundingClientRect();
            
                        var inputElements = graphElement.getElementsByClassName('graph-input');
                        if (inputElements && inputElements.length > 0) {
                            // Center view based on input elements
                            var xs = [];
                            var ys = [];
                            for (var i = 0; i < inputElements.length; i++) {
                                var inputTransform = inputElements[i].transform.baseVal.consolidate().matrix;
                                xs.push(inputTransform.e);
                                ys.push(inputTransform.f);
                            }
                            var x = xs[0];
                            var y = ys[0];
                            if (ys.every(y => y == ys[0])) {
                                x = xs.reduce((a,b) => { return a + b; }) / xs.length;
                            }
                            this._zoom.transform(svg, d3.zoomIdentity.translate((svgSize.width / 2) - x, (svgSize.height / 4) - y));
                        }
                        else {
                            this._zoom.transform(svg, d3.zoomIdentity.translate((svgSize.width - g.graph().width) / 2, (svgSize.height - g.graph().height) / 2));
                        }

                        callback(null);
                    }
                    catch (err) {
                        callback(err);
                    }
                }, 20);
            }
        }
        catch (err) {
            callback(err);
        }
    }

    applyStyleSheet(element, name) {
        var rules = [];
        for (var i = 0; i < document.styleSheets.length; i++) {
            var styleSheet = document.styleSheets[i];
            if (styleSheet && styleSheet.href && styleSheet.href.endsWith('/' + name)) {
                rules = styleSheet.cssRules;
                break;
            }
        }
        var nodes = element.getElementsByTagName('*');
        for (var j = 0; j < nodes.length; j++) {
            var node = nodes[j];
            for (var k = 0; k < rules.length; k++) {
                var rule = rules[k];                
                if (node.matches(rule.selectorText)) {
                    for (var l = 0; l < rule.style.length; l++) {
                        var item = rule.style.item(l);
                        node.style[item] = rule.style[item];
                    }
                }
            }
        }
    }

    export(file) {
        var extension = '';
        var lastIndex = file.lastIndexOf('.');
        if (lastIndex != -1) {
            extension = file.substring(lastIndex + 1);
        }
        if (this._activeGraph && (extension == 'png' || extension == 'svg')) {
            var graphElement = document.getElementById('graph');
            var exportElement = graphElement.cloneNode(true);
            this.applyStyleSheet(exportElement, 'view-grapher.css');
            exportElement.setAttribute('id', 'export');
            exportElement.removeAttribute('width');
            exportElement.removeAttribute('height');
            exportElement.style.removeProperty('opacity');
            exportElement.style.removeProperty('display');
            var backgroundElement = exportElement.querySelector('#background');
            var originElement = exportElement.querySelector('#origin');
            originElement.setAttribute('transform', 'translate(0,0) scale(1)');
            backgroundElement.removeAttribute('width');
            backgroundElement.removeAttribute('height');
    
            var parentElement = graphElement.parentElement;
            parentElement.insertBefore(exportElement, graphElement);
            var size = exportElement.getBBox();
            parentElement.removeChild(exportElement);
            parentElement.removeChild(graphElement);
            parentElement.appendChild(graphElement);

            var delta = (Math.min(size.width, size.height) / 2.0) * 0.1;
            var width = Math.ceil(delta + size.width + delta);
            var height = Math.ceil(delta + size.height + delta);
            originElement.setAttribute('transform', 'translate(' + delta.toString() + ', ' + delta.toString() + ') scale(1)');
            exportElement.setAttribute('width', width);
            exportElement.setAttribute('height', height);
            backgroundElement.setAttribute('width', width);
            backgroundElement.setAttribute('height', height);
            backgroundElement.setAttribute('fill', '#fff');
    
            var data = new XMLSerializer().serializeToString(exportElement);
    
            if (extension == 'svg') {
                this._host.export(file, new Blob([ data ], { type: 'image/svg' }));
            }
    
            if (extension == 'png') {
                var imageElement = new Image();
                imageElement.onload = () => {
                    var max = Math.max(width, height);
                    var scale = ((max * 2.0) > 24000) ? (24000.0 / max) : 2.0;
                    var canvas = document.createElement('canvas');
                    canvas.width = Math.ceil(width * scale);
                    canvas.height = Math.ceil(height * scale);    
                    var context = canvas.getContext('2d');
                    context.scale(scale, scale);
                    context.drawImage(imageElement, 0, 0);
                    document.body.removeChild(imageElement);
                    canvas.toBlob((blob) => {
                        this._host.export(file, blob);
                    }, 'image/png');
                };
                imageElement.src = 'data:image/svg+xml;base64,' + window.btoa(unescape(encodeURIComponent(data)));
                document.body.insertBefore(imageElement, document.body.firstChild);
            }
        }
    }

    showModelProperties() {
        if (this._model) {
            var view = new ModelSidebar(this._model, this._host);
            view.on('update-active-graph', (sender, name) => {
                this.updateActiveGraph(name);
            });
            this._sidebar.open(view.elements, 'Model Properties');
        }
    }
    
    showNodeProperties(node, input) {
        if (node) {
            var view = new NodeSidebar(node, this._host);
            view.on('show-documentation', (sender, e) => {
                this.showOperatorDocumentation(node);
            });
            view.on('export-tensor', (sender, tensor) => {
                this._host.require('./numpy', (err, numpy) => {
                    if (!err) {
                        var defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
                        this._host.save('NumPy Array', 'npy', defaultPath, (file) => {
                            try {
                                var array = new numpy.Array(tensor.value, tensor.type.dataType, tensor.type.shape.dimensions);
                                this._host.export(file, new Blob([ array.toBuffer() ], { type: 'application/octet-stream' }));
                            }
                            catch (error) {
                                this.error('Error saving NumPy tensor.', error);
                            }
                        });
                    }
                });
            });
            if (input) {
                view.toggleInput(input.name);
            }
            this._sidebar.open(view.elements, 'Node Properties');
        }
    }

    showOperatorDocumentation(node) {
        var documentation = node.documentation;
        if (documentation) {
            var view = new OperatorDocumentationSidebar(documentation);
            view.on('navigate', (sender, e) => {
                this._host.openURL(e.link);
            });
            this._sidebar.open(view.elements, 'Documentation');
        }
    }

    static formatAttributeValue(value, type) {
        if (typeof value === 'function') {
            return value();
        }
        if (typeof value === 'string' && type && type != 'string') {
            return value;
        }
        if (value && value.__isLong__) {
            return value.toString();
        }
        if (value instanceof base.Int64 || value instanceof base.Uint64) {
            return value.toString();
        }
        if (Number.isNaN(value)) {
            return 'NaN';
        }
        if (type == 'shape') {
            return value.toString();
        }
        if (type == 'shape[]') {
            return value.map((item) => item.toString()).join(', ');
        }
        if (Array.isArray(value)) {
            return value.map((item) => {
                if (item && item.__isLong__) {
                    return item.toString();
                }
                if (Number.isNaN(item)) {
                    return 'NaN';
                }
                return JSON.stringify(item);
            }).join(', ');
        }
        return JSON.stringify(value);
    }
};

class ArchiveContext {

    constructor(entries, rootFolder, identifier, buffer) {
        this._entries = {};
        if (entries) {
            entries.forEach((entry) => {
                if (entry.name.startsWith(rootFolder)) {
                    var name = entry.name.substring(rootFolder.length);
                    if (identifier.length > 0 && identifier.indexOf('/') < 0) {
                        this._entries[name] = entry;
                    }
                }
            });
        }
        this._identifier = identifier;
        this._buffer = buffer;
    }

    request(file, encoding, callback) {
        var entry = this._entries[file];
        if (!entry) {
            callback(new Error('File not found.'), null);
            return;
        }
        var data = entry.data;
        if (type != null) {
            data = new TextDecoder(encoding).decode(data);
        }
        callback(null, data);
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }

    get text() {
        if (!this._text) {
            var decoder = new TextDecoder('utf-8');
            this._text = decoder.decode(this._buffer);
        }
        return this._text;
    }

    get tags() {
        if (!this._tags) {
            this._tags = {};
            try {
                var reader = protobuf.TextReader.create(this.text);
                reader.start(false);
                while (!reader.end(false)) {
                    var tag = reader.tag();
                    this._tags[tag] = true;
                    reader.skip();
                }
            }
            catch (error) {
            }
        }
        return this._tags;
    }
}

class ArchiveError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading archive.';
    }
}

class ModelError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading model.'; 
    }
}

view.ModelFactoryService = class {

    constructor(host) {
        this._host = host;
        this._factories = [
            new onnx.ModelFactory(),
            new mxnet.ModelFactory(),
            new keras.ModelFactory(),
            new coreml.ModelFactory(),
            new caffe.ModelFactory(),
            new caffe2.ModelFactory(), 
            new pytorch.ModelFactory(),
            new tflite.ModelFactory(),
            new tf.ModelFactory(),
            new sklearn.ModelFactory(),
            new cntk.ModelFactory()
        ];
    }

    some(context) {
        return this._factories.some((factory) => factory.match(context, this._host));
    }

    filter(context) {
        return this._factories.filter((factory) => factory.match(context, this._host));        
    }

    create(context, callback) {
        try {
            var extension;
            var archive;
            var entry;
    
            var identifier = context.identifier;
            var buffer = context.buffer;

            extension = identifier.split('.').pop().toLowerCase();
            if (extension == 'gz' || extension == 'tgz') {
                archive = new gzip.Archive(buffer);
                if (archive.entries.length == 1) {
                    entry = archive.entries[0];
                    if (entry.name) {
                        identifier = entry.name;
                    }
                    else {
                        identifier = identifier.substring(0, identifier.lastIndexOf('.'));
                        if (extension == 'tgz') {
                            identifier += '.tar';
                        }
                    }
                    buffer = entry.data;
                    archive = null;
                }
            }

            switch (identifier.split('.').pop().toLowerCase()) {
                case 'tar':
                    archive = new tar.Archive(buffer);
                    break;
                case 'zip':
                    archive = new zip.Archive(buffer);
                    break;
            }

            if (archive) {
                var folders = {};
                archive.entries.forEach((entry) => {
                    if (entry.name.indexOf('/') != -1) {
                        folders[entry.name.split('/').shift() + '/'] = true;
                    }
                    else {
                        folders['/'] = true;
                    }
                });
                var rootFolder = Object.keys(folders).length == 1 ? Object.keys(folders)[0] : '';
                rootFolder = rootFolder == '/' ? '' : rootFolder;
                var entries = archive.entries.filter((entry) => {
                    if (entry.name.startsWith(rootFolder)) {
                        var identifier = entry.name.substring(rootFolder.length);
                        if (identifier.length > 0 && identifier.indexOf('/') < 0) {
                            return this.some(new ArchiveContext(null, rootFolder, identifier, entry.data), this._host);
                        }
                    }
                    return false;
                });
                if (entries.length == 0) {
                    callback(new ArchiveError('Root does not contain model file.'), null);
                    return;
                }
                else if (entries.length > 1) {
                    callback(new ArchiveError('Root contains multiple model files.'), null);
                    return;
                }
                else {
                    entry = entries[0];
                    context = new ArchiveContext(entries, rootFolder, entry.name, entry.data);
                }
            }
        }
        catch (err) {
            callback(new ArchiveError(err.message), null);
            return;
        }

        var errorList = [];
        var factoryList = this.filter(context, this._host);
        var factoryCount = factoryList.length;
        var next = () => {
            if (factoryList.length > 0) {
                var modelFactory = factoryList.shift();
                modelFactory.open(context, this._host, (err, model) => {
                    if (err) {
                        errorList.push(err);
                    }
                    if (model || factoryList.length == 0) {
                        if (!model && factoryCount > 1 && errorList.length > 1) {
                            callback(new ModelError(errorList.map((err) => err.message).join('\n')), null);
                        }
                        else {
                            callback(err, model);
                        }
                    }
                    else {
                        next();
                    }
                });
            }
            else {
                var extension = context.identifier.split('.').pop().toLowerCase();
                switch (extension) {
                    case 'json':
                    case 'pb':
                    case 'pbtxt':
                    case 'prototxt':
                    case 'pth':
                    case 'h5':
                    case 'model':
                        callback(new ModelError("Unsupported file content for extension '." + extension + "' in '" + context.identifier + "'."), null);
                        break;
                    default:
                        callback(new ModelError("Unsupported file extension '." + extension + "'."), null);
                        break;
                }
            }
        };
        next();
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.View = view.View;
    module.exports.ModelFactoryService = view.ModelFactoryService;
}
