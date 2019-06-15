/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var view = view || {};

var zip = zip || require('./zip');
var gzip = gzip || require('./gzip');
var tar = tar || require('./tar');
var protobuf = protobuf || require('protobufjs');
var prototxt = prototxt || require('protobufjs/ext/prototxt');

var d3 = d3 || require('d3');
var dagre = dagre || require('dagre');

var sidebar = sidebar || require('./view-sidebar');
var grapher = grapher || require('./view-grapher');

view.View = class {

    constructor(host) {
        this._host = host;
        this._model = null;
        this._selection = [];
        this._sidebar = new sidebar.Sidebar(this._host);
        this._host.initialize(this);
        this._showAttributes = false;
        this._showInitializers = true;
        this._showNames = false;
        this._searchText = '';
        this._modelFactoryService = new view.ModelFactoryService(this._host);
        this._host.document.documentElement.style.overflow = 'hidden';
        this._host.document.body.scroll = 'no';
        this._host.document.getElementById('model-properties-button').addEventListener('click', () => {
            this.showModelProperties();
        });
        this._host.document.getElementById('zoom-in-button').addEventListener('click', () => {
            this.zoomIn();
        });
        this._host.document.getElementById('zoom-out-button').addEventListener('click', () => {
            this.zoomOut();
        });
        this._host.document.getElementById('toolbar').addEventListener('mousewheel', (e) => {
            this.preventZoom(e);
        });
        this._host.document.getElementById('sidebar').addEventListener('mousewheel', (e) => {
            this.preventZoom(e);
        });
        this._host.document.addEventListener('keydown', () => {
            this.clearSelection();
        });
        if (this._host.environment('zoom') == 'scroll') {
            this._host.document.getElementById('graph-container').addEventListener('mousewheel', (e) => {
                this._mouseWheelHandler(e);
            });
            this._host.document.getElementById('graph').addEventListener('mousewheel', (e) => {
                this._mouseWheelHandler(e);
            });
        }
    }
    
    show(page) {

        if (!page) {
            page = (!this._model && !this._activeGraph) ? 'Welcome' : 'Graph';
        }

        this._host.screen(page);

        this._sidebar.close();

        var welcomeElement = this._host.document.getElementById('welcome');
        var openFileButton = this._host.document.getElementById('open-file-button');
        var spinnerElement = this._host.document.getElementById('spinner');
        var graphElement = this._host.document.getElementById('graph');
        var toolbarElement = this._host.document.getElementById('toolbar');
    
        if (page == 'Welcome') {
            this._host.document.body.style.cursor = 'default';
            welcomeElement.style.display = 'block';
            openFileButton.style.display = 'block';
            openFileButton.style.opacity = 1;
            spinnerElement.style.display = 'none';
            graphElement.style.display = 'none';
            graphElement.style.opacity = 0;
            toolbarElement.style.display = 'none';
        }

        if (page == 'Spinner') {
            this._host.document.body.style.cursor = 'wait';
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
            this._host.document.body.style.cursor = 'default';
        }
    }

    cut() {
        this._host.document.execCommand('cut');
    }

    copy() {
        this._host.document.execCommand('copy');
    }

    paste() {
        this._host.document.execCommand('paste');
    }

    selectAll() {
        this._host.document.execCommand('selectall');
    }

    find() {
        if (this._activeGraph) {
            this.clearSelection();
            var graphElement = document.getElementById('graph');
            var view = new sidebar.FindSidebar(graphElement, this._activeGraph);
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

    toggleAttributes() {
        this._showAttributes = !this._showAttributes;
        this._reload();
    }

    get showAttributes() {
        return this._showAttributes;
    }

    toggleInitializers() {
        this._showInitializers = !this._showInitializers;
        this._reload();
    }

    get showInitializers() {
        return this._showInitializers;
    }

    toggleNames() {
        this._showNames = !this._showNames;
        this._reload();
    }

    get showNames() {
        return this._showNames;
    }

    _reload() {
        this.show('Spinner');
        if (this._model && this._activeGraph) {
            this._updateGraph(this._model, this._activeGraph).catch((error) => {
                if (error) {
                    this.error('Graph update failed.', error);
                }
            });
        }
    }

    _timeout(time) {
        return new Promise((resolve) => {
            setTimeout(() => { resolve(); }, time);
        });
    }

    zoomIn() {
        switch (this._host.environment('zoom')) {
            case 'scroll':
                if (this._zoom) {
                    this._zoom = this._zoom * 1.05;
                    if (this._zoom > 2) {
                        this._zoom = 2;
                    }
                    this.applyZoom();
                }
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleBy(d3.select(this._host.document.getElementById('graph')), 1.2);
                }
                break;
        }
    }

    zoomOut() {
        switch (this._host.environment('zoom')) {
            case 'scroll':
                if (this._zoom) {
                    this._zoom = this._zoom * 0.95;
                    if (this._zoom < 0.1) {
                        this._zoom = 0.1;
                    }
                    this.applyZoom();
                }
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleBy(d3.select(this._host.document.getElementById('graph')), 0.8);
                }
                break;
        }
    }

    resetZoom() { 
        switch (this._host.environment('zoom')) {
            case 'scroll':
                if (this._zoom) {
                    this._zoom = 1;
                    this.applyZoom();
                }
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleTo(d3.select(this._host.document.getElementById('graph')), 1);
                }
                break;
        }
    }

    preventZoom(e) {
        if (e.shiftKey || e.ctrlKey) {
            e.preventDefault();
        }
    }

    applyZoom() {
        var svgElement = this._host.document.getElementById('graph');
        svgElement.setAttribute('style', 'zoom: ' + this._zoom + ';');
        // svgElement.setAttribute('style', 'transform: scale(' + this._zoom + ',' + this._zoom + ')');
        // svgElement.setAttribute('width', this._width * this._zoom);
        // svgElement.setAttribute('height', this._height * this._zoom);
    }

    _mouseWheelHandler(e) {
        if (e.shiftKey || e.ctrlKey) {
            if (this._zoom) {
                // var oldWidth = this._width * this._zoom;
                // var oldHeight = this._height * this._zoom;
                this._zoom = this._zoom + (e.wheelDelta * 1.0 / 6000.0);
                if (this._zoom < 0.1) { this._zoom = 0.1; }
                if (this._zoom > 2) { this._zoom = 2; }
                this.applyZoom();

                /* var svgElement = document.getElementById('graph');
                va r newWidth = this._width * this._zoom;
                var newHeight = this._height * this._zoom;
                svgElement.setAttribute('width', newWidth);
                svgElement.setAttribute('height', newHeight); */

                // var dx = (oldWidth - newWidth) / 2;
                // var dy = (oldHeight - newHeight) / 2;
                // window.scrollBy(dx, dy);

                e.preventDefault();
            }
        }
    }
    
    select(selection) {
        this.clearSelection();
        if (selection && selection.length > 0) {
            var graphElement = this._host.document.getElementById('graph');
            var graphRect = graphElement.getBoundingClientRect();
            var x = 0;
            var y = 0;
            for (var element of selection) {
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
            }
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

    error(message, err) {
        this._sidebar.close();
        this._host.exception(err, false);
        this._host.error(message, err.toString());
        this.show('Welcome');
    }

    accept(file) {
        return this._modelFactoryService.accept(file);
    }

    open(context) {
        this._host.event('Model', 'Open', 'Size', context.buffer.length);
        this._sidebar.close();
        return this._timeout(2).then(() => {
            return this._modelFactoryService.open(context).then((model) => {
                var format = model.format;
                if (format) {
                    format = format + (model.producer ? ' (' + model.producer + ')' : '');
                    this._host.event('Model', 'Format', format);
                }
                return this._timeout(20).then(() => {
                    var graph = model.graphs.length > 0 ? model.graphs[0] : null;
                    return this._updateGraph(model, graph);
                });
            });
        });
    }

    _updateActiveGraph(name) {
        this._sidebar.close();
        if (this._model) {
            var model = this._model;
            var graph = model.graphs.filter(graph => name == graph.name).shift();
            if (graph) {
                this.show('Spinner');
                this._timeout(200).then(() => {
                    return this._updateGraph(model, graph).catch((error) => {
                        if (error) {
                            this.error('Graph update failed.', error);
                        }
                    });
                });
            }
        }
    }

    _updateGraph(model, graph) {
        return this._timeout(100).then(() => {
            if (graph && graph != this._activeGraph) {
                var nodes = graph.nodes;
                if (nodes.length > 1400) {
                    if (!this._host.confirm('Large model detected.', 'This graph contains a large number of nodes and might take a long time to render. Do you want to continue?')) {
                        this._host.event('Graph', 'Render', 'Skip', nodes.length);
                        this.show(null);
                        return null;
                    }  
                }
            }
            return this.renderGraph(graph).then(() => {
                this._model = model;
                this._activeGraph = graph;
                this.show('Graph');
                return this._model;
            }).catch((error) => {
                this.renderGraph(this._activeGraph).then(() => {
                    this.show('Graph');
                    throw error;
                }).catch(() => {
                    throw error;
                });
            });
        });
    }

    renderGraph(graph) {
        try {
            if (!graph) {
                return Promise.resolve();
            }
            else {
                var graphElement = this._host.document.getElementById('graph');
                while (graphElement.lastChild) {
                    graphElement.removeChild(graphElement.lastChild);
                }
    
                switch (this._host.environment('zoom')) {
                    case 'scroll':
                        this._zoom = 0;
                        graphElement.style.position = 'static';
                        graphElement.style.margin = 'auto';
                        break;
                    case 'd3':
                        this._zoom = null;
                        graphElement.style.position = 'absolute';
                        graphElement.style.margin = '0';
                        break;
                }
    
                var groups = graph.groups;
    
                var graphOptions = {};
                graphOptions.nodesep = 25;
                graphOptions.ranksep = 20;

                var g = new dagre.graphlib.Graph({ compound: groups });
                g.setGraph(graphOptions);
                g.setDefaultEdgeLabel(() => { return {}; });
            
                var nodeId = 0;
                var edgeMap = {};
            
                var clusterMap = {};
                var clusterParentMap = {};
    
                var id = new Date().getTime();
                var nodes = graph.nodes;

                if (nodes.length > 1500) {
                    graphOptions.ranker = 'longest-path';
                }

                this._host.event('Graph', 'Render', 'Size', nodes.length);

                var node;
                if (groups) {
                    for (node of nodes) {
                        if (node.group) {
                            var path = node.group.split('/');
                            while (path.length > 0) {
                                var name = path.join('/');
                                path.pop();
                                clusterParentMap[name] = path.join('/');
                            }
                        }
                    }
                }

                var input;
                var output;
                var connection;
                var tuple;

                var self = this;
                for (node of nodes) {
    
                    var element = new grapher.NodeElement(this._host.document);

                    var addNode = function(element, node, edges) {

                        var header =  element.block('header');
                        var styles = [ 'node-item-operator' ];
                        var category = node.category;
                        if (category) {
                            styles.push('node-item-operator-' + category.toLowerCase());
                        }
                        var content = self.showNames && node.name ? node.name : node.operator;
                        var tooltip = self.showNames && node.name ? node.operator : node.name;
                        header.add(null, styles, content, tooltip, () => { 
                            self.showNodeProperties(node, null);
                        });

                        if (node.function) {
                            header.add(null, [ 'node-item-function' ], '+', null, () => {
                                // debugger;
                            });
                        }

                        var initializers = [];
                        var hiddenInitializers = false;
                        if (self._showInitializers) {
                            for (var input of node.inputs) {
                                if (input.visible && input.connections.length == 1 && input.connections[0].initializer != null) {
                                    initializers.push(input);
                                }
                                if (!input.visible && input.connections.some((connection) => connection.initializer != null)) {
                                    hiddenInitializers = true;
                                }
                            }
                        }
                        var attributes = [];
                        if (self.showAttributes && node.attributes) {
                            attributes = node.attributes.filter((attribute) => attribute.visible);
                        }
                        if (initializers.length > 0 || hiddenInitializers || attributes.length > 0) {
                            var block = element.block('list');
                            block.handler = () => {
                                self.showNodeProperties(node);
                            };
                            for (var initializer of initializers) {
                                var connection = initializer.connections[0];
                                var type = connection.type;
                                var shape = '';
                                var separator = '';
                                if (type && type.shape && type.shape.dimensions && type.shape.dimensions.hasOwnProperty('length')) {
                                    shape = '\u3008' + type.shape.dimensions.join('\u00D7') + '\u3009';
                                    if (type.shape.dimensions.length == 0 && connection.initializer && !connection.initializer.state) {
                                        shape = connection.initializer.toString();
                                        if (shape && shape.length > 25) {
                                            shape = shape.substring(0, 25) + '...';
                                        }
                                        separator = ' = ';
                                    }
                                }
                                block.add('initializer-' + connection.id, initializer.name, shape, type ? type.toString() : '', separator);
                            }
                            if (hiddenInitializers) {
                                block.add(null, '\u3008' + '...' + '\u3009', '', null, '');
                            }

                            for (var attribute of attributes) {
                                if (attribute.visible) {
                                    var attributeValue = sidebar.NodeSidebar.formatAttributeValue(attribute.value, attribute.type);
                                    if (attributeValue && attributeValue.length > 25) {
                                        attributeValue = attributeValue.substring(0, 25) + '...';
                                    }
                                    block.add(null, attribute.name, attributeValue, attribute.type, ' = ');
                                }
                            }
                        }

                        if (edges) {
                            var inputs = node.inputs;
                            for (input of inputs) {
                                for (connection of input.connections) {
                                    if (connection.id != '' && !connection.initializer) {
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
                                }
                            }
                            var outputs = node.outputs;
                            if (node.chain && node.chain.length > 0) {
                                var chainOutputs = node.chain[node.chain.length - 1].outputs;
                                if (chainOutputs.length > 0) {
                                    outputs = chainOutputs;
                                }
                            }
                            for (output of outputs) {
                                for (connection of output.connections) {
                                    if (connection.id != '') {
                                        tuple = edgeMap[connection.id];
                                        if (!tuple) {
                                            tuple = { from: null, to: [] };
                                            edgeMap[connection.id] = tuple;
                                        }
                                        tuple.from = { 
                                            node: nodeId,
                                            name: output.name,
                                            type: connection.type
                                        };
                                    }
                                }
                            }
                        }
    
                        if (node.chain && node.chain.length > 0) {
                            for (var innerNode of node.chain) {
                                addNode(element, innerNode, false);
                            }
                        }

                        if (node.inner) {
                            addNode(element, node.inner, false);
                        }
                    }
    
                    addNode(element, node, true);

                    if (node.controlDependencies && node.controlDependencies.length > 0) {
                        for (var controlDependency of node.controlDependencies) {
                            tuple = edgeMap[controlDependency];
                            if (!tuple) {
                                tuple = { from: null, to: [] };
                                edgeMap[controlDependency] = tuple;
                            }
                            tuple.to.push({
                                node: nodeId,
                                name: controlDependency,
                                controlDependency: true
                            });
                        }
                    }

                    var nodeName = node.name;
                    if (nodeName) {
                        g.setNode(nodeId, { label: element.format(graphElement), id: 'node-' + nodeName });
                    }
                    else {
                        g.setNode(nodeId, { label: element.format(graphElement), id: 'node-' + id.toString() });
                        id++;
                    }
            
                    var createCluster = function(name) {
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
                }

                for (input of graph.inputs) {
                    for (connection of input.connections) {
                        tuple = edgeMap[connection.id];
                        if (!tuple) {
                            tuple = { from: null, to: [] };
                            edgeMap[connection.id] = tuple;
                        }
                        tuple.from = { 
                            node: nodeId,
                            type: connection.type
                        };
                    }
                    var types = input.connections.map(connection => connection.type || '').join('\n');
                    var inputName = input.name || '';
                    if (inputName.length > 16) {
                        inputName = inputName.split('/').pop();
                    }
    
                    var inputElement = new grapher.NodeElement(this._host.document);
                    var inputHeader = inputElement.block('header');
                    inputHeader.add(null, [ 'graph-item-input' ], inputName, types, () => {
                        this.showModelProperties();
                    });
                    g.setNode(nodeId++, { label: inputElement.format(graphElement), class: 'graph-input' } ); 
                }
            
                for (output of graph.outputs) {
                    for (connection of output.connections) {
                        tuple = edgeMap[connection.id];
                        if (!tuple) {
                            tuple = { from: null, to: [] };
                            edgeMap[connection.id] = tuple;
                        }
                        tuple.to.push({ node: nodeId });
                    }
                    var outputTypes = output.connections.map(connection => connection.type || '').join('\n');
                    var outputName = output.name || '';
                    if (outputName.length > 16) {
                        outputName = outputName.split('/').pop();
                    }
            
                    var outputElement = new grapher.NodeElement(this._host.document);
                    var outputHeader = outputElement.block('header');
                    outputHeader.add(null, [ 'graph-item-output' ], outputName, outputTypes, () => {
                        this.showModelProperties();
                    });
                    g.setNode(nodeId++, { label: outputElement.format(graphElement) } ); 
                }

                for (var edge of Object.keys(edgeMap)) {
                    tuple = edgeMap[edge];
                    if (tuple.from != null) {
                        for (var to of tuple.to) {
                            var text = '';
                            var type = tuple.from.type;
                            if (type && type.shape && type.shape.dimensions && type.shape.dimensions.length > 0) {
                                text = type.shape.dimensions.join('\u00D7');
                            }
            
                            if (this._showNames) {
                                text = edge.split('\n').shift(); // custom connection id
                            }
    
                            if (to.controlDependency) {
                                g.setEdge(tuple.from.node, to.node, { label: text, id: 'edge-' + edge, arrowhead: 'vee', class: 'edge-path-control-dependency' } );
                            }
                            else {
                                g.setEdge(tuple.from.node, to.node, { label: text, id: 'edge-' + edge, arrowhead: 'vee' } );
                            }
                        }
                    }
                }

                // Workaround for Safari background drag/zoom issue:
                // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
                var backgroundElement = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                backgroundElement.setAttribute('id', 'background');
                if (this._host.environment('zoom') == 'd3') {
                    backgroundElement.setAttribute('width', '100%');
                    backgroundElement.setAttribute('height', '100%');
                }
                backgroundElement.setAttribute('fill', 'none');
                backgroundElement.setAttribute('pointer-events', 'all');
                graphElement.appendChild(backgroundElement);

                var originElement = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'g');
                originElement.setAttribute('id', 'origin');
                graphElement.appendChild(originElement);
            
                if (this._host.environment('zoom') == 'd3') {
                    // Set up zoom support
                    var svg = d3.select(graphElement);
                    this._zoom = d3.zoom();
                    this._zoom(svg);
                    this._zoom.scaleExtent([0.1, 2]);
                    this._zoom.on('zoom', () => {
                        originElement.setAttribute('transform', d3.event.transform.toString());
                    });
                    this._zoom.transform(svg, d3.zoomIdentity);
                }

                return this._timeout(20).then(() => {
                    var graphRenderer = new grapher.Renderer(this._host.document, originElement);
                    graphRenderer.render(g);

                    var inputElements = graphElement.getElementsByClassName('graph-input');

                    switch (this._host.environment('zoom')) {
                        case 'scroll':
                            var size = graphElement.getBBox();
                            var graphMin = Math.min(size.width, size.height);
                            var windowMin = Math.min(window.innerWidth, window.innerHeight);
                            var delta = (Math.max(graphMin, windowMin) / 2.0) * 0.2;
                            var width = Math.ceil(delta + size.width + delta);
                            var height = Math.ceil(delta + size.height + delta);
                            originElement.setAttribute('transform', 'translate(' + delta.toString() + ', ' + delta.toString() + ') scale(1)');
                            backgroundElement.setAttribute('width', width);
                            backgroundElement.setAttribute('height', height);
                            this._width = width;
                            this._height = height;
                            this._zoom = 1;
                            graphElement.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
                            graphElement.setAttribute('width', width / this._zoom);
                            graphElement.setAttribute('height', height / this._zoom);
                            if (inputElements && inputElements.length > 0) {
                                // Center view based on input elements
                                for (var j = 0; j < inputElements.length; j++) {
                                    inputElements[j].scrollIntoView({ behavior: 'instant' });
                                    break;
                                }
                            }
                            else {
                                // this._zoom.transform(svg, d3.zoomIdentity.translate((svgSize.width - g.graph().width) / 2, (svgSize.height - g.graph().height) / 2));
                            }
                            break;
                        case 'd3':
                            var svgSize = graphElement.getBoundingClientRect();
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
                            break;
                    }
                    return;
                });
            }
        }
        catch (error) {
            return Promise.reject(error);
        }
    }

    applyStyleSheet(element, name) {
        var rules = [];
        for (var i = 0; i < this._host.document.styleSheets.length; i++) {
            var styleSheet = this._host.document.styleSheets[i];
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
            var graphElement = this._host.document.getElementById('graph');
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
                var blob = new Blob([ data ], { type: 'image/svg' });
                this._host.export(file, blob);
            }
    
            if (extension == 'png') {
                var imageElement = new Image();
                imageElement.onload = () => {
                    var max = Math.max(width, height);
                    var scale = ((max * 2.0) > 24000) ? (24000.0 / max) : 2.0;
                    var canvas = this._host.document.createElement('canvas');
                    canvas.width = Math.ceil(width * scale);
                    canvas.height = Math.ceil(height * scale);
                    var context = canvas.getContext('2d');
                    context.scale(scale, scale);
                    context.drawImage(imageElement, 0, 0);
                    this._host.document.body.removeChild(imageElement);
                    canvas.toBlob((blob) => {
                        if (blob) {
                            this._host.export(file, blob);
                        }
                        else {
                            var err = new Error();
                            err.name = 'Error exporting image.';
                            err.message = 'Image may be too large to render as PNG.';
                            this._host.exception(err, false);
                            this._host.error(err.name, err.message);
                        }
                    }, 'image/png');
                };
                imageElement.src = 'data:image/svg+xml;base64,' + window.btoa(unescape(encodeURIComponent(data)));
                this._host.document.body.insertBefore(imageElement, this._host.document.body.firstChild);
            }
        }
    }

    showModelProperties() {
        if (this._model) {
            var view = new sidebar.ModelSidebar(this._model, this._host);
            view.on('update-active-graph', (sender, name) => {
                this._updateActiveGraph(name);
            });
            this._sidebar.open(view.render(), 'Model Properties');
        }
    }
    
    showNodeProperties(node, input) {
        if (node) {
            var view = new sidebar.NodeSidebar(node, this._host);
            view.on('show-documentation', (/* sender, e */) => {
                this.showOperatorDocumentation(node);
            });
            view.on('export-tensor', (sender, tensor) => {
                this._host.require('./numpy').then((numpy) => {
                    var defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
                    this._host.save('NumPy Array', 'npy', defaultPath, (file) => {
                        try {
                            var array = new numpy.Array(tensor.value, tensor.type.dataType, tensor.type.shape.dimensions);
                            var blob = new Blob([ array.toBuffer() ], { type: 'application/octet-stream' });
                            this._host.export(file, blob);
                        }
                        catch (error) {
                            this.error('Error saving NumPy tensor.', error);
                        }
                    });
                }).catch(() => {
                });
            });
            if (input) {
                view.toggleInput(input.name);
            }
            this._sidebar.open(view.render(), 'Node Properties');
        }
    }

    showOperatorDocumentation(node) {
        var documentation = node.documentation;
        if (documentation) {
            var view = new sidebar.OperatorDocumentationSidebar(documentation);
            view.on('navigate', (sender, e) => {
                this._host.openURL(e.link);
            });
            this._sidebar.open(view.render(), 'Documentation');
        }
    }
};

class ModelError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading model.'; 
    }
}

class ModelContext {

    constructor(context) {
        this._context = context;
        this._tags = {};
    }

    request(file, encoding) {
        return this._context.request(file, encoding);
    }

    get identifier() {
        return this._context.identifier;
    }

    get buffer() {
        return this._context.buffer;
    }

    get text() {
        if (!this._text) {
            this._text = new TextDecoder('utf-8').decode(this.buffer);
        }
        return this._text;
    }

    tags(extension) {
        var tags = this._tags[extension];
        if (!tags) {
            tags = {};
            try {
                var reader = null;
                switch (extension) {
                    case 'pbtxt':
                        reader = prototxt.TextReader.create(this.text);
                        reader.start(false);
                        while (!reader.end(false)) {
                            var tag = reader.tag();
                            tags[tag] = true;
                            reader.skip();
                        }
                        break;
                    case 'pb':
                        reader = new protobuf.Reader.create(this.buffer);
                        while (reader.pos < reader.len) {
                            var tagType = reader.uint32();
                            tags[tagType >>> 3] = tagType & 7;
                            try {
                                reader.skipType(tagType & 7);
                            }
                            catch (err) {
                                tags = {};
                                break;
                            }
                        }
                        break;
                }
            }
            catch (error) {
                tags = {};
            }
            this._tags[extension] = tags;
        }
        return tags;
    }
}

class ArchiveContext {

    constructor(entries, rootFolder, identifier, buffer) {
        this._entries = {};
        if (entries) {
            for (var entry of entries) {
                if (entry.name.startsWith(rootFolder)) {
                    var name = entry.name.substring(rootFolder.length);
                    if (identifier.length > 0 && identifier.indexOf('/') < 0) {
                        this._entries[name] = entry.substring(rootFolder.length);
                    }
                }
            }
        }
        this._identifier = identifier.substring(rootFolder.length);
        this._buffer = buffer;
    }

    request(file, encoding) {
        var entry = this._entries[file];
        if (!entry) {
            return Promise.reject(new Error('File not found.'));
        }
        var data = entry.data;
        if (data != null) {
            data = new TextDecoder(encoding).decode(data);
        }
        return Promise.resolve(data);
    }

    get identifier() {
        return this._identifier;
    }

    get buffer() {
        return this._buffer;
    }
}

class ArchiveError extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading archive.';
    }
}

view.ModelFactoryService = class {

    constructor(host) {
        this._host = host;
        this._extensions = [];
        this.register('./onnx', [ '.onnx', '.pb', '.pbtxt', '.prototxt' ]);
        this.register('./mxnet', [ '.model', '.json', '.params' ]);
        this.register('./keras', [ '.h5', '.keras', '.hdf5', '.json', '.model' ]);
        this.register('./coreml', [ '.mlmodel' ]);
        this.register('./caffe', [ '.caffemodel', '.pbtxt', '.prototxt', '.pt' ]);
        this.register('./caffe2', [ '.pb', '.pbtxt', '.prototxt' ]);
        this.register('./pytorch', [ '.pt', '.pth', '.pkl', '.h5', '.t7', '.model', '.dms', '.pth.tar', '.ckpt' ]);
        this.register('./torch', [ '.t7' ]);
        this.register('./torchscript', [ '.pt', '.pth' ]);
        this.register('./tflite', [ '.tflite', '.lite' ]);
        this.register('./tf', [ '.pb', '.meta', '.pbtxt', '.prototxt' ]);
        this.register('./sklearn', [ '.pkl', '.joblib' ]);
        this.register('./cntk', [ '.model', '.cntk', '.cmf', '.dnn' ]);
        this.register('./openvino', [ '.xml' ]);
        this.register('./darknet', [ '.cfg' ]);
        this.register('./paddle', [ '.paddle', '__model__' ]);
    }

    register(id, extensions) {
        for (var extension of extensions) {
            this._extensions.push({ extension: extension, id: id });
        }
    }
 
    open(context) {
        return this._openArchive(context).then((context) => {
            context = new ModelContext(context);
            var extension = context.identifier.split('.').pop().toLowerCase();
            var modules = this._filter(context);
            if (modules.length == 0) {
                throw new ModelError("Unsupported file extension '." + extension + "'.");
            }
            var errors = [];
            var match = false;
            var nextModule = () => {
                if (modules.length > 0) {
                    var id = modules.shift();
                    return this._host.require(id).then((module) => {
                        if (!module.ModelFactory) {
                            throw new ModelError("Failed to load module '" + id + "'.");
                        }
                        var modelFactory = new module.ModelFactory(); 
                        if (!modelFactory.match(context)) {
                            return nextModule();
                        }
                        match++;
                        return modelFactory.open(context, this._host).then((model) => {
                            return model;
                        }).catch((error) => {
                            errors.push(error);
                            return nextModule();
                        });
                    });
                }
                else {
                    if (match) {
                        if (errors.length == 1) {
                            throw errors[0];
                        }
                        throw new ModelError(errors.map((err) => err.message).join('\n'));
                    }
                    throw new ModelError("Unsupported file content for extension '." + extension + "' in '" + context.identifier + "'.");
                }
            };
            return nextModule();
        });
    }

    _openArchive(context) {
        var extension;
        var archive;
        var entry;
        var message;

        var identifier = context.identifier;
        var buffer = context.buffer;

        try {
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
        }
        catch (error) {
            message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            return Promise.reject(new ArchiveError(message + " in '" + identifier + "'."));
        }

        try {
            extension = identifier.split('.').pop().toLowerCase();
            switch (extension) {
                case 'tar':
                    // handle .pth.tar
                    var torch = [ 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ];
                    if (!buffer || buffer.length < 14 || buffer[0] != 0x80 || !torch.every((v, i) => v == buffer[i + 2])) {
                        archive = new tar.Archive(buffer);
                    }
                    break;
                case 'zip':
                    archive = new zip.Archive(buffer);
                    break;
            }
        }
        catch (error) {
            message = error && error.message ? error.message : error.toString();
            message = message.endsWith('.') ? message.substring(0, message.length - 1) : message;
            return Promise.reject(new ArchiveError(message + " in '" + identifier + "'."));
        }

        if (!archive) {
            return Promise.resolve(context);
        }

        try {
            var folders = {};
            for (entry of archive.entries) {
                if (entry.name.indexOf('/') != -1) {
                    folders[entry.name.split('/').shift() + '/'] = true;
                }
                else {
                    folders['/'] = true;
                }
            }
            if (extension == 'tar') {
                delete folders['PaxHeader/'];
            }
            var rootFolder = Object.keys(folders).length == 1 ? Object.keys(folders)[0] : '';
            rootFolder = rootFolder == '/' ? '' : rootFolder;
            var matches = [];
            var entries = archive.entries.slice();
            var nextEntry = () => {
                if (entries.length > 0) {
                    var entry = entries.shift();
                    if (entry.name.startsWith(rootFolder)) {
                        var identifier = entry.name.substring(rootFolder.length);
                        if (identifier.length > 0 && identifier.indexOf('/') < 0 && !identifier.startsWith('.')) {
                            var context = new ModelContext(new ArchiveContext(null, rootFolder, entry.name, entry.data));
                            var modules = this._filter(context);
                            var nextModule = () => {
                                if (modules.length > 0) {
                                    var id = modules.shift();
                                    return this._host.require(id).then((module) => {
                                        if (!module.ModelFactory) {
                                            throw new ArchiveError("Failed to load module '" + id + "'.", null);
                                        }
                                        var factory = new module.ModelFactory();
                                        if (factory.match(context)) {
                                            matches.push(entry);
                                            modules = [];
                                        }
                                        return nextModule();
                                    });
                                }
                                else {
                                    return nextEntry();
                                }
                            };
                            return nextModule();
                        }
                    }
                    return nextEntry();
                }
                else {
                    if (matches.length == 0) {
                        return Promise.reject(new ArchiveError('Archive does not contain model file.'));
                    }
                    else if (matches.length > 1) {
                        return Promise.reject(new ArchiveError('Archive contains multiple model files.'));
                    }
                    var match = matches[0];
                    return Promise.resolve(new ModelContext(new ArchiveContext(entries, rootFolder, match.name, match.data)));
                }
            };
            return nextEntry();
        }
        catch (error) {
            return Promise.reject(new ArchiveError(error.message));
        }
    }

    accept(identifier) {
        var extension = identifier.toLowerCase().split('.').pop();
        var excludes = [
            'blockmap', 'checkpoint', 'dat', 'test', 'bytes', 'desktop', 'graph',
            'index', 'data-00000-of-00001',
            'exe', 'dll', 'bin', 'raw', 'msg',
            'html', 'pdf', 'rtf', 'txt', 'md', 'svg', 'csv',
            'xls', 'doc', 'ppt', 'xlsx', 'docx', 'pptx',
            'jpeg', 'jpg', 'png', 'gif', 'ico', 'icns',
            'js', 'py', 'pyc', 'ipynb',
            'weights',
            'mp3', 'mp4', 'mov',
            'npy', 'npz',
            'tmp'
        ];
        if (excludes.some((exclude) => exclude == extension)) {
            return false;
        }
        if (extension.startsWith('data-')) {
            return false;
        }
        return true;
    }

    _filter(context) {
        var moduleList = [];
        var moduleMap = {};
        var identifier = context.identifier.toLowerCase();
        for (var extension of this._extensions) {
            if (identifier.endsWith(extension.extension)) {
                if (!moduleMap[extension.id]) {
                    moduleList.push(extension.id);
                    moduleMap[extension.id] = true;
                }
            }
        }
        return moduleList;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.View = view.View;
    module.exports.ModelFactoryService = view.ModelFactoryService;
}
