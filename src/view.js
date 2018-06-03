/*jshint esversion: 6 */

class View {

    constructor(host) {
        this._host = host;
        this._model = null;
        this._sidebar = new Sidebar();
        this._host.initialize(this);
        this._showDetails = true;
        this._showNames = false;
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
    }
    
    show(page) {

        if (!page) {
            page = (!this._model && !this._graph) ? 'welcome' : 'graph';
        }

        this._sidebar.close();

        var welcomeElement = document.getElementById('welcome');
        var openFileButton = document.getElementById('open-file-button');
        var spinnerElement = document.getElementById('spinner');
        var graphElement = document.getElementById('graph');
        var toolbarElement = document.getElementById('toolbar');
    
        if (page == 'welcome') {
            document.body.style.cursor = 'default';
            welcomeElement.style.display = 'block';
            var offsetHeight1 = welcomeElement.offsetHeight; 
            openFileButton.style.display = 'block';
            openFileButton.style.opacity = 1;
            spinnerElement.style.display = 'none';
            graphElement.style.display = 'none';
            graphElement.style.opacity = 0;
            toolbarElement.style.display = 'none';

            this._model = null;
            this._graph = false;
        }
    
        if (page == 'spinner') {
            document.body.style.cursor = 'wait';
            welcomeElement.style.display = 'block';
            openFileButton.style.opacity = 0;
            spinnerElement.style.display = 'block';
            var offsetHeight2 = spinnerElement.offsetHeight;
            graphElement.style.display = 'block';
            graphElement.style.opacity = 0;
            toolbarElement.style.display = 'none';
        }
    
        if (page == 'graph') {
            welcomeElement.style.display = 'none';
            openFileButton.style.display = 'none';
            spinnerElement.style.display = 'none';
            graphElement.style.display = 'block';
            graphElement.style.opacity = 1;
            toolbarElement.style.display = 'block';
            document.body.style.cursor = 'default';
        }
    }

    copy() {
        document.execCommand('copy');
    }

    find() {
        this._sidebar.open('<div></div>', 'Find');
    }

    toggleDetails() {
        this._showDetails = !this._showDetails;
        this.show('spinner');
        this.updateGraph(this._model, this._activeGraph);
    }

    get showDetails() {
        return this._showDetails;
    }

    toggleNames() {
        this._showNames = !this._showNames;
        this.show('spinner');
        this.updateGraph(this._model, this._activeGraph);
    }

    get showNames() {
        return this._showNames;
    }

    zoomIn() {
        if (this._zoom) {
            var svgElement = document.getElementById('graph');
            d3.select(svgElement).call(this._zoom.scaleBy, 1.2);
        }
    }

    zoomOut() {
        if (this._zoom) {
            var svgElement = document.getElementById('graph');
            d3.select(svgElement).call(this._zoom.scaleBy, 0.8);
        }
    }

    resetZoom() { 
        if (this._zoom) {
            var svgElement = document.getElementById('graph');
            d3.select(svgElement).call(this._zoom.scaleTo, 1);
        }
    }

    preventZoom(e) {
        if (e.shiftKey || e.ctrlKey) {
            e.preventDefault();
        }
    }

    loadBuffer(buffer, identifier, callback) {
        var modelFactoryRegistry = [
            new OnnxModelFactory(),
            new MXNetModelFactory(),
            new KerasModelFactory(),
            new CoreMLModelFactory(),
            new CaffeModelFactory(),
            new Caffe2ModelFactory(), 
            new TensorFlowLiteModelFactory(),
            new TensorFlowModelFactory()
        ];
        var matches = modelFactoryRegistry.filter((factory) => factory.match(buffer, identifier));
        var next = () => {
            if (matches.length > 0) {
                var modelFactory = matches.shift();
                modelFactory.open(buffer, identifier, this._host, (err, model) => {
                    if (model || matches.length == 0) {
                        callback(err, model);
                    }
                    else {
                        next();
                    }
                });
            }
            else {
                var extension = identifier.split('.').pop();
                callback(new Error('Unsupported file extension \'.' + extension + '\'.'), null);
            }
        };
        next();
    }

    openBuffer(buffer, identifier, callback) {
        this._sidebar.close();
        setTimeout(() => {
            this.loadBuffer(buffer, identifier, (err, model) => {
                if (err) {
                    callback(err);
                }
                else {
                    setTimeout(() => {
                        this._graph = false;
                        try {
                            var graph = model.graphs.length > 0 ? model.graphs[0] : null;
                            this.updateGraph(model, graph);
                            this._model = model;
                            this._activeGraph = graph;
                            callback(null);
                        }
                        catch (err) {
                            try {
                                this.updateGraph(this._model, this._activeGraph);
                            }
                            catch (obj) {
                                this._model = null;
                                this._activeGraph = null;
                            }
                            callback(err);
                        }
                    }, 2);   
                }
            });    
        }, 2);
    }

    showError(err) {
        this._sidebar.close();
        this._host.showError(err.toString());
        this.show('welcome');
    }

    updateActiveGraph(name) {
        this._sidebar.close();
        if (this._model) {
            var model = this._model;
            var graph = model.graphs.filter(graph => graph.name).shift();
            if (graph) {
                this.show('spinner');
                setTimeout(() => {
                    try {
                        this.updateGraph(model, graph);
                        this._model = model;
                        this._activeGraph = graph;
                    }
                    catch (obj) {
                        this._model = null;
                        this._activeGraph = null;
                    }
                }, 2);
    
            }
        }
    }
    
    updateGraph(model, graph) {

        if (!graph) {
            this.show('graph');
            return;
        }
    
        var svgElement = document.getElementById('graph');
        while (svgElement.lastChild) {
            svgElement.removeChild(svgElement.lastChild);
        }

        this._zoom = null;

        var groups = graph.groups;

        var graphOptions = {};
        if (!this._showDetails) {
            graphOptions.nodesep = 25;
            graphOptions.ranksep = 25;
        }

        var g = new dagre.graphlib.Graph({ compound: groups });
        g.setGraph(graphOptions);
        // g.setGraph({ align: 'DR' });
        // g.setGraph({ ranker: 'network-simplex' });
        // g.setGraph({ ranker: 'tight-tree' });
        // g.setGraph({ ranker: 'longest-path' });
        // g.setGraph({ acyclicer: 'greedy' });
        g.setDefaultEdgeLabel(() => { return {}; });
    
        var nodeId = 0;
        var edgeMap = {};
    
        var clusterMap = {};
        var clusterParentMap = {};
    
        if (groups) {
            graph.nodes.forEach((node) => {
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

        graph.nodes.forEach((node) => {
            var formatter = new NodeFormatter();
    
            function addOperator(view, formatter, node) {
                if (node) {
                    var styles = [ 'node-item-operator' ];
                    var category = node.category;
                    if (category) {
                        styles.push('node-item-operator-' + category.toLowerCase());
                    }
                    var text = view.showNames && node.name ? node.name : (node.primitive ? node.primitive : node.operator);
                    var title = view.showNames && node.name ? node.operator : node.name;
                    formatter.addItem(text, styles, title, () => { 
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
                    var inputClass = 'node-item-input';
                    if (initializers.length == 0) {
                        inputClass = 'node-item-input';
                        if (input.hidden) {
                            hiddenInputs = true;
                        }
                    }
                    else {
                        if (initializers.length == input.connections.length) {
                            inputClass = 'node-item-constant';
                            if (input.hidden) {
                                hiddenInitializers = true;
                            }
                        }
                        else {
                            inputClass = 'node-item-constant';
                            if (input.hidden) {
                                hiddenInputs = true;
                            }
                        }
                    }
    
                    if (this._showDetails) {
                        if (!input.hidden) {
                            var types = input.connections.map(connection => connection.type ? connection.type : '').join('\n');
                            formatter.addItem(input.name, [ inputClass ], types, () => {
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
                    formatter.addItem('...', [ 'node-item-input' ], '', () => {
                        this.showNodeProperties(node, null);
                    });    
                }
                if (hiddenInitializers) {
                    formatter.addItem('...', [ 'node-item-constant' ], '', () => {
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
                        name: output.name
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
                                attributeValue = attribute.value;
                                if (attributeValue.length > 25) {
                                    attributeValue = attributeValue.substring(0, 25) + '...';
                                }
                            }
                            formatter.addAttribute(attribute.name, attributeValue, attribute.type);
                        }
                    });
                }
            }
    
            g.setNode(nodeId, { label: formatter.format(svgElement) });
    
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
                var name = node.group;
                if (name && name.length > 0) {
                    if (!clusterParentMap.hasOwnProperty(name)) {
                        var lastIndex = name.lastIndexOf('/');
                        if (lastIndex != -1) {
                            name = name.substring(0, lastIndex);
                            if (!clusterParentMap.hasOwnProperty(name)) {
                                name = null;
                            }
                        }
                        else {
                            name = null;
                        }
                    }
                    if (name) {
                        createCluster(name);
                        g.setParent(nodeId, name);
                    }
                }
            }
        
            nodeId++;
        });
    
        graph.inputs.forEach((input) => {
            var tuple = edgeMap[input.id];
            if (!tuple) {
                tuple = { from: null, to: [] };
                edgeMap[input.id] = tuple;
            }
            tuple.from = { 
                node: nodeId,
            };

            var formatter = new NodeFormatter();
            formatter.addItem(input.name, [ 'graph-item-input' ], input.type, () => {
                this.showModelProperties();
            });
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
            formatter.addItem(output.name, [ 'graph-item-output' ], output.type, () => {
                this.showProperties();
            });
            g.setNode(nodeId++, { label: formatter.format(svgElement) } ); 
        });
    
        Object.keys(edgeMap).forEach((edge) => {
            var tuple = edgeMap[edge];
            if (tuple.from != null) {
                tuple.to.forEach((to) => {
                    var text = '';
                    if (tuple.from.name && to.name) {
                        text = tuple.from.name + ' \u21E8 ' + to.name;
                    }
                    else if (tuple.from.name) {
                        text = tuple.from.name;
                    }
                    else {
                        text = to.name;
                    }
    
                    if (this._showNames) {
                        text = edge;
                    }
                    if (!this._showDetails) {
                        text = '';
                    }

                    if (to.dependency) { 
                        g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee', curve: d3.curveBasis, class: 'edge-path-control' } );
                    }
                    else {
                        g.setEdge(tuple.from.node, to.node, { label: text, arrowhead: 'vee', curve: d3.curveBasis } );
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
        svgElement.appendChild(backgroundElement);
    
        var originElement = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        originElement.setAttribute('id', 'origin');
        svgElement.appendChild(originElement);
    
        // Set up zoom support
        this._zoom = d3.zoom();
        this._zoom.scaleExtent([0.1, 2]);
        this._zoom.on('zoom', (e) => {
            d3.select(originElement).attr('transform', d3.event.transform);
        });
        var svg = d3.select(svgElement);
        svg.call(this._zoom);
        svg.call(this._zoom.transform, d3.zoomIdentity);
        this._svg = svg;
    
        setTimeout(() => {
    
            var graphRenderer = new GraphRenderer(originElement);
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
    
                svg.call(this._zoom.transform, d3.zoomIdentity.translate((svgSize.width / 2) - x, (svgSize.height / 4) - y));
            }
            else {
                svg.call(this._zoom.transform, d3.zoomIdentity.translate((svgSize.width - g.graph().width) / 2, (svgSize.height - g.graph().height) / 2));
            }    
        
            this.show('graph');
        }, 2);
    }

    copyStylesInline(destinationNode, sourceNode) {
        var containerElements = ["svg","g"];
        for (var cd = 0; cd < destinationNode.childNodes.length; cd++) {
            var child = destinationNode.childNodes[cd];
            if (containerElements.indexOf(child.tagName) != -1) {
                 this.copyStylesInline(child, sourceNode.childNodes[cd]);
                 continue;
            }
            var style = sourceNode.childNodes[cd].currentStyle || window.getComputedStyle(sourceNode.childNodes[cd]);
            if (style == "undefined" || style == null) continue;
            for (var st = 0; st < style.length; st++){
                 child.style.setProperty(style[st], style.getPropertyValue(style[st]));
            }
        }
    }

    transferStyleSheet(element, name) {

        var result = [];
        result.push('<style type="text/css">');
        for (var i = 0; i < document.styleSheets.length; i++) {
            var styleSheet = document.styleSheets[i];
            if (styleSheet && styleSheet.href && styleSheet.href.endsWith('/' + name)) {
                for (var j = 0; j < styleSheet.rules.length; j++) {
                    var rule = styleSheet.rules[j];
                    result.push(rule.cssText);
                }
            }
        }
        result.push('</style>');

        var defsElement = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        defsElement.innerHTML = result.join('\n') + '\n';

        element.insertBefore(defsElement, element.firstChild);
    }

    export(file) {
        var extension = '';
        var lastIndex = file.lastIndexOf('.');
        if (lastIndex != -1) {
            extension = file.substring(lastIndex + 1);
        }
        if (extension != 'png' && extension != 'svg') {
            return;
        }

        var svgElement = document.getElementById('graph');
        var exportElement = svgElement.cloneNode(true);
        this.transferStyleSheet(exportElement, 'view-render.css');
        exportElement.setAttribute('id', 'export');
        exportElement.removeAttribute('width');
        exportElement.removeAttribute('height');
        exportElement.style.removeProperty('opacity');
        exportElement.style.removeProperty('display');
        var originElement = exportElement.getElementById('origin');
        originElement.setAttribute('transform', 'translate(0,0) scale(1)');
        var backgroundElement = exportElement.getElementById('background');
        backgroundElement.removeAttribute('width');
        backgroundElement.removeAttribute('height');

        var parentElement = svgElement.parentElement;
        parentElement.insertBefore(exportElement, svgElement);
        var size = exportElement.getBBox();
        parentElement.removeChild(exportElement);

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
            this._host.export(file, data, 'image/svg');
        }

        if (extension == 'png') {
            var imageElement = new Image();
            document.body.insertBefore(imageElement, document.body.firstChild);
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
                var pngBase64 = canvas.toDataURL('image/png');
                this._host.export(file, pngBase64, 'image/png');
            };
            imageElement.src = 'data:image/svg+xml;base64,' + window.btoa(unescape(encodeURIComponent(data)));
        }
    }

    showModelProperties() {
        if (this._model) {
            var template = Handlebars.compile(summaryTemplate, 'utf-8');
            var data = template(this._model);
            this._sidebar.open(data, 'Model Properties');
        }
    }
    
    showNodeProperties(node, input) {
        if (node) {
            var documentationHandler = () => {
                this.showDocumentation(node);
            };
            var view = new NodeView(node, documentationHandler);
            if (input) {
                view.toggleInput(input.name);
            }
            this._sidebar.open(view.elements, 'Node Properties');
        }
    }

    showDocumentation(node) {
        var documentation = node.documentation;
        if (documentation) {
            this._sidebar.open(documentation, 'Documentation');
    
            var documentationElement = document.getElementById('documentation');
            if (documentationElement) {
                documentationElement.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        var link = e.target.href;
                        if (link.startsWith('http://') || link.startsWith('https://')) {
                            this._host.openURL(link);
                            e.preventDefault();
                        }
                    }
                });
            }
        }
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
            while (contentElement.firstChild) {
                contentElement.removeChild(contentElement.firstChild);
            }
            if (typeof content == 'string') {
                contentElement.innerHTML = content;
            }
            else if (content instanceof Array) {
                content.forEach((element) => {
                    contentElement.appendChild(element);
                });
            }
            else {
                contentElement.appendChild(content);
            }
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

window.view = new View(window.host);

function updateActiveGraph(name) {
    window.view.updateActiveGraph(name);
}

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
