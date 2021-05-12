/* jshint esversion: 6 */

var view = view || {};

var base = base || require('./base');
var zip = zip || require('./zip');
var gzip = gzip || require('./gzip');
var tar = tar || require('./tar');
var json = json || require('./json');
var protobuf = protobuf || require('./protobuf');
var python = python || require('./python');

var d3 = d3 || require('d3');

var sidebar = sidebar || require('./view-sidebar');
var grapher = grapher || require('./view-grapher');

view.View = class {

    constructor(host, id) {
        this._host = host;
        this._id = id ? ('-' + id) : '';
        this._host.initialize(this).then(() => {
            this._model = null;
            this._selection = [];
            this._sidebar = new sidebar.Sidebar(this._host, id);
            this._showAttributes = false;
            this._showInitializers = true;
            this._showNames = false;
            this._showHorizontal = false;
            this._searchText = '';
            this._modelFactoryService = new view.ModelFactoryService(this._host);
            this._getElementById('zoom-in-button').addEventListener('click', () => {
                this.zoomIn();
            });
            this._getElementById('zoom-out-button').addEventListener('click', () => {
                this.zoomOut();
            });
            this._getElementById('sidebar').addEventListener('mousewheel', (e) => {
                this._preventZoom(e);
            });
            this._host.document.addEventListener('keydown', () => {
                this.clearSelection();
            });
            this._host.start();
            switch (this._host.environment('zoom')) {
                case 'scroll': {
                    const userAgent = navigator.userAgent.toLowerCase();
                    const safari = userAgent.indexOf('safari') !== -1 && userAgent.indexOf('chrome') === -1;
                    const elements = [ 'graph', 'toolbar' ];
                    for (const id of elements) {
                        const element = this._getElementById(id);
                        element.addEventListener('mousewheel', (e) => {
                            this._mouseWheelHandler(e);
                        });
                        element.addEventListener('scroll', (e) => {
                            this._scrollHandler(e);
                        });
                        element.addEventListener('wheel', (e) => {
                            this._mouseWheelHandler(e);
                        });
                        if (safari) {
                            element.addEventListener('gesturestart', (e) => {
                                e.preventDefault();
                                this._gestureZoom = this._zoom;
                            }, false);
                            element.addEventListener('gesturechange', (e) => {
                                e.preventDefault();
                                this._updateZoom(this._gestureZoom * e.scale, e);
                            }, false);
                            element.addEventListener('gestureend', (e) => {
                                e.preventDefault();
                                this._updateZoom(this._gestureZoom * e.scale, e);
                            }, false);
                        }
                        else {
                            element.addEventListener('touchstart', (e) => {
                                if (e.touches.length === 2) {
                                    this._touchPoints = Array.from(e.touches);
                                    this._touchZoom = this._zoom;
                                }
                            }, { passive: true });
                            element.addEventListener('touchmove', (e) => {
                                if (Array.isArray(this._touchPoints) && this._touchPoints.length === 2 && e.touches.length === 2) {
                                    const distance = (points) => {
                                        const dx =(points[1].clientX - points[0].clientX);
                                        const dy =(points[1].clientY - points[0].clientY);
                                        return Math.sqrt(dx * dx + dy * dy);
                                    };
                                    const d1 = distance(Array.from(e.touches));
                                    const d2 = distance(this._touchPoints);
                                    if (d2 !== 0) {
                                        const points = this._touchPoints;
                                        const e = {
                                            pageX: (points[1].pageX + points[0].pageX) / 2,
                                            pageY: (points[1].pageY + points[0].pageY) / 2
                                        };
                                        const zoom = d2 === 0 ? d1 : d1 / d2;
                                        this._updateZoom(this._touchZoom * zoom, e);
                                    }
                                }
                            }, { passive: true });
                            element.addEventListener('touchcancel', () => {
                                delete this._touchPoints;
                                delete this._touchZoom;
                            }, { passive: true });
                            element.addEventListener('touchend', () => {
                                delete this._touchPoints;
                                delete this._touchZoom;
                            }, { passive: true });
                        }
                    }
                    break;
                }
                case 'd3': {
                    this._getElementById('toolbar').addEventListener('mousewheel', (e) => {
                        this._preventZoom(e);
                    });
                    break;
                }
            }
        }).catch((err) => {
            this.error(err, null, null);
        });
    }

    show(page) {
        if (!page) {
            page = (!this._model && !this._activeGraph) ? 'welcome' : 'default';
        }
        this._host.screen(page);
        if (this._sidebar) {
            this._sidebar.close();
        }
        this._host.document.body.setAttribute('class', page);
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
            const graphElement = this._getElementById('canvas');
            const view = new sidebar.FindSidebar(this._host, graphElement, this._activeGraph);
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

    get model() {
        return this._model;
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

    toggleDirection() {
        this._showHorizontal = !this._showHorizontal;
        this._reload();
    }

    get showHorizontal() {
        return this._showHorizontal;
    }

    _reload() {
        this.show('welcome spinner');
        if (this._model && this._activeGraph) {
            this._updateGraph(this._model, this._activeGraph).catch((error) => {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            });
        }
    }

    _timeout(time) {
        return new Promise((resolve) => {
            setTimeout(() => { resolve(); }, time);
        });
    }

    _getElementById(id) {
        return this._host.document.getElementById(id + this._id);
    }

    zoomIn() {
        switch (this._host.environment('zoom')) {
            case 'scroll':
                this._updateZoom(this._zoom * 1.1);
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleBy(d3.select(this._getElementById('canvas')), 1.2);
                }
                break;
        }
    }

    zoomOut() {
        switch (this._host.environment('zoom')) {
            case 'scroll':
                this._updateZoom(this._zoom * 0.9);
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleBy(d3.select(this._getElementById('canvas')), 0.8);
                }
                break;
        }
    }

    resetZoom() {
        switch (this._host.environment('zoom')) {
            case 'scroll':
                this._updateZoom(1);
                break;
            case 'd3':
                if (this._zoom) {
                    this._zoom.scaleTo(d3.select(this._getElementById('canvas')), 1);
                }
                break;
        }
    }

    _preventZoom(e) {
        if (e.shiftKey || e.ctrlKey) {
            e.preventDefault();
        }
    }

    _updateZoom(zoom, e) {

        const graphElement = this._getElementById('graph');

        const min = Math.min(Math.max(graphElement.clientHeight / this._height, 0.2), 1);

        zoom = Math.min(zoom, 1.4);
        zoom = Math.max(min, zoom);

        const scrollLeft = this._scrollLeft || graphElement.scrollLeft;
        const scrollTop = this._scrollTop || graphElement.scrollTop;

        const x = (e ? e.pageX : (graphElement.clientWidth / 2)) + scrollLeft;
        const y = (e ? e.pageY : (graphElement.clientHeight / 2)) + scrollTop;

        const canvasElement = this._getElementById('canvas');
        canvasElement.style.width = zoom * this._width;
        canvasElement.style.height = zoom * this._height;

        this._scrollLeft = ((x * zoom) / this._zoom) - (x - scrollLeft);
        this._scrollTop = ((y * zoom) / this._zoom) - (y - scrollTop);
        this._scrollLeft = Math.max(0, this._scrollLeft);
        this._scrollTop = Math.max(0, this._scrollTop);
        graphElement.scrollLeft = this._scrollLeft;
        graphElement.scrollTop = this._scrollTop;

        this._zoom = zoom;
    }

    _mouseWheelHandler(e) {
        if (e.shiftKey || e.ctrlKey) {
            this._updateZoom(this._zoom + (e.wheelDelta * 1.0 / 4000.0), e);
            e.preventDefault();
        }
    }

    _scrollHandler(e) {

        if (this._scrollLeft && e.target.scrollLeft !== Math.floor(this._scrollLeft)) {
            delete this._scrollLeft;
        }
        if (this._scrollTop && e.target.scrollTop !== Math.floor(this._scrollTop)) {
            delete this._scrollTop;
        }
    }

    select(selection) {
        this.clearSelection();
        if (selection && selection.length > 0) {
            const graphElement = this._getElementById('graph');
            switch (this._host.environment('zoom')) {
                case 'd3': {
                    let x = 0;
                    let y = 0;
                    for (const element of selection) {
                        element.classList.add('select');
                        this._selection.push(element);
                        const transform = element.transform.baseVal.consolidate();
                        const box = element.getBBox();
                        const ex = transform ? transform.matrix.e : box.x + (box.width / 2);
                        const ey = transform ? transform.matrix.f : box.y + (box.height / 2);
                        x += ex;
                        y += ey;
                    }
                    x = x / selection.length;
                    y = y / selection.length;
                    const canvasElement = this._getElementById('canvas');
                    const canvasRect = canvasElement.getBoundingClientRect();
                    this._zoom.transform(d3.select(canvasElement), d3.zoomIdentity.translate((canvasRect.width / 2) - x, (canvasRect.height / 2) - y));
                    break;
                }
                case 'scroll': {
                    let x = 0;
                    let y = 0;
                    for (const element of selection) {
                        element.classList.add('select');
                        this._selection.push(element);
                        const rect = element.getBoundingClientRect();
                        x += rect.left + (rect.width / 2);
                        y += rect.top + (rect.height / 2);
                    }
                    x = x / selection.length;
                    y = y / selection.length;
                    const rect = graphElement.getBoundingClientRect();
                    const left = (graphElement.scrollLeft + x - rect.left) - (rect.width / 2);
                    const top = (graphElement.scrollTop + y - rect.top) - (rect.height / 2);
                    graphElement.scrollTo({ left: left, top: top, behavior: 'smooth' });
                    break;
                }
            }
        }
    }

    clearSelection() {
        while (this._selection.length > 0) {
            const element = this._selection.pop();
            element.classList.remove('select');
        }
    }

    error(err, name, screen) {
        if (this._sidebar) {
            this._sidebar.close();
        }
        this._host.exception(err, false);

        const knowns = [
            { name: '', message: /^Invalid argument identifier/, url: 'https://github.com/lutzroeder/netron/issues/540' },
            { name: '', message: /^Cannot read property/, url: 'https://github.com/lutzroeder/netron/issues/647' },
            { name: '', message: /^Failed to render tensor/, url: 'https://github.com/lutzroeder/netron/issues/681' },
            { name: 'Error', message: /^EPERM: operation not permitted/, url: 'https://github.com/lutzroeder/netron/issues/551' },
            { name: 'Error', message: /^EACCES: permission denied/, url: 'https://github.com/lutzroeder/netron/issues/504' },
            { name: 'RangeError', message: /^Offset is outside the bounds of the DataView/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'RangeError', message: /^start offset of Int32Array/, url: 'https://github.com/lutzroeder/netron/issues/565' },
            { name: 'RangeError', message: /^Maximum call stack size exceeded/, url: 'https://github.com/lutzroeder/netron/issues/589' },
            { name: 'RangeError', message: /^Invalid string length/, url: 'https://github.com/lutzroeder/netron/issues/648' },
            { name: 'Error loading model.', message: /^Unsupported file content \(/, url: 'https://github.com/lutzroeder/netron/issues/550' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers content/, url: 'https://github.com/lutzroeder/netron/issues/593' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers text content/, url: 'https://github.com/lutzroeder/netron/issues/594' },
            { name: 'Error loading model.', message: /^Unsupported JSON content/, url: 'https://github.com/lutzroeder/netron/issues/595' },
            { name: 'Error loading Caffe model.', message: /^File format is not caffe.NetParameter (Offset is outside the bounds of the DataView)/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'Error loading Darknet model.', message: /^Invalid tensor shape/, url: 'https://github.com/lutzroeder/netron/issues/541' },
            { name: 'Error loading Keras model.', message: /^Unsupported data object header version/, url: 'https://github.com/lutzroeder/netron/issues/548' },
            { name: 'Error loading MNN model.', message: /^Offset is outside the bounds of the DataView/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'Error loading PyTorch model.', message: /^File does not contain root module or state dictionary/, url: 'https://github.com/lutzroeder/netron/issues/543' },
            { name: 'Error loading PyTorch model.', message: /^Module does not contain modules/, url: 'https://github.com/lutzroeder/netron/issues/544' },
            { name: 'Error loading PyTorch model.', message: /^Failed to resolve module/, url: 'https://github.com/lutzroeder/netron/issues/545' },
            { name: 'Error loading PyTorch model.', message: /^Unsupported function/, url: 'https://github.com/lutzroeder/netron/issues/546' },
            { name: 'Error loading PyTorch model.', message: /^Unsupported uninitialized argument/, url: 'https://github.com/lutzroeder/netron/issues/547' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto/, url: 'https://github.com/lutzroeder/netron/issues/549' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx.ModelProto (Offset is outside the bounds of the DataView)/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'Error loading TensorFlow Lite model.', message: /^Offset is outside the bounds of the DataView/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'Error loading UFF model.', message: /^Unknown attribute/, url: 'https://github.com/lutzroeder/netron/issues/649' }
        ];
        const known = knowns.find((known) => (known.name.length === 0 || known.name === err.name) && err.message.match(known.message));
        const message = err.message + (known ? '\n\nPlease provide information about this issue at ' + known.url + '.' : '');
        name = name || err.name;
        this._host.error(name, message);
        this.show(screen !== undefined ? screen : 'welcome');
        if (known) {
            this._host.openURL(known.url);
        }
    }

    accept(file) {
        return this._modelFactoryService.accept(file);
    }

    open(context) {
        this._host.event('Model', 'Open', 'Size', context.stream.length);
        this._sidebar.close();
        return this._timeout(2).then(() => {
            return this._modelFactoryService.open(context).then((model) => {
                const format = model.format;
                if (format) {
                    this._host.event('Model', 'Format', format + (model.producer ? ' (' + model.producer + ')' : ''));
                }
                return this._timeout(20).then(() => {
                    const graph = model.graphs.length > 0 ? model.graphs[0] : null;
                    return this._updateGraph(model, graph);
                });
            });
        });
    }

    _updateActiveGraph(name) {
        this._sidebar.close();
        if (this._model) {
            const model = this._model;
            const graph = model.graphs.filter(graph => name === graph.name).shift();
            if (graph) {
                this.show('welcome spinner');
                this._timeout(200).then(() => {
                    return this._updateGraph(model, graph).catch((error) => {
                        if (error) {
                            this.error(error, 'Graph update failed.', 'welcome');
                        }
                    });
                });
            }
        }
    }

    _updateGraph(model, graph) {
        return this._timeout(100).then(() => {
            if (graph && graph != this._activeGraph) {
                const nodes = graph.nodes;
                if (nodes.length > 1400) {
                    if (!this._host.confirm('Large model detected.', 'This graph contains a large number of nodes and might take a long time to render. Do you want to continue?')) {
                        this._host.event('Graph', 'Render', 'Skip', nodes.length);
                        this.show(null);
                        return null;
                    }
                }
            }
            return this.renderGraph(model, graph).then(() => {
                this._model = model;
                this._activeGraph = graph;
                this.show('default');
                return this._model;
            }).catch((error) => {
                return this.renderGraph(this._model, this._activeGraph).then(() => {
                    this.show('default');
                    throw error;
                }).catch(() => {
                    throw error;
                });
            });
        });
    }

    renderGraph(model, graph) {
        try {
            const graphElement = this._getElementById('graph');
            const canvasElement = this._getElementById('canvas');
            while (canvasElement.lastChild) {
                canvasElement.removeChild(canvasElement.lastChild);
            }
            if (!graph) {
                return Promise.resolve();
            }
            else {
                switch (this._host.environment('zoom')) {
                    case 'scroll':
                        this._zoom = 1;
                        canvasElement.style.position = 'static';
                        canvasElement.style.margin = 'auto';
                        break;
                    case 'd3':
                        this._zoom = null;
                        canvasElement.style.position = 'absolute';
                        canvasElement.style.margin = '0';
                        break;
                }

                const groups = graph.groups;
                const nodes = graph.nodes;
                this._host.event('Graph', 'Render', 'Size', nodes.length);

                const options = {};
                options.nodesep = 25;
                options.ranksep = 20;
                const rotate = graph.nodes.every((node) => node.inputs.filter((input) => input.arguments.every((argument) => !argument.initializer)).length === 0 && node.outputs.length === 0);
                const showHorizontal = rotate ? !this._showHorizontal : this._showHorizontal;
                if (showHorizontal) {
                    options.rankdir = "LR";
                }
                if (nodes.length > 1500) {
                    options.ranker = 'longest-path';
                }

                const viewGraph = new view.Graph(this, groups, options);

                const clusters = new Set();
                const clusterParentMap = new Map();

                if (groups) {
                    for (const node of nodes) {
                        if (node.group) {
                            const path = node.group.split('/');
                            while (path.length > 0) {
                                const name = path.join('/');
                                path.pop();
                                clusterParentMap.set(name, path.join('/'));
                            }
                        }
                    }
                }

                for (const node of nodes) {

                    const viewNode = viewGraph.createNode(node);

                    const inputs = node.inputs;
                    for (const input of inputs) {
                        for (const argument of input.arguments) {
                            if (argument.name != '' && !argument.initializer) {
                                viewGraph.createArgument(argument).to(viewNode);
                            }
                        }
                    }
                    let outputs = node.outputs;
                    if (node.chain && node.chain.length > 0) {
                        const chainOutputs = node.chain[node.chain.length - 1].outputs;
                        if (chainOutputs.length > 0) {
                            outputs = chainOutputs;
                        }
                    }
                    for (const output of outputs) {
                        for (const argument of output.arguments) {
                            if (!argument) {
                                throw new view.Error("Invalid null argument in '" + model.format + "'.");
                            }
                            if (argument.name != '') {
                                viewGraph.createArgument(argument).from(viewNode);
                            }
                        }
                    }

                    if (node.controlDependencies && node.controlDependencies.length > 0) {
                        for (const name of node.controlDependencies) {
                            viewGraph.createArgument({ name: name, controlDependency: true }).to(viewNode);
                        }
                    }

                    const createCluster = function(name) {
                        if (!clusters.has(name)) {
                            viewGraph.setNode({ name: name, rx: 5, ry: 5});
                            clusters.add(name);
                            const parent = clusterParentMap.get(name);
                            if (parent) {
                                createCluster(parent);
                                viewGraph.setParent(name, parent);
                            }
                        }
                    };

                    if (groups) {
                        let groupName = node.group;
                        if (groupName && groupName.length > 0) {
                            if (!clusterParentMap.has(groupName)) {
                                const lastIndex = groupName.lastIndexOf('/');
                                if (lastIndex != -1) {
                                    groupName = groupName.substring(0, lastIndex);
                                    if (!clusterParentMap.has(groupName)) {
                                        groupName = null;
                                    }
                                }
                                else {
                                    groupName = null;
                                }
                            }
                            if (groupName) {
                                createCluster(groupName);
                                viewGraph.setParent(viewNode.name, groupName);
                            }
                        }
                    }
                }

                for (const input of graph.inputs) {
                    const viewInput = viewGraph.createInput(input);
                    for (const argument of input.arguments) {
                        viewGraph.createArgument(argument).from(viewInput);
                    }
                }

                for (const output of graph.outputs) {
                    const viewOutput = viewGraph.createOutput(output);
                    for (const argument of output.arguments) {
                        viewGraph.createArgument(argument).to(viewOutput);
                    }
                }

                // Workaround for Safari background drag/zoom issue:
                // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
                const backgroundElement = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                backgroundElement.setAttribute('id', 'background');
                if (this._host.environment('zoom') === 'd3') {
                    backgroundElement.setAttribute('width', '100%');
                    backgroundElement.setAttribute('height', '100%');
                }
                backgroundElement.setAttribute('fill', 'none');
                backgroundElement.setAttribute('pointer-events', 'all');
                canvasElement.appendChild(backgroundElement);

                const originElement = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'g');
                originElement.setAttribute('id', 'origin');
                canvasElement.appendChild(originElement);

                viewGraph.build(this._host.document, originElement);

                let svg = null;
                switch (this._host.environment('zoom')) {
                    case 'd3': {
                        svg = d3.select(canvasElement);
                        this._zoom = d3.zoom();
                        this._zoom(svg);
                        this._zoom.scaleExtent([ 0.1, 1.4 ]);
                        this._zoom.on('zoom', (event) => {
                            originElement.setAttribute('transform', event.transform.toString());
                        });
                        this._zoom.transform(svg, d3.zoomIdentity);
                        break;
                    }
                    case 'scroll': {
                        this._zoom = 1;
                        break;
                    }
                }

                return this._timeout(20).then(() => {

                    viewGraph.layout();

                    const elements = Array.from(canvasElement.getElementsByClassName('graph-input') || []);
                    if (elements.length === 0) {
                        const nodeElements = Array.from(canvasElement.getElementsByClassName('graph-node') || []);
                        if (nodeElements.length > 0) {
                            elements.push(nodeElements[0]);
                        }
                    }

                    switch (this._host.environment('zoom')) {
                        case 'd3': {
                            const svgSize = canvasElement.getBoundingClientRect();
                            if (elements && elements.length > 0) {
                                // Center view based on input elements
                                const xs = [];
                                const ys = [];
                                for (let i = 0; i < elements.length; i++) {
                                    const transform = elements[i].transform.baseVal.consolidate();
                                    if (transform) {
                                        xs.push(transform.matrix.e);
                                        ys.push(transform.matrix.f);
                                    }
                                }
                                let x = xs[0];
                                const y = ys[0];
                                if (ys.every(y => y === ys[0])) {
                                    x = xs.reduce((a,b) => { return a + b; }) / xs.length;
                                }
                                const sx = (svgSize.width / (this._showHorizontal ? 4 : 2)) - x;
                                const sy = (svgSize.height / (this._showHorizontal ? 2 : 4)) - y;
                                this._zoom.transform(svg, d3.zoomIdentity.translate(sx, sy));
                            }
                            else {
                                this._zoom.transform(svg, d3.zoomIdentity.translate((svgSize.width - viewGraph.graph().width) / 2, (svgSize.height - viewGraph.graph().height) / 2));
                            }
                            break;
                        }
                        case 'scroll': {
                            const size = canvasElement.getBBox();
                            const margin = 100;
                            const width = Math.ceil(margin + size.width + margin);
                            const height = Math.ceil(margin + size.height + margin);
                            originElement.setAttribute('transform', 'translate(' + margin.toString() + ', ' + margin.toString() + ') scale(1)');
                            backgroundElement.setAttribute('width', width);
                            backgroundElement.setAttribute('height', height);
                            this._width = width;
                            this._height = height;
                            this._zoom = 1;
                            delete this._scrollLeft;
                            delete this._scrollRight;
                            canvasElement.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
                            canvasElement.setAttribute('width', width);
                            canvasElement.setAttribute('height', height);

                            this._updateZoom(this._zoom);

                            if (elements && elements.length > 0) {
                                // Center view based on input elements
                                const xs = [];
                                const ys = [];
                                for (let i = 0; i < elements.length; i++) {
                                    const element = elements[i];
                                    const rect = element.getBoundingClientRect();
                                    xs.push(rect.left + (rect.width / 2));
                                    ys.push(rect.top + (rect.height / 2));
                                }
                                let x = xs[0];
                                const y = ys[0];
                                if (ys.every(y => y === ys[0])) {
                                    x = xs.reduce((a,b) => { return a + b; }) / xs.length;
                                }
                                // const canvasRect = graphElement.getBoundingClientRect();
                                const graphRect = graphElement.getBoundingClientRect();
                                // const sx = (canvasRect.width / (this._showHorizontal ? 4 : 2)) - x;
                                // const sy = (canvasRect.height / (this._showHorizontal ? 2 : 4)) - y;
                                const left = (graphElement.scrollLeft + x - graphRect.left) - (graphRect.width / 2);
                                const top = (graphElement.scrollTop + y - graphRect.top) - (graphRect.height / 2);
                                graphElement.scrollTo({ left: left, top: top, behavior: 'auto' });
                            }
                            else {
                                const canvasRect = graphElement.getBoundingClientRect();
                                const graphRect = graphElement.getBoundingClientRect();
                                const left = (graphElement.scrollLeft + (canvasRect.width / 2) - graphRect.left) - (graphRect.width / 2);
                                const top = (graphElement.scrollTop + (canvasRect.height / 2) - graphRect.top) - (graphRect.height / 2);
                                graphElement.scrollTo({ left: left, top: top, behavior: 'auto' });
                            }
                            break;
                        }
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
        let rules = [];
        for (const styleSheet of this._host.document.styleSheets) {
            if (styleSheet && styleSheet.href && styleSheet.href.endsWith('/' + name)) {
                rules = styleSheet.cssRules;
                break;
            }
        }
        const nodes = element.getElementsByTagName('*');
        for (const node of nodes) {
            for (const rule of rules) {
                if (node.matches(rule.selectorText)) {
                    for (const item of rule.style) {
                        node.style[item] = rule.style[item];
                    }
                }
            }
        }
    }

    export(file) {
        const lastIndex = file.lastIndexOf('.');
        const extension = (lastIndex != -1) ? file.substring(lastIndex + 1) : '';
        if (this._activeGraph && (extension === 'png' || extension === 'svg')) {
            const graphElement = this._getElementById('canvas');
            const exportElement = graphElement.cloneNode(true);
            this.applyStyleSheet(exportElement, 'view-grapher.css');
            exportElement.setAttribute('id', 'export');
            exportElement.removeAttribute('width');
            exportElement.removeAttribute('height');
            exportElement.style.removeProperty('opacity');
            exportElement.style.removeProperty('display');
            const backgroundElement = exportElement.querySelector('#background');
            const originElement = exportElement.querySelector('#origin');
            originElement.setAttribute('transform', 'translate(0,0) scale(1)');
            backgroundElement.removeAttribute('width');
            backgroundElement.removeAttribute('height');

            const parentElement = graphElement.parentElement;
            parentElement.insertBefore(exportElement, graphElement);
            const size = exportElement.getBBox();
            parentElement.removeChild(exportElement);
            parentElement.removeChild(graphElement);
            parentElement.appendChild(graphElement);

            const delta = (Math.min(size.width, size.height) / 2.0) * 0.1;
            const width = Math.ceil(delta + size.width + delta);
            const height = Math.ceil(delta + size.height + delta);
            originElement.setAttribute('transform', 'translate(' + delta.toString() + ', ' + delta.toString() + ') scale(1)');
            exportElement.setAttribute('width', width);
            exportElement.setAttribute('height', height);
            backgroundElement.setAttribute('width', width);
            backgroundElement.setAttribute('height', height);
            backgroundElement.setAttribute('fill', '#fff');

            const data = new XMLSerializer().serializeToString(exportElement);

            if (extension === 'svg') {
                const blob = new Blob([ data ], { type: 'image/svg' });
                this._host.export(file, blob);
            }

            if (extension === 'png') {
                const imageElement = new Image();
                imageElement.onload = () => {
                    const max = Math.max(width, height);
                    const scale = Math.min(24000.0 / max, 2.0);
                    const canvas = this._host.document.createElement('canvas');
                    canvas.width = Math.ceil(width * scale);
                    canvas.height = Math.ceil(height * scale);
                    const context = canvas.getContext('2d');
                    context.scale(scale, scale);
                    context.drawImage(imageElement, 0, 0);
                    canvas.toBlob((blob) => {
                        if (blob) {
                            this._host.export(file, blob);
                        }
                        else {
                            const err = new Error();
                            err.name = 'Error exporting image.';
                            err.message = 'Image may be too large to render as PNG.';
                            this._host.exception(err, false);
                            this._host.error(err.name, err.message);
                        }
                    }, 'image/png');
                };
                imageElement.src = 'data:image/svg+xml;base64,' + this._host.window.btoa(unescape(encodeURIComponent(data)));
            }
        }
    }

    showModelProperties() {
        if (this._model) {
            const modelSidebar = new sidebar.ModelSidebar(this._host, this._model, this._activeGraph);
            modelSidebar.on('update-active-graph', (sender, name) => {
                this._updateActiveGraph(name);
            });
            this._sidebar.open(modelSidebar.render(), 'Model Properties');
        }
    }

    showNodeProperties(node, input) {
        if (node) {
            const nodeSidebar = new sidebar.NodeSidebar(this._host, node);
            nodeSidebar.on('show-documentation', (/* sender, e */) => {
                this.showNodeDocumentation(node);
            });
            nodeSidebar.on('export-tensor', (sender, tensor) => {
                this._host.require('./numpy').then((numpy) => {
                    const defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
                    this._host.save('NumPy Array', 'npy', defaultPath, (file) => {
                        try {
                            const dataTypeMap = new Map([
                                [ 'float16', 'f2' ], [ 'float32', 'f4' ], [ 'float64', 'f8' ],
                                [ 'int8', 'i1' ], [ 'int16', 'i2'], [ 'int32', 'i4' ], [ 'int64', 'i8' ],
                                [ 'uint8', 'u1' ], [ 'uint16', 'u2' ], [ 'uint32', 'u4' ], [ 'uint64', 'u8' ],
                                [ 'qint8', 'i1' ], [ 'qint16', 'i2' ],
                                [ 'quint8', 'u1' ], [ 'quint16', 'u2' ]
                            ]);
                            const array = new numpy.Array();
                            array.shape = tensor.type.shape.dimensions;
                            array.data = tensor.value;
                            array.dataType = dataTypeMap.has(tensor.type.dataType) ? dataTypeMap.get(tensor.type.dataType) : tensor.type.dataType;
                            const blob = new Blob([ array.toBuffer() ], { type: 'application/octet-stream' });
                            this._host.export(file, blob);
                        }
                        catch (error) {
                            this.error(error, 'Error saving NumPy tensor.', null);
                        }
                    });
                }).catch(() => {
                });
            });
            nodeSidebar.on('error', (sender, error) => {
                if (this._model) {
                    error.message = error.message.replace(/\.$/, '') + " in format '" + this._model.format + "'.";
                }
                this.error(error, null, null);
            });
            if (input) {
                nodeSidebar.toggleInput(input.name);
            }
            this._sidebar.open(nodeSidebar.render(), 'Node Properties');
        }
    }

    showNodeDocumentation(node) {
        const metadata = node.metadata;
        if (metadata) {
            const documentationSidebar = new sidebar.DocumentationSidebar(this._host, metadata);
            documentationSidebar.on('navigate', (sender, e) => {
                this._host.openURL(e.link);
            });
            this._sidebar.push(documentationSidebar.render(), 'Documentation');
        }
    }
};

view.Graph = class extends grapher.Graph {

    constructor(view, compound, options) {
        super(compound);
        this.view = view;
        this._arguments = new Map();
        this._nodeKey = 0;
        this.setGraph(options);
    }

    createNode(node) {
        const value = new view.Node(this, node);
        value.name = this._nodeKey++;
        this.setNode(value);
        return value;
    }

    createInput(input) {
        const value = new view.Input(this, input);
        value.name = this._nodeKey++;
        this.setNode(value);
        return value;
    }

    createOutput(output) {
        const value = new view.Output(this, output);
        value.name = this._nodeKey++;
        this.setNode(value);
        return value;
    }

    createArgument(argument) {
        const name = argument.name;
        if (!this._arguments.has(name)) {
            this._arguments.set(name, new view.Argument(this, argument));
        }
        return this._arguments.get(name);
    }

    createEdge(from, to) {
        const value = new view.Edge(from, to);
        return value;
    }

    build(document, originElement) {

        for (const argument of this._arguments.values()) {
            argument.build();
        }

        super.build(document, originElement);
    }
};

view.Node = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        view.Node.counter = view.Node.counter || 0;
        this.id = 'node-' + (value.name ? 'name-' + value.name : 'id-' + (view.Node.counter++).toString());
        this._add(this.value);
    }

    get class() {
        return 'graph-node';
    }

    _add(node) {

        const header =  this.header();
        const styles = [ 'node-item-type' ];
        const metadata = node.metadata;
        const category = metadata && metadata.category ? metadata.category : '';
        if (category) {
            styles.push('node-item-type-' + category.toLowerCase());
        }
        const type = node.type;
        if (typeof type !== 'string' || !type.split) { // #416
            const format = this.context.view.model && this.context.view.model.format ? this.context.view.model.format : '?';
            throw new view.Error("Unknown node type '" + JSON.stringify(type) + "' in format '" + format + "'.");
        }
        const content = this.context.view.showNames && (node.name || node.location) ? (node.name || node.location) : type.split('.').pop();
        const tooltip = this.context.view.showNames && (node.name || node.location) ? type : (node.name || node.location);
        header.add(null, styles, content, tooltip, () => {
            this.context.view.showNodeProperties(node, null);
        });
        if (node.function) {
            header.add(null, [ 'node-item-function' ], '+', null, () => {
                // debugger;
            });
        }
        const initializers = [];
        let hiddenInitializers = false;
        if (this.context.view.showInitializers) {
            for (const input of node.inputs) {
                if (input.visible && input.arguments.length === 1 && input.arguments[0].initializer != null) {
                    initializers.push(input);
                }
                if ((!input.visible || input.arguments.length > 1) &&
                    input.arguments.some((argument) => argument.initializer != null)) {
                    hiddenInitializers = true;
                }
            }
        }
        let sortedAttributes = [];
        const attributes = node.attributes || [];
        if (this.context.view.showAttributes) {
            sortedAttributes = attributes.filter((attribute) => attribute.visible).slice();
        }
        sortedAttributes.sort((a, b) => {
            const au = a.name.toUpperCase();
            const bu = b.name.toUpperCase();
            return (au < bu) ? -1 : (au > bu) ? 1 : 0;
        });
        if (initializers.length > 0 || hiddenInitializers || sortedAttributes.length > 0) {
            const block = this.list();
            block.handler = () => {
                this.context.view.showNodeProperties(node);
            };
            for (const initializer of initializers) {
                const argument = initializer.arguments[0];
                const type = argument.type;
                let shape = '';
                let separator = '';
                if (type && type.shape && type.shape.dimensions && Array.isArray(type.shape.dimensions)) {
                    shape = '\u3008' + type.shape.dimensions.map((d) => d ? d : '?').join('\u00D7') + '\u3009';
                    if (type.shape.dimensions.length === 0 && argument.initializer && !argument.initializer.state) {
                        try {
                            shape = argument.initializer.toString();
                            if (shape && shape.length > 10) {
                                shape = shape.substring(0, 10) + '\u2026';
                            }
                            separator = ' = ';
                        }
                        catch (err) {
                            let type = '?';
                            try {
                                type = argument.initializer.type.toString();
                            }
                            catch (error) {
                                // continue regardless of error
                            }
                            const format = this.context.view.model && this.context.view.model.format ? this.context.view.model.format : '?';
                            throw new view.Error("Failed to render tensor of type '" + type + "' in format '" + format + "' (" + err.message + ").");
                        }
                    }
                }
                block.add(argument.name ? 'initializer-' + argument.name : '', initializer.name, shape, type ? type.toString() : '', separator);
            }
            if (hiddenInitializers) {
                block.add(null, '\u3008' + '\u2026' + '\u3009', '', null, '');
            }

            for (const attribute of sortedAttributes) {
                if (attribute.visible) {
                    let value = sidebar.NodeSidebar.formatAttributeValue(attribute.value, attribute.type);
                    if (value && value.length > 25) {
                        value = value.substring(0, 25) + '\u2026';
                    }
                    block.add(null, attribute.name, value, attribute.type, ' = ');
                }
            }
        }
        if (Array.isArray(node.chain) && node.chain.length > 0) {
            for (const innerNode of node.chain) {
                this._add(innerNode);
            }
        }
        if (node.inner) {
            this._add(node.inner);
        }
    }
};

view.Input = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        view.Input.counter = view.Input.counter || 0;
        const types = value.arguments.map((argument) => argument.type || '').join('\n');
        let name = value.name || '';
        if (name.length > 16) {
            name = name.split('/').pop();
        }
        const header = this.header();
        header.add(null, [ 'graph-item-input' ], name, types, () => this.context.view.showModelProperties());
        this.id = 'input-' + (name ? 'name-' + name : 'id-' + (view.Input.counter++).toString());
    }

    get class() {
        return 'graph-input';
    }
};

view.Output = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        const types = value.arguments.map((argument) => argument.type || '').join('\n');
        let name = value.name || '';
        if (name.length > 16) {
            name = name.split('/').pop();
        }
        const header = this.header();
        header.add(null, [ 'graph-item-output' ], name, types, () => this.context.view.showModelProperties());
    }
};

view.Argument = class {

    constructor(context, argument) {
        this.context = context;
        this._argument = argument;
    }

    from(node) {
        this._from = node;
    }

    to(node) {
        this._to = this._to || [];
        this._to.push(node);
    }

    build() {
        this._edges = this._edges || [];
        if (this._from && this._to) {
            for (const to of this._to) {
                let text = '';
                const type = this._argument.type;
                if (type && type.shape && type.shape.dimensions && type.shape.dimensions.length > 0) {
                    text = type.shape.dimensions.map((dimension) => dimension || '?').join('\u00D7');
                }
                if (this.context.view.showNames) {
                    text = this._argument.name.split('\n').shift(); // custom argument id
                }
                const edge = this.context.createEdge(this._from, to);
                edge.v = this._from.name;
                edge.w = to.name;
                edge.label = text;
                edge.id = 'edge-' + this._argument.name;
                if (this._argument.controlDependency) {
                    edge.class = 'edge-path-control-dependency';
                }
                this.context.setEdge(edge);
                this._edges.push(edge);
            }
        }
    }
};

view.Edge = class extends grapher.Edge {

    constructor(from, to) {
        super(from, to);
    }
};

view.ModelContext = class {

    constructor(context, entries) {
        this._context = context;
        this._tags = new Map();
        this._content = new Map();
        this._entries = entries || new Map();
    }

    get identifier() {
        return this._context.identifier;
    }

    get stream() {
        return this._context.stream;
    }

    request(file, encoding, base) {
        return this._context.request(file, encoding, base);
    }

    require(id) {
        return this._context.require(id);
    }

    exception(error, fatal) {
        this._context.exception(error, fatal);
    }

    entries(format) {
        return this._entries.get(format) || [];
    }

    open(type) {
        if (!this._content.has(type)) {
            this._content.set(type, undefined);
            let reset = false;
            switch (type) {
                case 'json': {
                    try {
                        reset = true;
                        const reader = json.TextReader.create(this.stream.peek());
                        const obj = reader.read();
                        this._content.set(type, obj);
                    }
                    catch (err) {
                        // continue regardless of error
                    }
                    break;
                }
                case 'pkl': {
                    try {
                        if (this.stream.length > 2) {
                            const stream = this.stream.peek(1)[0] === 0x78 ? zip.Archive.open(this.stream).entries[0].stream : this.stream;
                            const match = (stream) => {
                                const head = stream.peek(2);
                                if (head[0] === 0x80 && head[1] < 7) {
                                    return true;
                                }
                                stream.seek(-1);
                                const tail = stream.peek(1);
                                stream.seek(0);
                                if (tail[0] === 0x2e) {
                                    return true;
                                }
                                return false;
                            };
                            if (match(stream)) {
                                reset = true;
                                const unpickler = new python.Unpickler(stream);
                                const execution = new python.Execution(null, (error, fatal) => {
                                    const message = error && error.message ? error.message : error.toString();
                                    this.exception(new view.Error(message.replace(/\.$/, '') + " in '" + this.identifier + "'."), fatal);
                                });
                                const obj = unpickler.load((name, args) => execution.invoke(name, args));
                                this._content.set(type, obj);
                            }
                        }
                    }
                    catch (err) {
                        // continue regardless of error
                    }
                    break;
                }
            }
            if (reset) {
                this.stream.seek(0);
            }
        }
        return this._content.get(type);
    }

    tags(type) {
        let tags = this._tags.get(type);
        if (!tags) {
            tags = new Map();
            let reset = false;
            const signatures = [
                // Reject PyTorch models
                [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ],
                // Reject TorchScript models
                [ 0x50, 0x4b ]
            ];
            const stream = this.stream;
            if (!signatures.some((signature) => signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value))) {
                try {
                    switch (type) {
                        case 'pbtxt': {
                            reset = true;
                            const buffer = stream.peek();
                            const decoder = base.TextDecoder.create(buffer);
                            let count = 0;
                            for (let i = 0; i < 0x100; i++) {
                                const c = decoder.decode();
                                switch (c) {
                                    case '\n': case '\r': case '\t': case '\0': break;
                                    case undefined: i = 0x100; break;
                                    default: count += c < ' ' ? 1 : 0; break;
                                }
                            }
                            if (count < 4) {
                                const buffer = stream.peek();
                                const reader = protobuf.TextReader.create(buffer);
                                reader.start(false);
                                while (!reader.end(false)) {
                                    const tag = reader.tag();
                                    tags.set(tag, true);
                                    if (reader.token() === '{') {
                                        reader.start();
                                        while (!reader.end()) {
                                            const subtag = reader.tag();
                                            tags.set(tag + '.' + subtag, true);
                                            reader.skip();
                                            reader.match(',');
                                        }
                                    }
                                    else {
                                        reader.skip();
                                    }
                                }
                            }
                            break;
                        }
                        case 'pb': {
                            reset = true;
                            const buffer = stream.peek();
                            const reader = protobuf.Reader.create(buffer);
                            const length = reader.length;
                            while (reader.position < length) {
                                const tag = reader.uint32();
                                const number = tag >>> 3;
                                const type = tag & 7;
                                if (type > 5 || number === 0) {
                                    tags = new Map();
                                    break;
                                }
                                tags.set(number, type);
                                try {
                                    reader.skipType(type);
                                }
                                catch (err) {
                                    tags = new Map();
                                    break;
                                }
                            }
                            break;
                        }
                    }
                }
                catch (error) {
                    tags = new Map();
                }
            }
            if (reset) {
                this.stream.seek(0);
            }
            this._tags.set(type, tags);
        }
        return tags;
    }
};

view.ArchiveContext = class {

    constructor(host, entries, rootFolder, identifier, stream) {
        this._host = host;
        this._entries = {};
        if (entries) {
            for (const entry of entries) {
                if (entry.name.startsWith(rootFolder)) {
                    const name = entry.name.substring(rootFolder.length);
                    if (name.length > 0 && (name.indexOf('/') === -1 || name.startsWith('MAR-INF/'))) {
                        this._entries[name] = entry;
                    }
                }
            }
        }
        this._identifier = identifier.substring(rootFolder.length);
        this._stream = stream;
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    request(file, encoding, base) {
        if (base === undefined) {
            const entry = this._entries[file];
            if (!entry) {
                return Promise.reject(new Error('File not found.'));
            }
            return Promise.resolve(encoding ? new TextDecoder(encoding).decode(entry.data) : entry.stream);
        }
        return this._host.request(file, encoding, base);
    }

    require(id) {
        return this._host.require(id);
    }

    exception(error, fatal) {
        this._host.exception(error, fatal);
    }
};

view.ArchiveError = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading archive.';
    }
};

view.ModelFactoryService = class {

    constructor(host) {
        this._host = host;
        this._extensions = [];
        this.register('./pytorch', [ '.pt', '.pth', '.pt1', '.pyt', '.pkl', '.h5', '.t7', '.model', '.dms', '.tar', '.ckpt', '.chkpt', '.tckpt', '.bin', '.pb', '.zip', '.nn', '.torchmodel' ]);
        this.register('./onnx', [ '.onnx', '.pb', '.pbtxt', '.prototxt', '.model', '.pt', '.pth', '.pkl' ]);
        this.register('./mxnet', [ '.json', '.params' ]);
        this.register('./coreml', [ '.mlmodel' ]);
        this.register('./caffe', [ '.caffemodel', '.pbtxt', '.prototxt', '.pt', '.txt' ]);
        this.register('./caffe2', [ '.pb', '.pbtxt', '.prototxt' ]);
        this.register('./torch', [ '.t7' ]);
        this.register('./tflite', [ '.tflite', '.lite', '.tfl', '.bin', '.pb', '.tmfile', '.h5', '.model', '.json' ]);
        this.register('./tf', [ '.pb', '.meta', '.pbtxt', '.prototxt', '.pt', '.json', '.index', '.ckpt', '.graphdef', /.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/, /^events.out.tfevents./ ]);
        this.register('./mediapipe', [ '.pbtxt' ]);
        this.register('./uff', [ '.uff', '.pb', '.pbtxt', '.uff.txt', '.trt', '.engine' ]);
        this.register('./tensorrt', [ '.trt', '.engine', '.model', '.txt', '.uff', '.pb', '.tmfile', '.onnx' ]);
        this.register('./npz', [ '.npz', '.npy', '.pkl' ]);
        this.register('./lasagne', [ '.pkl', '.pickle', '.joblib', '.model', '.pkl.z', '.joblib.z' ]);
        this.register('./lightgbm', [ '.txt', '.pkl', '.model' ]);
        this.register('./sklearn', [ '.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z' ]);
        this.register('./pickle', [ '.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z' ]);
        this.register('./cntk', [ '.model', '.cntk', '.cmf', '.dnn' ]);
        this.register('./paddle', [ '.pdmodel', '.pdparams', '.paddle', '__model__', '.pbtxt', '.txt', '.tar', '.tar.gz' ]);
        this.register('./bigdl', [ '.model', '.bigdl' ]);
        this.register('./darknet', [ '.cfg', '.model', '.txt', '.weights' ]);
        this.register('./weka', [ '.model' ]);
        this.register('./rknn', [ '.rknn', '.onnx' ]);
        this.register('./dlc', [ '.dlc' ]);
        this.register('./keras', [ '.h5', '.hd5', '.hdf5', '.keras', '.json', '.cfg', '.model', '.pb', '.pth', '.weights', '.pkl', '.lite', '.tflite', '.ckpt' ]);
        this.register('./armnn', [ '.armnn', '.json' ]);
        this.register('./mnn', ['.mnn']);
        this.register('./ncnn', [ '.param', '.bin', '.cfg.ncnn', '.weights.ncnn' ]);
        this.register('./tnn', [ '.tnnproto', '.tnnmodel' ]);
        this.register('./tengine', ['.tmfile']);
        this.register('./mslite', [ '.ms']);
        this.register('./barracuda', [ '.nn' ]);
        this.register('./dnn', [ '.dnn' ]);
        this.register('./xmodel', [ '.xmodel' ]);
        this.register('./openvino', [ '.xml', '.bin' ]);
        this.register('./flux', [ '.bson' ]);
        this.register('./dl4j', [ '.zip' ]);
        this.register('./mlnet', [ '.zip' ]);
        this.register('./acuity', [ '.json' ]);
    }

    register(id, extensions) {
        for (const extension of extensions) {
            this._extensions.push({ extension: extension, id: id });
        }
    }

    open(context) {
        return this._openSignature(context).then((context) => {
            const entries = this._openArchive(context);
            const modelContext = new view.ModelContext(context, entries);
            return this._openContext(modelContext).then((model) => {
                if (model) {
                    return model;
                }
                if (entries.size > 0) {
                    return this._openEntries(entries.values().next().value).then((context) => {
                        if (context) {
                            return this._openContext(context);
                        }
                        this._unsupported(modelContext);
                    });
                }
                this._unsupported(modelContext);
            });
        });
    }

    _unsupported(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        for (const format of new Map([ [ 'Zip', zip ], [ 'tar', tar ] ])) {
            const name = format[0];
            const module = format[1];
            let archive = null;
            try {
                archive = module.Archive.open(context.stream);
            }
            catch (error) {
                // continue regardless of error
            }
            if (archive) {
                throw new view.Error("Invalid file content. File contains " + name + " archive in '" + identifier + "'.", true);
            }
        }
        const knownUnsupportedIdentifiers = new Set([
            'natives_blob.bin',
            'v8_context_snapshot.bin',
            'snapshot_blob.bin',
            'image_net_labels.json',
            'package.json',
            'models.json',
            'LICENSE.meta',
            'input_0.pb',
            'output_0.pb'
        ]);
        const skip = knownUnsupportedIdentifiers.has(identifier);
        const encodings = [
            {
                type: 'pb',
                name: 'Protocol Buffers',
                formats: []
            },
            {
                type: 'pbtxt',
                name: 'Protocol Buffers text',
                formats: [
                    { name: 'ImageNet LabelMap data', tags: [ 'entry', 'entry.target_class' ] },
                    { name: 'StringIntLabelMapProto data', tags: [ 'item', 'item.id', 'item.name' ] },
                    { name: 'caffe.LabelMap data', tags: [ 'item', 'item.name', 'item.label' ] },
                    { name: 'Triton Inference Server configuration', tags: [ 'name', 'platform', 'input', 'output' ] },
                    { name: 'TensorFlow OpList data', tags: [ 'op', 'op.name', 'op.input_arg' ] }
                ]
            },
            {
                type: 'json',
                name: 'JSON',
                formats: [
                    { name: 'Netron metadata', tags: [ '[].name', '[].schema' ] },
                    { name: 'Netron metadata', tags: [ '[].name', '[].attributes' ] },
                    { name: 'Darkflow metadata', tags: [ 'net', 'type', 'model' ] },
                    { name: 'keras-yolo2 configuration', tags: [ 'model', 'train', 'valid' ] },
                    { name: 'Vulkan SwiftShader ICD manifest', tags: [ 'file_format_version', 'ICD' ] },
                    { name: 'DeepLearningExamples configuration', tags: [ 'attention_probs_dropout_prob', 'hidden_act', 'hidden_dropout_prob', 'hidden_size', ] },
                    { name: 'NuGet assets', tags: [ 'version', 'targets', 'packageFolders' ] },
                    { name: 'NuGet data', tags: [ 'format', 'restore', 'projects' ] },
                    { name: 'NPM package', tags: [ 'name', 'version', 'dependencies' ] }
                ]
            }
        ];
        for (const encoding of encodings) {
            const tags = context.tags(encoding.type);
            if (tags.size > 0) {
                for (const format of encoding.formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
                const entries = [];
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') === -1));
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') !== -1));
                const content = entries.map((pair) => pair[1] === true ? pair[0] : pair[0] + ':' + JSON.stringify(pair[1])).join(',');
                throw new view.Error("Unsupported " + encoding.name + " content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "' in '" + identifier + "'.", !skip);
            }
            const obj = context.open(encoding.type);
            if (obj) {
                const match = (obj, tag) => {
                    if (tag.startsWith('[].')) {
                        tag = tag.substring(3);
                        return (Array.isArray(obj) && obj.some((item) => Object.prototype.hasOwnProperty.call(item, tag)));
                    }
                    return Object.prototype.hasOwnProperty.call(obj, tag);
                };
                for (const format of encoding.formats) {
                    if (format.tags.every((tag) => match(obj, tag))) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
                const content = JSON.stringify(obj).substring(0, 100).replace(/\s/, '').substr(0, 48) + '...';
                throw new view.Error("Unsupported " + encoding.name + " content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "' in '" + identifier + "'.", !skip);
            }
        }
        const stream = context.stream;
        stream.seek(0);
        const buffer = stream.peek(Math.min(16, stream.length));
        const bytes = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
        const content = stream.length > 268435456 ? '(' + bytes + ') [' + stream.length.toString() + ']': '(' + bytes + ')';
        throw new view.Error("Unsupported file content " + content + " for extension '." + extension + "' in '" + identifier + "'.", !skip);
    }

    _openArchive(context) {
        const entries = new Map();
        let stream = context.stream;
        let extension;
        let identifier = context.identifier;
        try {
            extension = identifier.split('.').pop().toLowerCase();
            const gzipArchive = gzip.Archive.open(stream);
            if (gzipArchive) {
                const entries = gzipArchive.entries;
                if (entries.length === 1) {
                    const entry = entries[0];
                    if (entry.name) {
                        identifier = entry.name;
                    }
                    else {
                        identifier = identifier.substring(0, identifier.lastIndexOf('.'));
                        switch (extension) {
                            case 'tgz':
                            case 'tar': {
                                if (identifier.split('.').pop().toLowerCase() !== 'tar') {
                                    identifier += '.tar';
                                }
                                break;
                            }
                        }
                    }
                    stream = entry.stream;
                }
            }
        }
        catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new view.ArchiveError(message.replace(/\.$/, '') + " in '" + identifier + "'.");
        }

        try {
            const formats = new Map([ [ 'zip', zip ], [ 'tar', tar ] ]);
            for (const pair of formats) {
                const format = pair[0];
                const module = pair[1];
                const archive = module.Archive.open(stream);
                if (archive) {
                    entries.set(format, archive.entries);
                    break;
                }
            }
        }
        catch (error) {
            const message = error && error.message ? error.message : error.toString();
            throw new view.ArchiveError(message.replace(/\.$/, '') + " in '" + identifier + "'.");
        }
        return entries;
    }

    _openContext(context) {
        const modules = this._filter(context).filter((module) => module && module.length > 0);
        const errors = [];
        let match = false;
        const nextModule = () => {
            if (modules.length > 0) {
                const id = modules.shift();
                return this._host.require(id).then((module) => {
                    if (!module.ModelFactory) {
                        throw new view.Error("Failed to load module '" + id + "'.");
                    }
                    const modelFactory = new module.ModelFactory();
                    if (!modelFactory.match(context)) {
                        return nextModule();
                    }
                    match = true;
                    return modelFactory.open(context).then((model) => {
                        return model;
                    }).catch((error) => {
                        const text = " in '" + context.identifier + "'.";
                        if (error && !error.message.endsWith(text) && (error.context === undefined || error.context === true)) {
                            error.message = error.message.replace(/\.$/, '') + text;
                        }
                        errors.push(error);
                        return nextModule();
                    });
                });
            }
            else {
                if (match) {
                    if (errors.length === 1) {
                        return Promise.reject(errors[0]);
                    }
                    return Promise.reject(new view.Error(errors.map((err) => err.message).join('\n')));
                }
                return Promise.resolve(null);
            }
        };
        return nextModule();
    }

    _openEntries(entries) {
        try {
            const rootFolder = (files) => {
                const map = files.map((file) => file.split('/').slice(0, -1));
                const at = index => list => list[index];
                const rotate = list => list.length === 0 ? [] : list[0].map((item, index) => list.map(at(index)));
                const equals = list => list.every((item) => item === list[0]);
                const folder = rotate(map).filter(equals).map(at(0)).join('/');
                return folder.length === 0 ? folder : folder + '/';
            };
            const files = entries.filter((entry) => {
                if (entry.name.endsWith('/')) {
                    return false;
                }
                if (entry.name.split('/').pop().startsWith('.')) {
                    return false;
                }
                if (!entry.name.startsWith('./') && entry.name.startsWith('.')) {
                    return false;
                }
                return true;
            });
            const folder = rootFolder(files.map((entry) => entry.name));
            let matches = [];
            const queue = files.slice(0).filter((entry) => entry.name.substring(folder.length).indexOf('/') < 0);
            const nextEntry = () => {
                if (queue.length > 0) {
                    const entry = queue.shift();
                    const context = new view.ModelContext(new view.ArchiveContext(this._host, null, folder, entry.name, entry.stream));
                    let modules = this._filter(context);
                    const nextModule = () => {
                        if (modules.length > 0) {
                            const id = modules.shift();
                            return this._host.require(id).then((module) => {
                                if (!module.ModelFactory) {
                                    throw new view.ArchiveError("Failed to load module '" + id + "'.", null);
                                }
                                const factory = new module.ModelFactory();
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
                else {
                    if (matches.length === 0) {
                        return Promise.resolve(null);
                    }
                    // MXNet
                    if (matches.length === 2 &&
                        matches.some((e) => e.name.toLowerCase().endsWith('.params')) &&
                        matches.some((e) => e.name.toLowerCase().endsWith('-symbol.json'))) {
                        matches = matches.filter((e) => e.name.toLowerCase().endsWith('.params'));
                    }
                    // TensorFlow.js
                    if (matches.length > 0 &&
                        matches.some((e) => e.name.toLowerCase().endsWith('.bin')) &&
                        matches.some((e) => e.name.toLowerCase().endsWith('.json'))) {
                        matches = matches.filter((e) => e.name.toLowerCase().endsWith('.json'));
                    }
                    // TensorFlow Bundle
                    if (matches.length > 1 &&
                        matches.some((e) => e.name.toLowerCase().endsWith('.data-00000-of-00001'))) {
                        matches = matches.filter((e) => !e.name.toLowerCase().endsWith('.data-00000-of-00001'));
                    }
                    if (matches.length > 1) {
                        return Promise.reject(new view.ArchiveError('Archive contains multiple model files.'));
                    }
                    const match = matches.shift();
                    return Promise.resolve(new view.ModelContext(new view.ArchiveContext(this._host, entries, folder, match.name, match.stream)));
                }
            };
            return nextEntry();
        }
        catch (error) {
            return Promise.reject(new view.ArchiveError(error.message));
        }
    }

    accept(identifier) {
        const extension = identifier.indexOf('.') === -1 ? '' : identifier.split('.').pop().toLowerCase();
        identifier = identifier.toLowerCase().split('/').pop();
        for (const entry of this._extensions) {
            if ((typeof entry.extension === 'string' && identifier.endsWith(entry.extension)) ||
                (entry.extension instanceof RegExp && entry.extension.exec(identifier))) {
                this._host.event('File', 'Accept', extension, 1);
                return true;
            }
        }
        if (identifier.endsWith('.zip') ||
            identifier.endsWith('.tar') ||
            identifier.endsWith('.tar.gz') ||
            identifier.endsWith('.tgz') ||
            identifier.endsWith('.mar') ||
            identifier.endsWith('.model')) {
            this._host.event('File', 'Accept', extension, 1);
            return true;
        }
        this._host.event('File', 'Reject', extension, 1);
        return false;
    }

    _filter(context) {
        const identifier = context.identifier.toLowerCase().split('/').pop();
        const list = this._extensions.filter((entry) =>
            (typeof entry.extension === 'string' && identifier.endsWith(entry.extension)) ||
            (entry.extension instanceof RegExp && entry.extension.exec(identifier)));
        return Array.from(new Set(list.map((entry) => entry.id)));
    }

    _openSignature(context) {
        const stream = context.stream;
        let empty = true;
        let position = 0;
        while (empty && position < stream.length) {
            const buffer = stream.read(Math.min(4096, stream.length - position));
            position += buffer.length;
            if (!buffer.every((value) => value === 0x00)) {
                empty = false;
                break;
            }
        }
        stream.seek(0);
        if (empty) {
            return Promise.reject(new view.Error('File has no content.', true));
        }
        /* eslint-disable no-control-regex */
        const entries = [
            { name: 'ELF executable', value: /^\x7FELF/ },
            { name: 'PNG image', value: /^\x89PNG/ },
            { name: 'Git LFS header', value: /^version https:\/\/git-lfs.github.com/ },
            { name: 'Git LFS header', value: /^\s*oid sha256:/ },
            { name: 'HTML markup', value: /^\s*<html>/ },
            { name: 'HTML markup', value: /^\s*<!doctype\s*html>/ },
            { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*html>/ },
            { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*HTML>/ },
            { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*HTML\s+(PUBLIC|SYSTEM)?/ },
            { name: 'Unity metadata', value: /^fileFormatVersion:/ },
            { name: 'Python source code', value: /^\s*import[ ]+(os|sys|types|torch|argparse|onnx|numpy|tensorflow)(,|;|\s)/ },
            { name: 'Python source code', value: /^\s*import[ ]+([a-z])+[ ]+as[ ]+/ },
            { name: 'Python source code', value: /^\s*from[ ]+(torch)[ ]+import[ ]+/ },
            { name: 'TSD header', value: /^%TSD-Header-###%/ },
            { name: "TensorFlow Hub module", value: /^\x08\x03$/, identifier: 'tfhub_module.pb' }
        ];
        /* eslint-enable no-control-regex */
        const buffer = stream.peek(Math.min(4096, stream.length));
        const text = String.fromCharCode.apply(null, buffer);
        for (const entry of entries) {
            if (text.match(entry.value) && (!entry.identifier || entry.identifier === context.identifier)) {
                return Promise.reject(new view.Error("Invalid file content. File contains " + entry.name + ".", true));
            }
        }
        return Promise.resolve(context);
    }
};

view.Error = class extends Error {

    constructor(message, telemetry) {
        super(message);
        this.name = 'Error loading model.';
        this.telemetry = telemetry;
        this.stack = undefined;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.View = view.View;
    module.exports.ModelFactoryService = view.ModelFactoryService;
}
