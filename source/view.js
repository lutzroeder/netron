
var view =  {};
var base = require('./base');
var zip = require('./zip');
var tar = require('./tar');
var json = require('./json');
var xml = require('./xml');
var protobuf = require('./protobuf');
var flatbuffers = require('./flatbuffers');
var hdf5 = require('./hdf5');
var python = require('./python');
var grapher = require('./grapher');

view.View = class {

    constructor(host, id) {
        this._host = host;
        this._id = id ? ('-' + id) : '';
        this._options = {
            initializers: true,
            attributes: false,
            names: false,
            direction: 'vertical',
            mousewheel: 'scroll'
        };
        this._host.view(this).then(() => {
            this._model = null;
            this._graphs = [];
            this._selection = [];
            this._sidebar = new view.Sidebar(this._host, id);
            this._searchText = '';
            this._modelFactoryService = new view.ModelFactoryService(this._host);
            this._getElementById('sidebar-button').addEventListener('click', () => {
                this.showModelProperties();
            });
            this._getElementById('zoom-in-button').addEventListener('click', () => {
                this.zoomIn();
            });
            this._getElementById('zoom-out-button').addEventListener('click', () => {
                this.zoomOut();
            });
            this._getElementById('back-button').addEventListener('click', () => {
                this.popGraph();
            });
            this._getElementById('name-button').addEventListener('click', () => {
                this.showDocumentation(this.activeGraph);
            });
            this._getElementById('sidebar').addEventListener('mousewheel', (e) => {
                if (e.shiftKey || e.ctrlKey) {
                    e.preventDefault();
                }
            }, { passive: true });
            this._host.document.addEventListener('keydown', () => {
                this.clearSelection();
            });
            if (this._host.environment('menu')) {
                this._menu = new view.Menu(this._host, 'menu-button', 'menu');
                this._menu.add({
                    label: 'Properties...',
                    accelerator: 'CmdOrCtrl+Enter',
                    click: () => this.showModelProperties(),
                    enabled: () => this.activeGraph
                });
                this._menu.add({});
                this._menu.add({
                    label: 'Find...',
                    accelerator: 'CmdOrCtrl+F',
                    click: () => this.find(),
                    enabled: () => this.activeGraph
                });
                this._menu.add({});
                this._menu.add({
                    label: () => this.options.attributes ? 'Hide Attributes' : 'Show Attributes',
                    accelerator: 'CmdOrCtrl+D',
                    click: () => this.toggle('attributes'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: () => this.options.initializers ? 'Hide Initializers' : 'Show Initializers',
                    accelerator: 'CmdOrCtrl+I',
                    click: () => this.toggle('initializers'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: () => this.options.names ? 'Hide Names' : 'Show Names',
                    accelerator: 'CmdOrCtrl+U',
                    click: () => this.toggle('names'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: () => this.options.direction === 'vertical' ? 'Show Horizontal' : 'Show Vertical',
                    accelerator: 'CmdOrCtrl+K',
                    click: () => this.toggle('direction'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: () => this.options.mousewheel === 'scroll' ? 'Mouse Wheel: Zoom' : 'Mouse Wheel: Scroll',
                    accelerator: 'CmdOrCtrl+M',
                    click: () => this.toggle('mousewheel'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({});
                this._menu.add({
                    label: 'Zoom In',
                    accelerator: 'Shift+Up',
                    click: () => this.zoomIn(),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: 'Zoom Out',
                    accelerator: 'Shift+Down',
                    click: () => this.zoomOut(),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: 'Actual Size',
                    accelerator: 'Shift+Backspace',
                    click: () => this.resetZoom(),
                    enabled: () => this.activeGraph
                });
                this._menu.add({});
                this._menu.add({
                    label: 'Export as PNG',
                    accelerator: 'CmdOrCtrl+Shift+E',
                    click: () => this.export(this._host.document.title + '.png'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({
                    label: 'Export as SVG',
                    accelerator: 'CmdOrCtrl+Alt+E',
                    click: () => this.export(this._host.document.title + '.svg'),
                    enabled: () => this.activeGraph
                });
                this._menu.add({});
                this._menu.add({
                    label: 'About ' + this._host.document.title,
                    click: () => this.about()
                });
                this._getElementById('menu-button').addEventListener('click', (e) => {
                    this._menu.toggle();
                    e.preventDefault();
                });
            }
            this._host.start();
        }).catch((err) => {
            this.error(err, null, null);
        });
    }

    show(page) {
        if (!page) {
            page = (!this._model && !this.activeGraph) ? 'welcome' : 'default';
        }
        this._host.event('screen_view', {
            screen_name: page,
        });
        if (this._sidebar) {
            this._sidebar.close();
        }
        this._host.document.body.classList.remove(...Array.from(this._host.document.body.classList).filter((_) => _ !== 'active'));
        this._host.document.body.classList.add(...page.split(' '));
        if (page === 'default') {
            this._activate();
        }
        else {
            this._deactivate();
        }
        if (page === 'welcome') {
            const element = this._getElementById('open-file-button');
            if (element) {
                element.focus();
            }
        }
        this._page = page;
    }

    progress(percent) {
        const bar = this._getElementById('progress-bar');
        if (bar) {
            bar.style.width = percent.toString() + '%';
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
        if (this._graph) {
            this.clearSelection();
            const graphElement = this._getElementById('canvas');
            const content = new view.FindSidebar(this._host, graphElement, this._graph);
            content.on('search-text-changed', (sender, text) => {
                this._searchText = text;
            });
            content.on('select', (sender, selection) => {
                this.select(selection);
            });
            this._sidebar.open(content.content, 'Find');
            content.focus(this._searchText);
        }
    }

    get model() {
        return this._model;
    }

    get options() {
        return this._options;
    }

    toggle(name) {
        switch (name) {
            case 'names':
            case 'attributes':
            case 'initializers':
                this._options[name] = !this._options[name];
                this._reload();
                break;
            case 'direction':
                this._options.direction = this._options.direction === 'vertical' ? 'horizontal' : 'vertical';
                this._reload();
                break;
            case 'mousewheel':
                this._options.mousewheel = this._options.mousewheel === 'scroll' ? 'zoom' : 'scroll';
                break;
            default:
                throw new view.Error("Unsupported toogle '" + name + "'.");
        }
    }

    _reload() {
        this.show('welcome spinner');
        if (this._model && this._graphs.length > 0) {
            this._updateGraph(this._model, this._graphs).catch((error) => {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            });
        }
    }

    _timeout(time) {
        return new Promise((resolve) => {
            setTimeout(() => {
                resolve();
            }, time);
        });
    }

    _getElementById(id) {
        return this._host.document.getElementById(id + this._id);
    }

    zoomIn() {
        this._updateZoom(this._zoom * 1.1);
    }

    zoomOut() {
        this._updateZoom(this._zoom * 0.9);
    }

    resetZoom() {
        this._updateZoom(1);
    }

    _activate() {
        if (!this._events) {
            this._events = {};
            this._events.scroll = (e) => this._scrollHandler(e);
            this._events.wheel = (e) => this._wheelHandler(e);
            this._events.mousedown = (e) => this._mouseDownHandler(e);
            this._events.gesturestart = (e) => this._gestureStartHandler(e);
            this._events.touchstart = (e) => this._touchStartHandler(e);
        }
        const graph = this._getElementById('graph');
        if (graph) {
            graph.focus();
        }
        graph.addEventListener('scroll', this._events.scroll);
        graph.addEventListener('wheel', this._events.wheel, { passive: false });
        graph.addEventListener('mousedown', this._events.mousedown);
        if (this._host.agent === 'safari') {
            graph.addEventListener('gesturestart', this._events.gesturestart, false);
        }
        else {
            graph.addEventListener('touchstart', this._events.touchstart, { passive: true });
        }
    }

    _deactivate() {
        if (this._events) {
            const graph = this._getElementById('graph');
            graph.removeEventListener('scroll', this._events.scroll);
            graph.removeEventListener('wheel', this._events.wheel);
            graph.removeEventListener('mousedown', this._events.mousedown);
            graph.removeEventListener('gesturestart', this._events.gesturestart);
            graph.removeEventListener('touchstart', this._events.touchstart);
        }
    }

    _updateZoom(zoom, e) {
        const container = this._getElementById('graph');
        const canvas = this._getElementById('canvas');
        const limit = this._options.direction === 'vertical' ?
            container.clientHeight / this._height :
            container.clientWidth / this._width;
        const min = Math.min(Math.max(limit, 0.15), 1);
        zoom = Math.max(min, Math.min(zoom, 1.4));
        const scrollLeft = this._scrollLeft || container.scrollLeft;
        const scrollTop = this._scrollTop || container.scrollTop;
        const x = (e ? e.pageX : (container.clientWidth / 2)) + scrollLeft;
        const y = (e ? e.pageY : (container.clientHeight / 2)) + scrollTop;
        const width = zoom * this._width;
        const height = zoom * this._height;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        this._scrollLeft = Math.max(0, ((x * zoom) / this._zoom) - (x - scrollLeft));
        this._scrollTop = Math.max(0, ((y * zoom) / this._zoom) - (y - scrollTop));
        container.scrollLeft = this._scrollLeft;
        container.scrollTop = this._scrollTop;
        this._zoom = zoom;
    }

    _mouseDownHandler(e) {
        if (e.buttons === 1) {
            const document = this._host.document.documentElement;
            const container = this._getElementById('graph');
            this._mousePosition = {
                left: container.scrollLeft,
                top: container.scrollTop,
                x: e.clientX,
                y: e.clientY
            };
            const background = this._getElementById('background');
            background.setAttribute('cursor', 'grabbing');
            e.stopImmediatePropagation();
            const mouseMoveHandler = (e) => {
                e.preventDefault();
                e.stopImmediatePropagation();
                const dx = e.clientX - this._mousePosition.x;
                const dy = e.clientY - this._mousePosition.y;
                this._mousePosition.moved = dx * dx + dy * dy > 0;
                if (this._mousePosition.moved) {
                    const container = this._getElementById('graph');
                    container.scrollTop = this._mousePosition.top - dy;
                    container.scrollLeft = this._mousePosition.left - dx;
                }
            };
            const mouseUpHandler = () => {
                background.setAttribute('cursor', 'default');
                container.removeEventListener('mouseup', mouseUpHandler);
                container.removeEventListener('mouseleave', mouseUpHandler);
                container.removeEventListener('mousemove', mouseMoveHandler);
                if (this._mousePosition && this._mousePosition.moved) {
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    delete this._mousePosition;
                    document.addEventListener('click', clickHandler, true);
                }
            };
            const clickHandler = (e) => {
                e.stopPropagation();
                document.removeEventListener('click', clickHandler, true);
            };
            container.addEventListener('mousemove', mouseMoveHandler);
            container.addEventListener('mouseup', mouseUpHandler);
            container.addEventListener('mouseleave', mouseUpHandler);
        }
    }

    _touchStartHandler(e) {
        if (e.touches.length === 2) {
            this._touchPoints = Array.from(e.touches);
            this._touchZoom = this._zoom;
        }
        const touchMoveHandler = (e) => {
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
        };
        const touchEndHandler = () => {
            container.removeEventListener('touchmove', touchMoveHandler, { passive: true });
            container.removeEventListener('touchcancel', touchEndHandler, { passive: true });
            container.removeEventListener('touchend', touchEndHandler, { passive: true });
            delete this._touchPoints;
            delete this._touchZoom;
        };
        const container = this._getElementById('graph');
        container.addEventListener('touchmove', touchMoveHandler, { passive: true });
        container.addEventListener('touchcancel', touchEndHandler, { passive: true });
        container.addEventListener('touchend', touchEndHandler, { passive: true });
    }

    _gestureStartHandler(e) {
        e.preventDefault();
        this._gestureZoom = this._zoom;
        const container = this._getElementById('graph');
        const gestureChangeHandler = (e) => {
            e.preventDefault();
            this._updateZoom(this._gestureZoom * e.scale, e);
        };
        const gestureEndHandler = (e) => {
            container.removeEventListener('gesturechange', gestureChangeHandler, false);
            container.removeEventListener('gestureend', gestureEndHandler, false);
            e.preventDefault();
            if (this._gestureZoom) {
                this._updateZoom(this._gestureZoom * e.scale, e);
                delete this._gestureZoom;
            }
        };
        container.addEventListener('gesturechange', gestureChangeHandler, false);
        container.addEventListener('gestureend', gestureEndHandler, false);
    }

    _scrollHandler(e) {
        if (this._scrollLeft && e.target.scrollLeft !== Math.floor(this._scrollLeft)) {
            delete this._scrollLeft;
        }
        if (this._scrollTop && e.target.scrollTop !== Math.floor(this._scrollTop)) {
            delete this._scrollTop;
        }
    }

    _wheelHandler(e) {
        if (e.shiftKey || e.ctrlKey || this._options.mousewheel === 'zoom') {
            const delta = -e.deltaY * (e.deltaMode === 1 ? 0.05 : e.deltaMode ? 1 : 0.002) * (e.ctrlKey ? 10 : 1);
            this._updateZoom(this._zoom * Math.pow(2, delta), e);
            e.preventDefault();
        }
    }

    select(selection) {
        this.clearSelection();
        if (selection && selection.length > 0) {
            const container = this._getElementById('graph');
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
            const rect = container.getBoundingClientRect();
            const left = (container.scrollLeft + x - rect.left) - (rect.width / 2);
            const top = (container.scrollTop + y - rect.top) - (rect.height / 2);
            container.scrollTo({ left: left, top: top, behavior: 'smooth' });
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
            { name: 'RangeError', message: /^Maximum call stack size exceeded/, url: 'https://github.com/lutzroeder/netron/issues/589' },
            { name: 'RangeError', message: /^Invalid string length/, url: 'https://github.com/lutzroeder/netron/issues/648' },
            { name: 'Python Error', message: /^Unknown function/, url: 'https://github.com/lutzroeder/netron/issues/546' },
            { name: 'Error loading model.', message: /^Unsupported file content \(/, url: 'https://github.com/lutzroeder/netron/issues/550' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers content/, url: 'https://github.com/lutzroeder/netron/issues/593' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers text content/, url: 'https://github.com/lutzroeder/netron/issues/594' },
            { name: 'Error loading model.', message: /^Unsupported JSON content/, url: 'https://github.com/lutzroeder/netron/issues/595' },
            { name: 'Error loading Caffe model.', message: /^File format is not caffe\.NetParameter/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'Error loading Darknet model.', message: /^Invalid tensor shape/, url: 'https://github.com/lutzroeder/netron/issues/541' },
            { name: 'Error loading DaVinci model.', message: /^Unsupported attribute type/, url: 'https://github.com/lutzroeder/netron/issues/926' },
            { name: 'Error loading Keras model.', message: /^Unsupported data object header version/, url: 'https://github.com/lutzroeder/netron/issues/548' },
            { name: 'Error loading MNN model.', message: /^File format is not mnn\.Net/, url: 'https://github.com/lutzroeder/netron/issues/746' },
            { name: 'Error loading NNEF model.', message: /^.*/, url: 'https://github.com/lutzroeder/netron/issues/992' },
            { name: 'Error loading PyTorch model.', message: /^File does not contain root module or state dictionary/, url: 'https://github.com/lutzroeder/netron/issues/543' },
            { name: 'Error loading PyTorch model.', message: /^Module does not contain modules/, url: 'https://github.com/lutzroeder/netron/issues/544' },
            { name: 'Error loading PyTorch model.', message: /^Unknown type name/, url: 'https://github.com/lutzroeder/netron/issues/969' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto/, url: 'https://github.com/lutzroeder/netron/issues/549' },
            { name: 'Error loading TensorFlow Lite model.', message: /^Offset is outside the bounds of the DataView/, url: 'https://github.com/lutzroeder/netron/issues/563' },
        ];
        const known = knowns.find((known) => (known.name.length === 0 || known.name === err.name) && err.message.match(known.message));
        const message = err.message;
        name = name || err.name;
        this._host.error(name, message, known ? known.url : undefined);
        this.show(screen !== undefined ? screen : 'welcome');
    }

    accept(file, size) {
        return this._modelFactoryService.accept(file, size);
    }

    open(context) {
        this._sidebar.close();
        return this._timeout(2).then(() => {
            return this._modelFactoryService.open(context).then((model) => {
                const format = [];
                if (model.format) {
                    format.push(model.format);
                }
                if (model.producer) {
                    format.push('(' + model.producer + ')');
                }
                if (format.length > 0) {
                    this._host.event_ua('Model', 'Format', format.join(' '));
                    this._host.event('model_open', {
                        model_format: model.format || '',
                        model_producer: model.producer || ''
                    });
                }
                return this._timeout(20).then(() => {
                    const graphs = Array.isArray(model.graphs) && model.graphs.length > 0 ? [ model.graphs[0] ] : [];
                    return this._updateGraph(model, graphs);
                });
            }).catch((error) => {
                if (error && context.identifier) {
                    error.context = context.identifier;
                }
                throw error;
            });
        });
    }

    _updateActiveGraph(graph) {
        this._sidebar.close();
        if (this._model) {
            const model = this._model;
            this.show('welcome spinner');
            this._timeout(200).then(() => {
                return this._updateGraph(model, [ graph ]).catch((error) => {
                    if (error) {
                        this.error(error, 'Graph update failed.', 'welcome');
                    }
                });
            });
        }
    }

    get activeGraph() {
        return Array.isArray(this._graphs) && this._graphs.length > 0 ? this._graphs[0] : null;
    }

    _updateGraph(model, graphs) {
        return this._timeout(100).then(() => {
            const graph = Array.isArray(graphs) && graphs.length > 0 ? graphs[0] : null;
            if (graph && graph != this._graphs[0]) {
                const nodes = graph.nodes;
                if (nodes.length > 2048) {
                    if (!this._host.confirm('Large model detected.', 'This graph contains a large number of nodes and might take a long time to render. Do you want to continue?')) {
                        this._host.event('graph_view', {
                            graph_node_count: nodes.length,
                            graph_skip: 1 }
                        );
                        this.show(null);
                        return null;
                    }
                }
            }
            const update = () => {
                const nameButton = this._getElementById('name-button');
                const backButton = this._getElementById('back-button');
                if (this._graphs.length > 1) {
                    const graph = this.activeGraph;
                    nameButton.innerHTML = graph ? graph.name : '';
                    backButton.style.opacity = 1;
                    nameButton.style.opacity = 1;
                }
                else {
                    backButton.style.opacity = 0;
                    nameButton.style.opacity = 0;
                }
            };
            const lastModel = this._model;
            const lastGraphs = this._graphs;
            this._model = model;
            this._graphs = graphs;
            return this.renderGraph(this._model, this.activeGraph).then(() => {
                if (this._page !== 'default') {
                    this.show('default');
                }
                update();
                return this._model;
            }).catch((error) => {
                this._model = lastModel;
                this._graphs = lastGraphs;
                return this.renderGraph(this._model, this.activeGraph).then(() => {
                    if (this._page !== 'default') {
                        this.show('default');
                    }
                    update();
                    throw error;
                });
            });
        });
    }

    pushGraph(graph) {
        if (graph !== this.activeGraph) {
            this._sidebar.close();
            this._updateGraph(this._model, [ graph ].concat(this._graphs));
        }
    }

    popGraph() {
        if (this._graphs.length > 1) {
            this._sidebar.close();
            return this._updateGraph(this._model, this._graphs.slice(1));
        }
        return null;
    }

    renderGraph(model, graph) {
        try {
            this._graph = null;

            const canvas = this._getElementById('canvas');
            while (canvas.lastChild) {
                canvas.removeChild(canvas.lastChild);
            }
            if (!graph) {
                return Promise.resolve();
            }
            this._zoom = 1;

            const groups = graph.groups;
            const nodes = graph.nodes;
            this._host.event('graph_view', {
                graph_node_count: nodes.length,
                graph_skip: 0
            });

            const options = {};
            options.nodesep = 20;
            options.ranksep = 20;
            const rotate = graph.nodes.every((node) => node.inputs.filter((input) => input.arguments.every((argument) => !argument.initializer)).length === 0 && node.outputs.length === 0);
            const horizontal = rotate ? this._options.direction === 'vertical' : this._options.direction !== 'vertical';
            if (horizontal) {
                options.rankdir = "LR";
            }
            if (nodes.length > 3000) {
                options.ranker = 'longest-path';
            }

            const viewGraph = new view.Graph(this, model, groups, options);
            viewGraph.add(graph);

            // Workaround for Safari background drag/zoom issue:
            // https://stackoverflow.com/questions/40887193/d3-js-zoom-is-not-working-with-mousewheel-in-safari
            const background = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            background.setAttribute('id', 'background');
            background.setAttribute('fill', 'none');
            background.setAttribute('pointer-events', 'all');
            canvas.appendChild(background);

            const origin = this._host.document.createElementNS('http://www.w3.org/2000/svg', 'g');
            origin.setAttribute('id', 'origin');
            canvas.appendChild(origin);

            viewGraph.build(this._host.document, origin);

            this._zoom = 1;

            return this._timeout(20).then(() => {

                viewGraph.update();

                const elements = Array.from(canvas.getElementsByClassName('graph-input') || []);
                if (elements.length === 0) {
                    const nodeElements = Array.from(canvas.getElementsByClassName('graph-node') || []);
                    if (nodeElements.length > 0) {
                        elements.push(nodeElements[0]);
                    }
                }

                const size = canvas.getBBox();
                const margin = 100;
                const width = Math.ceil(margin + size.width + margin);
                const height = Math.ceil(margin + size.height + margin);
                origin.setAttribute('transform', 'translate(' + margin.toString() + ', ' + margin.toString() + ') scale(1)');
                background.setAttribute('width', width);
                background.setAttribute('height', height);
                this._width = width;
                this._height = height;
                delete this._scrollLeft;
                delete this._scrollRight;
                canvas.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
                canvas.setAttribute('width', width);
                canvas.setAttribute('height', height);

                this._zoom = 1;
                this._updateZoom(this._zoom);

                const container = this._getElementById('graph');
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
                        x = xs.reduce((a, b) => a + b, 0) / xs.length;
                    }
                    const graphRect = container.getBoundingClientRect();
                    const left = (container.scrollLeft + x - graphRect.left) - (graphRect.width / 2);
                    const top = (container.scrollTop + y - graphRect.top) - (graphRect.height / 2);
                    container.scrollTo({ left: left, top: top, behavior: 'auto' });
                }
                else {
                    const canvasRect = canvas.getBoundingClientRect();
                    const graphRect = container.getBoundingClientRect();
                    const left = (container.scrollLeft + (canvasRect.width / 2) - graphRect.left) - (graphRect.width / 2);
                    const top = (container.scrollTop + (canvasRect.height / 2) - graphRect.top) - (graphRect.height / 2);
                    container.scrollTo({ left: left, top: top, behavior: 'auto' });
                }
                this._graph = viewGraph;
                return;
            });
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
        const extension = (lastIndex != -1) ? file.substring(lastIndex + 1).toLowerCase() : 'png';
        if (this.activeGraph && (extension === 'png' || extension === 'svg')) {
            const canvas = this._getElementById('canvas');
            const clone = canvas.cloneNode(true);
            this.applyStyleSheet(clone, 'grapher.css');
            clone.setAttribute('id', 'export');
            clone.removeAttribute('viewBox');
            clone.removeAttribute('width');
            clone.removeAttribute('height');
            clone.style.removeProperty('opacity');
            clone.style.removeProperty('display');
            clone.style.removeProperty('width');
            clone.style.removeProperty('height');
            const background = clone.querySelector('#background');
            const origin = clone.querySelector('#origin');
            origin.setAttribute('transform', 'translate(0,0) scale(1)');
            background.removeAttribute('width');
            background.removeAttribute('height');

            const parent = canvas.parentElement;
            parent.insertBefore(clone, canvas);
            const size = clone.getBBox();
            parent.removeChild(clone);
            parent.removeChild(canvas);
            parent.appendChild(canvas);
            const delta = (Math.min(size.width, size.height) / 2.0) * 0.1;
            const width = Math.ceil(delta + size.width + delta);
            const height = Math.ceil(delta + size.height + delta);
            origin.setAttribute('transform', 'translate(' + (delta - size.x).toString() + ', ' + (delta - size.y).toString() + ') scale(1)');
            clone.setAttribute('width', width);
            clone.setAttribute('height', height);
            background.setAttribute('width', width);
            background.setAttribute('height', height);
            background.setAttribute('fill', '#fff');

            const data = new XMLSerializer().serializeToString(clone);

            if (extension === 'svg') {
                const blob = new Blob([ data ], { type: 'image/svg' });
                this._host.export(file, blob);
            }

            if (extension === 'png') {
                const image = new Image();
                image.onload = () => {
                    const max = Math.max(width, height);
                    const scale = Math.min(24000.0 / max, 2.0);
                    const canvas = this._host.document.createElement('canvas');
                    canvas.width = Math.ceil(width * scale);
                    canvas.height = Math.ceil(height * scale);
                    const context = canvas.getContext('2d');
                    context.scale(scale, scale);
                    context.drawImage(image, 0, 0);
                    canvas.toBlob((blob) => {
                        if (blob) {
                            this._host.export(file, blob);
                        }
                        else {
                            const error = new Error('Image may be too large to render as PNG.');
                            error.name = 'Error exporting image.';
                            this._host.exception(error, false);
                            this._host.error(error.name, error.message);
                        }
                    }, 'image/png');
                };
                image.src = 'data:image/svg+xml;base64,' + this._host.window.btoa(unescape(encodeURIComponent(data)));
            }
        }
    }

    showModelProperties() {
        if (this._model) {
            try {
                const modelSidebar = new view.ModelSidebar(this._host, this._model, this.activeGraph);
                modelSidebar.on('update-active-graph', (sender, graph) => {
                    this._updateActiveGraph(graph);
                });
                const content = modelSidebar.render();
                this._sidebar.open(content, 'Model Properties');
            }
            catch (error) {
                if (error) {
                    error.context = this._model.identifier;
                }
                this.error(error, 'Error showing model properties.', null);
            }
        }
    }

    showNodeProperties(node, input) {
        if (node) {
            try {
                const nodeSidebar = new view.NodeSidebar(this._host, node);
                nodeSidebar.on('show-documentation', (/* sender, e */) => {
                    this.showDocumentation(node.type);
                });
                nodeSidebar.on('show-graph', (sender, graph) => {
                    this.pushGraph(graph);
                });
                nodeSidebar.on('export-tensor', (sender, tensor) => {
                    const defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
                    this._host.save('NumPy Array', 'npy', defaultPath, (file) => {
                        try {
                            let data_type = tensor.type.dataType;
                            if (data_type === 'boolean') {
                                data_type = 'bool';
                            }
                            const execution = new python.Execution();
                            const bytes = execution.invoke('io.BytesIO', []);
                            const dtype = execution.invoke('numpy.dtype', [ data_type ]);
                            const array = execution.invoke('numpy.asarray', [ tensor.value, dtype ]);
                            execution.invoke('numpy.save', [ bytes, array ]);
                            bytes.seek(0);
                            const blob = new Blob([ bytes.read() ], { type: 'application/octet-stream' });
                            this._host.export(file, blob);
                        }
                        catch (error) {
                            this.error(error, 'Error saving NumPy tensor.', null);
                        }
                    });
                });
                nodeSidebar.on('error', (sender, error) => {
                    if (this._model) {
                        error.context = this._model.identifier;
                    }
                    this.error(error, null, null);
                });
                if (input) {
                    nodeSidebar.toggleInput(input.name);
                }
                this._sidebar.open(nodeSidebar.render(), 'Node Properties');
            }
            catch (error) {
                if (error) {
                    error.context = this._model.identifier;
                }
                this.error(error, 'Error showing node properties.', null);
            }
        }
    }

    showDocumentation(type) {
        if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
            if (type.nodes && type.nodes.length > 0) {
                this.pushGraph(type);
            }
            const documentationSidebar = new view.DocumentationSidebar(this._host, type);
            documentationSidebar.on('navigate', (sender, e) => {
                this._host.openURL(e.link);
            });
            const title = type.type === 'function' ? 'Function' : 'Documentation';
            this._sidebar.push(documentationSidebar.render(), title);
        }
    }

    about() {
        const handler = () => {
            this._host.window.removeEventListener('keydown', handler);
            this._host.document.body.removeEventListener('click', handler);
            this._host.document.body.classList.remove('about');
        };
        this._host.window.addEventListener('keydown', handler);
        this._host.document.body.addEventListener('click', handler);
        this._host.document.body.classList.add('about');
    }
};

view.Menu = class {

    constructor(host, button, dropdown) {
        this._host = host;
        this._dropdown = this._host.document.getElementById(dropdown);
        this._button = this._host.document.getElementById(button);
        this._items = [];
        this._darwin = this._host.environment('platform') === 'darwin';
        this._accelerators = new Map();
        this._host.window.addEventListener('keydown', (e) => {
            let code = e.keyCode;
            code |= ((e.ctrlKey && !this._darwin) || (e.metaKey && this._darwin)) ? 0x0400 : 0;
            code |= e.altKey ? 0x0200 : 0;
            code |= e.shiftKey ? 0x0100 : 0;
            if (code == 0x001b) { // Escape
                this.close();
                return;
            }
            const item = this._accelerators.get(code.toString());
            if (item && (!item.enabled || item.enabled())) {
                item.click();
                e.preventDefault();
            }
        });
        this._host.document.body.addEventListener('click', (e) => {
            if (!this._button.contains(e.target)) {
                this.close();
            }
        });
    }

    add(item) {
        const accelerator = item.accelerator;
        if (accelerator) {
            let cmdOrCtrl = false;
            let alt = false;
            let shift = false;
            let key = '';
            for (const part of item.accelerator.split('+')) {
                switch (part) {
                    case 'CmdOrCtrl': cmdOrCtrl = true; break;
                    case 'Alt': alt = true; break;
                    case 'Shift': shift = true; break;
                    default: key = part; break;
                }
            }
            if (key !== '') {
                item.accelerator = {};
                item.accelerator.text = '';
                if (this._darwin) {
                    item.accelerator.text += alt ? '&#x2325;' : '';
                    item.accelerator.text += shift ? '&#x21e7;' : '';
                    item.accelerator.text += cmdOrCtrl ? '&#x2318;' : '';
                    const keyTable = { 'Enter': '&#x23ce;', 'Up': '&#x2191;', 'Down': '&#x2193;', 'Backspace': '&#x232B;' };
                    item.accelerator.text += keyTable[key] ? keyTable[key] : key;
                }
                else {
                    const list = [];
                    if (cmdOrCtrl) {
                        list.push('Ctrl');
                    }
                    if (alt) {
                        list.push('Alt');
                    }
                    if (shift) {
                        list.push('Shift');
                    }
                    list.push(key);
                    item.accelerator.text = list.join('+');
                }
                let code = 0;
                switch (key) {
                    case 'Backspace': code = 0x08; break;
                    case 'Enter': code = 0x0D; break;
                    case 'Up': code = 0x26; break;
                    case 'Down': code = 0x28; break;
                    default: code = key.charCodeAt(0); break;
                }
                code |= cmdOrCtrl ? 0x0400 : 0;
                code |= alt ? 0x0200 : 0;
                code |= shift ? 0x0100 : 0;
                this._accelerators.set(code.toString(), item);
            }
        }
        this._items.push(item);
    }

    toggle() {

        if (this._dropdown.style.opacity >= 1) {
            this.close();
            return;
        }

        while (this._dropdown.lastChild) {
            this._dropdown.removeChild(this._dropdown.lastChild);
        }

        for (const item of this._items) {
            if (Object.keys(item).length > 0) {
                const button = this._host.document.createElement('button');
                button.innerText = (typeof item.label == 'function') ? item.label() : item.label;
                button.addEventListener('click', () => {
                    this.close();
                    setTimeout(() => {
                        item.click();
                    }, 10);
                });
                if (item.enabled && !item.enabled()) {
                    button.setAttribute('disabled', '');
                }
                this._dropdown.appendChild(button);
                if (item.accelerator) {
                    const accelerator = this._host.document.createElement('span');
                    accelerator.setAttribute('class', 'shortcut');
                    accelerator.innerHTML = item.accelerator.text;
                    button.appendChild(accelerator);
                }
            }
            else {
                const separator = this._host.document.createElement('div');
                separator.setAttribute('class', 'separator');
                this._dropdown.appendChild(separator);
            }
        }

        this._dropdown.style.opacity = 1.0;
        this._dropdown.style.left = '0px';
    }

    close() {
        this._dropdown.style.opacity = 0;
        this._dropdown.style.left = '-200px';
    }
};

view.Graph = class extends grapher.Graph {

    constructor(view, model, compound, options) {
        super(compound, options);
        this.view = view;
        this.model = model;
        this._arguments = new Map();
        this._nodeKey = 0;
    }

    createNode(node) {
        const value = new view.Node(this, node);
        value.name = (this._nodeKey++).toString();
        // value.name = node.name;
        this.setNode(value);
        return value;
    }

    createInput(input) {
        const value = new view.Input(this, input);
        value.name = (this._nodeKey++).toString();
        this.setNode(value);
        return value;
    }

    createOutput(output) {
        const value = new view.Output(this, output);
        value.name = (this._nodeKey++).toString();
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

    add(graph) {
        const clusters = new Set();
        const clusterParentMap = new Map();
        const groups = graph.groups;
        if (groups) {
            for (const node of graph.nodes) {
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

        for (const input of graph.inputs) {
            const viewInput = this.createInput(input);
            for (const argument of input.arguments) {
                this.createArgument(argument).from(viewInput);
            }
        }

        for (const node of graph.nodes) {

            const viewNode = this.createNode(node);

            const inputs = node.inputs;
            for (const input of inputs) {
                for (const argument of input.arguments) {
                    if (argument.name != '' && !argument.initializer) {
                        this.createArgument(argument).to(viewNode);
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
                        const error = new view.Error('Invalid null argument.');
                        error.context = this.model.identifier;
                        throw error;
                    }
                    if (argument.name != '') {
                        this.createArgument(argument).from(viewNode);
                    }
                }
            }

            if (node.controlDependencies && node.controlDependencies.length > 0) {
                for (const argument of node.controlDependencies) {
                    this.createArgument(argument).to(viewNode, true);
                }
            }

            const createCluster = (name) => {
                if (!clusters.has(name)) {
                    this.setNode({ name: name, rx: 5, ry: 5 });
                    clusters.add(name);
                    const parent = clusterParentMap.get(name);
                    if (parent) {
                        createCluster(parent);
                        this.setParent(name, parent);
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
                        createCluster(groupName + '\ngroup');
                        this.setParent(viewNode.name, groupName + '\ngroup');
                    }
                }
            }
        }

        for (const output of graph.outputs) {
            const viewOutput = this.createOutput(output);
            for (const argument of output.arguments) {
                this.createArgument(argument).to(viewOutput);
            }
        }
    }

    build(document, origin) {
        for (const argument of this._arguments.values()) {
            argument.build();
        }
        super.build(document, origin);
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

    get inputs() {
        return this.value.inputs;
    }

    get outputs() {
        return this.value.outputs;
    }

    _add(node) {
        const header =  this.header();
        const styles = [ 'node-item-type' ];
        const type = node.type;
        const category = type && type.category ? type.category : '';
        if (category) {
            styles.push('node-item-type-' + category.toLowerCase());
        }
        if (typeof type.name !== 'string' || !type.name.split) { // #416
            const error = new view.Error("Unsupported node type '" + JSON.stringify(type.name) + "'.");
            if (this.context.model && this.context.model.identifier) {
                error.context = this.context.model.identifier;
            }
            throw error;
        }
        const content = this.context.view.options.names && (node.name || node.location) ? (node.name || node.location) : type.name.split('.').pop();
        const tooltip = this.context.view.options.names && (node.name || node.location) ? type.name : (node.name || node.location);
        const title = header.add(null, styles, content, tooltip);
        title.on('click', () => this.context.view.showNodeProperties(node, null));
        if (node.type.nodes && node.type.nodes.length > 0) {
            const definition = header.add(null, styles, '\u0192', 'Show Function Definition');
            definition.on('click', () => this.context.view.pushGraph(node.type));
        }
        if (node.nodes) {
            // this._expand = header.add(null, styles, '+', null);
            // this._expand.on('click', () => this.toggle());
        }
        const initializers = [];
        let hiddenInitializers = false;
        if (this.context.view.options.initializers) {
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
        if (this.context.view.options.attributes) {
            sortedAttributes = attributes.filter((attribute) => attribute.visible).slice();
        }
        sortedAttributes.sort((a, b) => {
            const au = a.name.toUpperCase();
            const bu = b.name.toUpperCase();
            return (au < bu) ? -1 : (au > bu) ? 1 : 0;
        });
        if (initializers.length > 0 || hiddenInitializers || sortedAttributes.length > 0) {
            const list = this.list();
            list.on('click', () => this.context.view.showNodeProperties(node));
            for (const initializer of initializers) {
                const argument = initializer.arguments[0];
                const type = argument.type;
                let shape = '';
                let separator = '';
                if (type && type.shape && type.shape.dimensions && Array.isArray(type.shape.dimensions)) {
                    shape = '\u3008' + type.shape.dimensions.map((d) => (d !== null && d !== undefined) ? d : '?').join('\u00D7') + '\u3009';
                    if (type.shape.dimensions.length === 0 && argument.initializer) {
                        try {
                            const initializer = argument.initializer;
                            const tensor = new view.Tensor(initializer);
                            if ((tensor.layout === '<' || tensor.layout === '>' || tensor.layout === '|') && !tensor.empty && tensor.type.dataType !== '?') {
                                shape = tensor.toString();
                                if (shape && shape.length > 10) {
                                    shape = shape.substring(0, 10) + '\u2026';
                                }
                                separator = ' = ';
                            }
                        }
                        catch (err) {
                            let type = '?';
                            try {
                                type = argument.initializer.type.toString();
                            }
                            catch (error) {
                                // continue regardless of error
                            }
                            const error = new view.Error("Failed to render tensor of type '" + type + "' (" + err.message + ").");
                            if (this.context.view.model && this.context.view.model.identifier) {
                                error.context = this.context.view.model.identifier;
                            }
                            throw error;
                        }
                    }
                }
                list.add(argument.name ? 'initializer-' + argument.name : '', initializer.name, shape, type ? type.toString() : '', separator);
            }
            if (hiddenInitializers) {
                list.add(null, '\u3008' + '\u2026' + '\u3009', '', null, '');
            }

            for (const attribute of sortedAttributes) {
                if (attribute.visible) {
                    let value = new view.Formatter(attribute.value, attribute.type).toString();
                    if (value && value.length > 25) {
                        value = value.substring(0, 25) + '\u2026';
                    }
                    list.add(null, attribute.name, value, attribute.type, ' = ');
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
        if (node.nodes) {
            // this.canvas = this.canvas();
        }
    }

    toggle() {
        this._expand.content = '-';
        this._graph = new view.Graph(this.context.view, this.context.model, false, {});
        this._graph.add(this.value);
        // const document = this.element.ownerDocument;
        // const parent = this.element.parentElement;
        // this._graph.build(document, parent);
        // this._graph.update();
        this.canvas.width = 300;
        this.canvas.height = 300;
        this.layout();
        this.context.update();
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
        const title = header.add(null, [ 'graph-item-input' ], name, types);
        title.on('click', () => this.context.view.showModelProperties());
        this.id = 'input-' + (name ? 'name-' + name : 'id-' + (view.Input.counter++).toString());
    }

    get class() {
        return 'graph-input';
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [ this.value ];
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
        const title = header.add(null, [ 'graph-item-output' ], name, types);
        title.on('click', () => this.context.view.showModelProperties());
    }

    get inputs() {
        return [ this.value ];
    }

    get outputs() {
        return [];
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

    to(node, controlDependency) {
        this._to = this._to || [];
        if (controlDependency) {
            this._controlDependencies = this._controlDependencies || new Set();
            this._controlDependencies.add(this._to.length);
        }
        this._to.push(node);
    }

    build() {
        this._edges = this._edges || [];
        if (this._from && this._to) {
            for (let i = 0; i < this._to.length; i++) {
                const to = this._to[i];
                let content = '';
                const type = this._argument.type;

                if (type &&
                    type.shape &&
                    type.shape.dimensions &&
                    type.shape.dimensions.length > 0 &&
                    type.shape.dimensions.every((dim) => !dim || Number.isInteger(dim) || dim instanceof base.Int64 || (typeof dim === 'string'))) {
                    content = type.shape.dimensions.map((dim) => (dim !== null && dim !== undefined) ? dim : '?').join('\u00D7');
                    content = content.length > 16 ? '' : content;
                }
                if (this.context.view.options.names) {
                    content = this._argument.name.split('\n').shift(); // custom argument id
                }
                const edge = this.context.createEdge(this._from, to);
                edge.v = this._from.name;
                edge.w = to.name;
                if (content) {
                    edge.label = content;
                }
                edge.id = 'edge-' + this._argument.name;
                if (this._controlDependencies && this._controlDependencies.has(i)) {
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

    get minlen() {
        if (this.from.inputs.every((parameter) => parameter.arguments.every((argument) => argument.initializer))) {
            return 2;
        }
        return 1;
    }
};

view.Sidebar = class {

    constructor(host, id) {
        this._host = host;
        this._id = id ? ('-' + id) : '';
        this._stack = [];
        this._closeSidebarHandler = () => {
            this._pop();
        };
        this._closeSidebarKeyDownHandler = (e) => {
            if (e.keyCode == 27) {
                e.preventDefault();
                this._pop();
            }
        };
        const sidebar = this._element('sidebar');
        sidebar.addEventListener('transitionend', (event) => {
            if (event.propertyName === 'opacity' && sidebar.style.opacity === '0') {
                const content = this._element('sidebar-content');
                content.innerHTML = '';
            }
        });
    }

    _element(id) {
        return this._host.document.getElementById(id + this._id);
    }

    open(content, title) {
        this.close();
        this.push(content, title);
    }

    close() {
        this._deactivate();
        this._stack = [];
        this._hide();
    }

    push(content, title) {
        const item = { title: title, content: content };
        this._stack.push(item);
        this._activate(item);
    }

    _pop() {
        this._deactivate();
        if (this._stack.length > 0) {
            this._stack.pop();
        }
        if (this._stack.length > 0) {
            this._activate(this._stack[this._stack.length - 1]);
        }
        else {
            this._hide();
        }
    }

    _hide() {
        const sidebar = this._element('sidebar');
        if (sidebar) {
            sidebar.style.right = 'calc(0px - min(calc(100% * 0.6), 500px))';
            sidebar.style.opacity = 0;
        }
        const container = this._element('graph');
        if (container) {
            container.style.width = '100%';
            container.focus();
        }
    }

    _deactivate() {
        const sidebar = this._element('sidebar');
        if (sidebar) {
            const closeButton = this._element('sidebar-closebutton');
            closeButton.removeEventListener('click', this._closeSidebarHandler);
            this._host.document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
    }

    _activate(item) {
        const sidebar = this._element('sidebar');
        if (sidebar) {

            const title = this._element('sidebar-title');
            title.innerHTML = item.title ? item.title.toUpperCase() : '';
            const closeButton = this._element('sidebar-closebutton');
            closeButton.addEventListener('click', this._closeSidebarHandler);
            const content = this._element('sidebar-content');

            if (typeof item.content == 'string') {
                content.innerHTML = item.content;
            }
            else if (item.content instanceof Array) {
                content.innerHTML = '';
                for (const element of item.content) {
                    content.appendChild(element);
                }
            }
            else {
                content.innerHTML = '';
                content.appendChild(item.content);
            }
            sidebar.style.width = 'min(calc(100% * 0.6), 500px)';
            sidebar.style.right = 0;
            sidebar.style.opacity = 1;
            this._host.document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
        const container = this._element('graph');
        if (container) {
            container.style.width = 'max(40vw, calc(100vw - 500px))';
        }
    }
};

view.Control = class {

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
};

view.NodeSidebar = class extends view.Control {

    constructor(host, node) {
        super();
        this._host = host;
        this._node = node;
        this._elements = [];
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        if (node.type) {
            let showDocumentation = null;
            const type = node.type;
            if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
                showDocumentation = {};
                showDocumentation.text = type.nodes ? '\u0192': '?';
                showDocumentation.callback = () => {
                    this.emit('show-documentation', null);
                };
            }
            this._addProperty('type', new view.ValueTextView(this._host, node.type.identifier || node.type.name, showDocumentation));
            if (node.type.module) {
                this._addProperty('module', new view.ValueTextView(this._host, node.type.module));
            }
        }

        if (node.name) {
            this._addProperty('name', new view.ValueTextView(this._host, node.name));
        }

        if (node.location) {
            this._addProperty('location', new view.ValueTextView(this._host, node.location));
        }

        if (node.description) {
            this._addProperty('description', new view.ValueTextView(this._host, node.description));
        }

        if (node.device) {
            this._addProperty('device', new view.ValueTextView(this._host, node.device));
        }

        const attributes = node.attributes;
        if (attributes && attributes.length > 0) {
            const sortedAttributes = node.attributes.slice();
            sortedAttributes.sort((a, b) => {
                const au = a.name.toUpperCase();
                const bu = b.name.toUpperCase();
                return (au < bu) ? -1 : (au > bu) ? 1 : 0;
            });
            this._addHeader('Attributes');
            for (const attribute of sortedAttributes) {
                this._addAttribute(attribute.name, attribute);
            }
        }

        const inputs = node.inputs;
        if (inputs && inputs.length > 0) {
            this._addHeader('Inputs');
            for (const input of inputs) {
                this._addInput(input.name, input);
            }
        }

        const outputs = node.outputs;
        if (outputs && outputs.length > 0) {
            this._addHeader('Outputs');
            for (const output of outputs) {
                this._addOutput(output.name, output);
            }
        }

        const separator = this._host.document.createElement('div');
        separator.className = 'sidebar-view-separator';
        this._elements.push(separator);
    }

    render() {
        return this._elements;
    }

    _addHeader(title) {
        const headerElement = this._host.document.createElement('div');
        headerElement.className = 'sidebar-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    _addProperty(name, value) {
        const item = new view.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    _addAttribute(name, attribute) {
        const value = new view.AttributeView(this._host, attribute);
        value.on('show-graph', (sender, graph) => {
            this.emit('show-graph', graph);
        });
        const item = new view.NameValueView(this._host, name, value);
        this._attributes.push(item);
        this._elements.push(item.render());
    }

    _addInput(name, input) {
        if (input.arguments.length > 0) {
            const value = new view.ParameterView(this._host, input);
            value.on('export-tensor', (sender, tensor) => {
                this.emit('export-tensor', tensor);
            });
            value.on('error', (sender, tensor) => {
                this.emit('error', tensor);
            });
            const item = new view.NameValueView(this._host, name, value);
            this._inputs.push(item);
            this._elements.push(item.render());
        }
    }

    _addOutput(name, output) {
        if (output.arguments.length > 0) {
            const item = new view.NameValueView(this._host, name, new view.ParameterView(this._host, output));
            this._outputs.push(item);
            this._elements.push(item.render());
        }
    }

    toggleInput(name) {
        for (const input of this._inputs) {
            if (name == input.name) {
                input.toggle();
            }
        }
    }
};

view.NameValueView = class {

    constructor(host, name, value) {
        this._host = host;
        this._name = name;
        this._value = value;

        const nameElement = this._host.document.createElement('div');
        nameElement.className = 'sidebar-view-item-name';

        const nameInputElement = this._host.document.createElement('input');
        nameInputElement.setAttribute('type', 'text');
        nameInputElement.setAttribute('value', name);
        nameInputElement.setAttribute('title', name);
        nameInputElement.setAttribute('readonly', 'true');
        nameElement.appendChild(nameInputElement);

        const valueElement = this._host.document.createElement('div');
        valueElement.className = 'sidebar-view-item-value-list';

        for (const element of value.render()) {
            valueElement.appendChild(element);
        }

        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item';
        this._element.appendChild(nameElement);
        this._element.appendChild(valueElement);
    }

    get name() {
        return this._name;
    }

    render() {
        return this._element;
    }

    toggle() {
        this._value.toggle();
    }
};

view.SelectView = class extends view.Control {

    constructor(host, values, selected) {
        super();
        this._host = host;
        this._elements = [];
        this._values = values;

        const selectElement = this._host.document.createElement('select');
        selectElement.setAttribute('class', 'sidebar-view-item-select');
        selectElement.addEventListener('change', (e) => {
            this.emit('change', this._values[e.target.selectedIndex]);
        });
        this._elements.push(selectElement);

        for (const value of values) {
            const optionElement = this._host.document.createElement('option');
            optionElement.innerText = value.name || '';
            if (value == selected) {
                optionElement.setAttribute('selected', 'selected');
            }
            selectElement.appendChild(optionElement);
        }
    }

    render() {
        return this._elements;
    }
};

view.ValueTextView = class {

    constructor(host, value, action) {
        this._host = host;
        this._elements = [];
        const element = this._host.document.createElement('div');
        element.className = 'sidebar-view-item-value';
        this._elements.push(element);

        if (action) {
            this._action = this._host.document.createElement('div');
            this._action.className = 'sidebar-view-item-value-expander';
            this._action.innerHTML = action.text;
            this._action.addEventListener('click', () => {
                action.callback();
            });
            element.appendChild(this._action);
        }

        const list = Array.isArray(value) ? value : [ value ];
        let className = 'sidebar-view-item-value-line';
        for (const item of list) {
            const line = this._host.document.createElement('div');
            line.className = className;
            line.innerText = item;
            element.appendChild(line);
            className = 'sidebar-view-item-value-line-border';
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
    }
};

view.ValueView = class extends view.Control {

    _bold(name, value) {
        const line = this._host.document.createElement('div');
        line.innerHTML = name + ': ' + '<b>' + value + '</b>';
        this._add(line);
    }

    _code(name, value) {
        const line = this._host.document.createElement('div');
        line.innerHTML = name + ': ' + '<code><b>' + value + '</b></code>';
        this._add(line);
    }

    _add(child) {
        child.className = this._element.childNodes.length < 2 ? 'sidebar-view-item-value-line' : 'sidebar-view-item-value-line-border';
        this._element.appendChild(child);
    }

    _tensor(value) {
        const contentLine = this._host.document.createElement('pre');
        try {
            const tensor = new view.Tensor(value);
            const layout = tensor.layout;
            if (layout) {
                const layouts = new Map([
                    [ 'sparse', 'Sparse' ],
                    [ 'sparse.coo', 'Sparse COO' ],
                    [ 'sparse.csr', 'Sparse CSR' ],
                    [ 'sparse.csc', 'Sparse CSC' ],
                    [ 'sparse.bsr', 'Sparse BSR' ],
                    [ 'sparse.bsc', 'Sparse BSC' ]
                ]);
                if (layouts.has(layout)) {
                    this._bold('layout', layouts.get(layout));
                }
            }
            if (Array.isArray(tensor.stride) && tensor.stride.length > 0) {
                this._code('stride', tensor.stride.join(','));
            }
            if (tensor.layout !== '<' && tensor.layout !== '>' && tensor.layout !== '|' && tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo') {
                contentLine.innerHTML = "Tensor layout '" + tensor.layout + "' is not implemented.";
            }
            else if (tensor.empty) {
                contentLine.innerHTML = 'Tensor data is empty.';
            }
            else if (tensor.type && tensor.type.dataType === '?') {
                contentLine.innerHTML = 'Tensor data type is not defined.';
            }
            else if (tensor.type && !tensor.type.shape) {
                contentLine.innerHTML = 'Tensor shape is not defined.';
            }
            else {
                contentLine.innerHTML = tensor.toString();

                if (this._host.save &&
                    value.type.shape && value.type.shape.dimensions &&
                    value.type.shape.dimensions.length > 0) {
                    this._saveButton = this._host.document.createElement('div');
                    this._saveButton.className = 'sidebar-view-item-value-expander';
                    this._saveButton.innerHTML = '&#x1F4BE;';
                    this._saveButton.addEventListener('click', () => {
                        this.emit('export-tensor', tensor);
                    });
                    this._element.appendChild(this._saveButton);
                }
            }
        }
        catch (err) {
            contentLine.innerHTML = err.toString();
            this.emit('error', err);
        }
        const valueLine = this._host.document.createElement('div');
        valueLine.className = 'sidebar-view-item-value-line-border';
        valueLine.appendChild(contentLine);
        this._element.appendChild(valueLine);
    }
};

view.AttributeView = class extends view.ValueView {

    constructor(host, attribute) {
        super();
        this._host = host;
        this._attribute = attribute;
        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        const type = this._attribute.type;
        if (type) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }
        const value = this._attribute.value;
        switch (type) {
            case 'graph': {
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line-link';
                line.innerHTML = value.name;
                line.addEventListener('click', () => {
                    this.emit('show-graph', value);
                });
                this._element.appendChild(line);
                break;
            }
            case 'function': {
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line-link';
                line.innerHTML = value.type.name;
                line.addEventListener('click', () => {
                    this.emit('show-graph', value.type);
                });
                this._element.appendChild(line);
                break;
            }
            default: {
                let content = new view.Formatter(value, type).toString();
                if (content && content.length > 1000) {
                    content = content.substring(0, 1000) + '\u2026';
                }
                if (content && typeof content === 'string') {
                    content = content.split('<').join('&lt;').split('>').join('&gt;');
                }
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line';
                line.innerHTML = content ? content : '&nbsp;';
                this._element.appendChild(line);
            }
        }
    }

    render() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander.innerText == '+') {
            this._expander.innerText = '-';

            const type = this._attribute.type;
            const value = this._attribute.value;
            const content = type == 'tensor' && value && value.type ? value.type.toString() : this._attribute.type;
            const typeLine = this._host.document.createElement('div');
            typeLine.className = 'sidebar-view-item-value-line-border';
            typeLine.innerHTML = 'type: ' + '<code><b>' + content + '</b></code>';
            this._element.appendChild(typeLine);

            const description = this._attribute.description;
            if (description) {
                const descriptionLine = this._host.document.createElement('div');
                descriptionLine.className = 'sidebar-view-item-value-line-border';
                descriptionLine.innerHTML = description;
                this._element.appendChild(descriptionLine);
            }

            if (this._attribute.type == 'tensor' && value) {
                this._tensor(value);
            }
        }
        else {
            this._expander.innerText = '+';
            while (this._element.childElementCount > 2) {
                this._element.removeChild(this._element.lastChild);
            }
        }
    }
};

view.ParameterView = class extends view.Control {

    constructor(host, list) {
        super();
        this._list = list;
        this._elements = [];
        this._items = [];
        for (const argument of list.arguments) {
            const item = new view.ArgumentView(host, argument);
            item.on('export-tensor', (sender, tensor) => {
                this.emit('export-tensor', tensor);
            });
            item.on('error', (sender, tensor) => {
                this.emit('error', tensor);
            });
            this._items.push(item);
            this._elements.push(item.render());
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
        for (const item of this._items) {
            item.toggle();
        }
    }
};

view.ArgumentView = class extends view.ValueView {

    constructor(host, argument) {
        super();
        this._host = host;
        this._argument = argument;

        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        const type = this._argument.type;
        const initializer = this._argument.initializer;
        const quantization = this._argument.quantization;
        const location = this._argument.location !== undefined;

        if (initializer) {
            this._element.classList.add('sidebar-view-item-value-dark');
        }

        if (type || initializer || quantization || location) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        const name = this._argument.name ? this._argument.name.split('\n').shift() : ''; // custom argument id
        this._hasId = name ? true : false;
        this._hasCategory = initializer && initializer.category ? true : false;
        if (this._hasId || (!this._hasCategory && !type)) {
            this._hasId = true;
            const nameLine = this._host.document.createElement('div');
            nameLine.className = 'sidebar-view-item-value-line';
            if (typeof name !== 'string') {
                throw new Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
            }
            nameLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>name: <b>' + (name || ' ') + '</b></span>';
            this._element.appendChild(nameLine);
        }
        else if (this._hasCategory) {
            this._bold('category', initializer.category);
        }
        else if (type) {
            this._code('type', type.toString().split('<').join('&lt;').split('>').join('&gt;'));
        }
    }

    render() {
        return this._element;
    }

    toggle() {
        if (this._expander) {
            if (this._expander.innerText == '+') {
                this._expander.innerText = '-';

                const initializer = this._argument.initializer;
                if (this._hasId && this._hasCategory) {
                    this._bold('category', initializer.category);
                }

                let type = null;
                let denotation = null;
                if (this._argument.type) {
                    type = this._argument.type.toString();
                    denotation = this._argument.type.denotation || null;
                }
                if (type && (this._hasId || this._hasCategory)) {
                    this._code('type', type.split('<').join('&lt;').split('>').join('&gt;'));
                }
                if (denotation) {
                    this._code('denotation', denotation);
                }

                const description = this._argument.description;
                if (description) {
                    const descriptionLine = this._host.document.createElement('div');
                    descriptionLine.className = 'sidebar-view-item-value-line-border';
                    descriptionLine.innerHTML = description;
                    this._element.appendChild(descriptionLine);
                }

                const quantization = this._argument.quantization;
                if (quantization) {
                    const quantizationLine = this._host.document.createElement('div');
                    quantizationLine.className = 'sidebar-view-item-value-line-border';
                    const content = !Array.isArray(quantization) ? quantization : '<br><br>' + quantization.map((value) => '  ' + value).join('<br>');
                    quantizationLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>quantization: ' + '<b>' + content + '</b></span>';
                    this._element.appendChild(quantizationLine);
                }

                const location = this._argument.location;
                if (location !== undefined) {
                    this._bold('location', location);
                }

                if (initializer) {
                    this._tensor(initializer);
                }
            }
            else {
                this._expander.innerText = '+';
                while (this._element.childElementCount > 2) {
                    this._element.removeChild(this._element.lastChild);
                }
            }
        }
    }
};

view.ModelSidebar = class extends view.Control {

    constructor(host, model, graph) {
        super();
        this._host = host;
        this._model = model;
        this._elements = [];

        if (model.format) {
            this._addProperty('format', new view.ValueTextView(this._host, model.format));
        }
        if (model.producer) {
            this._addProperty('producer', new view.ValueTextView(this._host, model.producer));
        }
        if (model.name) {
            this._addProperty('name', new view.ValueTextView(this._host, model.name));
        }
        if (model.version) {
            this._addProperty('version', new view.ValueTextView(this._host, model.version));
        }
        if (model.description) {
            this._addProperty('description', new view.ValueTextView(this._host, model.description));
        }
        if (model.domain) {
            this._addProperty('domain', new view.ValueTextView(this._host, model.domain));
        }
        if (model.imports) {
            this._addProperty('imports', new view.ValueTextView(this._host, model.imports));
        }
        if (model.runtime) {
            this._addProperty('runtime', new view.ValueTextView(this._host, model.runtime));
        }
        if (model.metadata) {
            for (const entry of model.metadata) {
                this._addProperty(entry.name, new view.ValueTextView(this._host, entry.value));
            }
        }
        const graphs = Array.isArray(model.graphs) ? model.graphs : [];
        if (graphs.length > 1) {
            const graphSelector = new view.SelectView(this._host, model.graphs, graph);
            graphSelector.on('change', (sender, data) => {
                this.emit('update-active-graph', data);
            });
            this._addProperty('subgraph', graphSelector);
        }

        if (graph) {
            if (graph.version) {
                this._addProperty('version', new view.ValueTextView(this._host, graph.version));
            }
            if (graph.type) {
                this._addProperty('type', new view.ValueTextView(this._host, graph.type));
            }
            if (graph.tags) {
                this._addProperty('tags', new view.ValueTextView(this._host, graph.tags));
            }
            if (graph.description) {
                this._addProperty('description', new view.ValueTextView(this._host, graph.description));
            }
            if (Array.isArray(graph.inputs) && graph.inputs.length > 0) {
                this._addHeader('Inputs');
                for (const input of graph.inputs) {
                    this.addArgument(input.name, input);
                }
            }
            if (Array.isArray(graph.outputs) && graph.outputs.length > 0) {
                this._addHeader('Outputs');
                for (const output of graph.outputs) {
                    this.addArgument(output.name, output);
                }
            }
        }

        const separator = this._host.document.createElement('div');
        separator.className = 'sidebar-view-separator';
        this._elements.push(separator);
    }

    render() {
        return this._elements;
    }

    _addHeader(title) {
        const headerElement = this._host.document.createElement('div');
        headerElement.className = 'sidebar-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    _addProperty(name, value) {
        const item = new view.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    addArgument(name, argument) {
        const value = new view.ParameterView(this._host, argument);
        value.toggle();
        const item = new view.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }
};

view.DocumentationSidebar = class extends view.Control {

    constructor(host, type) {
        super();
        this._host = host;
        this._type = type;
    }

    render() {
        if (!this._elements) {
            this._elements = [];

            const type = view.Documentation.format(this._type);

            const element = this._host.document.createElement('div');
            element.setAttribute('class', 'sidebar-view-documentation');

            this._append(element, 'h1', type.name);

            if (type.summary) {
                this._append(element, 'p', type.summary);
            }

            if (type.description) {
                this._append(element, 'p', type.description);
            }

            if (Array.isArray(type.attributes) && type.attributes.length > 0) {
                this._append(element, 'h2', 'Attributes');
                const attributes = this._append(element, 'dl');
                for (const attribute of type.attributes) {
                    this._append(attributes, 'dt', attribute.name + (attribute.type ? ': <tt>' + attribute.type + '</tt>' : ''));
                    this._append(attributes, 'dd', attribute.description);
                }
                element.appendChild(attributes);
            }

            if (Array.isArray(type.inputs) && type.inputs.length > 0) {
                this._append(element, 'h2', 'Inputs' + (type.inputs_range ? ' (' + type.inputs_range + ')' : ''));
                const inputs = this._append(element, 'dl');
                for (const input of type.inputs) {
                    this._append(inputs, 'dt', input.name + (input.type ? ': <tt>' + input.type + '</tt>' : '') + (input.option ? ' (' + input.option + ')' : ''));
                    this._append(inputs, 'dd', input.description);
                }
            }

            if (Array.isArray(type.outputs) && type.outputs.length > 0) {
                this._append(element, 'h2', 'Outputs' + (type.outputs_range ? ' (' + type.outputs_range + ')' : ''));
                const outputs = this._append(element, 'dl');
                for (const output of type.outputs) {
                    this._append(outputs, 'dt', output.name + (output.type ? ': <tt>' + output.type + '</tt>' : '') + (output.option ? ' (' + output.option + ')' : ''));
                    this._append(outputs, 'dd', output.description);
                }
            }

            if (Array.isArray(type.type_constraints) && type.type_constraints.length > 0) {
                this._append(element, 'h2', 'Type Constraints');
                const type_constraints = this._append(element, 'dl');
                for (const type_constraint of type.type_constraints) {
                    this._append(type_constraints, 'dt', type_constraint.type_param_str + ': ' + type_constraint.allowed_type_strs.map((item) => '<tt>' + item + '</tt>').join(', '));
                    this._append(type_constraints, 'dd', type_constraint.description);
                }
            }

            if (Array.isArray(type.examples) && type.examples.length > 0) {
                this._append(element, 'h2', 'Examples');
                for (const example of type.examples) {
                    this._append(element, 'h3', example.summary);
                    this._append(element, 'pre', example.code);
                }
            }

            if (Array.isArray(type.references) && type.references.length > 0) {
                this._append(element, 'h2', 'References');
                const references = this._append(element, 'ul');
                for (const reference of type.references) {
                    this._append(references, 'li', reference.description);
                }
            }

            if (type.domain && type.version && type.support_level) {
                this._append(element, 'h2', 'Support');
                this._append(element, 'dl', 'In domain <tt>' + type.domain + '</tt> since version <tt>' + type.version + '</tt> at support level <tt>' + type.support_level + '</tt>.');
            }

            if (this._host.type === 'Electron') {
                element.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        const url = e.target.href;
                        if (url.startsWith('http://') || url.startsWith('https://')) {
                            e.preventDefault();
                            this.emit('navigate', { link: url });
                        }
                    }
                });
            }

            this._elements = [ element ];

            const separator = this._host.document.createElement('div');
            separator.className = 'sidebar-view-separator';
            this._elements.push(separator);
        }
        return this._elements;
    }

    _append(parent, type, content) {
        const element = this._host.document.createElement(type);
        if (content) {
            element.innerHTML = content;
        }
        parent.appendChild(element);
        return element;
    }
};

view.FindSidebar = class extends view.Control {

    constructor(host, element, graph) {
        super();
        this._host = host;
        this._graphElement = element;
        this._graph = graph;
        this._contentElement = this._host.document.createElement('div');
        this._contentElement.setAttribute('class', 'sidebar-view-find');
        this._searchElement = this._host.document.createElement('input');
        this._searchElement.setAttribute('id', 'search');
        this._searchElement.setAttribute('type', 'text');
        this._searchElement.setAttribute('spellcheck', 'false');
        this._searchElement.setAttribute('placeholder', 'Search...');
        this._searchElement.setAttribute('style', 'width: 100%');
        this._searchElement.addEventListener('input', (e) => {
            this.update(e.target.value);
            this.emit('search-text-changed', e.target.value);
        });
        this._resultElement = this._host.document.createElement('ol');
        this._resultElement.addEventListener('click', (e) => {
            this.select(e);
        });
        this._contentElement.appendChild(this._searchElement);
        this._contentElement.appendChild(this._resultElement);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    emit(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    select(e) {
        const selection = [];
        const id = e.target.id;

        const nodesElement = this._graphElement.getElementById('nodes');
        let nodeElement = nodesElement.firstChild;
        while (nodeElement) {
            if (nodeElement.id == id) {
                selection.push(nodeElement);
            }
            nodeElement = nodeElement.nextSibling;
        }

        const edgePathsElement = this._graphElement.getElementById('edge-paths');
        let edgePathElement = edgePathsElement.firstChild;
        while (edgePathElement) {
            if (edgePathElement.id == id) {
                selection.push(edgePathElement);
            }
            edgePathElement = edgePathElement.nextSibling;
        }

        let initializerElement = this._graphElement.getElementById(id);
        if (initializerElement) {
            while (initializerElement.parentElement) {
                initializerElement = initializerElement.parentElement;
                if (initializerElement.id && initializerElement.id.startsWith('node-')) {
                    selection.push(initializerElement);
                    break;
                }
            }
        }

        if (selection.length > 0) {
            this.emit('select', selection);
        }
    }

    focus(searchText) {
        this._searchElement.focus();
        this._searchElement.value = '';
        this._searchElement.value = searchText;
        this.update(searchText);
    }

    update(searchText) {
        while (this._resultElement.lastChild) {
            this._resultElement.removeChild(this._resultElement.lastChild);
        }

        let terms = null;
        let callback = null;
        const unquote = searchText.match(new RegExp(/^'(.*)'|"(.*)"$/));
        if (unquote) {
            const term = unquote[1] || unquote[2];
            terms = [ term ];
            callback = (name) => {
                return term == name;
            };
        }
        else {
            terms = searchText.trim().toLowerCase().split(' ').map((term) => term.trim()).filter((term) => term.length > 0);
            callback = (name) => {
                return terms.every((term) => name.toLowerCase().indexOf(term) !== -1);
            };
        }

        const nodes = new Set();
        const edges = new Set();

        for (const node of this._graph.nodes.values()) {
            const label = node.label;
            const initializers = [];
            if (label.class === 'graph-node' || label.class === 'graph-input') {
                for (const input of label.inputs) {
                    for (const argument of input.arguments) {
                        if (argument.name && !edges.has(argument.name)) {
                            const match = (argument, term) => {
                                if (argument.name && argument.name.toLowerCase().indexOf(term) !== -1) {
                                    return true;
                                }
                                if (argument.type) {
                                    if (argument.type.dataType && term === argument.type.dataType.toLowerCase()) {
                                        return true;
                                    }
                                    if (argument.type.shape) {
                                        if (term === argument.type.shape.toString().toLowerCase()) {
                                            return true;
                                        }
                                        if (argument.type.shape && Array.isArray(argument.type.shape.dimensions)) {
                                            const dimensions = argument.type.shape.dimensions.map((dimension) => dimension ? dimension.toString().toLowerCase() : '');
                                            if (term === dimensions.join(',')) {
                                                return true;
                                            }
                                            if (dimensions.some((dimension) => term === dimension)) {
                                                return true;
                                            }
                                        }
                                    }
                                }
                                return false;
                            };
                            if (terms.every((term) => match(argument, term))) {
                                if (!argument.initializer) {
                                    const inputItem = this._host.document.createElement('li');
                                    inputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                                    inputItem.id = 'edge-' + argument.name;
                                    this._resultElement.appendChild(inputItem);
                                    edges.add(argument.name);
                                }
                                else {
                                    initializers.push(argument);
                                }
                            }
                        }
                    }
                }
            }
            if (label.class === 'graph-node') {
                const name = label.value.name;
                const type = label.value.type.name;
                if (!nodes.has(label.id) &&
                    ((name && callback(name) || (type && callback(type))))) {
                    const nameItem = this._host.document.createElement('li');
                    nameItem.innerText = '\u25A2 ' + (name || '[' + type + ']');
                    nameItem.id = label.id;
                    this._resultElement.appendChild(nameItem);
                    nodes.add(label.id);
                }
            }
            for (const argument of initializers) {
                if (argument.name) {
                    const initializeItem = this._host.document.createElement('li');
                    initializeItem.innerText = '\u25A0 ' + argument.name.split('\n').shift(); // custom argument id
                    initializeItem.id = 'initializer-' + argument.name;
                    this._resultElement.appendChild(initializeItem);
                }
            }
        }

        for (const node of this._graph.nodes.values()) {
            const label = node.label;
            if (label.class === 'graph-node' || label.class === 'graph-output') {
                for (const output of label.outputs) {
                    for (const argument of output.arguments) {
                        if (argument.name && !edges.has(argument.name) && terms.every((term) => argument.name.toLowerCase().indexOf(term) != -1)) {
                            const outputItem = this._host.document.createElement('li');
                            outputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                            outputItem.id = 'edge-' + argument.name;
                            this._resultElement.appendChild(outputItem);
                            edges.add(argument.name);
                        }
                    }
                }
            }
        }

        this._resultElement.style.display = this._resultElement.childNodes.length != 0 ? 'block' : 'none';
    }

    get content() {
        return this._contentElement;
    }
};

view.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this._type = tensor.type;
        this._stride = tensor.stride;
        switch (tensor.layout) {
            case undefined:
            case '':
            case '<': {
                this._data = this._tensor.values;
                this._layout = '<';
                this._littleEndian = true;
                break;
            }
            case '>': {
                this._data = this._tensor.values;
                this._layout = '>';
                this._littleEndian = false;
                break;
            }
            case '|': {
                this._values = this._tensor.values;
                this._layout = '|';
                break;
            }
            case 'sparse': {
                this._indices = this._tensor.indices;
                this._values = this._tensor.values;
                this._layout = 'sparse';
                break;
            }
            case 'sparse.coo': {
                this._indices = this._tensor.indices;
                this._values = this._tensor.values;
                this._layout = 'sparse.coo';
                break;
            }
            default: {
                this._layout = tensor.layout;
                break;
            }
        }
        view.Tensor.dataTypes = view.Tensor.dataTypeSizes || new Map([
            [ 'boolean', 1 ],
            [ 'qint8', 1 ], [ 'qint16', 2 ], [ 'qint32', 4 ],
            [ 'quint8', 1 ], [ 'quint16', 2 ], [ 'quint32', 4 ],
            [ 'int8', 1 ], [ 'int16', 2 ], [ 'int32', 4 ], [ 'int64', 8 ],
            [ 'uint8', 1 ], [ 'uint16', 2 ], [ 'uint32', 4, ], [ 'uint64', 8 ],
            [ 'float16', 2 ], [ 'float32', 4 ], [ 'float64', 8 ], [ 'bfloat16', 2 ],
            [ 'complex64', 8 ], [ 'complex128', 15 ]
        ]);
    }

    get type() {
        return this._type;
    }

    get layout() {
        return this._layout;
    }

    get stride() {
        return this._stride;
    }

    get empty() {
        switch (this._layout) {
            case '<':
            case '>': {
                return !(Array.isArray(this._data) || this._data instanceof Uint8Array || this._data instanceof Int8Array) || this._data.length === 0;
            }
            case '|': {
                return !(Array.isArray(this._values) || ArrayBuffer.isView(this._values)) || this._values.length === 0;
            }
            case 'sparse':
            case 'sparse.coo': {
                return !this._values || this.indices || this._values.values.length === 0;
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    get value() {
        const context = this._context();
        context.limit = Number.MAX_SAFE_INTEGER;
        switch (context.layout) {
            case '<':
            case '>': {
                return this._decodeData(context, 0);
            }
            case '|': {
                return this._decodeValues(context, 0);
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    toString() {
        const context = this._context();
        context.limit = 10000;
        switch (context.layout) {
            case '<':
            case '>': {
                const value = this._decodeData(context, 0);
                return view.Tensor._stringify(value, '', '    ');
            }
            case '|': {
                const value = this._decodeValues(context, 0);
                return view.Tensor._stringify(value, '', '    ');
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    _context() {
        if (this._layout !== '<' && this._layout !== '>' && this._layout !== '|' && this._layout !== 'sparse' && this._layout !== 'sparse.coo') {
            throw new Error("Tensor layout '" + this._layout + "' is not supported.");
        }
        const dataType = this._type.dataType;
        const context = {};
        context.layout = this._layout;
        context.dimensions = this._type.shape.dimensions.map((value) => !Number.isInteger(value) && value.toNumber ? value.toNumber() : value);
        context.dataType = dataType;
        const size = context.dimensions.reduce((a, b) => a * b, 1);
        switch (this._layout) {
            case '<':
            case '>': {
                context.data = (this._data instanceof Uint8Array || this._data instanceof Int8Array) ? this._data : this._data.peek();
                context.view = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
                if (view.Tensor.dataTypes.has(dataType)) {
                    context.itemsize = view.Tensor.dataTypes.get(dataType);
                    if (context.data.length < (context.itemsize * size)) {
                        throw new Error('Invalid tensor data size.');
                    }
                }
                else if (dataType.startsWith('uint') && !isNaN(parseInt(dataType.substring(4), 10))) {
                    context.dataType = 'uint';
                    context.bits = parseInt(dataType.substring(4), 10);
                    context.itemsize = 1;
                }
                else if (dataType.startsWith('int') && !isNaN(parseInt(dataType.substring(3), 10))) {
                    context.dataType = 'int';
                    context.bits = parseInt(dataType.substring(3), 10);
                    context.itemsize = 1;
                }
                else {
                    throw new Error("Tensor data type '" + dataType + "' is not implemented.");
                }
                break;
            }
            case '|': {
                context.data = this._values;
                if (!view.Tensor.dataTypes.has(dataType) && dataType !== 'string' && dataType !== 'object') {
                    throw new Error("Tensor data type '" + dataType + "' is not implemented.");
                }
                if (size !== this._values.length) {
                    throw new Error('Invalid tensor data length.');
                }
                break;
            }
            case 'sparse': {
                const indices = new view.Tensor(this._indices).value;
                const values = new view.Tensor(this._values).value;
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.layout = '|';
                break;
            }
            case 'sparse.coo': {
                const values = new view.Tensor(this._values).value;
                const data = new view.Tensor(this._indices).value;
                const dimensions = context.dimensions.length;
                let stride = 1;
                const strides = context.dimensions.slice().reverse().map((dim) => {
                    const value = stride;
                    stride *= dim;
                    return value;
                }).reverse();
                const indices = new Uint32Array(values.length);
                for (let i = 0; i < dimensions; i++) {
                    const stride = strides[i];
                    const dimension = data[i];
                    for (let i = 0; i < indices.length; i++) {
                        indices[i] += dimension[i] * stride;
                    }
                }
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.layout = '|';
                break;
            }
            default: {
                throw new view.Tensor("Unsupported tensor layout '" + this._layout + "'.");
            }
        }
        context.index = 0;
        context.count = 0;
        return context;
    }

    _decodeSparse(dataType, dimensions, indices, values) {
        const size = dimensions.reduce((a, b) => a * b, 1);
        const array = new Array(size);
        switch (dataType) {
            case 'boolean':
                array.fill(false);
                break;
            default:
                array.fill(0);
                break;
        }
        if (indices.length > 0) {
            if (Object.prototype.hasOwnProperty.call(indices[0], 'low')) {
                for (let i = 0; i < indices.length; i++) {
                    const index = indices[i];
                    array[index.high === 0 ? index.low : index.toNumber()] = values[i];
                }
            }
            else {
                for (let i = 0; i < indices.length; i++) {
                    array[indices[i]] = values[i];
                }
            }
        }
        return array;
    }

    _decodeData(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [ 1 ] : context.dimensions;
        const size = dimensions[dimension];
        const dataType = context.dataType;
        const view = context.view;
        if (dimension == dimensions.length - 1) {
            const ellipsis = (context.count + size) > context.limit;
            const length = ellipsis ? context.limit - context.count : size;
            let i = context.index;
            const max = i + (length * context.itemsize);
            switch (dataType) {
                case 'boolean':
                    for (; i < max; i += 1) {
                        results.push(view.getUint8(i) === 0 ? false : true);
                    }
                    break;
                case 'qint8':
                case 'int8':
                    for (; i < max; i++) {
                        results.push(view.getInt8(i));
                    }
                    break;
                case 'qint16':
                case 'int16':
                    for (; i < max; i += 2) {
                        results.push(view.getInt16(i, this._littleEndian));
                    }
                    break;
                case 'qint32':
                case 'int32':
                    for (; i < max; i += 4) {
                        results.push(view.getInt32(i, this._littleEndian));
                    }
                    break;
                case 'int64':
                    for (; i < max; i += 8) {
                        results.push(view.getInt64(i, this._littleEndian));
                    }
                    break;
                case 'int':
                    for (; i < size; i++) {
                        results.push(view.getIntBits(i, context.bits));
                    }
                    break;
                case 'quint8':
                case 'uint8':
                    for (; i < max; i++) {
                        results.push(view.getUint8(i));
                    }
                    break;
                case 'quint16':
                case 'uint16':
                    for (; i < max; i += 2) {
                        results.push(view.getUint16(i, true));
                    }
                    break;
                case 'quint32':
                case 'uint32':
                    for (; i < max; i += 4) {
                        results.push(view.getUint32(i, true));
                    }
                    break;
                case 'uint64':
                    for (; i < max; i += 8) {
                        results.push(view.getUint64(i, true));
                    }
                    break;
                case 'uint':
                    for (; i < max; i++) {
                        results.push(view.getUintBits(i, context.bits));
                    }
                    break;
                case 'float16':
                    for (; i < max; i += 2) {
                        results.push(view.getFloat16(i, this._littleEndian));
                    }
                    break;
                case 'float32':
                    for (; i < max; i += 4) {
                        results.push(view.getFloat32(i, this._littleEndian));
                    }
                    break;
                case 'float64':
                    for (; i < max; i += 8) {
                        results.push(view.getFloat64(i, this._littleEndian));
                    }
                    break;
                case 'bfloat16':
                    for (; i < max; i += 2) {
                        results.push(view.getBfloat16(i, this._littleEndian));
                    }
                    break;
                case 'complex64':
                    for (; i < max; i += 8) {
                        results.push(view.getComplex64(i, this._littleEndian));
                        context.index += 8;
                    }
                    break;
                case 'complex128':
                    for (; i < size; i += 16) {
                        results.push(view.getComplex128(i, this._littleEndian));
                    }
                    break;
                default:
                    throw new Error("Unsupported tensor data type '" + dataType + "'.");
            }
            context.index = i;
            context.count += length;
            if (ellipsis) {
                results.push('...');
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count >= context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decodeData(context, dimension + 1));
            }
        }
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    _decodeValues(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [ 1 ] : context.dimensions;
        const size = dimensions[dimension];
        const dataType = context.dataType;
        if (dimension == dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (dataType) {
                    case 'boolean':
                        results.push(context.data[context.index] === 0 ? false : true);
                        break;
                    default:
                        results.push(context.data[context.index]);
                        break;
                }
                context.index++;
                context.count++;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decodeValues(context, dimension + 1));
            }
        }
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => view.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value === null) {
            return indentation + 'null';
        }
        switch (typeof value) {
            case 'boolean':
                return indentation + value.toString();
            case 'string':
                return indentation + '"' + value + '"';
            case 'number':
                if (value == Infinity) {
                    return indentation + 'Infinity';
                }
                if (value == -Infinity) {
                    return indentation + '-Infinity';
                }
                if (isNaN(value)) {
                    return indentation + 'NaN';
                }
                return indentation + value.toString();
            default:
                if (value && value.toString) {
                    return indentation + value.toString();
                }
                return indentation + '(undefined)';
        }
    }
};

view.Documentation = class {

    static format(source) {
        if (source) {
            const generator = new markdown.Generator();
            const target = {};
            if (source.name !== undefined) {
                target.name = source.name;
            }
            if (source.module !== undefined) {
                target.module = source.module;
            }
            if (source.category !== undefined) {
                target.category = source.category;
            }
            if (source.summary !== undefined) {
                target.summary = generator.html(source.summary);
            }
            if (source.description !== undefined) {
                target.description = generator.html(source.description);
            }
            if (Array.isArray(source.attributes)) {
                target.attributes = source.attributes.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.required !== undefined) {
                        target.required = source.required;
                    }
                    if (source.minimum !== undefined) {
                        target.minimum = source.minimum;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.src_type !== undefined) {
                        target.src_type = source.src_type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.inputs)) {
                target.inputs = source.inputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeListAttr;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.outputs)) {
                target.outputs = source.outputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type) {
                        target.type = source.type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.references)) {
                target.references = source.references.map((source) => {
                    if (source) {
                        target.description = generator.html(source.description);
                    }
                    return target;
                });
            }
            if (source.version !== undefined) {
                target.version = source.version;
            }
            if (source.operator !== undefined) {
                target.operator = source.operator;
            }
            if (source.identifier !== undefined) {
                target.identifier = source.identifier;
            }
            if (source.package !== undefined) {
                target.package = source.package;
            }
            if (source.support_level !== undefined) {
                target.support_level = source.support_level;
            }
            if (source.min_input !== undefined) {
                target.min_input = source.min_input;
            }
            if (source.max_input !== undefined) {
                target.max_input = source.max_input;
            }
            if (source.min_output !== undefined) {
                target.min_output = source.min_output;
            }
            if (source.max_input !== undefined) {
                target.max_output = source.max_output;
            }
            if (source.inputs_range !== undefined) {
                target.inputs_range = source.inputs_range;
            }
            if (source.outputs_range !== undefined) {
                target.outputs_range = source.outputs_range;
            }
            if (source.examples !== undefined) {
                target.examples = source.examples;
            }
            if (source.constants !== undefined) {
                target.constants = source.constants;
            }
            if (source.type_constraints !== undefined) {
                target.type_constraints = source.type_constraints;
            }
            return target;
        }
        return '';
    }
};

view.Formatter = class {

    constructor(value, type, quote) {
        this._value = value;
        this._type = type;
        this._quote = quote;
        this._values = new Set();
    }

    toString() {
        return this._format(this._value, this._type, this._quote);
    }

    _format(value, type, quote) {

        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'type') {
            return value.__module__ + '.' + value.__name__;
        }
        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'function') {
            return value.__module__ + '.' + value.__name__;
        }
        if (typeof value === 'function') {
            return value();
        }
        if (value && (value instanceof base.Int64 || value instanceof base.Uint64)) {
            return value.toString();
        }
        if (Number.isNaN(value)) {
            return 'NaN';
        }
        switch (type) {
            case 'shape':
                return value ? value.toString() : '(null)';
            case 'shape[]':
                if (value && !Array.isArray(value)) {
                    throw new Error("Invalid shape '" + JSON.stringify(value) + "'.");
                }
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'graph':
                return value ? value.name : '(null)';
            case 'graph[]':
                return value ? value.map((graph) => graph.name).join(', ') : '(null)';
            case 'tensor':
                if (value && value.type && value.type.shape && value.type.shape.dimensions && value.type.shape.dimensions.length == 0) {
                    return value.toString();
                }
                return '[...]';
            case 'function':
                return value.type.name;
            case 'function[]':
                return value ? value.map((item) => item.type.name).join(', ') : '(null)';
            case 'type':
                return value ? value.toString() : '(null)';
            case 'type[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            default:
                break;
        }
        if (typeof value === 'string' && (!type || type != 'string')) {
            return quote ? '"' + value + '"' : value;
        }
        if (Array.isArray(value)) {
            if (value.length == 0) {
                return quote ? '[]' : '';
            }
            let ellipsis = false;
            if (value.length > 1000) {
                value = value.slice(0, 1000);
                ellipsis = true;
            }
            const itemType = (type && type.endsWith('[]')) ? type.substring(0, type.length - 2) : null;
            const array = value.map((item) => {
                if (item && (item instanceof base.Int64 || item instanceof base.Uint64)) {
                    return item.toString();
                }
                if (Number.isNaN(item)) {
                    return 'NaN';
                }
                const quote = !itemType || itemType === 'string';
                return this._format(item, itemType, quote);
            });
            if (ellipsis) {
                array.push('\u2026');
            }
            return quote ? [ '[', array.join(', '), ']' ].join(' ') : array.join(', ');
        }
        if (value === null) {
            return quote ? 'null' : '';
        }
        if (value === undefined) {
            return 'undefined';
        }
        if (value !== Object(value)) {
            return value.toString();
        }
        if (this._values.has(value)) {
            return '\u2026';
        }
        this._values.add(value);
        let list = null;
        const entries = Object.entries(value).filter((entry) => !entry[0].startsWith('__') && !entry[0].endsWith('__'));
        if (entries.length == 1) {
            list = [ this._format(entries[0][1], null, true) ];
        }
        else {
            list = new Array(entries.length);
            for (let i = 0; i < entries.length; i++) {
                const entry = entries[i];
                list[i] = entry[0] + ': ' + this._format(entry[1], null, true);
            }
        }
        let objectType = value.__type__;
        if (!objectType && value.constructor.name && value.constructor.name !== 'Object') {
            objectType = value.constructor.name;
        }
        if (objectType) {
            return objectType + (list.length == 0 ? '()' : [ '(', list.join(', '), ')' ].join(''));
        }
        switch (list.length) {
            case 0:
                return quote ? '()' : '';
            case 1:
                return list[0];
            default:
                return quote ? [ '(', list.join(', '), ')' ].join(' ') : list.join(', ');
        }
    }
};

const markdown = {};

markdown.Generator = class {

    constructor() {
        this._newlineRegExp = /^\n+/;
        this._codeRegExp = /^( {4}[^\n]+\n*)+/;
        this._fencesRegExp = /^ {0,3}(`{3,}(?=[^`\n]*\n)|~{3,})([^\n]*)\n(?:|([\s\S]*?)\n)(?: {0,3}\1[~`]* *(?:\n+|$)|$)/;
        this._hrRegExp = /^ {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)/;
        this._headingRegExp = /^ {0,3}(#{1,6}) +([^\n]*?)(?: +#+)? *(?:\n+|$)/;
        this._blockquoteRegExp = /^( {0,3}> ?(([^\n]+(?:\n(?! {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--))[^\n]+)*)|[^\n]*)(?:\n|$))+/;
        this._listRegExp = /^( {0,3})((?:[*+-]|\d{1,9}[.)])) [\s\S]+?(?:\n+(?=\1?(?:(?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$))|\n+(?= {0,3}\[((?!\s*\])(?:\\[[\]]|[^[\]])+)\]: *\n? *<?([^\s>]+)>?(?:(?: +\n? *| *\n *)((?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))))? *(?:\n+|$))|\n{2,}(?! )(?!\1(?:[*+-]|\d{1,9}[.)]) )\n*|\s*$)/;
        this._htmlRegExp = /^ {0,3}(?:<(script|pre|style)[\s>][\s\S]*?(?:<\/\1>[^\n]*\n+|$)|<!--(?!-?>)[\s\S]*?(?:-->|$)[^\n]*(\n+|$)|<\?[\s\S]*?(?:\?>\n*|$)|<![A-Z][\s\S]*?(?:>\n*|$)|<!\[CDATA\[[\s\S]*?(?:\]\]>\n*|$)|<\/?(address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)[\s\S]*?(?:\n{2,}|$)|<(?!script|pre|style)([a-z][\w-]*)(?: +[a-zA-Z:_][\w.:-]*(?: *= *"[^"\n]*"| *= *'[^'\n]*'| *= *[^\s"'=<>`]+)?)*? *\/?>(?=[ \t]*(?:\n|$))[\s\S]*?(?:\n{2,}|$)|<\/(?!script|pre|style)[a-z][\w-]*\s*>(?=[ \t]*(?:\n|$))[\s\S]*?(?:\n{2,}|$))/i;
        this._defRegExp = /^ {0,3}\[((?!\s*\])(?:\\[[\]]|[^[\]])+)\]: *\n? *<?([^\s>]+)>?(?:(?: +\n? *| *\n *)((?:"(?:\\"?|[^"\\])*"|'[^'\n]*(?:\n[^'\n]+)*\n?'|\([^()]*\))))? *(?:\n+|$)/;
        this._nptableRegExp = /^ *([^|\n ].*\|.*)\n {0,3}([-:]+ *\|[-| :]*)(?:\n((?:(?!\n| {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {4}[^\n]| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--)).*(?:\n|$))*)\n*|$)/;
        this._tableRegExp = /^ *\|(.+)\n {0,3}\|?( *[-:]+[-| :]*)(?:\n *((?:(?!\n| {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {4}[^\n]| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--)).*(?:\n|$))*)\n*|$)/;
        this._lheadingRegExp = /^([^\n]+)\n {0,3}(=+|-+) *(?:\n+|$)/;
        this._textRegExp = /^[^\n]+/;
        this._bulletRegExp = /(?:[*+-]|\d{1,9}[.)])/;
        this._itemRegExp = /^( *)((?:[*+-]|\d{1,9}[.)])) ?[^\n]*(?:\n(?!\1(?:[*+-]|\d{1,9}[.)]) ?)[^\n]*)*/gm;
        this._paragraphRegExp = /^([^\n]+(?:\n(?! {0,3}((?:- *){3,}|(?:_ *){3,}|(?:\* *){3,})(?:\n+|$)| {0,3}#{1,6} | {0,3}>| {0,3}(?:`{3,}(?=[^`\n]*\n)|~{3,})[^\n]*\n| {0,3}(?:[*+-]|1[.)]) |<\/?(?:address|article|aside|base|basefont|blockquote|body|caption|center|col|colgroup|dd|details|dialog|dir|div|dl|dt|fieldset|figcaption|figure|footer|form|frame|frameset|h[1-6]|head|header|hr|html|iframe|legend|li|link|main|menu|menuitem|meta|nav|noframes|ol|optgroup|option|p|param|section|source|summary|table|tbody|td|tfoot|th|thead|title|tr|track|ul)(?: +|\n|\/?>)|<(?:script|pre|style|!--))[^\n]+)*)/;
        this._backpedalRegExp = /(?:[^?!.,:;*_~()&]+|\([^)]*\)|&(?![a-zA-Z0-9]+;$)|[?!.,:;*_~)]+(?!$))+/;
        this._escapeRegExp = /^\\([!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~~|])/;
        this._escapesRegExp = /\\([!"#$%&'()*+,\-./:;<=>?@[\]\\^_`{|}~])/g;
        /* eslint-disable no-control-regex */
        this._autolinkRegExp = /^<([a-zA-Z][a-zA-Z0-9+.-]{1,31}:[^\s\x00-\x1f<>]*|[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+(@)[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)+(?![-_]))>/;
        this._linkRegExp = /^!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\(\s*(<(?:\\[<>]?|[^\s<>\\])*>|[^\s\x00-\x1f]*)(?:\s+("(?:\\"?|[^"\\])*"|'(?:\\'?|[^'\\])*'|\((?:\\\)?|[^)\\])*\)))?\s*\)/;
        /* eslint-enable no-control-regex */
        this._urlRegExp = /^((?:ftp|https?):\/\/|www\.)(?:[a-zA-Z0-9-]+\.?)+[^\s<]*|^[A-Za-z0-9._+-]+(@)[a-zA-Z0-9-_]+(?:\.[a-zA-Z0-9-_]*[a-zA-Z0-9])+(?![-_])/i;
        this._tagRegExp = /^<!--(?!-?>)[\s\S]*?-->|^<\/[a-zA-Z][\w:-]*\s*>|^<[a-zA-Z][\w-]*(?:\s+[a-zA-Z:_][\w.:-]*(?:\s*=\s*"[^"]*"|\s*=\s*'[^']*'|\s*=\s*[^\s"'=<>`]+)?)*?\s*\/?>|^<\?[\s\S]*?\?>|^<![a-zA-Z]+\s[\s\S]*?>|^<!\[CDATA\[[\s\S]*?\]\]>/;
        this._reflinkRegExp = /^!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\[(?!\s*\])((?:\\[[\]]?|[^[\]\\])+)\]/;
        this._nolinkRegExp = /^!?\[(?!\s*\])((?:\[[^[\]]*\]|\\[[\]]|[^[\]])*)\](?:\[\])?/;
        this._reflinkSearchRegExp = /!?\[((?:\[(?:\\.|[^[\]\\])*\]|\\.|`[^`]*`|[^[\]\\`])*?)\]\[(?!\s*\])((?:\\[[\]]?|[^[\]\\])+)\]|!?\[(?!\s*\])((?:\[[^[\]]*\]|\\[[\]]|[^[\]])*)\](?:\[\])?(?!\()/g;
        this._strongStartRegExp = /^(?:(\*\*(?=[*!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]))|\*\*)(?![\s])|__/;
        this._strongMiddleRegExp = /^\*\*(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|\*(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?\*)+?\*\*$|^__(?![\s])((?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|_(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?_)+?)__$/;
        this._strongEndAstRegExp = /[^!"#$%&'()+\-.,/:;<=>?@[\]`{|}~\s]\*\*(?!\*)|[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]\*\*(?!\*)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~_\s]|$))/g;
        this._strongEndUndRegExp = /[^\s]__(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g;
        this._emStartRegExp = /^(?:(\*(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]))|\*)(?![*\s])|_/;
        this._emMiddleRegExp = /^\*(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|\*(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^*]|\\\*)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?\*)+?\*$|^_(?![_\s])(?:(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)|_(?:(?!__[^_]*?__|\*\*\[^\*\]*?\*\*)(?:[^_]|\\_)|__[^_]*?__|\*\*\[^\*\]*?\*\*)*?_)+?_$/;
        this._emEndAstRegExp = /[^!"#$%&'()+\-.,/:;<=>?@[\]`{|}~\s]\*(?!\*)|[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~]\*(?!\*)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~_\s]|$))/g;
        this._emEndUndRegExp = /[^\s]_(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g,
        this._codespanRegExp = /^(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/;
        this._brRegExp = /^( {2,}|\\)\n(?!\s*$)/;
        this._delRegExp = /^~+(?=\S)([\s\S]*?\S)~+/;
        this._textspanRegExp = /^(`+|[^`])(?:(?= {2,}\n)|[\s\S]*?(?:(?=[\\<![`*~]|\b_|https?:\/\/|ftp:\/\/|www\.|$)|[^ ](?= {2,}\n)|[^a-zA-Z0-9.!#$%&'*+/=?_`{|}~-](?=[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+@))|(?=[a-zA-Z0-9.!#$%&'*+/=?_`{|}~-]+@))/;
        this._punctuationRegExp = /^([\s*!"#$%&'()+\-.,/:;<=>?@[\]`{|}~])/;
        this._blockSkipRegExp = /\[[^\]]*?\]\([^)]*?\)|`[^`]*?`|<[^>]*?>/g;
        this._escapeTestRegExp = /[&<>"']/;
        this._escapeReplaceRegExp = /[&<>"']/g;
        this._escapeTestNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/;
        this._escapeReplaceNoEncodeRegExp = /[<>"']|&(?!#?\w+;)/g;
        this._escapeReplacementsMap = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
    }

    html(source) {
        const tokens = [];
        const links = new Map();
        source = source.replace(/\r\n|\r/g, '\n').replace(/\t/g, '    ');
        this._tokenize(source, tokens, links, true);
        this._tokenizeBlock(tokens, links);
        const result = this._render(tokens, true);
        return result;
    }

    _tokenize(source, tokens, links, top) {
        source = source.replace(/^ +$/gm, '');
        while (source) {
            let match = this._newlineRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                if (match[0].length > 1) {
                    tokens.push({ type: 'space' });
                }
                continue;
            }
            match = this._codeRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const lastToken = tokens[tokens.length - 1];
                if (lastToken && lastToken.type === 'paragraph') {
                    lastToken.text += '\n' + match[0].trimRight();
                }
                else {
                    const text = match[0].replace(/^ {4}/gm, '').replace(/\n*$/, '');
                    tokens.push({ type: 'code', text: text });
                }
                continue;
            }
            match = this._fencesRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const language = match[2] ? match[2].trim() : match[2];
                let content = match[3] || '';
                const matchIndent = match[0].match(/^(\s+)(?:```)/);
                if (matchIndent !== null) {
                    const indent = matchIndent[1];
                    content = content.split('\n').map(node => {
                        const match = node.match(/^\s+/);
                        return (match !== null && match[0].length >= indent.length) ? node.slice(indent.length) : node;
                    }).join('\n');
                }
                tokens.push({ type: 'code', language: language, text: content });
                continue;
            }
            match = this._headingRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'heading', depth: match[1].length, text: match[2] });
                continue;
            }
            match = this._nptableRegExp.exec(source);
            if (match) {
                const header = this._splitCells(match[1].replace(/^ *| *\| *$/g, ''));
                const align = match[2].replace(/^ *|\| *$/g, '').split(/ *\| */);
                if (header.length === align.length) {
                    const cells = match[3] ? match[3].replace(/\n$/, '').split('\n') : [];
                    const token = { type: 'table', header: header, align: align, cells: cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        }
                        else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        }
                        else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        }
                        else {
                            token.align[i] = null;
                        }
                    }
                    token.cells = token.cells.map((cell) => this._splitCells(cell, token.header.length));
                    source = source.substring(token.raw.length);
                    tokens.push(token);
                    continue;
                }
            }
            match = this._hrRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'hr' });
                continue;
            }
            match = this._blockquoteRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = match[0].replace(/^ *> ?/gm, '');
                tokens.push({ type: 'blockquote', text: text, tokens: this._tokenize(text, [], links, top) });
                continue;
            }
            match = this._listRegExp.exec(source);
            if (match) {
                let raw = match[0];
                const bull = match[2];
                const ordered = bull.length > 1;
                const parent = bull[bull.length - 1] === ')';
                const list = { type: 'list', raw: raw, ordered: ordered, start: ordered ? +bull.slice(0, -1) : '', loose: false, items: [] };
                const itemMatch = match[0].match(this._itemRegExp);
                let next = false;
                const length = itemMatch.length;
                for (let i = 0; i < length; i++) {
                    let item = itemMatch[i];
                    raw = item;
                    let space = item.length;
                    item = item.replace(/^ *([*+-]|\d+[.)]) ?/, '');
                    if (~item.indexOf('\n ')) {
                        space -= item.length;
                        item = item.replace(new RegExp('^ {1,' + space + '}', 'gm'), '');
                    }
                    if (i !== length - 1) {
                        const bullet = this._bulletRegExp.exec(itemMatch[i + 1])[0];
                        if (ordered ? bullet.length === 1 || (!parent && bullet[bullet.length - 1] === ')') : (bullet.length > 1)) {
                            const addBack = itemMatch.slice(i + 1).join('\n');
                            list.raw = list.raw.substring(0, list.raw.length - addBack.length);
                            i = length - 1;
                        }
                    }
                    let loose = next || /\n\n(?!\s*$)/.test(item);
                    if (i !== length - 1) {
                        next = item.charAt(item.length - 1) === '\n';
                        if (!loose) {
                            loose = next;
                        }
                    }
                    if (loose) {
                        list.loose = true;
                    }
                    const task = /^\[[ xX]\] /.test(item);
                    let checked = undefined;
                    if (task) {
                        checked = item[1] !== ' ';
                        item = item.replace(/^\[[ xX]\] +/, '');
                    }
                    list.items.push({ type: 'list_item', raw, task: task, checked: checked, loose: loose, text: item });
                }
                source = source.substring(list.raw.length);
                for (const item of list.items) {
                    item.tokens = this._tokenize(item.text, [], links, false);
                }
                tokens.push(list);
                continue;
            }
            match = this._htmlRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'html', pre: (match[1] === 'pre' || match[1] === 'script' || match[1] === 'style'), text: match[0] });
                continue;
            }
            if (top) {
                match = this._defRegExp.exec(source);
                if (match) {
                    source = source.substring(match[0].length);
                    match[3] = match[3] ? match[3].substring(1, match[3].length - 1) : match[3];
                    const tag = match[1].toLowerCase().replace(/\s+/g, ' ');
                    if (!links.has(tag)) {
                        links.set(tag, { href: match[2], title: match[3] });
                    }
                    continue;
                }
            }
            match = this._tableRegExp.exec(source);
            if (match) {
                const header = this._splitCells(match[1].replace(/^ *| *\| *$/g, ''));
                const align = match[2].replace(/^ *|\| *$/g, '').split(/ *\| */);
                if (header.length === align.length) {
                    const cells = match[3] ? match[3].replace(/\n$/, '').split('\n') : [];
                    const token = { type: 'table', header: header, align: align, cells: cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        }
                        else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        }
                        else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        }
                        else {
                            token.align[i] = null;
                        }
                    }
                    token.cells = token.cells.map((cell) => this._splitCells(cell.replace(/^ *\| *| *\| *$/g, ''), token.header.length));
                    source = source.substring(token.raw.length);
                    tokens.push(token);
                    continue;
                }
            }
            match = this._lheadingRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'heading', depth: match[2].charAt(0) === '=' ? 1 : 2, text: match[1] });
                continue;
            }
            if (top) {
                match = this._paragraphRegExp.exec(source);
                if (match) {
                    source = source.substring(match[0].length);
                    tokens.push({ type: 'paragraph', text: match[1].charAt(match[1].length - 1) === '\n' ? match[1].slice(0, -1) : match[1] });
                    continue;
                }
            }
            match = this._textRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const lastToken = tokens[tokens.length - 1];
                if (lastToken && lastToken.type === 'text') {
                    lastToken.text += '\n' + match[0];
                }
                else {
                    tokens.push({ type: 'text', text: match[0] });
                }
                continue;
            }
            throw new Error("Unexpected '" + source.charCodeAt(0) + "'.");
        }
        return tokens;
    }

    _tokenizeInline(source, links, inLink, inRawBlock, prevChar) {
        const tokens = [];
        let maskedSource = source;
        if (links.size > 0) {
            while (maskedSource) {
                const match = this._reflinkSearchRegExp.exec(maskedSource);
                if (match) {
                    if (links.has(match[0].slice(match[0].lastIndexOf('[') + 1, -1))) {
                        maskedSource = maskedSource.slice(0, match.index) + '[' + 'a'.repeat(match[0].length - 2) + ']' + maskedSource.slice(this._reflinkSearchRegExp.lastIndex);
                    }
                    continue;
                }
                break;
            }
        }
        while (maskedSource) {
            const match = this._blockSkipRegExp.exec(maskedSource);
            if (match) {
                maskedSource = maskedSource.slice(0, match.index) + '[' + 'a'.repeat(match[0].length - 2) + ']' + maskedSource.slice(this._blockSkipRegExp.lastIndex);
                continue;
            }
            break;
        }
        while (source) {
            let match = this._escapeRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'escape', text: this._escape(match[1]) });
                continue;
            }
            match = this._tagRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                if (!inLink && /^<a /i.test(match[0])) {
                    inLink = true;
                }
                else if (inLink && /^<\/a>/i.test(match[0])) {
                    inLink = false;
                }
                if (!inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = true;
                }
                else if (inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = false;
                }
                tokens.push({ type: 'html', raw: match[0], text: match[0] });
                continue;
            }
            match = this._linkRegExp.exec(source);
            if (match) {
                let index = -1;
                const ref = match[2];
                if (ref.indexOf(')') !== -1) {
                    let level = 0;
                    for (let i = 0; i < ref.length; i++) {
                        switch (ref[i]) {
                            case '\\':
                                i++;
                                break;
                            case '(':
                                level++;
                                break;
                            case ')':
                                level--;
                                if (level < 0) {
                                    index = i;
                                    i = ref.length;
                                }
                                break;
                            default:
                                break;
                        }
                    }
                }
                if (index > -1) {
                    const length = (match[0].indexOf('!') === 0 ? 5 : 4) + match[1].length + index;
                    match[2] = match[2].substring(0, index);
                    match[0] = match[0].substring(0, length).trim();
                    match[3] = '';
                }
                const title = (match[3] ? match[3].slice(1, -1) : '').replace(this._escapesRegExp, '$1');
                const href = match[2].trim().replace(/^<([\s\S]*)>$/, '$1').replace(this._escapesRegExp, '$1');
                const token = this._outputLink(match, href, title);
                source = source.substring(match[0].length);
                if (token.type === 'link') {
                    token.tokens = this._tokenizeInline(token.text, links, true, inRawBlock, '');
                }
                tokens.push(token);
                continue;
            }
            match = this._reflinkRegExp.exec(source) || this._nolinkRegExp.exec(source);
            if (match) {
                let link = (match[2] || match[1]).replace(/\s+/g, ' ');
                link = links.get(link.toLowerCase());
                if (!link || !link.href) {
                    const text = match[0].charAt(0);
                    source = source.substring(text.length);
                    tokens.push({ type: 'text', text: text });
                }
                else {
                    source = source.substring(match[0].length);
                    const token = this._outputLink(match, link);
                    if (token.type === 'link') {
                        token.tokens = this._tokenizeInline(token.text, links, true, inRawBlock, '');
                    }
                    tokens.push(token);
                }
                continue;
            }
            match = this._strongStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '**' ? this._strongEndAstRegExp : this._strongEndUndRegExp;
                endReg.lastIndex = 0;
                let cap;
                while ((match = endReg.exec(masked)) != null) {
                    cap = this._strongMiddleRegExp.exec(masked.slice(0, match.index + 3));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.substring(2, cap[0].length - 2);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'strong', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._emStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '*' ? this._emEndAstRegExp : this._emEndUndRegExp;
                endReg.lastIndex = 0;
                let cap;
                while ((match = endReg.exec(masked)) != null) {
                    cap = this._emMiddleRegExp.exec(masked.slice(0, match.index + 2));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.slice(1, cap[0].length - 1);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'em', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._codespanRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                let content = match[2].replace(/\n/g, ' ');
                if (/[^ ]/.test(content) && content.startsWith(' ') && content.endsWith(' ')) {
                    content = content.substring(1, content.length - 1);
                }
                tokens.push({ type: 'codespan', text: this._encode(content) });
                continue;
            }
            match = this._brRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                tokens.push({ type: 'br' });
                continue;
            }
            match = this._delRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = match[1];
                tokens.push({ type: 'del', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                continue;
            }
            match = this._autolinkRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = this._escape(match[1]);
                const href = match[2] === '@' ? 'mailto:' + text : text;
                tokens.push({ type: 'link', text: text, href: href, tokens: [ { type: 'text', raw: text, text } ] });
                continue;
            }
            if (!inLink) {
                match = this._urlRegExp.exec(source);
                if (match) {
                    const email = match[2] === '@';
                    if (!email) {
                        let prevCapZero;
                        do {
                            prevCapZero = match[0];
                            match[0] = this._backpedalRegExp.exec(match[0])[0];
                        } while (prevCapZero !== match[0]);
                    }
                    const text = this._escape(match[0]);
                    const href = email ? ('mailto:' + text) : (match[1] === 'www.' ? 'http://' + text : text);
                    source = source.substring(match[0].length);
                    tokens.push({ type: 'link', text: text, href: href, tokens: [ { type: 'text', text: text } ] });
                    continue;
                }
            }
            match = this._textspanRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                prevChar = match[0].slice(-1);
                tokens.push({ type: 'text' , text: inRawBlock ? match[0] : this._escape(match[0]) });
                continue;
            }
            throw new Error("Unexpected '" + source.charCodeAt(0) + "'.");
        }
        return tokens;
    }

    _tokenizeBlock(tokens, links) {
        for (const token of tokens) {
            switch (token.type) {
                case 'paragraph':
                case 'text':
                case 'heading': {
                    token.tokens  = this._tokenizeInline(token.text, links, false, false, '');
                    break;
                }
                case 'table': {
                    token.tokens = {};
                    token.tokens.header = token.header.map((header) => this._tokenizeInline(header, links, false, false, ''));
                    token.tokens.cells = token.cells.map((cell) => cell.map((row) => this._tokenizeInline(row, links, false, false, '')));
                    break;
                }
                case 'blockquote': {
                    this._tokenizeBlock(token.tokens, links);
                    break;
                }
                case 'list': {
                    for (const item of token.items) {
                        this._tokenizeBlock(item.tokens, links);
                    }
                    break;
                }
                default: {
                    break;
                }
            }
        }
    }

    _render(tokens, top) {
        let html = '';
        while (tokens.length > 0) {
            const token = tokens.shift();
            switch (token.type) {
                case 'space': {
                    continue;
                }
                case 'hr': {
                    html += '<hr>\n';
                    continue;
                }
                case 'heading': {
                    const level = token.depth;
                    html += '<h' + level + '">' + this._renderInline(token.tokens) + '</h' + level + '>\n';
                    continue;
                }
                case 'code': {
                    const code = token.text;
                    const language = (token.language || '').match(/\S*/)[0];
                    html += '<pre><code' + (language ? ' class="' + 'language-' + this._encode(language) + '"' : '') + '>' + (token.escaped ? code : this._encode(code)) + '</code></pre>\n';
                    continue;
                }
                case 'table': {
                    let header = '';
                    let cell = '';
                    for (let j = 0; j < token.header.length; j++) {
                        const content = this._renderInline(token.tokens.header[j]);
                        const align = token.align[j];
                        cell += '<th' + (align ? ' align="' + align + '"' : '') + '>' + content + '</th>\n';
                    }
                    header += '<tr>\n' + cell + '</tr>\n';
                    let body = '';
                    for (let j = 0; j < token.cells.length; j++) {
                        const row = token.tokens.cells[j];
                        cell = '';
                        for (let k = 0; k < row.length; k++) {
                            const content = this._renderInline(row[k]);
                            const align = token.align[k];
                            cell += '<td' + (align ? ' align="' + align + '"' : '') + '>' + content + '</td>\n';
                        }
                        body += '<tr>\n' + cell + '</tr>\n';
                    }
                    html += '<table>\n<thead>\n' + header + '</thead>\n' + (body ? '<tbody>' + body + '</tbody>' : body) + '</table>\n';
                    continue;
                }
                case 'blockquote': {
                    html += '<blockquote>\n' + this._render(token.tokens, true) + '</blockquote>\n';
                    continue;
                }
                case 'list': {
                    const ordered = token.ordered;
                    const start = token.start;
                    const loose = token.loose;
                    let body = '';
                    for (const item of token.items) {
                        let itemBody = '';
                        if (item.task) {
                            const checkbox = '<input ' + (item.checked ? 'checked="" ' : '') + 'disabled="" type="checkbox"' + '> ';
                            if (loose) {
                                if (item.tokens.length > 0 && item.tokens[0].type === 'text') {
                                    item.tokens[0].text = checkbox + ' ' + item.tokens[0].text;
                                    if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === 'text') {
                                        item.tokens[0].tokens[0].text = checkbox + ' ' + item.tokens[0].tokens[0].text;
                                    }
                                }
                                else {
                                    item.tokens.unshift({ type: 'text', text: checkbox });
                                }
                            }
                            else {
                                itemBody += checkbox;
                            }
                        }
                        itemBody += this._render(item.tokens, loose);
                        body += '<li>' + itemBody + '</li>\n';
                    }
                    const type = (ordered ? 'ol' : 'ul');
                    html += '<' + type + (ordered && start !== 1 ? (' start="' + start + '"') : '') + '>\n' + body + '</' + type + '>\n';
                    continue;
                }
                case 'html': {
                    html += token.text;
                    continue;
                }
                case 'paragraph': {
                    html += '<p>' + this._renderInline(token.tokens) + '</p>\n';
                    continue;
                }
                case 'text': {
                    html += top ? '<p>' : '';
                    html += token.tokens ? this._renderInline(token.tokens) : token.text;
                    while (tokens.length > 0 && tokens[0].type === 'text') {
                        const token = tokens.shift();
                        html += '\n' + (token.tokens ? this._renderInline(token.tokens) : token.text);
                    }
                    html += top ? '</p>\n' : '';
                    continue;
                }
                default: {
                    throw new Error("Unexpected token type '" + token.type + "'.");
                }
            }
        }
        return html;
    }

    _renderInline(tokens) {
        let html = '';
        for (const token of tokens) {
            switch (token.type) {
                case 'escape':
                case 'html':
                case 'text': {
                    html += token.text;
                    break;
                }
                case 'link': {
                    const text = this._renderInline(token.tokens);
                    html += '<a href="' + token.href + '"' + (token.title ? ' title="' + token.title + '"' : '') + ' target="_blank">' + text + '</a>';
                    break;
                }
                case 'image': {
                    html += '<img src="' + token.href + '" alt="' + token.text + '"' + (token.title ? ' title="' + token.title + '"' : '') + '>';
                    break;
                }
                case 'strong': {
                    const text = this._renderInline(token.tokens);
                    html += '<strong>' + text + '</strong>';
                    break;
                }
                case 'em': {
                    const text = this._renderInline(token.tokens);
                    html += '<em>' + text + '</em>';
                    break;
                }
                case 'codespan': {
                    html += '<code>' + token.text + '</code>';
                    break;
                }
                case 'br': {
                    html += '<br>';
                    break;
                }
                case 'del': {
                    const text = this._renderInline(token.tokens);
                    html += '<del>' + text + '</del>';
                    break;
                }
                default: {
                    throw new Error("Unexpected token type '" + token.type + "'.");
                }
            }
        }
        return html;
    }

    _outputLink(match, href, title) {
        title = title ? this._escape(title) : null;
        const text = match[1].replace(/\\([[\]])/g, '$1');
        return match[0].charAt(0) !== '!' ?
            { type: 'link', href: href, title: title, text: text } :
            { type: 'image', href: href, title: title, text: this._escape(text) };
    }

    _splitCells(tableRow, count) {
        const row = tableRow.replace(/\|/g, (match, offset, str) => {
            let escaped = false;
            let position = offset;
            while (--position >= 0 && str[position] === '\\') {
                escaped = !escaped;
            }
            return escaped ? '|' : ' |';
        });
        const cells = row.split(/ \|/);
        if (cells.length > count) {
            cells.splice(count);
        }
        else {
            while (cells.length < count) {
                cells.push('');
            }
        }
        return cells.map((cell) => cell.trim().replace(/\\\|/g, '|'));
    }

    _encode(content) {
        if (this._escapeTestRegExp.test(content)) {
            return content.replace(this._escapeReplaceRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }

    _escape(content) {
        if (this._escapeTestNoEncodeRegExp.test(content)) {
            return content.replace(this._escapeReplaceNoEncodeRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }
};

view.ModelContext = class {

    constructor(context) {
        this._context = context;
        this._tags = new Map();
        this._content = new Map();
        let stream = context.stream;
        const entries = context.entries;
        if (!stream && entries && entries.size > 0) {
            this._entries = entries;
            this._format = '';
        }
        else {
            this._entries = new Map();
            const entry = context instanceof view.EntryContext;
            try {
                const archive = zip.Archive.open(stream, 'gzip');
                if (archive) {
                    this._entries = archive.entries;
                    this._format = 'gzip';
                    if (this._entries.size === 1) {
                        stream = this._entries.values().next().value;
                    }
                }
            }
            catch (error) {
                if (!entry) {
                    throw error;
                }
            }
            try {
                const formats = new Map([ [ 'zip', zip ], [ 'tar', tar ] ]);
                for (const pair of formats) {
                    const format = pair[0];
                    const module = pair[1];
                    const archive = module.Archive.open(stream);
                    if (archive) {
                        this._entries = archive.entries;
                        this._format = format;
                        break;
                    }
                }
            }
            catch (error) {
                if (!entry) {
                    throw error;
                }
            }
        }
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
        if (error && this.identifier) {
            error.context = this.identifier;
        }
        this._context.exception(error, fatal);
    }

    entries(format) {
        if (format !== undefined && format !== this._format) {
            return new Map();
        }
        return this._entries;
    }

    open(type) {
        if (!this._content.has(type)) {
            this._content.set(type, undefined);
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const signatures = [
                    [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ] // PyTorch
                ];
                const skip =
                    signatures.some((signature) => signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) ||
                    Array.from(this._tags).some((pair) => pair[0] !== 'flatbuffers' && pair[1].size > 0) ||
                    Array.from(this._content.values()).some((obj) => obj !== undefined);
                if (!skip) {
                    switch (type) {
                        case 'json': {
                            try {
                                const reader = json.TextReader.open(this.stream);
                                if (reader) {
                                    const obj = reader.read();
                                    this._content.set(type, obj);
                                }
                            }
                            catch (err) {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'json.gz': {
                            try {
                                const archive = zip.Archive.open(this.stream, 'gzip');
                                if (archive && archive.entries.size === 1) {
                                    const stream = archive.entries.values().next().value;
                                    const reader = json.TextReader.open(stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            }
                            catch (err) {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'pkl': {
                            let unpickler = null;
                            try {
                                const archive = zip.Archive.open(stream, 'zlib');
                                const data = archive ? archive.entries.get('') : stream;
                                let condition = false;
                                if (data.length > 2) {
                                    const head = data.peek(2);
                                    condition = head[0] === 0x80 && head[1] < 7;
                                    if (!condition) {
                                        data.seek(-1);
                                        const tail = data.peek(1);
                                        data.seek(0);
                                        condition = tail[0] === 0x2e;
                                    }
                                }
                                if (condition) {
                                    const signature = [ 0x80, undefined, 0x63, 0x5F, 0x5F, 0x74, 0x6F, 0x72, 0x63, 0x68, 0x5F, 0x5F, 0x2E]; // __torch__.
                                    const torch = signature.length <= data.length && data.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value);
                                    const execution = new python.Execution();
                                    execution.on('resolve', (_, name) => {
                                        if (!torch || !name.startsWith('__torch__.')) {
                                            this.exception(new view.Error("Unknown type name '" + name + "'."));
                                        }
                                    });
                                    const pickle = execution.__import__('pickle');
                                    unpickler = new pickle.Unpickler(data);
                                }
                            }
                            catch (err) {
                                // continue regardless of error
                            }
                            if (unpickler) {
                                unpickler.persistent_load = (saved_id) => saved_id;
                                const obj = unpickler.load();
                                this._content.set(type, obj);
                            }
                            break;
                        }
                        case 'hdf5': {
                            const file = hdf5.File.open(stream);
                            if (file && file.rootGroup && file.rootGroup.attributes) {
                                this._content.set(type, file.rootGroup);
                            }
                            break;
                        }
                        default: {
                            throw new view.Error("Unsupported open format type '" + type + "'.");
                        }
                    }
                }
                if (stream.position !== position) {
                    stream.seek(0);
                }
            }
        }
        return this._content.get(type);
    }

    tags(type) {
        if (!this._tags.has(type)) {
            let tags = new Map();
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const signatures = [
                    [ 0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A ], // HDF5
                    [ 0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19 ], // PyTorch
                    [ 0x50, 0x4b ], // Zip
                    [ 0x1f, 0x8b ] // Gzip
                ];
                const skip =
                    signatures.some((signature) => signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) ||
                    (Array.from(this._tags).some((pair) => pair[0] !== 'flatbuffers' && pair[1].size > 0) && type !== 'pb+') ||
                    Array.from(this._content.values()).some((obj) => obj !== undefined);
                if (!skip) {
                    try {
                        switch (type) {
                            case 'pbtxt': {
                                const reader = protobuf.TextReader.open(stream);
                                tags = reader ? reader.signature() : tags;
                                break;
                            }
                            case 'pb': {
                                const reader = protobuf.BinaryReader.open(stream);
                                tags = reader.signature();
                                break;
                            }
                            case 'pb+': {
                                const reader = protobuf.BinaryReader.open(stream);
                                tags = reader.decode();
                                break;
                            }
                            case 'flatbuffers': {
                                if (stream.length >= 8) {
                                    const buffer = stream.peek(Math.min(32, stream.length));
                                    const reader = flatbuffers.BinaryReader.open(buffer);
                                    const identifier = reader.identifier;
                                    if (identifier.length > 0) {
                                        tags.set('file_identifier', identifier);
                                    }
                                }
                                break;
                            }
                            case 'xml': {
                                const reader = xml.TextReader.open(stream);
                                if (reader) {
                                    const document = reader.peek();
                                    const element = document.documentElement;
                                    const namespaceURI = element.namespaceURI;
                                    const localName = element.localName;
                                    const name = namespaceURI ? namespaceURI + ':' + localName : localName;
                                    tags.set(name, element);
                                }
                                break;
                            }
                            default: {
                                throw new view.Error("Unsupported tags format type '" + type + "'.");
                            }
                        }
                    }
                    catch (error) {
                        tags.clear();
                    }
                }
                if (stream.position !== position) {
                    stream.seek(position);
                }
            }
            this._tags.set(type, tags);
        }
        return this._tags.get(type);
    }

    metadata(name) {
        return view.Metadata.open(this, name);
    }
};

view.EntryContext = class {

    constructor(host, entries, rootFolder, identifier, stream) {
        this._host = host;
        this._entries = new Map();
        if (entries) {
            for (const entry of entries) {
                if (entry[0].startsWith(rootFolder)) {
                    const name = entry[0].substring(rootFolder.length);
                    this._entries.set(name, entry[1]);
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
            const stream = this._entries.get(file);
            if (!stream) {
                return Promise.reject(new Error('File not found.'));
            }
            if (encoding) {
                const decoder = new TextDecoder(encoding);
                const buffer = stream.peek();
                const value = decoder.decode(buffer);
                return Promise.resolve(value);
            }
            return Promise.resolve(stream);
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
        this._extensions = new Set([ '.zip', '.tar', '.tar.gz', '.tgz', '.gz' ]);
        this._factories = [];
        this.register('./server', [ '.netron']);
        this.register('./pytorch', [ '.pt', '.pth', '.ptl', '.pt1', '.pyt', '.pyth', '.pkl', '.pickle', '.h5', '.t7', '.model', '.dms', '.tar', '.ckpt', '.chkpt', '.tckpt', '.bin', '.pb', '.zip', '.nn', '.torchmodel', '.torchscript', '.pytorch', '.ot', '.params', '.trt', '.ff', '.ptmf' ], [ '.model' ]);
        this.register('./onnx', [ '.onnx', '.onn', '.pb', '.onnxtxt', '.pbtxt', '.prototxt', '.txt', '.model', '.pt', '.pth', '.pkl', '.ort', '.ort.onnx', 'onnxmodel', 'ngf' ]);
        this.register('./mxnet', [ '.json', '.params' ], [ '.mar']);
        this.register('./coreml', [ '.mlmodel', '.bin', 'manifest.json', 'metadata.json', 'featuredescriptions.json', '.pb' ], [ '.mlpackage' ]);
        this.register('./caffe', [ '.caffemodel', '.pbtxt', '.prototxt', '.pt', '.txt' ]);
        this.register('./caffe2', [ '.pb', '.pbtxt', '.prototxt' ]);
        this.register('./torch', [ '.t7', '.net' ]);
        this.register('./tflite', [ '.tflite', '.lite', '.tfl', '.bin', '.pb', '.tmfile', '.h5', '.model', '.json', '.txt' ]);
        this.register('./circle', [ '.circle' ]);
        this.register('./tf', [ '.pb', '.meta', '.pbtxt', '.prototxt', '.txt', '.pt', '.json', '.index', '.ckpt', '.graphdef', '.pbmm', /.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/, /^events.out.tfevents./ ], [ '.zip' ]);
        this.register('./mediapipe', [ '.pbtxt' ]);
        this.register('./uff', [ '.uff', '.pb', '.pbtxt', '.uff.txt', '.trt', '.engine' ]);
        this.register('./tensorrt', [ '.trt', '.trtmodel', '.engine', '.model', '.txt', '.uff', '.pb', '.tmfile', '.onnx', '.pth', '.dnn', '.plan' ]);
        this.register('./numpy', [ '.npz', '.npy', '.pkl', '.pickle', '.model', '.model2' ]);
        this.register('./lasagne', [ '.pkl', '.pickle', '.joblib', '.model', '.pkl.z', '.joblib.z' ]);
        this.register('./lightgbm', [ '.txt', '.pkl', '.model' ]);
        this.register('./keras', [ '.h5', '.hd5', '.hdf5', '.keras', '.json', '.cfg', '.model', '.pb', '.pth', '.weights', '.pkl', '.lite', '.tflite', '.ckpt' ], [ '.zip' ]);
        this.register('./sklearn', [ '.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z' ]);
        this.register('./megengine', [ '.tm', '.mge' ]);
        this.register('./pickle', [ '.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z', '.pdstates', '.mge' ]);
        this.register('./cntk', [ '.model', '.cntk', '.cmf', '.dnn' ]);
        this.register('./paddle', [ '.pdmodel', '.pdiparams', '.pdparams', '.pdopt', '.paddle', '__model__', '.__model__', '.pbtxt', '.txt', '.tar', '.tar.gz', '.nb' ]);
        this.register('./bigdl', [ '.model', '.bigdl' ]);
        this.register('./darknet', [ '.cfg', '.model', '.txt', '.weights' ]);
        this.register('./weka', [ '.model' ]);
        this.register('./rknn', [ '.rknn', '.nb', '.onnx' ]);
        this.register('./dlc', [ '.dlc', 'model', '.params' ]);
        this.register('./armnn', [ '.armnn', '.json' ]);
        this.register('./mnn', ['.mnn']);
        this.register('./ncnn', [ '.param', '.bin', '.cfg.ncnn', '.weights.ncnn', '.ncnnmodel' ]);
        this.register('./tnn', [ '.tnnproto', '.tnnmodel' ]);
        this.register('./tengine', ['.tmfile']);
        this.register('./mslite', [ '.ms']);
        this.register('./barracuda', [ '.nn' ]);
        this.register('./dnn', [ '.dnn' ]);
        this.register('./xmodel', [ '.xmodel' ]);
        this.register('./kmodel', [ '.kmodel' ]);
        this.register('./flux', [ '.bson' ]);
        this.register('./dl4j', [ '.json', '.bin' ]);
        this.register('./openvino', [ '.xml', '.bin' ]);
        this.register('./mlnet', [ '.zip' ]);
        this.register('./acuity', [ '.json' ]);
        this.register('./imgdnn', [ '.dnn', 'params', '.json' ]);
        this.register('./flax', [ '.msgpack' ]);
        this.register('./om', [ '.om', '.onnx', '.pb', '.engine' ]);
        this.register('./nnabla', [ '.nntxt' ], [ '.nnp' ]);
        this.register('./hickle', [ '.h5', '.hkl' ]);
        this.register('./nnef', [ '.nnef', '.dat' ]);
        this.register('./cambricon', [ '.cambricon' ]);
        this.register('./onednn', [ '.json']);
    }

    register(id, factories, containers) {
        for (const extension of factories) {
            this._factories.push({ extension: extension, id: id });
            this._extensions.add(extension);
        }
        for (const extension of containers || []) {
            this._extensions.add(extension);
        }
    }

    open(context) {
        return this._openSignature(context).then((context) => {
            const modelContext = new view.ModelContext(context);
            /* eslint-disable consistent-return */
            return this._openContext(modelContext).then((model) => {
                if (model) {
                    return model;
                }
                const entries = modelContext.entries();
                if (entries && entries.size > 0) {
                    return this._openEntries(entries).then((context) => {
                        if (context) {
                            return this._openContext(context);
                        }
                        this._unsupported(modelContext);
                    });
                }
                this._unsupported(modelContext);
            });
            /* eslint-enable consistent-return */
        }).catch((error) => {
            if (error && context.identifier) {
                error.context = context.identifier;
            }
            throw error;
        });
    }

    _unsupported(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        const stream = context.stream;
        const callbacks = [
            (stream) => zip.Archive.open(stream, 'zip'),
            (stream) => tar.Archive.open(stream),
            (stream) => zip.Archive.open(stream, 'gzip')
        ];
        for (const callback of callbacks) {
            let archive = null;
            try {
                archive = callback(stream);
            }
            catch (error) {
                // continue regardless of error
            }
            if (archive) {
                throw new view.Error("Archive contains no model files.");
            }
        }
        const json = () => {
            const obj = context.open('json');
            if (obj) {
                const formats = [
                    { name: 'Netron metadata', tags: [ '[].name', '[].schema' ] },
                    { name: 'Netron metadata', tags: [ '[].name', '[].attributes' ] },
                    { name: 'Netron metadata', tags: [ '[].name', '[].category' ] },
                    { name: 'Netron test data', tags: [ '[].type', '[].target', '[].source', '[].format', '[].link' ] },
                    { name: 'Darkflow metadata', tags: [ 'net', 'type', 'model' ] },
                    { name: 'keras-yolo2 configuration', tags: [ 'model', 'train', 'valid' ] },
                    { name: 'Vulkan SwiftShader ICD manifest', tags: [ 'file_format_version', 'ICD' ] },
                    { name: 'DeepLearningExamples configuration', tags: [ 'attention_probs_dropout_prob', 'hidden_act', 'hidden_dropout_prob', 'hidden_size', ] },
                    { name: 'NuGet assets', tags: [ 'version', 'targets', 'packageFolders' ] },
                    { name: 'NuGet data', tags: [ 'format', 'restore', 'projects' ] },
                    { name: 'NPM package', tags: [ 'name', 'version', 'dependencies' ] },
                    { name: 'NetworkX adjacency_data', tags: [ 'directed', 'graph', 'nodes' ] },
                    { name: 'Waifu2x data', tags: [ 'name', 'arch_name', 'channels' ] },
                    { name: 'Waifu2x data', tags: [ '[].nInputPlane', '[].nOutputPlane', '[].weight', '[].bias' ] },
                    { name: 'Brain.js data', tags: [ 'type', 'sizes', 'layers' ] },
                    { name: 'Custom Vision metadata', tags: [ 'CustomVision.Metadata.Version' ] },
                    { name: 'W&B metadata', tags: [ 'program', 'host', 'executable' ] }

                ];
                const match = (obj, tag) => {
                    if (tag.startsWith('[].')) {
                        tag = tag.substring(3);
                        return (Array.isArray(obj) && obj.some((item) => Object.prototype.hasOwnProperty.call(item, tag)));
                    }
                    return Object.prototype.hasOwnProperty.call(obj, tag);
                };
                for (const format of formats) {
                    if (format.tags.every((tag) => match(obj, tag))) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.');
                    }
                }
                const content = JSON.stringify(obj).substring(0, 100).replace(/\s/, '').substring(0, 48) + '...';
                throw new view.Error("Unsupported JSON content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "'.");
            }
        };
        const pbtxt = () => {
            const formats = [
                { name: 'ImageNet LabelMap data', tags: [ 'entry', 'entry.target_class' ] },
                { name: 'StringIntLabelMapProto data', tags: [ 'item', 'item.id', 'item.name' ] },
                { name: 'caffe.LabelMap data', tags: [ 'item', 'item.name', 'item.label' ] },
                { name: 'Triton Inference Server configuration', tags: [ 'name', 'platform', 'input', 'output' ] },
                { name: 'TensorFlow OpList data', tags: [ 'op', 'op.name', 'op.input_arg' ] },
                { name: 'vitis.ai.proto.DpuModelParamList data', tags: [ 'model', 'model.name', 'model.kernel' ] },
                { name: 'object_detection.protos.DetectionModel data', tags: [ 'model', 'model.ssd' ] },
                { name: 'object_detection.protos.DetectionModel data', tags: [ 'model', 'model.faster_rcnn' ] },
                { name: 'tensorflow.CheckpointState data', tags: [ 'model_checkpoint_path', 'all_model_checkpoint_paths' ] },
                { name: 'apollo.perception.camera.traffic_light.detection.DetectionParam data', tags: [ 'min_crop_size', 'crop_method' ] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'caffe_ssd' ] }, // https://github.com/TexasInstruments/edgeai-mmdetection/blob/master/mmdet/utils/proto/mmdet_meta_arch.proto
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'tf_od_api_ssd' ] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'tidl_ssd' ] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'tidl_faster_rcnn' ] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'tidl_yolo' ] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: [ 'tidl_retinanet' ] },
                { name: 'domi.InsertNewOps data', tags: [ 'aipp_op' ] } // https://github.com/Ascend/parser/blob/development/parser/proto/insert_op.proto
            ];
            const tags = context.tags('pbtxt');
            if (tags.size > 0) {
                for (const format of formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        const error = new view.Error('Invalid file content. File contains ' + format.name + '.');
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const entries = [];
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') === -1));
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') !== -1));
                const content = entries.map((pair) => pair[1] === true ? pair[0] : pair[0] + ':' + JSON.stringify(pair[1])).join(',');
                throw new view.Error("Unsupported Protocol Buffers text content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "'.");
            }
        };
        const pb = () => {
            const tags = context.tags('pb+');
            if (Object.keys(tags).length > 0) {
                const formats = [
                    { name: 'sentencepiece.ModelProto data', tags: [[1,[[1,2],[2,5],[3,0]]],[2,[[1,2],[2,2],[3,0],[4,0],[5,2],[6,0],[7,2],[10,5],[16,0],[40,0],[41,0],[42,0],[43,0]]],[3,[]],[4,[]],[5,[]]] },
                    { name: 'mediapipe.BoxDetectorIndex data', tags: [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]] },
                    { name: 'third_party.tensorflow.python.keras.protobuf.SavedMetadata data', tags: [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]] },
                    { name: 'pblczero.Net data', tags: [[1,5],[2,2],[3,[[1,0],[2,0],[3,0]],[10,[[1,[]],[2,[]],[3,[]],[4,[]],[5,[]],[6,[]]]],[11,[]]]] } // https://github.com/LeelaChessZero/lczero-common/blob/master/proto/net.proto
                ];
                const match = (tags, schema) => {
                    for (const pair of schema) {
                        const key = pair[0];
                        const inner = pair[1];
                        const value = tags[key];
                        if (value === undefined) {
                            continue;
                        }
                        if (inner === false) {
                            return false;
                        }
                        if (Array.isArray(inner)) {
                            if (typeof value !== 'object' || !match(value, inner)) {
                                return false;
                            }
                        }
                        else if (inner !== value) {
                            if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return true;
                };
                const tags = context.tags('pb+');
                for (const format of formats) {
                    if (match(tags, format.tags)) {
                        const error = new view.Error('Invalid file content. File contains ' + format.name + '.');
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const format = (tags) => {
                    const content = Object.entries(tags).map((pair) => {
                        const key = pair[0];
                        const value = pair[1];
                        return key.toString() + ':' + (Object(value) === value ? '{' + format(value) + '}' : value.toString());
                    });
                    return content.join(',');
                };
                const content = format(tags);
                throw new view.Error("Unsupported Protocol Buffers content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "'.");
            }
        };
        const flatbuffers = () => {
            const tags = context.tags('flatbuffers');
            if (tags.has('file_identifier')) {
                const file_identifier = tags.get('file_identifier');
                const formats = [
                    { name: 'onnxruntime.experimental.fbs.InferenceSession data', identifier: 'ORTM' },
                    { name: 'tflite.Model data', identifier: 'TFL3' },
                    { name: 'FlatBuffers ENNC data', identifier: 'ENNC' },
                ];
                for (const format of formats) {
                    if (file_identifier === format.identifier) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.');
                    }
                }
            }
        };
        const xml = () => {
            const tags = context.tags('xml');
            if (tags.size > 0) {
                const formats = [
                    { name: 'OpenCV storage data', tags: [ 'opencv_storage' ] },
                    { name: 'XHTML markup', tags: [ 'http://www.w3.org/1999/xhtml:html' ] }
                ];
                for (const format of formats) {
                    if (format.tags.some((tag) => tags.has(tag))) {
                        const error = new view.Error('Invalid file content. File contains ' + format.name + '.');
                        error.content = context.identifier;
                        throw error;
                    }
                }
                throw new view.Error("Unsupported XML content '" + tags.keys().next().value + "'.");
            }
        };
        const unknown = () => {
            if (stream) {
                stream.seek(0);
                const buffer = stream.peek(Math.min(16, stream.length));
                const bytes = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                const content = stream.length > 268435456 ? '(' + bytes + ') [' + stream.length.toString() + ']': '(' + bytes + ')';
                throw new view.Error("Unsupported file content " + content + " for extension '." + extension + "'.");
            }
            throw new view.Error("Unsupported file directory.");
        };
        json();
        pbtxt();
        pb();
        flatbuffers();
        xml();
        unknown();
    }

    _openContext(context) {
        const modules = this._filter(context).filter((module) => module && module.length > 0);
        const errors = [];
        let success = false;
        const nextModule = () => {
            if (modules.length > 0) {
                const id = modules.shift();
                return this._host.require(id).then((module) => {
                    if (!module.ModelFactory) {
                        throw new view.Error("Failed to load module '" + id + "'.");
                    }
                    const modelFactory = new module.ModelFactory();
                    let match = undefined;
                    try {
                        match = modelFactory.match(context);
                        if (!match) {
                            return nextModule();
                        }
                    }
                    catch (error) {
                        return Promise.reject(error);
                    }
                    success = true;
                    try {
                        return modelFactory.open(context, match).then((model) => {
                            if (!model.identifier) {
                                model.identifier = context.identifier;
                            }
                            return model;
                        }).catch((error) => {
                            if (context.stream && context.stream.position !== 0) {
                                context.stream.seek(0);
                            }
                            errors.push(error);
                            return nextModule();
                        });
                    }
                    catch (error) {
                        if (context.stream && context.stream.position !== 0) {
                            context.stream.seek(0);
                        }
                        errors.push(error);
                        return nextModule();
                    }
                });
            }
            if (success) {
                if (errors.length === 1) {
                    const error = errors[0];
                    return Promise.reject(error);
                }
                return Promise.reject(new view.Error(errors.map((err) => err.message).join('\n')));
            }
            return Promise.resolve(null);
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
            const filter = (queue) => {
                let matches = [];
                const nextEntry = () => {
                    if (queue.length > 0) {
                        const entry = queue.shift();
                        const context = new view.ModelContext(new view.EntryContext(this._host, entries, folder, entry.name, entry.stream));
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
                                        matches.push(context);
                                        modules = [];
                                    }
                                    return nextModule();
                                });
                            }
                            return nextEntry();
                        };
                        return nextModule();
                    }
                    if (matches.length === 0) {
                        return Promise.resolve(null);
                    }
                    // MXNet
                    if (matches.length === 2 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.params')) &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('-symbol.json'))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.params'));
                    }
                    // TensorFlow.js
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.bin')) &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.json'))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.json'));
                    }
                    // ncnn
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.bin')) &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.param'))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.param'));
                    }
                    // ncnn
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.bin')) &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.param.bin'))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.param.bin'));
                    }
                    // NNEF
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.nnef')) &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.dat'))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.nnef'));
                    }
                    // Paddle
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.pdmodel')) &&
                        (matches.some((context) => context.identifier.toLowerCase().endsWith('.pdparams')) ||
                            matches.some((context) => context.identifier.toLowerCase().endsWith('.pdopt')) ||
                            matches.some((context) => context.identifier.toLowerCase().endsWith('.pdiparams')))) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().endsWith('.pdmodel'));
                    }
                    // Paddle Lite
                    if (matches.length > 0 &&
                        matches.some((context) => context.identifier.toLowerCase().split('/').pop() === '__model__.nb') &&
                        matches.some((context) => context.identifier.toLowerCase().split('/').pop() === 'param.nb')) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().split('/').pop() == '__model__.nb');
                    }
                    // TensorFlow Bundle
                    if (matches.length > 1 &&
                        matches.some((context) => context.identifier.toLowerCase().endsWith('.data-00000-of-00001'))) {
                        matches = matches.filter((context) => !context.identifier.toLowerCase().endsWith('.data-00000-of-00001'));
                    }
                    // TensorFlow SavedModel
                    if (matches.length === 2 &&
                        matches.some((context) => context.identifier.toLowerCase().split('/').pop() === 'keras_metadata.pb')) {
                        matches = matches.filter((context) => context.identifier.toLowerCase().split('/').pop() !== 'keras_metadata.pb');
                    }
                    if (matches.length > 1) {
                        return Promise.reject(new view.ArchiveError('Archive contains multiple model files.'));
                    }
                    const match = matches.shift();
                    return Promise.resolve(match);
                };
                return nextEntry();
            };
            const list = Array.from(entries).map((entry) => {
                return { name: entry[0], stream: entry[1] };
            });
            const files = list.filter((entry) => {
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
            const queue = files.slice(0).filter((entry) => entry.name.substring(folder.length).indexOf('/') < 0);
            return filter(queue).then((context) => {
                if (context) {
                    return Promise.resolve(context);
                }
                const queue = files.slice(0).filter((entry) => entry.name.substring(folder.length).indexOf('/') >= 0);
                return filter(queue);
            });
        }
        catch (error) {
            return Promise.reject(new view.ArchiveError(error.message));
        }
    }

    accept(identifier, size) {
        const extension = identifier.indexOf('.') === -1 ? '' : identifier.split('.').pop().toLowerCase();
        identifier = identifier.toLowerCase().split('/').pop();
        let accept = false;
        for (const extension of this._extensions) {
            if ((typeof extension === 'string' && identifier.endsWith(extension)) || (extension instanceof RegExp && extension.exec(identifier))) {
                accept = true;
                break;
            }
        }
        this._host.event('model_file', {
            file_extension: extension,
            file_size: size || 0,
            file_accept: accept ? 1 : 0
        });
        return accept;
    }

    _filter(context) {
        const identifier = context.identifier.toLowerCase().split('/').pop();
        const list = this._factories.filter((entry) =>
            (typeof entry.extension === 'string' && identifier.endsWith(entry.extension)) ||
            (entry.extension instanceof RegExp && entry.extension.exec(identifier)));
        return Array.from(new Set(list.map((entry) => entry.id)));
    }

    _openSignature(context) {
        const stream = context.stream;
        if (stream) {
            let empty = true;
            let position = 0;
            while (position < stream.length) {
                const buffer = stream.read(Math.min(4096, stream.length - position));
                position += buffer.length;
                if (!buffer.every((value) => value === 0x00)) {
                    empty = false;
                    break;
                }
            }
            stream.seek(0);
            if (empty) {
                return Promise.reject(new view.Error('File has no content.'));
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
                { name: 'Python source code', value: /^\s*from[ ]+(keras)[ ]+import[ ]+/ },
                { name: 'Bash script', value: /^#!\/usr\/bin\/env\s/ },
                { name: 'Bash script', value: /^#!\/bin\/bash\s/ },
                { name: 'TSD header', value: /^%TSD-Header-###%/ },
                { name: 'AppleDouble data', value: /^\x00\x05\x16\x07/ },
                { name: 'TensorFlow Hub module', value: /^\x08\x03$/, identifier: 'tfhub_module.pb' },
                { name: 'ViSQOL model', value: /^svm_type\s/ },
                { name: 'SenseTime model', value: /^STEF/ }
            ];
            /* eslint-enable no-control-regex */
            const buffer = stream.peek(Math.min(4096, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            for (const entry of entries) {
                if (content.match(entry.value) && (!entry.identifier || entry.identifier === context.identifier)) {
                    return Promise.reject(new view.Error('Invalid file content. File contains ' + entry.name + '.'));
                }
            }
        }
        return Promise.resolve(context);
    }
};

view.Metadata = class {

    static open(context, name) {
        view.Metadata._metadata = view.Metadata._metadata || new Map();
        if (view.Metadata._metadata.has(name)) {
            return Promise.resolve(view.Metadata._metadata.get(name));
        }
        return context.request(name, 'utf-8', null).then((data) => {
            const library = new view.Metadata(data);
            view.Metadata._metadata.set(name, library);
            return library;
        }).catch(() => {
            const library = new view.Metadata(null);
            view.Metadata._metadata.set(name, library);
            return library;
        });
    }

    constructor(data) {
        this._types = new Map();
        this._attributes = new Map();
        this._inputs = new Map();
        if (data) {
            const metadata = JSON.parse(data);
            for (const entry of metadata) {
                this._types.set(entry.name, entry);
                if (entry.identifier !== undefined) {
                    this._types.set(entry.identifier, entry);
                }
            }
        }
    }

    type(name) {
        if (!this._types.has(name)) {
            this._types.set(name, { name: name.toString() });
        }
        return this._types.get(name);
    }

    attribute(type, name) {
        const key = type + ':' + name;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.attributes)) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(type + ':' + attribute.name, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }

    input(type, name) {
        const key = type + ':' + name;
        if (!this._inputs.has(key)) {
            this._inputs.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.inputs)) {
                for (const input of metadata.inputs) {
                    this._inputs.set(type + ':' + input.name, input);
                }
            }
        }
        return this._inputs.get(key);
    }
};

view.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.View = view.View;
    module.exports.ModelFactoryService = view.ModelFactoryService;
    module.exports.Documentation = view.Documentation;
    module.exports.Formatter = view.Formatter;
    module.exports.Tensor = view.Tensor;
}
