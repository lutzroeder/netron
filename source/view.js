
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
var dialog = require('./dialog');
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
        this._host.initialize(this).then(() => {
            this._model = null;
            this._graphs = [];
            this._selection = [];
            this._sidebar = new dialog.Sidebar(this._host, id);
            this._searchText = '';
            this._modelFactoryService = new view.ModelFactoryService(this._host);
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
                this._preventDefault(e);
            }, { passive: true });
            this._host.document.addEventListener('keydown', () => {
                this.clearSelection();
            });
            this._host.start();
            const container = this._getElementById('graph');
            container.addEventListener('scroll', (e) => this._scrollHandler(e));
            container.addEventListener('wheel', (e) => this._wheelHandler(e), { passive: false });
            container.addEventListener('mousedown', (e) => this._mouseDownHandler(e));
            switch (this._host.agent) {
                case 'safari':
                    container.addEventListener('gesturestart', (e) => this._gestureStartHandler(e), false);
                    break;
                default:
                    container.addEventListener('touchstart', (e) => this._touchStartHandler(e), { passive: true });
                    break;
            }
        }).catch((err) => {
            this.error(err, null, null);
        });
    }

    show(page) {
        if (!page) {
            page = (!this._model && !this.activeGraph) ? 'welcome' : 'default';
        }
        this._host.screen(page);
        if (this._sidebar) {
            this._sidebar.close();
        }
        this._host.document.body.setAttribute('class', page);
        if (page === 'default') {
            const container = this._getElementById('graph');
            if (container) {
                container.focus();
            }
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
            const view = new dialog.FindSidebar(this._host, graphElement, this._graph);
            view.on('search-text-changed', (sender, text) => {
                this._searchText = text;
            });
            view.on('select', (sender, selection) => {
                this.select(selection);
            });
            this._sidebar.open(view.content, 'Find');
            view.focus(this._searchText);
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

    _preventDefault(e) {
        if (e.shiftKey || e.ctrlKey) {
            e.preventDefault();
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
            document.style.cursor = 'grabbing';
            const container = this._getElementById('graph');
            this._mousePosition = {
                left: container.scrollLeft,
                top: container.scrollTop,
                x: e.clientX,
                y: e.clientY
            };
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
                document.style.cursor = null;
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
        const message = err.message + (known ? '\n\n' + known.url : '');
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
        this._host.event('Model', 'Open', 'Size', context.stream ? context.stream.length : 0);
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
                    this._host.event('Model', 'Format', format.join(' '));
                }
                return this._timeout(20).then(() => {
                    const graphs = Array.isArray(model.graphs) && model.graphs.length > 0 ? [ model.graphs[0] ] : [];
                    return this._updateGraph(model, graphs);
                });
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
                        this._host.event('Graph', 'Render', 'Skip', nodes.length);
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
            this._host.event('Graph', 'Render', 'Size', nodes.length);

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
                const modelSidebar = new dialog.ModelSidebar(this._host, this._model, this.activeGraph);
                modelSidebar.on('update-active-graph', (sender, graph) => {
                    this._updateActiveGraph(graph);
                });
                const content = modelSidebar.render();
                this._sidebar.open(content, 'Model Properties');
            }
            catch (error) {
                const content = " in '" + this._model.identifier + "'.";
                if (error && !error.message.endsWith(content) && (error.context === undefined || error.context === true)) {
                    error.message = error.message.replace(/\.$/, '') + content;
                }
                this.error(error, 'Error showing model properties.', null);
            }
        }
    }

    showNodeProperties(node, input) {
        if (node) {
            try {
                const nodeSidebar = new dialog.NodeSidebar(this._host, node);
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
                        error.message = error.message.replace(/\.$/, '') + " in '" + this._model.identifier + "'.";
                    }
                    this.error(error, null, null);
                });
                if (input) {
                    nodeSidebar.toggleInput(input.name);
                }
                this._sidebar.open(nodeSidebar.render(), 'Node Properties');
            }
            catch (error) {
                const content = " in '" + this._model.identifier + "'.";
                if (error && !error.message.endsWith(content) && (error.context === undefined || error.context === true)) {
                    error.message = error.message.replace(/\.$/, '') + content;
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
            const documentationSidebar = new dialog.DocumentationSidebar(this._host, type);
            documentationSidebar.on('navigate', (sender, e) => {
                this._host.openURL(e.link);
            });
            const title = type.type === 'function' ? 'Function' : 'Documentation';
            this._sidebar.push(documentationSidebar.render(), title);
        }
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
                        throw new view.Error("Invalid null argument in '" + this.model.identifier + "'.");
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
                    this.setNode({ name: name, rx: 5, ry: 5});
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
            const identifier = this.context.model && this.context.model.identifier ? this.context.model.identifier : '?';
            throw new view.Error("Unsupported node type '" + JSON.stringify(type.name) + "' in '" + identifier + "'.");
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
                            const tensor = new dialog.Tensor(initializer);
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
                            const identifier = this.context.view.model && this.context.view.model.identifier ? this.context.view.model.identifier : '?';
                            throw new view.Error("Failed to render tensor of type '" + type + "' in '" + identifier + "' (" + err.message + ").");
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
                    let value = new dialog.Formatter(attribute.value, attribute.type).toString();
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
            const identifier = context.identifier;
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
                    const message = error && error.message ? error.message : error.toString();
                    throw new view.ArchiveError(message.replace(/\.$/, '') + " in '" + identifier + "'.");
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
                    const message = error && error.message ? error.message : error.toString();
                    throw new view.ArchiveError(message.replace(/\.$/, '') + " in '" + identifier + "'.");
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
                                if (stream.length > 2) {
                                    const archive = zip.Archive.open(stream, 'zlib');
                                    const data = archive ? archive.entries.get('') : stream;
                                    unpickler = python.Unpickler.open(data, () => {
                                        return new python.Execution(null, (error, fatal) => {
                                            const message = error && error.message ? error.message : error.toString();
                                            this.exception(new view.Error(message.replace(/\.$/, '') + " in '" + this.identifier + "'."), fatal);
                                        });
                                    });
                                }
                            }
                            catch (err) {
                                // continue regardless of error
                            }
                            if (unpickler) {
                                unpickler.persistent_load = (saved_id) => {
                                    return saved_id;
                                };
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
        this.register('./pytorch', [ '.pt', '.pth', '.ptl', '.pt1', '.pyt', '.pyth', '.pkl', '.pickle', '.h5', '.t7', '.model', '.dms', '.tar', '.ckpt', '.chkpt', '.tckpt', '.bin', '.pb', '.zip', '.nn', '.torchmodel', '.torchscript', '.pytorch', '.ot', '.params', '.trt' ], [ '.model' ]);
        this.register('./onnx', [ '.onnx', '.onn', '.pb', '.onnxtxt', '.pbtxt', '.prototxt', '.txt', '.model', '.pt', '.pth', '.pkl', '.ort', '.ort.onnx', 'onnxmodel' ]);
        this.register('./mxnet', [ '.json', '.params' ], [ '.mar'] );
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
        this.register('./numpy', [ '.npz', '.npy', '.pkl', '.pickle' ]);
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
                throw new view.Error("Archive contains no model files in '" + identifier + "'.", true);
            }
        }
        const skip = () => {
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
            return knownUnsupportedIdentifiers.has(context.identifier);
        };
        const json = () => {
            const obj = context.open('json');
            if (obj) {
                const formats = [
                    { name: 'Netron metadata', tags: [ '[].name', '[].schema' ] },
                    { name: 'Netron metadata', tags: [ '[].name', '[].attributes' ] },
                    { name: 'Netron metadata', tags: [ '[].name', '[].category' ] },
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
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
                const content = JSON.stringify(obj).substring(0, 100).replace(/\s/, '').substr(0, 48) + '...';
                throw new view.Error("Unsupported JSON content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "' in '" + identifier + "'.", !skip());
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
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
                const entries = [];
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') === -1));
                entries.push(...Array.from(tags).filter((pair) => pair[0].toString().indexOf('.') !== -1));
                const content = entries.map((pair) => pair[1] === true ? pair[0] : pair[0] + ':' + JSON.stringify(pair[1])).join(',');
                throw new view.Error("Unsupported Protocol Buffers text content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "' in '" + identifier + "'.", !skip());
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
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
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
                throw new view.Error("Unsupported Protocol Buffers content '" + (content.length > 64 ? content.substring(0, 100) + '...' : content) + "' for extension '." + extension + "' in '" + identifier + "'.", !skip());
            }
        };
        const flatbuffers = () => {
            const tags = context.tags('flatbuffers');
            if (tags.has('file_identifier')) {
                const file_identifier = tags.get('file_identifier');
                const formats = [
                    { name: 'onnxruntime.experimental.fbs.InferenceSession data', identifier: 'ORTM' },
                    { name: 'tflite.Model data', identifier: 'TFL3' },
                    { name: 'torch.jit.mobile.serialization.Module data', identifier: 'PTMF' }, // https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/serialization/mobile_bytecode.fbs
                    { name: 'FlatBuffers ENNC data', identifier: 'ENNC' },
                ];
                for (const format of formats) {
                    if (file_identifier === format.identifier) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
            }
        };
        const xml = () => {
            const tags = context.tags('xml');
            if (tags.size > 0) {
                const formats = [
                    { name: 'OpenCV storage data', tags: [ 'opencv_storage' ] },
                    { name: 'XHTML markup', tags: [ 'http://www.w3.org/1999/xhtml:html' ]}
                ];
                for (const format of formats) {
                    if (format.tags.some((tag) => tags.has(tag))) {
                        throw new view.Error('Invalid file content. File contains ' + format.name + '.', true);
                    }
                }
                throw new view.Error("Unsupported XML content '" + tags.keys().next().value + "' in '" + identifier + "'.", !skip());
            }
        };
        const unknown = () => {
            if (stream) {
                stream.seek(0);
                const buffer = stream.peek(Math.min(16, stream.length));
                const bytes = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                const content = stream.length > 268435456 ? '(' + bytes + ') [' + stream.length.toString() + ']': '(' + bytes + ')';
                throw new view.Error("Unsupported file content " + content + " for extension '." + extension + "' in '" + identifier + "'.", !skip());
            }
            throw new view.Error("Unsupported file directory in '" + identifier + "'.", !skip());
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
                    const updateErrorContext = (error, context) => {
                        const content = " in '" + context.identifier + "'.";
                        if (error && typeof error.message === 'string' && !error.message.endsWith(content) && (error.context === undefined || error.context === true)) {
                            error.message = error.message.replace(/\.$/, '') + content;
                        }
                    };
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
                        updateErrorContext(error, context);
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
                            updateErrorContext(error, context);
                            errors.push(error);
                            return nextModule();
                        });
                    }
                    catch (error) {
                        if (context.stream && context.stream.position !== 0) {
                            context.stream.seek(0);
                        }
                        updateErrorContext(error, context);
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

    accept(identifier) {
        const extension = identifier.indexOf('.') === -1 ? '' : identifier.split('.').pop().toLowerCase();
        identifier = identifier.toLowerCase().split('/').pop();
        for (const extension of this._extensions) {
            if ((typeof extension === 'string' && identifier.endsWith(extension)) || (extension instanceof RegExp && extension.exec(identifier))) {
                this._host.event('File', 'Accept', extension, 1);
                return true;
            }
        }
        this._host.event('File', 'Reject', extension, 1);
        return false;
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
                    return Promise.reject(new view.Error('Invalid file content. File contains ' + entry.name + '.', true));
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
