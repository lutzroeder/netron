
import * as base from './base.js';
import * as flatbuffers from './flatbuffers.js';
import * as grapher from './grapher.js';
import * as hdf5 from './hdf5.js';
import * as json from './json.js';
import * as protobuf from './protobuf.js';
import * as python from './python.js';
import * as tar from './tar.js';
import * as text from './text.js';
import * as xml from './xml.js';
import * as zip from './zip.js';

const view =  {};
const markdown = {};
const metrics = {};

view.View = class {

    constructor(host) {
        this._host = host;
        this._defaultOptions = {
            weights: true,
            attributes: false,
            names: false,
            direction: 'vertical',
            mousewheel: 'scroll'
        };
        this._options = { ...this._defaultOptions };
        this._model = null;
        this._stack = [];
        this._selection = [];
        this._sidebar = new view.Sidebar(this._host);
        this._find = null;
        this._modelFactoryService = new view.ModelFactoryService(this._host);
        this._modelFactoryService.import();
        this._worker = this._host.environment('measure') ? null : new view.Worker(this._host);
    }

    async start() {
        try {
            await zip.Archive.import();
            await this._host.view(this);
            const options = this._host.get('options') || {};
            for (const [name, value] of Object.entries(options)) {
                this._options[name] = value;
            }
            this._element('sidebar-button').addEventListener('click', () => {
                this.showModelProperties();
            });
            this._element('zoom-in-button').addEventListener('click', () => {
                this.zoomIn();
            });
            this._element('zoom-out-button').addEventListener('click', () => {
                this.zoomOut();
            });
            this._element('toolbar-path-back-button').addEventListener('click', async () => {
                await this.popGraph();
            });
            this._element('sidebar').addEventListener('mousewheel', (e) => {
                if (e.shiftKey || e.ctrlKey) {
                    e.preventDefault();
                }
            }, { passive: true });
            this._host.document.addEventListener('keydown', () => {
                if (this._graph) {
                    this._graph.select(null);
                }
            });
            if (this._host.type === 'Electron') {
                this._host.update({ 'can-copy': false });
                this._host.document.addEventListener('selectionchange', () => {
                    const selection = this._host.document.getSelection();
                    const selected = selection.rangeCount === 0 || selection.toString().trim() !== '';
                    this._host.update({ 'can-copy': selected });
                });
            }
            const platform = this._host.environment('platform');
            this._menu = new view.Menu(this._host);
            this._menu.add({
                accelerator: platform === 'darwin' ? 'Ctrl+Cmd+F' : 'F11',
                execute: async () => await this._host.execute('fullscreen')
            });
            this._menu.add({
                accelerator: 'Backspace',
                execute: async () => await this.popGraph()
            });
            if (this._host.environment('menu')) {
                this._menu.attach(this._element('menu'), this._element('menu-button'));
                const file = this._menu.group('&File');
                file.add({
                    label: '&Open...',
                    accelerator: 'CmdOrCtrl+O',
                    execute: async () => await this._host.execute('open')
                });
                if (this._host.type === 'Electron') {
                    this._recents = file.group('Open &Recent');
                    file.add({
                        label: '&Export...',
                        accelerator: 'CmdOrCtrl+Shift+E',
                        execute: async () => await this._host.execute('export'),
                        enabled: () => this.activeGraph
                    });
                    file.add({
                        label: platform === 'darwin' ? '&Close Window' : '&Close',
                        accelerator: 'CmdOrCtrl+W',
                        execute: async () => await this._host.execute('close'),
                    });
                    file.add({
                        label: platform === 'win32' ? 'E&xit' : '&Quit',
                        accelerator: platform === 'win32' ? '' : 'CmdOrCtrl+Q',
                        execute: async () => await this._host.execute('quit'),
                    });
                } else {
                    file.add({
                        label: 'Export as &PNG',
                        accelerator: 'CmdOrCtrl+Shift+E',
                        execute: async () => await this.export(`${this._host.document.title}.png`),
                        enabled: () => this.activeGraph
                    });
                    file.add({
                        label: 'Export as &SVG',
                        accelerator: 'CmdOrCtrl+Alt+E',
                        execute: async () => await this.export(`${this._host.document.title}.svg`),
                        enabled: () => this.activeGraph
                    });
                }
                const edit = this._menu.group('&Edit');
                edit.add({
                    label: '&Find...',
                    accelerator: 'CmdOrCtrl+F',
                    execute: () => this.find(),
                    enabled: () => this.activeGraph
                });
                const view = this._menu.group('&View');
                view.add({
                    label: () => this.options.attributes ? 'Hide &Attributes' : 'Show &Attributes',
                    accelerator: 'CmdOrCtrl+D',
                    execute: () => this.toggle('attributes'),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: () => this.options.weights ? 'Hide &Weights' : 'Show &Weights',
                    accelerator: 'CmdOrCtrl+I',
                    execute: () => this.toggle('weights'),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: () => this.options.names ? 'Hide &Names' : 'Show &Names',
                    accelerator: 'CmdOrCtrl+U',
                    execute: () => this.toggle('names'),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: () => this.options.direction === 'vertical' ? 'Show &Horizontal' : 'Show &Vertical',
                    accelerator: 'CmdOrCtrl+K',
                    execute: () => this.toggle('direction'),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: () => this.options.mousewheel === 'scroll' ? '&Mouse Wheel: Zoom' : '&Mouse Wheel: Scroll',
                    accelerator: 'CmdOrCtrl+M',
                    execute: () => this.toggle('mousewheel'),
                    enabled: () => this.activeGraph
                });
                view.add({});
                if (this._host.type === 'Electron') {
                    view.add({
                        label: '&Reload',
                        accelerator: platform === 'darwin' ? 'CmdOrCtrl+R' : 'F5',
                        execute: async () => await this._host.execute('reload'),
                        enabled: () => this.activeGraph
                    });
                    view.add({});
                }
                view.add({
                    label: 'Zoom &In',
                    accelerator: 'Shift+Up',
                    execute: () => this.zoomIn(),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: 'Zoom &Out',
                    accelerator: 'Shift+Down',
                    execute: () => this.zoomOut(),
                    enabled: () => this.activeGraph
                });
                view.add({
                    label: 'Actual &Size',
                    accelerator: 'Shift+Backspace',
                    execute: () => this.resetZoom(),
                    enabled: () => this.activeGraph
                });
                view.add({});
                view.add({
                    label: '&Properties...',
                    accelerator: 'CmdOrCtrl+Enter',
                    execute: () => this.showModelProperties(),
                    enabled: () => this.activeGraph
                });
                if (this._host.type === 'Electron' && !this._host.environment('packaged')) {
                    view.add({});
                    view.add({
                        label: '&Developer Tools...',
                        accelerator: 'CmdOrCtrl+Alt+I',
                        execute: async () => await this._host.execute('toggle-developer-tools')
                    });
                }
                const help = this._menu.group('&Help');
                help.add({
                    label: 'Report &Issue',
                    execute: async () => await this._host.execute('report-issue')
                });
                help.add({
                    label: `&About ${this._host.environment('name')}`,
                    execute: async () => await this._host.execute('about')
                });
            }
            await this._host.start();
        } catch (error) {
            this.error(error, null, null);
        }
    }

    get host() {
        return this._host;
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
        if (this._menu) {
            this._menu.close();
        }
        this._host.document.body.classList.remove(...Array.from(this._host.document.body.classList).filter((_) => _ !== 'active'));
        this._host.document.body.classList.add(...page.split(' '));
        if (page === 'default') {
            this._activate();
        } else {
            this._deactivate();
        }
        if (page === 'welcome') {
            const element = this._element('open-file-button');
            if (element) {
                element.focus();
            }
        }
        this._page = page;
    }

    progress(percent) {
        const bar = this._element('progress-bar');
        if (bar) {
            bar.style.width = `${percent}%`;
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
            this._graph.select(null);
            const sidebar = new view.FindSidebar(this, this.activeGraph, this.activeSignature);
            sidebar.on('state-changed', (sender, state) => {
                this._find = state;
            });
            sidebar.on('select', (sender, value) => {
                this.scrollTo(this._graph.select([value]));
            });
            sidebar.on('focus', (sender, value) => {
                this._graph.focus([value]);
            });
            sidebar.on('blur', (sender, value) => {
                this._graph.blur([value]);
            });
            sidebar.on('activate', (sender, value) => {
                this.scrollTo(this._graph.activate(value));
            });
            this._sidebar.open(sidebar, 'Find');
            sidebar.focus(this._find);
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
            case 'weights':
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
                throw new view.Error(`Unsupported toggle '${name}'.`);
        }
        const options = {};
        for (const [name, value] of Object.entries(this._options)) {
            if (this._defaultOptions[name] !== value) {
                options[name] = value;
            }
        }
        if (Object.entries(options).length === 0) {
            this._host.delete('options');
        } else {
            this._host.set('options', options);
        }
    }

    recents(recents) {
        if (this._recents) {
            this._recents.clear();
            for (let i = 0; i < recents.length; i++) {
                const recent = recents[i];
                this._recents.add({
                    label: recent.label,
                    accelerator: `CmdOrCtrl+${(i + 1)}`,
                    execute: () => this._host.execute('open', recent.path)
                });
            }
        }
    }

    _reload() {
        this.show('welcome spinner');
        if (this._model && this._stack.length > 0) {
            this._updateGraph(this._model, this._stack).catch((error) => {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            });
        }
    }

    _timeout(delay) {
        return new Promise((resolve) => {
            setTimeout(resolve, delay);
        });
    }

    _element(id) {
        return this._host.document.getElementById(id);
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
            this._events.gesturestart = (e) => this._gestureStartHandler(e);
            this._events.pointerdown = (e) => this._pointerDownHandler(e);
            this._events.touchstart = (e) => this._touchStartHandler(e);
        }
        const graph = this._element('graph');
        graph.focus();
        graph.addEventListener('scroll', this._events.scroll);
        graph.addEventListener('wheel', this._events.wheel, { passive: false });
        graph.addEventListener('pointerdown', this._events.pointerdown);
        if (this._host.environment('agent') === 'safari') {
            graph.addEventListener('gesturestart', this._events.gesturestart, false);
        } else {
            graph.addEventListener('touchstart', this._events.touchstart, { passive: true });
        }
    }

    _deactivate() {
        if (this._events) {
            const graph = this._element('graph');
            graph.removeEventListener('scroll', this._events.scroll);
            graph.removeEventListener('wheel', this._events.wheel);
            graph.removeEventListener('pointerdown', this._events.pointerdown);
            graph.removeEventListener('gesturestart', this._events.gesturestart);
            graph.removeEventListener('touchstart', this._events.touchstart);
        }
    }

    _updateZoom(zoom, e) {
        const container = this._element('graph');
        const canvas = this._element('canvas');
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
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        this._scrollLeft = Math.max(0, ((x * zoom) / this._zoom) - (x - scrollLeft));
        this._scrollTop = Math.max(0, ((y * zoom) / this._zoom) - (y - scrollTop));
        container.scrollLeft = this._scrollLeft;
        container.scrollTop = this._scrollTop;
        this._zoom = zoom;
    }

    _pointerDownHandler(e) {
        if (e.pointerType === 'touch' || e.buttons !== 1) {
            return;
        }
        // Workaround for Firefox emitting 'pointerdown' event when scrollbar is pressed
        if (e.originalTarget) {
            try {
                /* eslint-disable no-unused-expressions */
                e.originalTarget.id;
                /* eslint-enable no-unused-expressions */
            } catch {
                return;
            }
        }
        const container = this._element('graph');
        e.target.setPointerCapture(e.pointerId);
        this._mousePosition = {
            left: container.scrollLeft,
            top: container.scrollTop,
            x: e.clientX,
            y: e.clientY
        };
        container.style.cursor = 'grabbing';
        e.preventDefault();
        e.stopImmediatePropagation();
        const pointerMoveHandler = (e) => {
            e.preventDefault();
            e.stopImmediatePropagation();
            if (this._mousePosition) {
                const dx = e.clientX - this._mousePosition.x;
                const dy = e.clientY - this._mousePosition.y;
                this._mousePosition.moved = dx * dx + dy * dy > 0;
                if (this._mousePosition.moved) {
                    const container = this._element('graph');
                    container.scrollTop = this._mousePosition.top - dy;
                    container.scrollLeft = this._mousePosition.left - dx;
                }
            }
        };
        const clickHandler = (e) => {
            e.stopPropagation();
            document.removeEventListener('click', clickHandler, true);
        };
        const pointerUpHandler = (e) => {
            e.target.releasePointerCapture(e.pointerId);
            container.style.removeProperty('cursor');
            container.removeEventListener('pointerup', pointerUpHandler);
            container.removeEventListener('pointermove', pointerMoveHandler);
            if (this._mousePosition && this._mousePosition.moved) {
                e.preventDefault();
                e.stopImmediatePropagation();
                delete this._mousePosition;
                document.addEventListener('click', clickHandler, true);
            }
        };
        container.addEventListener('pointermove', pointerMoveHandler);
        container.addEventListener('pointerup', pointerUpHandler);
    }

    _touchStartHandler(e) {
        if (e.touches.length === 2) {
            this._touchPoints = Array.from(e.touches);
            this._touchZoom = this._zoom;
        }
        const touchMoveHandler = (e) => {
            if (Array.isArray(this._touchPoints) && this._touchPoints.length === 2 && e.touches.length === 2) {
                const distance = (points) => {
                    const dx = (points[1].clientX - points[0].clientX);
                    const dy = (points[1].clientY - points[0].clientY);
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
        const container = this._element('graph');
        const touchEndHandler = () => {
            container.removeEventListener('touchmove', touchMoveHandler, { passive: true });
            container.removeEventListener('touchcancel', touchEndHandler, { passive: true });
            container.removeEventListener('touchend', touchEndHandler, { passive: true });
            delete this._touchPoints;
            delete this._touchZoom;
        };
        container.addEventListener('touchmove', touchMoveHandler, { passive: true });
        container.addEventListener('touchcancel', touchEndHandler, { passive: true });
        container.addEventListener('touchend', touchEndHandler, { passive: true });
    }

    _gestureStartHandler(e) {
        e.preventDefault();
        this._gestureZoom = this._zoom;
        const container = this._element('graph');
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
            let factor = 1;
            if (e.deltaMode === 1) {
                factor = 0.05;
            } else if (e.deltaMode) {
                factor = 1;
            } else {
                factor = 0.002;
            }
            const delta = -e.deltaY * factor * (e.ctrlKey ? 10 : 1);
            this._updateZoom(this._zoom * Math.pow(2, delta), e);
            e.preventDefault();
        }
    }

    scrollTo(selection, behavior) {
        if (selection && selection.length > 0) {
            const container = this._element('graph');
            let x = 0;
            let y = 0;
            for (const element of selection) {
                const rect = element.getBoundingClientRect();
                x += rect.left + (rect.width / 2);
                y += rect.top + (rect.height / 2);
            }
            x /= selection.length;
            y /= selection.length;
            const rect = container.getBoundingClientRect();
            const left = (container.scrollLeft + x - rect.left) - (rect.width / 2);
            const top = (container.scrollTop + y - rect.top) - (rect.height / 2);
            behavior = behavior || 'smooth';
            container.scrollTo({ left, top, behavior });
        }
    }

    async error(error, name, screen) {
        if (this._sidebar) {
            this._sidebar.close();
        }
        this.exception(error, false);
        const knowns = [
            { name: '', message: /^Invalid value identifier/, url: 'https://github.com/lutzroeder/netron/issues/540' },
            { name: '', message: /^Cannot read property/, url: 'https://github.com/lutzroeder/netron/issues/647' },
            { name: 'Error', message: /^EPERM: operation not permitted/, url: 'https://github.com/lutzroeder/netron/issues/551' },
            { name: 'Error', message: /^EACCES: permission denied/, url: 'https://github.com/lutzroeder/netron/issues/504' },
            { name: 'RangeError', message: /^Offset is outside the bounds of the DataView/, url: 'https://github.com/lutzroeder/netron/issues/563' },
            { name: 'RangeError', message: /^Invalid string length/, url: 'https://github.com/lutzroeder/netron/issues/648' },
            { name: 'Python Error', message: /^Unknown function/, url: 'https://github.com/lutzroeder/netron/issues/546' },
            { name: 'Error loading model.', message: /^Unsupported file content/, url: 'https://github.com/lutzroeder/netron/issues/550' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers content/, url: 'https://github.com/lutzroeder/netron/issues/593' },
            { name: 'Error loading model.', message: /^Unsupported Protocol Buffers text content/, url: 'https://github.com/lutzroeder/netron/issues/594' },
            { name: 'Error loading model.', message: /^Unsupported JSON content/, url: 'https://github.com/lutzroeder/netron/issues/595' },
            { name: 'Error loading PyTorch model.', message: /^File does not contain root module or state dictionary/, url: 'https://github.com/lutzroeder/netron/issues/543' },
            { name: 'Error loading PyTorch model.', message: /^Module does not contain modules/, url: 'https://github.com/lutzroeder/netron/issues/544' },
            { name: 'Error loading PyTorch model.', message: /^Unknown type name/, url: 'https://github.com/lutzroeder/netron/issues/969' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto \(Unexpected end of file\)\./, url: 'https://github.com/lutzroeder/netron/issues/1155' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto \(Cannot read properties of undefined \(reading 'ModelProto'\)\)\./, url: 'https://github.com/lutzroeder/netron/issues/1156' },
            { name: 'Error loading ONNX model.', message: /^File format is not onnx\.ModelProto/, url: 'https://github.com/lutzroeder/netron/issues/549' }
        ];
        const known = knowns.find((known) => (known.name.length === 0 || known.name === error.name) && error.message.match(known.message));
        const url = known && known.url ? known.url : null;
        const message = error.message;
        name = name || error.name;
        const report = url ? true : false;
        await this._host.message(message, true, report ? 'Report' : 'OK');
        if (report) {
            this._host.openURL(url || `${this._host.environment('repository')}/issues`);
        }
        this.show(screen);
    }

    accept(file, size) {
        return this._modelFactoryService.accept(file, size);
    }

    async open(context) {
        this._sidebar.close();
        await this._timeout(2);
        try {
            const model = await this._modelFactoryService.open(context);
            const format = [];
            if (model.format) {
                format.push(model.format);
            }
            if (model.producer) {
                format.push(`(${model.producer})`);
            }
            if (format.length > 0) {
                this._host.event('model_open', {
                    model_format: model.format || '',
                    model_producer: model.producer || ''
                });
            }
            await this._timeout(20);
            const stack = [];
            if (Array.isArray(model.graphs) && model.graphs.length > 0) {
                const [graph] = model.graphs;
                const entry = {
                    graph,
                    signature: Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures[0] : null
                };
                stack.push(entry);
            }
            return await this._updateGraph(model, stack);
        } catch (error) {
            error.context = !error.context && context && context.identifier ? context.identifier : error.context || '';
            throw error;
        }
    }

    async _updateActive(stack) {
        this._sidebar.close();
        if (this._model) {
            this.show('welcome spinner');
            await this._timeout(200);
            try {
                await this._updateGraph(this._model, stack);
            } catch (error) {
                if (error) {
                    this.error(error, 'Graph update failed.', 'welcome');
                }
            }
        }
    }

    get activeGraph() {
        if (Array.isArray(this._stack) && this._stack.length > 0) {
            return this._stack[0].graph;
        }
        return null;
    }

    get activeSignature() {
        if (Array.isArray(this._stack) && this._stack.length > 0) {
            return this._stack[0].signature;
        }
        return null;
    }

    async _updateGraph(model, stack) {
        const update = async (model, stack) => {
            this._model = model;
            this._stack = stack;
            const status = await this.renderGraph(this._model, this.activeGraph, this.activeSignature, this._options);
            if (status !== '') {
                this._model = null;
                this._stack = [];
                this._activeGraph = null;
            }
            this.show(null);
            const path = this._element('toolbar-path');
            const back = this._element('toolbar-path-back-button');
            while (path.children.length > 1) {
                path.removeChild(path.lastElementChild);
            }
            if (status === '') {
                if (this._stack.length <= 1) {
                    back.style.opacity = 0;
                } else {
                    back.style.opacity = 1;
                    const last = this._stack.length - 2;
                    const count = Math.min(2, last);
                    if (count < last) {
                        const element = this._host.document.createElement('button');
                        element.setAttribute('class', 'toolbar-path-name-button');
                        element.innerHTML = '&hellip;';
                        path.appendChild(element);
                    }
                    for (let i = count; i >= 0; i--) {
                        const graph = this._stack[i].graph;
                        const element = this._host.document.createElement('button');
                        element.setAttribute('class', 'toolbar-path-name-button');
                        element.addEventListener('click', async () => {
                            if (i > 0) {
                                this._stack = this._stack.slice(i);
                                await this._updateGraph(this._model, this._stack);
                            } else {
                                await this.showDefinition(graph);
                            }
                        });
                        let name = '';
                        if (graph && graph.identifier) {
                            name = graph.identifier;
                        } else if (graph && graph.name) {
                            name = graph.name;
                        }
                        if (name.length > 24) {
                            element.setAttribute('title', name);
                            element.innerHTML = `&hellip;${name.substring(name.length - 24, name.length)}`;
                        } else {
                            element.removeAttribute('title');
                            element.innerHTML = name;
                        }
                        path.appendChild(element);
                    }
                }
            }
        };
        const lastModel = this._model;
        const lastStack = this._stack;
        try {
            await update(model, stack);
            return this._model;
        } catch (error) {
            await update(lastModel, lastStack);
            throw error;
        }
    }

    async pushGraph(graph, context) {
        if (graph && graph !== this.activeGraph && Array.isArray(graph.nodes)) {
            this._sidebar.close();
            if (context) {
                this._stack[0].context = context;
                this._stack[0].zoom = this._zoom;
            }
            const signature = Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures[0] : null;
            const entry = { graph, signature };
            const stack = [entry].concat(this._stack);
            await this._updateGraph(this._model, stack);
        }
    }

    async popGraph() {
        if (this._stack.length > 1) {
            this._sidebar.close();
            return await this._updateGraph(this._model, this._stack.slice(1));
        }
        return null;
    }

    async renderGraph(model, graph, signature, options) {
        this._graph = null;
        const canvas = this._element('canvas');
        while (canvas.lastChild) {
            canvas.removeChild(canvas.lastChild);
        }
        if (!graph) {
            return '';
        }
        this._zoom = 1;
        const groups = graph.groups || false;
        const nodes = graph.nodes;
        this._host.event('graph_view', {
            graph_node_count: nodes.length,
            graph_skip: 0
        });
        const viewGraph = new view.Graph(this, this._host, model, options, groups);
        viewGraph.add(graph, signature);
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
        await this._timeout(20);
        viewGraph.measure();
        const status = await viewGraph.layout(this._worker);
        if (status === '') {
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
            origin.setAttribute('transform', `translate(${margin}, ${margin}) scale(1)`);
            background.setAttribute('width', width);
            background.setAttribute('height', height);
            this._width = width;
            this._height = height;
            delete this._scrollLeft;
            delete this._scrollRight;
            canvas.setAttribute('viewBox', `0 0 ${width} ${height}`);
            canvas.setAttribute('width', width);
            canvas.setAttribute('height', height);
            this._zoom = this._stack && this._stack.length > 0 && this._stack[0].zoom ? this._stack[0].zoom : 1;
            this._updateZoom(this._zoom);
            const container = this._element('graph');
            const context = this._stack && this._stack.length > 0 && this._stack[0].context ? viewGraph.select([this._stack[0].context]) : [];
            if (context.length > 0) {
                this.scrollTo(context, 'instant');
            } else if (elements && elements.length > 0) {
                // Center view based on input elements
                const xs = [];
                const ys = [];
                for (let i = 0; i < elements.length; i++) {
                    const element = elements[i];
                    const rect = element.getBoundingClientRect();
                    xs.push(rect.left + (rect.width / 2));
                    ys.push(rect.top + (rect.height / 2));
                }
                let [x] = xs;
                const [y] = ys;
                if (ys.every((y) => y === ys[0])) {
                    x = xs.reduce((a, b) => a + b, 0) / xs.length;
                }
                const graphRect = container.getBoundingClientRect();
                const left = (container.scrollLeft + x - graphRect.left) - (graphRect.width / 2);
                const top = (container.scrollTop + y - graphRect.top) - (graphRect.height / 2);
                container.scrollTo({ left, top, behavior: 'auto' });
            } else {
                const canvasRect = canvas.getBoundingClientRect();
                const graphRect = container.getBoundingClientRect();
                const left = (container.scrollLeft + (canvasRect.width / 2) - graphRect.left) - (graphRect.width / 2);
                const top = (container.scrollTop + (canvasRect.height / 2) - graphRect.top) - (graphRect.height / 2);
                container.scrollTo({ left, top, behavior: 'auto' });
            }
            this._graph = viewGraph;
        }
        return status;
    }

    applyStyleSheet(element, name) {
        let rules = [];
        for (const styleSheet of this._host.document.styleSheets) {
            if (styleSheet && styleSheet.href && styleSheet.href.endsWith(`/${name}`)) {
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

    async export(file) {
        const lastIndex = file.lastIndexOf('.');
        const extension = lastIndex === -1 ? 'png' : file.substring(lastIndex + 1).toLowerCase();
        if (this.activeGraph && (extension === 'png' || extension === 'svg')) {
            const canvas = this._element('canvas');
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
            origin.setAttribute('transform', `translate(${(delta - size.x)}, ${(delta - size.y)}) scale(1)`);
            clone.setAttribute('width', width);
            clone.setAttribute('height', height);
            background.setAttribute('width', width);
            background.setAttribute('height', height);
            background.setAttribute('fill', '#fff');
            const data = new XMLSerializer().serializeToString(clone);
            if (extension === 'svg') {
                const blob = new Blob([data], { type: 'image/svg' });
                await this._host.export(file, blob);
            }
            if (extension === 'png') {
                try {
                    const blob = await new Promise((resolve, reject) => {
                        const image = new Image();
                        image.onload = async () => {
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
                                    resolve(blob);
                                } else {
                                    const error = new Error('Image may be too large to render as PNG.');
                                    error.name = 'Error exporting image.';
                                    reject(error);
                                }
                            }, 'image/png');
                        };
                        image.onerror = (error) => {
                            reject(error);
                        };
                        image.src = `data:image/svg+xml;base64,${this._host.window.btoa(unescape(encodeURIComponent(data)))}`;
                    });
                    await this._host.export(file, blob);
                } catch (error) {
                    await this.error(error);
                }
            }
        }
    }

    showModelProperties() {
        if (this._model) {
            try {
                const sidebar = new view.ModelSidebar(this, this.model, this.activeGraph, this.activeSignature);
                sidebar.on('update-active-graph', (sender, graph) => {
                    const entry = {
                        graph,
                        signature: Array.isArray(graph.signatures) && graph.signatures.length > 0 ? graph.signatures[0] : null
                    };
                    this._updateActive([entry]);
                });
                sidebar.on('update-active-graph-signature', (sender, signature) => {
                    const stack = this._stack.map((entry) => {
                        return { graph: entry.graph, signature: entry.signature };
                    });
                    stack[0].signature = signature;
                    this._updateActive(stack);
                });
                sidebar.on('focus', (sender, value) => {
                    this._graph.focus([value]);
                });
                sidebar.on('blur', (sender, value) => {
                    this._graph.blur([value]);
                });
                sidebar.on('select', (sender, value) => {
                    this.scrollTo(this._graph.activate(value));
                });
                sidebar.on('activate', (sender, value) => {
                    this.scrollTo(this._graph.select([value]));
                });
                sidebar.on('deactivate', () => {
                    this._graph.select(null);
                });
                this._sidebar.open(sidebar, 'Model Properties');
            } catch (error) {
                this.error(error, 'Error showing model properties.', null);
            }
        }
    }

    showNodeProperties(node) {
        if (node) {
            try {
                if (this._menu) {
                    this._menu.close();
                }
                const sidebar = new view.NodeSidebar(this, node);
                sidebar.on('show-definition', async (/* sender, e */) => {
                    await this.showDefinition(node.type);
                });
                sidebar.on('focus', (sender, value) => {
                    this._graph.focus([value]);
                });
                sidebar.on('blur', (sender, value) => {
                    this._graph.blur([value]);
                });
                sidebar.on('select', (sender, value) => {
                    this.scrollTo(this._graph.select([value]));
                });
                sidebar.on('activate', (sender, value) => {
                    this.scrollTo(this._graph.activate(value));
                });
                this._sidebar.open(sidebar, 'Node Properties');
            } catch (error) {
                this.error(error, 'Error showing node properties.', null);
            }
        }
    }

    showConnectionProperties(value, from, to) {
        try {
            if (this._menu) {
                this._menu.close();
            }
            const sidebar = new view.ConnectionSidebar(this, value, from, to);
            sidebar.on('focus', (sender, value) => {
                this._graph.focus([value]);
            });
            sidebar.on('blur', (sender, value) => {
                this._graph.blur([value]);
            });
            sidebar.on('select', (sender, value) => {
                this.scrollTo(this._graph.select([value]));
            });
            sidebar.on('activate', (sender, value) => {
                this.scrollTo(this._graph.activate(value));
            });
            this._sidebar.push(sidebar, 'Connection Properties');
        } catch (error) {
            this.error(error, 'Error showing connection properties.', null);
        }
    }

    showTensorProperties(value) {
        try {
            if (this._menu) {
                this._menu.close();
            }
            const sidebar = new view.TensorSidebar(this, value);
            sidebar.on('focus', (sender, value) => {
                this._graph.focus([value]);
            });
            sidebar.on('blur', () => {
                this._graph.blur(null);
            });
            sidebar.on('select', (sender, value) => {
                this.scrollTo(this._graph.select([value]));
            });
            sidebar.on('activate', (sender, value) => {
                this.scrollTo(this._graph.activate(value));
            });
            this._sidebar.push(sidebar, 'Tensor Properties');
        } catch (error) {
            this.error(error, 'Error showing tensor properties.', null);
        }
    }

    exception(error, fatal) {
        if (error && !error.context && this._model && this._model.identifier) {
            error.context = this._model.identifier;
        }
        this._host.exception(error, fatal);
    }

    async showDefinition(type) {
        if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
            if (type.nodes && type.nodes.length > 0) {
                await this.pushGraph(type);
            }
            if (type.type !== 'weights') {
                const sidebar = new view.DocumentationSidebar(this, type);
                sidebar.on('navigate', (sender, e) => {
                    this._host.openURL(e.link);
                });
                const title = type.type === 'function' ? 'Function' : 'Documentation';
                this._sidebar.push(sidebar, title);
            }
        }
    }

    about() {
        this._host.document.getElementById('version').innerText = this._host.version;
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

    constructor(host) {
        this.items = [];
        this._darwin = host.environment('platform') === 'darwin';
        this._document = host.document;
        this._stack = [];
        this._root = [];
        this._buttons = [];
        this._accelerators = new Map();
        this._keyCodes = new Map([
            ['Backspace', 0x08], ['Enter', 0x0D], ['Escape', 0x1B],
            ['Left', 0x25], ['Up', 0x26], ['Right', 0x27], ['Down', 0x28],
            ['F5', 0x74], ['F11', 0x7a]
        ]);
        this._symbols = new Map([
            ['Backspace', '&#x232B;'], ['Enter', '&#x23ce;'],
            ['Up', '&#x2191;'], ['Down', '&#x2193;'],
        ]);
        this._keydown = (e) => {
            this._alt = false;
            const code = e.keyCode | (e.altKey ? 0x0200 : 0) | (e.shiftKey ? 0x0100 : 0);
            const modifier = (e.ctrlKey ? 0x0400 : 0) | (e.metaKey ? 0x0800 : 0);
            if ((code | modifier) === 0x0212) { // Alt
                this._alt = true;
            } else {
                const action =
                    this._accelerators.get(code | modifier) ||
                    this._accelerators.get(code | ((e.ctrlKey && !this._darwin) || (e.metaKey && this._darwin) ? 0x1000 : 0));
                if (action && this._execute(action)) {
                    e.preventDefault();
                } else {
                    const item = this._mnemonic(code | modifier);
                    if (item && this._activate(item)) {
                        e.preventDefault();
                    }
                }
            }
        };
        this._keyup = (e) => {
            const code = e.keyCode;
            if (code === 0x0012 && this._alt) { // Alt
                switch (this._stack.length) {
                    case 0: {
                        if (this.open()) {
                            e.preventDefault();
                        }
                        break;
                    }
                    case 1: {
                        if (this.close()) {
                            e.preventDefault();
                        }
                        break;
                    }
                    default: {
                        this._stack = [this];
                        if (this._root.length > 1) {
                            this._root =  [this];
                            this._rebuild();
                        }
                        this._update();
                        e.preventDefault();
                        break;
                    }
                }
            }
            this._alt = false;
        };
        this._next = () => {
            const button = this._element.ownerDocument.activeElement;
            const index = this._buttons.indexOf(button);
            if (index !== -1 && index < this._buttons.length - 1) {
                const next = this._buttons[index + 1];
                next.focus();
            }
        };
        this._previous = () => {
            const button = this._element.ownerDocument.activeElement;
            const index = this._buttons.indexOf(button);
            if (index > 0) {
                const next = this._buttons[index - 1];
                next.focus();
            }
        };
        this._push = () => {
            const button = this._element.ownerDocument.activeElement;
            if (button && button.getAttribute('data-type') === 'group') {
                button.click();
            }
        };
        this._pop = () => {
            if (this._stack.length > 1) {
                this._deactivate();
            }
        };
        this._exit = () => {
            this._deactivate();
            if (this._stack.length === 0) {
                this.close();
            }
        };
        host.window.addEventListener('keydown', this._keydown);
        host.window.addEventListener('keyup', this._keyup);
    }

    attach(element, button) {
        this._element = element;
        button.addEventListener('click', (e) => {
            this.toggle();
            e.preventDefault();
        });
    }

    add(value) {
        const item = new view.Menu.Command(value);
        this.register(item, item.accelerator);
    }

    group(label) {
        const item = new view.Menu.Group(this, label);
        item.identifier = `menu-item-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.register(item.accelerator);
        return item;
    }

    toggle() {
        if (this._element.style.opacity >= 1) {
            this.close();
        } else {
            this._root = [this];
            this._stack = [this];
            this.open();
        }
    }

    open() {
        if (this._element) {
            if (this._stack.length === 0) {
                this.toggle();
                this._stack = [this];
            }
            this._rebuild();
            this._update();
            this.register(this._exit, 'Escape');
            this.register(this._previous, 'Up');
            this.register(this._next, 'Down');
            this.register(this._pop, 'Left');
            this.register(this._push, 'Right');
        }
    }

    close() {
        if (this._element) {
            this.unregister(this._exit);
            this.unregister(this._previous);
            this.unregister(this._next);
            this.unregister(this._pop);
            this.unregister(this._push);
            this._element.style.opacity = 0;
            this._element.style.left = '-17em';
            const button = this._element.ownerDocument.activeElement;
            if (this._buttons.indexOf(button) > 0) {
                button.blur();
            }
            while (this._root.length > 1) {
                this._deactivate();
            }
            this._stack = [];
        }
    }

    register(action, accelerator) {
        let shortcut = '';
        if (accelerator) {
            let shift = false;
            let alt = false;
            let ctrl = false;
            let cmd = false;
            let cmdOrCtrl = false;
            let key = '';
            for (const part of accelerator.split('+')) {
                switch (part) {
                    case 'CmdOrCtrl': cmdOrCtrl = true; break;
                    case 'Cmd': cmd = true; break;
                    case 'Ctrl': ctrl = true; break;
                    case 'Alt': alt = true; break;
                    case 'Shift': shift = true; break;
                    default: key = part; break;
                }
            }
            if (key !== '') {
                if (this._darwin) {
                    shortcut += ctrl ? '&#x2303' : '';
                    shortcut += alt ? '&#x2325;' : '';
                    shortcut += shift ? '&#x21e7;' : '';
                    shortcut += cmdOrCtrl || cmd ? '&#x2318;' : '';
                    shortcut += this._symbols.has(key) ? this._symbols.get(key) : key;
                } else {
                    shortcut += cmdOrCtrl || ctrl ? 'Ctrl+' : '';
                    shortcut += alt ? 'Alt+' : '';
                    shortcut += shift ? 'Shift+' : '';
                    shortcut += key;
                }
                let code = (cmdOrCtrl ? 0x1000 : 0) | (cmd ? 0x0800 : 0) | (ctrl ? 0x0400 : 0) | (alt ? 0x0200 : 0) | (shift ? 0x0100 : 0);
                code |= this._keyCodes.has(key) ? this._keyCodes.get(key) : key.charCodeAt(0);
                this._accelerators.set(code, action);
            }
        }
        return shortcut;
    }

    unregister(action) {
        this._accelerators = new Map(Array.from(this._accelerators.entries()).filter(([, value]) => value !== action));
    }

    _execute(action) {
        if (typeof action === 'function') {
            action();
            return true;
        }
        switch (action ? action.type : null) {
            case 'group': {
                while (this._stack.length > this._root.length) {
                    this._stack.pop();
                }
                this._root.push({ items: [action] });
                this._stack.push(action);
                this._rebuild();
                this._update();
                return true;
            }
            case 'command': {
                this.close();
                setTimeout(() => action.execute(), 10);
                return true;
            }
            default: {
                return false;
            }
        }
    }

    _mnemonic(code) {
        const key = /[a-zA-Z0-9]/.test(String.fromCharCode(code & 0x00FF));
        const modifier = (code & 0xFF00) !== 0;
        const alt = (code & 0xFF00) === 0x0200;
        if (alt && key) {
            this.open();
        }
        if (this._stack.length > 0 && key && (alt || !modifier)) {
            const key = String.fromCharCode(code & 0x00FF);
            const group = this._stack.length > 0 ? this._stack[this._stack.length - 1] : this;
            const item = group.items.find((item) => key === item.mnemonic && (item.type === 'group' || item.type === 'command') && item.enabled);
            if (item) {
                return item;
            }
        }
        return null;
    }

    _activate(item) {
        switch (item ? item.type : null) {
            case 'group': {
                this._stack.push(item);
                this._rebuild();
                this._update();
                return true;
            }
            case 'command': {
                return this._execute(item);
            }
            default: {
                return false;
            }
        }
    }

    _deactivate() {
        if (this._root.length > 1) {
            this._root.pop();
            const group = this._stack.pop();
            this._rebuild();
            this._update();
            if (group) {
                const button = this._buttons.find((button) => button.getAttribute('id') === group.identifier);
                if (button) {
                    button.focus();
                }
            }
        } else if (this._stack.length > 0) {
            this._stack.pop();
            this._update();
        }
    }

    _label(item, mnemonic) {
        delete item.mnemonic;
        const value = item.label;
        if (value) {
            const index = value.indexOf('&');
            if (index !== -1) {
                if (mnemonic) {
                    item.mnemonic = value[index + 1].toUpperCase();
                    return `${value.substring(0, index)}<u>${value[index + 1]}</u>${value.substring(index + 2)}`;
                }
                return value.substring(0, index) + value.substring(index + 1);
            }
        }
        return value || '';
    }

    _rebuild() {
        this._element.innerHTML = '';
        const root = this._root[this._root.length - 1];
        for (const group of root.items) {
            const container = this._document.createElement('div');
            container.setAttribute('id', group.identifier);
            container.setAttribute('class', 'menu-group');
            container.innerHTML = "<div class='menu-group-header'></div>";
            for (const item of group.items) {
                switch (item.type) {
                    case 'group':
                    case 'command': {
                        const button = this._document.createElement('button');
                        button.setAttribute('class', 'menu-command');
                        button.setAttribute('id', item.identifier);
                        button.setAttribute('data-type', item.type);
                        button.addEventListener('mouseenter', () => button.focus());
                        button.addEventListener('click', () => this._execute(item));
                        const accelerator = this._document.createElement('span');
                        accelerator.setAttribute('class', 'menu-shortcut');
                        if (item.type === 'group') {
                            accelerator.innerHTML = '&#10095;';
                        } else if (item.shortcut) {
                            accelerator.innerHTML = item.shortcut;
                        }
                        button.appendChild(accelerator);
                        const content = this._document.createElement('span');
                        content.setAttribute('class', 'menu-label');
                        button.appendChild(content);
                        container.appendChild(button);
                        break;
                    }
                    case 'separator': {
                        const element = this._document.createElement('div');
                        element.setAttribute('class', 'menu-separator');
                        element.setAttribute('id', item.identifier);
                        container.appendChild(element);
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            this._element.appendChild(container);
        }
        this._element.style.opacity = 1.0;
        this._element.style.left = '0px';
        if (this._root.length > 1) {
            this._element.style.width = 'auto';
            this._element.style.maxWidth = '60%';
        } else {
            this._element.style.removeProperty('width');
            this._element.style.maxWidth = 'auto';
        }
    }

    _update() {
        this._buttons = [];
        const selected = this._stack.length > 0 ? this._stack[this._stack.length - 1] : null;
        const root = this._root[this._root.length - 1];
        for (const group of root.items) {
            let visible = false;
            let block = false;
            const active = this._stack.length <= 1 || this._stack[1] === group;
            const container = this._document.getElementById(group.identifier);
            container.childNodes[0].innerHTML = this._label(group, this === selected);
            for (const item of group.items) {
                switch (item.type) {
                    case 'group':
                    case 'command': {
                        const label = this._label(item, group === selected);
                        const button = this._document.getElementById(item.identifier);
                        button.childNodes[1].innerHTML = label;
                        if (item.enabled) {
                            button.removeAttribute('disabled');
                            button.style.display = 'block';
                            visible = true;
                            block = true;
                            if (active) {
                                this._buttons.push(button);
                            }
                        } else {
                            button.setAttribute('disabled', '');
                            button.style.display = 'none';
                        }
                        break;
                    }
                    case 'separator': {
                        const element = this._document.getElementById(item.identifier);
                        element.style.display = block ? 'block' : 'none';
                        block = false;
                        break;
                    }
                    default: {
                        break;
                    }
                }
            }
            for (let i = group.items.length - 1; i >= 0; i--) {
                const item = group.items[i];
                if ((item.type === 'group' || item.type === 'command') && item.enabled) {
                    break;
                } else if (item.type === 'separator') {
                    const element = this._document.getElementById(item.identifier);
                    element.style.display = 'none';
                }
            }
            if (!visible) {
                container.style.display = 'none';
            }
            container.style.opacity = active ? 1 : 0;
        }
        const button = this._element.ownerDocument.activeElement;
        const index = this._buttons.indexOf(button);
        if (index === -1 && this._buttons.length > 0) {
            this._buttons[0].focus();
        }
    }
};

view.Menu.Group = class {

    constructor(parent, label) {
        this.type = 'group';
        this.parent = parent;
        this.label = label;
        this.items = [];
    }

    get enabled() {
        return this.items.some((item) => item.enabled);
    }

    add(value) {
        const item = Object.keys(value).length > 0 ? new view.Menu.Command(value) : new view.Menu.Separator();
        item.identifier = `${this.identifier}-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.parent.register(item, item.accelerator);
    }

    group(label) {
        const item = new view.Menu.Group(this, label);
        item.identifier = `${this.identifier}-${this.items.length}`;
        this.items.push(item);
        item.shortcut = this.parent.register(item, item.accelerator);
        return item;
    }

    clear() {
        for (const item of this.items) {
            if (item.clear) {
                item.clear();
            }
            this.parent.unregister(item);
        }
        this.items = [];
    }

    register(item, accelerator) {
        return this.parent.register(item, accelerator);
    }

    unregister(item) {
        this.parent.unregister(item);
    }
};

view.Menu.Command = class {

    constructor(item) {
        this.type = 'command';
        this.accelerator = item.accelerator;
        this._label = item.label;
        this._enabled = item.enabled;
        this._execute = item.execute;
    }

    get label() {
        return typeof this._label === 'function' ? this._label() : this._label;
    }

    get enabled() {
        return this._enabled ? this._enabled() : true;
    }

    execute() {
        if (this._execute && this.enabled) {
            this._execute();
        }
    }
};

view.Menu.Separator = class {

    constructor() {
        this.type = 'separator';
        this.enabled = false;
    }
};

view.Worker = class {

    constructor(host) {
        this._host = host;
        const type = this._host.type;
        this._browser = type === 'Browser' || type === 'Python';
        if (this._browser) {
            this._create();
        }
    }

    async request(message, delay, notification) {
        this._timeout = -1;
        return new Promise((resolve, reject) => {
            this._resolve = resolve;
            this._reject = reject;
            if (!this._worker) {
                this._create();
            }
            this._worker.postMessage(message);
            this._timeout = setTimeout(async () => {
                await this._host.message(notification, null, 'Cancel');
                this._cancel(true);
                delete this._resolve;
                delete this._reject;
                resolve({ type: 'cancel' });
            }, delay);
        });
    }

    _create() {
        this._worker = this._host.worker('./worker');
        this._worker.addEventListener('message', (e) => {
            this._cancel(false);
            const message = e.data;
            if (this._reject && message.type === 'error') {
                this._reject(new Error(message.message));
            } else if (this._resolve) {
                this._resolve(message);
            }
            delete this._resolve;
            delete this._reject;
        });
        this._worker.addEventListener('error', (e) => {
            this._cancel(true);
            if (this._reject) {
                this._reject(new Error(`Unknown worker error type '${e.type}'.`));
                delete this._resolve;
                delete this._reject;
            }
        });
    }

    _cancel(terminate) {
        terminate = terminate || !this._browser;
        if (this._worker && terminate) {
            this._worker.terminate();
            this._worker = null;
        }
        if (this._timeout >= 0) {
            clearTimeout(this._timeout);
            this._timeout = -1;
            this._host.message();
        }
    }
};

view.Graph = class extends grapher.Graph {

    constructor(view, host, model, options, compound) {
        super(compound);
        this.view = view;
        this.host = host;
        this.model = model;
        this.options = options;
        this.counter = 0;
        this._nodeKey = 0;
        this._values = new Map();
        this._tensors = new Map();
        this._table = new Map();
        this._selection = new Set();
    }

    createNode(node, type) {
        if (type) {
            const obj = new view.Node(this, { type });
            obj.name = (this._nodeKey++).toString();
            this._table.set(type, obj);
            return obj;
        }
        const obj = new view.Node(this, node);
        obj.name = (this._nodeKey++).toString();
        this._table.set(node, obj);
        const inputs = node.inputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                if (!argument.type || argument.type.endsWith('*')) {
                    if (Array.isArray(argument.value) && argument.value.length === 1 && argument.value[0].initializer) {
                        this.createArgument(argument);
                    } else {
                        for (const value of argument.value) {
                            if (value.name !== '' && !value.initializer) {
                                this.createValue(value).to.push(obj);
                            } else if (value.initializer) {
                                this.createValue(value);
                            }
                        }
                    }
                }
            }
        }
        return obj;
    }

    createInput(input) {
        const obj = new view.Input(this, input);
        obj.name = (this._nodeKey++).toString();
        this._table.set(input, obj);
        return obj;
    }

    createOutput(output) {
        const obj = new view.Output(this, output);
        obj.name = (this._nodeKey++).toString();
        this._table.set(output, obj);
        return obj;
    }

    createValue(value) {
        const key = value && value.name && !value.initializer ? value.name : value;
        if (this._values.has(key)) {
            // duplicate argument name
            const obj = this._values.get(key);
            this._table.set(value, obj);
        } else {
            const obj = new view.Value(this, value);
            this._values.set(key, obj);
            this._table.set(value, obj);
        }
        return this._values.get(key);
    }

    createArgument(value) {
        if (Array.isArray(value.value) && value.value.length === 1 && value.value[0].initializer) {
            if (!this._tensors.has(value)) {
                const obj = new view.Argument(this, value);
                this._tensors.set(value, obj);
                this._table.set(value, obj);
            }
            return this._tensors.get(value);
        }
        return null;
    }

    add(graph, signature) {
        this.identifier = this.model.identifier;
        this.identifier += graph && graph.name ? `.${graph.name.replace(/\/|\\/, '.')}` : '';
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
        const inputs = signature ? signature.inputs : graph.inputs;
        const outputs = signature ? signature.outputs : graph.outputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                if (argument.visible !== false) {
                    const viewInput = this.createInput(argument);
                    this.setNode(viewInput);
                    for (const value of argument.value) {
                        this.createValue(value).from = viewInput;
                    }
                }
            }
        }
        for (const node of graph.nodes) {
            const viewNode = this.createNode(node);
            this.setNode(viewNode);
            let outputs = node.outputs;
            if (node.chain && node.chain.length > 0) {
                const chainOutputs = node.chain[node.chain.length - 1].outputs;
                if (chainOutputs.length > 0) {
                    outputs = chainOutputs;
                }
            }
            if (Array.isArray(outputs)) {
                for (const argument of outputs) {
                    for (const value of argument.value) {
                        if (!value) {
                            throw new view.Error('Invalid null argument.');
                        }
                        if (value.name !== '') {
                            this.createValue(value).from = viewNode;
                        }
                    }
                }
            }
            if (Array.isArray(node.controlDependencies) && node.controlDependencies.length > 0) {
                for (const value of node.controlDependencies) {
                    this.createValue(value).controlDependency(viewNode);
                }
            }
            const createCluster = (name) => {
                if (!clusters.has(name)) {
                    this.setNode({ name, rx: 5, ry: 5 });
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
                        if (lastIndex === -1) {
                            groupName = null;
                        } else {
                            groupName = groupName.substring(0, lastIndex);
                            if (!clusterParentMap.has(groupName)) {
                                groupName = null;
                            }
                        }
                    }
                    if (groupName) {
                        createCluster(`${groupName}\ngroup`);
                        this.setParent(viewNode.name, `${groupName}\ngroup`);
                    }
                }
            }
        }
        if (Array.isArray(outputs)) {
            for (const argument of outputs) {
                if (argument.visible !== false) {
                    const viewOutput = this.createOutput(argument);
                    this.setNode(viewOutput);
                    for (const value of argument.value) {
                        this.createValue(value).to.push(viewOutput);
                    }
                }
            }
        }
    }

    build(document, origin) {
        for (const value of this._values.values()) {
            value.build();
        }
        super.build(document, origin);
    }

    select(selection) {
        if (this._selection.size > 0) {
            for (const element of this._selection) {
                element.deselect();
            }
            this._selection.clear();
        }
        if (selection) {
            let array = [];
            for (const value of selection) {
                if (this._table.has(value)) {
                    const element = this._table.get(value);
                    array = array.concat(element.select());
                    this._selection.add(element);
                }
            }
            return array;
        }
        return null;
    }

    activate(value) {
        if (this._table.has(value)) {
            this.select(null);
            const element = this._table.get(value);
            element.activate();
            return this.select([value]);
        }
        return [];
    }

    focus(selection) {
        for (const value of selection) {
            const element = this._table.get(value);
            if (element && !this._selection.has(element)) {
                element.select();
            }
        }
    }

    blur(selection) {
        for (const value of selection) {
            const element = this._table.get(value);
            if (element && !this._selection.has(element)) {
                element.deselect();
            }
        }
    }
};

view.Node = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        this.id = `node-${value.name ? `name-${value.name}` : `id-${(context.counter++)}`}`;
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
        const options = this.context.options;
        const header =  this.header();
        const type = node.type;
        const category = type && type.category ? type.category : '';
        if (typeof type.name !== 'string' || !type.name.split) { // #416
            const error = new view.Error(`Unsupported node type '${JSON.stringify(type.name)}'.`);
            if (this.context.model && this.context.model.identifier) {
                error.context = this.context.model.identifier;
            }
            throw error;
        }
        let content = options.names && (node.name || node.identifier) ? (node.name || node.identifier) : type.name.split('.').pop();
        const tooltip = options.names && (node.name || node.identifier) ? type.name : (node.name || node.identifier);
        if (content.length > 18) {
            content = `${content.substring(0, 9)}\u2026${content.substring(content.length - 9, content.length)}`;
        }
        const styles = category ? ['node-item-type', `node-item-type-${category.toLowerCase()}`] : ['node-item-type'];
        const title = header.add(null, styles, content, tooltip);
        title.on('click', () => {
            this.context.activate(node);
        });
        if (Array.isArray(node.type.nodes) && node.type.nodes.length > 0) {
            let icon = '\u0192';
            let tooltip = 'Show Function Definition';
            if (node.type.type === 'weights') {
                icon = '\u25CF';
                tooltip = 'Show Weights';
            }
            const definition = header.add(null, styles, icon, tooltip);
            definition.on('click', async () => await this.context.view.pushGraph(node.type, this.value));
        }
        if (Array.isArray(node.nodes)) {
            // this._expand = header.add(null, styles, '+', null);
            // this._expand.on('click', () => this.toggle());
        }
        let current = null;
        const list = () => {
            if (!current) {
                current = this.list();
                current.on('click', () => this.context.activate(node));
            }
            return current;
        };
        let hiddenTensors = false;
        const objects = [];
        const attribute = (argument) => {
            let content = new view.Formatter(argument.value, argument.type).toString();
            if (content && content.length > 12) {
                content = `${content.substring(0, 12)}\u2026`;
            }
            const item = list().argument(argument.name, content);
            item.tooltip = argument.type;
            if (!content.startsWith('\u3008')) {
                item.separator = ' = ';
            }
            return item;
        };
        const isObject = (node) => {
            if (node.name || node.identifier || node.description ||
                (Array.isArray(node.inputs) && node.inputs.length > 0) ||
                (Array.isArray(node.outputs) && node.outputs.length > 0) ||
                (Array.isArray(node.attributes) && node.attributes.length > 0) ||
                (Array.isArray(node.chain) && node.chain.length > 0) ||
                (node.type && Array.isArray(node.type.nodes) && node.type.nodes.length > 0)) {
                return true;
            }

            return false;
        };
        const inputs = node.inputs;
        if (Array.isArray(inputs)) {
            for (const argument of inputs) {
                const type = argument.type;
                if (argument.visible !== false &&
                    ((type === 'graph') ||
                    (type === 'object' && isObject(argument.value)) ||
                    (type === 'object[]' || type === 'function' || type === 'function[]'))) {
                    objects.push(argument);
                } else if (options.weights && argument.visible !== false && argument.type !== 'attribute' && Array.isArray(argument.value) && argument.value.length === 1 && argument.value[0].initializer) {
                    const item = this.context.createArgument(argument);
                    list().add(item);
                } else if (options.weights && (argument.visible === false || Array.isArray(argument.value) && argument.value.length > 1) && (!argument.type || argument.type.endsWith('*')) && argument.value.some((value) => value.initializer)) {
                    hiddenTensors = true;
                } else if (options.attributes && argument.visible !== false && argument.type && !argument.type.endsWith('*')) {
                    const item = attribute(argument);
                    list().add(item);
                }
            }
        }
        if (Array.isArray(node.attributes)) {
            const attributes = node.attributes.slice();
            attributes.sort((a, b) => a.name.toUpperCase().localeCompare(b.name.toUpperCase()));
            for (const argument of attributes) {
                const type = argument.type;
                if (argument.visible !== false &&
                    ((type === 'graph') ||
                    (type === 'object') ||
                    type === 'object[]' || type === 'function' || type === 'function[]')) {
                    objects.push(argument);
                } else if (options.attributes && argument.visible !== false) {
                    const item = attribute(argument);
                    list().add(item);
                }
            }
        }
        if (hiddenTensors) {
            const item = list().argument('\u3008\u2026\u3009', '');
            list().add(item);
        }
        for (const argument of objects) {
            const type = argument.type;
            let content = null;
            if (type === 'graph') {
                content = this.context.createNode(null, argument.value);
            } else if (type === 'function' || argument.type === 'object') {
                content = this.context.createNode(argument.value);
            } else if (type === 'function[]' || argument.type === 'object[]') {
                content = argument.value.map((value) => this.context.createNode(value));
            }
            const item = list().argument(argument.name, content);
            list().add(item);
        }
        if (Array.isArray(node.nodes) && node.nodes.length > 0) {
            // this.canvas = this.canvas();
        }
        if (Array.isArray(node.chain) && node.chain.length > 0) {
            for (const innerNode of node.chain) {
                this.context.createNode(innerNode);
                this._add(innerNode);
            }
        }
        if (node.inner) {
            this.context.createNode(node.inner);
            this._add(node.inner);
        }
    }

    toggle() {
        this._expand.content = '-';
        this._graph = new view.Graph(this.context.view, this.context.view.host, this.context.model, this.context.options, false, {});
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

    activate() {
        this.context.view.showNodeProperties(this.value);
    }

    edge(to) {
        this._edges = this._edges || new Map();
        if (!this._edges.has(to)) {
            this._edges.set(to, new view.Edge(this, to));
        }
        return this._edges.get(to);
    }
};

view.Input = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        view.Input.counter = view.Input.counter || 0;
        const types = value.value.map((argument) => argument.type || '').join('\n');
        let name = value.name || '';
        if (name.length > 16) {
            name = name.split('/').pop();
        }
        const header = this.header();
        const title = header.add(null, ['graph-item-input'], name, types);
        title.on('click', () => this.context.view.showModelProperties());
        this.id = `input-${name ? `name-${name}` : `id-${(view.Input.counter++)}`}`;
    }

    get class() {
        return 'graph-input';
    }

    get inputs() {
        return [];
    }

    get outputs() {
        return [this.value];
    }

    activate() {
        this.context.view.showModelProperties();
    }

    edge(to) {
        this._edges = this._edges || new Map();
        if (!this._edges.has(to)) {
            this._edges.set(to, new view.Edge(this, to));
        }
        return this._edges.get(to);
    }
};

view.Output = class extends grapher.Node {

    constructor(context, value) {
        super();
        this.context = context;
        this.value = value;
        const types = value.value.map((argument) => argument.type || '').join('\n');
        let name = value.name || '';
        if (name.length > 16) {
            name = name.split('/').pop();
        }
        const header = this.header();
        const title = header.add(null, ['graph-item-output'], name, types);
        title.on('click', () => this.context.view.showModelProperties());
    }

    get inputs() {
        return [this.value];
    }

    get outputs() {
        return [];
    }

    activate() {
        this.context.view.showModelProperties();
    }
};

view.Value = class {

    constructor(context, value) {
        this.context = context;
        this.value = value;
        this.from = null;
        this.to = [];
    }

    controlDependency(node) {
        this._controlDependencies = this._controlDependencies || new Set();
        this._controlDependencies.add(this.to.length);
        this.to.push(node);
    }

    build() {
        this._edges = this._edges || [];
        if (this.from && Array.isArray(this.to)) {
            for (let i = 0; i < this.to.length; i++) {
                const to = this.to[i];
                let content = '';
                const type = this.value.type;
                if (type &&
                    type.shape &&
                    type.shape.dimensions &&
                    type.shape.dimensions.length > 0 &&
                    type.shape.dimensions.every((dim) => !dim || Number.isInteger(dim) || typeof dim === 'bigint' || (typeof dim === 'string'))) {
                    content = type.shape.dimensions.map((dim) => (dim !== null && dim !== undefined && dim !== -1) ? dim : '?').join('\u00D7');
                    content = content.length > 16 ? '' : content;
                }
                if (this.context.options.names) {
                    content = this.value.name.split('\n').shift(); // custom argument id
                }
                const edge = this.from.edge(to);
                if (!edge.value) {
                    edge.value = this;
                    if (content) {
                        edge.label = content;
                    }
                    edge.id = `edge-${this.value.name}`;
                    if (this._controlDependencies && this._controlDependencies.has(i)) {
                        edge.class = 'edge-path-control-dependency';
                    }
                }
                this.context.setEdge(edge);
                this._edges.push(edge);
            }
        }
    }

    select() {
        let array = [];
        if (Array.isArray(this._edges)) {
            for (const edge of this._edges) {
                array = array.concat(edge.select());
            }
        }
        return array;
    }

    deselect() {
        if (Array.isArray(this._edges)) {
            for (const edge of this._edges) {
                edge.deselect();
            }
        }
    }

    activate() {
        if (this.value && this.from && Array.isArray(this.to) && !this.value.initializer) {
            const from = this.from.value;
            const to = this.to.map((node) => node.value);
            this.context.view.showConnectionProperties(this.value, from, to);
        } else if (this.value && this.value.initializer) {
            this.context.view.showTensorProperties({ value: [this.value] });
        }
    }
};

view.Argument = class extends grapher.Argument {

    constructor(context, value) {
        const name = value.name;
        let content = '';
        let separator = '';
        let tooltip = '';
        if (Array.isArray(value.value) && value.value.length === 1 && value.value[0].initializer) {
            const tensor = value.value[0].initializer;
            const type = value.value[0].type;
            tooltip = type.toString();
            content = view.Formatter.tensor(tensor);
            if (!content.startsWith('\u3008')) {
                separator = ' = ';
            }
        }
        super(name, content);
        this.context = context;
        this.value = value;
        this.separator = separator;
        this.tooltip = tooltip;
    }

    focus() {
        this.context.focus([this.value]);
    }

    blur() {
        this.context.blur([this.value]);
    }

    activate() {
        this.context.view.showTensorProperties(this.value);
    }
};

view.Edge = class extends grapher.Edge {

    constructor(from, to) {
        super(from, to);
        this.v = from.name;
        this.w = to.name;
    }

    get minlen() {
        if (this.from.inputs.every((argument) => (!argument.type || argument.type.endsWith('*')) && argument.value.every((value) => value.initializer))) {
            return 2;
        }
        return 1;
    }

    focus() {
        this.value.context.focus([this.value.value]);
    }

    blur() {
        this.value.context.blur([this.value.value]);
    }

    activate() {
        this.value.context.activate(this.value.value);
    }
};

view.Sidebar = class {

    constructor(host) {
        this._host = host;
        this._stack = [];
        const pop = () => this._update(this._stack.slice(0, -1));
        this._closeSidebarHandler = () => pop();
        this._closeSidebarKeyDownHandler = (e) => {
            if (e.keyCode === 27) {
                e.stopPropagation();
                e.preventDefault();
                pop();
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
        return this._host.document.getElementById(id);
    }

    open(content, title) {
        const element = this._render(content);
        const entry = { title, element, content };
        this._update([entry]);
    }

    close() {
        this._update([]);
    }

    push(content, title) {
        const element = this._render(content);
        const entry = { title, content, element };
        this._update(this._stack.concat(entry));
    }

    _render(content) {
        try {
            content.render();
        } catch (error) {
            content.error(error, false);
        }
        const element = content.element;
        return Array.isArray(element) ? element : [element];
    }

    _update(stack) {
        const sidebar = this._element('sidebar');
        const content = this._element('sidebar-content');
        const container = this._element('graph');
        const closeButton = this._element('sidebar-closebutton');
        closeButton.removeEventListener('click', this._closeSidebarHandler);
        this._host.document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
        if (this._stack.length > 0) {
            const entry = this._stack.pop();
            if (entry.content && entry.content.deactivate) {
                entry.content.deactivate();
            }
        }
        if (stack) {
            this._stack = stack;
        }
        if (this._stack.length > 0) {
            const entry = this._stack[this._stack.length - 1];
            this._element('sidebar-title').innerHTML = entry.title || '';
            closeButton.addEventListener('click', this._closeSidebarHandler);
            if (typeof entry.content === 'string') {
                content.innerHTML = entry.element;
            } else if (entry.element instanceof Array) {
                content.innerHTML = '';
                for (const element of entry.element) {
                    content.appendChild(element);
                }
            } else {
                content.innerHTML = '';
                content.appendChild(entry.element);
            }
            sidebar.style.width = 'min(calc(100% * 0.6), 42em)';
            sidebar.style.right = 0;
            sidebar.style.opacity = 1;
            this._host.document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
            container.style.width = 'max(40vw, calc(100vw - 42em))';
            if (entry.content && entry.content.activate) {
                entry.content.activate();
            }
        } else {
            sidebar.style.right = 'calc(0px - min(calc(100% * 0.6), 42em))';
            sidebar.style.opacity = 0;
            const clone = content.cloneNode(true);
            content.parentNode.replaceChild(clone, content);
            container.style.width = '100%';
            container.focus();
        }
    }
};

view.Control = class {

    constructor(context) {
        this._view = context;
        this._host = context.host;
    }

    createElement(tagName, className) {
        const element = this._host.document.createElement(tagName);
        if (className) {
            element.setAttribute('class', className);
        }
        return element;
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

    error(error, fatal) {
        this._view.exception(error, fatal || false);
    }
};

view.Expander = class extends view.Control {

    constructor(context) {
        super(context);
        this.element = this.createElement('div', 'sidebar-item-value');
        this._count = -1;
    }

    render() {
        return [this.element];
    }

    enable() {
        this._expander = this.createElement('div', 'sidebar-item-value-expander');
        this._expander.innerText = '+';
        this._expander.addEventListener('click', () => this.toggle());
        this.add(this._expander);
    }

    add(element) {
        this.element.appendChild(element);
    }

    control(element) {
        this.add(element);
    }

    toggle() {
        this._count = this._count === -1 ? this.element.childElementCount : this._count;
        if (this._expander) {
            while (this.element.childElementCount > this._count) {
                this.element.removeChild(this.element.lastChild);
            }
            if (this._expander.innerText === '+') {
                this._expander.innerText = '-';
                this.expand();
            } else {
                this._expander.innerText = '+';
                this.collapse();
            }
        }
    }

    expand() {
    }

    collapse() {
    }
};

view.ObjectSidebar = class extends view.Control {

    constructor(context) {
        super(context);
        this.element = this.createElement('div', 'sidebar-object');
    }

    addEntry(name, item) {
        const entry = new view.NameValueView(this._view, name, item);
        const element = entry.render();
        this.element.appendChild(element);
    }

    addProperty(name, value, style) {
        const item = new view.TextView(this._view, value, style);
        this.addEntry(name, item);
        return item;
    }

    addHeader(title) {
        const element = this.createElement('div', 'sidebar-header');
        element.innerText = title;
        this.element.appendChild(element);
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('span');
        element.innerHTML = `<b>ERROR:</b> ${error.message}`;
        this.element.appendChild(element);
    }
};

view.NodeSidebar = class extends view.ObjectSidebar {

    constructor(context, node) {
        super(context);
        this._node = node;
    }

    render() {
        const node = this._node;
        if (node.type) {
            const type = node.type;
            const item = this.addProperty('type', node.type.identifier || node.type.name);
            if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
                let icon = '?';
                let tooltip = 'Show Definition';
                if (type.type === 'weights') {
                    icon = '\u25CF';
                    tooltip = 'Show Weights';
                } else if (Array.isArray(type.nodes)) {
                    icon = '\u0192';
                }
                item.action(icon, tooltip, () => {
                    this.emit('show-definition', null);
                });
            }
            const module = node.type.module;
            const version = node.type.version;
            const status = node.type.status;
            if (module || version || status) {
                const list = [module, version ? `v${version}` : '', status];
                const value = list.filter((value) => value).join(' ');
                this.addProperty('module', value, 'nowrap');
            }
        }
        if (node.name) {
            this.addProperty('name', node.name, 'nowrap');
        }
        if (node.identifier) {
            this.addProperty('identifier', node.identifier, 'nowrap');
        }
        if (node.description) {
            this.addProperty('description', node.description);
        }
        if (node.device) {
            this.addProperty('device', node.device);
        }
        const attributes = node.attributes;
        if (Array.isArray(attributes) && attributes.length > 0) {
            this.addHeader('Attributes');
            attributes.sort((a, b) => a.name.localeCompare(b.name, undefined, { sensitivity: 'base' }));
            for (const attribute of attributes) {
                this.addArgument(attribute.name, attribute, 'attribute');
            }
        }
        const inputs = node.inputs;
        if (Array.isArray(inputs) && inputs.length > 0) {
            this.addHeader('Inputs');
            for (const input of inputs) {
                const name = input.name;
                this.addArgument(name, input);
            }
        }
        const outputs = node.outputs;
        if (Array.isArray(outputs) && outputs.length > 0) {
            this.addHeader('Outputs');
            for (const output of outputs) {
                const name = output.name;
                this.addArgument(name, output);
            }
        }
        const metadata = node.metadata;
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addHeader('Metadata');
            for (const entry of metadata) {
                this.addArgument(entry.name, entry, 'attribute');
            }
        }
    }

    addArgument(name, argument, source) {
        const value = new view.ArgumentView(this._view, argument, source);
        value.on('focus', (sender, value) => {
            this.emit('focus', value);
            this._focused = this._focused || new Set();
            this._focused.add(value);
        });
        value.on('blur', (sender, value) => {
            this.emit('blur', value);
            this._focused = this._focused || new Set();
            this._focused.delete(value);
        });
        value.on('select', (sender, value) => this.emit('select', value));
        value.on('activate', (sender, value) => this.emit('activate', value));
        this.addEntry(name, value);
    }

    activate() {
        this.emit('select', this._node);
    }

    deactivate() {
        this.emit('select', null);
        if (this._focused) {
            for (const value of this._focused) {
                this.emit('blur', value);
            }
            this._focused.clear();
        }
    }
};

view.NameValueView = class extends view.Control {

    constructor(context, name, value) {
        super(context);
        this._name = name;
        this._value = value;
        const nameElement = this.createElement('div', 'sidebar-item-name');
        const input = this.createElement('input');
        input.setAttribute('type', 'text');
        input.setAttribute('value', name);
        input.setAttribute('title', name);
        input.setAttribute('readonly', 'true');
        nameElement.appendChild(input);
        const valueElement = this.createElement('div', 'sidebar-item-value-list');
        for (const element of value.render()) {
            valueElement.appendChild(element);
        }
        this.element = this.createElement('div', 'sidebar-item');
        this.element.appendChild(nameElement);
        this.element.appendChild(valueElement);
    }

    get name() {
        return this._name;
    }

    render() {
        return this.element;
    }

    toggle() {
        this._value.toggle();
    }
};

view.SelectView = class extends view.Control {

    constructor(context, entries, selected) {
        super(context);
        this._elements = [];
        this._entries = Array.from(entries);
        const selectElement = this.createElement('select', 'sidebar-item-selector');
        selectElement.addEventListener('change', (e) => {
            this.emit('change', this._entries[e.target.selectedIndex][1]);
        });
        this._elements.push(selectElement);
        for (const [name, value] of this._entries) {
            const element = this.createElement('option');
            element.innerText = name;
            if (value === selected) {
                element.setAttribute('selected', 'selected');
            }
            selectElement.appendChild(element);
        }
    }

    render() {
        return this._elements;
    }
};

view.TextView = class extends view.Control {

    constructor(context, value, style) {
        super(context);
        this.element = this.createElement('div', 'sidebar-item-value');
        let className = 'sidebar-item-value-line';
        if (value) {
            const list = Array.isArray(value) ? value : [value];
            for (const item of list) {
                const line = this.createElement('div', className);
                switch (style) {
                    case 'code':
                        line.innerHTML = `<code>${item}<code>`;
                        break;
                    case 'bold':
                        line.innerHTML = `<b>${item}<b>`;
                        break;
                    case 'nowrap':
                        line.innerText = item;
                        line.style.whiteSpace = style;
                        break;
                    default:
                        line.innerText = item;
                        break;
                }
                this.element.appendChild(line);
                className = 'sidebar-item-value-line-border';
            }
        } else {
            const line = this.createElement('div', className);
            line.classList.add('sidebar-item-disable-select');
            line.innerHTML = '&nbsp';
            this.element.appendChild(line);
        }
    }

    action(text, description, callback) {
        const action = this.createElement('div', 'sidebar-item-value-expander');
        action.setAttribute('title', description);
        action.addEventListener('click', () => callback());
        action.innerHTML = text;
        this.element.insertBefore(action, this.element.childNodes[0]);
    }

    render() {
        return [this.element];
    }

    toggle() {
    }
};

view.ArgumentView = class extends view.Control {

    constructor(context, argument, source) {
        super(context);
        this._argument = argument;
        this._source = source;
        this._elements = [];
        this._items = [];
        const type = argument.type === 'attribute' ? null : argument.type;
        let value = argument.value;
        if (argument.type === 'attribute') {
            this._source = 'attribute';
        }
        if (argument.type === 'tensor') {
            value = [{ type: value.type, initializer: value }];
        } else if (argument.type === 'tensor[]') {
            value = value.map((value) => value === null ? value : { type: value.type, initializer: value });
        }
        this._source = typeof type === 'string' && !type.endsWith('*') ? 'attribute' : this._source;
        if (this._source === 'attribute' && type !== 'tensor' && type !== 'tensor[]') {
            this._source = 'attribute';
            const item = new view.PrimitiveView(context, argument);
            this._items.push(item);
        } else if (Array.isArray(value) && value.length === 0) {
            const item = new view.TextView(this._view, null);
            this._items.push(item);
        } else {
            const values = value;
            for (const value of values) {
                const emit = values.length === 1 && value.initializer;
                const target = emit ? argument : value;
                if (value === null) {
                    const item = new view.TextView(this._view, null);
                    this._items.push(item);
                } else {
                    const item = new view.ValueView(context, value, this._source);
                    item.on('focus', () => this.emit('focus', target));
                    item.on('blur', () => this.emit('blur', target));
                    item.on('activate', () => this.emit('activate', target));
                    item.on('select', () => this.emit('select', target));
                    this._items.push(item);
                }
            }
        }
        for (const item of this._items) {
            this._elements.push(...item.render());
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
        if (this._source === 'attribute') {
            if (this._expander.innerText === '+') {
                this._expander.innerText = '-';
            } else {
                this._expander.innerText = '+';
                while (this._element.childElementCount > 2) {
                    this._element.removeChild(this._element.lastChild);
                }
            }
        } else {
            for (const item of this._items) {
                item.toggle();
            }
        }
    }
};

view.PrimitiveView = class extends view.Expander {

    constructor(context, argument) {
        super(context);
        try {
            this._argument = argument;
            const type = argument.type === 'attribute' ? null : argument.type;
            const value = argument.value;
            if (type) {
                this.enable();
            }
            switch (type) {
                case 'graph': {
                    const line = this.createElement('div', 'sidebar-item-value-line-link');
                    line.innerHTML = value.name || '&nbsp;';
                    line.addEventListener('click', () => this.emit('activate', value));
                    this.add(line);
                    break;
                }
                case 'function': {
                    const line = this.createElement('div', 'sidebar-item-value-line-link');
                    line.innerHTML = value.type.name;
                    line.addEventListener('click', () => this.emit('activate', value));
                    this.add(line);
                    break;
                }
                case 'object[]': {
                    for (const obj of argument.value) {
                        const line = this.createElement('div', 'sidebar-item-value-line');
                        line.innerHTML = obj.type.name;
                        this.add(line);
                    }
                    break;
                }
                default: {
                    let content = new view.Formatter(value, type).toString();
                    if (content && content.length > 1000) {
                        content = `${content.substring(0, 1000)}\u2026`;
                    }
                    if (content && typeof content === 'string') {
                        content = content.split('<').join('&lt;').split('>').join('&gt;');
                    }
                    const line = this.createElement('div', 'sidebar-item-value-line');
                    line.innerHTML = content ? content : '&nbsp;';
                    this.add(line);
                }
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    expand() {
        try {
            const type = this._argument.type;
            const value = this._argument.value;
            const content = type === 'tensor' && value && value.type ? value.type.toString() : this._argument.type;
            const line = this.createElement('div', 'sidebar-item-value-line-border');
            line.innerHTML = `type: <code><b>${content}</b></code>`;
            this.add(line);
            const description = this._argument.description;
            if (description) {
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.innerHTML = description;
                this.add(line);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    _info(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `<b>${name}:</b> ${value}`;
        this._add(line);
    }

    _add(child) {
        child.className = this._first === false ? 'sidebar-item-value-line-border' : 'sidebar-item-value-line';
        this.add(child);
        this._first = false;
    }
};

view.ValueView = class extends view.Expander {

    constructor(context, value, source) {
        super(context);
        this._value = value;
        try {
            const type = this._value.type;
            const initializer = this._value.initializer;
            const quantization = this._value.quantization;
            const location = this._value.location !== undefined;
            if (initializer) {
                this.element.classList.add('sidebar-item-value-content');
            }
            if (type || initializer || quantization || location || source === 'attribute') {
                this.enable();
            }
            if (initializer && source !== 'attribute') {
                const element = this.createElement('div', 'sidebar-item-value-button');
                element.classList.add('sidebar-item-value-button-tool');
                element.setAttribute('title', 'Show Tensor');
                element.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-weight"></use></svg>`;
                element.addEventListener('pointerenter', () => this.emit('focus', this._value));
                element.addEventListener('pointerleave', () => this.emit('blur', this._value));
                element.style.cursor = 'pointer';
                element.addEventListener('click', () => this.emit('activate', this._value));
                this.control(element);
            }
            const name = this._value.name ? this._value.name.split('\n').shift() : ''; // custom argument id
            this._hasId = name && source !== 'attribute' ? true : false;
            this._hasCategory = initializer && initializer.category && source !== 'attribute' ? true : false;
            if (this._hasId || (!this._hasCategory && !type && source !== 'attribute')) {
                this._hasId = true;
                const element = this.createElement('div', 'sidebar-item-value-line');
                if (typeof name !== 'string') {
                    throw new Error(`Invalid value identifier '${JSON.stringify(name)}'.`);
                }
                const text = this.createElement('b');
                text.innerText = name || ' ';
                const line = this.createElement('span', 'sidebar-item-value-line-content');
                line.innerText = 'name: ';
                line.appendChild(text);
                element.appendChild(line);
                element.addEventListener('pointerenter', () => this.emit('focus', this._value));
                element.addEventListener('pointerleave', () => this.emit('blur', this._value));
                element.style.cursor = 'pointer';
                element.addEventListener('click', () => this.emit('activate', this._value));
                this._add(element);
            } else if (this._hasCategory) {
                this._bold('category', initializer.category);
            } else if (type) {
                const value = type.toString().split('<').join('&lt;').split('>').join('&gt;');
                this._code('tensor', value);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    render() {
        return [this.element];
    }

    expand() {
        try {
            const initializer = this._value.initializer;
            if (this._hasId && this._hasCategory) {
                this._bold('category', initializer.category);
            }
            let type = null;
            let denotation = null;
            if (this._value.type) {
                type = this._value.type.toString();
                denotation = this._value.type.denotation || null;
            }
            if (type && (this._hasId || this._hasCategory)) {
                this._code('tensor', type.split('<').join('&lt;').split('>').join('&gt;'));
            }
            if (denotation) {
                this._code('denotation', denotation);
            }
            const description = this._value.description;
            if (description) {
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.innerHTML = description;
                this.add(line);
            }
            const identifier = this._value.identifier;
            if (identifier !== undefined) {
                this._bold('identifier', identifier);
            }
            const layout = this._value.type ? this._value.type.layout : null;
            if (layout) {
                this._bold('layout', layout.replace('.', ' '));
            }
            const quantization = this._value.quantization;
            if (quantization) {
                if (typeof quantization.type !== 'string') {
                    throw new view.Error('Unsupported quantization value.');
                }
                const value = new view.Quantization(quantization).toString();
                if (quantization.type && (quantization.type !== 'linear' || (value && value !== 'q'))) {
                    const line = this.createElement('div', 'sidebar-item-value-line-border');
                    const content = [
                        `<span class='sidebar-item-value-line-content'>quantization: <b>${quantization.type}</b></span>`
                    ];
                    if (value) {
                        content.push(`<pre style='margin: 4px 0 2px 0'>${value}</pre>`);
                    }
                    line.innerHTML = content.join('');
                    this._add(line);
                }
            }
            if (initializer) {
                if (initializer.location) {
                    this._bold('location', initializer.location);
                }
                const stride = initializer.stride;
                if (Array.isArray(stride) && stride.length > 0) {
                    this._code('stride', stride.join(','));
                }
                const tensor = new view.TensorView(this._view, initializer);
                const content = tensor.content;
                const line = this.createElement('div', 'sidebar-item-value-line-border');
                line.appendChild(content);
                this._add(line);
            }
        } catch (error) {
            super.error(error, false);
            this._info('ERROR', error.message);
        }
    }

    _bold(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `${name}: <b>${value}</b>`;
        this._add(line);
    }

    _code(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `${name}: <code><b>${value}</b></code>`;
        this._add(line);
    }

    _info(name, value) {
        const line = this.createElement('div');
        line.innerHTML = `<b>${name}:</b> ${value}`;
        this._add(line);
    }

    _add(child) {
        child.className = this._first === false ? 'sidebar-item-value-line-border' : 'sidebar-item-value-line';
        this.add(child);
        this._first = false;
    }
};

view.TensorView = class extends view.Expander {

    constructor(context, value, tensor) {
        super(context);
        this._value = value;
        this._tensor = tensor || new base.Tensor(value);
    }

    render() {
        if (!this._button) {
            this.enable();
            this._button = this.createElement('div', 'sidebar-item-value-button');
            this._button.setAttribute('style', 'float: left;');
            this._button.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-weight"></use></svg>`;
            this._button.addEventListener('click', () => this.toggle());
            this.control(this._button);
            const line = this.createElement('div', 'sidebar-item-value-line');
            line.classList.add('sidebar-item-disable-select');
            line.innerHTML = '&nbsp';
            this.element.appendChild(line);
        }
        return super.render();
    }

    expand() {
        try {
            const content = this.content;
            const container = this.createElement('div', 'sidebar-item-value-line-border');
            container.appendChild(content);
            this.element.appendChild(container);
        } catch (error) {
            this.error(error, false);
        }
    }

    get content() {
        const content = this.createElement('pre');
        const value = this._value;
        const tensor = this._tensor;
        if (tensor.encoding !== '<' && tensor.encoding !== '>' && tensor.encoding !== '|') {
            content.innerHTML = `Tensor encoding '${tensor.layout}' is not implemented.`;
        } else if (tensor.layout && (tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo')) {
            content.innerHTML = `Tensor layout '${tensor.layout}' is not implemented.`;
        } else if (tensor.empty) {
            content.innerHTML = 'Tensor data is empty.';
        } else if (tensor.type && tensor.type.dataType === '?') {
            content.innerHTML = 'Tensor data type is not defined.';
        } else if (tensor.type && !tensor.type.shape) {
            content.innerHTML = 'Tensor shape is not defined.';
        } else {
            content.innerHTML = tensor.toString();
            if (this._host.save && value.type.shape && value.type.shape.dimensions && value.type.shape.dimensions.length > 0) {
                this._saveButton = this.createElement('div', 'sidebar-item-value-button');
                this._saveButton.classList.add('sidebar-item-value-button-context');
                this._saveButton.setAttribute('style', 'float: right;');
                this._saveButton.innerHTML = '&#x1F4BE;';
                this._saveButton.addEventListener('click', async () => {
                    await this.export();
                });
                content.insertBefore(this._saveButton, content.firstChild);
            }
        }
        return content;
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('div', 'sidebar-item-value-line');
        element.innerHTML = `<b>ERROR:</b> ${error.message}`;
        this.element.appendChild(element);
    }

    async export() {
        const tensor = this._tensor;
        const defaultPath = tensor.name ? tensor.name.split('/').join('_').split(':').join('_').split('.').join('_') : 'tensor';
        const file = await this._host.save('NumPy Array', 'npy', defaultPath);
        if (file) {
            try {
                let data_type = '?';
                switch (tensor.type.dataType) {
                    case 'boolean': data_type = 'bool'; break;
                    case 'bfloat16': data_type = 'float32'; break;
                    case 'float8e5m2': data_type = 'float16'; break;
                    case 'float8e5m2fnuz': data_type = 'float16'; break;
                    case 'float8e4m3fn': data_type = 'float16'; break;
                    case 'float8e4m3fnuz': data_type = 'float16'; break;
                    case 'int4': data_type = 'int8'; break;
                    default: data_type = tensor.type.dataType; break;
                }
                const execution = new python.Execution();
                const bytes = execution.invoke('io.BytesIO', []);
                const dtype = execution.invoke('numpy.dtype', [data_type]);
                const array = execution.invoke('numpy.asarray', [tensor.value, dtype]);
                execution.invoke('numpy.save', [bytes, array]);
                bytes.seek(0);
                const blob = new Blob([bytes.read()], { type: 'application/octet-stream' });
                await this._host.export(file, blob);
            } catch (error) {
                this.error(error, 'Error saving NumPy tensor.', null);
            }
        }
    }
};

view.NodeView = class extends view.Expander {

    constructor(context, node) {
        super(context);
        this._node = node;
        const name = node.name;
        const type = node.type ? node.type.name : '';
        if (name && type) {
            this.enable();
        }
        if (type) {
            const type = node.type.name;
            const element = this.createElement('div', 'sidebar-item-value-line');
            element.innerHTML = `<span class='sidebar-item-value-line-content'>node: <b>${type || ' '}</b></span>`;
            element.addEventListener('pointerenter', () => this.emit('focus', this._node));
            element.addEventListener('pointerleave', () => this.emit('blur', this._node));
            element.addEventListener('click', () => this.emit('activate', this._node));
            element.style.cursor = 'pointer';
            this.element.appendChild(element);
        } else {
            const element = this.createElement('div', 'sidebar-item-value-line');
            element.innerHTML = `<span class='sidebar-item-value-line-content'>name: <b>${name || ' '}</b></span>`;
            element.addEventListener('pointerenter', () => this.emit('focus', this._node));
            element.addEventListener('pointerleave', () => this.emit('blur', this._node));
            element.addEventListener('click', () => this.emit('activate', this._node));
            element.style.cursor = 'pointer';
            this.element.appendChild(element);
        }
    }

    expand() {
        const name = this._node.name;
        const element = this.createElement('div', 'sidebar-item-value-line-border');
        element.innerHTML = `<span class='sidebar-item-value-line-content'>name: <b>${name}</b></span>`;
        element.addEventListener('pointerenter', () => this.emit('focus', this._node));
        element.addEventListener('pointerleave', () => this.emit('blur', this._node));
        element.addEventListener('click', () => this.emit('activate', this._node));
        element.style.cursor = 'pointer';
        this.element.appendChild(element);
    }
};

view.NodeListView = class extends view.Control {

    constructor(context, list) {
        super(context);
        this._elements = [];
        for (const node of list) {
            const item = new view.NodeView(this._view, node);
            item.on('focus', (sender, value) => this.emit('focus', value));
            item.on('blur', (sender, value) => this.emit('blur', value));
            item.on('activate', (sender, value) => this.emit('activate', value));
            item.on('deactivate', (sender, value) => this.emit('deactivate', value));
            item.on('select', (sender, value) => this.emit('select', value));
            item.toggle();
            for (const element of item.render()) {
                this._elements.push(element);
            }
        }
    }

    render() {
        return this._elements;
    }
};

view.ConnectionSidebar = class extends view.ObjectSidebar {

    constructor(context, value, from, to) {
        super(context);
        this._value = value;
        this._from = from;
        this._to = to;
    }

    render() {
        const value = this._value;
        const from = this._from;
        const to = this._to;
        const [name] = value.name.split('\n');
        this.addProperty('name', name);
        if (value.type) {
            const item = new view.ValueView(this._view, value);
            this.addEntry('type', item);
            item.toggle();
        }
        if (from) {
            this.addHeader('Inputs');
            this.addNodeList('from', [from]);
        }
        if (Array.isArray(to) && to.length > 0) {
            this.addHeader('Outputs');
            this.addNodeList('to', to);
        }
    }

    addNodeList(name, list) {
        const entry = new view.NodeListView(this._view, list);
        entry.on('focus', (sender, value) => {
            this.emit('focus', value);
            this._focused = this._focused || new Set();
            this._focused.add(value);
        });
        entry.on('blur', (sender, value) => {
            this.emit('blur', value);
            this._focused = this._focused || new Set();
            this._focused.delete(value);
        });
        entry.on('select', (sender, value) => this.emit('select', value));
        entry.on('activate', (sender, value) => this.emit('activate', value));
        this.addEntry(name, entry);
    }

    activate() {
        this.emit('select', this._value);
    }

    deactivate() {
        this.emit('select', null);
        if (this._focused) {
            for (const value of this._focused) {
                this.emit('blur', value);
            }
            this._focused.clear();
        }
    }
};

view.TensorSidebar = class extends view.ObjectSidebar {

    constructor(context, value) {
        super(context);
        this._value = value;
        this._tensor = new base.Tensor(value.value[0].initializer);
    }

    render() {
        const [value] = this._value.value;
        const tensor = value.initializer;
        const name = tensor && tensor.name ? tensor.name : value.name.split('\n')[0];
        if (name) {
            this.addProperty('name', name);
        }
        if (tensor) {
            const category = tensor.category;
            if (category) {
                this.addProperty('category', category);
            }
            const description = tensor.description;
            if (description) {
                this.addProperty('description', description);
            }
            const type = tensor.type;
            if (type) {
                const dataType = type.dataType;
                this.addProperty('type', `${dataType}`, 'code');
                const shape = type.shape && Array.isArray(type.shape.dimensions) ? type.shape.dimensions.toString(', ') : '?';
                this.addProperty('shape', `${shape}`, 'code');
                const denotation = type.denotation;
                if (denotation) {
                    this.addProperty('denotation', denotation, 'code');
                }
                const layout = type.layout;
                if (layout) {
                    this.addProperty('layout', layout.replace('.', ' '));
                }
            }
            const location = tensor.location;
            if (location) {
                this.addProperty('location', tensor.location);
            }
            const stride = tensor.stride;
            if (Array.isArray(stride) && stride.length > 0) {
                this.addProperty('stride', stride.join(','), 'code');
            }
            const value = new view.TensorView(this._view, tensor, this._tensor);
            this.addEntry('value', value);
            const metadata = tensor.metadata;
            if (Array.isArray(metadata) && metadata.length > 0) {
                this.addHeader('Metadata');
                for (const argument of tensor.metadata) {
                    this.addProperty(argument.name, argument.value);
                }
            }
        }
        // Metrics
        if (value.initializer) {
            if (!this._tensor.empty) {
                if (!this._metrics) {
                    this._metrics = new metrics.Tensor(this._tensor);
                }
                if (this._metrics.metrics.length > 0) {
                    this.addHeader('Metrics');
                    for (const metric of this._metrics.metrics) {
                        const value = metric.type === 'percentage' ? `${(metric.value * 100).toFixed(1)}%` : metric.value;
                        this.addProperty(metric.name, [value]);
                    }
                }
            }
        }
    }

    activate() {
        this.emit('select', this._value);
    }

    deactivate() {
        this.emit('select', null);
    }
};

view.ModelSidebar = class extends view.ObjectSidebar {

    constructor(context, model, graph, signature) {
        super(context);
        this._model = model;
        this._graph = graph;
        this._signature = signature;
    }

    render() {
        const model = this._model;
        const graph = this._graph;
        const signature = this._signature;
        if (model.format) {
            this.addProperty('format', model.format);
        }
        if (model.producer) {
            this.addProperty('producer', model.producer);
        }
        if (model.name) {
            this.addProperty('name', model.name);
        }
        if (model.version) {
            this.addProperty('version', model.version);
        }
        if (model.description) {
            this.addProperty('description', model.description);
        }
        if (model.domain) {
            this.addProperty('domain', model.domain);
        }
        if (model.imports) {
            this.addProperty('imports', model.imports);
        }
        if (model.runtime) {
            this.addProperty('runtime', model.runtime);
        }
        if (model.source) {
            this.addProperty('source', model.source);
        }
        const graphs = Array.isArray(model.graphs) ? model.graphs : [];
        if (graphs.length === 1 && graphs[0].name) {
            this.addProperty('graph', graphs[0].name);
        } else if (graphs.length > 1) {
            const entries = new Map();
            for (const graph of model.graphs) {
                entries.set(graph.name, graph);
            }
            const selector = new view.SelectView(this._view, entries, graph);
            selector.on('change', (sender, data) => this.emit('update-active-graph', data));
            this.addEntry('graph', selector);
        }
        if (graph && Array.isArray(graph.signatures) && graph.signatures.length > 0) {
            const entries = new Map();
            entries.set('', graph);
            for (const signature of graph.signatures) {
                entries.set(signature.name, signature);
            }
            const selector = new view.SelectView(this._view, entries, signature || graph);
            selector.on('change', (sender, data) => this.emit('update-active-graph-signature', data));
            this.addEntry('signature', selector);
        }
        const metadata = model.metadata;
        if (Array.isArray(metadata) && metadata.length > 0) {
            this.addHeader('Metadata');
            for (const argument of metadata) {
                this.addProperty(argument.name, argument.value);
            }
        }
        if (graph) {
            if (graph.version) {
                this.addProperty('version', graph.version);
            }
            if (graph.type) {
                this.addProperty('type', graph.type);
            }
            if (graph.tags) {
                this.addProperty('tags', graph.tags);
            }
            if (graph.description) {
                this.addProperty('description', graph.description);
            }
            const inputs = signature ? signature.inputs : graph.inputs;
            const outputs = signature ? signature.outputs : graph.outputs;
            if (Array.isArray(inputs) && inputs.length > 0) {
                this.addHeader('Inputs');
                for (const input of inputs) {
                    this.addArgument(input.name, input);
                }
            }
            if (Array.isArray(outputs) && outputs.length > 0) {
                this.addHeader('Outputs');
                for (const output of outputs) {
                    this.addArgument(output.name, output);
                }
            }
            const metadata = graph.metadata;
            if (Array.isArray(metadata) && metadata.length > 0) {
                this.addHeader('Metadata');
                for (const argument of metadata) {
                    this.addProperty(argument.name, argument.value);
                }
            }
        }
    }

    addArgument(name, argument) {
        const value = new view.ArgumentView(this._view, argument);
        value.on('focus', (sender, value) => this.emit('focus', value));
        value.on('blur', (sender, value) => this.emit('blur', value));
        value.on('activate', (sender, value) => this.emit('activate', value));
        value.on('deactivate', (sender, value) => this.emit('deactivate', value));
        value.on('select', (sender, value) => this.emit('select', value));
        value.toggle();
        this.addEntry(name, value);
    }
};

view.DocumentationSidebar = class extends view.Control {

    constructor(context, type) {
        super(context);
        this._type = type;
    }

    render() {
        if (!this.element) {
            this.element = this.createElement('div', 'sidebar-documentation');
            const type = view.Documentation.open(this._type);
            this._append(this.element, 'h1', type.name);
            if (type.summary) {
                this._append(this.element, 'p', type.summary);
            }
            if (type.description) {
                this._append(this.element, 'p', type.description);
            }
            if (Array.isArray(type.attributes) && type.attributes.length > 0) {
                this._append(this.element, 'h2', 'Attributes');
                const attributes = this._append(this.element, 'dl');
                for (const attribute of type.attributes) {
                    this._append(attributes, 'dt', attribute.name + (attribute.type ? `: <tt>${attribute.type}</tt>` : ''));
                    this._append(attributes, 'dd', attribute.description);
                }
                this.element.appendChild(attributes);
            }
            if (Array.isArray(type.inputs) && type.inputs.length > 0) {
                this._append(this.element, 'h2', `Inputs${type.inputs_range ? ` (${type.inputs_range})` : ''}`);
                const inputs = this._append(this.element, 'dl');
                for (const input of type.inputs) {
                    this._append(inputs, 'dt', input.name + (input.type ? `: <tt>${input.type}</tt>` : '') + (input.option ? ` (${input.option})` : ''));
                    this._append(inputs, 'dd', input.description);
                }
            }
            if (Array.isArray(type.outputs) && type.outputs.length > 0) {
                this._append(this.element, 'h2', `Outputs${type.outputs_range ? ` (${type.outputs_range})` : ''}`);
                const outputs = this._append(this.element, 'dl');
                for (const output of type.outputs) {
                    this._append(outputs, 'dt', output.name + (output.type ? `: <tt>${output.type}</tt>` : '') + (output.option ? ` (${output.option})` : ''));
                    this._append(outputs, 'dd', output.description);
                }
            }
            if (Array.isArray(type.type_constraints) && type.type_constraints.length > 0) {
                this._append(this.element, 'h2', 'Type Constraints');
                const type_constraints = this._append(this.element, 'dl');
                for (const type_constraint of type.type_constraints) {
                    this._append(type_constraints, 'dt', `${type_constraint.type_param_str}: ${type_constraint.allowed_type_strs.map((item) => `<tt>${item}</tt>`).join(', ')}`);
                    this._append(type_constraints, 'dd', type_constraint.description);
                }
            }
            if (Array.isArray(type.examples) && type.examples.length > 0) {
                this._append(this.element, 'h2', 'Examples');
                for (const example of type.examples) {
                    this._append(this.element, 'h3', example.summary);
                    this._append(this.element, 'pre', example.code);
                }
            }
            if (Array.isArray(type.references) && type.references.length > 0) {
                this._append(this.element, 'h2', 'References');
                const references = this._append(this.element, 'ul');
                for (const reference of type.references) {
                    this._append(references, 'li', reference.description);
                }
            }
            if (this._host.type === 'Electron') {
                this.element.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        const url = e.target.href;
                        if (url.startsWith('http://') || url.startsWith('https://')) {
                            e.preventDefault();
                            this.emit('navigate', { link: url });
                        }
                    }
                });
            }
        }
    }

    _append(parent, type, content) {
        const element = this.createElement(type);
        if (content) {
            element.innerHTML = content;
        }
        parent.appendChild(element);
        return element;
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('span');
        element.innerHTML = `<b>ERROR:</b> ${error.message}`;
        this.element.appendChild(element);
    }
};

view.FindSidebar = class extends view.Control {

    constructor(context, graph, signature) {
        super(context);
        this._graph = graph;
        this._signature = signature;
        this._state = {
            query: '',
            node: true,
            connection: true,
            weight: true
        };
        this._toggles = {
            node: { hide: 'Hide Nodes', show: 'Show Nodes' },
            connection: { hide: 'Hide Connections', show: 'Show Connections' },
            weight: { hide: 'Hide Weights', show: 'Show Weights' }
        };
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

    focus(state) {
        this._state = state || this._state;
        this._query.focus();
        this._query.value = '';
        this._query.value = this._state.query;
        for (const [name, toggle] of Object.entries(this._toggles)) {
            toggle.checkbox.checked = this._state[name];
            toggle.element.setAttribute('title', this._state[name] ? toggle.hide : toggle.show);
        }
        this._update();
    }

    _clear() {
        for (const element of this._focused) {
            this._blur(element);
        }
        this._focused.clear();
        this._table.clear();
        this._edges.clear();
        const unquote = this._state.query.match(new RegExp(/^'(.*)'|"(.*)"$/));
        if (unquote) {
            this._exact = true;
            const term = unquote[1] || unquote[2];
            this._terms = [term];
        } else {
            this._exact = false;
            this._terms = this._state.query.trim().toLowerCase().split(' ').map((term) => term.trim()).filter((term) => term.length > 0);
        }
    }

    _term(value) {
        if (this._exact) {
            return value === this._terms[0];
        }
        value = value.toLowerCase();
        return this._terms.every((term) => value.indexOf(term) !== -1);
    }

    _value(value) {
        if (this._terms.length === 0) {
            return true;
        }
        if (value.name && this._term(value.name.split('\n').shift())) {
            return true;
        }
        if (value.identifier && this._term(value.identifier)) {
            return true;
        }
        if (value.type && !this._exact) {
            for (const term of this._terms) {
                if (value.type.dataType && term === value.type.dataType.toLowerCase()) {
                    return true;
                }
                if (value.type.shape) {
                    if (term === value.type.shape.toString().toLowerCase()) {
                        return true;
                    }
                    if (value.type.shape && Array.isArray(value.type.shape.dimensions)) {
                        const dimensions = value.type.shape.dimensions.map((dimension) => dimension ? dimension.toString().toLowerCase() : '');
                        if (term === dimensions.join(',')) {
                            return true;
                        }
                        if (dimensions.some((dimension) => term === dimension)) {
                            return true;
                        }
                    }
                }
            }
        }
        return false;
    }

    _edge(value) {
        if (value.name && !this._edges.has(value.name) && this._value(value)) {
            const content = `${value.name.split('\n').shift()}`;
            this._add(value, content, 'connection'); // split custom argument id
            this._edges.add(value.name);
        }
    }

    _node(node) {
        if (this._state.connection) {
            const inputs = node.inputs;
            if (Array.isArray(inputs)) {
                for (const input of node.inputs) {
                    if (!input.type || input.type.endsWith('*')) {
                        for (const value of input.value) {
                            if (!value.initializer) {
                                this._edge(value);
                            }
                        }
                    }
                }
            }
        }
        if (this._state.node) {
            const name = node.name;
            const type = node.type.name;
            const identifier = node.identifier;
            if ((name && this._term(name)) || (type && this._term(type)) || (identifier && this._term(identifier))) {
                const content = `${name || `[${type}]`}`;
                this._add(node, content, 'node');
            }
        }
        if (this._state.weight) {
            const inputs = node.inputs;
            if (Array.isArray(inputs)) {
                for (const argument of node.inputs) {
                    if (!argument.type || argument.type.endsWith('*')) {
                        for (const value of argument.value) {
                            if (value.initializer && this._value(value)) {
                                let content = null;
                                if (value.name) {
                                    content = `${value.name.split('\n').shift()}`; // split custom argument id
                                } else if (Array.isArray(argument.value) && argument.value.length === 1 && argument.name.indexOf('.') !== -1) {
                                    content = argument.name;
                                } else if (value.type && value.type.shape && Array.isArray(value.type.shape.dimensions) && value.type.shape.dimensions.length > 0) {
                                    content = `${value.type.shape.dimensions.map((d) => (d !== null && d !== undefined) ? d : '?').join('\u00D7')}`;
                                }
                                if (content) {
                                    const target = argument.value.length === 1 ? argument : node;
                                    this._add(target, content, 'weight');
                                }
                            }
                        }
                    } else if (argument.type === 'object') {
                        this._node(argument.value);
                    } else if (argument.type === 'object[]') {
                        for (const value of argument.value) {
                            this._node(value);
                        }
                    }
                }
            }
        }
    }

    _add(value, content, icon) {
        const element = this.createElement('li');
        element.innerHTML = `<svg class='sidebar-find-content-icon'><use href="#sidebar-icon-${icon}"></use></svg>`;
        const text = this.createElement('span');
        text.innerText = content;
        element.appendChild(text);
        this._table.set(element, value);
        this._content.appendChild(element);
    }

    _focus(element) {
        if (this._table.has(element)) {
            this.emit('focus', this._table.get(element));
            this._focused.add(element);
        }
    }

    _blur(element) {
        if (this._table.has(element)) {
            this.emit('blur', this._table.get(element));
            this._focused.delete(element);
        }
    }

    _update() {
        this._content.innerHTML = '';
        try {
            this._clear();
            const inputs = this._signature ? this._signature.inputs : this._graph.inputs;
            if (this._state.connection) {
                for (const input of inputs) {
                    for (const value of input.value) {
                        this._edge(value);
                    }
                }
            }
            for (const node of this._graph.nodes) {
                this._node(node);
            }
            if (this._state.connection) {
                const outputs = this._signature ? this._signature.outputs : this._graph.inputs;
                for (const output of outputs) {
                    if (!output.type || output.type.endsWith('*')) {
                        for (const value of output.value) {
                            this._edge(value);
                        }
                    }
                }
            }
        } catch (error) {
            this.error(error, false);
        }
    }

    render() {
        this._table = new Map();
        this._focused = new Set();
        this._edges = new Set();
        this._search = this.createElement('div', 'sidebar-find-search');
        this._query = this.createElement('input', 'sidebar-find-query');
        this._search.appendChild(this._query);
        this._content = this.createElement('ol', 'sidebar-find-content');
        this._elements = [this._query, this._content];
        this._query.setAttribute('id', 'search');
        this._query.setAttribute('type', 'text');
        this._query.setAttribute('spellcheck', 'false');
        this._query.setAttribute('placeholder', 'Search');
        this._query.addEventListener('input', (e) => {
            this._state.query = e.target.value;
            this.emit('state-changed', this._state);
            this._update();
        });
        this._query.addEventListener('keydown', (e) => {
            if (e.keyCode === 0x08 && !e.altKey && !e.ctrlKey && !e.shiftKey && !e.metaKey) {
                e.stopPropagation();
            }
        });
        for (const [name, toggle] of Object.entries(this._toggles)) {
            toggle.element = this.createElement('label', 'sidebar-find-toggle');
            toggle.element.innerHTML = `<svg class='sidebar-find-toggle-icon'><use href="#sidebar-icon-${name}"></use></svg>`;
            toggle.element.setAttribute('title', this._state[name] ? toggle.hide : toggle.show);
            toggle.checkbox = this.createElement('input');
            toggle.checkbox.setAttribute('type', 'checkbox');
            toggle.checkbox.setAttribute('data', name);
            toggle.checkbox.addEventListener('change', (e) => {
                const name = e.target.getAttribute('data');
                this._state[name] = e.target.checked;
                const toggle = this._toggles[name];
                toggle.element.setAttribute('title', e.target.checked ? toggle.hide : toggle.show);
                this.emit('state-changed', this._state);
                this._update();
            });
            toggle.element.insertBefore(toggle.checkbox, toggle.element.firstChild);
            this._search.appendChild(toggle.element);
        }
        this._content.addEventListener('click', (e) => {
            if (this._table.has(e.target)) {
                this.emit('select', this._table.get(e.target));
            }
        });
        this._content.addEventListener('dblclick', (e) => {
            if (this._table.has(e.target)) {
                this.emit('activate', this._table.get(e.target));
            }
        });
        this._content.addEventListener('pointerover', (e) => {
            for (const element of this._focused) {
                this._blur(element);
            }
            this._focus(e.target);
        });
    }

    get element() {
        return [this._search, this._content];
    }

    deactivate() {
        for (const element of this._focused) {
            this._blur(element);
        }
        this._table.clear();
        this._focused.clear();
    }

    error(error, fatal) {
        super.error(error, fatal);
        const element = this.createElement('li');
        element.innerHTML = `<b>ERROR:</b> ${error.message}`;
        this._content.appendChild(element);
    }
};

view.Quantization = class {

    constructor(quantization) {
        Object.assign(this, quantization);
    }

    toString() {
        if (this.type === 'linear' || /^quant\d\d?_.*$/.test(this.type)) {
            const content = [];
            const scale = this.scale || [];
            const offset = this.offset || [];
            const bias = this.bias || [];
            const max = this.max || [];
            const min = this.min || [];
            const length = Math.max(scale.length, offset.length, bias.length, min.length, max.length);
            const size = length.toString().length;
            for (let i = 0; i < length; i++) {
                let s = 'q';
                let bracket = false;
                if (i < offset.length && offset[i] !== undefined && offset[i] !== 0 && offset[i] !== 0n) {
                    const value = offset[i];
                    s = value > 0 ? `${s} - ${value}` : `${s} + ${-value}`;
                    bracket = true;
                }
                if (i < scale.length && scale[i] !== undefined && scale[i] !== 1 && scale[i] !== 1n) {
                    const value = scale[i];
                    s = bracket ? `(${s})` : s;
                    s = `${value} * ${s}`;
                    bracket = true;
                }
                if (i < bias.length && bias[i] !== undefined && bias[i] !== 0 && bias[i] !== 0n) {
                    const value = bias[i];
                    s = bracket ? `(${s})` : s;
                    s = value < 0 ? `${s} - ${-value}` : `${s} + ${value}`;
                }
                if (i < min.length && min[i] !== undefined && min[i] !== 0 && min[i] !== 0n) {
                    s = `${min[i]} \u2264 ${s}`;
                }
                if (i < max.length && max[i] !== undefined && max[i] !== 0 && max[i] !== 0n) {
                    s = `${s} \u2264 ${max[i]}`;
                }
                content.push(length > 1 ? `${i.toString().padStart(size, ' ')}: ${s}` : `${s}`);
            }
            return content.join('\n');
        } else if (this.type === 'lookup') {
            const size = this.value.length.toString().length;
            return this.value.map((value, index) => `${index.toString().padStart(size, ' ')}: ${value}`).join('\n');
        } else if (this.type === 'annotation') {
            return Array.from(this.value).map(([name, value]) => `${name} = ${value}`).join('\n');
        } else if (/^q\d_[01k]$/.test(this.type) || /^iq\d_[xsnlm]+$/.test(this.type)) {
            return '';
        }
        throw new view.Error(`Unknown quantization type '${this.type}'.`);
    }
};

view.Documentation = class {

    static open(source) {
        if (source) {
            const generator = markdown.Generator.open();
            const target = {};
            if (source.name) {
                target.name = source.name;
            }
            if (source.module) {
                target.module = source.module;
            }
            if (source.category) {
                target.category = source.category;
            }
            if (source.summary) {
                target.summary = generator.html(source.summary);
            }
            if (source.description) {
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
                    if (source.description) {
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
                    if (source.description) {
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
            if (source.status !== undefined) {
                target.status = source.status;
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
        return null;
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
            return `${value.__module__}.${value.__name__}`;
        }
        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'function') {
            return `${value.__module__}.${value.__name__}`;
        }
        if (typeof value === 'function') {
            return value();
        }
        if (value && typeof value === 'bigint') {
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
                    throw new Error(`Invalid shape '${JSON.stringify(value)}'.`);
                }
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'graph':
                return value ? value.name : '(null)';
            case 'graph[]':
                return value ? value.map((graph) => graph.name).join(', ') : '(null)';
            case 'tensor': {
                if (value === null) {
                    return '(null)';
                }
                return view.Formatter.tensor(value);
            }
            case 'object':
            case 'function':
                return value.type.name;
            case 'object[]':
            case 'function[]':
                return value ? value.map((item) => item.type.name).join(', ') : '(null)';
            case 'type':
                return value ? value.toString() : '(null)';
            case 'type[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            default:
                break;
        }
        if (typeof value === 'string' && (!type || type !== 'string')) {
            return quote ? `"${value}"` : value;
        }
        if (Array.isArray(value)) {
            if (value.length === 0) {
                return quote ? '[]' : '';
            }
            let ellipsis = false;
            if (value.length > 1000) {
                value = value.slice(0, 1000);
                ellipsis = true;
            }
            const itemType = (type && type.endsWith('[]')) ? type.substring(0, type.length - 2) : null;
            const array = value.map((value) => {
                if (value && typeof value === 'bigint') {
                    return value.toString();
                }
                if (Number.isNaN(value)) {
                    return 'NaN';
                }
                const quote = !itemType || itemType === 'string';
                return this._format(value, itemType, quote);
            });
            if (ellipsis) {
                array.push('\u2026');
            }
            return quote ? ['[', array.join(', '), ']'].join(' ') : array.join(', ');
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
        const entries = Object.entries(value).filter(([name]) => !name.startsWith('__') && !name.endsWith('__'));
        if (entries.length === 1) {
            list = [this._format(entries[0][1], null, true)];
        } else {
            list = entries.map(([name, value]) => `${name}: ${this._format(value, null, true)}`);
        }
        let objectType = value.__type__;
        if (!objectType && value.constructor.name && value.constructor.name !== 'Object') {
            objectType = value.constructor.name;
        }
        if (objectType) {
            return objectType + (list.length === 0 ? '()' : ['(', list.join(', '), ')'].join(''));
        }
        switch (list.length) {
            case 0:
                return quote ? '()' : '';
            case 1:
                return list[0];
            default:
                return quote ? ['(', list.join(', '), ')'].join(' ') : list.join(', ');
        }
    }

    static tensor(value) {
        const type = value.type;
        if (type && type.shape && type.shape.dimensions && Array.isArray(type.shape.dimensions)) {
            if (type.shape.dimensions.length === 0) {
                const tensor = new base.Tensor(value);
                const encoding = tensor.encoding;
                if ((encoding === '<' || encoding === '>' || encoding === '|') && !tensor.empty && tensor.type.dataType !== '?') {
                    let content = tensor.toString();
                    if (content && content.length > 10) {
                        content = `${content.substring(0, 10)}\u2026`;
                    }
                    return content;
                }
            }
            const content = type.shape.dimensions.map((d) => (d !== null && d !== undefined) ? d : '?').join('\u00D7');
            return `\u3008${content}\u3009`;
        }
        return '\u3008\u2026\u3009';
    }
};

markdown.Generator = class {

    static open() {
        if (!markdown.Generator.generator) {
            markdown.Generator.generator = new markdown.Generator();
        }
        return markdown.Generator.generator;
    }

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
        this._emEndUndRegExp = /[^\s]_(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g;
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
        this._cache = new Map();
    }

    html(source) {
        if (this._cache.has(source)) {
            return this._cache.get(source);
        }
        const tokens = [];
        const links = new Map();
        source = source.replace(/\r\n|\r/g, '\n').replace(/\t/g, '    ');
        this._tokenize(source, tokens, links, true);
        this._tokenizeBlock(tokens, links);
        const target = this._render(tokens, true);
        if (this._cache.size > 256) {
            this._cache.delete(this._cache.keys().next().value);
        }
        this._cache.set(source, target);
        return target;
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
                    lastToken.text += `\n${match[0].trimRight()}`;
                } else {
                    const text = match[0].replace(/^ {4}/gm, '').replace(/\n*$/, '');
                    tokens.push({ type: 'code', text });
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
                    const [, indent] = matchIndent;
                    content = content.split('\n').map((node) => {
                        const match = node.match(/^\s+/);
                        return (match !== null && match[0].length >= indent.length) ? node.slice(indent.length) : node;
                    }).join('\n');
                }
                tokens.push({ type: 'code', language, text: content });
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
                    const token = { type: 'table', header, align, cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        } else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        } else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        } else {
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
                tokens.push({ type: 'blockquote', text, tokens: this._tokenize(text, [], links, top) });
                continue;
            }
            match = this._listRegExp.exec(source);
            if (match) {
                const [value, , bull] = match;
                const ordered = bull.length > 1;
                const parent = bull[bull.length - 1] === ')';
                let raw = value;
                const list = { type: 'list', raw, ordered, start: ordered ? Number(bull.slice(0, -1)) : '', loose: false, items: [] };
                const itemMatch = value.match(this._itemRegExp);
                let next = false;
                const length = itemMatch.length;
                for (let i = 0; i < length; i++) {
                    let item = itemMatch[i];
                    raw = item;
                    let space = item.length;
                    item = item.replace(/^ *([*+-]|\d+[.)]) ?/, '');
                    if (item.indexOf('\n ') !== -1) {
                        space -= item.length;
                        item = item.replace(new RegExp(`^ {1,${space}}`, 'gm'), '');
                    }
                    if (i !== length - 1) {
                        const [bullet] = this._bulletRegExp.exec(itemMatch[i + 1]);
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
                    let checked = false;
                    if (task) {
                        checked = item[1] !== ' ';
                        item = item.replace(/^\[[ xX]\] +/, '');
                    }
                    list.items.push({ type: 'list_item', raw, task, checked, loose, text: item });
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
                    const token = { type: 'table', header, align, cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        } else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        } else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        } else {
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
                    lastToken.text += `\n${match[0]}`;
                } else {
                    tokens.push({ type: 'text', text: match[0] });
                }
                continue;
            }
            throw new Error(`Unexpected '${source.charCodeAt(0)}'.`);
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
                        maskedSource = `${maskedSource.slice(0, match.index)}[${'a'.repeat(match[0].length - 2)}]${maskedSource.slice(this._reflinkSearchRegExp.lastIndex)}`;
                    }
                    continue;
                }
                break;
            }
        }
        while (maskedSource) {
            const match = this._blockSkipRegExp.exec(maskedSource);
            if (match) {
                maskedSource = `${maskedSource.slice(0, match.index)}[${'a'.repeat(match[0].length - 2)}]${maskedSource.slice(this._blockSkipRegExp.lastIndex)}`;
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
                } else if (inLink && /^<\/a>/i.test(match[0])) {
                    inLink = false;
                }
                if (!inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = true;
                } else if (inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = false;
                }
                tokens.push({ type: 'html', raw: match[0], text: match[0] });
                continue;
            }
            match = this._linkRegExp.exec(source);
            if (match) {
                let index = -1;
                const [, , ref] = match;
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
                    tokens.push({ type: 'text', text });
                } else {
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
                let cap = '';
                while ((match = endReg.exec(masked)) !== null) {
                    cap = this._strongMiddleRegExp.exec(masked.slice(0, match.index + 3));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.substring(2, cap[0].length - 2);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'strong', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._emStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '*' ? this._emEndAstRegExp : this._emEndUndRegExp;
                endReg.lastIndex = 0;
                let cap = '';
                while ((match = endReg.exec(masked)) !== null) {
                    cap = this._emMiddleRegExp.exec(masked.slice(0, match.index + 2));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.slice(1, cap[0].length - 1);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'em', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
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
                const [value, text] = match;
                source = source.substring(value.length);
                tokens.push({ type: 'del', text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                continue;
            }
            match = this._autolinkRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = this._escape(match[1]);
                const href = match[2] === '@' ? `mailto:${text}` : text;
                tokens.push({ type: 'link', text, href, tokens: [{ type: 'text', raw: text, text }] });
                continue;
            }
            if (!inLink) {
                match = this._urlRegExp.exec(source);
                if (match) {
                    const email = match[2] === '@';
                    let [value] = match;
                    if (!email) {
                        let prevCapZero = '';
                        do {
                            prevCapZero = value;
                            [value] = this._backpedalRegExp.exec(value);
                        } while (prevCapZero !== value);
                    }
                    const text = this._escape(value);
                    let href = text;
                    if (email) {
                        href = `mailto:${text}`;
                    } else if (text.startsWith('www.')) {
                        href = `http://${text}`;
                    }
                    source = source.substring(value.length);
                    tokens.push({ type: 'link', text, href, tokens: [{ type: 'text', text }] });
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
            throw new Error(`Unexpected '${source.charCodeAt(0)}'.`);
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
                    html += `<h${level}">${this._renderInline(token.tokens)}</h${level}>\n`;
                    continue;
                }
                case 'code': {
                    const code = token.text;
                    const [language] = (token.language || '').match(/\S*/);
                    html += `<pre><code${language ? ` class="language-${this._encode(language)}"` : ''}>${token.escaped ? code : this._encode(code)}</code></pre>\n`;
                    continue;
                }
                case 'table': {
                    let header = '';
                    let cell = '';
                    for (let j = 0; j < token.header.length; j++) {
                        const content = this._renderInline(token.tokens.header[j]);
                        const align = token.align[j];
                        cell += `<th${align ? ` align="${align}"` : ''}>${content}</th>\n`;
                    }
                    header += `<tr>\n${cell}</tr>\n`;
                    let body = '';
                    for (let j = 0; j < token.cells.length; j++) {
                        const row = token.tokens.cells[j];
                        cell = '';
                        for (let k = 0; k < row.length; k++) {
                            const content = this._renderInline(row[k]);
                            const align = token.align[k];
                            cell += `<td${align ? ` align="${align}"` : ''}>${content}</td>\n`;
                        }
                        body += `<tr>\n${cell}</tr>\n`;
                    }
                    html += `<table>\n<thead>\n${header}</thead>\n${body ? `<tbody>${body}</tbody>` : body}</table>\n`;
                    continue;
                }
                case 'blockquote': {
                    html += `<blockquote>\n${this._render(token.tokens, true)}</blockquote>\n`;
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
                            const checkbox = `<input ${item.checked ? 'checked="" ' : ''}disabled="" type="checkbox"> `;
                            if (loose) {
                                if (item.tokens.length > 0 && item.tokens[0].type === 'text') {
                                    item.tokens[0].text = `${checkbox} ${item.tokens[0].text}`;
                                    if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === 'text') {
                                        item.tokens[0].tokens[0].text = `${checkbox} ${item.tokens[0].tokens[0].text}`;
                                    }
                                } else {
                                    item.tokens.unshift({ type: 'text', text: checkbox });
                                }
                            } else {
                                itemBody += checkbox;
                            }
                        }
                        itemBody += this._render(item.tokens, loose);
                        body += `<li>${itemBody}</li>\n`;
                    }
                    const type = (ordered ? 'ol' : 'ul');
                    html += `<${type}${ordered && start !== 1 ? (` start="${start}"`) : ''}>\n${body}</${type}>\n`;
                    continue;
                }
                case 'html': {
                    html += token.text;
                    continue;
                }
                case 'paragraph': {
                    html += `<p>${this._renderInline(token.tokens)}</p>\n`;
                    continue;
                }
                case 'text': {
                    html += top ? '<p>' : '';
                    html += token.tokens ? this._renderInline(token.tokens) : token.text;
                    while (tokens.length > 0 && tokens[0].type === 'text') {
                        const token = tokens.shift();
                        html += `\n${token.tokens ? this._renderInline(token.tokens) : token.text}`;
                    }
                    html += top ? '</p>\n' : '';
                    continue;
                }
                default: {
                    throw new Error(`Unexpected token type '${token.type}'.`);
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
                    html += `<a href="${token.href}"${token.title ? ` title="${token.title}"` : ''} target="_blank">${text}</a>`;
                    break;
                }
                case 'image': {
                    html += `<img src="${token.href}" alt="${token.text}"${token.title ? ` title="${token.title}"` : ''}>`;
                    break;
                }
                case 'strong': {
                    const text = this._renderInline(token.tokens);
                    html += `<strong>${text}</strong>`;
                    break;
                }
                case 'em': {
                    const text = this._renderInline(token.tokens);
                    html += `<em>${text}</em>`;
                    break;
                }
                case 'codespan': {
                    html += `<code>${token.text}</code>`;
                    break;
                }
                case 'br': {
                    html += '<br>';
                    break;
                }
                case 'del': {
                    const text = this._renderInline(token.tokens);
                    html += `<del>${text}</del>`;
                    break;
                }
                default: {
                    throw new Error(`Unexpected token type '${token.type}'.`);
                }
            }
        }
        return html;
    }

    _outputLink(match, href, title) {
        title = title ? this._escape(title) : null;
        const text = match[1].replace(/\\([[\]])/g, '$1');
        return match[0].charAt(0) === '!' ?
            { type: 'image', href, title, text: this._escape(text) } :
            { type: 'link', href, title, text };
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
        } else {
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

metrics.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type;
    }
};

metrics.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this._metrics = null;
    }

    get metrics() {
        if (this._metrics === null) {
            this._metrics = [];
            this._metrics = Array.from(this._tensor.metrics || []);
            const keys = new Set(this._metrics.map((metrics) => metrics.name));
            const type = this._tensor.type;
            const shape = type.shape.dimensions;
            const size = shape.reduce((a, b) => a * b, 1);
            if (size < 0x800000 &&
                (type.dataType.startsWith('float') || type.dataType.startsWith('bfloat')) &&
                (!keys.has('sparsity') || !keys.has('min') || !keys.has('max') && !keys.has('mean') || !keys.has('max') || !keys.has('std'))) {
                const data = this._tensor.value;
                let zeros = 0;
                let min = null;
                let max = null;
                let sum = 0;
                let count = 0;
                const stack = [data];
                while (stack.length > 0) {
                    const data = stack.pop();
                    if (Array.isArray(data)) {
                        for (const element of data) {
                            stack.push(element);
                        }
                    } else {
                        zeros += data === 0 || data === 0n || data === '';
                        min = Math.min(data, min === null ? data : min);
                        max = Math.max(data, max === null ? data : max);
                        sum += data;
                        count += 1;
                    }
                }
                const mean = sum / count;
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metrics.Argument('min', min, type.dataType));
                }
                if (!keys.has('max')) {
                    this._metrics.push(new metrics.Argument('max', max, type.dataType));
                }
                if (!keys.has('mean')) {
                    this._metrics.push(new metrics.Argument('mean', mean, type.dataType));
                }
                if (!keys.has('std')) {
                    let variance = 0;
                    const stack = [data];
                    while (stack.length > 0) {
                        const data = stack.pop();
                        if (Array.isArray(data)) {
                            for (const element of data) {
                                stack.push(element);
                            }
                        } else {
                            variance += Math.pow(data - mean, 2);
                        }
                    }
                    this._metrics.push(new metrics.Argument('std', Math.sqrt(variance / count)));
                }
                if (!keys.has('sparsity')) {
                    this._metrics.push(new metrics.Argument('sparsity', count > 0 ? zeros / count : 0, 'percentage'));
                }
            }
        }
        return this._metrics;
    }
};

view.Context = class {

    constructor(context, identifier, stream) {
        this._context = context;
        this._tags = new Map();
        this._content = new Map();
        this._stream = stream || context.stream;
        identifier = typeof identifier === 'string' ? identifier : context.identifier;
        const index = Math.max(identifier.lastIndexOf('/'), identifier.lastIndexOf('\\'));
        this._base = index === -1 ? undefined : identifier.substring(0, index);
        this._identifier = index === -1 ? identifier : identifier.substring(index + 1);
    }

    get identifier() {
        return this._identifier;
    }

    get stream() {
        return this._stream;
    }

    async request(file) {
        return this._context.request(file, 'utf-8', null);
    }

    async fetch(file) {
        const stream = await this._context.request(file, null, this._base);
        return new view.Context(this, file, stream, new Map());
    }

    async require(id) {
        return this._context.require(id);
    }

    error(error, fatal) {
        if (error && this.identifier) {
            error.context = this.identifier;
        }
        this._context.error(error, fatal);
    }

    peek(type) {
        if (!this._content.has(type)) {
            this._content.set(type, undefined);
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const match = (buffer, signature) => {
                    return signature.length <= buffer.length && buffer.every((value, index) => signature[index] === undefined || signature[index] === value);
                };
                const buffer = stream.peek(Math.min(stream.length, 16));
                const skip =
                    match(buffer, [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19]) || // PyTorch
                    (type !== 'npz' && type !== 'zip' && match(buffer, [0x50, 0x4B, 0x03, 0x04])) || // Zip
                    (type !== 'hdf5' && match(buffer, [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A])) || // \x89HDF\r\n\x1A\n
                    Array.from(this._tags).some(([key, value]) => key !== 'flatbuffers' && key !== 'xml' && value.size > 0) ||
                    Array.from(this._content.values()).some((obj) => obj !== undefined);
                if (!skip) {
                    switch (type) {
                        case 'json': {
                            try {
                                const buffer = stream.peek(Math.min(this.stream.length, 0x1000));
                                if (stream.length < 0x7ffff000 &&
                                    (buffer.length < 8 || String.fromCharCode.apply(null, buffer.slice(0, 8)) !== '\x89HDF\r\n\x1A\n') &&
                                    (buffer.some((v) => v === 0x22 || v === 0x5b || v === 0x5d || v === 0x7b || v === 0x7d))) {
                                    const reader = json.TextReader.open(stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'json.gz': {
                            try {
                                const entries = this.peek('gzip');
                                if (entries && entries.size === 1) {
                                    const stream = entries.values().next().value;
                                    const reader = json.TextReader.open(stream);
                                    if (reader) {
                                        const obj = reader.read();
                                        this._content.set(type, obj);
                                    }
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'xml': {
                            try {
                                const reader = xml.TextReader.open(this._stream);
                                if (reader) {
                                    const obj = reader.read();
                                    this._content.set(type, obj);
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        case 'pkl': {
                            let unpickler = null;
                            const types = new Set();
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
                                    const execution = new python.Execution();
                                    execution.on('resolve', (_, name) => types.add(name));
                                    const pickle = execution.__import__('pickle');
                                    unpickler = new pickle.Unpickler(data);
                                }
                            } catch {
                                // continue regardless of error
                            }
                            if (unpickler) {
                                const storages = new Map();
                                unpickler.persistent_load = (saved_id) => {
                                    if (Array.isArray(saved_id) && saved_id.length > 3) {
                                        switch (saved_id[0]) {
                                            case 'storage': {
                                                const [, storage_type, key, , size] = saved_id;
                                                if (!storages.has(key)) {
                                                    const storage = new storage_type(size);
                                                    storages.set(key, storage);
                                                }
                                                return storages.get(key);
                                            }
                                            default: {
                                                throw new python.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
                                            }
                                        }
                                    }
                                    throw new view.Error("Unsupported 'persistent_load'.");
                                };
                                try {
                                    const obj = unpickler.load();
                                    this._content.set(type, obj);
                                } catch (error) {
                                    this._content.set(type, error);
                                }
                                if (Array.from(types).every((name) => !name.startsWith('__torch__.'))) {
                                    for (const name of types) {
                                        this.error(new view.Error(`Unknown type name '${name}'.`));
                                    }
                                } else {
                                    this._content.set(type, new view.Error("PyTorch standalone 'data.pkl' format not supported."));
                                }
                            }
                            break;
                        }
                        case 'hdf5': {
                            const file = hdf5.File.open(stream);
                            if (file) {
                                try {
                                    this._content.set(type, file.read());
                                } catch (error) {
                                    this._content.set(type, error);
                                }
                            }
                            break;
                        }
                        case 'zip':
                        case 'tar':
                        case 'gzip': {
                            this._content.set('zip', undefined);
                            this._content.set('tar', undefined);
                            this._content.set('gzip', undefined);
                            let stream = this._stream;
                            try {
                                const archive = zip.Archive.open(this._stream, 'gzip');
                                if (archive) {
                                    let entries = archive.entries;
                                    if (entries.size === 1) {
                                        const key = entries.keys().next().value;
                                        stream = entries.values().next().value;
                                        const name = key === '' ? this.identifier.replace(/\.gz$/, '') : key;
                                        entries = new Map([[name, stream]]);
                                    }
                                    this._content.set('gzip', entries);
                                }
                            } catch (error) {
                                this._content.set('gzip', error);
                            }
                            let skipTar = false;
                            try {
                                const archive = zip.Archive.open(stream, 'zip');
                                if (archive) {
                                    this._content.set('zip', archive.entries);
                                    skipTar = true;
                                }
                            } catch (error) {
                                this._content.set('zip', error);
                            }
                            if (!skipTar) {
                                try {
                                    const archive = tar.Archive.open(stream);
                                    if (archive) {
                                        this._content.set('tar', archive.entries);
                                    }
                                } catch (error) {
                                    this._content.set('tar', error);
                                }
                            }
                            break;
                        }
                        case 'flatbuffers.binary': {
                            try {
                                const reader = flatbuffers.BinaryReader.open(this._stream);
                                if (reader) {
                                    this._content.set('flatbuffers.binary', reader);
                                }
                            } catch (error) {
                                this._content.set('flatbuffers.binary', error);
                            }
                            break;
                        }
                        case 'npz': {
                            try {
                                const content = new Map();
                                const entries = this.peek('zip');
                                if (entries instanceof Map && entries.size > 0 &&
                                    Array.from(entries.keys()).every((name) => name.endsWith('.npy'))) {
                                    const execution = new python.Execution();
                                    for (const [name, stream] of entries) {
                                        const buffer = stream.peek();
                                        const bytes = execution.invoke('io.BytesIO', [buffer]);
                                        const array = execution.invoke('numpy.load', [bytes]);
                                        content.set(name, array);
                                    }
                                    this._content.set(type, content);
                                }
                            } catch {
                                // continue regardless of error
                            }
                            break;
                        }
                        default: {
                            throw new view.Error(`Unsupported open format type '${type}'.`);
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

    read(type, ...args) {
        if (!this._content.has(type)) {
            switch (type) {
                case 'json': {
                    const reader = json.TextReader.open(this._stream);
                    if (reader) {
                        const obj = reader.read();
                        this._content.set('json', obj);
                        return obj;
                    }
                    throw new view.Error('Invalid JSON content.');
                }
                case 'bson': {
                    const reader = json.BinaryReader.open(this._stream);
                    if (reader) {
                        return reader.read();
                    }
                    throw new view.Error('Invalid BSON content.');
                }
                case 'xml': {
                    const reader = xml.TextReader.open(this._stream);
                    if (reader) {
                        return reader.read();
                    }
                    throw new view.Error(`Invalid XML content.`);
                }
                case 'flatbuffers.binary': {
                    const reader = flatbuffers.BinaryReader.open(this._stream);
                    if (reader) {
                        this._content.set('flatbuffers.reader', reader);
                        return reader;
                    }
                    throw new view.Error('Invalid FlatBuffers content.');
                }
                case 'flatbuffers.text': {
                    const obj = this.peek('json');
                    return flatbuffers.TextReader.open(obj);
                }
                case 'protobuf.binary': {
                    return protobuf.BinaryReader.open(this._stream);
                }
                case 'protobuf.text': {
                    return protobuf.TextReader.open(this._stream);
                }
                case 'binary.big-endian': {
                    return base.BinaryReader.open(this._stream, false);
                }
                case 'binary': {
                    return base.BinaryReader.open(this._stream);
                }
                case 'text': {
                    if (typeof args[0] === 'number') {
                        const length = Math.min(this._stream.length, args[0]);
                        const buffer = this._stream.peek(length);
                        text.Reader.open(buffer);
                    }
                    return text.Reader.open(this._stream);
                }
                case 'text.decoder': {
                    return text.Decoder.open(this._stream);
                }
                default: {
                    break;
                }
            }
        }
        return this.peek(type);
    }

    tags(type) {
        if (!this._tags.has(type)) {
            let tags = new Map();
            const stream = this.stream;
            if (stream) {
                const position = stream.position;
                const signatures = [
                    [0x89, 0x48, 0x44, 0x46, 0x0D, 0x0A, 0x1A, 0x0A], // HDF5
                    [0x80, undefined, 0x8a, 0x0a, 0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19], // PyTorch
                    [0x50, 0x4b], // Zip
                    [0x1f, 0x8b] // Gzip
                ];
                const skip =
                    signatures.some((signature) => signature.length <= stream.length && stream.peek(signature.length).every((value, index) => signature[index] === undefined || signature[index] === value)) ||
                    (Array.from(this._tags).some(([key, value]) => key !== 'flatbuffers' && value.size > 0) && type !== 'pb+') ||
                    Array.from(this._content.values()).some((obj) => obj !== undefined) ||
                    (stream.length < 0x7ffff000 && json.TextReader.open(stream));
                if (!skip && stream.length < 0x7ffff000) {
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
                            case 'xml': {
                                const reader = xml.TextReader.open(stream);
                                if (reader) {
                                    const document = reader.peek();
                                    const element = document.documentElement;
                                    const namespaceURI = element.namespaceURI;
                                    const localName = element.localName;
                                    const name = namespaceURI ? `${namespaceURI}:${localName}` : localName;
                                    tags.set(name, element);
                                }
                                break;
                            }
                            default: {
                                throw new view.Error(`Unsupported tags format type '${type}'.`);
                            }
                        }
                    } catch {
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

    constructor(host, entries) {
        this._host = host;
        this._entries = entries;
    }

    async request(file, encoding, base) {
        if (base === null) {
            return this._host.request(file, encoding, base);
        }
        let stream = null;
        if (typeof base === 'string') {
            stream = this._entries.get(`${base}/${file}`) || this._entries.get(`${base}\\${file}`);
        } else {
            stream = this._entries.get(file);
        }
        if (!stream) {
            throw new view.Error('File not found.');
        }
        if (encoding) {
            const decoder = new TextDecoder(encoding);
            const buffer = stream.peek();
            const value = decoder.decode(buffer);
            return value;
        }
        return stream;
    }

    async require(id) {
        return this._host.require(id);
    }

    error(error, fatal) {
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
        this._patterns = new Set(['.zip', '.tar', '.tar.gz', '.tgz', '.gz']);
        this._factories = [];
        this.register('./server', ['.netron']);
        this.register('./pytorch', ['.pt', '.pth', '.ptl', '.pt1', '.pyt', '.pyth', '.pkl', '.pickle', '.h5', '.t7', '.model', '.dms', '.tar', '.ckpt', '.chkpt', '.tckpt', '.bin', '.pb', '.zip', '.nn', '.torchmodel', '.torchscript', '.pytorch', '.ot', '.params', '.trt', '.ff', '.ptmf', '.jit', '.pte', '.bin.index.json', 'serialized_exported_program.json', 'model.json'], ['.model', '.pt2']);
        this.register('./onnx', ['.onnx', '.onnx.data', '.onn', '.pb', '.onnxtxt', '.pbtxt', '.prototxt', '.txt', '.model', '.pt', '.pth', '.pkl', '.ort', '.ort.onnx', '.ngf', '.json', '.bin', 'onnxmodel']);
        this.register('./tflite', ['.tflite', '.lite', '.tfl', '.bin', '.pb', '.tmfile', '.h5', '.model', '.json', '.txt', '.dat', '.nb', '.ckpt', '.onnx']);
        this.register('./mxnet', ['.json', '.params'], ['.mar']);
        this.register('./coreml', ['.mlmodel', '.bin', 'manifest.json', 'metadata.json', 'featuredescriptions.json', '.pb', '.pbtxt', '.mil'], ['.mlpackage', '.mlmodelc']);
        this.register('./caffe', ['.caffemodel', '.pbtxt', '.prototxt', '.pt', '.txt']);
        this.register('./caffe2', ['.pb', '.pbtxt', '.prototxt']);
        this.register('./torch', ['.t7', '.net']);
        this.register('./tf', ['.pb', '.meta', '.pbtxt', '.prototxt', '.txt', '.pt', '.json', '.index', '.ckpt', '.graphdef', '.pbmm', /.data-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9]$/, /^events.out.tfevents./], ['.zip']);
        this.register('./tensorrt', ['.trt', '.trtmodel', '.engine', '.model', '.txt', '.uff', '.pb', '.tmfile', '.onnx', '.pth', '.dnn', '.plan', '.pt', '.dat', '.bin']);
        this.register('./keras', ['.h5', '.hd5', '.hdf5', '.keras', '.json', '.cfg', '.model', '.pb', '.pth', '.weights', '.pkl', '.lite', '.tflite', '.ckpt', '.pb', 'model.weights.npz'], ['.zip']);
        this.register('./numpy', ['.npz', '.npy', '.pkl', '.pickle', '.model', '.model2', '.mge', '.joblib']);
        this.register('./lasagne', ['.pkl', '.pickle', '.joblib', '.model', '.pkl.z', '.joblib.z']);
        this.register('./lightgbm', ['.txt', '.pkl', '.model']);
        this.register('./sklearn', ['.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z', '.pickle.dat', '.bin']);
        this.register('./megengine', ['.tm', '.mge', '.pkl']);
        this.register('./pickle', ['.pkl', '.pickle', '.joblib', '.model', '.meta', '.pb', '.pt', '.h5', '.pkl.z', '.joblib.z', '.pdstates', '.mge', '.bin', '.npy', '.pth']);
        this.register('./cntk', ['.model', '.cntk', '.cmf', '.dnn']);
        this.register('./uff', ['.uff', '.pb', '.pbtxt', '.uff.txt', '.trt', '.engine']);
        this.register('./paddle', ['.pdmodel', '.pdiparams', '.pdparams', '.pdopt', '.paddle', '__model__', '.__model__', '.pbtxt', '.txt', '.tar', '.tar.gz', '.nb']);
        this.register('./bigdl', ['.model', '.bigdl']);
        this.register('./darknet', ['.cfg', '.model', '.txt', '.weights']);
        this.register('./mediapipe', ['.pbtxt']);
        this.register('./rknn', ['.rknn', '.nb', '.onnx', '.json', '.bin']);
        this.register('./dlc', ['.dlc', 'model', '.params']);
        this.register('./armnn', ['.armnn', '.json']);
        this.register('./mnn', ['.mnn']);
        this.register('./ncnn', ['.param', '.bin', '.cfg.ncnn', '.weights.ncnn', '.ncnnmodel']);
        this.register('./tnn', ['.tnnproto', '.tnnmodel']);
        this.register('./tengine', ['.tmfile']);
        this.register('./mslite', ['.ms', '.bin']);
        this.register('./barracuda', ['.nn']);
        this.register('./circle', ['.circle']);
        this.register('./dnn', ['.dnn']);
        this.register('./xmodel', ['.xmodel']);
        this.register('./kmodel', ['.kmodel']);
        this.register('./flux', ['.bson']);
        this.register('./dl4j', ['.json', '.bin']);
        this.register('./openvino', ['.xml', '.bin']);
        this.register('./mlnet', ['.zip', '.mlnet']);
        this.register('./acuity', ['.json']);
        this.register('./imgdnn', ['.dnn', 'params', '.json']);
        this.register('./flax', ['.msgpack']);
        this.register('./om', ['.om', '.onnx', '.pb', '.engine', '.bin']);
        this.register('./gguf', ['.gguf', /^[^.]+$/]);
        this.register('./nnabla', ['.nntxt'], ['.nnp']);
        this.register('./hickle', ['.h5', '.hkl']);
        this.register('./nnef', ['.nnef', '.dat']);
        this.register('./onednn', ['.json']);
        this.register('./espresso', ['.espresso.net', '.espresso.shape', '.espresso.weights'], ['.mlmodelc']);
        this.register('./mlir', ['.mlir', '.mlir.txt']);
        this.register('./sentencepiece', ['.model']);
        this.register('./hailo', ['.hn', '.har', '.metadata.json']);
        this.register('./nnc', ['.nnc','.tflite']);
        this.register('./safetensors', ['.safetensors', '.safetensors.index.json']);
        this.register('./tvm', ['.json', '.params']);
        this.register('./modular', ['.maxviz']);
        this.register('./catboost', ['.cbm']);
        this.register('./cambricon', ['.cambricon']);
        this.register('./weka', ['.model']);
        this.register('./qnn', ['.json', '.bin', '.serialized']);
    }

    register(module, factories, containers) {
        for (const pattern of factories) {
            this._factories.push({ pattern, module });
            this._patterns.add(pattern);
        }
        for (const pattern of containers || []) {
            this._patterns.add(pattern);
        }
    }

    async open(context) {
        try {
            await this._openSignature(context);
            const content = new view.Context(context);
            const model = await this._openContext(content);
            if (!model) {
                const check = (obj) => {
                    if (obj instanceof Error) {
                        throw obj;
                    }
                    return obj instanceof Map && obj.size > 0;
                };
                let entries = context.entries;
                if (!check(entries)) {
                    entries = content.peek('zip');
                    if (!check(entries)) {
                        entries = content.peek('tar');
                        if (!check(entries)) {
                            entries = content.peek('gzip');
                        }
                    }
                }
                if (!check(entries)) {
                    this._unsupported(content);
                }
                const entryContext = await this._openEntries(entries);
                if (!entryContext) {
                    this._unsupported(content);
                }
                return this._openContext(entryContext);
            }
            return model;
        } catch (error) {
            error.context = !error.context && context && context.identifier ? context.identifier : error.context || '';
            throw error;
        }
    }

    _unsupported(context) {
        const identifier = context.identifier;
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
            } catch {
                // continue regardless of error
            }
            if (archive) {
                throw new view.Error("Archive contains no model files.");
            }
        }
        const json = () => {
            const obj = context.peek('json');
            if (obj) {
                const formats = [
                    { name: 'Netron metadata', tags: ['[].name', '[].schema'] },
                    { name: 'Netron metadata', tags: ['[].name', '[].attributes'] },
                    { name: 'Netron metadata', tags: ['[].name', '[].category'] },
                    { name: 'Netron test data', tags: ['[].type', '[].target', '[].source', '[].format', '[].link'] },
                    { name: 'Netron configuration', tags: ['recents', 'consent'] },
                    { name: 'Darkflow metadata', tags: ['net', 'type', 'model'] },
                    { name: 'keras-yolo2 configuration', tags: ['model', 'train', 'valid'] },
                    { name: 'Vulkan SwiftShader ICD manifest', tags: ['file_format_version', 'ICD'] },
                    { name: 'DeepLearningExamples configuration', tags: ['attention_probs_dropout_prob', 'hidden_act', 'hidden_dropout_prob', 'hidden_size',] },
                    { name: 'GitHub page data', tags: ['payload', 'title'] },
                    { name: 'NuGet assets', tags: ['version', 'targets', 'packageFolders'] },
                    { name: 'NuGet data', tags: ['format', 'restore', 'projects'] },
                    { name: 'NPM package', tags: ['name', 'version', 'dependencies'] },
                    { name: 'NetworkX adjacency_data', tags: ['directed', 'graph', 'nodes'] },
                    { name: 'Waifu2x data', tags: ['name', 'arch_name', 'channels'] },
                    { name: 'Waifu2x data', tags: ['[].nInputPlane', '[].nOutputPlane', '[].weight', '[].bias'] },
                    { name: 'Brain.js data', tags: ['type', 'sizes', 'layers'] },
                    { name: 'Custom Vision metadata', tags: ['CustomVision.Metadata.Version'] },
                    { name: 'W&B metadata', tags: ['program', 'host', 'executable'] },
                    { name: 'TypeScript configuration data', tags: ['compilerOptions'] },
                    { name: 'CatBoost model', tags: ['features_info', 'model_info'] },
                    { name: 'TPU-MLIR tensor location data', tags: ['file-line', 'subnet_id', 'core_id'] }, // https://github.com/sophgo/tpu-mlir/blob/master/lib/Dialect/Tpu/Transforms/Codegen/TensorLocation.cpp
                    { name: 'HTTP Archive data', tags: ['log.version', 'log.creator', 'log.entries'] }, // https://w3c.github.io/web-performance/specs/HAR/Overview.html
                    { name: 'Trace Event data', tags: ['traceEvents'] },
                    { name: 'Trace Event data', tags: ['[].pid', '[].ph'] },
                    { name: 'Diffusers configuration', tags: ['_class_name', '_diffusers_version'] },
                    { name: 'Transformers configuration', tags: ['architectures', 'model_type'] }, // https://huggingface.co/docs/transformers/en/create_a_model
                    { name: 'Transformers generation configuration', tags: ['transformers_version'] },
                    { name: 'Transformers tokenizer configuration', tags: ['tokenizer_class'] },
                    { name: 'Transformers tokenizer configuration', tags: ['<|im_start|>'] },
                    { name: 'Transformers preprocessor configuration', tags: ['crop_size', 'do_center_crop', 'image_mean', 'image_std', 'do_resize'] },
                    { name: 'Tokenizers data', tags: ['version', 'added_tokens', 'model'] }, // https://github.com/huggingface/tokenizers/blob/main/tokenizers/src/tokenizer/serialization.rs
                    { name: 'Jupyter Notebook data', tags: ['cells', 'nbformat'] },
                    { name: 'Kaggle credentials', tags: ['username','key'] },
                    { name: '.NET runtime configuration', tags: ['runtimeOptions.configProperties'] },
                    { name: '.NET dependency manifest', tags: ['runtimeTarget', 'targets', 'libraries'] }
                ];
                const match = (obj, tag) => {
                    if (tag.startsWith('[].')) {
                        tag = tag.substring(3);
                        return (Array.isArray(obj) && obj.some((item) => Object.prototype.hasOwnProperty.call(item, tag)));
                    }
                    tag = tag.split('.');
                    while (tag.length > 1) {
                        const key = tag.shift();
                        obj = obj[key];
                        if (!obj) {
                            return false;
                        }
                    }
                    return Object.prototype.hasOwnProperty.call(obj, tag[0]);
                };
                for (const format of formats) {
                    if (format.tags.every((tag) => match(obj, tag))) {
                        throw new view.Error(`Invalid file content. File contains ${format.name}.`);
                    }
                }
                const content = `${JSON.stringify(obj).substring(0, 100).replace(/\s/, '').substring(0, 48)}...`;
                throw new view.Error(`Unsupported JSON content '${content.length > 64 ? `${content.substring(0, 100)}...` : content}'.`);
            }
        };
        const pbtxt = () => {
            const formats = [
                { name: 'ImageNet LabelMap data', tags: ['entry', 'entry.target_class'] },
                { name: 'StringIntLabelMapProto data', tags: ['item', 'item.id', 'item.name'] },
                { name: 'caffe.LabelMap data', tags: ['item', 'item.name', 'item.label'] },
                { name: 'Triton Inference Server configuration', tags: ['name', 'platform', 'input', 'output'] },
                { name: 'TensorFlow OpList data', tags: ['op', 'op.name', 'op.input_arg'] },
                { name: 'vitis.ai.proto.DpuModelParamList data', tags: ['model', 'model.name', 'model.kernel'] },
                { name: 'object_detection.protos.DetectionModel data', tags: ['model', 'model.ssd'] },
                { name: 'object_detection.protos.DetectionModel data', tags: ['model', 'model.faster_rcnn'] },
                { name: 'tensorflow.CheckpointState data', tags: ['model_checkpoint_path', 'all_model_checkpoint_paths'] },
                { name: 'apollo.perception.camera.traffic_light.detection.DetectionParam data', tags: ['min_crop_size', 'crop_method'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['caffe_ssd'] }, // https://github.com/TexasInstruments/edgeai-mmdetection/blob/master/mmdet/utils/proto/mmdet_meta_arch.proto
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tf_od_api_ssd'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_ssd'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_faster_rcnn'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_yolo'] },
                { name: 'tidl_meta_arch.TIDLMetaArch data', tags: ['tidl_retinanet'] },
                { name: 'domi.InsertNewOps data', tags: ['aipp_op'] } // https://github.com/Ascend/parser/blob/development/parser/proto/insert_op.proto
            ];
            const tags = context.tags('pbtxt');
            if (tags.size > 0) {
                for (const format of formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const entries = [];
                entries.push(...Array.from(tags).filter(([key]) => key.toString().indexOf('.') === -1));
                entries.push(...Array.from(tags).filter(([key]) => key.toString().indexOf('.') !== -1));
                const content = entries.map(([key, value]) => value === true ? key : `${key}:${JSON.stringify(value)}`).join(',');
                throw new view.Error(`Unsupported Protocol Buffers text content '${content.length > 64 ? `${content.substring(0, 100)}...` : content}'.`);
            }
        };
        const pb = () => {
            const tags = context.tags('pb+');
            if (Object.keys(tags).length > 0) {
                const formats = [
                    { name: 'sentencepiece.ModelProto data', tags: [[1,[[1,2],[2,5],[3,0]]],[2,[[1,2],[2,2],[3,0],[4,0],[5,2],[6,0],[7,2],[10,5],[16,0],[40,0],[41,0],[42,0],[43,0]]],[3,[]],[4,[]],[5,[]]] },
                    { name: 'mediapipe.BoxDetectorIndex data', tags: [[1,[[1,[[1,[[1,5],[2,5],[3,5],[4,5],[6,0],[7,5],[8,5],[10,5],[11,0],[12,0]]],[2,5],[3,[]]]],[2,false],[3,false],[4,false],[5,false]]],[2,false],[3,false]] },
                    { name: 'third_party.tensorflow.python.keras.protobuf.SavedMetadata data', tags: [[1,[[1,[[1,0],[2,0]]],[2,0],[3,2],[4,2],[5,2]]]] },
                    { name: 'pblczero.Net data', tags: [[1,5],[2,2],[3,[[1,0],[2,0],[3,0]],[10,[[1,[]],[2,[]],[3,[]],[4,[]],[5,[]],[6,[]]]],[11,[]]]] }, // https://github.com/LeelaChessZero/lczero-common/blob/master/proto/net.proto
                    { name: 'chrome_browser_media.PreloadedData', tags: [[1,2]], identifier: 'preloaded_data.pb' }, // https://github.com/kiwibrowser/src/blob/86afd150b847c9dd6f9ad3faddee1a28b8c9b23b/chrome/browser/media/media_engagement_preload.proto#L9
                    { name: 'mind_ir.ModelProto', tags: [[1,2],[2,2],[5,2],[7,[]],[10,0],[12,[]],[13,0]] }, // https://github.com/mindspore-ai/mindspore/blob/master/mindspore/core/proto/mind_ir.proto
                    { name: 'mindspore.irpb.Checkpoint', tags: [[1,[[1,2],[2,[[1,0],[2,2],[3,2]]]]]] }, // https://github.com/mindspore-ai/mindspore/blob/master/mindspore/ccsrc/utils/checkpoint.proto
                    { name: 'optimization_guide.proto.PageTopicsOverrideList data', tags: [[1,[[1,2],[2,[]]]]] }, // https://github.com/chromium/chromium/blob/main/components/optimization_guide/proto/page_topics_override_list.proto
                    { name: 'optimization_guide.proto.ModelInfo data', tags: [[1,0],[2,0],[4,0],[6,false],[7,[]],[9,0]] }, // https://github.com/chromium/chromium/blob/22b0d711657b451b61d50dd2e242b3c6e38e6ef5/components/optimization_guide/proto/models.proto#L80
                    { name: 'Hobot Dnn data', tags: [[1,0],[2,0],[4,[[1,2],[2,2]]]] }, // https://github.com/HorizonRDK/hobot_dnn
                    { name: 'Hobot Dnn data', tags: [[1,0],[2,0],[6,[1,[[1,2],[2,2]]]]] }, // https://github.com/HorizonRDK/hobot_dnn
                ];
                const match = (tags, schema) => {
                    for (const [key, inner] of schema) {
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
                        } else if (inner !== value) {
                            if (inner === 2 && !Array.isArray(value) && Object(value) === (value) && Object.keys(value).length === 0) {
                                return true;
                            }
                            return false;
                        }
                    }
                    return true;
                };
                for (const format of formats) {
                    if (match(tags, format.tags) && (!format.identifier || identifier === context.identifier)) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.context = context.identifier;
                        throw error;
                    }
                }
                const format = (tags) => {
                    const content = Object.entries(tags).map(([key, value]) => {
                        return `${key}:${Object(value) === value ? `{${format(value)}}` : value}`;
                    });
                    return content.join(',');
                };
                const content = format(tags);
                throw new view.Error(`Unsupported Protocol Buffers content '${content.length > 64 ? `${content.substring(0, 100)}...` : content}'.`);
            }
        };
        const flatbuffers = () => {
            const stream = context.stream;
            if (stream && stream.length >= 8) {
                let identifier = null;
                const reader = context.peek('flatbuffers.binary');
                if (reader) {
                    identifier = reader.identifier;
                } else {
                    const data = stream.peek(8);
                    if ((data[0] === 0x08 || data[0] === 0x18 || data[0] === 0x1C || data[0] === 0x20 || data[0] === 0x28) && data[1] === 0x00 && data[2] === 0x00 && data[2] === 0x00) {
                        identifier = String.fromCharCode.apply(null, data.slice(4, 8));
                    }
                }
                if (identifier) {
                    const formats = [
                        { name: 'ONNX Runtime model data', identifier: 'ORTM' },
                        { name: 'TensorFlow Lite model data', identifier: 'TFL3' },
                        { name: 'NNC model data', identifier: 'ENNC' },
                        { name: 'KaNN model data', identifier: 'KaNN' },
                        { name: 'Circle model data', identifier: 'CIR0' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL0' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL1' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL2' },
                        { name: 'MindSpore Lite model data', identifier: 'MSL3' },
                    ];
                    for (const format of formats) {
                        if (identifier === format.identifier) {
                            throw new view.Error(`Invalid file content. File contains ${format.name}.`);
                        }
                    }
                }
            }
        };
        const xml = () => {
            const document = context.peek('xml');
            if (document && document.documentElement) {
                const tags = new Set();
                const qualifiedName = (element) => {
                    const namespaceURI = element.namespaceURI;
                    const localName = element.localName;
                    return namespaceURI ? `${namespaceURI}:${localName}` : localName;
                };
                const root = qualifiedName(document.documentElement);
                tags.add(root);
                for (const element of document.documentElement.childNodes) {
                    const name = qualifiedName(element);
                    tags.add(`${root}/${name}`);
                }
                const formats = [
                    { name: 'OpenCV storage data', tags: ['opencv_storage'] },
                    { name: 'XHTML markup', tags: ['http://www.w3.org/1999/xhtml:html'] },
                    { name: '.NET XML documentation', tags: ['doc', 'doc/assembly'] },
                    { name: '.NET XML documentation', tags: ['doc', 'doc/members'] }
                ];
                for (const format of formats) {
                    if (format.tags.every((tag) => tags.has(tag))) {
                        const error = new view.Error(`Invalid file content. File contains ${format.name}.`);
                        error.content = context.identifier;
                        throw error;
                    }
                }
                throw new view.Error(`Unsupported XML content '${tags.keys().next().value}'.`);
            }
        };
        const hdf5 = () => {
            const obj = context.peek('hdf5');
            if (obj instanceof Error) {
                throw obj;
            }
            if (obj) {
                throw new view.Error(`Invalid file content. File contains HDF5 content.`);
            }
        };
        const unknown = () => {
            if (stream) {
                stream.seek(0);
                const buffer = stream.peek(Math.min(16, stream.length));
                const content = Array.from(buffer).map((c) => (c < 16 ? '0' : '') + c.toString(16)).join('');
                throw new view.Error(`Unsupported file content '${content}'.`);
            }
            throw new view.Error("Unsupported file directory.");
        };
        json();
        pbtxt();
        pb();
        flatbuffers();
        xml();
        hdf5();
        unknown();
    }

    async _require(id) {
        const module = await this._host.require(id);
        if (!module || !module.ModelFactory) {
            throw new view.Error(`Failed to load module '${id}'.`);
        }
        return new module.ModelFactory();
    }

    async _openContext(context) {
        const modules = this._filter(context).filter((module) => module && module.length > 0);
        const errors = [];
        for (const module of modules) {
            /* eslint-disable no-await-in-loop */
            const factory = await this._require(module);
            /* eslint-enable no-await-in-loop */
            factory.match(context);
            if (context.stream && context.stream.position !== 0) {
                throw new view.Error('Invalid stream position.');
            }
            if (context.type) {
                try {
                    /* eslint-disable no-await-in-loop */
                    const model = await factory.open(context);
                    /* eslint-enable no-await-in-loop */
                    if (!model.identifier) {
                        model.identifier = context.identifier;
                    }
                    return model;
                } catch (error) {
                    delete context.type;
                    delete context.target;
                    if (context.stream && context.stream.position !== 0) {
                        context.stream.seek(0);
                    }
                    errors.push(error);
                }
            }
            if (context.stream && context.stream.position !== 0) {
                throw new view.Error('Invalid stream position.');
            }
        }
        if (errors.length > 0) {
            if (errors.length === 1) {
                throw errors[0];
            }
            throw new view.Error(errors.map((err) => err.message).join('\n'));
        }
        return null;
    }

    async _openEntries(entries) {
        try {
            const rootFolder = (files) => {
                const map = files.map((file) => file.split('/').slice(0, -1));
                const at = (index) => (list) => list[index];
                const rotate = (list) => list.length === 0 ? [] : list[0].map((item, index) => list.map(at(index)));
                const equals = (list) => list.every((item) => item === list[0]);
                const folder = rotate(map).filter(equals).map(at(0)).join('/');
                return folder.length === 0 ? folder : `${folder}/`;
            };
            const files = Array.from(entries).filter(([name]) => !(name.endsWith('/') || name.split('/').pop().startsWith('.') || (!name.startsWith('./') && name.startsWith('.'))));
            const folder = rootFolder(files.map(([name]) => name));
            const filter = async (queue, entries) => {
                entries = new Map(Array.from(entries)
                    .filter(([path]) => path.startsWith(folder))
                    .map(([path, stream]) => [path.substring(folder.length), stream]));
                const entryContext = new view.EntryContext(this._host, entries);
                let matches = [];
                for (const [name, stream] of queue) {
                    const identifier = name.substring(folder.length);
                    const context = new view.Context(entryContext, identifier, stream);
                    const modules = this._filter(context);
                    for (const module of modules) {
                        /* eslint-disable no-await-in-loop */
                        const factory = await this._require(module);
                        /* eslint-enable no-await-in-loop */
                        factory.match(context);
                        if (context.stream && context.stream.position !== 0) {
                            throw new view.Error('Invalid stream position.');
                        }
                        delete context.target;
                        if (context.type) {
                            matches = matches.filter((match) => !factory.filter || factory.filter(context, match.type));
                            if (matches.every((match) => !match.factory.filter || match.factory.filter(match, context.type))) {
                                context.factory = factory;
                                matches.push(context);
                            }
                            break;
                        }
                    }
                }
                if (matches.length > 1) {
                    const content = matches.map((context) => context.type).join(',');
                    throw new view.ArchiveError(`Archive contains multiple model files '${content}'.`);
                }
                if (matches.length > 0) {
                    const match = matches.shift();
                    delete match.type;
                    delete match.factory;
                    return match;
                }
                return null;
            };
            const queue = files.filter(([name]) => name.substring(folder.length).indexOf('/') < 0);
            let context = await filter(queue, entries);
            if (!context) {
                const queue = files.filter(([name]) => name.substring(folder.length).indexOf('/') >= 0);
                context = await filter(queue, entries);
            }
            return context;
        } catch (error) {
            throw new view.ArchiveError(error.message);
        }
    }

    accept(identifier, size) {
        const extension = identifier.indexOf('.') === -1 ? '' : identifier.split('.').pop().toLowerCase();
        identifier = identifier.toLowerCase().split('/').pop();
        let accept = false;
        for (const extension of this._patterns) {
            if ((typeof extension === 'string' && identifier.endsWith(extension)) ||
                (extension instanceof RegExp && extension.exec(identifier))) {
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
            (typeof entry.pattern === 'string' && identifier.endsWith(entry.pattern)) ||
            (entry.pattern instanceof RegExp && entry.pattern.test(identifier)));
        return Array.from(new Set(list.map((entry) => entry.module)));
    }

    async _openSignature(context) {
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
                throw new view.Error('File has no content.');
            }
            /* eslint-disable no-control-regex */
            const entries = [
                { name: 'ELF executable', value: /^\x7FELF/ },
                { name: 'PNG image', value: /^\x89PNG/ },
                { name: 'Git LFS header', value: /^version https:\/\/git-lfs.github.com/ },
                { name: 'Git LFS header', value: /^\s*oid sha256:/ },
                { name: 'GGML data', value: /^lmgg|fmgg|tjgg|algg|fugg/ },
                { name: 'HTML markup', value: /^\s*<html(\s+[^>]+)?>/ },
                { name: 'HTML markup', value: /^\s*<!doctype\s*html>/ },
                { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*html>/ },
                { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*HTML>/ },
                { name: 'HTML markup', value: /^\s*<!DOCTYPE\s*HTML\s+(PUBLIC|SYSTEM)?/ },
                { name: 'Unity metadata', value: /^fileFormatVersion:/ },
                { name: 'Python source code', value: /^((#.*(\n|\r\n))|('''.*'''(\n|\r\n))|("""[\s\S]*""")|(\n|\r\n))*(import[ ]+[a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*([ ]+as[ ]+[a-zA-Z]\w*)?[ ]*(,|;|\n|\r\n))/ },
                { name: 'Python source code', value: /^((#.*(\n|\r\n))|('''.*'''(\n|\r\n))|("""[\s\S]*""")|(\n|\r\n))*(from[ ]+([a-zA-Z_]\w*(\.[a-zA-Z_]\w*)*)[ ]+import[ ]+[a-zA-Z]\w*)/ },
                { name: 'Python virtual environment configuration', value: /^home[ ]*=[ ]*/, identifier: 'pyvenv.cfg' },
                { name: 'Bash script', value: /^#!\/usr\/bin\/env\s/ },
                { name: 'Bash script', value: /^#!\/bin\/bash\s/ },
                { name: 'TSD header', value: /^%TSD-Header-###%/ },
                { name: 'AppleDouble data', value: /^\x00\x05\x16\x07/ },
                { name: 'TensorFlow Hub module', value: /^\x08\x03$/, identifier: 'tfhub_module.pb' },
                { name: 'V8 snapshot', value: /^.\x00\x00\x00.\x00\x00\x00/, identifier: 'snapshot_blob.bin' },
                { name: 'V8 context snapshot', value: /^.\x00\x00\x00.\x00\x00\x00/, identifier: 'v8_context_snapshot.bin' },
                { name: 'V8 natives blob', value: /^./, identifier: 'natives_blob.bin' },
                { name: 'ViSQOL model', value: /^svm_type\s/ },
                { name: 'SenseTime model', value: /^STEF/ },
                { name: 'AES Crypt data', value: /^AES[\x01|\x02]\x00/ },
                { name: 'BModel data', value: /^\xEE\xAA\x55\xFF/ }, // https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Builder/BM168x/bmodel.fbs
                { name: 'CviModel data', value: /^CviModel/ }, // https://github.com/sophgo/tpu-mlir/blob/master/include/tpu_mlir/Builder/CV18xx/proto/cvimodel.fbs
                { name: 'Tokenizer data', value: /^IQ== 0\n/ },
                { name: 'BCNN model', value: /^BCNN/ },
                { name: 'base64 data', value: /^gAAAAAB/ },
                { name: 'Mathematica Notebook data', value: /^\(\*\sContent-type:\sapplication\/vnd\.wolfram\.mathematica\s\*\)/ }
            ];
            /* eslint-enable no-control-regex */
            const buffer = stream.peek(Math.min(4096, stream.length));
            const content = String.fromCharCode.apply(null, buffer);
            for (const entry of entries) {
                if (content.match(entry.value) && (!entry.identifier || entry.identifier === context.identifier)) {
                    throw new view.Error(`Invalid file content. File contains ${entry.name}.`);
                }
            }
        }
    }

    async import() {
        if (this._host.type === 'Browser' || this._host.type === 'Python') {
            const files = [
                './server', './onnx', './pytorch', './tflite', './mlnet',
                './onnx-proto', './onnx-schema', './tflite-schema',
                'onnx-metadata.json', 'pytorch-metadata.json', 'tflite-metadata.json'
            ];
            for (const file of files) {
                /* eslint-disable no-await-in-loop */
                try {
                    if (file.startsWith('./')) {
                        await this._host.require(file);
                    } else if (file.endsWith('.json')) {
                        await this._host.request(file, 'utf-8', null);
                    }
                } catch {
                    // continue regardless of error
                }
                /* eslint-enable no-await-in-loop */
            }
        }
    }
};

view.Metadata = class {

    static async open(context, name) {
        view.Metadata._metadata = view.Metadata._metadata || new Map();
        const metadata = view.Metadata._metadata;
        if (!metadata.has(name)) {
            try {
                const content = await context.request(name);
                const types = JSON.parse(content);
                metadata.set(name, new view.Metadata(types));
            } catch {
                metadata.set(name, new view.Metadata(null));
            }
        }
        return metadata.get(name);
    }

    constructor(types) {
        this._types = new Map();
        this._attributes = new Map();
        this._inputs = new Map();
        if (Array.isArray(types)) {
            for (const type of types) {
                this._types.set(type.name, type);
                if (type.identifier !== undefined) {
                    this._types.set(type.identifier, type);
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
        const key = `${type}:${name}`;
        if (!this._attributes.has(key)) {
            this._attributes.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.attributes)) {
                for (const attribute of metadata.attributes) {
                    this._attributes.set(`${type}:${attribute.name}`, attribute);
                }
            }
        }
        return this._attributes.get(key);
    }

    input(type, name) {
        const key = `${type}:${name}`;
        if (!this._inputs.has(key)) {
            this._inputs.set(key, null);
            const metadata = this.type(type);
            if (metadata && Array.isArray(metadata.inputs)) {
                for (const input of metadata.inputs) {
                    this._inputs.set(`${type}:${input.name}`, input);
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

if (typeof window !== 'undefined' && window.exports) {
    window.exports.view = view;
}

export const View = view.View;
export const ModelFactoryService = view.ModelFactoryService;
export const ModelSidebar = view.ModelSidebar;
export const NodeSidebar = view.NodeSidebar;
export const TensorSidebar = view.TensorSidebar;
export const Documentation = view.Documentation;
export const Formatter = view.Formatter;
export const Tensor = view.Tensor;
export const Quantization = view.Quantization;
