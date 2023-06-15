
var grapher = {};
var dagre = require('./dagre');

grapher.Graph = class {

    constructor(compound, options) {
        this._isCompound = compound;
        this._options = options;
        this._nodes = new Map();
        this._edges = new Map();
        this._children = {};
        this._children['\x00'] = {};
        this._parent = {};
    }

    get options() {
        return this._options;
    }

    setNode(node) {
        const key = node.name;
        const value = this._nodes.get(key);
        if (value) {
            value.label = node;
        } else {
            this._nodes.set(key, { v: key, label: node });
            if (this._isCompound) {
                this._parent[key] = '\x00';
                this._children[key] = {};
                this._children['\x00'][key] = true;
            }
        }
    }

    setEdge(edge) {
        if (!this._nodes.has(edge.v)) {
            throw new grapher.Error("Invalid edge '" + JSON.stringify(edge.v) + "'.");
        }
        if (!this._nodes.has(edge.w)) {
            throw new grapher.Error("Invalid edge '" + JSON.stringify(edge.w) + "'.");
        }
        const key = edge.v + ':' + edge.w;
        if (!this._edges.has(key)) {
            this._edges.set(key, { v: edge.v, w: edge.w, label: edge });
        }
    }

    setParent(node, parent) {
        if (!this._isCompound) {
            throw new Error("Cannot set parent in a non-compound graph");
        }
        parent += "";
        for (let ancestor = parent; ancestor; ancestor = this.parent(ancestor)) {
            if (ancestor === node) {
                throw new Error("Setting " + parent + " as parent of " + node + " would create a cycle");
            }
        }
        delete this._children[this._parent[node]][node];
        this._parent[node] = parent;
        this._children[parent][node] = true;
        return this;
    }

    get nodes() {
        return this._nodes;
    }

    hasNode(key) {
        return this._nodes.has(key);
    }

    node(key) {
        return this._nodes.get(key);
    }

    get edges() {
        return this._edges;
    }

    parent(key) {
        if (this._isCompound) {
            const parent = this._parent[key];
            if (parent !== '\x00') {
                return parent;
            }
        }
        return null;
    }

    children(key) {
        key = key === undefined ? '\x00' : key;
        if (this._isCompound) {
            const children = this._children[key];
            if (children) {
                return Object.keys(children);
            }
        } else if (key === '\x00') {
            return this.nodes.keys();
        } else if (this.hasNode(key)) {
            return [];
        }
        return null;
    }

    build(document, origin) {
        const createGroup = (name) => {
            const element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
            element.setAttribute('id', name);
            element.setAttribute('class', name);
            origin.appendChild(element);
            return element;
        };

        const clusterGroup = createGroup('clusters');
        const edgePathGroup = createGroup('edge-paths');
        const edgeLabelGroup = createGroup('edge-labels');
        const nodeGroup = createGroup('nodes');

        const edgePathGroupDefs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        edgePathGroup.appendChild(edgePathGroupDefs);
        const marker = (id) => {
            const element = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
            element.setAttribute('id', id);
            element.setAttribute('viewBox', '0 0 10 10');
            element.setAttribute('refX', 9);
            element.setAttribute('refY', 5);
            element.setAttribute('markerUnits', 'strokeWidth');
            element.setAttribute('markerWidth', 8);
            element.setAttribute('markerHeight', 6);
            element.setAttribute('orient', 'auto');
            const markerPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
            markerPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z');
            markerPath.style.setProperty('stroke-width', 1);
            element.appendChild(markerPath);
            return element;
        };
        edgePathGroupDefs.appendChild(marker("arrowhead"));
        edgePathGroupDefs.appendChild(marker("arrowhead-select"));
        edgePathGroupDefs.appendChild(marker("arrowhead-hover"));

        for (const nodeId of this.nodes.keys()) {
            const node = this.node(nodeId);
            if (this.children(nodeId).length == 0) {
                // node
                node.label.build(document, nodeGroup);
            } else {
                // cluster
                node.label.rectangle = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
                if (node.label.rx) {
                    node.label.rectangle.setAttribute('rx', node.rx);
                }
                if (node.label.ry) {
                    node.label.rectangle.setAttribute('ry', node.ry);
                }
                node.label.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
                node.label.element.setAttribute('class', 'cluster');
                node.label.element.appendChild(node.label.rectangle);
                clusterGroup.appendChild(node.label.element);
            }
        }

        for (const edge of this.edges.values()) {
            edge.label.build(document, edgePathGroup, edgeLabelGroup);
        }
    }

    update() {
        dagre.layout(this);
        for (const nodeId of this.nodes.keys()) {
            const node = this.node(nodeId);
            if (this.children(nodeId).length == 0) {
                // node
                node.label.update();
            } else {
                // cluster
                const node = this.node(nodeId);
                node.label.element.setAttribute('transform', 'translate(' + node.label.x + ',' + node.label.y + ')');
                node.label.rectangle.setAttribute('x', - node.label.width / 2);
                node.label.rectangle.setAttribute('y', - node.label.height / 2);
                node.label.rectangle.setAttribute('width', node.label.width);
                node.label.rectangle.setAttribute('height', node.label.height);
            }
        }
        for (const edge of this.edges.values()) {
            edge.label.update();
        }
    }
};

grapher.Node = class {

    constructor() {
        this._blocks = [];
    }

    header() {
        const block = new grapher.Node.Header();
        this._blocks.push(block);
        return block;
    }

    list() {
        const block = new grapher.Node.List();
        this._blocks.push(block);
        return block;
    }

    canvas() {
        const block = new grapher.Node.Canvas();
        this._blocks.push(block);
        return block;
    }

    build(document, parent) {
        this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        if (this.id) {
            this.element.setAttribute('id', this.id);
        }
        this.element.setAttribute('class', this.class ? 'node ' + this.class : 'node');
        this.element.style.opacity = 0;
        parent.appendChild(this.element);
        this.border = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.border.setAttribute('class', [ 'node', 'border' ].join(' '));
        for (let i = 0; i < this._blocks.length; i++) {
            const block = this._blocks[i];
            block.first = i === 0;
            block.last = i === this._blocks.length - 1;
            block.build(document, this.element);
        }
        this.element.appendChild(this.border);
        this.layout();
    }

    layout() {
        const width = Math.max(...this._blocks.map((block) => block.width));
        let height = 0;
        for (let i = 0; i < this._blocks.length; i++) {
            const block = this._blocks[i];
            block.y = height;
            block.update(this.element, height, width, i == 0, i == this._blocks.length - 1);
            height = height + block.height;
        }
        this.border.setAttribute('d', grapher.Node.roundedRect(0, 0, width, height, true, true, true, true));
        const nodeBox = this.element.getBBox();
        this.width = nodeBox.width;
        this.height = nodeBox.height;
    }

    update() {
        this.element.setAttribute('transform', 'translate(' + (this.x - (this.width / 2)) + ',' + (this.y - (this.height / 2)) + ')');
        this.element.style.opacity = 1;
    }

    select() {
        this.element.classList.add('select');
        return [ this.element ];
    }

    deselect() {
        this.element.classList.remove('select');
    }

    static roundedRect(x, y, width, height, r1, r2, r3, r4) {
        const radius = 5;
        r1 = r1 ? radius : 0;
        r2 = r2 ? radius : 0;
        r3 = r3 ? radius : 0;
        r4 = r4 ? radius : 0;
        return "M" + (x + r1) + "," + y +
            "h" + (width - r1 - r2) +
            "a" + r2 + "," + r2 + " 0 0 1 " + r2 + "," + r2 +
            "v" + (height - r2 - r3) +
            "a" + r3 + "," + r3 + " 0 0 1 " + -r3 + "," + r3 +
            "h" + (r3 + r4 - width) +
            "a" + r4 + "," + r4 + " 0 0 1 " + -r4 + "," + -r4 +
            'v' + (-height + r4 + r1) +
            "a" + r1 + "," + r1 + " 0 0 1 " + r1 + "," + -r1 +
            "z";
    }
};

grapher.Node.Header = class {

    constructor() {
        this._entries = [];
    }

    add(id, classList, content, tooltip, handler) {
        const entry = new grapher.Node.Header.Entry(id, classList, content, tooltip, handler);
        this._entries.push(entry);
        return entry;
    }

    build(document, parent) {
        this._document = document;
        this.width = 0;
        this.height = 0;
        let x = 0;
        const y = 0;
        for (const entry of this._entries) {
            entry.x = x;
            entry.y = y;
            entry.build(document, parent);
            x += entry.width;
            this.height = Math.max(entry.height, this.height);
            this.width = Math.max(x, this.width);
        }
        if (!this.first) {
            this.line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            parent.appendChild(this.line);
        }
    }

    update(parent, top, width) {
        const document = this._document;
        const dx = width - this.width;
        for (let i = 0; i < this._entries.length; i++) {
            const entry = this._entries[i];
            if (i == 0) {
                entry.width = entry.width + dx;
            } else {
                entry.x = entry.x + dx;
                entry.tx = entry.tx + dx;
            }
            entry.y = entry.y + top;
        }
        for (let i = 0; i < this._entries.length; i++) {
            const entry = this._entries[i];
            entry.element.setAttribute('transform', 'translate(' + entry.x + ',' + entry.y + ')');
            const r1 = i == 0 && this.first;
            const r2 = i == this._entries.length - 1 && this.first;
            const r3 = i == this._entries.length - 1 && this.last;
            const r4 = i == 0 && this.last;
            entry.path.setAttribute('d', grapher.Node.roundedRect(0, 0, entry.width, entry.height, r1, r2, r3, r4));
            entry.text.setAttribute('x', 6);
            entry.text.setAttribute('y', entry.ty);
        }
        for (let i = 0; i < this._entries.length; i++) {
            const entry = this._entries[i];
            if (i != 0) {
                const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
                line.setAttribute('class', 'node');
                line.setAttribute('x1', entry.x);
                line.setAttribute('x2', entry.x);
                line.setAttribute('y1', top);
                line.setAttribute('y2', top + this.height);
                parent.appendChild(line);
            }
        }
        if (this.line) {
            this.line.setAttribute('class', 'node');
            this.line.setAttribute('x1', 0);
            this.line.setAttribute('x2', width);
            this.line.setAttribute('y1', top);
            this.line.setAttribute('y2', top);
        }
    }
};

grapher.Node.Header.Entry = class {

    constructor(id, classList, content, tooltip, handler) {
        this.id = id;
        this.classList = classList;
        this.content = content;
        this.tooltip = tooltip;
        this.handler = handler;
        this._events = {};
    }

    on(event, callback) {
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

    build(document, parent) {
        const yPadding = 4;
        const xPadding = 7;
        this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        parent.appendChild(this.element);
        this.path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        this.element.appendChild(this.path);
        this.element.appendChild(this.text);
        const classList = [ 'node-item' ];
        if (this.classList) {
            classList.push(...this.classList);
        }
        this.element.setAttribute('class', classList.join(' '));
        if (this.id) {
            this.element.setAttribute('id', this.id);
        }
        if (this._events.click) {
            this.element.addEventListener('click', () => this.emit('click'));
        }
        if (this.tooltip) {
            const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
            title.textContent = this.tooltip;
            this.element.appendChild(title);
        }
        if (this.content) {
            this.text.textContent = this.content;
        }
        const boundingBox = this.text.getBBox();
        this.width = boundingBox.width + xPadding + xPadding;
        this.height = boundingBox.height + yPadding + yPadding;
        this.tx = xPadding;
        this.ty = yPadding - boundingBox.y;
    }
};

grapher.Node.List = class {

    constructor() {
        this._items = [];
        this._events = {};
    }

    add(id, name, value, tooltip, separator) {
        const item = new grapher.Node.List.Item(id, name, value, tooltip, separator);
        this._items.push(item);
        return item;
    }

    on(event, callback) {
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

    build(document, parent) {
        this._document = document;
        this.width = 0;
        this.height = 0;
        const x = 0;
        const y = 0;
        this.element = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        this.element.setAttribute('class', 'node-attribute');
        if (this._events.click) {
            this.element.addEventListener('click', () => this.emit('click'));
        }
        this.element.setAttribute('transform', 'translate(' + x + ',' + y + ')');
        this.background = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        this.element.appendChild(this.background);
        parent.appendChild(this.element);
        this.height += 3;
        for (const item of this._items) {
            const yPadding = 1;
            const xPadding = 6;
            const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            if (item.id) {
                text.setAttribute('id', item.id);
            }
            text.setAttribute('xml:space', 'preserve');
            this.element.appendChild(text);
            if (item.tooltip) {
                const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
                title.textContent = item.tooltip;
                text.appendChild(title);
            }
            const name = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
            name.textContent = item.name;
            if (item.separator.trim() != '=') {
                name.style.fontWeight = 'bold';
            }
            text.appendChild(name);
            const textValueElement = document.createElementNS('http://www.w3.org/2000/svg', 'tspan');
            textValueElement.textContent = item.separator + item.value;
            text.appendChild(textValueElement);
            const size = text.getBBox();
            const width = xPadding + size.width + xPadding;
            this.width = Math.max(width, this.width);
            text.setAttribute('x', x + xPadding);
            text.setAttribute('y', this.height + yPadding - size.y);
            this.height += yPadding + size.height + yPadding;
        }
        this.height += 3;
        this.width = Math.max(75, this.width);
        if (!this.first) {
            this.line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
            this.line.setAttribute('class', 'node');
            this.element.appendChild(this.line);
        }
    }

    update(parent, top, width) {
        this.element.setAttribute('transform', 'translate(0,' + this.y + ')');
        this.background.setAttribute('d', grapher.Node.roundedRect(0, 0, width, this.height, this.first, this.first, this.last, this.last));
        if (this.line) {
            this.line.setAttribute('x1', 0);
            this.line.setAttribute('x2', width);
            this.line.setAttribute('y1', 0);
            this.line.setAttribute('y2', 0);
        }
    }
};

grapher.Node.List.Item = class {

    constructor(id, name, value, tooltip, separator) {
        this.id = id;
        this.name = name;
        this.value = value;
        this.tooltip = tooltip;
        this.separator = separator;
    }
};

grapher.Node.Canvas = class {

    constructor() {
        this.width = 0;
        this.height = 0;
    }

    build(/* document, parent */) {
    }

    update(/* parent, top, width , first, last */) {
    }
};

grapher.Edge = class {

    constructor(from, to) {
        this.from = from;
        this.to = to;
    }

    build(document, edgePathGroupElement, edgeLabelGroupElement) {
        const createElement = (name) => {
            return document.createElementNS('http://www.w3.org/2000/svg', name);
        };
        this.element = createElement('path');
        if (this.id) {
            this.element.setAttribute('id', this.id);
        }
        this.element.setAttribute('class', this.class ? 'edge-path ' + this.class : 'edge-path');
        edgePathGroupElement.appendChild(this.element);
        this.hitTest = createElement('path');
        this.hitTest.setAttribute('class', 'edge-path-hit-test');
        this.hitTest.addEventListener('pointerover', () => this.emit('pointerover'));
        this.hitTest.addEventListener('pointerleave', () => this.emit('pointerleave'));
        edgePathGroupElement.appendChild(this.hitTest);
        if (this.label) {
            const tspan = createElement('tspan');
            tspan.setAttribute('xml:space', 'preserve');
            tspan.setAttribute('dy', '1em');
            tspan.setAttribute('x', '1');
            tspan.appendChild(document.createTextNode(this.label));
            this.labelElement = createElement('text');
            this.labelElement.appendChild(tspan);
            this.labelElement.style.opacity = 0;
            this.labelElement.setAttribute('class', 'edge-label');
            if (this.id) {
                this.labelElement.setAttribute('id', 'edge-label-' + this.id);
            }
            edgeLabelGroupElement.appendChild(this.labelElement);
            const edgeBox = this.labelElement.getBBox();
            this.width = edgeBox.width;
            this.height = edgeBox.height;
        }
    }

    update() {
        const intersectRect = (node, point) => {
            const x = node.x;
            const y = node.y;
            const dx = point.x - x;
            const dy = point.y - y;
            let h = node.height / 2;
            let w = node.width / 2;
            if (Math.abs(dy) * w > Math.abs(dx) * h) {
                if (dy < 0) {
                    h = -h;
                }
                return { x: x + (dy === 0 ? 0 : h * dx / dy), y: y + h };
            }
            if (dx < 0) {
                w = -w;
            }
            return { x: x + w, y: y + (dx === 0 ? 0 : w * dy / dx) };
        };
        const curvePath = (edge, tail, head) => {
            const points = edge.points.slice(1, edge.points.length - 1);
            points.unshift(intersectRect(tail, points[0]));
            points.push(intersectRect(head, points[points.length - 1]));
            return new grapher.Edge.Curve(points).path.data;
        };
        const edgePath = curvePath(this, this.from, this.to);
        this.element.setAttribute('d', edgePath);
        this.hitTest.setAttribute('d', edgePath);
        if (this.labelElement) {
            this.labelElement.setAttribute('transform', 'translate(' + (this.x - (this.width / 2)) + ',' + (this.y - (this.height / 2)) + ')');
            this.labelElement.style.opacity = 1;
        }
    }

    select() {
        if (this.element) {
            if (!this.element.classList.contains('select')) {
                const path = this.element;
                path.classList.add('select');
                this.element = path.cloneNode(true);
                path.parentNode.replaceChild(this.element, path);
            }
            return [ this.element ];
        }
        return [];
    }

    deselect() {
        if (this.element && this.element.classList.contains('select')) {
            const path = this.element;
            path.classList.remove('select');
            this.element = path.cloneNode(true);
            path.parentNode.replaceChild(this.element, path);
        }
    }
};

grapher.Edge.Curve = class {

    constructor(points) {
        this._path = new grapher.Edge.Path();
        this._x0 = NaN;
        this._x1 = NaN;
        this._y0 = NaN;
        this._y1 = NaN;
        this._state = 0;
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            this.point(point.x, point.y);
            if (i === points.length - 1) {
                switch (this._state) {
                    case 3:
                        this.curve(this._x1, this._y1);
                        this._path.lineTo(this._x1, this._y1);
                        break;
                    case 2:
                        this._path.lineTo(this._x1, this._y1);
                        break;
                    default:
                        break;
                }
                if (this._line || (this._line !== 0 && this._point === 1)) {
                    this._path.closePath();
                }
                this._line = 1 - this._line;
            }
        }
    }

    get path() {
        return this._path;
    }

    point(x, y) {
        x = +x;
        y = +y;
        switch (this._state) {
            case 0:
                this._state = 1;
                if (this._line) {
                    this._path.lineTo(x, y);
                } else {
                    this._path.moveTo(x, y);
                }
                break;
            case 1:
                this._state = 2;
                break;
            case 2:
                this._state = 3;
                this._path.lineTo((5 * this._x0 + this._x1) / 6, (5 * this._y0 + this._y1) / 6);
                this.curve(x, y);
                break;
            default:
                this.curve(x, y);
                break;
        }
        this._x0 = this._x1;
        this._x1 = x;
        this._y0 = this._y1;
        this._y1 = y;
    }

    curve(x, y) {
        this._path.bezierCurveTo(
            (2 * this._x0 + this._x1) / 3,
            (2 * this._y0 + this._y1) / 3,
            (this._x0 + 2 * this._x1) / 3,
            (this._y0 + 2 * this._y1) / 3,
            (this._x0 + 4 * this._x1 + x) / 6,
            (this._y0 + 4 * this._y1 + y) / 6
        );
    }
};

grapher.Edge.Path = class {

    constructor() {
        this._x0 = null;
        this._y0 = null;
        this._x1 = null;
        this._y1 = null;
        this._data = '';
    }

    moveTo(x, y) {
        this._data += "M" + (this._x0 = this._x1 = +x) + "," + (this._y0 = this._y1 = +y);
    }

    lineTo(x, y) {
        this._data += "L" + (this._x1 = +x) + "," + (this._y1 = +y);
    }

    bezierCurveTo(x1, y1, x2, y2, x, y) {
        this._data += "C" + (+x1) + "," + (+y1) + "," + (+x2) + "," + (+y2) + "," + (this._x1 = +x) + "," + (this._y1 = +y);
    }

    closePath() {
        if (this._x1 !== null) {
            this._x1 = this._x0;
            this._y1 = this._y0;
            this._data += "Z";
        }
    }

    get data() {
        return this._data;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Graph = grapher.Graph;
    module.exports.Node = grapher.Node;
    module.exports.Edge = grapher.Edge;
}