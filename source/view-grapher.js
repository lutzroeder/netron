/* jshint esversion: 6 */

var grapher = grapher || {};
var dagre = dagre || require('dagre');

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

    graph() {
        return this._options;
    }

    setNode(node) {
        this._nodes.set(node.name, node);
        if (this._isCompound) {
            this._parent[node.name] = '\x00';
            this._children[node.name] = {};
            this._children['\x00'][node.name] = true;
        }
        return this;
    }

    setEdge(edge) {
        if (!this._nodes.has(edge.v)) {
            throw new grapher.Error();
        }
        if (!this._nodes.has(edge.w)) {
            throw new grapher.Error();
        }
        const key = edge.v + ' ' + edge.w + ' ';
        if (!this._edges.has(key)) {
            this._edges.set(key, edge);
        }
        return this;
    }

    setParent(node, parent) {
        if (!this._isCompound) {
            throw new Error("Cannot set parent in a non-compound graph");
        }
        parent += "";
        for (var ancestor = parent; ancestor; ancestor = this.parent(ancestor)) {
            if (ancestor === node) {
                throw new Error("Setting " + parent + " as parent of " + node + " would create a cycle");
            }
        }
        delete this._children[this._parent[node]][node];
        this._parent[node] = parent;
        this._children[parent][node] = true;
        return this;
    }

    nodes() {
        return Array.from(this._nodes.keys());
    }

    hasNode(key) {
        return this._nodes.has(key);
    }

    node(key) {
        return this._nodes.get(key);
    }

    edges() {
        return Array.from(this._edges.values());
    }

    edge(key) {
        return key;
    }

    parent(key) {
        if (this._isCompound) {
            var parent = this._parent[key];
            if (parent !== '\x00') {
                return parent;
            }
        }
    }

    children(key) {
        key = key === undefined ? '\x00' : key;
        if (this._isCompound) {
            const children = this._children[key];
            if (children) {
                return Object.keys(children);
            }
        }
        else if (key === '\x00') {
            return this.nodes();
        }
        else if (this.hasNode(key)) {
            return [];
        }
    }

    build(document, originElement) {

        const createElement = (name) => {
            return document.createElementNS('http://www.w3.org/2000/svg', name);
        };
        const createGroup = (name) => {
            const element = createElement('g');
            element.setAttribute('id', name);
            element.setAttribute('class', name);
            originElement.appendChild(element);
            return element;
        };

        const clusterGroup = createGroup('clusters');
        const edgePathGroup = createGroup('edge-paths');
        const edgeLabelGroup = createGroup('edge-labels');
        const nodeGroup = createGroup('nodes');

        const edgePathGroupDefs = createElement('defs');
        edgePathGroup.appendChild(edgePathGroupDefs);
        const marker = (id) => {
            const element = createElement('marker');
            element.setAttribute('id', id);
            element.setAttribute('viewBox', '0 0 10 10');
            element.setAttribute('refX', 9);
            element.setAttribute('refY', 5);
            element.setAttribute('markerUnits', 'strokeWidth');
            element.setAttribute('markerWidth', 8);
            element.setAttribute('markerHeight', 6);
            element.setAttribute('orient', 'auto');
            const markerPath = createElement('path');
            markerPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z');
            markerPath.style.setProperty('stroke-width', 1);
            element.appendChild(markerPath);
            return element;
        };
        edgePathGroupDefs.appendChild(marker("arrowhead-vee"));
        edgePathGroupDefs.appendChild(marker("arrowhead-vee-select"));

        for (const nodeId of this.nodes()) {
            const node = this.node(nodeId);
            if (this.children(nodeId).length == 0) {
                // node
                node.build(document, nodeGroup);
            }
            else {
                // cluster
                node.rectangle = createElement('rect');
                if (node.rx) {
                    node.rectangle.setAttribute('rx', node.rx);
                }
                if (node.ry) {
                    node.rectangle.setAttribute('ry', node.ry);
                }
                node.element = createElement('g');
                node.element.setAttribute('class', 'cluster');
                node.element.appendChild(node.rectangle);
                clusterGroup.appendChild(node.element);
            }
        }

        for (const edgeId of this.edges()) {
            const edge = this.edge(edgeId);
            edge.build(document, edgePathGroup, edgeLabelGroup);
        }
    }

    layout() {

        dagre.layout(this);

        for (const nodeId of this.nodes()) {
            const node = this.node(nodeId);
            if (this.children(nodeId).length == 0) {
                // node
                node.layout();
            }
            else {
                // cluster
                const node = this.node(nodeId);
                node.element.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
                node.rectangle.setAttribute('x', - node.width / 2);
                node.rectangle.setAttribute('y', - node.height / 2 );
                node.rectangle.setAttribute('width', node.width);
                node.rectangle.setAttribute('height', node.height);
            }
        }

        for (const edgeId of this.edges()) {
            const edge = this.edge(edgeId);
            edge.layout();
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

    build(document, contextElement) {
        const createElement = (name) => {
            return document.createElementNS('http://www.w3.org/2000/svg', name);
        };
        this.label = createElement('g');

        this.element = createElement('g');
        if (this.id) {
            this.element.setAttribute('id', this.id);
        }
        this.element.setAttribute('class', this.class ? 'node ' + this.class : 'node');
        this.element.style.opacity = 0;
        this.element.appendChild(this.label);

        contextElement.appendChild(this.element);

        // contextElement.appendChild(this.label);

        let width = 0;
        let height = 0;
        const tops = [];

        for (const block of this._blocks) {
            tops.push(height);
            block.build(document, this.label);
            if (width < block.width) {
                width = block.width;
            }
            height = height + block.height;
        }

        for (let i = 0; i < this._blocks.length; i++) {
            const top = tops.shift();
            this._blocks[i].update(this.label, top, width, i == 0, i == this._blocks.length - 1);
        }

        const borderElement = createElement('path');
        borderElement.setAttribute('class', [ 'node', 'border' ].join(' '));
        borderElement.setAttribute('d', grapher.Node.roundedRect(0, 0, width, height, true, true, true, true));
        this.label.appendChild(borderElement);

        const nodeBox = this.label.getBBox();
        const nodeX = - nodeBox.width / 2;
        const nodeY = - nodeBox.height / 2;
        this.width = nodeBox.width;
        this.height = nodeBox.height;
        this.label.setAttribute('transform', 'translate(' + nodeX + ',' + nodeY + ')');
    }

    layout() {
        this.element.setAttribute('transform', 'translate(' + this.x + ',' + this.y + ')');
        this.element.style.opacity = 1;
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
        this._items = [];
    }

    add(id, classList, content, tooltip, handler) {
        this._items.push({
            id: id,
            classList: classList,
            content: content,
            tooltip: tooltip,
            handler: handler
        });
    }

    build(document, parentElement) {
        this._document = document;
        this._width = 0;
        this._height = 0;
        this._elements = [];
        let x = 0;
        const y = 0;
        for (const item of this._items) {
            const yPadding = 4;
            const xPadding = 7;
            const element = this.createElement('g');
            const classList = [ 'node-item' ];
            parentElement.appendChild(element);
            const pathElement = this.createElement('path');
            const textElement = this.createElement('text');
            element.appendChild(pathElement);
            element.appendChild(textElement);
            if (item.classList) {
                classList.push(...item.classList);
            }
            element.setAttribute('class', classList.join(' '));
            if (item.id) {
                element.setAttribute('id', item.id);
            }
            if (item.handler) {
                element.addEventListener('click', item.handler);
            }
            if (item.tooltip) {
                const titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                element.appendChild(titleElement);
            }
            if (item.content) {
                textElement.textContent = item.content;
            }
            const boundingBox = textElement.getBBox();
            const width = boundingBox.width + xPadding + xPadding;
            const height = boundingBox.height + yPadding + yPadding;
            this._elements.push({
                'group': element,
                'text': textElement,
                'path': pathElement,
                'x': x, 'y': y,
                'width': width, 'height': height,
                'tx': xPadding, 'ty': yPadding - boundingBox.y,
            });
            x += width;
            if (this._height < height) {
                this._height = height;
            }
            if (x > this._width) {
                this._width = x;
            }
        }
    }

    get width() {
        return this._width;
    }

    get height() {
        return this._height;
    }

    update(parentElement, top, width, first, last) {

        const dx = width - this._width;
        let i;
        let element;

        for (i = 0; i < this._elements.length; i++) {
            element = this._elements[i];
            if (i == 0) {
                element.width = element.width + dx;
            }
            else {
                element.x = element.x + dx;
                element.tx = element.tx + dx;
            }
            element.y = element.y + top;
        }

        for (i = 0; i < this._elements.length; i++) {
            element = this._elements[i];
            element.group.setAttribute('transform', 'translate(' + element.x + ',' + element.y + ')');
            const r1 = i == 0 && first;
            const r2 = i == this._elements.length - 1 && first;
            const r3 = i == this._elements.length - 1 && last;
            const r4 = i == 0 && last;
            element.path.setAttribute('d', grapher.Node.roundedRect(0, 0, element.width, element.height, r1, r2, r3, r4));
            element.text.setAttribute('x', 6);
            element.text.setAttribute('y', element.ty);
        }

        let lineElement;
        for (i = 0; i < this._elements.length; i++) {
            element = this._elements[i];
            if (i != 0) {
                lineElement = this.createElement('line');
                lineElement.setAttribute('class', 'node');
                lineElement.setAttribute('x1', element.x);
                lineElement.setAttribute('x2', element.x);
                lineElement.setAttribute('y1', top);
                lineElement.setAttribute('y2', top + this._height);
                parentElement.appendChild(lineElement);
            }
        }

        if (!first) {
            lineElement = this.createElement('line');
            lineElement.setAttribute('class', 'node');
            lineElement.setAttribute('x1', 0);
            lineElement.setAttribute('x2', width);
            lineElement.setAttribute('y1', top);
            lineElement.setAttribute('y2', top);
            parentElement.appendChild(lineElement);
        }
    }

    createElement(name) {
        return this._document.createElementNS('http://www.w3.org/2000/svg', name);
    }
};

grapher.Node.List = class {

    constructor() {
        this._items = [];
    }

    add(id, name, value, tooltip, separator) {
        this._items.push({ id: id, name: name, value: value, tooltip: tooltip, separator: separator });
    }

    get handler() {
        return this._handler;
    }

    set handler(handler) {
        this._handler = handler;
    }

    build(document, parentElement) {
        this._document = document;
        this._width = 0;
        this._height = 0;
        const x = 0;
        const y = 0;
        this._element = this.createElement('g');
        this._element.setAttribute('class', 'node-attribute');
        parentElement.appendChild(this._element);
        if (this._handler) {
            this._element.addEventListener('click', this._handler);
        }
        this._backgroundElement = this.createElement('path');
        this._element.appendChild(this._backgroundElement);
        this._element.setAttribute('transform', 'translate(' + x + ',' + y + ')');
        this._height += 3;
        for (const item of this._items) {
            const yPadding = 1;
            const xPadding = 6;
            const textElement = this.createElement('text');
            if (item.id) {
                textElement.setAttribute('id', item.id);
            }
            textElement.setAttribute('xml:space', 'preserve');
            this._element.appendChild(textElement);
            if (item.tooltip) {
                const titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                textElement.appendChild(titleElement);
            }
            const textNameElement = this.createElement('tspan');
            textNameElement.textContent = item.name;
            if (item.separator.trim() != '=') {
                textNameElement.style.fontWeight = 'bold';
            }
            textElement.appendChild(textNameElement);
            const textValueElement = this.createElement('tspan');
            textValueElement.textContent = item.separator + item.value;
            textElement.appendChild(textValueElement);
            const size = textElement.getBBox();
            const width = xPadding + size.width + xPadding;
            if (this._width < width) {
                this._width = width;
            }
            textElement.setAttribute('x', x + xPadding);
            textElement.setAttribute('y', this._height + yPadding - size.y);
            this._height += yPadding + size.height + yPadding;
        }
        this._height += 3;

        if (this._width < 100) {
            this._width = 100;
        }
    }

    get width() {
        return this._width;
    }

    get height() {
        return this._height;
    }

    update(parentElement, top, width , first, last) {

        this._element.setAttribute('transform', 'translate(0,' + top + ')');

        const r1 = first;
        const r2 = first;
        const r3 = last;
        const r4 = last;
        this._backgroundElement.setAttribute('d', grapher.Node.roundedRect(0, 0, width, this._height, r1, r2, r3, r4));

        if (!first) {
            const lineElement = this.createElement('line');
            lineElement.setAttribute('class', 'node');
            lineElement.setAttribute('x1', 0);
            lineElement.setAttribute('x2', width);
            lineElement.setAttribute('y1', 0);
            lineElement.setAttribute('y2', 0);
            this._element.appendChild(lineElement);
        }
    }

    createElement(name) {
        return this._document.createElementNS('http://www.w3.org/2000/svg', name);
    }
};

class Path {

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
}

class Curve {

    constructor(context) {
        this._context = context;
    }

    lineStart() {
        this._x0 = NaN;
        this._x1 = NaN;
        this._y0 = NaN;
        this._y1 = NaN;
        this._point = 0;
    }

    lineEnd() {
        switch (this._point) {
            case 3:
                this.curve(this._x1, this._y1);
                this._context.lineTo(this._x1, this._y1);
                break;
            case 2:
                this._context.lineTo(this._x1, this._y1);
                break;
        }
        if (this._line || (this._line !== 0 && this._point === 1)) {
            this._context.closePath();
        }
        this._line = 1 - this._line;
    }

    point(x, y) {
        x = +x;
        y = +y;
        switch (this._point) {
            case 0:
                this._point = 1;
                if (this._line) {
                    this._context.lineTo(x, y);
                }
                else {
                    this._context.moveTo(x, y);
                }
                break;
            case 1:
                this._point = 2;
                break;
            case 2:
                this._point = 3;
                this._context.lineTo((5 * this._x0 + this._x1) / 6, (5 * this._y0 + this._y1) / 6);
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
        this._context.bezierCurveTo(
            (2 * this._x0 + this._x1) / 3,
            (2 * this._y0 + this._y1) / 3,
            (this._x0 + 2 * this._x1) / 3,
            (this._y0 + 2 * this._y1) / 3,
            (this._x0 + 4 * this._x1 + x) / 6,
            (this._y0 + 4 * this._y1 + y) / 6
        );
    }
}

grapher.Edge = class {

    constructor(from, to) {
        this.from = from;
        this.to = to;
    }

    get arrowhead() {
        return 'vee';
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

        if (this.label) {
            const tspan = createElement('tspan');
            tspan.setAttribute('xml:space', 'preserve');
            tspan.setAttribute('dy', '1em');
            tspan.setAttribute('x', '1');
            tspan.appendChild(document.createTextNode(this.label));
            const text = createElement('text');
            text.appendChild(tspan);
            const textContainer = createElement('g');
            textContainer.appendChild(text);
            this.labelElement = createElement('g');
            this.labelElement.style.opacity = 0;
            this.labelElement.setAttribute('class', 'edge-label');
            if (this.id) {
                this.labelElement.setAttribute('id', 'edge-label-' + this.id);
            }
            this.labelElement.appendChild(textContainer);
            edgeLabelGroupElement.appendChild(this.labelElement);
            const edgeBox = textContainer.getBBox();
            const edgeX = - edgeBox.width / 2;
            const edgeY = - edgeBox.height / 2;
            textContainer.setAttribute('transform', 'translate(' + edgeX + ',' + edgeY + ')');
            this.width = edgeBox.width;
            this.height = edgeBox.height;
        }
    }

    layout() {
        const edgePath = grapher.Edge._computeCurvePath(this, this.from, this.to);
        this.element.setAttribute('d', edgePath);

        if (this.labelElement) {
            this.labelElement.setAttribute('transform', 'translate(' + this.x + ',' + this.y + ')');
            this.labelElement.style.opacity = 1;
        }
    }

    static _computeCurvePath(edge, tail, head) {
        const points = edge.points.slice(1, edge.points.length - 1);
        points.unshift(grapher.Edge._intersectRect(tail, points[0]));
        points.push(grapher.Edge._intersectRect(head, points[points.length - 1]));

        const path = new Path();
        const curve = new Curve(path);
        for (let i = 0; i < points.length; i++) {
            const point = points[i];
            if (i == 0) {
                curve.lineStart();
            }
            curve.point(point.x, point.y);
            if (i == points.length - 1) {
                curve.lineEnd();
            }
        }

        return path.data;
    }

    static _intersectRect(node, point) {
        const x = node.x;
        const y = node.y;
        const dx = point.x - x;
        const dy = point.y - y;
        let w = node.width / 2;
        let h = node.height / 2;
        let sx;
        let sy;
        if (Math.abs(dy) * w > Math.abs(dx) * h) {
            if (dy < 0) {
                h = -h;
            }
            sx = dy === 0 ? 0 : h * dx / dy;
            sy = h;
        }
        else {
            if (dx < 0) {
                w = -w;
            }
            sx = w;
            sy = dx === 0 ? 0 : w * dy / dx;
        }
        return {x: x + sx, y: y + sy};
    }
};


if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Graph = grapher.Graph;
    module.exports.Node = grapher.Node;
    module.exports.Edge = grapher.Edge;
}