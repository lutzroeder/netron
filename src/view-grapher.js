/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

var grapher = grapher || {};
var dagre = dagre || require('dagre');

grapher.Renderer = class {

    constructor(document, svgElement) {
        this._document = document;
        this._svgElement = svgElement;
    }

    render(graph) {

        let svgClusterGroup = this.createElement('g');
        svgClusterGroup.setAttribute('id', 'clusters');
        svgClusterGroup.setAttribute('class', 'clusters');
        this._svgElement.appendChild(svgClusterGroup);

        let svgEdgePathGroup = this.createElement('g');
        svgEdgePathGroup.setAttribute('id', 'edge-paths');
        svgEdgePathGroup.setAttribute('class', 'edge-paths');
        this._svgElement.appendChild(svgEdgePathGroup);

        let svgEdgeLabelGroup = this.createElement('g');
        svgEdgeLabelGroup.setAttribute('id', 'edge-labels');
        svgEdgeLabelGroup.setAttribute('class', 'edge-labels');
        this._svgElement.appendChild(svgEdgeLabelGroup);

        let svgNodeGroup = this.createElement('g');
        svgNodeGroup.setAttribute('id', 'nodes');
        svgNodeGroup.setAttribute('class', 'nodes');
        this._svgElement.appendChild(svgNodeGroup);

        for (const nodeId of graph.nodes()) {
            if (graph.children(nodeId).length == 0) {
                const node = graph.node(nodeId);
                const element = this.createElement('g');
                if (node.id) {
                    element.setAttribute('id', node.id);
                }
                element.setAttribute('class', Object.prototype.hasOwnProperty.call(node, 'class') ? ('node ' + node.class) : 'node');
                element.style.opacity = 0;
                const container = this.createElement('g');
                container.appendChild(node.label);
                element.appendChild(container);
                svgNodeGroup.appendChild(element);
                const nodeBox = node.label.getBBox();
                const nodeX = - nodeBox.width / 2;
                const nodeY = - nodeBox.height / 2;
                container.setAttribute('transform', 'translate(' + nodeX + ',' + nodeY + ')');
                node.width = nodeBox.width;
                node.height = nodeBox.height;
                node.element = element;
            }
        }

        for (const edgeId of graph.edges()) {
            const edge = graph.edge(edgeId);
            if (edge.label) {
                let tspan = this.createElement('tspan');
                tspan.setAttribute('xml:space', 'preserve');
                tspan.setAttribute('dy', '1em');
                tspan.setAttribute('x', '1');
                tspan.appendChild(this._document.createTextNode(edge.label));
                const text = this.createElement('text');
                text.appendChild(tspan);
                const textContainer = this.createElement('g');
                textContainer.appendChild(text);
                const labelElement = this.createElement('g');
                labelElement.style.opacity = 0;
                labelElement.setAttribute('class', 'edge-label');
                labelElement.appendChild(textContainer);
                svgEdgeLabelGroup.appendChild(labelElement);
                const edgeBox = textContainer.getBBox();
                const edgeX = - edgeBox.width / 2;
                const edgeY = - edgeBox.height / 2;
                textContainer.setAttribute('transform', 'translate(' + edgeX + ',' + edgeY + ')');
                edge.width = edgeBox.width;
                edge.height = edgeBox.height;
                edge.labelElement = labelElement;
            }
        }

        dagre.layout(graph);

        for (const nodeId of graph.nodes()) {
            if (graph.children(nodeId).length == 0) {
                const node = graph.node(nodeId);
                node.element.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
                node.element.style.opacity = 1;
                delete node.element;
            }
        }

        for (const edgeId of graph.edges()) {
            const edge = graph.edge(edgeId);
            if (edge.labelElement) {
                edge.labelElement.setAttribute('transform', 'translate(' + edge.x + ',' + edge.y + ')');
                edge.labelElement.style.opacity = 1;
                delete edge.labelElement;
            }
        }

        const edgePathGroupDefs = this.createElement('defs');
        svgEdgePathGroup.appendChild(edgePathGroupDefs);
        const marker = this.createElement('marker');
        marker.setAttribute('id', 'arrowhead-vee');
        marker.setAttribute('viewBox', '0 0 10 10');
        marker.setAttribute('refX', 9);
        marker.setAttribute('refY', 5);
        marker.setAttribute('markerUnits', 'strokeWidth');
        marker.setAttribute('markerWidth', 8);
        marker.setAttribute('markerHeight', 6);
        marker.setAttribute('orient', 'auto');
        edgePathGroupDefs.appendChild(marker);
        const markerPath = this.createElement('path');
        markerPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z');
        markerPath.style.setProperty('stroke-width', 1);
        markerPath.style.setProperty('stroke-dasharray', '1,0');
        marker.appendChild(markerPath);

        for (const edgeId of graph.edges()) {
            const edge = graph.edge(edgeId);
            const edgePath = grapher.Renderer._computeCurvePath(edge, graph.node(edgeId.v), graph.node(edgeId.w));
            const edgeElement = this.createElement('path');
            edgeElement.setAttribute('class', Object.prototype.hasOwnProperty.call(edge, 'class') ? ('edge-path ' + edge.class) : 'edge-path');
            edgeElement.setAttribute('d', edgePath);
            edgeElement.setAttribute('marker-end', 'url(#arrowhead-vee)');
            if (edge.id) {
                edgeElement.setAttribute('id', edge.id);
            }
            svgEdgePathGroup.appendChild(edgeElement);
        }

        for (const nodeId of graph.nodes()) {
            if (graph.children(nodeId).length > 0) {
                const node = graph.node(nodeId);
                const nodeElement = this.createElement('g');
                nodeElement.setAttribute('class', 'cluster');
                nodeElement.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
                const rect = this.createElement('rect');
                rect.setAttribute('x', - node.width / 2);
                rect.setAttribute('y', - node.height / 2 );
                rect.setAttribute('width', node.width);
                rect.setAttribute('height', node.height);
                if (node.rx) {
                    rect.setAttribute('rx', node.rx);
                }
                if (node.ry) {
                    rect.setAttribute('ry', node.ry);
                }
                nodeElement.appendChild(rect);
                svgClusterGroup.appendChild(nodeElement);
            }
        }
    }

    createElement(name) {
        return this._document.createElementNS('http://www.w3.org/2000/svg', name);
    }

    static _computeCurvePath(edge, tail, head) {
        let points = edge.points.slice(1, edge.points.length - 1);
        points.unshift(grapher.Renderer.intersectRect(tail, points[0]));
        points.push(grapher.Renderer.intersectRect(head, points[points.length - 1]));

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

    static intersectRect(node, point) {
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

grapher.NodeElement = class {

    constructor(document) {
        this._document = document;
        this._blocks = [];
    }

    block(type) {
        this._block = null;
        switch (type) {
            case 'header':
                this._block = new grapher.NodeElement.Header(this._document);
                break;
            case 'list':
                this._block = new grapher.NodeElement.List(this._document);
                break;
        }
        this._blocks.push(this._block);
        return this._block;
    }

    format(contextElement) {
        let rootElement = this.createElement('g');
        contextElement.appendChild(rootElement);

        let width = 0;
        let height = 0;
        let tops = [];

        for (const block of this._blocks) {
            tops.push(height);
            block.layout(rootElement);
            if (width < block.width) {
                width = block.width;
            }
            height = height + block.height;
        }

        for (let i = 0; i < this._blocks.length; i++) {
            let top = tops.shift();
            this._blocks[i].update(rootElement, top, width, i == 0, i == this._blocks.length - 1);
        }

        let borderElement = this.createElement('path');
        borderElement.setAttribute('class', [ 'node', 'border' ].join(' '));
        borderElement.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, width, height, true, true, true, true));
        rootElement.appendChild(borderElement);

        contextElement.innerHTML = '';
        return rootElement;
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

    createElement(name) {
        return this._document.createElementNS('http://www.w3.org/2000/svg', name);
    }
};

grapher.NodeElement.Header = class {

    constructor(document) {
        this._document = document;
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

    layout(parentElement) {
        this._width = 0;
        this._height = 0;
        this._elements = [];
        let x = 0;
        let y = 0;
        for (const item of this._items) {
            let yPadding = 4;
            let xPadding = 7;
            let element = this.createElement('g');
            let classList = [ 'node-item' ];
            parentElement.appendChild(element);
            let pathElement = this.createElement('path');
            let textElement = this.createElement('text');
            element.appendChild(pathElement);
            element.appendChild(textElement);
            if (item.classList) {
                classList = classList.concat(item.classList);
            }
            element.setAttribute('class', classList.join(' '));
            if (item.id) {
                element.setAttribute('id', item.id);
            }
            if (item.handler) {
                element.addEventListener('click', item.handler);
            }
            if (item.tooltip) {
                let titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                element.appendChild(titleElement);
            }
            if (item.content) {
                textElement.textContent = item.content;
            }
            let boundingBox = textElement.getBBox();
            let width = boundingBox.width + xPadding + xPadding;
            let height = boundingBox.height + yPadding + yPadding;
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

        let dx = width - this._width;
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
            let r1 = i == 0 && first;
            let r2 = i == this._elements.length - 1 && first;
            let r3 = i == this._elements.length - 1 && last;
            let r4 = i == 0 && last;
            element.path.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, element.width, element.height, r1, r2, r3, r4));
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

grapher.NodeElement.List = class {

    constructor(document) {
        this._document = document;
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

    layout(parentElement) {
        this._width = 0;
        this._height = 0;
        let x = 0;
        let y = 0;
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
            let textElement = this.createElement('text');
            if (item.id) {
                textElement.setAttribute('id', item.id);
            }
            textElement.setAttribute('xml:space', 'preserve');
            this._element.appendChild(textElement);
            if (item.tooltip) {
                let titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                textElement.appendChild(titleElement);
            }
            let textNameElement = this.createElement('tspan');
            textNameElement.textContent = item.name;
            if (item.separator.trim() != '=') {
                textNameElement.style.fontWeight = 'bold';
            }
            textElement.appendChild(textNameElement);
            let textValueElement = this.createElement('tspan');
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

        let r1 = first;
        let r2 = first;
        let r3 = last;
        let r4 = last;
        this._backgroundElement.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, width, this._height, r1, r2, r3, r4));

        if (!first) {
            let lineElement = this.createElement('line');
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

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Renderer = grapher.Renderer;
    module.exports.NodeElement = grapher.NodeElement;
}