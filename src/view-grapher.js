/*jshint esversion: 6 */

var grapher = grapher || {};

var dagre = dagre || require('dagre');

grapher.Renderer = class {

    constructor(document, svgElement) {
        this._document = document;
        this._svgElement = svgElement;
    }

    render(graph) {

        var svgClusterGroup = this.createElement('g');
        svgClusterGroup.setAttribute('id', 'clusters');
        svgClusterGroup.setAttribute('class', 'clusters');
        this._svgElement.appendChild(svgClusterGroup);

        var svgEdgePathGroup = this.createElement('g');
        svgEdgePathGroup.setAttribute('id', 'edge-paths');
        svgEdgePathGroup.setAttribute('class', 'edge-paths');
        this._svgElement.appendChild(svgEdgePathGroup);

        var svgEdgeLabelGroup = this.createElement('g');
        svgEdgeLabelGroup.setAttribute('id', 'edge-labels');
        svgEdgeLabelGroup.setAttribute('class', 'edge-labels');
        this._svgElement.appendChild(svgEdgeLabelGroup);

        var svgNodeGroup = this.createElement('g');
        svgNodeGroup.setAttribute('id', 'nodes');
        svgNodeGroup.setAttribute('class', 'nodes');
        this._svgElement.appendChild(svgNodeGroup);

        graph.nodes().forEach((nodeId) => {
            if (graph.children(nodeId).length == 0) {
                var node = graph.node(nodeId);
                var element = this.createElement('g');
                if (node.id) {
                    element.setAttribute('id', node.id);
                }
                element.setAttribute('class', node.hasOwnProperty('class') ? ('node ' + node.class) : 'node');
                element.style.opacity = 0;
                var container = this.createElement('g');
                container.appendChild(node.label);
                element.appendChild(container);
                svgNodeGroup.appendChild(element);
                var bbox = node.label.getBBox();
                var x = - bbox.width / 2;
                var y = - bbox.height / 2;
                container.setAttribute('transform', 'translate(' + x + ',' + y + ')');
                node.width = bbox.width;
                node.height = bbox.height;
                node.element = element;
            }
        });

        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var tspan = this.createElement('tspan');
            tspan.setAttribute('xml:space', 'preserve');
            tspan.setAttribute('dy', '1em');
            tspan.setAttribute('x', '1');
            tspan.appendChild(this._document.createTextNode(edge.label));
            var text = this.createElement('text');
            text.appendChild(tspan);
            var container = this.createElement('g');
            container.appendChild(text);
            var element = this.createElement('g');
            element.style.opacity = 0;
            element.setAttribute('class', 'edge-label');
            element.appendChild(container);
            svgEdgeLabelGroup.appendChild(element);
            var bbox = container.getBBox();
            var x = - bbox.width / 2;
            var y = - bbox.height / 2;
            container.setAttribute('transform', 'translate(' + x + ',' + y + ')');
            edge.width = bbox.width;
            edge.height = bbox.height;
            edge.element = element;
        });

        dagre.layout(graph);

        graph.nodes().forEach((nodeId) => {
            if (graph.children(nodeId).length == 0) {
                var node = graph.node(nodeId);
                var element = node.element;
                element.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
                element.style.opacity = 1;
                delete node.element;
            }
        });

        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var element = edge.element;
            element.setAttribute('transform', 'translate(' + edge.x + ',' + edge.y + ')');
            element.style.opacity = 1;
            delete edge.element;
        });

        var edgePathGroupDefs = this.createElement('defs');
        svgEdgePathGroup.appendChild(edgePathGroupDefs);
        var marker = this.createElement('marker');
        marker.setAttribute('id', 'arrowhead-vee');
        marker.setAttribute('viewBox', '0 0 10 10');
        marker.setAttribute('refX', 9);
        marker.setAttribute('refY', 5);
        marker.setAttribute('markerUnits', 'strokeWidth');
        marker.setAttribute('markerWidth', 8);
        marker.setAttribute('markerHeight', 6);
        marker.setAttribute('orient', 'auto');
        edgePathGroupDefs.appendChild(marker);
        var markerPath = this.createElement('path');
        markerPath.setAttribute('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z');
        markerPath.style.setProperty('stroke-width', 1);
        markerPath.style.setProperty('stroke-dasharray', '1,0');
        marker.appendChild(markerPath);

        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var edgePath = grapher.Renderer._computeCurvePath(edge, graph.node(edgeId.v), graph.node(edgeId.w));
            var edgeElement = this.createElement('path');
            edgeElement.setAttribute('class', edge.hasOwnProperty('class') ? ('edge-path ' + edge.class) : 'edge-path');
            edgeElement.setAttribute('d', edgePath);
            edgeElement.setAttribute('marker-end', 'url(#arrowhead-vee)');
            if (edge.id) {
                edgeElement.setAttribute('id', edge.id);
            }
            svgEdgePathGroup.appendChild(edgeElement);
        });

        graph.nodes().forEach((nodeId) => {
            if (graph.children(nodeId).length > 0) {
                var node = graph.node(nodeId);
                var element = this.createElement('g');
                element.setAttribute('class', 'cluster');
                element.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
                var rect = this.createElement('rect');
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
                element.appendChild(rect);
                svgClusterGroup.appendChild(element);
            }
        });
    }

    createElement(name) {
        return this._document.createElementNS('http://www.w3.org/2000/svg', name);
    }

    static _computeCurvePath(edge, tail, head) {
        var points = edge.points.slice(1, edge.points.length - 1);
        points.unshift(grapher.Renderer.intersectRect(tail, points[0]));
        points.push(grapher.Renderer.intersectRect(head, points[points.length - 1]));

        var path = new Path();
        var curve = new Curve(path);
        points.forEach((point, index) => {
            if (index == 0) {
                curve.lineStart();
            }
            curve.point(point.x, point.y);
            if (index == points.length - 1) {
                curve.lineEnd();
            }
        });

        return path.data;
    }
    
    static intersectRect(node, point) {
        var x = node.x;
        var y = node.y;
        var dx = point.x - x;
        var dy = point.y - y;
        var w = node.width / 2;
        var h = node.height / 2;
        var sx;
        var sy;
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

    setControlDependencies() {
        this._controlDependencies = true;
    }

    format(contextElement) {
        var rootElement = this.createElement('g');
        contextElement.appendChild(rootElement);

        var width = 0;
        var height = 0;
        var tops = [];

        this._blocks.forEach((block) => {
            tops.push(height);
            block.layout(rootElement);
            if (width < block.width) {
                width = block.width;
            }
            height = height + block.height;
        });

        this._blocks.forEach((block, index) => {
            var top = tops.shift();
            block.update(rootElement, top, width, index == 0, index == this._blocks.length - 1);
        });

        var borderElement = this.createElement('path');
        var borderClassList = [ 'node', 'border' ];
        if (this._controlDependencies) {
            borderClassList.push('node-control-dependency');
        }
        borderElement.setAttribute('class', borderClassList.join(' '));
        borderElement.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, width, height, true, true, true, true));
        rootElement.appendChild(borderElement);

        contextElement.innerHTML = '';
        return rootElement;
    }

    static roundedRect(x, y, width, height, r1, r2, r3, r4) {
        var radius = 5;    
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
        var x = 0;
        var y = 0;
        this._items.forEach((item, index) => {
            var yPadding = 4;
            var xPadding = 7;
            var element = this.createElement('g');
            var classList = [ 'node-item' ];
            parentElement.appendChild(element);
            var pathElement = this.createElement('path');
            var textElement = this.createElement('text');
            element.appendChild(pathElement);
            element.appendChild(textElement);
            if (item.classList) {
                item.classList.forEach((className) => {
                    classList.push(className);
                });
            }
            element.setAttribute('class', classList.join(' '));
            if (item.id) {
                element.setAttribute('id', item.id);
            }
            if (item.handler) {
                element.addEventListener('click', item.handler);
            }
            if (item.tooltip) {
                var titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                element.appendChild(titleElement);
            }
            if (item.content) {
                textElement.textContent = item.content;
            }
            var boundingBox = textElement.getBBox();
            var width = boundingBox.width + xPadding + xPadding;
            var height = boundingBox.height + yPadding + yPadding;
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
        });
    }

    get width() {
        return this._width;
    }

    get height() {
        return this._height;
    }

    update(parentElement, top, width, first, last) {

        var dx = width - this._width;
        this._elements.forEach((element, index) => {
            if (index == 0) {
                element.width = element.width + dx;
            }
            else {
                element.x = element.x + dx;
                element.tx = element.tx + dx;
            }
            element.y = element.y + top;
        });

        this._elements.forEach((element, index) => {
            element.group.setAttribute('transform', 'translate(' + element.x + ',' + element.y + ')');        
            var r1 = index == 0 && first;
            var r2 = index == this._elements.length - 1 && first;
            var r3 = index == this._elements.length - 1 && last;
            var r4 = index == 0 && last;
            element.path.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, element.width, element.height, r1, r2, r3, r4));
            element.text.setAttribute('x', 6);
            element.text.setAttribute('y', element.ty);
        });

        this._elements.forEach((element, index) => {
            if (index != 0) {
                var lineElement = this.createElement('line');
                lineElement.setAttribute('class', 'node');
                lineElement.setAttribute('x1', element.x);
                lineElement.setAttribute('x2', element.x);
                lineElement.setAttribute('y1', top);
                lineElement.setAttribute('y2', top + this._height);
                parentElement.appendChild(lineElement);
            }
        });

        if (!first) 
        {
            var lineElement = this.createElement('line');
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
        var x = 0;
        var y = 0;
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
        this._items.forEach((item) => {
            var yPadding = 1;
            var xPadding = 6;
            var textElement = this.createElement('text');
            if (item.id) {
                textElement.setAttribute('id', item.id);
            }
            textElement.setAttribute('xml:space', 'preserve');
            this._element.appendChild(textElement);
            if (item.tooltip) {
                var titleElement = this.createElement('title');
                titleElement.textContent = item.tooltip;
                textElement.appendChild(titleElement);
            }
            var textNameElement = this.createElement('tspan');
            textNameElement.textContent = item.name;
            textNameElement.style.fontWeight = 'bold';
            textElement.appendChild(textNameElement);
            var textValueElement = this.createElement('tspan');
            textValueElement.textContent = item.separator + item.value;
            textElement.appendChild(textValueElement);
            var size = textElement.getBBox();
            var width = xPadding + size.width + xPadding;
            if (this._width < width) {
                this._width = width;
            }
            textElement.setAttribute('x', x + xPadding);
            textElement.setAttribute('y', this._height + yPadding - size.y);
            this._height += yPadding + size.height + yPadding;
        });
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

        var r1 = first;
        var r2 = first;
        var r3 = last;
        var r4 = last;
        this._backgroundElement.setAttribute('d', grapher.NodeElement.roundedRect(0, 0, width, this._height, r1, r2, r3, r4));

        if (!first) 
        {
            var lineElement = this.createElement('line');
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