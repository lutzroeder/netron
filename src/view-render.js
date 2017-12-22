/*jshint esversion: 6 */

class GraphRenderer {

    constructor(svg) {
        this._svg = svg;
        this._svgElement = svg.node();
        this._svgNamespace = 'http://www.w3.org/2000/svg';
    }

    render(graph) {

        var svgEdgePathGroup = document.createElementNS(this._svgNamespace, 'g');
        svgEdgePathGroup.setAttribute('class', 'edgePaths');
        this._svgElement.appendChild(svgEdgePathGroup);

        var svgEdgeLabelGroup = document.createElementNS(this._svgNamespace, 'g');
        svgEdgeLabelGroup.setAttribute('class', 'edgeLabes');
        this._svgElement.appendChild(svgEdgeLabelGroup);

        var svgNodeGroup = document.createElementNS(this._svgNamespace, 'g');
        svgNodeGroup.setAttribute('class', 'nodes');
        this._svgElement.appendChild(svgNodeGroup);

        graph.nodes().forEach((nodeId) => {
            var node = graph.node(nodeId);
            var element = document.createElementNS(this._svgNamespace, 'g');
            element.setAttribute('class', node.hasOwnProperty('class') ? ('node ' + node.class) : 'node');
            element.style.setProperty('opacity', 0);
            var container = document.createElementNS(this._svgNamespace, 'g');
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
        });

        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var element = document.createElementNS(this._svgNamespace, 'g');
            element.style.setProperty('opacity', 0);
            element.setAttribute('class', 'edgeLabel');
            svgEdgeLabelGroup.appendChild(element);
            var container = document.createElementNS(this._svgNamespace, 'g');
            d3.select(container).append('text').append('tspan').attr('xml:space', 'preserve').attr('dy', '1em').attr('x', '1').text(edge.label);
            element.appendChild(container);
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
            var node = graph.node(nodeId);
            var element = node.element;
            element.setAttribute('transform', 'translate(' + node.x + ',' + node.y + ')');
            element.style.setProperty('opacity', 1);
            delete node.element;
        });

        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var element = edge.element;
            element.setAttribute('transform', 'translate(' + edge.x + ',' + edge.y + ')');
            element.style.setProperty('opacity', 1);
            delete edge.element;
        });

        d3.select(svgEdgePathGroup).append('defs')
            .append('marker')
                .attr('id', 'arrowhead-vee')
                .attr('viewBox', '0 0 10 10').attr('refX', 9).attr('refY', 5)
                .attr('markerUnits', 'strokeWidth').attr('markerWidth', 8).attr('markerHeight', 6).attr('orient', 'auto')
            .append('path')
                .attr('d', 'M 0 0 L 10 5 L 0 10 L 4 5 z')
                .style('stroke-width', 1).style('stroke-dasharray', '1,0');   
        graph.edges().forEach((edgeId) => {
            var edge = graph.edge(edgeId);
            var points = GraphRenderer.calcPoints(graph, edgeId);
            var element = document.createElementNS(this._svgNamespace, 'path');
            element.setAttribute('class', 'edgePath');
            element.setAttribute('d', points);
            element.setAttribute('marker-end', 'url(#arrowhead-vee)');
            if (edge.hasOwnProperty('style')) {
                element.setAttribute('style', edge.style);
            }
            svgEdgePathGroup.appendChild(element);
        });
    }

    static calcPoints(g, e) {
        const edge = g.edge(e);
        const tail = g.node(e.v);
        const head = g.node(e.w);
        const points = edge.points.slice(1, edge.points.length - 1);
        points.unshift(GraphRenderer.intersectRect(tail, points[0]));
        points.push(GraphRenderer.intersectRect(head, points[points.length - 1]));
        var line = d3.line().x(d => d.x).y(d => d.y);
        if (edge.hasOwnProperty('curve')) {
            line.curve(edge.curve);
        }
        return line(points);
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
}

class NodeFormatter {

    constructor(context) {
        this.items = [];
        this.attributes = [];
    }

    addItem(content, className, title, handler) {
        var item = {};
        if (content) {
            item.content = content;
        }
        if (className) {
            item.class = className;
        }
        if (title) {
            item.title = title;
        }
        if (handler) {
            item.handler = handler;
        }
        this.items.push(item);
    }

    addAttribute(name, value, title) {
        this.attributes.push({ name: name, value: value, title: title });
    }

    setAttributeHandler(handler) {
        this.attributeHandler = handler;
    }

    format(context) {
        var root = context.append('g');
        var hasAttributes = this.attributes && this.attributes.length > 0;
        var x = 0;
        var y = 0;
        var maxWidth = 0;
        var itemHeight = 0;
        var itemBoxes = [];
        this.items.forEach((item, index) => {
            var yPadding = 4;
            var xPadding = 7;
            var itemGroup = root.append('g').classed('node-item', true);
            var path = itemGroup.append('path');
            var text = itemGroup.append('text');
            var content = item.content;
            var className = item.class; 
            var handler = item.handler;
            var title = item.title;
            if (className) {
                itemGroup.classed(className, true);
            }
            if (handler) {
                itemGroup.on('click', handler);
            }
            if (title) {
                itemGroup.append('title').text(title);
            }
            if (content) {
                text.text(content);
            }
            var boundingBox = text.node().getBBox();
            var width = boundingBox.width + xPadding + xPadding;
            var height = boundingBox.height + yPadding + yPadding;
            itemBoxes.push({
                'group': itemGroup, 'text': text, 'path': path,
                'x': x, 'y': y,
                'width': width, 'height': height,
                'tx': xPadding, 'ty': yPadding - boundingBox.y
            });
            x += width;
            if (itemHeight < height) {
                itemHeight = height;
            }
            if (x > maxWidth) { 
                maxWidth = x;
            }
        });

        var itemWidth = maxWidth;

        x = 0;
        y += itemHeight;

        var attributesHeight = 0;
        var attributesPath = null;
        if (hasAttributes)
        {
            var attributeGroup = root.append('g').classed('node-attribute', true);
            if (this.attributeHandler) {
                attributeGroup.on('click', this.attributeHandler);
            }
            attributesPath = attributeGroup.append('path');
            attributeGroup.attr('transform', 'translate(' + x + ',' + y + ')');
            attributesHeight += 4;
            this.attributes.forEach((attribute) => {
                var yPadding = 1;
                var xPadding = 4;
                var text = attributeGroup.append('text').attr('xml:space', 'preserve');
                if (attribute.title) {
                    text.append('title').text(attribute.title);
                }
                var text_name = text.append('tspan').style('font-weight', 'bold').text(attribute.name);
                var text_value = text.append('tspan').text(' = ' + attribute.value);
                var size = text.node().getBBox();
                var width = xPadding + size.width + xPadding;
                if (maxWidth < width) {
                    maxWidth = width;
                }
                text.attr('x', x + xPadding);
                text.attr('y', attributesHeight + yPadding - size.y);
                attributesHeight += yPadding + size.height + yPadding;
            });
            attributesHeight += 4;
        }

        if (maxWidth > itemWidth) {
            var d = (maxWidth - itemWidth) / this.items.length;
            itemBoxes.forEach((itemBox, index) => {
                itemBox.x = itemBox.x + (index * d);
                itemBox.width = itemBox.width + d;
                itemBox.tx = itemBox.tx + (0.5 * d);
            });
        }

        itemBoxes.forEach((itemBox, index) => {
            itemBox.group.attr('transform', 'translate(' + itemBox.x + ',' + itemBox.y + ')');        
            var r1 = index == 0;
            var r2 = index == itemBoxes.length - 1;
            var r3 = !hasAttributes && r2;
            var r4 = !hasAttributes && r1;
            itemBox.path.attr('d', this.roundedRect(0, 0, itemBox.width, itemBox.height, r1, r2, r3, r4));
            itemBox.text.attr('x', itemBox.tx).attr('y', itemBox.ty);
        });

        if (hasAttributes) {
            attributesPath.attr('d', this.roundedRect(0, 0, maxWidth, attributesHeight, false, false, true, true));
        }

        itemBoxes.forEach((itemBox, index) => {
            if (index != 0) {
                root.append('line').classed('node', true).attr('x1', itemBox.x).attr('y1', 0).attr('x2', itemBox.x).attr('y2', itemHeight);
            }
        });
        if (hasAttributes) {
            root.append('line').classed('node', true).attr('x1', 0).attr('y1', itemHeight).attr('x2', maxWidth).attr('y2', itemHeight);
        }
        root.append('path').classed('node', true).attr('d', this.roundedRect(0, 0, maxWidth, itemHeight + attributesHeight, true, true, true, true));

        context.html("");
        return root;
    }

    roundedRect(x, y, width, height, r1, r2, r3, r4) {
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
}