/*jshint esversion: 6 */

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