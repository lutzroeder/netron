
function NodeFormatter(context) {
    this.items = [];
    this.properties = [];
    this.attributes = [];
}

NodeFormatter.prototype.addItem = function(content, className, title, handler) {
    var item = {};
    if (content) {
        item['content'] = content;
    }
    if (className) {
        item['class'] = className;
    }
    if (title) {
        item['title'] = title;
    }
    if (handler) {
        item['handler'] = handler;
    }
    this.items.push(item);
};

NodeFormatter.prototype.addProperty = function(name, value) {
    this.properties.push({ 'name': name, 'value': value });
}

NodeFormatter.prototype.setPropertyHandler = function(handler) {
    this.propertyHandler = handler;
}

NodeFormatter.prototype.addAttribute = function(name, value, title) {
    this.attributes.push({ 'name': name, 'value': value, 'title': title });
}

NodeFormatter.prototype.setAttributeHandler = function(handler) {
    this.attributeHandler = handler;
}

NodeFormatter.prototype.format = function(context) {
    var self = this;
    var root = context.append('g');
    var hasProperties = self.properties && self.properties.length > 0;
    var hasAttributes = self.attributes && self.attributes.length > 0;
    var x = 0;
    var y = 0;
    var maxWidth = 0;
    var itemHeight = 0;
    self.items.forEach(function (item, index) {
        var yPadding = 4;
        var xPadding = 7;
        var itemGroup = root.append('g').classed('node-item', true);
        itemGroup.attr('transform', 'translate(' + x + ',' + y + ')');
        var path = itemGroup.append('path');
        var text = itemGroup.append('text');
        var content = item['content'];
        var className = item['class']; 
        var handler = item['handler'];
        var title = item['title']
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
        text.attr('x', xPadding);
        text.attr('y', yPadding - boundingBox.y);
        var r1 = index == 0;
        var r2 = index == self.items.length - 1;
        var r3 = !hasAttributes && !hasProperties && r2;
        var r4 = !hasAttributes && !hasProperties && r1;
        path.attr('d', self.roundedRect(0, 0, width, height, r1, r2, r3, r4));
        x += width;
        if (height > itemHeight) {
            itemHeight = height;
        }
        if (x > maxWidth) { 
            maxWidth = x;
        }
    });

    var itemWidth = maxWidth;

    x = 0;
    y += itemHeight;

    var propertiesHeight = 0;
    var propertiesPath = null;
    if (hasProperties) {
        var propertiesGroup = root.append('g').classed('node-property', true);
        if (self.propertyHandler) {
            propertiesGroup.on('click', self.propertyHandler);
        }
        propertiesPath = propertiesGroup.append('path');
        propertiesGroup.attr('transform', 'translate(' + x + ',' + y + ')');
        propertiesHeight += 4;
        self.properties.forEach(function (property) {
            var yPadding = 1;
            var xPadding = 4;
            var text = propertiesGroup.append('text').attr('xml:space', 'preserve');
            var text_name = text.append('tspan').style('font-weight', 'bold').text(property.name);
            var text_value = text.append('tspan').text(': ' + property.value)
            var size = text.node().getBBox();
            var width = xPadding + size.width + xPadding;
            if (maxWidth < width) {
                maxWidth = width;
            }
            text.attr('x', x + xPadding);
            text.attr('y', propertiesHeight + yPadding - size.y);
            propertiesHeight += yPadding + size.height + yPadding;
        });
        propertiesHeight += hasAttributes ? 1 : 4;
    }

    y += propertiesHeight;

    var attributesHeight = 0;
    var attributesPath = null;
    if (hasAttributes)
    {
        var attributeGroup = root.append('g').classed('node-attribute', true);
        if (self.attributeHandler) {
            attributeGroup.on('click', self.attributeHandler);
        }
        attributesPath = attributeGroup.append('path');
        attributeGroup.attr('transform', 'translate(' + x + ',' + y + ')');
        attributesHeight += hasProperties ? 1 : 4;
        self.attributes.forEach(function (attribute) {
            var yPadding = 1;
            var xPadding = 4;
            var text = attributeGroup.append('text').attr('xml:space', 'preserve');
            if (attribute['title']) {
                text.append('title').text(attribute['title']);
            }
            var text_name = text.append('tspan').style('font-weight', 'bold').text(attribute.name);
            var text_value = text.append('tspan').text(' = ' + attribute.value)
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
        var d = (maxWidth - itemWidth) / self.items.length;
        self.items.forEach(function (item, index) {
            var itemGroup = dagreD3.d3.select(root.node().children[index]);
            var t = dagreD3.d3.transform(itemGroup.attr('transform')).translate;
            itemGroup.attr('transform', 'translate(' + (t[0] + index * d) + ',' + t[1] + ')');
            var path = dagreD3.d3.select(itemGroup.node().children[0]);
            var r1 = index == 0;
            var r2 = index == self.items.length - 1;
            var r3 = !hasAttributes && !hasProperties && r2;
            var r4 = !hasAttributes && !hasProperties && r1;
            var box = path.node().getBBox();
            path.attr('d', self.roundedRect(0, 0, box.width + d, box.height, r1, r2, r3, r4));
            var text = dagreD3.d3.select(itemGroup.node().children[1]);
            var t2 = dagreD3.d3.transform(text.attr('transform')).translate;
            text.attr('transform', 'translate(' + (t2[0] + 0.5 * d) + ',' + t2[1] + ')');
        });
    }

    if (hasProperties) {
        propertiesPath.attr('d', self.roundedRect(0, 0, maxWidth, propertiesHeight, false, false, !hasAttributes, !hasAttributes));
    }

    if (hasAttributes) {
        attributesPath.attr('d', self.roundedRect(0, 0, maxWidth, attributesHeight, false, false, true, true));
    }

    self.items.forEach(function(item, index) {
        if (index != 0) {
            var itemGroup = dagreD3.d3.select(root.node().children[index]);
            var t = dagreD3.d3.transform(itemGroup.attr('transform')).translate;
            root.append('line').classed('node', true).attr('x1', t[0]).attr('y1', 0).attr('x2', t[0]).attr('y2', itemHeight);
        }
    });

    if (hasAttributes || hasProperties) {
        root.append('line').classed('node', true).attr('x1', 0).attr('y1', itemHeight).attr('x2', maxWidth).attr('y2', itemHeight);
    }

    root.append('path').classed('node', true).attr('d', self.roundedRect(0, 0, maxWidth, itemHeight + propertiesHeight + attributesHeight, true, true, true, true));

    context.html("");
    return root;
};

NodeFormatter.prototype.roundedRect = function(x, y, width, height, r1, r2, r3, r4) {
    var radius = 5;
    r1 = r1 ? radius : 0;
    r2 = r2 ? radius : 0;
    r3 = r3 ? radius : 0;
    r4 = r4 ? radius : 0;
    return "M" + (x + r1) + "," + y
       + "h" + (width - r1 - r2)
       + "a" + r2 + "," + r2 + " 0 0 1 " + r2 + "," + r2
       + "v" + (height - r2 - r3)
       + "a" + r3 + "," + r3 + " 0 0 1 " + -r3 + "," + r3
       + "h" + (r3 + r4 - width)
       + "a" + r4 + "," + r4 + " 0 0 1 " + -r4 + "," + -r4
       + 'v' + (-height + r4 + r1)
       + "a" + r1 + "," + r1 + " 0 0 1 " + r1 + "," + -r1
       + "z";
};
