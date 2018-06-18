/*jshint esversion: 6 */

class NodeView {

    constructor(node) {
        this._node = node;
        this._elements = [];
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        var operatorElement = document.createElement('div');
        operatorElement.className = 'node-view-title';
        operatorElement.innerText = node.operator;
        this._elements.push(operatorElement);

        if (node.documentation) {
            operatorElement.innerText += ' ';
            var documentationButton = document.createElement('a');
            documentationButton.className = 'node-view-documentation-button';
            documentationButton.innerText = '?';
            documentationButton.addEventListener('click', (e) => {
                this.raise('show-documentation', null);
            });
            operatorElement.appendChild(documentationButton);
        }

        if (node.name) {
            this.addProperty('name', new ValueContentView(node.name));
        }

        if (node.domain) {
            this.addProperty('domain', new ValueContentView(node.domain));
        }

        if (node.description) {
            this.addProperty('description', new ValueContentView(node.description));
        }

        if (node.device) {
            this.addProperty('device', new ValueContentView(node.device));
        }

        var attributes = node.attributes;
        if (attributes && attributes.length > 0) {
            this.addHeader('Attributes');
            attributes.forEach((attribute) => {
                this.addAttribute(attribute.name, attribute);
            });
        }

        var inputs = node.inputs;
        if (inputs && inputs.length > 0) {
            this.addHeader('Inputs');
            inputs.forEach((input) => {
                this.addInput(input.name, input);
            });
        }

        var outputs = node.outputs;
        if (outputs && outputs.length > 0) {
            this.addHeader('Outputs');
            outputs.forEach((output) => {
                this.addOutput(output.name, output);
            });
        }

        var divider = document.createElement('div');
        divider.setAttribute('style', 'margin-bottom: 20px');
        this._elements.push(divider);
    }

    get elements() {
        return this._elements;
    }

    addHeader(title) {
        var headerElement = document.createElement('div');
        headerElement.className = 'node-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    addProperty(name, value) {
        var item = new NameValueView(name, value);
        this._elements.push(item.element);
    }

    addAttribute(name, attribute) {
        var item = new NameValueView(name, new NodeAttributeValueView(attribute));
        this._attributes.push(item);
        this._elements.push(item.element);
    }

    addInput(name, input) {
        if (input.connections.length > 0) {
            var item = new NameValueView(name, new NodeArgumentView(input));
            this._inputs.push(item);
            this._elements.push(item.element);
        }
    }

    addOutput(name, output) {
        if (output.connections.length > 0) {
            var item = new NameValueView(name, new NodeArgumentView(output));
            this._outputs.push(item);
            this._elements.push(item.element);
        }
    }

    toggleInput(name) {
        this._inputs.forEach((input) => {
            if (name == input.name) {
                input.toggle();
            }
        });
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    raise(event, data) {
        if (this._events && this._events[event]) {
            this._events[event].forEach((callback) => {
                callback(this, data);
            });
        }
    }
}

class NameValueView {
    constructor(name, value) {
        this._name = name;
        this._value = value;

        var itemName = document.createElement('div');
        itemName.className = 'node-view-item-name';

        var inputName = document.createElement('input');
        inputName.setAttribute('type', 'text');
        inputName.setAttribute('value', name);
        inputName.setAttribute('title', name);
        inputName.setAttribute('readonly', 'true');
        itemName.appendChild(inputName);

        var itemValueList = document.createElement('div');
        itemValueList.className = 'node-view-item-value-list';

        value.elements.forEach((element) => {
            itemValueList.appendChild(element);
        });

        this._element = document.createElement('div');
        this._element.className = 'node-view-item';
        this._element.appendChild(itemName);
        this._element.appendChild(itemValueList);
    }

    get name() {
        return this._name;
    }

    get element() {
        return this._element;
    }

    toggle() {
        this._value.toggle();
    }
}

class ValueContentView {
    constructor(value) {
        var line = document.createElement('div');
        line.className = 'node-view-item-value-line';
        line.innerHTML = '<code>' + value + '</code>';
        var element = document.createElement('div');
        element.className = 'node-view-item-value';
        element.appendChild(line);
        this._elements = [ element ];
    }

    get elements() {
        return this._elements;
    }

    toggle() {
    }
}

class NodeAttributeValueView {

    constructor(attribute) {
        this._attribute = attribute;
        this._element = document.createElement('div');
        this._element.className = 'node-view-item-value';

        if (attribute.type) {
            this._expander = document.createElement('div');
            this._expander.className = 'node-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', (e) => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        var value = this._attribute.value;
        if (value.length > 1000) {
            value = value.substring(0, 1000) + '...';
        }
        var valueLine = document.createElement('div');
        valueLine.className = 'node-view-item-value-line';
        valueLine.innerHTML = '<code>' + (value ? value : '&nbsp;') + '</code>';
        this._element.appendChild(valueLine);
    }

    get elements() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander.innerText == '+') {
            this._expander.innerText = '-';

            var typeLine = document.createElement('div');
            typeLine.className = 'node-view-item-value-line-border';
            typeLine.innerHTML = 'type: ' + '<code><b>' + this._attribute.type + '</b></code>';
            this._element.appendChild(typeLine);
        }
        else {
            this._expander.innerText = '+';
            while (this._element.childElementCount > 2) {
                this._element.removeChild(this._element.lastChild);
            }
        }
    }
}

class NodeArgumentView {

    constructor(list) {
        this._list = list;
        this._elements = [];
        this._items = [];
        list.connections.forEach((connection) => {
            var item = new NodeConnectionView(connection);
            this._items.push(item);
            this._elements.push(item.element);
        });
    }

    get elements() {
        return this._elements;
    }

    toggle() {
        this._items.forEach((item) => {
            item.toggle();
        });
    }
}

class NodeConnectionView {
    constructor(connection) {
        this._connection = connection;
        this._element = document.createElement('div');
        this._element.className = 'node-view-item-value';

        var initializer = connection.initializer;
        if (!initializer) {
            this._element.style.backgroundColor = '#f4f4f4';
        }

        var type = connection.type;
        if (type || initializer) {
            this._expander = document.createElement('div');
            this._expander.className = 'node-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', (e) => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        var id = this._connection.id || '';
        this._hasId = id ? true : false;
        if (initializer && !this._hasId) {
            var kindLine = document.createElement('div');
            kindLine.className = 'node-view-item-value-line';
            kindLine.innerHTML = 'kind: <b>' + initializer.kind + '</b>';
            this._element.appendChild(kindLine);
        }
        else {
            var idLine = document.createElement('div');
            idLine.className = 'node-view-item-value-line';
            id = this._connection.id.split('@').shift();
            id = id || ' ';
            idLine.innerHTML = '<span class=\'node-view-item-value-line-content\'>id: <b>' + id + '</b></span>';
            this._element.appendChild(idLine);
        }
    }

    get element() {
        return this._element;
    }

    toggle() {
        if (this._expander) {
            if (this._expander.innerText == '+') {
                this._expander.innerText = '-';
    
                var initializer = this._connection.initializer;
                if (initializer && this._hasId) {
                    var kind = initializer.kind;
                    if (kind) {
                        var kindLine = document.createElement('div');
                        kindLine.className = 'node-view-item-value-line-border';
                        kindLine.innerHTML = 'kind: ' + '<b>' + kind + '</b>';
                        this._element.appendChild(kindLine);
                    }
                }
    
                var type = this._connection.type;
                if (type) {
                    var typeLine = document.createElement('div');
                    typeLine.className = 'node-view-item-value-line-border';
                    typeLine.innerHTML = 'type: ' + '<code><b>' + type + '</b></code>';
                    this._element.appendChild(typeLine);
                }
    
                if (initializer) {
                    var quantization = initializer.quantization;
                    if (quantization) {
                        var quantizationLine = document.createElement('div');
                        quantizationLine.className = 'node-view-item-value-line-border';
                        quantizationLine.innerHTML = 'quantization: ' + '<code><b>' + quantization + '</b></code>';
                        this._element.appendChild(quantizationLine);   
                    }
                    var reference = initializer.reference;
                    if (reference) {
                        var referenceLine = document.createElement('div');
                        referenceLine.className = 'node-view-item-value-line-border';
                        referenceLine.innerHTML = 'reference: ' + '<b>' + reference + '</b>';
                        this._element.appendChild(referenceLine);   
                    }
                    var value = initializer.value;
                    if (value) {
                        var valueLine = document.createElement('div');
                        valueLine.className = 'node-view-item-value-line-border';
                        valueLine.innerHTML = '<pre>' + value + '</pre>';
                        this._element.appendChild(valueLine);
                    }   
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
}

class ModelView {

    constructor(model) {
        this._model = model;
        this._elements = [];

        this._model.properties.forEach((property) => {
            this.addProperty(property.name, new ValueContentView(property.value));
        });

        var graphs = this._model.graphs;
        graphs.forEach((graph, index) => {

            var name = graph.name ? ("'" + graph.name + "'") : ('(' + index.toString() + ')');

            var graphTitleElement = document.createElement('div');
            graphTitleElement.className = 'node-view-title';
            graphTitleElement.style.marginTop = '16px';
            graphTitleElement.innerText = 'Graph';
            if (graphs.length > 1) {
                graphTitleElement.innerText += " " + name;
                graphTitleElement.innerText += ' ';
                var graphButton = document.createElement('a');
                graphButton.className = 'node-view-documentation-button';
                graphButton.id = graph.name;
                graphButton.innerText = '\u21a9';
                graphButton.addEventListener('click', (e) => {
                    this.raise('update-active-graph', e.target.id);
                });
                graphTitleElement.appendChild(graphButton);
            }
            this._elements.push(graphTitleElement);
    
            if (graph.name) {
                this.addProperty('name', new ValueContentView(graph.name));
            }
            if (graph.version) {
                this.addProperty('version', new ValueContentView(graph.version));
            }
            if (graph.type) {
                this.addProperty('type', new ValueContentView(graph.type));                
            }
            if (graph.tags) {
                this.addProperty('tags', new ValueContentView(graph.tags));
            }
            if (graph.description) {
                this.addProperty('description', new ValueContentView(graph.description));                
            }

            if (graph.operators) {
                var item = new NameValueView('operators', new GraphOperatorListView(graph.operators));
                this._elements.push(item.element);
            }

            if (graph.inputs.length > 0) {
                this.addHeader('Inputs');
                graph.inputs.forEach((input) => {
                    this.addArgument(input.name, input);
                });
            }

            if (graph.outputs.length > 0) {
                this.addHeader('Outputs');
                graph.outputs.forEach((output) => {
                    this.addArgument(output.name, output);
                });
            }
        });
    }

    get elements() {
        return this._elements;
    }

    addHeader(title) {
        var headerElement = document.createElement('div');
        headerElement.className = 'node-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    addProperty(name, value) {
        var item = new NameValueView(name, value);
        this._elements.push(item.element);
    }

    addArgument(name, argument) {
        var item = new NameValueView(name, new GraphArgumentView(argument));
        this._elements.push(item.element);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    raise(event, data) {
        if (this._events && this._events[event]) {
            this._events[event].forEach((callback) => {
                callback(this, data);
            });
        }
    }
}

class GraphOperatorListView {

    constructor(operators) {

        this._element = document.createElement('div');
        this._element.className = 'node-view-item-value';

        var count = 0;
        this._list = [];
        Object.keys(operators).forEach((operator) => {
            this._list.push({ name: operator, count: operators[operator] });
            count += operators[operator];
        });
        this._list = this._list.sort((a, b) => { return (a.name > b.name) - (a.name < b.name); });
        this._list = this._list.map((item) => { return item.name + ': ' + item.count.toString(); });

        this._expander = document.createElement('div');
        this._expander.className = 'node-view-item-value-expander';
        this._expander.innerText = '+';
        this._expander.addEventListener('click', (e) => {
            this.toggle();
        });

        this._element.appendChild(this._expander);

        var countLine = document.createElement('div');
        countLine.className = 'node-view-item-value-line';
        countLine.innerHTML = '<code>' + 'Total: ' + count.toString() + '<code>';
        this._element.appendChild(countLine);
    }

    get elements() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander) {
            if (this._expander.innerText == '+') {
                this._expander.innerText = '-';
    
                var valueLine = document.createElement('div');
                valueLine.className = 'node-view-item-value-line-border';
                valueLine.innerHTML = '<pre>' + this._list.join('\n') + '</pre>';
                this._element.appendChild(valueLine);
            }
            else {
                this._expander.innerText = '+';
                while (this._element.childElementCount > 2) {
                    this._element.removeChild(this._element.lastChild);
                }
            }
        }
    }
}

class GraphArgumentView {

    constructor(argument) {
        this._argument = argument;
        this._element = document.createElement('div');
        this._element.className = 'node-view-item-value';

        if (argument.description) {
            this._expander = document.createElement('div');
            this._expander.className = 'node-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', (e) => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        var type = this._argument.type || '?';
        var typeLine = document.createElement('div');
        typeLine.className = 'node-view-item-value-line';
        typeLine.innerHTML = '<code>' + type.replace('<', '&lt;').replace('>', '&gt;') + '</code>';
        this._element.appendChild(typeLine);

        if (argument.description) {
            this.toggle();
        }
    }

    get elements() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander.innerText == '+') {
            this._expander.innerText = '-';

            var typeLine = document.createElement('div');
            typeLine.className = 'node-view-item-value-line-border';
            typeLine.innerHTML = '<code>' + this._argument.description + '<code>';
            this._element.appendChild(typeLine);
        }
        else {
            this._expander.innerText = '+';
            while (this._element.childElementCount > 2) {
                this._element.removeChild(this._element.lastChild);
            }
        }
    }
}


