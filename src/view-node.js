/*jshint esversion: 6 */

class NodeView {

    constructor(node, documentationHandler) {
        this._node = node;
        this._documentationHandler = documentationHandler;
        this._elements = [];
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        var operatorElement = document.createElement('div');
        operatorElement.className = 'node-view-title';
        operatorElement.innerText = node.operator + ' ';
        this._elements.push(operatorElement);

        if (node.documentation) {
            var documentationButton = document.createElement('a');
            documentationButton.className = 'node-view-documentation-button';
            documentationButton.innerText = '?';
            documentationButton.addEventListener('click', (e) => {
                this._documentationHandler();
            });
            operatorElement.appendChild(documentationButton);
        }

        if (node.name) {
            this.addProperty('name', new NodeViewItemContent(node.name));
        }

        if (node.domain) {
            this.addProperty('domain', new NodeViewItemContent(node.domain));
        }

        if (node.description) {
            this.addProperty('description', new NodeViewItemContent(node.description));
        }

        if (node.device) {
            this.addProperty('device', new NodeViewItemContent(node.device));
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
        var item = new NodeViewItem(name, value);
        this._elements.push(item.element);
    }

    addAttribute(name, attribute) {
        var item = new NodeViewItem(name, new NodeViewItemAttribute(attribute));
        this._attributes.push(item);
        this._elements.push(item.element);
    }

    addInput(name, input) {
        if (input.connections.length > 0) {
            var item = new NodeViewItem(name, new NodeViewItemList(input));
            this._inputs.push(item);
            this._elements.push(item.element);
        }
    }

    addOutput(name, output) {
        if (output.connections.length > 0) {
            var item = new NodeViewItem(name, new NodeViewItemList(output));
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
}

class NodeViewItem {
    constructor(name, value) {
        this._name = name;
        this._value = value;

        var itemName = document.createElement('div');
        itemName.className = 'node-view-item-name';

        var inputName = document.createElement('input');
        inputName.setAttribute('type', 'text');
        inputName.setAttribute('value', name);
        inputName.setAttribute('title', name);
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

class NodeViewItemContent {
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

class NodeViewItemAttribute {

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

class NodeViewItemList {

    constructor(list) {
        this._list = list;
        this._elements = [];
        this._items = [];
        list.connections.forEach((connection) => {
            var item = new NodeViewItemConnection(connection);
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

class NodeViewItemConnection {
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

        this._hasId = this._connection.id ? true : false;
        if (this._hasId) {
            var idLine = document.createElement('div');
            idLine.className = 'node-view-item-value-line';
            var id = this._connection.id.split('@').shift();
            idLine.innerHTML = '<span class=\'node-view-item-value-line-content\'>id: <b>' + id + '</b></span>';
            this._element.appendChild(idLine);
        }
        else if (initializer) {
            var kindLine = document.createElement('div');
            kindLine.className = 'node-view-item-value-line';
            kindLine.innerHTML = 'kind: <b>' + initializer.kind + '</b>';
            this._element.appendChild(kindLine);
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
