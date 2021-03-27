/* jshint esversion: 6 */

var sidebar = sidebar || {};
var base = base || require('./base');

sidebar.Sidebar = class {

    constructor(host, id) {
        this._host = host;
        this._id = id ? ('-' + id) : '';
        this._stack = [];
        this._closeSidebarHandler = () => {
            this._pop();
        };
        this._closeSidebarKeyDownHandler = (e) => {
            if (e.keyCode == 27) {
                e.preventDefault();
                this._pop();
            }
        };
    }

    _getElementById(id) {
        return this._host.document.getElementById(id + this._id);
    }

    open(content, title) {
        this.close();
        this.push(content, title);
    }

    close() {
        this._deactivate();
        this._stack = [];
        this._hide();
    }

    push(content, title) {
        const item = { title: title, content: content };
        this._stack.push(item);
        this._activate(item);
    }

    _pop() {
        this._deactivate();
        if (this._stack.length > 0) {
            this._stack.pop();
        }
        if (this._stack.length > 0) {
            this._activate(this._stack[this._stack.length - 1]);
        }
        else {
            this._hide();
        }
    }

    _hide() {
        const sidebarElement = this._getElementById('sidebar');
        if (sidebarElement) {
            sidebarElement.style.width = '0';
        }
        const graphElement = this._getElementById('graph');
        if (graphElement) {
            graphElement.style.marginRight = '0';
        }
    }

    _deactivate() {
        const sidebarElement = this._getElementById('sidebar');
        if (sidebarElement) {
            const closeButton = this._getElementById('sidebar-closebutton');
            if (closeButton) {
                closeButton.removeEventListener('click', this._closeSidebarHandler);
                closeButton.style.color = '#f8f8f8';
            }

            this._host.document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
    }

    _activate(item) {
        const width = 'min(calc(100vw * 0.6), 500px)';
        const sidebarElement = this._getElementById('sidebar');
        if (sidebarElement) {
            sidebarElement.innerHTML = '';

            const titleElement = this._host.document.createElement('h1');
            titleElement.classList.add('sidebar-title');
            titleElement.innerHTML = item.title ? item.title.toUpperCase() : '';
            sidebarElement.appendChild(titleElement);

            const closeButton = this._host.document.createElement('a');
            closeButton.classList.add('sidebar-closebutton');
            closeButton.setAttribute('id', 'sidebar-closebutton');
            closeButton.setAttribute('href', 'javascript:void(0)');
            closeButton.innerHTML = '&times;';
            closeButton.addEventListener('click', this._closeSidebarHandler);
            sidebarElement.appendChild(closeButton);

            const contentElement = this._host.document.createElement('div');
            contentElement.classList.add('sidebar-content');
            contentElement.setAttribute('id', 'sidebar-content');
            sidebarElement.appendChild(contentElement);

            if (typeof content == 'string') {
                contentElement.innerHTML = item.content;
            }
            else if (item.content instanceof Array) {
                for (const element of item.content) {
                    contentElement.appendChild(element);
                }
            }
            else {
                contentElement.appendChild(item.content);
            }

            sidebarElement.style.width = width;
            this._host.document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
        const graphElement = this._getElementById('graph');
        if (graphElement) {
            graphElement.style.marginRight = width;
        }
    }
};

sidebar.NodeSidebar = class {

    constructor(host, node) {
        this._host = host;
        this._node = node;
        this._elements = [];
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        if (node.type) {
            let showDocumentation = null;
            if (node.metadata) {
                showDocumentation = {};
                showDocumentation.text = '?';
                showDocumentation.callback = () => {
                    this._raise('show-documentation', null);
                };
            }
            this._addProperty('type', new sidebar.ValueTextView(this._host, node.type, showDocumentation));
        }

        if (node.name) {
            this._addProperty('name', new sidebar.ValueTextView(this._host, node.name));
        }

        if (node.location) {
            this._addProperty('location', new sidebar.ValueTextView(this._host, node.location));
        }

        if (node.domain) {
            this._addProperty('domain', new sidebar.ValueTextView(this._host, node.domain));
        }

        if (node.description) {
            this._addProperty('description', new sidebar.ValueTextView(this._host, node.description));
        }

        if (node.device) {
            this._addProperty('device', new sidebar.ValueTextView(this._host, node.device));
        }

        const attributes = node.attributes;
        if (attributes && attributes.length > 0) {
            const sortedAttributes = node.attributes.slice();
            sortedAttributes.sort((a, b) => {
                const au = a.name.toUpperCase();
                const bu = b.name.toUpperCase();
                return (au < bu) ? -1 : (au > bu) ? 1 : 0;
            });
            this._addHeader('Attributes');
            for (const attribute of sortedAttributes) {
                this._addAttribute(attribute.name, attribute);
            }
        }

        const inputs = node.inputs;
        if (inputs && inputs.length > 0) {
            this._addHeader('Inputs');
            for (const input of inputs) {
                this._addInput(input.name, input);
            }
        }

        const outputs = node.outputs;
        if (outputs && outputs.length > 0) {
            this._addHeader('Outputs');
            for (const output of outputs) {
                this._addOutput(output.name, output);
            }
        }

        const separator = this._host.document.createElement('div');
        separator.className = 'sidebar-view-separator';
        this._elements.push(separator);
    }

    render() {
        return this._elements;
    }

    _addHeader(title) {
        const headerElement = this._host.document.createElement('div');
        headerElement.className = 'sidebar-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    _addProperty(name, value) {
        const item = new sidebar.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    _addAttribute(name, attribute) {
        const item = new sidebar.NameValueView(this._host, name, new NodeAttributeView(this._host, attribute));
        this._attributes.push(item);
        this._elements.push(item.render());
    }

    _addInput(name, input) {
        if (input.arguments.length > 0) {
            const view = new sidebar.ParameterView(this._host, input);
            view.on('export-tensor', (sender, tensor) => {
                this._raise('export-tensor', tensor);
            });
            view.on('error', (sender, tensor) => {
                this._raise('error', tensor);
            });
            const item = new sidebar.NameValueView(this._host, name, view);
            this._inputs.push(item);
            this._elements.push(item.render());
        }
    }

    _addOutput(name, output) {
        if (output.arguments.length > 0) {
            const item = new sidebar.NameValueView(this._host, name, new sidebar.ParameterView(this._host, output));
            this._outputs.push(item);
            this._elements.push(item.render());
        }
    }

    toggleInput(name) {
        for (const input of this._inputs) {
            if (name == input.name) {
                input.toggle();
            }
        }
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    static formatAttributeValue(value, type, quote) {
        if (typeof value === 'function') {
            return value();
        }
        if (value && (value instanceof base.Int64 || value instanceof base.Uint64)) {
            return value.toString();
        }
        if (Number.isNaN(value)) {
            return 'NaN';
        }
        switch (type) {
            case 'shape':
                return value ? value.toString() : '(null)';
            case 'shape[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'graph':
                return value ? value.toString() : '(null)';
            case 'graph[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'tensor':
                if (value && value.type && value.type.shape && value.type.shape.dimensions && value.type.shape.dimensions.length == 0) {
                    return value.toString();
                }
                return '[...]';
        }
        if (typeof value === 'string' && (!type || type != 'string')) {
            return quote ? '"' + value + '"' : value;
        }
        if (Array.isArray(value)) {
            if (value.length == 0) {
                return quote ? '[]' : '';
            }
            let ellipsis = false;
            if (value.length > 1000) {
                value = value.slice(0, 1000);
                ellipsis = true;
            }
            const itemType = (type && type.endsWith('[]')) ? type.substring(0, type.length - 2) : null;
            const array = value.map((item) => {
                if (item && (item instanceof base.Int64 || item instanceof base.Uint64)) {
                    return item.toString();
                }
                if (Number.isNaN(item)) {
                    return 'NaN';
                }
                const quote = !itemType || itemType === 'string';
                return sidebar.NodeSidebar.formatAttributeValue(item, itemType, quote);
            });
            if (ellipsis) {
                array.push('\u2026');
            }
            return quote ? [ '[', array.join(', '), ']' ].join(' ') : array.join(', ');
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
        const list = [];
        const keys = Object.keys(value).filter((key) => !key.startsWith('__') && !key.endsWith('__'));
        if (keys.length == 1) {
            list.push(sidebar.NodeSidebar.formatAttributeValue(value[Object.keys(value)[0]], null, true));
        }
        else {
            for (const key of keys) {
                list.push(key + ': ' + sidebar.NodeSidebar.formatAttributeValue(value[key], null, true));
            }
        }
        let objectType = value.__type__;
        if (!objectType && value.constructor.name && value.constructor.name !== 'Object') {
            objectType = value.constructor.name;
        }
        if (objectType) {
            return objectType + (list.length == 0 ? '()' : [ '(', list.join(', '), ')' ].join(''));
        }
        switch (list.length) {
            case 0:
                return quote ? '()' : '';
            case 1:
                return list[0];
            default:
                return quote ? [ '(', list.join(', '), ')' ].join(' ') : list.join(', ');
        }
    }
};

sidebar.NameValueView = class {

    constructor(host, name, value) {
        this._host = host;
        this._name = name;
        this._value = value;

        const nameElement = this._host.document.createElement('div');
        nameElement.className = 'sidebar-view-item-name';

        const nameInputElement = this._host.document.createElement('input');
        nameInputElement.setAttribute('type', 'text');
        nameInputElement.setAttribute('value', name);
        nameInputElement.setAttribute('title', name);
        nameInputElement.setAttribute('readonly', 'true');
        nameElement.appendChild(nameInputElement);

        const valueElement = this._host.document.createElement('div');
        valueElement.className = 'sidebar-view-item-value-list';

        for (const element of value.render()) {
            valueElement.appendChild(element);
        }

        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item';
        this._element.appendChild(nameElement);
        this._element.appendChild(valueElement);
    }

    get name() {
        return this._name;
    }

    render() {
        return this._element;
    }

    toggle() {
        this._value.toggle();
    }
};

sidebar.SelectView = class {

    constructor(host, values, selected) {
        this._host = host;
        this._elements = [];

        const selectElement = this._host.document.createElement('select');
        selectElement.setAttribute('class', 'sidebar-view-item-select');
        selectElement.addEventListener('change', (e) => {
            this._raise('change', e.target.value);
        });
        this._elements.push(selectElement);

        for (const value of values) {
            const optionElement = this._host.document.createElement('option');
            optionElement.innerText = value;
            if (value == selected) {
                optionElement.setAttribute('selected', 'selected');
            }
            selectElement.appendChild(optionElement);
        }
    }

    render() {
        return this._elements;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
};

sidebar.ValueTextView = class {

    constructor(host, value, action) {
        this._host = host;
        this._elements = [];
        const element = this._host.document.createElement('div');
        element.className = 'sidebar-view-item-value';
        this._elements.push(element);

        if (action) {
            this._action = this._host.document.createElement('div');
            this._action.className = 'sidebar-view-item-value-expander';
            this._action.innerHTML = action.text;
            this._action.addEventListener('click', () => {
                action.callback();
            });
            element.appendChild(this._action);
        }

        const line = this._host.document.createElement('div');
        line.className = 'sidebar-view-item-value-line';
        line.innerText = value;
        element.appendChild(line);
    }

    render() {
        return this._elements;
    }

    toggle() {
    }
};

class NodeAttributeView {

    constructor(host, attribute) {
        this._host = host;
        this._attribute = attribute;
        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        if (attribute.type) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }
        let value = sidebar.NodeSidebar.formatAttributeValue(this._attribute.value, this._attribute.type);
        if (value && value.length > 1000) {
            value = value.substring(0, 1000) + '\u2026';
        }
        if (value && typeof value === 'string') {
            value = value.split('<').join('&lt;').split('>').join('&gt;');
        }
        const valueLine = this._host.document.createElement('div');
        valueLine.className = 'sidebar-view-item-value-line';
        valueLine.innerHTML = (value ? value : '&nbsp;');
        this._element.appendChild(valueLine);
    }

    render() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander.innerText == '+') {
            this._expander.innerText = '-';

            const typeLine = this._host.document.createElement('div');
            typeLine.className = 'sidebar-view-item-value-line-border';
            const type = this._attribute.type;
            const value = this._attribute.value;
            if (type == 'tensor' && value && value.type) {
                typeLine.innerHTML = 'type: ' + '<code><b>' + value.type.toString() + '</b></code>';
                this._element.appendChild(typeLine);
            }
            else {
                typeLine.innerHTML = 'type: ' + '<code><b>' + this._attribute.type + '</b></code>';
                this._element.appendChild(typeLine);
            }

            const description = this._attribute.description;
            if (description) {
                const descriptionLine = this._host.document.createElement('div');
                descriptionLine.className = 'sidebar-view-item-value-line-border';
                descriptionLine.innerHTML = description;
                this._element.appendChild(descriptionLine);
            }

            if (this._attribute.type == 'tensor' && value) {
                const state = value.state;
                const valueLine = this._host.document.createElement('div');
                valueLine.className = 'sidebar-view-item-value-line-border';
                const contentLine = this._host.document.createElement('pre');
                contentLine.innerHTML = state || value.toString();
                valueLine.appendChild(contentLine);
                this._element.appendChild(valueLine);
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

sidebar.ParameterView = class {

    constructor(host, list) {
        this._list = list;
        this._elements = [];
        this._items = [];
        for (const argument of list.arguments) {
            const item = new sidebar.ArgumentView(host, argument);
            item.on('export-tensor', (sender, tensor) => {
                this._raise('export-tensor', tensor);
            });
            item.on('error', (sender, tensor) => {
                this._raise('error', tensor);
            });
            this._items.push(item);
            this._elements.push(item.render());
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
        for (const item of this._items) {
            item.toggle();
        }
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
};

sidebar.ArgumentView = class {

    constructor(host, argument) {
        this._host = host;
        this._argument = argument;

        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        const initializer = argument.initializer;
        if (initializer) {
            this._element.classList.add('sidebar-view-item-value-dark');
        }

        const quantization = argument.quantization;
        const type = argument.type;
        const location = this._argument.location !== undefined;
        if (type || initializer || quantization || location) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        let name = this._argument.name || '';
        this._hasId = name ? true : false;
        if (initializer && !this._hasId) {
            const kindLine = this._host.document.createElement('div');
            kindLine.className = 'sidebar-view-item-value-line';
            kindLine.innerHTML = 'kind: <b>' + initializer.kind + '</b>';
            this._element.appendChild(kindLine);
        }
        else {
            const nameLine = this._host.document.createElement('div');
            nameLine.className = 'sidebar-view-item-value-line';
            if (typeof name !== 'string') {
                throw new Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
            }
            name = name.split('\n').shift(); // custom argument id
            name = name || ' ';
            nameLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>name: <b>' + name + '</b></span>';
            this._element.appendChild(nameLine);
        }
    }

    render() {
        return this._element;
    }

    toggle() {
        if (this._expander) {
            if (this._expander.innerText == '+') {
                this._expander.innerText = '-';

                const initializer = this._argument.initializer;
                if (initializer && this._hasId) {
                    const kind = initializer.kind;
                    if (kind) {
                        const kindLine = this._host.document.createElement('div');
                        kindLine.className = 'sidebar-view-item-value-line-border';
                        kindLine.innerHTML = 'kind: ' + '<b>' + kind + '</b>';
                        this._element.appendChild(kindLine);
                    }
                }

                let type = null;
                let denotation = null;
                if (this._argument.type) {
                    type = this._argument.type.toString();
                    denotation = this._argument.type.denotation || null;
                }

                if (type) {
                    const typeLine = this._host.document.createElement('div');
                    typeLine.className = 'sidebar-view-item-value-line-border';
                    typeLine.innerHTML = 'type: <code><b>' + type.split('<').join('&lt;').split('>').join('&gt;') + '</b></code>';
                    this._element.appendChild(typeLine);
                }
                if (denotation) {
                    const denotationLine = this._host.document.createElement('div');
                    denotationLine.className = 'sidebar-view-item-value-line-border';
                    denotationLine.innerHTML = 'denotation: <code><b>' + denotation + '</b></code>';
                    this._element.appendChild(denotationLine);
                }

                const description = this._argument.description;
                if (description) {
                    const descriptionLine = this._host.document.createElement('div');
                    descriptionLine.className = 'sidebar-view-item-value-line-border';
                    descriptionLine.innerHTML = description;
                    this._element.appendChild(descriptionLine);
                }

                const quantization = this._argument.quantization;
                if (quantization) {
                    const quantizationLine = this._host.document.createElement('div');
                    quantizationLine.className = 'sidebar-view-item-value-line-border';
                    quantizationLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>quantization: ' + '<b>' + quantization + '</b></span>';
                    this._element.appendChild(quantizationLine);
                }

                if (this._argument.location !== undefined) {
                    const location = this._host.document.createElement('div');
                    location.className = 'sidebar-view-item-value-line-border';
                    location.innerHTML = 'location: ' + '<b>' + this._argument.location + '</b>';
                    this._element.appendChild(location);
                }

                if (initializer) {
                    const contentLine = this._host.document.createElement('pre');
                    const valueLine = this._host.document.createElement('div');
                    try {
                        const state = initializer.state;
                        if (state === null && this._host.save &&
                            initializer.type.dataType && initializer.type.dataType != '?' &&
                            initializer.type.shape && initializer.type.shape.dimensions && initializer.type.shape.dimensions.length > 0) {
                            this._saveButton = this._host.document.createElement('div');
                            this._saveButton.className = 'sidebar-view-item-value-expander';
                            this._saveButton.innerHTML = '&#x1F4BE;';
                            this._saveButton.addEventListener('click', () => {
                                this._raise('export-tensor', initializer);
                            });
                            this._element.appendChild(this._saveButton);
                        }

                        valueLine.className = 'sidebar-view-item-value-line-border';
                        contentLine.innerHTML = state || initializer.toString();
                    }
                    catch (err) {
                        contentLine.innerHTML = err.toString();
                        this._raise('error', err);
                    }
                    valueLine.appendChild(contentLine);
                    this._element.appendChild(valueLine);
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

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
};

sidebar.ModelSidebar = class {

    constructor(host, model, graph) {
        this._host = host;
        this._model = model;
        this._elements = [];

        if (this._model.format) {
            this._addProperty('format', new sidebar.ValueTextView(this._host, this._model.format));
        }
        if (this._model.producer) {
            this._addProperty('producer', new sidebar.ValueTextView(this._host, this._model.producer));
        }
        if (this._model.source) {
            this._addProperty('source', new sidebar.ValueTextView(this._host, this._model.source));
        }
        if (this._model.name) {
            this._addProperty('name', new sidebar.ValueTextView(this._host, this._model.name));
        }
        if (this._model.version) {
            this._addProperty('version', new sidebar.ValueTextView(this._host, this._model.version));
        }
        if (this._model.description) {
            this._addProperty('description', new sidebar.ValueTextView(this._host, this._model.description));
        }
        if (this._model.author) {
            this._addProperty('author', new sidebar.ValueTextView(this._host, this._model.author));
        }
        if (this._model.company) {
            this._addProperty('company', new sidebar.ValueTextView(this._host, this._model.company));
        }
        if (this._model.license) {
            this._addProperty('license', new sidebar.ValueTextView(this._host, this._model.license));
        }
        if (this._model.domain) {
            this._addProperty('domain', new sidebar.ValueTextView(this._host, this._model.domain));
        }
        if (this._model.imports) {
            this._addProperty('imports', new sidebar.ValueTextView(this._host, this._model.imports));
        }
        if (this._model.runtime) {
            this._addProperty('runtime', new sidebar.ValueTextView(this._host, this._model.runtime));
        }

        const metadata = this._model.metadata;
        if (metadata) {
            for (const property of this._model.metadata) {
                this._addProperty(property.name, new sidebar.ValueTextView(this._host, property.value));
            }
        }

        if (this._model._graphs.length > 1) {
            const graphSelector = new sidebar.SelectView(this._host, this._model.graphs.map((g) => g.name), graph.name);
            graphSelector.on('change', (sender, data) => {
                this._raise('update-active-graph', data);
            });
            this._addProperty('subgraph', graphSelector);
        }

        if (graph) {
            if (graph.version) {
                this._addProperty('version', new sidebar.ValueTextView(this._host, graph.version));
            }
            if (graph.type) {
                this._addProperty('type', new sidebar.ValueTextView(this._host, graph.type));
            }
            if (graph.tags) {
                this._addProperty('tags', new sidebar.ValueTextView(this._host, graph.tags));
            }
            if (graph.description) {
                this._addProperty('description', new sidebar.ValueTextView(this._host, graph.description));
            }

            if (graph.inputs.length > 0) {
                this._addHeader('Inputs');
                for (const input of graph.inputs) {
                    this.addArgument(input.name, input);
                }
            }

            if (graph.outputs.length > 0) {
                this._addHeader('Outputs');
                for (const output of graph.outputs) {
                    this.addArgument(output.name, output);
                }
            }
        }

        const separator = this._host.document.createElement('div');
        separator.className = 'sidebar-view-separator';
        this._elements.push(separator);
    }

    render() {
        return this._elements;
    }

    _addHeader(title) {
        const headerElement = this._host.document.createElement('div');
        headerElement.className = 'sidebar-view-header';
        headerElement.innerText = title;
        this._elements.push(headerElement);
    }

    _addProperty(name, value) {
        const item = new sidebar.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    addArgument(name, argument) {
        const view = new sidebar.ParameterView(this._host, argument);
        view.toggle();
        const item = new sidebar.NameValueView(this._host, name, view);
        this._elements.push(item.render());
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }
};

sidebar.DocumentationSidebar = class {

    constructor(host, metadata) {
        this._host = host;
        this._metadata = metadata;
    }

    render() {
        if (!this._elements) {
            this._elements = [];

            const documentation = sidebar.DocumentationSidebar.formatDocumentation(this._metadata);

            const element = this._host.document.createElement('div');
            element.setAttribute('class', 'sidebar-view-documentation');

            this._append(element, 'h1', documentation.name);

            if (documentation.summary) {
                this._append(element, 'p', documentation.summary);
            }

            if (documentation.description) {
                this._append(element, 'p', documentation.description);
            }

            if (documentation.attributes) {
                this._append(element, 'h2', 'Attributes');
                const attributes = this._append(element, 'dl');
                for (const attribute of documentation.attributes) {
                    this._append(attributes, 'dt', attribute.name + (attribute.type ? ': <tt>' + attribute.type + '</tt>' : ''));
                    this._append(attributes, 'dd', attribute.description);
                }
                element.appendChild(attributes);
            }

            if (documentation.inputs) {
                this._append(element, 'h2', 'Inputs' + (documentation.inputs_range ? ' (' + documentation.inputs_range + ')' : ''));
                const inputs = this._append(element, 'dl');
                for (const input of documentation.inputs) {
                    this._append(inputs, 'dt', input.name + (input.type ? ': <tt>' + input.type + '</tt>' : '') + (input.option ? ' (' + input.option + ')' : ''));
                    this._append(inputs, 'dd', input.description);
                }
            }

            if (documentation.outputs) {
                this._append(element, 'h2', 'Outputs' + (documentation.outputs_range ? ' (' + documentation.outputs_range + ')' : ''));
                const outputs = this._append(element, 'dl');
                for (const output of documentation.outputs) {
                    this._append(outputs, 'dt', output.name + (output.type ? ': <tt>' + output.type + '</tt>' : '') + (output.option ? ' (' + output.option + ')' : ''));
                    this._append(outputs, 'dd', output.description);
                }
            }

            if (documentation.type_constraints) {
                this._append(element, 'h2', 'Type Constraints');
                const type_constraints = this._append(element, 'dl');
                for (const type_constraint of documentation.type_constraints) {
                    this._append(type_constraints, 'dt', type_constraint.type_param_str + ': ' + type_constraint.allowed_type_strs.map((item) => '<tt>' + item + '</tt>').join(', '));
                    this._append(type_constraints, 'dd', type_constraint.description);
                }
            }

            if (documentation.examples) {
                this._append(element, 'h2', 'Examples');
                for (const example of documentation.examples) {
                    this._append(element, 'h3', example.summary);
                    this._append(element, 'pre', example.code);
                }
            }

            if (documentation.references) {
                this._append(element, 'h2', 'References');
                const references = this._append(element, 'ul');
                for (const reference of documentation.references) {
                    this._append(references, 'li', reference.description);
                }
            }

            if (documentation.domain && documentation.version && documentation.support_level) {
                this._append(element, 'h2', 'Support');
                this._append(element, 'dl', 'In domain <tt>' + documentation.domain + '</tt> since version <tt>' + documentation.version + '</tt> at support level <tt>' + documentation.support_level + '</tt>.');
            }

            if (!this._host.browser) {
                element.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        const link = e.target.href;
                        if (link.startsWith('http://') || link.startsWith('https://')) {
                            e.preventDefault();
                            this._raise('navigate', { link: link });
                        }
                    }
                });
            }

            this._elements = [ element ];

            const separator = this._host.document.createElement('div');
            separator.className = 'sidebar-view-separator';
            this._elements.push(separator);
        }
        return this._elements;
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    _append(parent, type, content) {
        const element = this._host.document.createElement(type);
        if (content) {
            element.innerHTML = content;
        }
        parent.appendChild(element);
        return element;
    }

    static formatDocumentation(data) {
        if (data) {
            data = JSON.parse(JSON.stringify(data));
            const generator = new markdown.Generator();
            if (data.summary) {
                data.summary = generator.html(data.summary);
            }
            if (data.description) {
                data.description = generator.html(data.description);
            }
            if (data.attributes) {
                for (const attribute of data.attributes) {
                    if (attribute.description) {
                        attribute.description = generator.html(attribute.description);
                    }
                }
            }
            if (data.inputs) {
                for (const input of data.inputs) {
                    if (input.description) {
                        input.description = generator.html(input.description);
                    }
                }
            }
            if (data.outputs) {
                for (const output of data.outputs) {
                    if (output.description) {
                        output.description = generator.html(output.description);
                    }
                }
            }
            if (data.references) {
                for (const reference of data.references) {
                    if (reference) {
                        reference.description = generator.html(reference.description);
                    }
                }
            }
            return data;
        }
        return '';
    }
};

sidebar.FindSidebar = class {

    constructor(host, element, graph) {
        this._host = host;
        this._graphElement = element;
        this._graph = graph;
        this._contentElement = this._host.document.createElement('div');
        this._contentElement.setAttribute('class', 'sidebar-view-find');
        this._searchElement = this._host.document.createElement('input');
        this._searchElement.setAttribute('id', 'search');
        this._searchElement.setAttribute('type', 'text');
        this._searchElement.setAttribute('placeholder', 'Search...');
        this._searchElement.setAttribute('style', 'width: 100%');
        this._searchElement.addEventListener('input', (e) => {
            this.update(e.target.value);
            this._raise('search-text-changed', e.target.value);
        });
        this._resultElement = this._host.document.createElement('ol');
        this._resultElement.addEventListener('click', (e) => {
            this.select(e);
        });
        this._contentElement.appendChild(this._searchElement);
        this._contentElement.appendChild(this._resultElement);
    }

    on(event, callback) {
        this._events = this._events || {};
        this._events[event] = this._events[event] || [];
        this._events[event].push(callback);
    }

    _raise(event, data) {
        if (this._events && this._events[event]) {
            for (const callback of this._events[event]) {
                callback(this, data);
            }
        }
    }

    select(e) {
        const selection = [];
        const id = e.target.id;

        const nodesElement = this._graphElement.getElementById('nodes');
        let nodeElement = nodesElement.firstChild;
        while (nodeElement) {
            if (nodeElement.id == id) {
                selection.push(nodeElement);
            }
            nodeElement = nodeElement.nextSibling;
        }

        const edgePathsElement = this._graphElement.getElementById('edge-paths');
        let edgePathElement = edgePathsElement.firstChild;
        while (edgePathElement) {
            if (edgePathElement.id == id) {
                selection.push(edgePathElement);
            }
            edgePathElement = edgePathElement.nextSibling;
        }

        let initializerElement = this._graphElement.getElementById(id);
        if (initializerElement) {
            while (initializerElement.parentElement) {
                initializerElement = initializerElement.parentElement;
                if (initializerElement.id && initializerElement.id.startsWith('node-')) {
                    selection.push(initializerElement);
                    break;
                }
            }
        }

        if (selection.length > 0) {
            this._raise('select', selection);
        }
    }

    focus(searchText) {
        this._searchElement.focus();
        this._searchElement.value = '';
        this._searchElement.value = searchText;
        this.update(searchText);
    }

    update(searchText) {
        while (this._resultElement.lastChild) {
            this._resultElement.removeChild(this._resultElement.lastChild);
        }

        const text = searchText.toLowerCase();

        const nodeMatches = new Set();
        const edgeMatches = new Set();

        for (const node of this._graph.nodes) {

            const initializers = [];

            for (const input of node.inputs) {
                for (const argument of input.arguments) {
                    const match =
                        (argument.name && argument.name.toLowerCase().indexOf(text) != -1) ||
                        (argument.type && argument.type.dataType && argument.type.dataType.toLowerCase() === text) ||
                        (argument.type && argument.type.shape && argument.type.shape.dimensions && argument.type.shape.dimensions.some((dimension) => dimension && dimension.toString() === text)) ||
                        (argument.type && argument.type.shape && argument.type.shape.dimensions && (argument.type.shape.toString() === text || argument.type.shape.toString() === '[' + text + ']'));
                    if (match && !edgeMatches.has(argument.name)) {
                        if (!argument.initializer) {
                            const inputItem = this._host.document.createElement('li');
                            inputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                            inputItem.id = 'edge-' + argument.name;
                            this._resultElement.appendChild(inputItem);
                            edgeMatches.add(argument.name);
                        }
                        else {
                            initializers.push(argument);
                        }
                    }
                }
            }

            const name = node.name;
            const operator = node.type;
            if (!nodeMatches.has(name) && name &&
                ((name.toLowerCase().indexOf(text) != -1) ||
                (operator && operator.toLowerCase().indexOf(text) != -1))) {
                const nameItem = this._host.document.createElement('li');
                nameItem.innerText = '\u25A2 ' + node.name;
                nameItem.id = 'node-name-' + node.name;
                this._resultElement.appendChild(nameItem);
                nodeMatches.add(node.name);
            }

            for (const argument of initializers) {
                if (argument.name) {
                    const initializeItem = this._host.document.createElement('li');
                    initializeItem.innerText = '\u25A0 ' + argument.name;
                    initializeItem.id = 'initializer-' + argument.name;
                    this._resultElement.appendChild(initializeItem);
                }
            }
        }

        for (const node of this._graph.nodes) {
            for (const output of node.outputs) {
                for (const argument of output.arguments) {
                    if (argument.name && argument.name.toLowerCase().indexOf(text) != -1 && !edgeMatches.has(argument.name)) {
                        const outputItem = this._host.document.createElement('li');
                        outputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                        outputItem.id = 'edge-' + argument.name;
                        this._resultElement.appendChild(outputItem);
                        edgeMatches.add(argument.name);
                    }
                }
            }
        }

        this._resultElement.style.display = this._resultElement.childNodes.length != 0 ? 'block' : 'none';
    }

    get content() {
        return this._contentElement;
    }
};

const markdown = {};

markdown.Generator = class {

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
        this._emEndUndRegExp = /[^\s]_(?!_)(?:(?=[!"#$%&'()+\-.,/:;<=>?@[\]`{|}~*\s])|$)/g,
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
    }

    html(source) {
        const tokens = [];
        const links = new Map();
        this._tokenize(source.replace(/\r\n|\r/g, '\n').replace(/\t/g, '    '), tokens, links, true);
        this._tokenizeBlock(tokens, links);
        const slugs = new Map();
        const result = this._render(tokens, slugs, true);
        return result;
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
                    lastToken.text += '\n' + match[0].trimRight();
                }
                else {
                    const text = match[0].replace(/^ {4}/gm, '').replace(/\n*$/, '');
                    tokens.push({ type: 'code', text: text });
                }
                continue;
            }
            match = this._fencesRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const language = match[2] ? match[2].trim() : match[2];
                let text = match[3] || '';
                const matchIndent = match[0].match(/^(\s+)(?:```)/);
                if (matchIndent !== null) {
                    const indent = matchIndent[1];
                    text = text.split('\n').map(node => {
                        const match = node.match(/^\s+/);
                        return (match !== null && match[0].length >= indent.length) ? node.slice(indent.length) : node;
                    }).join('\n');
                }
                tokens.push({ type: 'code', language: language, text: text });
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
                    const token = { type: 'table', header: header, align: align, cells: cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        }
                        else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        }
                        else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        }
                        else {
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
                tokens.push({ type: 'blockquote', text: text, tokens: this._tokenize(text, [], links, top) });
                continue;
            }
            match = this._listRegExp.exec(source);
            if (match) {
                let raw = match[0];
                const bull = match[2];
                const ordered = bull.length > 1;
                const parent = bull[bull.length - 1] === ')';
                const list = { type: 'list', raw: raw, ordered: ordered, start: ordered ? +bull.slice(0, -1) : '', loose: false, items: [] };
                const itemMatch = match[0].match(this._itemRegExp);
                let next = false;
                const length = itemMatch.length;
                for (let i = 0; i < length; i++) {
                    let item = itemMatch[i];
                    raw = item;
                    let space = item.length;
                    item = item.replace(/^ *([*+-]|\d+[.)]) ?/, '');
                    if (~item.indexOf('\n ')) {
                        space -= item.length;
                        item = item.replace(new RegExp('^ {1,' + space + '}', 'gm'), '');
                    }
                    if (i !== length - 1) {
                        const bullet = this._bulletRegExp.exec(itemMatch[i + 1])[0];
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
                    let checked = undefined;
                    if (task) {
                        checked = item[1] !== ' ';
                        item = item.replace(/^\[[ xX]\] +/, '');
                    }
                    list.items.push({ type: 'list_item', raw, task: task, checked: checked, loose: loose, text: item });
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
                    const token = { type: 'table', header: header, align: align, cells: cells, raw: match[0] };
                    for (let i = 0; i < token.align.length; i++) {
                        if (/^ *-+: *$/.test(token.align[i])) {
                            token.align[i] = 'right';
                        }
                        else if (/^ *:-+: *$/.test(token.align[i])) {
                            token.align[i] = 'center';
                        }
                        else if (/^ *:-+ *$/.test(token.align[i])) {
                            token.align[i] = 'left';
                        }
                        else {
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
                    lastToken.text += '\n' + match[0];
                }
                else {
                    tokens.push({ type: 'text', text: match[0] });
                }
                continue;
            }
            throw new Error("Unexpected '" + source.charCodeAt(0) + "'.");
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
                        maskedSource = maskedSource.slice(0, match.index) + '[' + 'a'.repeat(match[0].length - 2) + ']' + maskedSource.slice(this._reflinkSearchRegExp.lastIndex);
                    }
                    continue;
                }
                break;
            }
        }
        while (maskedSource) {
            const match = this._blockSkipRegExp.exec(maskedSource);
            if (match) {
                maskedSource = maskedSource.slice(0, match.index) + '[' + 'a'.repeat(match[0].length - 2) + ']' + maskedSource.slice(this._blockSkipRegExp.lastIndex);
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
                }
                else if (inLink && /^<\/a>/i.test(match[0])) {
                    inLink = false;
                }
                if (!inRawBlock && /^<(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = true;
                }
                else if (inRawBlock && /^<\/(pre|code|kbd|script)(\s|>)/i.test(match[0])) {
                    inRawBlock = false;
                }
                tokens.push({ type: 'html', raw: match[0], text: match[0] });
                continue;
            }
            match = this._linkRegExp.exec(source);
            if (match) {
                let index = -1;
                const ref = match[2];
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
                    tokens.push({ type: 'text', text: text });
                }
                else {
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
                let cap;
                while ((match = endReg.exec(masked)) != null) {
                    cap = this._strongMiddleRegExp.exec(masked.slice(0, match.index + 3));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.substring(2, cap[0].length - 2);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'strong', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._emStartRegExp.exec(source);
            if (match && (!match[1] || (match[1] && (prevChar === '' || this._punctuationRegExp.exec(prevChar))))) {
                const masked = maskedSource.slice(-1 * source.length);
                const endReg = match[0] === '*' ? this._emEndAstRegExp : this._emEndUndRegExp;
                endReg.lastIndex = 0;
                let cap;
                while ((match = endReg.exec(masked)) != null) {
                    cap = this._emMiddleRegExp.exec(masked.slice(0, match.index + 2));
                    if (cap) {
                        break;
                    }
                }
                if (cap) {
                    const text = source.slice(1, cap[0].length - 1);
                    source = source.substring(cap[0].length);
                    tokens.push({ type: 'em', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                    continue;
                }
            }
            match = this._codespanRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                let text = match[2].replace(/\n/g, ' ');
                if (/[^ ]/.test(text) && text.startsWith(' ') && text.endsWith(' ')) {
                    text = text.substring(1, text.length - 1);
                }
                tokens.push({ type: 'codespan', text: this._encode(text) });
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
                source = source.substring(match[0].length);
                const text = match[1];
                tokens.push({ type: 'del', text: text, tokens: this._tokenizeInline(text, links, inLink, inRawBlock, '') });
                continue;
            }
            match = this._autolinkRegExp.exec(source);
            if (match) {
                source = source.substring(match[0].length);
                const text = this._escape(match[1]);
                const href = match[2] === '@' ? 'mailto:' + text : text;
                tokens.push({ type: 'link', text: text, href: href, tokens: [ { type: 'text', raw: text, text } ] });
                continue;
            }
            if (!inLink) {
                match = this._urlRegExp.exec(source);
                if (match) {
                    const email = match[2] === '@';
                    if (!email) {
                        let prevCapZero;
                        do {
                            prevCapZero = match[0];
                            match[0] = this._backpedalRegExp.exec(match[0])[0];
                        } while (prevCapZero !== match[0]);
                    }
                    const text = this._escape(match[0]);
                    const href = email ? ('mailto:' + text) : (match[1] === 'www.' ? 'http://' + text : text);
                    source = source.substring(match[0].length);
                    tokens.push({ type: 'link', text: text, href: href, tokens: [ { type: 'text', text: text } ] });
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
            throw new Error("Unexpected '" + source.charCodeAt(0) + "'.");
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
            }
        }
    }

    _render(tokens, slugs, top) {
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
                    const id = this._slug(slugs, this._renderInline(token.tokens, true));
                    html += '<h' + level + ' id="' + id + '">' + this._renderInline(token.tokens) + '</h' + level + '>\n';
                    continue;
                }
                case 'code': {
                    const code = token.text;
                    const language = (token.language || '').match(/\S*/)[0];
                    html += '<pre><code' + (language ? ' class="' + 'language-' + this._encode(language) + '"' : '') + '>' + (token.escaped ? code : this._encode(code)) + '</code></pre>\n';
                    continue;
                }
                case 'table': {
                    let header = '';
                    let cell = '';
                    for (let j = 0; j < token.header.length; j++) {
                        const content = this._renderInline(token.tokens.header[j]);
                        const align = token.align[j];
                        cell += '<th' + (align ? ' align="' + align + '"' : '') + '>' + content + '</th>\n';
                    }
                    header += '<tr>\n' + cell + '</tr>\n';
                    let body = '';
                    for (let j = 0; j < token.cells.length; j++) {
                        const row = token.tokens.cells[j];
                        cell = '';
                        for (let k = 0; k < row.length; k++) {
                            const content = this._renderInline(row[k]);
                            const align = token.align[k];
                            cell += '<td' + (align ? ' align="' + align + '"' : '') + '>' + content + '</td>\n';
                        }
                        body += '<tr>\n' + cell + '</tr>\n';
                    }
                    html += '<table>\n<thead>\n' + header + '</thead>\n' + (body ? '<tbody>' + body + '</tbody>' : body) + '</table>\n';
                    continue;
                }
                case 'blockquote': {
                    html += '<blockquote>\n' + this._render(token.tokens, slugs, true) + '</blockquote>\n';
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
                            const checkbox = '<input ' + (item.checked ? 'checked="" ' : '') + 'disabled="" type="checkbox"' + '> ';
                            if (loose) {
                                if (item.tokens.length > 0 && item.tokens[0].type === 'text') {
                                    item.tokens[0].text = checkbox + ' ' + item.tokens[0].text;
                                    if (item.tokens[0].tokens && item.tokens[0].tokens.length > 0 && item.tokens[0].tokens[0].type === 'text') {
                                        item.tokens[0].tokens[0].text = checkbox + ' ' + item.tokens[0].tokens[0].text;
                                    }
                                }
                                else {
                                    item.tokens.unshift({ type: 'text', text: checkbox });
                                }
                            }
                            else {
                                itemBody += checkbox;
                            }
                        }
                        itemBody += this._render(item.tokens, slugs, loose);
                        body += '<li>' + itemBody + '</li>\n';
                    }
                    const type = (ordered ? 'ol' : 'ul');
                    html += '<' + type + (ordered && start !== 1 ? (' start="' + start + '"') : '') + '>\n' + body + '</' + type + '>\n';
                    continue;
                }
                case 'html': {
                    html += token.text;
                    continue;
                }
                case 'paragraph': {
                    html += '<p>' + this._renderInline(token.tokens) + '</p>\n';
                    continue;
                }
                case 'text': {
                    html += top ? '<p>' : '';
                    html += token.tokens ? this._renderInline(token.tokens) : token.text;
                    while (tokens.length > 0 && tokens[0].type === 'text') {
                        const token = tokens.shift();
                        html += '\n' + (token.tokens ? this._renderInline(token.tokens) : token.text);
                    }
                    html += top ? '</p>\n' : '';
                    continue;
                }
                default: {
                    throw new Error("Unexpected token type '" + token.type + "'.");
                }
            }
        }
        return html;
    }

    _renderInline(tokens, slug) {
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
                    const text = this._renderInline(token.tokens, slug);
                    html += slug ? text : '<a href="' + token.href + '"' + (token.title ? ' title="' + token.title + '"' : '') + ' target="_blank">' + text + '</a>';
                    break;
                }
                case 'image': {
                    html += slug ? token.text : '<img src="' + token.href + '" alt="' + token.text + '"' + (token.title ? ' title="' + token.title + '"' : '') + '>';
                    break;
                }
                case 'strong': {
                    const text = this._renderInline(token.tokens, slug);
                    html += slug ? text : '<strong>' + text + '</strong>';
                    break;
                }
                case 'em': {
                    const text = this._renderInline(token.tokens, slug);
                    html += slug ? text : '<em>' + text + '</em>';
                    break;
                }
                case 'codespan': {
                    html += slug ? token.text : '<code>' + token.text + '</code>';
                    break;
                }
                case 'br': {
                    html += slug ? '' : '<br>';
                    break;
                }
                case 'del': {
                    const text = this._renderInline(token.tokens, slug);
                    html += slug ? text : '<del>' + text + '</del>';
                    break;
                }
                default: {
                    throw new Error("Unexpected token type '" + token.type + "'.");
                }
            }
        }
        return html;
    }

    _outputLink(match, href, title) {
        title = title ? this._escape(title) : null;
        const text = match[1].replace(/\\([[\]])/g, '$1');
        return match[0].charAt(0) !== '!' ?
            { type: 'link', href: href, title: title, text: text } :
            { type: 'image', href: href, title: title, text: this._escape(text) };
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
        }
        else {
            while (cells.length < count) {
                cells.push('');
            }
        }
        return cells.map((cell) => cell.trim().replace(/\\\|/g, '|'));
    }

    _slug(slugs, value) {
        value = value.replace(/&(#(?:\d+)|(?:#x[0-9A-Fa-f]+)|(?:\w+));?/ig, (_, n) => {
            n = n.toLowerCase();
            if (n === 'colon') {
                return ':';
            }
            if (n.charAt(0) === '#') {
                return String.fromCharCode(n.charAt(1) === 'x' ? parseInt(n.substring(2), 16) : +n.substring(1));
            }
            return '';
        });
        value = value.toLowerCase().trim()
            .replace(/<[!/a-z].*?>/ig, '')
            .replace(/[\u2000-\u206F\u2E00-\u2E7F\\'!"#$%&()*+,./:;<=>?@[\]^`{|}~]/g, '')
            .replace(/\s/g, '-');
        let slug = value;
        let count = 0;
        if (slugs.has(value)) {
            count = slugs.get(value);
            do {
                count++;
                slug = value + '-' + count;
            }
            while (slugs.has(slug));
        }
        slugs.set(value, count);
        slugs.set(slug, 0);
        return slug;
    }

    _encode(text) {
        if (this._escapeTestRegExp.test(text)) {
            return text.replace(this._escapeReplaceRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return text;
    }

    _escape(text) {
        if (this._escapeTestNoEncodeRegExp.test(text)) {
            return text.replace(this._escapeReplaceNoEncodeRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return text;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Sidebar = sidebar.Sidebar;
    module.exports.ModelSidebar = sidebar.ModelSidebar;
    module.exports.NodeSidebar = sidebar.NodeSidebar;
    module.exports.DocumentationSidebar = sidebar.DocumentationSidebar;
    module.exports.FindSidebar = sidebar.FindSidebar;
}