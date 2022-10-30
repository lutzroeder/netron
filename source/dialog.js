
var dialog = {};
var base = require('./base');

dialog.Sidebar = class {

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
        const sidebar = this._getElementById('sidebar');
        if (sidebar) {
            sidebar.style.width = '0px';
        }
        const container = this._getElementById('graph');
        if (container) {
            container.style.width = '100%';
            container.focus();
        }
    }

    _deactivate() {
        const sidebar = this._getElementById('sidebar');
        if (sidebar) {
            const closeButton = this._getElementById('sidebar-closebutton');
            if (closeButton) {
                closeButton.removeEventListener('click', this._closeSidebarHandler);
                closeButton.style.color = '#f8f8f8';
            }

            this._host.document.removeEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
    }

    _activate(item) {
        const sidebar = this._getElementById('sidebar');
        if (sidebar) {
            sidebar.innerHTML = '';

            const title = this._host.document.createElement('h1');
            title.classList.add('sidebar-title');
            title.innerHTML = item.title ? item.title.toUpperCase() : '';
            sidebar.appendChild(title);

            const closeButton = this._host.document.createElement('a');
            closeButton.classList.add('sidebar-closebutton');
            closeButton.setAttribute('id', 'sidebar-closebutton');
            closeButton.setAttribute('href', 'javascript:void(0)');
            closeButton.innerHTML = '&times;';
            closeButton.addEventListener('click', this._closeSidebarHandler);
            sidebar.appendChild(closeButton);

            const content = this._host.document.createElement('div');
            content.classList.add('sidebar-content');
            content.setAttribute('id', 'sidebar-content');
            sidebar.appendChild(content);

            if (typeof item.content == 'string') {
                content.innerHTML = item.content;
            }
            else if (item.content instanceof Array) {
                for (const element of item.content) {
                    content.appendChild(element);
                }
            }
            else {
                content.appendChild(item.content);
            }
            sidebar.style.width = 'min(calc(100% * 0.6), 500px)';
            this._host.document.addEventListener('keydown', this._closeSidebarKeyDownHandler);
        }
        const container = this._getElementById('graph');
        if (container) {
            container.style.width = 'max(40vw, calc(100vw - 500px))';
        }
    }
};

dialog.Control = class {

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

dialog.NodeSidebar = class extends dialog.Control {

    constructor(host, node) {
        super();
        this._host = host;
        this._node = node;
        this._elements = [];
        this._attributes = [];
        this._inputs = [];
        this._outputs = [];

        if (node.type) {
            let showDocumentation = null;
            const type = node.type;
            if (type && (type.description || type.inputs || type.outputs || type.attributes)) {
                showDocumentation = {};
                showDocumentation.text = type.nodes ? '\u0192': '?';
                showDocumentation.callback = () => {
                    this._raise('show-documentation', null);
                };
            }
            this._addProperty('type', new dialog.ValueTextView(this._host, node.type.identifier || node.type.name, showDocumentation));
            if (node.type.module) {
                this._addProperty('module', new dialog.ValueTextView(this._host, node.type.module));
            }
        }

        if (node.name) {
            this._addProperty('name', new dialog.ValueTextView(this._host, node.name));
        }

        if (node.location) {
            this._addProperty('location', new dialog.ValueTextView(this._host, node.location));
        }

        if (node.description) {
            this._addProperty('description', new dialog.ValueTextView(this._host, node.description));
        }

        if (node.device) {
            this._addProperty('device', new dialog.ValueTextView(this._host, node.device));
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
        const item = new dialog.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    _addAttribute(name, attribute) {
        const item = new dialog.AttributeView(this._host, attribute);
        item.on('show-graph', (sender, graph) => {
            this._raise('show-graph', graph);
        });
        const view = new dialog.NameValueView(this._host, name, item);
        this._attributes.push(view);
        this._elements.push(view.render());
    }

    _addInput(name, input) {
        if (input.arguments.length > 0) {
            const view = new dialog.ParameterView(this._host, input);
            view.on('export-tensor', (sender, tensor) => {
                this._raise('export-tensor', tensor);
            });
            view.on('error', (sender, tensor) => {
                this._raise('error', tensor);
            });
            const item = new dialog.NameValueView(this._host, name, view);
            this._inputs.push(item);
            this._elements.push(item.render());
        }
    }

    _addOutput(name, output) {
        if (output.arguments.length > 0) {
            const item = new dialog.NameValueView(this._host, name, new dialog.ParameterView(this._host, output));
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
};

dialog.NameValueView = class {

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

dialog.SelectView = class extends dialog.Control {

    constructor(host, values, selected) {
        super();
        this._host = host;
        this._elements = [];
        this._values = values;

        const selectElement = this._host.document.createElement('select');
        selectElement.setAttribute('class', 'sidebar-view-item-select');
        selectElement.addEventListener('change', (e) => {
            this._raise('change', this._values[e.target.selectedIndex]);
        });
        this._elements.push(selectElement);

        for (const value of values) {
            const optionElement = this._host.document.createElement('option');
            optionElement.innerText = value.name || '';
            if (value == selected) {
                optionElement.setAttribute('selected', 'selected');
            }
            selectElement.appendChild(optionElement);
        }
    }

    render() {
        return this._elements;
    }
};

dialog.ValueTextView = class {

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

        const list = Array.isArray(value) ? value : [ value ];
        let className = 'sidebar-view-item-value-line';
        for (const item of list) {
            const line = this._host.document.createElement('div');
            line.className = className;
            line.innerText = item;
            element.appendChild(line);
            className = 'sidebar-view-item-value-line-border';
        }
    }

    render() {
        return this._elements;
    }

    toggle() {
    }
};

dialog.ValueView = class extends dialog.Control {

    _bold(name, value) {
        const line = this._host.document.createElement('div');
        line.innerHTML = name + ': ' + '<b>' + value + '</b>';
        this._add(line);
    }

    _code(name, value) {
        const line = this._host.document.createElement('div');
        line.innerHTML = name + ': ' + '<code><b>' + value + '</b></code>';
        this._add(line);
    }

    _add(child) {
        child.className = this._element.childNodes.length < 2 ? 'sidebar-view-item-value-line' : 'sidebar-view-item-value-line-border';
        this._element.appendChild(child);
    }

    _tensor(value) {
        const contentLine = this._host.document.createElement('pre');
        try {
            const tensor = new dialog.Tensor(value);
            const layout = tensor.layout;
            if (layout) {
                const layouts = new Map([
                    [ 'sparse', 'Sparse' ],
                    [ 'sparse.coo', 'Sparse COO' ],
                    [ 'sparse.csr', 'Sparse CSR' ],
                    [ 'sparse.csc', 'Sparse CSC' ],
                    [ 'sparse.bsr', 'Sparse BSR' ],
                    [ 'sparse.bsc', 'Sparse BSC' ]
                ]);
                if (layouts.has(layout)) {
                    this._bold('layout', layouts.get(layout));
                }
            }
            if (tensor.layout !== '<' && tensor.layout !== '>' && tensor.layout !== '|' && tensor.layout !== 'sparse' && tensor.layout !== 'sparse.coo') {
                contentLine.innerHTML = "Tensor layout '" + tensor.layout + "' is not implemented.";
            }
            else if (tensor.empty) {
                contentLine.innerHTML = 'Tensor data is empty.';
            }
            else if (tensor.type && tensor.type.dataType === '?') {
                contentLine.innerHTML = 'Tensor data type is not defined.';
            }
            else if (tensor.type && !tensor.type.shape) {
                contentLine.innerHTML = 'Tensor shape is not defined.';
            }
            else {
                contentLine.innerHTML = tensor.toString();

                if (this._host.save &&
                    value.type.shape && value.type.shape.dimensions &&
                    value.type.shape.dimensions.length > 0) {
                    this._saveButton = this._host.document.createElement('div');
                    this._saveButton.className = 'sidebar-view-item-value-expander';
                    this._saveButton.innerHTML = '&#x1F4BE;';
                    this._saveButton.addEventListener('click', () => {
                        this._raise('export-tensor', tensor);
                    });
                    this._element.appendChild(this._saveButton);
                }
            }
        }
        catch (err) {
            contentLine.innerHTML = err.toString();
            this._raise('error', err);
        }
        const valueLine = this._host.document.createElement('div');
        valueLine.className = 'sidebar-view-item-value-line-border';
        valueLine.appendChild(contentLine);
        this._element.appendChild(valueLine);
    }
};

dialog.AttributeView = class extends dialog.ValueView {

    constructor(host, attribute) {
        super();
        this._host = host;
        this._attribute = attribute;
        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        const type = this._attribute.type;
        if (type) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }
        const value = this._attribute.value;
        switch (type) {
            case 'graph': {
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line-link';
                line.innerHTML = value.name;
                line.addEventListener('click', () => {
                    this._raise('show-graph', value);
                });
                this._element.appendChild(line);
                break;
            }
            case 'function': {
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line-link';
                line.innerHTML = value.type.name;
                line.addEventListener('click', () => {
                    this._raise('show-graph', value.type);
                });
                this._element.appendChild(line);
                break;
            }
            default: {
                let content = new dialog.Formatter(value, type).toString();
                if (content && content.length > 1000) {
                    content = content.substring(0, 1000) + '\u2026';
                }
                if (content && typeof content === 'string') {
                    content = content.split('<').join('&lt;').split('>').join('&gt;');
                }
                const line = this._host.document.createElement('div');
                line.className = 'sidebar-view-item-value-line';
                line.innerHTML = content ? content : '&nbsp;';
                this._element.appendChild(line);
            }
        }
    }

    render() {
        return [ this._element ];
    }

    toggle() {
        if (this._expander.innerText == '+') {
            this._expander.innerText = '-';

            const type = this._attribute.type;
            const value = this._attribute.value;
            const content = type == 'tensor' && value && value.type ? value.type.toString() : this._attribute.type;
            const typeLine = this._host.document.createElement('div');
            typeLine.className = 'sidebar-view-item-value-line-border';
            typeLine.innerHTML = 'type: ' + '<code><b>' + content + '</b></code>';
            this._element.appendChild(typeLine);

            const description = this._attribute.description;
            if (description) {
                const descriptionLine = this._host.document.createElement('div');
                descriptionLine.className = 'sidebar-view-item-value-line-border';
                descriptionLine.innerHTML = description;
                this._element.appendChild(descriptionLine);
            }

            if (this._attribute.type == 'tensor' && value) {
                this._tensor(value);
            }
        }
        else {
            this._expander.innerText = '+';
            while (this._element.childElementCount > 2) {
                this._element.removeChild(this._element.lastChild);
            }
        }
    }
};

dialog.ParameterView = class extends dialog.Control {

    constructor(host, list) {
        super();
        this._list = list;
        this._elements = [];
        this._items = [];
        for (const argument of list.arguments) {
            const item = new dialog.ArgumentView(host, argument);
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
};

dialog.ArgumentView = class extends dialog.ValueView {

    constructor(host, argument) {
        super();
        this._host = host;
        this._argument = argument;

        this._element = this._host.document.createElement('div');
        this._element.className = 'sidebar-view-item-value';

        const type = this._argument.type;
        const initializer = this._argument.initializer;
        const quantization = this._argument.quantization;
        const location = this._argument.location !== undefined;

        if (initializer) {
            this._element.classList.add('sidebar-view-item-value-dark');
        }

        if (type || initializer || quantization || location) {
            this._expander = this._host.document.createElement('div');
            this._expander.className = 'sidebar-view-item-value-expander';
            this._expander.innerText = '+';
            this._expander.addEventListener('click', () => {
                this.toggle();
            });
            this._element.appendChild(this._expander);
        }

        const name = this._argument.name ? this._argument.name.split('\n').shift() : ''; // custom argument id
        this._hasId = name ? true : false;
        this._hasCategory = initializer && initializer.category ? true : false;
        if (this._hasId || (!this._hasCategory && !type)) {
            this._hasId = true;
            const nameLine = this._host.document.createElement('div');
            nameLine.className = 'sidebar-view-item-value-line';
            if (typeof name !== 'string') {
                throw new Error("Invalid argument identifier '" + JSON.stringify(name) + "'.");
            }
            nameLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>name: <b>' + (name || ' ') + '</b></span>';
            this._element.appendChild(nameLine);
        }
        else if (this._hasCategory) {
            this._bold('category', initializer.category);
        }
        else if (type) {
            this._code('type', type.toString().split('<').join('&lt;').split('>').join('&gt;'));
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
                if (this._hasId && this._hasCategory) {
                    this._bold('category', initializer.category);
                }

                let type = null;
                let denotation = null;
                if (this._argument.type) {
                    type = this._argument.type.toString();
                    denotation = this._argument.type.denotation || null;
                }
                if (type && (this._hasId || this._hasCategory)) {
                    this._code('type', type.split('<').join('&lt;').split('>').join('&gt;'));
                }
                if (denotation) {
                    this._code('denotation', denotation);
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
                    const content = !Array.isArray(quantization) ? quantization : '<br><br>' + quantization.map((value) => '  ' + value).join('<br>');
                    quantizationLine.innerHTML = '<span class=\'sidebar-view-item-value-line-content\'>quantization: ' + '<b>' + content + '</b></span>';
                    this._element.appendChild(quantizationLine);
                }

                const location = this._argument.location;
                if (location !== undefined) {
                    this._bold('location', location);
                }

                if (initializer) {
                    this._tensor(initializer);
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
};

dialog.ModelSidebar = class extends dialog.Control {

    constructor(host, model, graph) {
        super();
        this._host = host;
        this._model = model;
        this._elements = [];

        if (model.format) {
            this._addProperty('format', new dialog.ValueTextView(this._host, model.format));
        }
        if (model.producer) {
            this._addProperty('producer', new dialog.ValueTextView(this._host, model.producer));
        }
        if (model.name) {
            this._addProperty('name', new dialog.ValueTextView(this._host, model.name));
        }
        if (model.version) {
            this._addProperty('version', new dialog.ValueTextView(this._host, model.version));
        }
        if (model.description) {
            this._addProperty('description', new dialog.ValueTextView(this._host, model.description));
        }
        if (model.domain) {
            this._addProperty('domain', new dialog.ValueTextView(this._host, model.domain));
        }
        if (model.imports) {
            this._addProperty('imports', new dialog.ValueTextView(this._host, model.imports));
        }
        if (model.runtime) {
            this._addProperty('runtime', new dialog.ValueTextView(this._host, model.runtime));
        }
        if (model.metadata) {
            for (const entry of model.metadata) {
                this._addProperty(entry.name, new dialog.ValueTextView(this._host, entry.value));
            }
        }
        const graphs = Array.isArray(model.graphs) ? model.graphs : [];
        if (graphs.length > 1) {
            const graphSelector = new dialog.SelectView(this._host, model.graphs, graph);
            graphSelector.on('change', (sender, data) => {
                this._raise('update-active-graph', data);
            });
            this._addProperty('subgraph', graphSelector);
        }

        if (graph) {
            if (graph.version) {
                this._addProperty('version', new dialog.ValueTextView(this._host, graph.version));
            }
            if (graph.type) {
                this._addProperty('type', new dialog.ValueTextView(this._host, graph.type));
            }
            if (graph.tags) {
                this._addProperty('tags', new dialog.ValueTextView(this._host, graph.tags));
            }
            if (graph.description) {
                this._addProperty('description', new dialog.ValueTextView(this._host, graph.description));
            }
            if (Array.isArray(graph.inputs) && graph.inputs.length > 0) {
                this._addHeader('Inputs');
                for (const input of graph.inputs) {
                    this.addArgument(input.name, input);
                }
            }
            if (Array.isArray(graph.outputs) && graph.outputs.length > 0) {
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
        const item = new dialog.NameValueView(this._host, name, value);
        this._elements.push(item.render());
    }

    addArgument(name, argument) {
        const view = new dialog.ParameterView(this._host, argument);
        view.toggle();
        const item = new dialog.NameValueView(this._host, name, view);
        this._elements.push(item.render());
    }
};

dialog.DocumentationSidebar = class extends dialog.Control {

    constructor(host, type) {
        super();
        this._host = host;
        this._type = type;
    }

    render() {
        if (!this._elements) {
            this._elements = [];

            const type = dialog.DocumentationSidebar.formatDocumentation(this._type);

            const element = this._host.document.createElement('div');
            element.setAttribute('class', 'sidebar-view-documentation');

            this._append(element, 'h1', type.name);

            if (type.summary) {
                this._append(element, 'p', type.summary);
            }

            if (type.description) {
                this._append(element, 'p', type.description);
            }

            if (Array.isArray(type.attributes) && type.attributes.length > 0) {
                this._append(element, 'h2', 'Attributes');
                const attributes = this._append(element, 'dl');
                for (const attribute of type.attributes) {
                    this._append(attributes, 'dt', attribute.name + (attribute.type ? ': <tt>' + attribute.type + '</tt>' : ''));
                    this._append(attributes, 'dd', attribute.description);
                }
                element.appendChild(attributes);
            }

            if (Array.isArray(type.inputs) && type.inputs.length > 0) {
                this._append(element, 'h2', 'Inputs' + (type.inputs_range ? ' (' + type.inputs_range + ')' : ''));
                const inputs = this._append(element, 'dl');
                for (const input of type.inputs) {
                    this._append(inputs, 'dt', input.name + (input.type ? ': <tt>' + input.type + '</tt>' : '') + (input.option ? ' (' + input.option + ')' : ''));
                    this._append(inputs, 'dd', input.description);
                }
            }

            if (Array.isArray(type.outputs) && type.outputs.length > 0) {
                this._append(element, 'h2', 'Outputs' + (type.outputs_range ? ' (' + type.outputs_range + ')' : ''));
                const outputs = this._append(element, 'dl');
                for (const output of type.outputs) {
                    this._append(outputs, 'dt', output.name + (output.type ? ': <tt>' + output.type + '</tt>' : '') + (output.option ? ' (' + output.option + ')' : ''));
                    this._append(outputs, 'dd', output.description);
                }
            }

            if (Array.isArray(type.type_constraints) && type.type_constraints.length > 0) {
                this._append(element, 'h2', 'Type Constraints');
                const type_constraints = this._append(element, 'dl');
                for (const type_constraint of type.type_constraints) {
                    this._append(type_constraints, 'dt', type_constraint.type_param_str + ': ' + type_constraint.allowed_type_strs.map((item) => '<tt>' + item + '</tt>').join(', '));
                    this._append(type_constraints, 'dd', type_constraint.description);
                }
            }

            if (Array.isArray(type.examples) && type.examples.length > 0) {
                this._append(element, 'h2', 'Examples');
                for (const example of type.examples) {
                    this._append(element, 'h3', example.summary);
                    this._append(element, 'pre', example.code);
                }
            }

            if (Array.isArray(type.references) && type.references.length > 0) {
                this._append(element, 'h2', 'References');
                const references = this._append(element, 'ul');
                for (const reference of type.references) {
                    this._append(references, 'li', reference.description);
                }
            }

            if (type.domain && type.version && type.support_level) {
                this._append(element, 'h2', 'Support');
                this._append(element, 'dl', 'In domain <tt>' + type.domain + '</tt> since version <tt>' + type.version + '</tt> at support level <tt>' + type.support_level + '</tt>.');
            }

            if (this._host.type === 'Electron') {
                element.addEventListener('click', (e) => {
                    if (e.target && e.target.href) {
                        const url = e.target.href;
                        if (url.startsWith('http://') || url.startsWith('https://')) {
                            e.preventDefault();
                            this._raise('navigate', { link: url });
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

    _append(parent, type, content) {
        const element = this._host.document.createElement(type);
        if (content) {
            element.innerHTML = content;
        }
        parent.appendChild(element);
        return element;
    }

    static formatDocumentation(source) {
        if (source) {
            const generator = new markdown.Generator();
            const target = {};
            if (source.name !== undefined) {
                target.name = source.name;
            }
            if (source.module !== undefined) {
                target.module = source.module;
            }
            if (source.category !== undefined) {
                target.category = source.category;
            }
            if (source.summary !== undefined) {
                target.summary = generator.html(source.summary);
            }
            if (source.description !== undefined) {
                target.description = generator.html(source.description);
            }
            if (Array.isArray(source.attributes)) {
                target.attributes = source.attributes.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.required !== undefined) {
                        target.required = source.required;
                    }
                    if (source.minimum !== undefined) {
                        target.minimum = source.minimum;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.src_type !== undefined) {
                        target.src_type = source.src_type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.inputs)) {
                target.inputs = source.inputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type !== undefined) {
                        target.type = source.type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.default !== undefined) {
                        target.default = source.default;
                    }
                    if (source.src !== undefined) {
                        target.src = source.src;
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeListAttr;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    if (source.optional !== undefined) {
                        target.optional = source.optional;
                    }
                    if (source.visible !== undefined) {
                        target.visible = source.visible;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.outputs)) {
                target.outputs = source.outputs.map((source) => {
                    const target = {};
                    target.name = source.name;
                    if (source.type) {
                        target.type = source.type;
                    }
                    if (source.description !== undefined) {
                        target.description = generator.html(source.description);
                    }
                    if (source.list !== undefined) {
                        target.list = source.list;
                    }
                    if (source.typeAttr !== undefined) {
                        target.typeAttr = source.typeAttr;
                    }
                    if (source.typeListAttr !== undefined) {
                        target.typeListAttr = source.typeAttr;
                    }
                    if (source.numberAttr !== undefined) {
                        target.numberAttr = source.numberAttr;
                    }
                    if (source.isRef !== undefined) {
                        target.isRef = source.isRef;
                    }
                    if (source.option !== undefined) {
                        target.option = source.option;
                    }
                    return target;
                });
            }
            if (Array.isArray(source.references)) {
                target.references = source.references.map((source) => {
                    if (source) {
                        target.description = generator.html(source.description);
                    }
                    return target;
                });
            }
            if (source.version !== undefined) {
                target.version = source.version;
            }
            if (source.operator !== undefined) {
                target.operator = source.operator;
            }
            if (source.identifier !== undefined) {
                target.identifier = source.identifier;
            }
            if (source.package !== undefined) {
                target.package = source.package;
            }
            if (source.support_level !== undefined) {
                target.support_level = source.support_level;
            }
            if (source.min_input !== undefined) {
                target.min_input = source.min_input;
            }
            if (source.max_input !== undefined) {
                target.max_input = source.max_input;
            }
            if (source.min_output !== undefined) {
                target.min_output = source.min_output;
            }
            if (source.max_input !== undefined) {
                target.max_output = source.max_output;
            }
            if (source.inputs_range !== undefined) {
                target.inputs_range = source.inputs_range;
            }
            if (source.outputs_range !== undefined) {
                target.outputs_range = source.outputs_range;
            }
            if (source.examples !== undefined) {
                target.examples = source.examples;
            }
            if (source.constants !== undefined) {
                target.constants = source.constants;
            }
            if (source.type_constraints !== undefined) {
                target.type_constraints = source.type_constraints;
            }
            return target;
        }
        return '';
    }
};

dialog.FindSidebar = class extends dialog.Control {

    constructor(host, element, graph) {
        super();
        this._host = host;
        this._graphElement = element;
        this._graph = graph;
        this._contentElement = this._host.document.createElement('div');
        this._contentElement.setAttribute('class', 'sidebar-view-find');
        this._searchElement = this._host.document.createElement('input');
        this._searchElement.setAttribute('id', 'search');
        this._searchElement.setAttribute('type', 'text');
        this._searchElement.setAttribute('spellcheck', 'false');
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

        let terms = null;
        let callback = null;
        const unquote = searchText.match(new RegExp(/^'(.*)'|"(.*)"$/));
        if (unquote) {
            const term = unquote[1] || unquote[2];
            terms = [ term ];
            callback = (name) => {
                return term == name;
            };
        }
        else {
            terms = searchText.trim().toLowerCase().split(' ').map((term) => term.trim()).filter((term) => term.length > 0);
            callback = (name) => {
                return terms.every((term) => name.toLowerCase().indexOf(term) !== -1);
            };
        }

        const nodes = new Set();
        const edges = new Set();

        for (const node of this._graph.nodes.values()) {
            const label = node.label;
            const initializers = [];
            if (label.class === 'graph-node' || label.class === 'graph-input') {
                for (const input of label.inputs) {
                    for (const argument of input.arguments) {
                        if (argument.name && !edges.has(argument.name)) {
                            const match = (argument, term) => {
                                if (argument.name && argument.name.toLowerCase().indexOf(term) !== -1) {
                                    return true;
                                }
                                if (argument.type) {
                                    if (argument.type.dataType && term === argument.type.dataType.toLowerCase()) {
                                        return true;
                                    }
                                    if (argument.type.shape) {
                                        if (term === argument.type.shape.toString().toLowerCase()) {
                                            return true;
                                        }
                                        if (argument.type.shape && Array.isArray(argument.type.shape.dimensions)) {
                                            const dimensions = argument.type.shape.dimensions.map((dimension) => dimension ? dimension.toString().toLowerCase() : '');
                                            if (term === dimensions.join(',')) {
                                                return true;
                                            }
                                            if (dimensions.some((dimension) => term === dimension)) {
                                                return true;
                                            }
                                        }
                                    }
                                }
                                return false;
                            };
                            if (terms.every((term) => match(argument, term))) {
                                if (!argument.initializer) {
                                    const inputItem = this._host.document.createElement('li');
                                    inputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                                    inputItem.id = 'edge-' + argument.name;
                                    this._resultElement.appendChild(inputItem);
                                    edges.add(argument.name);
                                }
                                else {
                                    initializers.push(argument);
                                }
                            }
                        }
                    }
                }
            }
            if (label.class === 'graph-node') {
                const name = label.value.name;
                const type = label.value.type.name;
                if (!nodes.has(label.id) &&
                    ((name && callback(name) || (type && callback(type))))) {
                    const nameItem = this._host.document.createElement('li');
                    nameItem.innerText = '\u25A2 ' + (name || '[' + type + ']');
                    nameItem.id = label.id;
                    this._resultElement.appendChild(nameItem);
                    nodes.add(label.id);
                }
            }
            for (const argument of initializers) {
                if (argument.name) {
                    const initializeItem = this._host.document.createElement('li');
                    initializeItem.innerText = '\u25A0 ' + argument.name.split('\n').shift(); // custom argument id
                    initializeItem.id = 'initializer-' + argument.name;
                    this._resultElement.appendChild(initializeItem);
                }
            }
        }

        for (const node of this._graph.nodes.values()) {
            const label = node.label;
            if (label.class === 'graph-node' || label.class === 'graph-output') {
                for (const output of label.outputs) {
                    for (const argument of output.arguments) {
                        if (argument.name && !edges.has(argument.name) && terms.every((term) => argument.name.toLowerCase().indexOf(term) != -1)) {
                            const outputItem = this._host.document.createElement('li');
                            outputItem.innerText = '\u2192 ' + argument.name.split('\n').shift(); // custom argument id
                            outputItem.id = 'edge-' + argument.name;
                            this._resultElement.appendChild(outputItem);
                            edges.add(argument.name);
                        }
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

dialog.Tensor = class {

    constructor(tensor) {
        this._tensor = tensor;
        this._type = tensor.type;
        this._stride = tensor.stride;
        switch (tensor.layout) {
            case undefined:
            case '':
            case '<': {
                this._data = this._tensor.values;
                this._layout = '<';
                this._littleEndian = true;
                break;
            }
            case '>': {
                this._data = this._tensor.values;
                this._layout = '>';
                this._littleEndian = false;
                break;
            }
            case '|': {
                this._values = this._tensor.values;
                this._layout = '|';
                break;
            }
            case 'sparse': {
                this._indices = this._tensor.indices;
                this._values = this._tensor.values;
                this._layout = 'sparse';
                break;
            }
            case 'sparse.coo': {
                this._indices = this._tensor.indices;
                this._values = this._tensor.values;
                this._layout = 'sparse.coo';
                break;
            }
            default: {
                this._layout = tensor.layout;
                break;
            }
        }
        dialog.Tensor.dataTypes = dialog.Tensor.dataTypeSizes || new Map([
            [ 'boolean', 1 ],
            [ 'qint8', 1 ], [ 'qint16', 2 ], [ 'qint32', 4 ],
            [ 'quint8', 1 ], [ 'quint16', 2 ], [ 'quint32', 4 ],
            [ 'int8', 1 ], [ 'int16', 2 ], [ 'int32', 4 ], [ 'int64', 8 ],
            [ 'uint8', 1 ], [ 'uint16', 2 ], [ 'uint32', 4, ], [ 'uint64', 8 ],
            [ 'float16', 2 ], [ 'float32', 4 ], [ 'float64', 8 ], [ 'bfloat16', 2 ],
            [ 'complex64', 8 ], [ 'complex128', 15 ]
        ]);
    }

    get type() {
        return this._type;
    }

    get layout() {
        return this._layout;
    }

    get stride() {
        return this._stride;
    }

    get empty() {
        switch (this._layout) {
            case '<':
            case '>': {
                return !(Array.isArray(this._data) || this._data instanceof Uint8Array || this._data instanceof Int8Array) || this._data.length === 0;
            }
            case '|': {
                return !(Array.isArray(this._values) || ArrayBuffer.isView(this._values)) || this._values.length === 0;
            }
            case 'sparse':
            case 'sparse.coo': {
                return !this._values || this.indices || this._values.values.length === 0;
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    get value() {
        const context = this._context();
        context.limit = Number.MAX_SAFE_INTEGER;
        switch (context.layout) {
            case '<':
            case '>': {
                return this._decodeData(context, 0);
            }
            case '|': {
                return this._decodeValues(context, 0);
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    toString() {
        const context = this._context();
        context.limit = 10000;
        switch (context.layout) {
            case '<':
            case '>': {
                const value = this._decodeData(context, 0);
                return dialog.Tensor._stringify(value, '', '    ');
            }
            case '|': {
                const value = this._decodeValues(context, 0);
                return dialog.Tensor._stringify(value, '', '    ');
            }
            default: {
                throw new Error("Unsupported tensor format '" + this._format + "'.");
            }
        }
    }

    _context() {
        if (this._layout !== '<' && this._layout !== '>' && this._layout !== '|' && this._layout !== 'sparse' && this._layout !== 'sparse.coo') {
            throw new Error("Tensor layout '" + this._layout + "' is not supported.");
        }
        const dataType = this._type.dataType;
        const context = {};
        context.layout = this._layout;
        context.dimensions = this._type.shape.dimensions.map((value) => !Number.isInteger(value) && value.toNumber ? value.toNumber() : value);
        context.dataType = dataType;
        const size = context.dimensions.reduce((a, b) => a * b, 1);
        switch (this._layout) {
            case '<':
            case '>': {
                context.data = (this._data instanceof Uint8Array || this._data instanceof Int8Array) ? this._data : this._data.peek();
                context.view = new DataView(context.data.buffer, context.data.byteOffset, context.data.byteLength);
                if (dialog.Tensor.dataTypes.has(dataType)) {
                    context.itemsize = dialog.Tensor.dataTypes.get(dataType);
                    if (context.data.length < (context.itemsize * size)) {
                        throw new Error('Invalid tensor data size.');
                    }
                }
                else if (dataType.startsWith('uint') && !isNaN(parseInt(dataType.substring(4), 10))) {
                    context.dataType = 'uint';
                    context.bits = parseInt(dataType.substring(4), 10);
                    context.itemsize = 1;
                }
                else if (dataType.startsWith('int') && !isNaN(parseInt(dataType.substring(3), 10))) {
                    context.dataType = 'int';
                    context.bits = parseInt(dataType.substring(3), 10);
                    context.itemsize = 1;
                }
                else {
                    throw new Error("Tensor data type '" + dataType + "' is not implemented.");
                }
                break;
            }
            case '|': {
                context.data = this._values;
                if (!dialog.Tensor.dataTypes.has(dataType) && dataType !== 'string' && dataType !== 'object') {
                    throw new Error("Tensor data type '" + dataType + "' is not implemented.");
                }
                if (size !== this._values.length) {
                    throw new Error('Invalid tensor data length.');
                }
                break;
            }
            case 'sparse': {
                const indices = new dialog.Tensor(this._indices).value;
                const values = new dialog.Tensor(this._values).value;
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.layout = '|';
                break;
            }
            case 'sparse.coo': {
                const values = new dialog.Tensor(this._values).value;
                const data = new dialog.Tensor(this._indices).value;
                const dimensions = context.dimensions.length;
                let stride = 1;
                const strides = context.dimensions.slice().reverse().map((dim) => {
                    const value = stride;
                    stride *= dim;
                    return value;
                }).reverse();
                const indices = new Uint32Array(values.length);
                for (let i = 0; i < dimensions; i++) {
                    const stride = strides[i];
                    const dimension = data[i];
                    for (let i = 0; i < indices.length; i++) {
                        indices[i] += dimension[i] * stride;
                    }
                }
                context.data = this._decodeSparse(dataType, context.dimensions, indices, values);
                context.layout = '|';
                break;
            }
            default: {
                throw new dialog.Tensor("Unsupported tensor layout '" + this._layout + "'.");
            }
        }
        context.index = 0;
        context.count = 0;
        return context;
    }

    _decodeSparse(dataType, dimensions, indices, values) {
        const size = dimensions.reduce((a, b) => a * b, 1);
        const array = new Array(size);
        switch (dataType) {
            case 'boolean':
                array.fill(false);
                break;
            default:
                array.fill(0);
                break;
        }
        if (indices.length > 0) {
            if (Object.prototype.hasOwnProperty.call(indices[0], 'low')) {
                for (let i = 0; i < indices.length; i++) {
                    const index = indices[i];
                    array[index.high === 0 ? index.low : index.toNumber()] = values[i];
                }
            }
            else {
                for (let i = 0; i < indices.length; i++) {
                    array[indices[i]] = values[i];
                }
            }
        }
        return array;
    }

    _decodeData(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [ 1 ] : context.dimensions;
        const size = dimensions[dimension];
        const dataType = context.dataType;
        const view = context.view;
        if (dimension == dimensions.length - 1) {
            const ellipsis = (context.count + size) > context.limit;
            const length = ellipsis ? context.limit - context.count : size;
            let i = context.index;
            const max = i + (length * context.itemsize);
            switch (dataType) {
                case 'boolean':
                    for (; i < max; i += 1) {
                        results.push(view.getUint8(i) === 0 ? false : true);
                    }
                    break;
                case 'qint8':
                case 'int8':
                    for (; i < max; i++) {
                        results.push(view.getInt8(i));
                    }
                    break;
                case 'qint16':
                case 'int16':
                    for (; i < max; i += 2) {
                        results.push(view.getInt16(i, this._littleEndian));
                    }
                    break;
                case 'qint32':
                case 'int32':
                    for (; i < max; i += 4) {
                        results.push(view.getInt32(i, this._littleEndian));
                    }
                    break;
                case 'int64':
                    for (; i < max; i += 8) {
                        results.push(view.getInt64(i, this._littleEndian));
                    }
                    break;
                case 'int':
                    for (; i < size; i++) {
                        results.push(view.getIntBits(i, context.bits));
                    }
                    break;
                case 'quint8':
                case 'uint8':
                    for (; i < max; i++) {
                        results.push(view.getUint8(i));
                    }
                    break;
                case 'quint16':
                case 'uint16':
                    for (; i < max; i += 2) {
                        results.push(view.getUint16(i, true));
                    }
                    break;
                case 'quint32':
                case 'uint32':
                    for (; i < max; i += 4) {
                        results.push(view.getUint32(i, true));
                    }
                    break;
                case 'uint64':
                    for (; i < max; i += 8) {
                        results.push(view.getUint64(i, true));
                    }
                    break;
                case 'uint':
                    for (; i < max; i++) {
                        results.push(view.getUintBits(i, context.bits));
                    }
                    break;
                case 'float16':
                    for (; i < max; i += 2) {
                        results.push(view.getFloat16(i, this._littleEndian));
                    }
                    break;
                case 'float32':
                    for (; i < max; i += 4) {
                        results.push(view.getFloat32(i, this._littleEndian));
                    }
                    break;
                case 'float64':
                    for (; i < max; i += 8) {
                        results.push(view.getFloat64(i, this._littleEndian));
                    }
                    break;
                case 'bfloat16':
                    for (; i < max; i += 2) {
                        results.push(view.getBfloat16(i, this._littleEndian));
                    }
                    break;
                case 'complex64':
                    for (; i < max; i += 8) {
                        results.push(view.getComplex64(i, this._littleEndian));
                        context.index += 8;
                    }
                    break;
                case 'complex128':
                    for (; i < size; i += 16) {
                        results.push(view.getComplex128(i, this._littleEndian));
                    }
                    break;
                default:
                    throw new Error("Unsupported tensor data type '" + dataType + "'.");
            }
            context.index = i;
            context.count += length;
            if (ellipsis) {
                results.push('...');
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count >= context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decodeData(context, dimension + 1));
            }
        }
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    _decodeValues(context, dimension) {
        const results = [];
        const dimensions = (context.dimensions.length == 0) ? [ 1 ] : context.dimensions;
        const size = dimensions[dimension];
        const dataType = context.dataType;
        if (dimension == dimensions.length - 1) {
            for (let i = 0; i < size; i++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                switch (dataType) {
                    case 'boolean':
                        results.push(context.data[context.index] === 0 ? false : true);
                        break;
                    default:
                        results.push(context.data[context.index]);
                        break;
                }
                context.index++;
                context.count++;
            }
        }
        else {
            for (let j = 0; j < size; j++) {
                if (context.count > context.limit) {
                    results.push('...');
                    return results;
                }
                results.push(this._decodeValues(context, dimension + 1));
            }
        }
        if (context.dimensions.length == 0) {
            return results[0];
        }
        return results;
    }

    static _stringify(value, indentation, indent) {
        if (Array.isArray(value)) {
            const result = [];
            result.push(indentation + '[');
            const items = value.map((item) => dialog.Tensor._stringify(item, indentation + indent, indent));
            if (items.length > 0) {
                result.push(items.join(',\n'));
            }
            result.push(indentation + ']');
            return result.join('\n');
        }
        if (value === null) {
            return indentation + 'null';
        }
        switch (typeof value) {
            case 'boolean':
                return indentation + value.toString();
            case 'string':
                return indentation + '"' + value + '"';
            case 'number':
                if (value == Infinity) {
                    return indentation + 'Infinity';
                }
                if (value == -Infinity) {
                    return indentation + '-Infinity';
                }
                if (isNaN(value)) {
                    return indentation + 'NaN';
                }
                return indentation + value.toString();
            default:
                if (value && value.toString) {
                    return indentation + value.toString();
                }
                return indentation + '(undefined)';
        }
    }
};

dialog.Formatter = class {

    constructor(value, type, quote) {
        this._value = value;
        this._type = type;
        this._quote = quote;
        this._values = new Set();
    }

    toString() {
        return this._format(this._value, this._type, this._quote);
    }

    _format(value, type, quote) {

        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'type') {
            return value.__module__ + '.' + value.__name__;
        }
        if (value && value.__class__ && value.__class__.__module__ === 'builtins' && value.__class__.__name__ === 'function') {
            return value.__module__ + '.' + value.__name__;
        }
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
                if (value && !Array.isArray(value)) {
                    throw new Error("Invalid shape '" + JSON.stringify(value) + "'.");
                }
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            case 'graph':
                return value ? value.name : '(null)';
            case 'graph[]':
                return value ? value.map((graph) => graph.name).join(', ') : '(null)';
            case 'tensor':
                if (value && value.type && value.type.shape && value.type.shape.dimensions && value.type.shape.dimensions.length == 0) {
                    return value.toString();
                }
                return '[...]';
            case 'function':
                return value.type.name;
            case 'function[]':
                return value ? value.map((item) => item.type.name).join(', ') : '(null)';
            case 'type':
                return value ? value.toString() : '(null)';
            case 'type[]':
                return value ? value.map((item) => item.toString()).join(', ') : '(null)';
            default:
                break;
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
                return this._format(item, itemType, quote);
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
        if (this._values.has(value)) {
            return '\u2026';
        }
        this._values.add(value);
        let list = null;
        const entries = Object.entries(value).filter((entry) => !entry[0].startsWith('__') && !entry[0].endsWith('__'));
        if (entries.length == 1) {
            list = [ this._format(entries[0][1], null, true) ];
        }
        else {
            list = new Array(entries.length);
            for (let i = 0; i < entries.length; i++) {
                const entry = entries[i];
                list[i] = entry[0] + ': ' + this._format(entry[1], null, true);
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
        source = source.replace(/\r\n|\r/g, '\n').replace(/\t/g, '    ');
        this._tokenize(source, tokens, links, true);
        this._tokenizeBlock(tokens, links);
        const result = this._render(tokens, true);
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
                let content = match[3] || '';
                const matchIndent = match[0].match(/^(\s+)(?:```)/);
                if (matchIndent !== null) {
                    const indent = matchIndent[1];
                    content = content.split('\n').map(node => {
                        const match = node.match(/^\s+/);
                        return (match !== null && match[0].length >= indent.length) ? node.slice(indent.length) : node;
                    }).join('\n');
                }
                tokens.push({ type: 'code', language: language, text: content });
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
                            default:
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
                let content = match[2].replace(/\n/g, ' ');
                if (/[^ ]/.test(content) && content.startsWith(' ') && content.endsWith(' ')) {
                    content = content.substring(1, content.length - 1);
                }
                tokens.push({ type: 'codespan', text: this._encode(content) });
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
                default: {
                    break;
                }
            }
        }
    }

    _render(tokens, top) {
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
                    html += '<h' + level + '">' + this._renderInline(token.tokens) + '</h' + level + '>\n';
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
                    html += '<blockquote>\n' + this._render(token.tokens, true) + '</blockquote>\n';
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
                        itemBody += this._render(item.tokens, loose);
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

    _renderInline(tokens) {
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
                    const text = this._renderInline(token.tokens);
                    html += '<a href="' + token.href + '"' + (token.title ? ' title="' + token.title + '"' : '') + ' target="_blank">' + text + '</a>';
                    break;
                }
                case 'image': {
                    html += '<img src="' + token.href + '" alt="' + token.text + '"' + (token.title ? ' title="' + token.title + '"' : '') + '>';
                    break;
                }
                case 'strong': {
                    const text = this._renderInline(token.tokens);
                    html += '<strong>' + text + '</strong>';
                    break;
                }
                case 'em': {
                    const text = this._renderInline(token.tokens);
                    html += '<em>' + text + '</em>';
                    break;
                }
                case 'codespan': {
                    html += '<code>' + token.text + '</code>';
                    break;
                }
                case 'br': {
                    html += '<br>';
                    break;
                }
                case 'del': {
                    const text = this._renderInline(token.tokens);
                    html += '<del>' + text + '</del>';
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

    _encode(content) {
        if (this._escapeTestRegExp.test(content)) {
            return content.replace(this._escapeReplaceRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }

    _escape(content) {
        if (this._escapeTestNoEncodeRegExp.test(content)) {
            return content.replace(this._escapeReplaceNoEncodeRegExp, (ch) => this._escapeReplacementsMap[ch]);
        }
        return content;
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Sidebar = dialog.Sidebar;
    module.exports.ModelSidebar = dialog.ModelSidebar;
    module.exports.NodeSidebar = dialog.NodeSidebar;
    module.exports.DocumentationSidebar = dialog.DocumentationSidebar;
    module.exports.FindSidebar = dialog.FindSidebar;
    module.exports.Tensor = dialog.Tensor;
    module.exports.Formatter = dialog.Formatter;
}