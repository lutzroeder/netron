
var xml = {};
var text = require('./text');

// https://www.w3.org/TR/xml

xml.TextReader = class {

    static open(data, callback) {
        const decoder = text.Decoder.open(data);
        for (;;) {
            const c = decoder.decode();
            if (c === '<') {
                break;
            }
            if (c === ' ' || c === '\n' || c === '\r' || c === '\t') {
                continue;
            }
            return null;
        }
        return new xml.TextReader(data, callback);
    }

    constructor(data, callback) {
        this._data = data;
        this._callback = callback;
        this._entities = new Map([ [ 'quot', '"' ], [ 'amp', '&' ], [ 'apos', "'" ], [ 'lt', '<' ],  [ 'gt', '>' ] ]);
        this._nameStartCharRegExp = /[:A-Z_a-z\xC0-\xD6\xD8-\xF6\xF8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]/;
        this._nameCharRegExp = new RegExp("[-.0-9\\xB7" + this._nameStartCharRegExp.source.slice(1, -1) + "]");
        xml.Utility.nameStartCharRegExp = this._nameStartCharRegExp;
    }

    peek() {
        this._peek = true;
        const value = this.read();
        delete this._peek;
        return value;
    }

    read() {
        this._stack = [];
        this._context = [];
        this._pushBuffer(this._data, '', '', false);
        this._version = 0;
        /* eslint-disable */
        this._charRegExp = /[\x09\x0a\x0d\x20-\uD7FF\uE000-\uFFFD]/;
        /* eslint-enable */
        this._parameterEntities = false;
        this._characterData = true;
        this._push(new xml.Document());
        const document = this._document();
        for (;;) {
            this._start = this._position;
            switch (this._char) {
                case '<': {
                    this._next();
                    switch (this._char) {
                        case '?': {
                            this._processingInstruction();
                            break;
                        }
                        case '!': {
                            this._next();
                            if (this._match('--')) {
                                this._comment();
                            } else if (this._match('[CDATA')) {
                                this._assert(this._stack.length > 1);
                                this._characterData = true;
                                this._expect('[');
                                const data = this._terminal(']]>');
                                const node = document.createCDATASection(data);
                                this._appendChild(node);
                            } else if (this._match('DOCTYPE')) {
                                this._assert(this._stack.length > 1 || !document.documentElement || !document.documentType);
                                this._whitespace(1);
                                const name = this._name();
                                this._assert(name !== null);
                                let systemId = '';
                                let publicId = '';
                                let whitespace = this._whitespace(0);
                                if (whitespace && this._match('SYSTEM')) {
                                    this._whitespace(1);
                                    systemId = this._systemLiteral();
                                    this._whitespace(0);
                                    whitespace = true;
                                } else if (whitespace && this._match('PUBLIC')) {
                                    this._whitespace(1);
                                    publicId = this._pubidLiteral();
                                    this._whitespace(1);
                                    systemId = this._systemLiteral();
                                    this._whitespace(0);
                                    whitespace = true;
                                }
                                const node = document.createDocumentType(name, publicId, systemId);
                                this._appendChild(node);
                                this._push(node);
                                node.parameterEntities = new xml.NamedNodeMap();
                                node.elements = new xml.NamedNodeMap();
                                this._parameterEntities = true;
                                this._characterData = false;
                                const internalSubset = whitespace && this._match('[');
                                if (internalSubset) {
                                    this._internalSubset(']');
                                }
                                if (systemId && !this._standalone) {
                                    this._pushResource(systemId, '', true);
                                    this._internalSubset(undefined);
                                    this._popContext();
                                }
                                this._characterData = true;
                                this._parameterEntities = false;
                                const values = node.entities.filter((entity) => entity.value).map((entity) => entity.value);
                                for (const entity of node.entities.filter((entity) => entity.notationName)) {
                                    const reference = '&' + entity.localName + ';';
                                    if (values.some((value) => value.indexOf(reference) >= 0)) {
                                        this._error("Entity references unparsed entity '" + entity.localName + "'");
                                    }
                                }
                                if (internalSubset) {
                                    this._expect(']');
                                    this._whitespace(0);
                                }
                                this._expect('>');
                                this._assert(this._pop().nodeType === xml.NodeType.DocumentType);
                            } else {
                                this._unexpected();
                            }
                            break;
                        }
                        case '/': {
                            this._next();
                            const name = this._name();
                            this._assert(name !== null);
                            this._whitespace(0);
                            this._expect('>');
                            const node = this._pop();
                            const nodeName = node.prefix ? node.prefix + ':' + node.localName : node.localName;
                            if (name !== nodeName) {
                                this._error("Opening tag <" + nodeName + "> and ending tag </" + name + "> mismatch", this._start);
                            }
                            break;
                        }
                        default: {
                            this._assert(this._stack.length > 1 || !this._document.documentElement);
                            const name = this._name();
                            this._assert(name !== null);
                            this._assert(!name.startsWith('xmlns:'));
                            const attributes = [];
                            let whitespace = this._whitespace(0);
                            if (whitespace) {
                                while (this._char !== '/' && this._char !== '>') {
                                    if (!whitespace) {
                                        this._unexpected();
                                    }
                                    const position = this._position;
                                    const name = this._name();
                                    if (!name) {
                                        this._unexpected();
                                    }
                                    this._whitespace(0);
                                    this._expect('=');
                                    this._whitespace(0);
                                    const valuePosition = this._valuePosition;
                                    const value = this._attributeValue();
                                    attributes.push({
                                        qualifiedName: name,
                                        value: value,
                                        position: position,
                                        valuePosition: valuePosition
                                    });
                                    whitespace = this._whitespace(0);
                                    if (name === 'xmlns' && (!this._validateNamespace(value) || value === 'http://www.w3.org/2000/xmlns/' || value === 'http://www.w3.org/XML/1998/namespace')) {
                                        this._error("Invalid namespace '" + value + "'", valuePosition);
                                    }
                                    if (name === 'xml:space' && value !== 'preserve' && value !== 'default') {
                                        this._error("Unexpected xml:space attribute value '" + value + "'", position);
                                    }
                                }
                            }
                            const namespaces = new Map();
                            for (const entry of attributes.reverse()) {
                                const name = entry.qualifiedName;
                                const value = entry.value;
                                const pair = xml.Utility.split(name);
                                this._assert(name !== 'xmlns:');
                                entry.prefix = pair[0];
                                entry.localName = pair[1];
                                if (entry.prefix !== null) {
                                    this._assert(entry.localName !== '');
                                    if (entry.prefix === 'xmlns' && entry.localName) {
                                        if (!this._validateNamespace(value) || value === 'http://www.w3.org/2000/xmlns/') {
                                            this._error("Invalid namespace '" + value + "'", entry.valuePosition);
                                        }
                                        if (entry.localName === 'xmlns' || (entry.localName === 'xml' && value !== 'http://www.w3.org/XML/1998/namespace') || (entry.localName !== 'xml' && value === 'http://www.w3.org/XML/1998/namespace')) {
                                            this._error("Invalid namespace prefix '" + entry.localName + "'", entry.position);
                                        }
                                        if (this._version === 0 && value.length === 0) {
                                            this._error("Invalid namespace declaration'", entry.position);
                                        }
                                        namespaces.set(entry.localName, value);
                                    }
                                } else if (entry.localName === 'xmlns') {
                                    namespaces.set('', value);
                                }
                            }
                            const pair = xml.Utility.split(name);
                            const prefix = pair[0] || '';
                            const namespaceURI = namespaces.has(prefix) ? namespaces.get(prefix) : this._lookupNamespaceURI(prefix);
                            let element = null;
                            const documentType = document.documentType;
                            const elementType = documentType ? documentType.elements.getNamedItem(name) : null;
                            if (namespaceURI !== null) {
                                this._assert(name === ':' || (!name.endsWith(':') && !name.startsWith(':')));
                                if (prefix && namespaceURI === '') {
                                    this._error("Invalid namespace prefix '" + prefix + "'", this._start);
                                }
                                element = document.createElementNS(namespaceURI, name);
                            } else {
                                this._assert((pair[0] === null && !name.endsWith(':')) || name === ':' || elementType !== null);
                                element = document.createElement(name);
                            }
                            const parent = this._node();
                            if (parent.nodeType === xml.NodeType.Document && parent.documentElement !== null) {
                                this._error('Duplicate document element', this._start);
                            }
                            this._appendChild(element);
                            const keys = new Set();
                            for (const attr of attributes) {
                                const name = attr.qualifiedName;
                                const prefix = attr.prefix || '';
                                const namespaceURI = namespaces.has(prefix) ? namespaces.get(prefix) : this._lookupNamespaceURI(prefix);
                                let attribute = null;
                                if (namespaceURI) {
                                    attribute = document.createAttributeNS(namespaceURI, name);
                                } else {
                                    const attributeType = elementType ? elementType.attributes.getNamedItem(name) : null;
                                    this._assert(name.indexOf(':') === -1 || attributeType);
                                    attribute = document.createAttribute(name);
                                }
                                const key = (attribute.namespaceURI || '') + '|' + attribute.localName;
                                this._assert(!keys.has(key));
                                keys.add(key);
                                attribute.value = attr.value;
                                attribute.ownerElement = element;
                                element.setAttributeNode(attribute);
                            }
                            const close = this._match('/');
                            this._expect('>');
                            if (this._peek && this._stack.length === 1 && this._nodeType() === xml.NodeType.Document) {
                                return this._pop();
                            }
                            if (!close) {
                                this._push(element);
                            }
                            break;
                        }
                    }
                    break;
                }
                default: {
                    while (this._char === undefined && this._context.length > 0) {
                        this._popContext();
                    }
                    if (this._char === undefined) {
                        if (this._stack.length === 1 && this._nodeType() === xml.NodeType.Document) {
                            this._assert(document.documentElement);
                            const documentType = document.documentType;
                            if (documentType) {
                                delete documentType.parameterEntities;
                                delete documentType.elements;
                            }
                            const value = this._pop();
                            for (const key of Object.keys(this)) {
                                if (key !== '_data' && key !== '_callback' && key !== '_entities' && !key.startsWith('_name')) {
                                    delete this[key];
                                }
                            }
                            return value;
                        }
                        this._unexpected();
                    }
                    const node = this._node();
                    if (node.nodeType === xml.NodeType.Element) {
                        const documentType = document.documentType;
                        const name = node.prefix ? node.prefix + ':' + node.localName : node.localName;
                        const elementType = documentType ? documentType.elements.getNamedItem(name) : null;
                        this._characterData = elementType ? elementType.characterData : false;
                        this._seek(this._position);
                        const data = [];
                        while (this._char !== '<' && this._char !== undefined) {
                            if (this._char === ']' && this._match(']]>')) {
                                this._unexpected();
                            }
                            data.push(this._content());
                            if (data.length > 65536) {
                                this._error('Invalid character data buffer size.');
                            }
                        }
                        if (data.length > 0) {
                            const content = data.splice(0, data.length).join('');
                            if (content.trim().length > 0) {
                                const node = document.createTextNode(content);
                                this._appendChild(node);
                            }
                        }
                        continue;
                    }
                    if (!this._whitespace(0)) {
                        this._unexpected();
                    }
                    break;
                }
            }
        }
    }

    _internalSubset(terminal) {
        for (;;) {
            this._start = this._position;
            switch (this._char) {
                case '<': {
                    this._next();
                    switch (this._char) {
                        case '?': {
                            this._processingInstruction();
                            break;
                        }
                        case '!': {
                            this._next();
                            if (this._match('--')) {
                                this._parameterEntities = false;
                                this._characterData = true;
                                this._comment();
                                this._parameterEntities = true;
                            } else if (this._match('ENTITY')) {
                                const documentType = this._node();
                                this._assert(documentType.nodeType === xml.NodeType.DocumentType);
                                this._parameterEntities = false;
                                this._whitespace(1);
                                const parameter = this._char === '%';
                                if (parameter) {
                                    this._next();
                                    this._whitespace(1);
                                }
                                this._parameterEntities = true;
                                const name = this._entityName();
                                const node = documentType.createEntity(name);
                                let whitespace = this._whitespace(0);
                                if (whitespace && (this._char === '"' || this._char === "'")) {
                                    node.value = this._entityValue();
                                    this._whitespace(0);
                                } else {
                                    if (whitespace && this._match('SYSTEM')) {
                                        this._whitespace(1);
                                        node.systemId = this._systemLiteral();
                                        whitespace = this._whitespace(0);
                                    } else if (whitespace && this._match('PUBLIC')) {
                                        this._whitespace(1);
                                        node.publicId = this._pubidLiteral();
                                        this._whitespace(1);
                                        node.systemId = this._systemLiteral();
                                        whitespace = this._whitespace(0);
                                    } else {
                                        this._unexpected();
                                    }
                                    if (whitespace && !parameter) {
                                        if (this._match('NDATA')) {
                                            this._whitespace(1);
                                            const name = this._name();
                                            this._assert(name !== null);
                                            node.notationName = name;
                                            this._whitespace(0);
                                        }
                                    }
                                }
                                this._expect('>');
                                if (parameter) {
                                    documentType.parameterEntities.setNamedItem(node);
                                } else {
                                    this._appendChild(node);
                                }
                            } else if (this._match('ELEMENT')) {
                                const documentType = this._node();
                                this._assert(documentType.nodeType === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                const name = this._name();
                                this._assert(name !== null);
                                this._whitespace(1);
                                const elementType = this._elementType(name);
                                if (this._match('EMPTY')) {
                                    this._whitespace(0);
                                } else if (this._match('ANY')) {
                                    this._whitespace(0);
                                } else {
                                    this._expect('(');
                                    this._whitespace(0);
                                    if (this._match('#PCDATA')) {
                                        elementType.characterData = true;
                                        this._whitespace(0);
                                        if (this._match(')')) {
                                            this._match('*');
                                        } else {
                                            this._whitespace(0);
                                            while (this._match('|')) {
                                                this._whitespace(0);
                                                const name = this._name();
                                                this._assert(name);
                                                this._whitespace(0);
                                            }
                                            this._expect(')*');
                                        }
                                    } else {
                                        this._elementChildren();
                                    }
                                }
                                this._whitespace(0);
                                this._expect('>');
                            } else if (this._match('ATTLIST')) {
                                const documentType = this._node();
                                this._assert(documentType.nodeType === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                const name = this._name();
                                this._assert(name !== null);
                                const elementType = this._elementType(name);
                                while (this._whitespace(0)) {
                                    const attributeType = this._attributeDefinition();
                                    if (!attributeType) {
                                        break;
                                    }
                                    elementType.attributes.setNamedItem(attributeType);
                                }
                                this._whitespace(0);
                                this._expect('>');
                            } else if (this._match('NOTATION')) {
                                this._assert(this._nodeType() === xml.NodeType.DocumentType);
                                const notation = { systemId: null, publicId: null };
                                this._whitespace(1);
                                notation.name = this._entityName();
                                let whitespace = this._whitespace(0);
                                if (whitespace && this._match('SYSTEM')) {
                                    this._whitespace(1);
                                    notation.systemId = this._systemLiteral();
                                    whitespace = this._whitespace(0);
                                }
                                if (whitespace && this._match('PUBLIC')) {
                                    this._whitespace(1);
                                    notation.publicId = this._pubidLiteral();
                                    if (this._whitespace(0) && (this._char === '"') || this._char === "'") {
                                        notation.systemId = this._systemLiteral();
                                        this._whitespace(0);
                                    }
                                }
                                this._assert(notation.systemId || notation.publicId);
                                this._expect('>');
                            } else if (this._match('[')) {
                                this._whitespace(0);
                                if (this._match('INCLUDE')) {
                                    this._assert(this._context.length > 0);
                                    this._whitespace(0);
                                    this._expect('[');
                                    this._internalSubset(']');
                                    this._expect(']]>');
                                } else if (this._match('IGNORE')) {
                                    this._whitespace(0);
                                    this._expect('[');
                                    this._ignoreSectContents();
                                }
                            } else {
                                this._unexpected();
                            }
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                    break;
                }
                case '%': {
                    this._resolveParameterEntityReference();
                    break;
                }
                default: {
                    if (this._char === terminal) {
                        return;
                    }
                    if (!this._whitespace(0)) {
                        this._unexpected();
                    }
                    break;
                }
            }
        }
    }

    _ignoreSectContents() {
        while (!this._match(']]>')) {
            if (this._match('<![')) {
                this._ignoreSectContents();
            } else {
                this._next();
            }
        }
    }

    _push(value) {
        this._stack.push(value);
    }

    _pop() {
        return this._stack.pop();
    }

    _node() {
        return this._stack[this._stack.length - 1];
    }

    _document() {
        return this._stack[0];
    }

    _nodeType() {
        return this._node().nodeType;
    }

    _appendChild(newChild) {
        return this._node().appendChild(newChild);
    }

    _lookupNamespaceURI(prefix) {
        return this._node().lookupNamespaceURI(prefix);
    }

    _name() {
        const position = this._position;
        const name = [];
        const c = this._char.codePointAt(0);
        if (this._nameStartCharRegExp.test(this._char) || (c >= 0x10000 && c <= 0xEFFFF)) {
            name.push(this._char);
            this._next();
            if (this._char !== undefined) {
                let c = this._char.codePointAt(0);
                while (this._nameCharRegExp.test(this._char) || (c >= 0x300 && c <= 0x36f) || (c >= 0x203F && c <= 0x2040)) {
                    name.push(this._char);
                    this._next();
                    if (this._char === undefined || this._implicitSpace) {
                        break;
                    }
                    c = this._char.codePointAt(0);
                }
            }
        }
        if (name.length > 0) {
            return name.join('');
        }
        this._seek(position);
        return null;
    }

    _nmtoken() {
        const position = this._position;
        const name = [];
        let c = this._char.codePointAt(0);
        while (this._nameCharRegExp.test(this._char) || (c >= 0x300 && c <= 0x36f) || (c >= 0x203F && c <= 0x2040)) {
            name.push(this._char);
            this._next();
            if (this._char === undefined) {
                break;
            }
            c = this._char.codePointAt(0);
        }
        if (name.length > 0) {
            return name.join('');
        }
        this._seek(position);
        return null;
    }

    _entityName() {
        const position = this._position;
        const name = this._name();
        if (name === null) {
            this._error('Expected entity name', position);
        }
        if (!name.endsWith(':') && name.indexOf(':') !== -1) {
            this._error('Invalid colon in entity name', position);
        }
        return name;
    }

    _entityValue() {
        const quote = this._char;
        this._parameterEntities = false;
        this._characterData = true;
        const decoder = this._decoder;
        const position = this._position;
        this._next();
        while (this._char !== quote) {
            if (this._char === undefined) {
                this._unexpected();
            }
            this._next();
        }
        const end = this._position;
        this._parameterEntities = true;
        this._seek(position);
        this._next();
        const data = [];
        while (this._position !== end || this._decoder !== decoder) {
            if (this._char === undefined) {
                this._unexpected();
            }
            if (this._char === '%') {
                if (this._context.length === 0) {
                    this._error('Invalid parameter entity reference in internal subset');
                }
                this._assert();
                this._resolveParameterEntityReference();
                continue;
            }
            if (this._char === '&') {
                data.push(this._entityReference());
                this._expect(';');
                continue;
            }

            data.push(this._char);
            this._next();
        }
        this._next();
        this._parameterEntities = true;
        this._characterData = false;
        return data.join('');
    }

    _elementType(name) {
        const documentType = this._document().documentType;
        let elementType = documentType.elements.getNamedItem(name);
        if (!elementType) {
            elementType = { localName: name, characterData: false, attributes: new xml.NamedNodeMap() };
            documentType.elements.setNamedItem(elementType);
        }
        return elementType;
    }

    _elementChildren() {
        let separator = undefined;
        const choice = new Set();
        for (;;) {
            const name = this._name();
            if (name) {
                this._assert(separator !== '|' || !choice.has(name));
                choice.add(name);
                this._match('?') || this._match('*') || this._match('+');
                this._whitespace(0);
            } else if (this._match('(')) {
                this._elementChildren();
                this._whitespace(0);
            } else {
                this._unexpected();
            }
            if (this._match(')')) {
                break;
            }
            if (separator && separator !== this._char) {
                this._unexpected();
            }
            if (this._char !== '|' && this._char !== ',') {
                this._unexpected();
            }
            separator = this._char;
            this._next();
            this._whitespace(0);
        }
        this._match('?') || this._match('*') || this._match('+');
    }

    _attributeDefinition() {
        this._whitespace(0);
        const name = this._name();
        if (name) {
            this._whitespace(1);
            if (this._match('CDATA') || this._match('IDREFS') || this._match('IDREF') || this._match('ID') || this._match('ENTITIES') || this._match('ENTITY') || this._match('NMTOKENS') || this._match('NMTOKEN') ||
                this._enumeratedType()) {
                this._whitespace(1);
                if (!this._match('#REQUIRED') && !this._match('#IMPLIED')) {
                    if (this._match('#FIXED')) {
                        this._whitespace(1);
                    }
                    this._parameterEntities = false;
                    this._attributeValue();
                    this._parameterEntities = true;
                }
                return { localName: name };
            }
            this._assert(false);
        }
        return null;
    }

    _enumeratedType() {
        if (this._match('NOTATION')) {
            this._whitespace(1);
            this._expect('(');
            do {
                this._whitespace(0);
                const name = this._name();
                this._assert(name);
                this._whitespace(0);
            }
            while (this._match('|'));
            this._expect(')');
            return true;
        }
        if (this._match('(')) {
            do {
                this._whitespace(0);
                const name = this._nmtoken();
                this._assert(name);
                this._whitespace(0);
            }
            while (this._match('|'));
            this._expect(')');
            return true;
        }
        return false;
    }

    _content() {
        const c = this._char !== '&' ? this._char : this._resolveEntityReference();
        if (c === undefined) {
            return '';
        }
        const code = c.codePointAt(0);
        if ((!this._charRegExp.test(c) && (code < 0x10000 || c > 0x10FFFF))) {
            this._unexpected();
        }
        this._next();
        return c;
    }

    _attributeValue() {
        const quote = this._char;
        if (quote !== '"' && quote !== "'") {
            this._unexpected();
        }
        this._characterData = true;
        const decoder = this._decoder;
        const position = this._position;
        this._next();
        while (this._char !== quote) {
            if (this._char === undefined || this._char === '<') {
                this._unexpected();
            }
            this._next();
        }
        const end = this._position;
        this._characterData = false;
        this._seek(position);
        this._next();
        const data = [];
        while (this._position !== end || this._decoder !== decoder) {
            if (this._char === undefined && this._context.length > 0) {
                this._popContext();
                continue;
            }
            if (this._char === '<') {
                this._unexpected();
            }
            data.push(this._content());
            if (data.length > 65536) {
                this._error('Invalid character data buffer size.');
            }
        }
        this._characterData = true;
        this._next();
        return data.join('');
    }

    _validateNamespace(value) {
        if (value && (value.startsWith('#') || value.indexOf(':') === -1)) {
            return false;
        }
        if (this._version > 0) {
            return true;
        }
        return /^[A-Za-z0-9-._~:/?#[\]@!$&'()*+,;%=]*$/.exec(value) !== null;
    }

    _pubidLiteral() {
        const quote = this._char;
        if (quote !== '"' && quote !== "'") {
            this._unexpected();
        }
        this._next();
        const data = [];
        while (this._char !== quote) {
            if (/[a-zA-Z0-9-'()+,./:=?;!*#@$_%]/.test(this._char) || this._char === ' ' || this._char === '\r' || this._char === '\n') {
                data.push(this._char);
                this._next();
                if (this._char === undefined) {
                    this._unexpected();
                }
                continue;
            }
            this._unexpected();
        }
        this._next();
        return data.join('');
    }

    _systemLiteral() {
        const quote = this._char;
        if (quote !== '"' && quote !== "'") {
            this._unexpected();
        }
        this._next();
        const data = [];
        while (this._char !== quote) {
            data.push(this._char);
            this._next();
            if (this._char === undefined) {
                this._unexpected();
            }
        }
        this._next();
        const value = data.join('');
        if (value.indexOf('#') >= 0) {
            this._unexpected();
        }
        const match = /(.*\/)[^/]*/.exec(this._base);
        return (match ? match[1] : '') + value;
    }

    _terminal(terminal) {
        const data = [];
        while (!this._match(terminal)) {
            if (this._char === undefined) {
                this._unexpected();
            }
            const c = this._char.codePointAt(0);
            if (c !== 0x09 && c !== 0x0A && c !== 0x0D && (c < 0x20 || c > 0xD7FF) && (c < 0xE000 || c > 0xFFFD) && (c < 0x10000 || c > 0x10FFFF)) {
                this._unexpected();
            }
            data.push(this._char);
            this._next();
        }
        return data.join('');
    }

    _resolveParameterEntityReference() {
        const position = this._position;
        this._next();
        const name = this._name();
        this._assert(name !== null);
        if (this._char === ';') {
            const entity = this._document().documentType.parameterEntities.getNamedItem(name);
            if (entity) {
                const implicitSpace = !this._entity && !this._context.some((context) => context.entity);
                if (entity.systemId) {
                    this._pushResource(entity.systemId, name, false);
                } else {
                    this._pushString(entity.value, name, false);
                }
                if (implicitSpace) {
                    this._implicitSpace = true;
                }
                return;
            }
            this._error("Undefined ENTITY '" + name + "'", position);
        }
        this._unexpected();
    }

    _resolveEntityReference() {
        const position = this._position;
        let entity = this._entityReference();
        const name = entity.substring(1, entity.length - 1);
        if (name.startsWith('#x')) {
            const value = parseInt(name.substring(2), 16);
            return String.fromCodePoint(value);
        } else if (name.startsWith('#')) {
            const value = parseInt(name.substring(1), 10);
            return String.fromCodePoint(value);
        } else if (this._entities.has(name)) {
            return this._entities.get(name);
        }
        const documentType = this._document().documentType;
        entity = documentType ? documentType.entities.getNamedItem(name) : null;
        if (entity) {
            if (entity.systemId) {
                this._pushResource(entity.systemId, name, true);
            } else {
                this._pushString(entity.value, name, true);
            }
        } else if (this._context.length !== 0 || !documentType || documentType.parameterEntities.length === 0) {
            this._error("Undefined ENTITY '" + name + "'", position);
        }
        return undefined;
    }

    /* eslint-disable consistent-return */
    _entityReference() {
        if (this._char === '&') {
            const position = this._position;
            this._next();
            if (this._match('#x')) {
                const data = [];
                while (/[0-9a-fA-F]/.test(this._char)) {
                    data.push(this._char);
                    this._next();
                    if (this._char === undefined) {
                        this._unexpected();
                    }
                }
                this._assert(this._char === ';');
                if (data.length > 0) {
                    const text = data.join('');
                    const value = parseInt(text, 16);
                    this._assert(value <= 0x10FFFF, "Invalid value '&#x" + text + ";'", position);
                    return '&#x' + text + ';';
                }
            } else if (this._match('#')) {
                const data = [];
                while (/[0-9]/.test(this._char)) {
                    data.push(this._char);
                    this._next();
                    if (this._char === undefined) {
                        this._unexpected();
                    }
                }
                this._assert(this._char === ';');
                if (data.length > 0) {
                    const text = data.join('');
                    const value = parseInt(text, 10);
                    this._assert(value <= 0x10FFFF, "Invalid value '&#" + text + ";'", position);
                    return '&#' + text + ';';
                }
            } else {
                const name = this._name();
                this._assert(name !== null);
                this._assert(this._char === ';');
                return '&' + name + ';';
            }
        }
        this._unexpected();
    }
    /* eslint-enable consistent-return */

    _comment() {
        const data = this._terminal('--');
        const node = this._document().createComment(data);
        this._appendChild(node);
        this._expect('>');
    }

    _processingInstruction() {
        this._next();
        const name = this._entityName();
        let whitespace = this._char === '?' ? false : this._whitespace(1);
        const position = this._position;
        const data = this._terminal('?>');
        if (name.toLowerCase() === 'xml') {
            this._seek(position);
            this._assert(name === 'xml', "'" + name + "' must be lower case");
            this._assert(this._start === this._prolog, "Prolog must start with XML declaration", this._start);
            this._assert(typeof this._data !== 'string', 'Invalid text declaration', this._start);
            const obj = { version: '', encoding: '', standalone: 'no' };
            for (const name of Object.keys(obj)) {
                const expect = (name == 'version' && this._context.length === 0) || (name == 'encoding' && this._context.length > 0);
                if ((whitespace || expect) && (expect ? this._expect(name) : this._match(name))) {
                    this._whitespace(0);
                    this._expect('=');
                    this._whitespace(0);
                    obj[name] = this._attributeValue();
                    whitespace = this._whitespace(0);
                }
            }
            this._expect('?>');
            obj.encoding = obj.encoding.toLowerCase();
            if (this._decoder.encoding && obj.encoding !== this._decoder.encoding) {
                const position = this._position;
                this._decoder = text.Decoder.open(this._data, obj.encoding);
                this._seek(position);
            }
            if (obj.version.length > 0) {
                const match = /^(\d)\.(\d)$/.exec(obj.version);
                this._assert(match && match[1] === '1', "Invalid XML version '" + obj.version + "'");
                const version = Number.parseInt(match[2], 10);
                if (version > this._version) {
                    /* eslint-disable */
                    this._charRegExp = /[\x01-\uD7FF\uE000-\uFFFD]/;
                    /* eslint-enable */
                    this._version = version;
                }
                this._assert(this._context.length === 0 || this._context.some((context) => context.version >= this._version));
            }
            this._assert(obj.standalone === 'no' || (obj.standalone === 'yes' && !this._entity && this._context.length === 0));
            this._standalone = obj.standalone === 'yes';
        }
        const node = this._document().createProcessingInstruction(name, data);
        this._appendChild(node);
    }

    _whitespace(count) {
        const position = this._position;
        let index = 0;
        if (this._implicitSpace) {
            index++;
            this._implicitSpace = false;
        }
        while (this._char === ' ' || this._char === '\n' || this._char === '\r' || this._char === '\t' || (this._version > 0 && this._char === '\x85')) {
            index++;
            this._next();
        }
        if (index < count) {
            this._seek(position);
            this._unexpected();
        }
        return index > 0;
    }

    _pushResource(identifier, entity, stop) {
        const content = this._callback(identifier);
        this._pushBuffer(content, identifier, entity, stop);
    }

    _pushBuffer(data, base, entity, stop) {
        const signature = text.Decoder.open(data);
        const decoder = signature.encoding === 'utf-8' ? text.Decoder.open(data, 'utf-8') : signature;
        this._pushContext(decoder, data, base, entity, stop, false);
        this._data = data;
    }

    _pushString(value, entity, stop) {
        const decoder = text.Decoder.open(value);
        this._pushContext(decoder, value, this._base, entity, stop);
    }

    _pushContext(decoder, data, base, entity, stop) {
        if (this._context.some((context) => context && context.base === base && context.entity === entity)) {
            this._assert(!entity, "Recursive entity '" + entity + "'");
            this._assert(!base, "Recursive base '" + base + "'");
        }
        if (base.length !== 0 || entity.length !== 0) {
            this._context.push(this._state);
        }
        this._stop = stop;
        this._entity = entity;
        this._base = base;
        this._data = data;
        this._decoder = decoder;
        this._prolog = this._decoder.position;
        this._char = '';
        this._next();
    }

    _popContext() {
        const entity = this._entity;
        this._state = this._context.pop();
        if (entity) {
            this._expect(';');
            this._implicitSpace = !this._context.some((context) => context.entity);
        }
    }

    get _state() {
        return {
            base: this._base,
            data: this._data,
            decoder: this._decoder,
            position: this._position,
            version: this._version,
            entity: this._entity,
            prolog: this._prolog,
            stop: this._stop,
        };
    }

    set _state(value) {
        this._stop = value.stop;
        this._base = value.base;
        this._data = value.data;
        this._decoder = value.decoder;
        this._seek(value.position);
        this._version = value.version;
        this._entity = value.entity;
        this._prolog = value.prolog;
    }

    _next() {
        if (this._char === undefined) {
            this._unexpected();
        }
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
        this._implicitSpace = false;
        if (this._parameterEntities && this._char === '%' && (this._entity || this._base)) {
            this._resolveParameterEntityReference();
        }
        if (!this._characterData) {
            if (this._char === '&' && (this._entity || this._base)) {
                const c = this._resolveEntityReference();
                if (c !== undefined) {
                    this._char = c;
                }
            }
        }
        if (this._char === '\uffff' || this._char === '\ufffe' || (this._version > 0 && this._char >= '\x7f' && this._char <= '\x9f' && this._char != '\x85')) {
            this._unexpected();
        }
        if (this._char === undefined) {
            if (!this._stop && this._context.length > 0) {
                this._popContext();
            }
        }
    }

    _seek(position) {
        this._decoder.position = position;
        this._char = '';
        this._next();
    }

    _expect(value) {
        if (!this._match(value)) {
            this._unexpected();
        }
        return true;
    }

    _match(value) {
        if (this._char !== value[0]) {
            return false;
        }
        if (value.length === 1) {
            this._next();
            return true;
        }
        if (this._context.length === 0) {
            const position = this._position;
            for (let i = 0; i < value.length; i++) {
                if (this._char !== value[i]) {
                    this._seek(position);
                    return false;
                }
                this._next();
            }
            return true;
        }
        const context = Array.from(this._context);
        const state = this._state;
        for (let i = 0; i < value.length; i++) {
            if (this._char !== value[i]) {
                this._context = context;
                this._state = state;
                return false;
            }
            this._next();
        }
        return true;
    }

    _assert(value, message, position) {
        if (value === false || value === undefined || value === null) {
            this._error(message, position);
        }
    }

    _error(message, position) {
        if (position) {
            this._parameterEntities = false;
            this._characterData = true;
            this._seek(position);
        }
        if (message) {
            throw new xml.Error(message + this._location());
        }
        this._unexpected();
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new xml.Error('Unexpected end of XML input.');
        } else if (c === '"') {
            c = 'string';
        } else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        } else {
            if (c < ' ' || c > '\x7F') {
                c = c.codePointAt(0);
                if (c < 0x0100) {
                    c = '\\x' + ('0' + c.toString(16)).slice(-2);
                } else if (c < 0x010000) {
                    c = '\\u' + ('000' + c.toString(16)).slice(-4);
                } else {
                    c = '\\u' + ('00000' + c.toString(16)).slice(-6);
                }
            }
            c = "token '" + c + "'";
        }
        this._error('Unexpected ' + c);
    }

    _location() {
        while (typeof this._data === 'string') {
            this._popContext();
        }
        this._parameterEntities = false;
        this._characterData = true;
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this._position) {
                break;
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            } else {
                column++;
            }
        }
        while (c !== undefined);
        return ' at ' + (this._base ? this._base + ':' : '') +  line.toString() + ':' + column.toString() + '.';
    }
};

xml.NodeList = class extends Array {

    constructor() {
        super();
    }

    item(index) {
        return this[index] || null;
    }
};

xml.Node = class {

    constructor(document, nodeType) {
        this._ownerDocument = document;
        this._nodeType = nodeType;
        this._childNodes = new xml.NodeList();
    }

    get ownerDocument() {
        return this._ownerDocument;
    }

    get nodeType() {
        return this._nodeType;
    }

    get localName() {
        throw new xml.Error('Not implemented.');
    }

    get namespaceURI() {
        return null;
    }

    get childNodes() {
        return this._childNodes;
    }

    get parentNode() {
        return this._parentNode;
    }

    set parentNode(value) {
        this._parentNode = value;
    }

    get firstChild() {
        return this._firstChild;
    }

    set firstChild(value) {
        this._firstChild = value;
    }

    get lastChild() {
        return this._lastChild || null;
    }

    set lastChild(value) {
        this._lastChild = value;
    }

    get previousSibling() {
        return this._previousSibling;
    }

    set previousSibling(value) {
        this._previousSibling = value;
    }

    get nextSibling() {
        return this._nextSibling;
    }

    set nextSibling(value) {
        this._nextSibling = value;
    }

    appendChild(newChild) {
        this.firstChild = this.firstChild || newChild;
        newChild.previousSibling = this.lastChild;
        if (newChild.previousSibling) {
            newChild.previousSibling.nextSibling = newChild;
        }
        this.lastChild = newChild;
        this.childNodes[this.childNodes.length] = newChild;
        newChild.parentNode = this;
    }

    lookupNamespaceURI(prefix) {
        switch (prefix) {
            case 'xml':
                return 'http://www.w3.org/XML/1998/namespace';
            case 'xmlns':
                return 'http://www.w3.org/2000/xmlns/';
            default:
                return null;
        }
    }
};

xml.Element = class extends xml.Node {

    constructor(document, namespaceURI, qualifiedName) {
        super(document, xml.NodeType.Element);
        this._namespaces = new Map();
        this._attributes = new xml.NamedNodeMap();
        this._namespaceURI = namespaceURI;
        if (namespaceURI === null) {
            this._prefix = null;
            this._localName = qualifiedName;
        } else {
            const pair = xml.Utility.split(qualifiedName);
            this._prefix = pair[0];
            this._localName = pair[1];
        }
    }

    get localName() {
        return this._localName;
    }

    get prefix() {
        return this._prefix;
    }

    get namespaceURI() {
        return this._namespaceURI;
    }

    get attributes() {
        return this._attributes;
    }

    get textContent() {
        return this.childNodes.map((node) => node.nodeType === xml.NodeType.ProcessingInstruction || node.nodeType === xml.NodeType.Comment ? '' : node.textContent).join('');
    }

    getElementsByTagName(tagName) {
        const list = new xml.NodeList();
        let node = this.firstChild;
        while (node) {
            if (node.nodeType === xml.NodeType.Element && (tagName === '*' || tagName === (node.prefix ? node.prefix + ':' + node.localName : node.localName))) {
                list.push(node);
            }
            node = node.nextSibling;
        }
        return list;
    }

    getAttribute(name) {
        const node = this.getAttributeNode(name);
        return node ? node.value || '' : '';
    }

    getAttributeNode(name) {
        return this.attributes.getNamedItem(name);
    }

    setAttributeNode(node) {
        const oldNode = this.attributes.setNamedItem(node);
        if (node.namespaceURI === 'http://www.w3.org/2000/xmlns/') {
            const prefix = node.prefix ? node.localName : '';
            this._namespaces.set(prefix, node.value);
        }
        return oldNode;
    }

    lookupNamespaceURI(prefix) {
        if (this._namespaces.has(prefix)) {
            return this._namespaces.get(prefix);
        }
        if (this.parentNode) {
            return this.parentNode.lookupNamespaceURI(prefix);
        }
        return super.lookupNamespaceURI(prefix);
    }
};

xml.Attribute = class extends xml.Node {

    constructor(document, namespaceURI, qualifiedName) {
        super(document, xml.NodeType.Attribute);
        this._namespaceURI = namespaceURI;
        if (namespaceURI === null) {
            this._prefix = null;
            this._localName = qualifiedName;
        } else {
            const pair = xml.Utility.split(qualifiedName);
            this._prefix = pair[0];
            this._localName = pair[1];
        }
    }

    get ownerElement() {
        return this._ownerElement;
    }

    set ownerElement(value) {
        this._ownerElement = value;
    }

    get localName() {
        return this._localName;
    }

    get prefix() {
        return this._prefix;
    }

    get namespaceURI() {
        return this._namespaceURI;
    }

    get value() {
        return this._value;
    }

    set value(value) {
        this._value = value;
    }
};

xml.CharacterData = class extends xml.Node {

    constructor(document, nodeType, data) {
        super(document, nodeType);
        this._data = data;
    }

    get data() {
        return this._data;
    }

    get textContent() {
        return this._data;
    }
};

xml.Text = class extends xml.CharacterData {

    constructor(document, data) {
        super(document, xml.NodeType.Text, data);
    }

    get localName() {
        return '#text';
    }
};

xml.CDataSection = class extends xml.CharacterData {

    constructor(document, data) {
        super(document, xml.NodeType.CDATA, data);
    }
};

xml.Entity = class extends xml.Node {

    constructor(document, name) {
        super(document, xml.NodeType.Entity);
        this._name = name;
        this._publicId = '';
        this._systemId = '';
        this._notationName = '';
        this._value = '';
    }

    get localName() {
        return this._name;
    }

    get publicId() {
        return this._publicId;
    }

    set publicId(value) {
        this._publicId = value;
    }

    get systemId() {
        return this._systemId;
    }

    set systemId(value) {
        this._systemId = value;
    }

    get notationName() {
        return this._notationName;
    }

    set notationName(value) {
        this._notationName = value;
    }

    set value(value) {
        this._value = value;
    }

    get value() {
        return this._value;
    }
};

xml.ProcessingInstruction = class extends xml.Node {

    constructor(document, target, data) {
        super(document, xml.NodeType.ProcessingInstruction);
        this._target = target;
        this._data = data;
    }

    get localName() {
        return this._target;
    }

    get target() {
        return this._target;
    }

    get data() {
        return this._data;
    }
};

xml.Comment = class extends xml.CharacterData {

    constructor(document, data) {
        super(document, xml.NodeType.Comment, data);
    }

    get localName() {
        return '#comment';
    }
};

xml.Document = class extends xml.Node {

    constructor() {
        super(null, xml.NodeType.Document);
        this._documentElement = null;
        this._documentType = null;
    }

    get documentElement() {
        return this._documentElement;
    }

    get documentType() {
        return this._documentType;
    }

    appendChild(newChild) {
        super.appendChild(newChild);
        if (newChild.nodeType === xml.NodeType.Element) {
            this._documentElement = newChild;
        }
        if (newChild.nodeType === xml.NodeType.DocumentType) {
            this._documentType = newChild;
        }
    }

    createElement(localName) {
        return new xml.Element(this, null, localName);
    }

    createElementNS(namespaceURI, qualifiedName) {
        return new xml.Element(this, namespaceURI, qualifiedName);
    }

    createAttribute(localName) {
        return new xml.Attribute(this, null, localName);
    }

    createAttributeNS(namespaceURI, qualifiedName) {
        return new xml.Attribute(this, namespaceURI, qualifiedName);
    }

    createTextNode(data) {
        return new xml.Text(this, data);
    }

    createCDATASection(data) {
        return new xml.CDataSection(this, data);
    }

    createProcessingInstruction(target, data) {
        return new xml.ProcessingInstruction(this, target, data);
    }

    createComment(data) {
        return new xml.Comment(this, data);
    }

    createDocumentType(qualifiedName, publicId, systemId) {
        return new xml.DocumentType(this, qualifiedName, publicId, systemId);
    }
};

xml.DocumentType = class extends xml.Node {

    constructor(document, qualifiedName, publicId, systemId) {
        super(document, xml.NodeType.DocumentType);
        this._name = qualifiedName;
        this._publicId = publicId;
        this._systemId = systemId;
        this._entities = new xml.NamedNodeMap();
    }

    get name() {
        return this._name;
    }

    get publicId() {
        return this._publicId;
    }

    get systemId() {
        return this._systemId;
    }

    get entities() {
        return this._entities;
    }

    appendChild(newChild) {
        if (newChild.nodeType === xml.NodeType.Entity) {
            this._entities.setNamedItem(newChild);
        }
    }

    createEntity(name) {
        return new xml.Entity(this.ownerDocument, name);
    }
};

xml.NamedNodeMap = class extends Array {

    getNamedItem(qualifiedName) {
        for (let i = this.length - 1; i >= 0; i--) {
            const node = this[i];
            const key = node.prefix ? node.prefix + ':' + node.localName : node.localName;
            if (qualifiedName == key) {
                return node;
            }
        }
        return null;
    }

    getNamedItemNS(namespaceURI, localName) {
        for (let i = this.length - 1; i >= 0; i--) {
            const node = this[i];
            if (localName === node.localName && namespaceURI == node.namespaceURI) {
                return node;
            }
        }
        return null;
    }

    setNamedItem(node) {
        const qualifiedName = node.prefix ? node.prefix + ':' + node.localName : node.localName;
        for (let i = this.length - 1; i >= 0; i--) {
            const node = this[i];
            const key = node.prefix ? node.prefix + ':' + node.localName : node.localName;
            if (qualifiedName == key) {
                const oldNode = this[i];
                this[i] = node;
                return oldNode;
            }
        }
        this.push(node);
        return null;
    }
};

xml.NodeType = {
    None: 0,
    Element: 1,
    Attribute: 2,
    Text: 3,
    CDATA: 4,
    EntityReference: 5,
    Entity: 6,
    ProcessingInstruction: 7,
    Comment: 8,
    Document: 9,
    DocumentType: 10,
    DocumentFragment: 11,
    Notation: 12
};

xml.Utility = class {

    static split(name) {
        const index = name.indexOf(':');
        if (index < 0 || index === name.length - 1) {
            return [ null, name ];
        }
        const localName = name.substring(index + 1);
        const c = localName.codePointAt(0);
        if (localName.indexOf(':') !== -1 || !xml.Utility.nameStartCharRegExp.test(String.fromCodePoint(c)) && (c < 0x10000 || c > 0xEFFFF)) {
            return [ null, name ];
        }
        const prefix = name.substring(0, index);
        return [ prefix, localName ];
    }
};

xml.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'XML Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.TextReader = xml.TextReader;
}
