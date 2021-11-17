
var xml = xml || {};
var text = text || require('./text');

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
        this._nameStartCharRegExp = /[:A-Z_a-z\xC0-\xD6\xD8-\xF6\xF8-\u02FF\u0370-\u037D\u037F-\u1FFF\u200C-\u200D\u2070-\u218F\u2C00-\u2FEF\u3001-\uD7FF\uF900-\uFDCF\uFDF0-\uFFFD]/;
        this._nameCharRegExp = new RegExp("[-.0-9\\xB7" + this._nameStartCharRegExp.source.slice(1, -1) + "]");
        this._entities = new Set([ 'quot', 'amp', 'apos', 'lt', 'gt' ]);
    }

    read() {
        const decoder = text.Decoder.open(this._data);
        this._decoder = decoder.encoding === 'utf-8' ? text.Decoder.open(this._data, 'utf-8') : decoder;
        const document = new xml.Document();
        this._stack = [];
        this._document = document;
        this._push(document);
        this._position = 0;
        this._version = 0;
        this._char = this._decoder.decode();
        for (;;) {
            const start = this._position;
            switch (this._char) {
                case '<': {
                    this._next();
                    switch (this._char) {
                        case '?': {
                            this._next();
                            const name = this._entityName();
                            let whitespace = this._char === '?' ? false : this._whitespace(1);
                            const position = this._position;
                            const data = this._terminal('?>');
                            if (name.toLowerCase() === 'xml') {
                                this._seek(position);
                                this._assert(name === 'xml', "'" + name + "' must be lower case");
                                this._assert(start === 0, "Prolog must start with XML declaration", start);
                                const obj = { version: '1.0', encoding: '', standalone: 'no' };
                                for (const name of Object.keys(obj)) {
                                    if (whitespace && (name == 'version' ? this._expect(name) : this._match(name))) {
                                        this._whitespace(0);
                                        this._expect('=');
                                        this._whitespace(0);
                                        obj[name] = this._attValue();
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
                                const version = /(\d)\.(\d)/.exec(obj.version);
                                this._assert(version && version[1] === '1', "Invalid XML version '" + obj.version + "'.");
                                this._version = Number.parseInt(version[2], 10);
                                this._assert(obj.standalone === 'no' || obj.standalone === 'yes');
                            }
                            const node = document.createProcessingInstruction(name, data);
                            this._appendChild(node);
                            break;
                        }
                        case '!': {
                            this._next();
                            if (this._match('--')) {
                                const data = this._terminal('--');
                                const node = document.createComment(data);
                                this._appendChild(node);
                                this._expect('>');
                            }
                            else if (this._match('[CDATA[')) {
                                this._assert(this._stack.length > 1);
                                const data = this._terminal(']]>');
                                const node = document.createCDATASection(data);
                                this._appendChild(node);
                            }
                            else if (this._match('DOCTYPE')) {
                                this._assert(this._stack.length > 1 || !document.documentElement);
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
                                }
                                else if (whitespace && this._match('PUBLIC')) {
                                    this._whitespace(1);
                                    publicId = this._pubidLiteral();
                                    this._whitespace(1);
                                    systemId = this._systemLiteral();
                                    this._whitespace(0);
                                    whitespace = true;
                                }
                                const node = document.createDocumentType(name, publicId, systemId);
                                if (systemId) {
                                    // this._assert(!systemId || this._callback);
                                    // const data = this._callback(systemId);
                                    // const reader = xml.TextReader.open(data);
                                }
                                this._appendChild(node);
                                if (whitespace && this._match('[')) {
                                    this._push(node);
                                }
                                else {
                                    this._expect('>');
                                }
                            }
                            else if (this._match('ENTITY')) {
                                const documentType = this._top();
                                this._assert(documentType.nodeType === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                const parsed = this._char === '%';
                                if (parsed) {
                                    this._next();
                                    this._whitespace(1);
                                }
                                const name = this._entityName();
                                const node = documentType.createEntity(name);
                                let whitespace = this._whitespace(0);
                                if (whitespace && (this._char === '"' || this._char === "'")) {
                                    this._entityValue();
                                    whitespace = this._whitespace(0);
                                }
                                else {
                                    if (whitespace && this._match('SYSTEM')) {
                                        this._whitespace(1);
                                        node.systemId = this._systemLiteral();
                                        whitespace = this._whitespace(0);
                                    }
                                    else if (whitespace && this._match('PUBLIC')) {
                                        this._whitespace(1);
                                        node.publicId = this._pubidLiteral();
                                        this._whitespace(1);
                                        node.systemId = this._systemLiteral();
                                        whitespace = this._whitespace(0);
                                    }
                                    else {
                                        this._unexpected();
                                    }
                                    if (whitespace && !parsed) {
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
                                this._appendChild(node);
                            }
                            else if (this._match('ELEMENT')) {
                                this._assert(this._nodeType() === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                const name = this._name();
                                this._assert(name !== null);
                                this._terminal('>');
                            }
                            else if (this._match('ATTLIST')) {
                                this._assert(this._nodeType() === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                const name = this._name();
                                this._assert(name !== null);
                                this._terminal('>');
                            }
                            else if (this._match('NOTATION')) {
                                this._assert(this._nodeType() === xml.NodeType.DocumentType);
                                this._whitespace(1);
                                /* const name = */ this._entityName();
                                let whitespace = this._whitespace(0);
                                if (whitespace && this._match('SYSTEM')) {
                                    this._whitespace(1);
                                    /* node.systemId = */ this._systemLiteral();
                                    whitespace = this._whitespace(0);
                                }
                                if (whitespace && this._match('PUBLIC')) {
                                    this._whitespace(1);
                                    /* node.publicId = */ this._pubidLiteral();
                                    if (this._whitespace(0)) {
                                        /* node.systemId = */ this._systemLiteral();
                                        whitespace = this._whitespace(0);
                                    }
                                }
                                this._expect('>');
                            }
                            else {
                                this._unexpected();
                            }
                            break;
                        }
                        case '/': {
                            this._next();
                            const position = this._position;
                            const name = this._name();
                            this._assert(name !== null);
                            this._whitespace(0);
                            if (!this._match('>')) {
                                this._unexpected();
                            }
                            const node = this._pop();
                            const nodeName = node.prefix ? node.prefix + ':' + node.localName : node.localName;
                            if (name !== nodeName) {
                                this._seek(position);
                                throw new xml.Error("Opening tag <" + nodeName + "> and ending tag </" + name + "> mismatch" + this._location());
                            }
                            break;
                        }
                        default: {
                            this._assert(this._stack.length > 1 || !document.documentElement);
                            const position = this._position;
                            const name = this._name();
                            this._assert(name !== null);
                            const attributes = new Map();
                            let whitespace = this._whitespace(0);
                            if (whitespace) {
                                while (this._char !== '/' && this._char !== '>') {
                                    if (!whitespace) {
                                        this._unexpected();
                                    }
                                    const position = this._position;
                                    const name = this._name();
                                    if (name) {
                                        if (attributes.has(name)) {
                                            this._assert(false, "'" + name + "' is a duplicate attribute name", position);
                                        }
                                        this._whitespace(0);
                                        this._expect('=');
                                        this._whitespace(0);
                                        const value = this._attValue();
                                        attributes.set(name, {
                                            position: position,
                                            qualifiedName: name,
                                            value: value
                                        });
                                        whitespace = this._whitespace(0);
                                        continue;
                                    }
                                    else {
                                        this._unexpected();
                                    }
                                }
                            }
                            const namespaces = new Map();
                            for (const attr of Array.from(attributes.values()).reverse()) {
                                const name = attr.qualifiedName;
                                const value = attr.value;
                                if (name === 'xml:space') {
                                    this._assert(value === 'preserve' || value === 'default', "Unexpected xml:space attribute value '" + value + "'.", position);
                                }
                                const index = name.indexOf(':');
                                if (index > 0) {
                                    attr.prefix = name.substring(0, index);
                                    attr.localName = name.substring(index + 1);
                                    if (attr.prefix === 'xmlns' && attr.localName) {
                                        this._assert(attr.localName !== 'xmlns', "Invalid namespace prefix '" + attr.localName + "'", attr.position);
                                        this._assert(this._version > 0 || value, "Invalid namespace declaration'", attr.position);
                                        namespaces.set(attr.localName, value);
                                    }
                                }
                                else {
                                    attr.prefix = null;
                                    attr.localName = name;
                                    if (attr.localName === 'xmlns') {
                                        namespaces.set('', value);
                                    }
                                }
                            }
                            const index = name.indexOf(':');
                            const prefix = (index < 0 || index === name.length - 1 ? null : name.substring(0, index)) || '';
                            const namespaceURI = namespaces.get(prefix) || this._lookupNamespaceURI(prefix);
                            const element = document.createElementNS(namespaceURI, name);
                            this._appendChild(element);
                            for (const entry of attributes.values()) {
                                const prefix = entry.prefix;
                                const namespaceURI = namespaces.get(prefix) || element.parentNode.lookupNamespaceURI(prefix);
                                const qualifiedName = entry.qualifiedName;
                                const attribute = document.createAttributeNS(namespaceURI, qualifiedName);
                                attribute.value = entry.value;
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
                    if (this._char === undefined) {
                        if (this._stack.length === 1 && this._nodeType() === xml.NodeType.Document) {
                            return this._pop();
                        }
                        this._unexpected();
                    }
                    switch (this._nodeType()) {
                        case xml.NodeType.Element: {
                            const append = (data) => {
                                const content = data.splice(0, data.length).join('');
                                if (content.trim().length > 0) {
                                    const node = document.createTextNode(content);
                                    this._appendChild(node);
                                }
                            };
                            const data = [];
                            while (this._char !== '<') {
                                if (this._char === undefined) {
                                    this._unexpected();
                                }
                                if (this._char === '&') {
                                    if (data.length > 0) {
                                        append(data);
                                    }
                                    const name = this._reference();
                                    if (!name) {
                                        this._unexpected();
                                    }
                                    this._validateEntityReference(name);
                                    const node = document.createEntityReference(name);
                                    this._appendChild(node);
                                }
                                else if (this._char === ']' && this._match(']]>')) {
                                    this._unexpected();
                                }
                                else if (this._char === '\x07' || this._char === '\x0c' || this._char === '\x1b') {
                                    this._unexpected();
                                }
                                else {
                                    data.push(this._char);
                                    this._next();
                                }
                            }
                            if (data.length > 0) {
                                append(data);
                            }
                            continue;
                        }
                        case xml.NodeType.DocumentType: {
                            if (this._char === '%') {
                                this._next();
                                const name = this._name();
                                if (name && this._match(';')) {
                                    const node = document.createEntityReference(name);
                                    this._appendChild(node);
                                    continue;
                                }
                            }
                            else if (this._char === ']') {
                                this._next();
                                this._whitespace(0);
                                this._expect('>');
                                this._pop();
                                continue;
                            }
                        }
                    }
                    if (this._whitespace(0)) {
                        continue;
                    }
                    this._unexpected();
                }
            }
        }
    }

    peek() {
        this._peek = true;
        const value = this.read();
        delete this._peek;
        return value;
    }

    _push(value) {
        this._stack.push(value);
    }

    _pop() {
        return this._stack.pop();
    }

    _top() {
        return this._stack[this._stack.length - 1];
    }

    _nodeType() {
        return this._top().nodeType;
    }

    _appendChild(newChild) {
        return this._top().appendChild(newChild);
    }

    _lookupNamespaceURI(prefix) {
        return this._top().lookupNamespaceURI(prefix);
    }

    _next() {
        if (this._char === undefined) {
            this._unexpected();
        }
        this._position = this._decoder.position;
        this._char = this._decoder.decode();
        if (this._char === '\uffff' || this._char === '\ufffe') {
            this._unexpected();
        }
    }

    _seek(position) {
        this._decoder.position = position;
        this._char = '';
        this._next();
    }

    _whitespace(count) {
        const position = this._position;
        let index = 0;
        while (this._char === ' ' || this._char === '\n' || this._char === '\r' || this._char === '\t' || (this._version === 1 && this._char === '\x85')) {
            index++;
            this._next();
        }
        if (index < count) {
            this._seek(position);
            this._unexpected();
        }
        return index > 0;
    }

    _assert(value, message, position) {
        if (value == false || value === undefined || value === null) {
            if (message) {
                if (position) {
                    this._seek(position);
                }
                throw new xml.Error(message + this._location());
            }
            this._unexpected();
        }
    }

    _match(text) {
        const position = this._position;
        let i = 0;
        while (i < text.length) {
            if (this._char === text[i]) {
                this._next();
                i++;
                continue;
            }
            i = -1;
            break;
        }
        if (i > 0) {
            return true;
        }
        this._seek(position);
        return false;
    }

    _expect(value) {
        if (!this._match(value)) {
            this._unexpected();
        }
        return true;
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
                    if (this._char === undefined) {
                        break;
                    }
                    c = this._char.charCodeAt(0);
                }
            }
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
        this._assert(name !== null, 'Expected entity name', position);
        this._assert(name.endsWith(':') || name.indexOf(':') < 0, 'Invalid colon in entity name', position);
        return name;
    }

    _entityValue() {
        const quote = this._char;
        if (quote !== '"' && quote !== "'") {
            this._unexpected();
        }
        this._next();
        const data = [];
        while (this._char !== quote) {
            if (this._char === '%') {
                this._unexpected();
            }
            if (this._char === '&') {
                const char = this._char;
                const reference = this._reference();
                if (!reference) {
                    this._unexpected();
                }
                data.push(char + reference + ';');
            }
            else {
                data.push(this._char);
                this._next();
            }
            if (this._char === undefined) {
                this._unexpected();
            }
        }
        this._next();
        return data.join('');
    }

    _attValue() {
        const quote = this._char;
        if (quote !== '"' && quote !== "'") {
            this._unexpected();
        }
        this._next();
        const data = [];
        while (this._char !== quote) {
            if (this._char === '<') {
                this._unexpected();
            }
            else if (this._char === '&') {
                const reference = this._reference();
                if (!reference) {
                    this._unexpected();
                }
                this._validateEntityReference(reference);
                data.push('&' + reference + ';');
            }
            else {
                data.push(this._char);
                this._next();
            }
            if (this._char === undefined) {
                this._unexpected();
            }
        }
        this._next();
        return data.join('');
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
        const value = this._quote();
        if (value.indexOf('#') >= 0) {
            this._unexpected();
        }
        return value;
    }

    _quote() {
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
        return data.join('');
    }

    _terminal(terminal) {
        const data = [];
        while (!this._match(terminal)) {
            if (this._char === undefined) {
                this._unexpected();
            }
            const c = this._char.charCodeAt(0);
            if (c !== 0x09 && c !== 0x0A && c !== 0x0D && (c < 0x20 || c > 0xD7FF) && (c < 0xE000 || c > 0xFFFD) && (c < 0x10000 || c > 0x10FFFF)) {
                this._unexpected();
            }
            data.push(this._char);
            this._next();
        }
        return data.join('');
    }

    _validateEntityReference(name) {
        if (!name.startsWith('#') && !this._entities.has(name) && !this._document.documentType) {
            this._assert(false, "Undefined ENTITY '" + name + "'.");
        }
    }

    _reference() {
        const position = this._position;
        if (this._char === '&') {
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
                if (data.length > 0 && this._match(';')) {
                    const value = data.join('');
                    this._assert(parseInt(value, 16) <= 0x10FFFF, "Invalid value '&#x" + value + ";'", position);
                    return '#x' + value;
                }
            }
            else if (this._match('#')) {
                const data = [];
                while (/[0-9]/.test(this._char)) {
                    data.push(this._char);
                    this._next();
                    if (this._char === undefined) {
                        this._unexpected();
                    }
                }
                if (data.length > 0 && this._match(';')) {
                    const value = data.join('');
                    this._assert(parseInt(value, 10) <= 0x10FFFF, "Invalid value '&#" + value + ";'", position);
                    return '#' + value;
                }
            }
            else {
                const name = this._name();
                this._assert(name !== null);
                if (this._match(';')) {
                    return name;
                }
            }
        }
        this._seek(position);
        return null;
    }

    _location() {
        let line = 1;
        let column = 1;
        this._decoder.position = 0;
        let c;
        do {
            if (this._decoder.position === this._position) {
                return ' at ' + line.toString() + ':' + column.toString() + '.';
            }
            c = this._decoder.decode();
            if (c === '\n') {
                line++;
                column = 1;
            }
            else {
                column++;
            }
        }
        while (c !== undefined);
        return ' at ' + line.toString() + ':' + column.toString() + '.';
    }

    _unexpected() {
        let c = this._char;
        if (c === undefined) {
            throw new xml.Error('Unexpected end of XML input.');
        }
        else if (c === '"') {
            c = 'string';
        }
        else if ((c >= '0' && c <= '9') || c === '-') {
            c = 'number';
        }
        else {
            if (c < ' ' || c > '\x7F') {
                c = c < '\u0100' ?
                    '\\x' + ('0' + c.charCodeAt(0).toString(16)).slice(-2) :
                    '\\u' + ('000' + c.charCodeAt(0).toString(16)).slice(-4);
            }
            c = "token '" + c + "'";
        }
        throw new xml.Error('Unexpected ' + c + this._location());
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
            case 'xml': return 'http://www.w3.org/XML/1998/namespace';
            case 'xmlns': return 'http://www.w3.org/2000/xmlns/';
        }
        return null;
    }
};

xml.Element = class extends xml.Node {

    constructor(document, namespaceURI, qualifiedName) {
        super(document, xml.NodeType.Element);
        this._namespaces = new Map();
        this._attributes = new xml.NamedNodeMap();
        this._namespaceURI = namespaceURI;
        const index = qualifiedName.indexOf(':');
        this._prefix = index < 0 || index === qualifiedName.length - 1 ? null : qualifiedName.substring(0, index);
        this._localName = index < 0 || index === qualifiedName.length - 1 ? qualifiedName : qualifiedName.substring(index + 1);
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
        return this.childNodes.map((node) => node.nodeType === xml.NodeType.ProcessingInstruction || node.nodeType === xml.NodeType.Commnet ? '' : node.textContent).join('');
    }

    getElementsByTagName(tagName) {
        const list = new xml.NodeList();
        let node = this.firstChild;
        while (node) {
            if (tagName === '*' || tagName === (node.prefix ? node.prefix + ':' + node.localName : node.localName)) {
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
            return this._namespaces.has(prefix);
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
        const index = qualifiedName.indexOf(':');
        this._prefix = index > 0 ? qualifiedName.substring(0, index) : null;
        this._localName = index > 0 ? qualifiedName.substring(index + 1) : qualifiedName;
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
};

xml.EntityReference = class extends xml.Node {

    constructor(document, name) {
        super(document, xml.NodeType.EntityReference);
        this._name = name;
    }

    get localName() {
        return this._name;
    }

    get textContent() {
        return '&' + this._name + ';';
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
            if (this._documentElement !== null) {
                throw new xml.Error('');
            }
            this._documentElement = newChild;
        }
        if (newChild.nodeType === xml.NodeType.DocumentType) {
            if (this._documentType !== null) {
                throw new xml.Error('');
            }
            this._documentType = newChild;
        }
    }

    createElementNS(namespaceURI, qualifiedName) {
        return new xml.Element(this, namespaceURI, qualifiedName);
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

    createEntityReference(name) {
        return new xml.EntityReference(this, name);
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

xml.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'XML Error';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.TextReader = xml.TextReader;
}
