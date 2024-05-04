
// Experimental Python Execution

const python = {};

python.Parser = class {

    constructor(text, file, debug) {
        this._tokenizer = new python.Tokenizer(text, file);
        this._debug = debug;
        python.Parser._precedence = python.Parser._precedence || {
            'or': 2, 'and': 3, 'not' : 4,
            'in': 5, 'instanceof': 5, 'is': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '<>': 5, '==': 5, '!=': 5,
            '|': 6, '^' : 7, '&' : 8,
            '<<': 9, '>>': 9, '+': 10, '-': 10, '*': 11, '@': 11, '/': 11, '//': 11, '%': 11,
            // '+': 12, '-': 12,
            '~': 13, '**': 14
        };
    }

    parse() {
        const node = this._node('program');
        node.body = [];
        while (!this._tokenizer.match('eof')) {
            const statement = this._statement();
            if (statement) {
                node.body.push(statement);
                continue;
            }
            if (this._tokenizer.eat('\n') || this._tokenizer.eat(';') || this._tokenizer.peek().type === 'eof') {
                continue;
            }
            if (this._tokenizer.eat('indent') && this._tokenizer.peek().type === 'eof') {
                continue;
            }
            throw new python.Error(`Unsupported statement ${this._tokenizer.location()}`);
        }
        return node;
    }

    _suite() {
        const node = this._node('block');
        node.statements = [];
        let statement = null;
        if (this._tokenizer.eat('\n')) {
            if (this._tokenizer.eat('indent')) {
                while (!this._tokenizer.eat('eof') && !this._tokenizer.eat('dedent')) {
                    if (this._tokenizer.eat(';')) {
                        continue;
                    }
                    statement = this._statement();
                    if (statement) {
                        node.statements.push(statement);
                        continue;
                    }
                    if (this._tokenizer.eat('\n')) {
                        continue;
                    }
                    if (this._tokenizer.match('dedent') || this._tokenizer.match('eof')) {
                        continue;
                    }
                    throw new python.Error(`Empty statement ${this._tokenizer.location()}`);
                }
            }
        } else if (!this._tokenizer.eat('eof')) {
            while (!this._tokenizer.match('\n') && !this._tokenizer.match('eof') && !this._tokenizer.match('dedent')) {
                if (this._tokenizer.eat(';')) {
                    continue;
                }
                statement = this._statement();
                if (statement) {
                    node.statements.push(statement);
                    continue;
                }
                throw new python.Error(`Empty statement ${this._tokenizer.location()}`);
            }
            this._tokenizer.eat('\n');
        }

        return node;
    }

    _statement() {
        let node = this._eat('id', 'break');
        if (node) {
            return node;
        }
        node = this._eat('id', 'continue');
        if (node) {
            return node;
        }
        node = this._eat('id', 'return');
        if (node) {
            node.expression = this._expression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'raise');
        if (node) {
            node.exception = this._expression(-1, ['from']);
            if (this._tokenizer.eat('id', 'from')) {
                node.from = this._expression();
            } else if (this._tokenizer.eat(',')) {
                node.exception = [node.exception];
                node.exception.push(this._expression());
                if (this._tokenizer.eat(',')) {
                    node.exception.push(this._expression());
                }
            }
            return node;
        }
        node = this._eat('id', 'assert');
        if (node) {
            node.condition = this._expression(-1, [',']);
            if (this._tokenizer.eat(',')) {
                node.message = this._expression();
            }
            return node;
        }
        node = this._eat('id', 'exec');
        if (node) {
            node.variable = this._expression(-1, ['in']);
            if (this._tokenizer.eat('in')) {
                do {
                    node.target = node.target || [];
                    node.target.push(this._expression(-1, ['in'], false));
                }
                while (this._tokenizer.eat(','));
            }
            return node;
        }

        node = this._eat('id', 'global');
        if (node) {
            node.names = [];
            do {
                node.names.push(this._name(true).value);
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'nonlocal');
        if (node) {
            node.names = [];
            do {
                node.names.push(this._name(true).value);
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'import');
        if (node) {
            node.names = [];
            do {
                const alias = this._node('alias');
                alias.name = this._dottedName();
                if (this._tokenizer.eat('id', 'as')) {
                    alias.asname = this._name(true).value;
                }
                node.names.push(alias);
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'from');
        if (node) {
            node.type = 'import_from';
            node.level = 0;
            const dots = this._tokenizer.peek();
            if (dots && Array.from(dots.type).every((c) => c === '.')) {
                this._eat(dots.type);
                node.level = Array.from(dots.type).length;
            }
            node.module = this._dottedName();
            this._tokenizer.expect('id', 'import');
            node.names = [];
            const close = this._tokenizer.eat('(');
            do {
                const alias = this._node('alias');
                alias.name = this._name(true).value;
                if (this._tokenizer.eat('id', 'as')) {
                    alias.asname = this._name(true).value;
                }
                node.names.push(alias);
            }
            while (this._tokenizer.eat(','));
            if (close) {
                this._tokenizer.expect(')');
            }
            return node;
        }

        let decorator_list = this._decorator();

        node = this._eat('id', 'class');
        if (node) {
            node.name = this._name(true).value;
            if (decorator_list) {
                node.decorator_list = Array.from(decorator_list);
                decorator_list = null;
            }
            node.bases = this._tokenizer.peek().type === '(' ? this._arguments() : [];
            this._tokenizer.expect(':');
            node.body = this._suite();
            return node;
        }

        const async = this._eat('id', 'async');
        if (async &&
            !this._tokenizer.match('id', 'def') &&
            !this._tokenizer.match('id', 'with') &&
            !this._tokenizer.match('id', 'for')) {
            throw new python.Error(`Expected 'def', 'with' or 'for' ${this._tokenizer.location()}`);
        }

        node = this._eat('id', 'def');
        if (node) {
            if (async) {
                node.async = async;
            }
            node.name = this._name(true).value;
            if (decorator_list) {
                node.decorator_list = Array.from(decorator_list);
                decorator_list = null;
            }
            this._tokenizer.expect('(');
            node.parameters = this._parameters(')');
            if (this._tokenizer.eat('->')) {
                node.returnType = this._type();
            }
            this._tokenizer.expect(':');
            node.body = this._suite();
            return node;
        }

        if (decorator_list && decorator_list.length > 0) {
            throw new python.Error('Unexpected decorator.');
        }

        node = this._eat('id', 'del');
        if (node) {
            node.expression = this._expression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'print');
        if (node) {
            node.expression = this._expression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'if');
        if (node) {
            node.condition = this._expression();
            this._tokenizer.expect(':');
            node.then = this._suite();
            let current = node;
            this._tokenizer.eat('\n');
            while (this._tokenizer.eat('id', 'elif')) {
                current.else = this._node('if');
                current = current.else;
                current.condition = this._expression();
                this._tokenizer.expect(':');
                current.then = this._suite();
                this._tokenizer.eat('\n');
            }
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                current.else = this._suite();
            }
            return node;
        }
        node = this._eat('id', 'while');
        if (node) {
            node.condition = this._expression();
            this._tokenizer.expect(':');
            node.body = this._suite();
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                node.else = this._suite();
            }
            return node;
        }
        node = this._eat('id', 'pass');
        if (node) {
            return node;
        }
        node = this._eat('id', 'for');
        if (node) {
            node.variable = [];
            node.variable.push(this._expression(-1, ['in']));
            while (this._tokenizer.eat(',')) {
                if (this._tokenizer.match('id', 'in')) {
                    node.variable.push({});
                    break;
                }
                node.variable.push(this._expression(-1, ['in']));
            }
            this._tokenizer.expect('id', 'in');
            node.target = [];
            node.target.push(this._expression());
            while (this._tokenizer.eat(',')) {
                if (this._tokenizer.match(':')) {
                    node.target.push({});
                    break;
                }
                node.target.push(this._expression(-1, ['in']));
            }
            this._tokenizer.expect(':');
            node.body = this._suite();
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                node.else = this._suite();
            }
            return node;
        }
        node = this._eat('id', 'with');
        if (node) {
            if (async) {
                node.async = async;
            }
            node.item = [];
            do {
                const item = this._node();
                item.type = 'with_item';
                item.expression = this._expression();
                if (this._tokenizer.eat('id', 'as')) {
                    item.variable = this._expression();
                }
                node.item.push(item);
            }
            while (this._tokenizer.eat(','));
            this._tokenizer.expect(':');
            node.body = this._suite();
            return node;
        }
        node = this._eat('id', 'try');
        if (node) {
            this._tokenizer.expect(':');
            node.body = this._suite();
            node.except = [];
            while (this._tokenizer.match('id', 'except')) {
                const except = this._node('except');
                this._tokenizer.expect('id', 'except');
                except.clause = [];
                except.clause.push(this._expression());
                while (this._tokenizer.eat(',')) {
                    if (this._tokenizer.match(':') || this._tokenizer.match('as')) {
                        except.clause.push({});
                        break;
                    }
                    except.clause.push(this._expression());
                }
                if (this._tokenizer.eat('id', 'as')) {
                    except.variable = this._expression();
                }
                this._tokenizer.expect(':');
                except.body = this._suite();
                node.except.push(except);
            }
            if (this._tokenizer.match('id', 'else')) {
                node.else = this._node('else');
                this._tokenizer.expect('id', 'else');
                this._tokenizer.expect(':');
                node.else.body = this._suite();
            }
            if (this._tokenizer.match('id', 'finally')) {
                node.finally = this._node('finally');
                this._tokenizer.expect('id', 'finally');
                this._tokenizer.expect(':');
                node.finally.body = this._suite();
            }
            return node;
        }

        const expression = this._expression(-1, [], true);
        if (expression) {
            if (expression.type === 'id' && this._tokenizer.eat(':')) {
                node = this._node('var');
                node.name = expression.value;
                node.location = expression.location;
                node.variableType = this._expression(-1, ['=']);
                if (this._tokenizer.eat('=')) {
                    node.initializer = this._expression();
                }
                return node;
            }
            switch (expression.type) {
                case '=':
                case ':=':
                case '==':
                case '!=':
                case '+=':
                case '-=':
                case '*=':
                case '@=':
                case '/=':
                case '//=':
                case '**=':
                case '&=':
                case '|=':
                case '%=':
                case '>>=':
                case '<<=':
                case '>>':
                case '<<':
                case '>=':
                case '<=':
                case '<':
                case '>':
                case '%':
                case '^=':
                case '...':
                case 'call':
                case 'assert':
                case 'raise':
                case 'string':
                case 'list':
                case 'var':
                case '.':
                case '[]':
                case 'yield':
                case '+':
                case '-':
                case '*':
                case '**':
                case '@':
                case '/':
                case '//':
                case '~':
                case '&':
                case '^':
                case '|':
                case 'not':
                case 'id':
                case 'number':
                case 'in':
                case 'and':
                case 'or':
                case 'if':
                case 'for':
                case 'tuple':
                case 'lambda':
                case 'await':
                    return expression;
                default:
                    throw new python.Error(`Unhandled expression ${this._tokenizer.location()}`);
            }
        }

        return null;
    }

    _expression(minPrecedence, terminal, tuple) {
        minPrecedence = minPrecedence || -1;
        const terminalSet = new Set(terminal);
        const stack = [];
        for (;;) {
            let node = this._node();
            const token = this._tokenizer.peek();
            if (stack.length === 1 && terminalSet.has(token.value)) {
                break;
            }
            const precedence = python.Parser._precedence[token.value];
            if (precedence) {
                if (precedence >= minPrecedence) {
                    this._tokenizer.read();
                    node.type = token.value;
                    if (token.type === 'id' && (token.value === 'in' || token.value === 'not')) {
                        if (token.value === 'in') {
                            node.type = 'in';
                        } else if (this._tokenizer.eat('id', 'in')) {
                            node.type = 'not in';
                        } else {
                            node.type = 'not';
                            node.expression = this._expression(precedence, terminal, tuple === false ? false : true);
                            stack.push(node);
                            continue;
                        }
                    } else if (token.value === '~') {
                        node.type = '~';
                        node.expression = this._expression(precedence, terminal, tuple === false ? false : true);
                        stack.push(node);
                        continue;
                    } else if (token.type === 'id' && token.value === 'is') {
                        if (this._tokenizer.eat('id', 'not')) {
                            node.type = 'is not';
                        }
                    }
                    if (stack.length > 0) {
                        node.op = node.type;
                        node.type = 'binary';
                        node.left = stack.pop();
                        node.right = this._expression(precedence, terminal, tuple === true ? true : false);
                    } else {
                        node.op = node.type;
                        node.type = 'unary';
                        node.operand = this._expression(precedence, terminal, tuple === true ? true : false);
                    }
                    stack.push(node);
                    continue;
                }
            }
            if (this._tokenizer.eat(':=')) {
                node.type = ':=';
                node.target = stack.pop();
                node.expression = this._expression(-1, terminal, tuple === false ? false : true);
                stack.push(node);
                continue;
            }
            if (this._tokenizer.eat('=')) {
                node.type = '=';
                node.target = stack.pop();
                node.expression = this._expression(-1, terminal, tuple === false ? false : true);
                stack.push(node);
                continue;
            }
            switch (token.type) {
                case '-=':
                case '**=':
                case '*=':
                case '//=':
                case '/=':
                case '&=':
                case '%=':
                case '^=':
                case '+=':
                case '<<=':
                case '>>=':
                case '|=':
                case '@=':
                    node = this._node(token.type);
                    this._tokenizer.expect(token.type);
                    node.target = stack.pop();
                    node.expression = this._expression(-1, terminal, true);
                    stack.push(node);
                    continue;
                default:
                    break;
            }
            node = this._eat('id', 'if');
            if (node) {
                node.then = stack.pop();
                node.condition = this._expression();
                this._tokenizer.expect('id', 'else');
                node.else = this._expression();
                stack.push(node);
                continue;
            }
            while (this._tokenizer.match('id', 'for') || this._tokenizer.match('id', 'async')) {
                const async = this._eat('id', 'async');
                if (async && !this._tokenizer.match('id', 'for')) {
                    throw new python.Error(`Expected 'for' ${this._tokenizer.location()}`);
                }
                node = this._eat('id', 'for');
                if (node) {
                    if (async) {
                        node.async = async;
                    }
                    node.expression = stack.pop();
                    node.variable = this._expression(-1, ['in'], true);
                    this._tokenizer.expect('id', 'in');
                    node.target = this._expression(-1, ['for', 'if'], true);
                    while (this._tokenizer.eat('id', 'if')) {
                        node.condition = node.condition || [];
                        node.condition.push(this._expression(-1, ['for', 'if']));
                    }
                    stack.push(node);
                }
            }
            node = this._eat('id', 'lambda');
            if (node) {
                node.parameters = this._parameters(':');
                node.body = this._expression(-1, terminal, false);
                stack.push(node);
                continue;
            }
            node = this._eat('id', 'yield');
            if (node) {
                if (this._tokenizer.eat('id', 'from')) {
                    node.from = this._expression(-1, [], true);
                } else {
                    node.expression = [];
                    do {
                        node.expression.push(this._expression(-1, [], false));
                    }
                    while (this._tokenizer.eat(','));
                }
                stack.push(node);
                continue;
            }
            node = this._eat('id', 'await');
            if (node) {
                node.expression = this._expression(minPrecedence, terminal, tuple);
                stack.push(node);
                continue;
            }
            node = this._eat('.');
            if (node) {
                this._tokenizer.eat('\n');
                node.target = stack.pop();
                node.member = this._name();
                stack.push(node);
                continue;
            }
            if (this._tokenizer.peek().type === '(') {
                if (stack.length === 0) {
                    node = this._node('tuple');
                    const args = this._arguments();
                    if (args.length === 1) {
                        stack.push(args[0]);
                    } else {
                        node.value = args;
                        stack.push(node);
                    }
                } else {
                    node = this._node('call');
                    node.target = stack.pop();
                    node.args = this._arguments();
                    stack.push(node);
                }
                continue;
            }
            if (this._tokenizer.peek().type === '[') {
                if (stack.length === 0) {
                    stack.push(this._expressions());
                } else {
                    node = this._node('[]');
                    node.target = stack.pop();
                    node.arguments = this._slice();
                    stack.push(node);
                }
                continue;
            }
            if (this._tokenizer.peek().type === '{') {
                stack.push(this._dictOrSetMaker());
                continue;
            }
            node = this._node();
            const literal = this._literal();
            if (literal) {
                if (stack.length > 0 && literal.type === 'number' &&
                    (literal.value.startsWith('-') || literal.value.startsWith('+'))) {
                    node.type = literal.value.substring(0, 1);
                    literal.value = literal.value.substring(1);
                    node.left = stack.pop();
                    node.right = literal;
                    stack.push(node);
                } else if (stack.length === 1 && literal.type === 'string' && stack[0].type === 'string') {
                    stack[0].value += literal.value;
                } else {
                    if (literal.type === 'number') {
                        switch (literal.value) {
                            case 'inf': literal.value = Infinity; break;
                            case '-inf': literal.value = -Infinity; break;
                            default: break;
                        }
                    }
                    stack.push(literal);
                }
                continue;
            }
            if (this._tokenizer.peek().keyword) {
                break;
            }
            node = this._eat('...');
            if (node) {
                stack.push(node);
                continue;
            }
            const identifier = this._name();
            if (identifier) {
                stack.push(identifier);
                continue;
            }

            if (tuple === true && stack.length === 1 && this._tokenizer.eat(',')) {
                if (stack[0].type === 'tuple') {
                    [node] = stack;
                } else {
                    node = this._node('tuple');
                    node.value = [stack.pop()];
                    stack.push(node);
                }
                // for, bar, = <expr>
                if (this._tokenizer.peek().type === '=') {
                    continue;
                }
                if (!this._tokenizer.match('=') && !terminalSet.has(this._tokenizer.peek().value)) {
                    const nextTerminal = terminal.slice(0).concat([',', '=']);
                    const expression = this._expression(minPrecedence, nextTerminal, tuple);
                    if (expression) {
                        node.value.push(expression);
                        continue;
                    }
                }
                break;
            }
            break;
        }

        if (stack.length === 1) {
            return stack.pop();
        }
        if (stack.length !== 0) {
            throw new python.Error(`Unexpected expression ${this._tokenizer.location()}`);
        }
        return null;
    }

    _decorator() {
        let list = null;
        while (this._tokenizer.eat('@')) {
            const node = this._node('decorator');
            node.value = this._expression();
            if (!node.value || (node.value.type !== 'call' && node.value.type !== 'id' && node.value.type !== '.')) {
                throw new python.Error(`Invalid decorator ${this._tokenizer.location()}`);
            }
            this._tokenizer.eat('\n');
            list = list === null ? [] : list;
            list.push(node);
        }
        return list;
    }

    _dictOrSetMaker() {
        const list = [];
        this._tokenizer.expect('{');
        let dict = true;
        while (!this._tokenizer.eat('}')) {
            const item = this._expression(-1, [], false);
            if (item === null) {
                throw new python.Error(`Expected expression ${this._tokenizer.location()}`);
            }
            if (!this._tokenizer.eat(':')) {
                dict = false;
            }
            if (dict) {
                const value = this._expression(-1, [], false);
                if (value === null) {
                    throw new python.Error(`Expected expression ${this._tokenizer.location()}`);
                }
                list.push({ type: 'pair', key: item, value });
            } else {
                list.push(item);
            }
            this._tokenizer.eat(',');
            this._tokenizer.eat('\n');
            if (this._tokenizer.eat('}')) {
                break;
            }
        }
        if (dict) {
            return { type: 'dict', value: list };
        }
        return { type: 'set', value: list };
    }

    _expressions() {
        const list = [];
        this._tokenizer.expect('[');
        while (!this._tokenizer.eat(']')) {
            const expression = this._expression();
            if (expression === null) {
                throw new python.Error(`Expected expression ${this._tokenizer.location()}`);
            }
            list.push(expression);
            this._tokenizer.eat(',');
            while (this._tokenizer.eat('\n')) {
                // continue
            }
            if (this._tokenizer.eat(']')) {
                break;
            }
        }
        return { type: 'list', value: list };
    }

    _slice() {
        let node = { type: '::' };
        let list = [];
        const group = ['start', 'stop', 'step'];
        this._tokenizer.expect('[');
        while (!this._tokenizer.eat(']')) {
            if (this._tokenizer.eat(':')) {
                node[group.shift()] = { type: 'list', value: list };
                list = [];
                continue;
            }
            if (this._tokenizer.eat(',')) {
                // list.push({});
                continue;
            }
            if (this._tokenizer.peek().type !== ']') {
                const expression = this._expression();
                if (expression === null) {
                    throw new python.Error(`Expected expression ${this._tokenizer.location()}`);
                }
                list.push(expression);
            }
        }
        if (list.length > 0) {
            node[group.shift()] = { type: 'list', value: list };
        }
        if (node.start && !node.stop && !node.step) {
            node = node.start;
        }
        return node;
    }

    _name(required) {
        const token = this._tokenizer.peek();
        if (token.type === 'id' && !token.keyword) {
            this._tokenizer.read();
            return token;
        }
        if (required) {
            throw new python.Error(`Invalid syntax ${this._tokenizer.location()}`);
        }
        return null;
    }

    _dottedName() {
        const list = [];
        do {
            list.push(this._name(true).value);
        }
        while (this._tokenizer.eat('.'));
        return list.join('.');
    }

    _literal() {
        const token = this._tokenizer.peek();
        if (token.type === 'string' || token.type === 'number' || token.type === 'boolean') {
            this._tokenizer.read();
            return token;
        }
        return null;
    }

    _typeArguments() {
        const list = [];
        this._tokenizer.expect('[');
        while (!this._tokenizer.eat(']')) {
            const type = this._type();
            if (type === null) {
                throw new python.Error(`Expected type ${this._tokenizer.location()}`);
            }
            list.push(type);
            if (!this._tokenizer.eat(',')) {
                this._tokenizer.expect(']');
                break;
            }
        }
        return list;
    }

    _type() {
        const type = this._node();
        type.type = 'type';
        type.name = this._expression(-1, ['[', '=']);
        if (type.name) {
            if (this._tokenizer.peek().value === '[') {
                type.arguments = this._typeArguments();
            }
            return type;
        }
        return null;
    }

    _parameter(terminal) {
        const node = this._node('parameter');
        if (this._tokenizer.eat('/')) {
            node.name = '/';
            return node;
        }
        if (this._tokenizer.eat('**')) {
            node.parameterType = '**';
        }
        if (this._tokenizer.eat('*')) {
            node.parameterType = '*';
        }
        const identifier = this._name();
        if (identifier !== null) {
            node.name = identifier.value;
            if (terminal !== ':' && this._tokenizer.eat(':')) {
                node.parameterType = this._type();
            }
            if (this._tokenizer.eat('=')) {
                node.initializer = this._expression();
            }
            return node;
        }
        return null;
    }

    _parameters(terminal) {
        const list = [];
        while (!this._tokenizer.eat(terminal)) {
            this._tokenizer.eat('\n');
            if (this._tokenizer.eat('(')) {
                list.push(this._parameters(')'));
            } else {
                list.push(this._parameter(terminal));
            }
            this._tokenizer.eat('\n');
            if (!this._tokenizer.eat(',')) {
                this._tokenizer.expect(terminal);
                break;
            }
        }
        return list;
    }

    _arguments() {
        const list = [];
        this._tokenizer.expect('(');
        while (!this._tokenizer.eat(')')) {
            if (this._tokenizer.eat('\n')) {
                continue;
            }
            const expression = this._expression(-1, [], false);
            if (expression === null) {
                throw new python.Error(`Expected expression ${this._tokenizer.location()}`);
            }
            list.push(expression);
            if (!this._tokenizer.eat(',')) {
                this._tokenizer.eat('\n');
                this._tokenizer.expect(')');
                break;
            }
        }
        return list;
    }

    _node(type) {
        const node = {};
        node.location = this._tokenizer.location();
        if (type) {
            node.type = type;
        }
        return node;
    }

    _eat(type, value) {
        if (this._tokenizer.match(type, value)) {
            const node = this._node(type === 'id' ? value : type);
            this._tokenizer.expect(type, value);
            return node;
        }
        return null;
    }
};

python.Tokenizer = class {

    constructor(text, file) {
        this._text = text;
        this._file = file;
        this._position = 0;
        this._lineStart = 0;
        this._line = 0;
        this._token = { type: '', value: '' };
        this._brackets = 0;
        this._indentation = [];
        this._outdent = 0;
        if (!python.Tokenizer._whitespace) {
            python.Tokenizer._whitespace = /[\u1680\u180e\u2000-\u200a\u202f\u205f\u3000\ufeff]/;
            const identifierStartChars = '\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0370-\u0374\u0376\u0377\u037a-\u037d\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u0527\u0531-\u0556\u0559\u0561-\u0587\u05d0-\u05ea\u05f0-\u05f2\u0620-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06e5\u06e6\u06ee\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u07f4\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828\u0840-\u0858\u08a0\u08a2-\u08ac\u0904-\u0939\u093d\u0950\u0958-\u0961\u0971-\u0977\u0979-\u097f\u0985-\u098c\u098f\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc\u09dd\u09df-\u09e1\u09f0\u09f1\u0a05-\u0a0a\u0a0f\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32\u0a33\u0a35\u0a36\u0a38\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0\u0ae1\u0b05-\u0b0c\u0b0f\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32\u0b33\u0b35-\u0b39\u0b3d\u0b5c\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99\u0b9a\u0b9c\u0b9e\u0b9f\u0ba3\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c33\u0c35-\u0c39\u0c3d\u0c58\u0c59\u0c60\u0c61\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0\u0ce1\u0cf1\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d60\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32\u0e33\u0e40-\u0e46\u0e81\u0e82\u0e84\u0e87\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa\u0eab\u0ead-\u0eb0\u0eb2\u0eb3\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065\u1066\u106e-\u1070\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u13a0-\u13f4\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f0\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1877\u1880-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191c\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19c1-\u19c7\u1a00-\u1a16\u1a20-\u1a54\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5\u1cf6\u1d00-\u1dbf\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u212f-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u2e2f\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312d\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fcc\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua61f\ua62a\ua62b\ua640-\ua66e\ua67f-\ua697\ua6a0-\ua6ef\ua717-\ua71f\ua722-\ua788\ua78b-\ua78e\ua790-\ua793\ua7a0-\ua7aa\ua7f8-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9cf\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa76\uaa7a\uaa80-\uaaaf\uaab1\uaab5\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uabc0-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40\ufb41\ufb43\ufb44\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff21-\uff3a\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc';
            const identifierChars = '\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u0620-\u0649\u0672-\u06d3\u06e7-\u06e8\u06fb-\u06fc\u0730-\u074a\u0800-\u0814\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0840-\u0857\u08e4-\u08fe\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\u0966-\u096f\u0981-\u0983\u09bc\u09be-\u09c4\u09c7\u09c8\u09d7\u09df-\u09e0\u0a01-\u0a03\u0a3c\u0a3e-\u0a42\u0a47\u0a48\u0a4b-\u0a4d\u0a51\u0a66-\u0a71\u0a75\u0a81-\u0a83\u0abc\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0ae6-\u0aef\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47\u0b48\u0b4b-\u0b4d\u0b56\u0b57\u0b5f-\u0b60\u0b66-\u0b6f\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7\u0be6-\u0bef\u0c01-\u0c03\u0c46-\u0c48\u0c4a-\u0c4d\u0c55\u0c56\u0c62-\u0c63\u0c66-\u0c6f\u0c82\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5\u0cd6\u0ce2-\u0ce3\u0ce6-\u0cef\u0d02\u0d03\u0d46-\u0d48\u0d57\u0d62-\u0d63\u0d66-\u0d6f\u0d82\u0d83\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2\u0df3\u0e34-\u0e3a\u0e40-\u0e45\u0e50-\u0e59\u0eb4-\u0eb9\u0ec8-\u0ecd\u0ed0-\u0ed9\u0f18\u0f19\u0f20-\u0f29\u0f35\u0f37\u0f39\u0f41-\u0f47\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u1000-\u1029\u1040-\u1049\u1067-\u106d\u1071-\u1074\u1082-\u108d\u108f-\u109d\u135d-\u135f\u170e-\u1710\u1720-\u1730\u1740-\u1750\u1772\u1773\u1780-\u17b2\u17dd\u17e0-\u17e9\u180b-\u180d\u1810-\u1819\u1920-\u192b\u1930-\u193b\u1951-\u196d\u19b0-\u19c0\u19c8-\u19c9\u19d0-\u19d9\u1a00-\u1a15\u1a20-\u1a53\u1a60-\u1a7c\u1a7f-\u1a89\u1a90-\u1a99\u1b46-\u1b4b\u1b50-\u1b59\u1b6b-\u1b73\u1bb0-\u1bb9\u1be6-\u1bf3\u1c00-\u1c22\u1c40-\u1c49\u1c5b-\u1c7d\u1cd0-\u1cd2\u1d00-\u1dbe\u1e01-\u1f15\u200c\u200d\u203f\u2040\u2054\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2d81-\u2d96\u2de0-\u2dff\u3021-\u3028\u3099\u309a\ua640-\ua66d\ua674-\ua67d\ua69f\ua6f0-\ua6f1\ua7f8-\ua800\ua806\ua80b\ua823-\ua827\ua880-\ua881\ua8b4-\ua8c4\ua8d0-\ua8d9\ua8f3-\ua8f7\ua900-\ua909\ua926-\ua92d\ua930-\ua945\ua980-\ua983\ua9b3-\ua9c0\uaa00-\uaa27\uaa40-\uaa41\uaa4c-\uaa4d\uaa50-\uaa59\uaa7b\uaae0-\uaae9\uaaf2-\uaaf3\uabc0-\uabe1\uabec\uabed\uabf0-\uabf9\ufb20-\ufb28\ufe00-\ufe0f\ufe20-\ufe26\ufe33\ufe34\ufe4d-\ufe4f\uff10-\uff19\uff3f';
            python.Tokenizer._identifierStart = new RegExp(`[${identifierStartChars}]`);
            /* eslint-disable no-misleading-character-class */
            python.Tokenizer._identifierChar = new RegExp(`[${identifierStartChars}${identifierChars}]`);
            /* eslint-enable no-misleading-character-class */
        }
    }

    peek() {
        if (!this._cache) {
            this._tokenize();
            this._cache = true;
        }
        return this._token;
    }

    read() {
        if (!this._cache) {
            this._tokenize();
        }
        const next = this._position + this._token.value.length;
        while (this._position < next) {
            if (python.Tokenizer._isNewline(this._get(this._position))) {
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            } else {
                this._position++;
            }
        }
        this._cache = false;
        return this._token;
    }

    match(type, value) {
        const token = this.peek();
        if (token.type === type && (!value || token.value === value)) {
            return true;
        }
        return false;
    }

    eat(type, value) {
        const token = this.peek();
        if (token.type === type && (!value || token.value === value)) {
            this.read();
            return true;
        }
        return false;
    }

    expect(type, value) {
        const token = this.peek();
        if (token.type !== type) {
            throw new python.Error(`Unexpected '${token.value}' instead of '${type}' ${this.location()}`);
        }
        if (value && token.value !== value) {
            throw new python.Error(`Unexpected '${token.value}' instead of '${value}' ${this.location()}`);
        }
        this.read();
    }

    location() {
        const line = this._line + 1;
        const column = this._position - this._lineStart + 1;
        return `at ${this._file}:${line}:${column}.`;
    }

    static _isSpace(c) {
        switch (c) {
            case ' ':
            case '\t':
            case '\v': // 11
            case '\f': // 12
            case '\xA0': // 160
                return true;
            default:
                if (c.charCodeAt(0) >= 0x1680) {
                    return python.Tokenizer._whitespace.test(c);
                }
                return false;
        }
    }

    static _isNewline(c) {
        switch (c) {
            case '\n':
            case '\r':
            case '\u2028': // 8232
            case '\u2029': // 8233
                return true;
            default:
                return false;
        }
    }

    static _isIdentifierStartChar(c) {
        if (c < 'A') {
            return c === '$';
        }
        if (c <= 'Z') {
            return true;
        }
        if (c < 'a') {
            return c === '_';
        }
        if (c <= 'z') {
            return true;
        }
        const code = c.charCodeAt(0);
        if (code >= 0xAA) {
            return python.Tokenizer._identifierStart.test(c);
        }
        return false;
    }

    static _isIdentifierChar(c) {
        if (c < '0') {
            return c === '$';
        }
        if (c <= '9') {
            return true;
        }
        if (c < 'A') {
            return false;
        }
        if (c <= 'Z') {
            return true;
        }
        if (c < 'a') {
            return c === '_';
        }
        if (c <= 'z') {
            return true;
        }
        const code = c.charCodeAt(0);
        if (code >= 0xAA) {
            return python.Tokenizer._identifierChar.test(c);
        }
        return false;
    }

    _get(position) {
        return position >= this._text.length ? '\0' : this._text[position];
    }

    _skipLine() {
        while (this._position < this._text.length) {
            if (python.Tokenizer._isNewline(this._get(this._position))) {
                break;
            }
            this._position++;
        }
    }

    _skipWhitespace() {
        while (this._position < this._text.length) {
            const c = this._text[this._position];
            if (c === '#') {
                this._skipLine();
            } else if (python.Tokenizer._isSpace(c)) {
                this._position++;
            } else if (c === '\\') {
                // Explicit Line Continuation
                this._position++;
                if (python.Tokenizer._isNewline(this._get(this._position))) {
                    this._position = this._newLine(this._position);
                    this._lineStart = this._position;
                    this._line++;
                } else {
                    throw new python.Error(`Unexpected '${this._text[this._position]}' after line continuation ${this.location()}`);
                }
            } else if (this._brackets > 0 && python.Tokenizer._isNewline(c)) {
                // Implicit Line Continuation
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            } else {
                break;
            }
        }
    }

    _newLine(position) {
        if ((this._get(position) === '\n' && this._get(position + 1) === '\r') ||
            (this._get(position) === '\r' && this._get(position + 1) === '\n')) {
            return position + 2;
        }
        return position + 1;
    }

    _tokenize() {
        if (this._token.type !== '\n') {
            this._skipWhitespace();
        }
        if (this._token.type === 'dedent') {
            this._indentation.pop();
            this._outdent--;
            if (this._outdent > 0) {
                this._token = { type: 'dedent', value: '' };
                return;
            }
        }
        if (this._token.type === '\n') {
            let indent = '';
            let i = this._position;
            while (i < this._text.length) {
                const c = this._text[i];
                if (python.Tokenizer._isSpace(c)) {
                    indent += c;
                    i++;
                } else if (python.Tokenizer._isNewline(c)) {
                    indent = '';
                    i = this._newLine(i);
                    this._position = i;
                    this._lineStart = i;
                    this._line++;
                } else if (c === '#') {
                    indent = '';
                    while (i < this._text.length && !python.Tokenizer._isNewline(this._text[i])) {
                        i++;
                    }
                    continue;
                } else {
                    break;
                }
            }
            let type = null;
            if (indent.length > 0) {
                const current = this._indentation.length > 0 ? this._indentation[this._indentation.length - 1] : '';
                if (indent.length > current.length) {
                    type = 'indent';
                    this._indentation.push(indent);
                } else if (indent.length > 0 && indent.length < current.length) {
                    type = 'dedent';
                    this._outdent = 0;
                    for (let j = this._indentation.length - 1; j >= 0 && indent.length < this._indentation[j].length; j--) {
                        this._outdent++;
                    }
                } else {
                    this._position += indent.length;
                }
            } else if (i >= this._text.length) {
                this._token = { type: 'eof', value: '' };
                return;
            } else if (this._indentation.length > 0) {
                type = 'dedent';
                this._outdent = this._indentation.length;
            }
            if (type === 'indent' || type === 'dedent') {
                this._token = { type, value: indent };
                return;
            }
        }
        if (this._position >= this._text.length) {
            this._token = { type: 'eof', value: '' };
            return;
        }
        const c = this._get(this._position);
        const string = this._string();
        if (string) {
            this._token = string;
            return;
        }
        switch (c) {
            case '(':
            case '[':
            case '{':
                this._brackets++;
                this._token = { type: c, value: c };
                return;
            case ')':
            case ']':
            case '}':
                if (this._brackets === 0) {
                    throw new python.Error(`Unexpected '${c}' ${this.location}`);
                }
                this._brackets--;
                this._token = { type: c, value: c };
                return;
            case ',':
            case ';':
            case '?':
                this._token = { type: c, value: c };
                return;
            default: {
                const number = this._number();
                if (number) {
                    this._token = number;
                    return;
                }
                if (c === '.') {
                    let end = this._position + 1;
                    while (this._get(end) === '.') {
                        end++;
                    }
                    const text = this._text.substring(this._position, end);
                    this._token = { type: text, value: text };
                    return;
                }
                const identifier = this._identifier();
                if (identifier) {
                    this._token = identifier;
                    return;
                }
                const operator = this._operator();
                if (operator) {
                    this._token = operator;
                    return;
                }
                break;
            }
        }
        if (c === '.') {
            this._token = { type: c, value: c };
            return;
        }
        if (c === '\\') {
            this._token = { type: '\\', value: c };
            return;
        }
        if (python.Tokenizer._isNewline(c)) {
            this._token = { type: '\n', value: this._text.substring(this._position, this._newLine(this._position)) };
            return;
        }
        throw new python.Error(`Unexpected token '${c}' ${this.location()}`);
    }

    _number() {
        const octal = (c) => c >= '0' && c <= '7' || c === '_';
        const binary = (c) => c === '0' || c === '1' || c === '_';
        const decimal = (c) => c >= '0' && c <= '9' || c === '_';
        const hex = (c) => decimal(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c === '_';
        let c = this._get(this._position);
        const sign = (c === '-' || c === '+') ? 1 : 0;
        let i = this._position + sign;
        c = this._get(i);
        if (c === '0') {
            let radix = 0;
            const n = this._get(i + 1);
            if ((n === 'x' || n === 'X') && hex(this._get(i + 2))) {
                i += 2;
                while (hex(this._get(i))) {
                    i += 1;
                }
                if (this._get(i) === 'l' || this._get(i) === 'L') {
                    i += 1;
                }
                radix = 16;
            } else if ((n === 'b' || n === 'B') && binary(this._get(i + 2))) {
                i += 2;
                while (binary(this._get(i))) {
                    i++;
                }
                radix = 2;
            } else if ((n === 'o' || n === 'O') && octal(this._get(i + 2))) {
                i += 2;
                while (octal(this._get(i))) {
                    i++;
                }
                radix = 8;
            } else if (n >= '0' && n <= '7') {
                i++;
                while (octal(this._get(i))) {
                    i += 1;
                }
                if (this._get(i) === 'l' || this._get(i) === 'L') {
                    i += 1;
                }
                radix = 8;
            }
            if (radix > 0 && this._get(i) !== '.') {
                const radixText = this._text.substring(this._position, i);
                const radixParseText = radixText.indexOf('_') === -1 ? radixText : radixText.split('_').join('');
                if (!isNaN(parseInt(radixParseText, radix))) {
                    return { type: 'number', value: radixText };
                }
            }
        }
        i = this._position + sign;
        let isDecimal = false;
        if (this._get(i) >= '1' && this._get(i) <= '9') {
            while (decimal(this._get(i))) {
                i++;
            }
            c = this._get(i).toLowerCase();
            isDecimal = c !== '.' && c !== 'e';
        }
        if (this._get(i) === '0') {
            i++;
            c = this._get(i).toLowerCase();
            isDecimal = !decimal(c) && c !== '.' && c !== 'e' && c !== 'j';
        }
        if (isDecimal) {
            if (this._get(i) === 'j' || this._get(i) === 'J' || this._get(i) === 'l' || this._get(i) === 'L') {
                return { 'type': 'number', value: this._text.substring(this._position, i + 1) };
            }
            const intText = this._text.substring(this._position, i);
            if (!isNaN(parseInt(intText, 10))) {
                return { type: 'number', value: intText };
            }
        }
        i = this._position + sign;
        if ((this._get(i) >= '0' && this._get(i) <= '9') ||
            (this._get(i) === '.' && this._get(i + 1) >= '0' && this._get(i + 1) <= '9')) {
            while (decimal(this._get(i))) {
                i++;
            }
            if (this._get(i) === '.') {
                i++;
            }
            while (decimal(this._get(i))) {
                i++;
            }
            if (i > (this._position + sign)) {
                if (this._get(i) === 'e' || this._get(i) === 'E') {
                    i++;
                    if (this._get(i) === '-' || this._get(i) === '+') {
                        i++;
                    }
                    if (decimal(this._get(i))) {
                        while (decimal(this._get(i))) {
                            i++;
                        }
                    } else {
                        i = this._position;
                    }
                } else {
                    while (decimal(this._get(i))) {
                        i++;
                    }
                }
            }
            if (i > (this._position + sign)) {
                if (this._get(i) === 'j' || this._get(i) === 'J') {
                    return { type: 'number', value: this._text.substring(this._position, i + 1) };
                }
                const floatText = this._text.substring(this._position, i);
                const floatParseText = floatText.indexOf('_') === -1 ? floatText : floatText.split('_').join('');
                if (!isNaN(parseFloat(floatParseText))) {
                    return { type: 'number', value: floatText };
                }
            }
        }
        return null;
    }

    _identifier() {
        let i = this._position;
        if (python.Tokenizer._isIdentifierStartChar(this._get(i))) {
            i++;
            while (python.Tokenizer._isIdentifierChar(this._get(i))) {
                i++;
            }
        }
        if (i > this._position) {
            const text = this._text.substring(this._position, i);
            let keyword = false;
            switch (text) {
                case 'and':
                case 'as':
                case 'else':
                case 'for':
                case 'if':
                case 'import':
                case 'in':
                case 'is':
                case 'not':
                case 'or':
                    keyword = true;
                    break;
                default:
                    keyword = false;
                    break;
            }
            return { type: 'id', value: text, keyword };
        }
        return null;
    }

    _operator() {
        let length = 0;
        const c0 = this._get(this._position);
        const c1 = this._get(this._position + 1);
        const c2 = this._get(this._position + 2);
        switch (c0) {
            case '+':
            case '&':
            case '|':
            case '^':
            case '=':
            case '!':
            case '%':
            case '~':
                length = c1 === '=' ? 2 : 1;
                break;
            case '-':
                length = c1 === '=' || c1 === '>' ? 2 : 1;
                break;
            case '*':
                switch (c1) {
                    case '*': length = c2 === '=' ? 3 : 2; break;
                    case '=': length = 2; break;
                    default: length = 1; break;
                }
                break;
            case '/':
                switch (c1) {
                    case '/': length = c2 === '=' ? 3 : 2; break;
                    case '=': length = 2; break;
                    default: length = 1; break;
                }
                break;
            case '<':
                switch (c1) {
                    case '>': length = 2; break;
                    case '<': length = c2 === '=' ? 3 : 2; break;
                    case '=': length = 2; break;
                    default: length = 1; break;
                }
                break;
            case '>':
                switch (c1) {
                    case '>': length = c2 === '=' ? 3 : 2; break;
                    case '=': length = 2; break;
                    default: length = 1; break;
                }
                break;
            case '@':
                length = c1 === '=' ? 2 : 1;
                break;
            case ':':
                length = c1 === '=' ? 2 : 1;
                break;
            default:
                return null;
        }
        const text = this._text.substring(this._position, this._position + length);
        return { type: text, value: text };
    }

    _string() {
        let i = this._position;
        let prefix = -1;
        if (this._get(i) === "'" || this._get(i) === '"') {
            prefix = '';
        } else if (this._get(i + 1) === "'" || this._get(i + 1) === '"') {
            const c = this._get(i);
            switch (c.toLowerCase()) {
                case 'b':
                case 'f':
                case 'r':
                case 'u':
                    prefix = c;
                    break;
                default:
                    break;
            }
        } else if (this._get(i + 2) === "'" || this._get(i + 2) === '"') {
            const cc = this._text.substr(this._position, 2);
            switch (cc.toLowerCase()) {
                case 'br':
                case 'fr':
                case 'rb':
                case 'rf':
                case 'ur':
                    prefix = cc;
                    break;
                default:
                    break;
            }
        }
        if (prefix.length >= 0) {
            i += prefix.length;
            let quote = '';
            let count = 0;
            const q0 = this._get(i);
            const q1 = this._get(i + 1);
            const q2 = this._get(i + 2);
            switch (q0) {
                case "'":
                    quote = q0;
                    count = (q1 === "'" && q2 === "'") ? 3 : 1;
                    break;
                case '"':
                    quote = q0;
                    count = (q1 === '"' && q2 === '"') ? 3 : 1;
                    break;
                default:
                    throw new python.Error(`Unsupported string quote '${q0}'.`);
            }
            i += count;
            if (count === 1) {
                while (i < this._text.length) {
                    if (this._text[i] === quote) {
                        return { type: 'string', value: this._text.substring(this._position, i + 1) };
                    } else if (this._text[i] === '\\' &&
                             (this._get(i + 1) === quote || this._get(i + 1) === '\n' || this._get(i + 1) === '\\')) {
                        i += 2;
                    } else if (this._text[i] === '\r' || this._text[i] === '\n') {
                        break;
                    } else {
                        i++;
                    }
                }
            } else if (count === 3) {
                while (i < this._text.length) {
                    if (this._get(i) === quote && this._get(i + 1) === quote && this._get(i + 2) === quote) {
                        return { type: 'string', value: this._text.substring(this._position, i + 3) };
                    } else if (this._get(i) === '\\' && this._get(i + 1) === quote) {
                        i += 2;
                        continue;
                    }
                    i++;
                }
            }
        }
        i = this._position;
        if (this._get(i) === '`') {
            i++;
            while (i < this._text.length) {
                if (this._text[i] === '`') {
                    return { type: 'string', value: this._text.substring(this._position, i + 1) };
                }
                i++;
            }
        }
        return null;
    }
};

python.Execution = class {

    constructor(sources) {
        const self = this;
        const execution = self;
        this._sources = sources || new Map();
        this._events = new Map();
        this._utf8Decoder = new TextDecoder('utf-8');
        this._unresolved = new Map();
        const dict = class extends Map {
            constructor(items) {
                super();
                if (items) {
                    for (const [name, value] of items) {
                        this.__setitem__(name, value);
                    }
                }
            }
            __contains__(key) {
                return this.has(key);
            }
            __setitem__(key, value) {
                this.set(key, value);
            }
            __getitem__(key) {
                return this.get(key);
            }
        };
        this._modules = new dict();
        this._registry = new Map();
        const module = class {
            constructor(name) {
                this.__name__ = name;
            }
        };
        const builtins = this.register('builtins', new module('builtins'));
        this._builtins = builtins;
        this._registry.set('__builtin__', builtins);
        this.registerType('builtins.type', class {}).__class__ = builtins.type;
        this.registerType('builtins.module', module);
        this.registerType('builtins.method', class {});
        this.registerType('builtins.function', class {});
        this.registerType('builtins.code', class {});
        this.import('builtins');
        this.registerType('builtins.builtin_function_or_method', class {});
        const typing = this.register('typing');
        this._typing = typing;
        const operator = this.register('operator');
        this.register('_codecs');
        this.register('argparse');
        this._enum = this.register('enum');
        this.register('collections');
        this.register('copy_reg');
        this.register('cuml');
        const datetime = this.register('datetime');
        this.register('gensim');
        this.register('io');
        const joblib = this.register('joblib');
        const functools = this.register('functools');
        this.registerType('functools.partial', class {});
        const keras = this.register('keras');
        const catboost = this.register('catboost');
        this.register('lightgbm');
        this.register('nolearn');
        const fastcore = this.register('fastcore');
        const fastai = this.register('fastai');
        const math = this.register('math');
        math.inf = Infinity;
        const numpy = this.register('numpy');
        this.register('numpy.core._multiarray_umath');
        this.register('numpy.matrixlib.defmatrix');
        const pandas = this.register('pandas');
        this.register('pandas._libs.tslib');
        this.register('pandas._libs.internals');
        const pickle = this.register('pickle');
        const sklearn = this.register('sklearn');
        this.register('sklearn.externals.joblib.numpy_pickle');
        const torch = this.register('torch');
        const torchvision = this.register('torchvision');
        this.register('torch.storage');
        this.register('torch.nn.parameter');
        this.register('torch.ops');
        this.register('torch._ops');
        this.register('torch.ops.torchvision');
        this.register('torch.ops.torchaudio');
        this.register('torch.ops._caffe2');
        this.register('torchvision');
        this.register('__torch__');
        this.register('sys').modules = this._modules;
        this.register('xgboost');
        this.registerType('builtins.dict', dict);
        this.registerType('builtins.ellipsis', class {});
        this.registerType('builtins.cell', class {});
        this.registerType('builtins.list', class extends Array {});
        this.registerType('builtins.number', class {});
        this.registerFunction('builtins.__import__', (name, globals, locals, fromlist, level) => {
            return execution.__import__(name, globals, locals, fromlist, level);
        });
        this.registerFunction('builtins.bool', (value) => {
            if (value) {
                if (value.__bool__) {
                    return value.__bool__();
                }
                if (value.__len__) {
                    return value.__len__() > 0;
                }
            }
            return false;
        });
        this.registerFunction('builtins.int', (value) => {
            if (value) {
                if (value.__int__) {
                    return value.__int__();
                }
                if (Number.isInteger(value)) {
                    return value;
                }
            }
            return NaN;
        });
        this.registerFunction('builtins.float', (value) => {
            if (value) {
                if (value.__float__) {
                    return value.__float__();
                }
                if (Number(value) === value) {
                    return value;
                }
            }
            return NaN;
        });
        this.registerFunction('builtins.str', (value) => {
            if (value && value.__str__) {
                return value.__str__();
            }
            return JSON.stringify(value);
        });
        this.registerType('builtins.complex', class {
            constructor(real, imaginary) {
                this.real = real;
                this.imag = imaginary;
            }
        });
        this.registerType('builtins.NoneType', class {});
        this.registerType('builtins.object', class {});
        this.registerType('builtins.tuple', class extends Array {
            constructor(items) {
                super(items ? items.length : 0);
                if (items) {
                    for (let i = 0; i < items.length; i++) {
                        this[i] = items[i];
                    }
                }
            }
        });
        this.registerType('builtins.staticmethod', class {});
        this.registerFunction('builtins.long', this.builtins.int);
        this.registerFunction('builtins.print', () => {});
        this.registerFunction('builtins.unicode');
        this.registerType('builtins.Warning', class {});
        this.registerType('builtins.FutureWarning', class extends builtins.Warning {});
        this.registerType('builtins.BaseException', class {});
        this.registerType('builtins.Exception', class extends builtins.BaseException {});
        this.registerType('builtins.AttributeError', class extends builtins.Exception {});
        this.registerType('builtins.SyntaxError', class extends builtins.Exception {});
        this.registerType('typing._Final', class {});
        this.registerType('typing._SpecialForm', class extends typing._Final {});
        this.registerType('typing._BaseGenericAlias', class extends typing._Final {});
        this.registerType('typing._GenericAlias', class extends typing._BaseGenericAlias {});
        this.registerType('typing._SpecialGenericAlias', class extends typing._BaseGenericAlias {});
        this.registerType('typing._TupleType', class extends typing._SpecialGenericAlias {});
        this.registerType('typing._CallableType', class {});
        this.registerFunction('typing.cast');
        typing.Any = Reflect.construct(typing._SpecialForm, []);
        typing.Callable = Reflect.construct(typing._CallableType, []);
        typing.Dict = Reflect.construct(typing._SpecialGenericAlias, []);
        typing.List = Reflect.construct(typing._SpecialGenericAlias, []);
        typing.Optional = Reflect.construct(typing._SpecialForm, []);
        typing.OrderedDict = Reflect.construct(typing._SpecialGenericAlias, []);
        typing.Sequence = Reflect.construct(typing._SpecialGenericAlias, []);
        typing.Tuple = Reflect.construct(typing._TupleType, []);
        typing.Union = Reflect.construct(typing._SpecialForm, []);
        this.registerType('enum.Enum', class {});
        this.registerFunction('operator.add');
        this.registerFunction('operator.eq');
        this.registerFunction('operator.ge');
        this.registerFunction('operator.gt');
        this.registerFunction('operator.mul');
        this.registerFunction('operator.mod');
        this.registerFunction('operator.le');
        this.registerFunction('operator.lt');
        this.registerFunction('operator.ne');
        this.registerFunction('operator.floordiv');
        this.registerFunction('operator.sub');
        this.registerFunction('sys.path.insert', () => {});
        this.registerType('argparse.Namespace', class {
            constructor(args) {
                this.args = args;
            }
        });
        this.registerType('catboost.core._CatBoostBase', class {});
        this.registerType('catboost.core.CatBoost', class extends catboost.core._CatBoostBase {});
        this.registerType('catboost.core.CatBoostClassifier', class extends catboost.core.CatBoost {
            load_model() {
                throw new python.Error("'catboost.core.CatBoostClassifier.load_model' not implemented.");
            }
        });
        catboost.CatBoostClassifier = catboost.core.CatBoostClassifier;
        this.registerType('collections.deque', class extends Array {
            constructor(iterable) {
                super();
                if (Array.isArray(iterable)) {
                    for (const value of iterable) {
                        this.push(value);
                    }
                }
            }
        });
        this.registerType('collections.OrderedDict', class extends dict {});
        this.registerType('cuml.common.array_descriptor.CumlArrayDescriptorMeta', class {});
        this.registerType('cuml.ensemble.randomforestclassifier.RandomForestClassifier', class {});
        this.registerType('cuml.raft.common.handle.Handle', class {
            __setstate__(state) {
                this._handle = state;
            }
        });
        this.registerType('datetime.date', class {});
        this.registerType('datetime.datetime', class extends datetime.date {});
        this.registerType('datetime.timedelta', class {});
        this.registerType('datetime.tzinfo', class {});
        this.registerType('datetime.timezone', class extends datetime.tzinfo {});
        this.registerType('dnnlib.tflib.network.Network', class {});
        this.registerType('dnnlib.util.EasyDict', class extends dict {});
        this.registerType('haiku._src.data_structures.FlatMapping', class {
            constructor(dict) {
                Object.assign(this, dict);
            }
        });
        this.registerType('haiku._src.data_structures.frozendict', class {
            constructor(obj) {
                Object.assign(this, obj);
            }
        });
        this.registerType('hmmlearn.hmm.GaussianHMM', class {});
        this.registerType('hmmlearn.hmm.MultinomialHMM', class {});
        this.registerType('hmmlearn.base.ConvergenceMonitor', class {});
        this.registerType('io.BytesIO', class {
            constructor(buf, mode) {
                this.mode = mode || 'r';
                this._buf = this.mode === 'w' ? null : buf;
                this._point = 0;
            }
            seek(offset) {
                this._point = offset;
            }
            read(size) {
                const start = this._point;
                this._point = size === undefined ? this._buf.length : start + size;
                return this._buf.subarray(start, this._point);
            }
            write(data) {
                const src = this._buf || new Uint8Array();
                this._point = src.length + data.length;
                this._buf = new Uint8Array(this._point);
                this._buf.set(src, 0);
                this._buf.set(data, src.length);
            }
        });
        this.registerType('numpy.dtype', class {
            constructor(obj, align, copy) {
                if (typeof obj === 'string' && (obj.startsWith('<') || obj.startsWith('>'))) {
                    this.byteorder = obj.substring(0, 1);
                    obj = obj.substring(1);
                } else {
                    this.byteorder = '=';
                }
                switch (obj) {
                    case 'b1': case 'bool': this.itemsize = 1; this.kind = 'b'; break;
                    case 'i1': case 'int8': this.itemsize = 1; this.kind = 'i'; break;
                    case 'i2': case 'int16': this.itemsize = 2; this.kind = 'i'; break;
                    case 'i4': case 'int32': this.itemsize = 4; this.kind = 'i'; break;
                    case 'i8': case 'int64': case 'int': this.itemsize = 8; this.kind = 'i'; break;
                    case 'u1': case 'uint8': this.itemsize = 1; this.kind = 'u'; break;
                    case 'u2': case 'uint16': this.itemsize = 2; this.kind = 'u'; break;
                    case 'u4': case 'uint32': this.itemsize = 4; this.kind = 'u'; break;
                    case 'u8': case 'uint64': case 'uint': this.itemsize = 8; this.kind = 'u'; break;
                    case 'f2': case 'float16': this.itemsize = 2; this.kind = 'f'; break;
                    case 'f4': case 'float32': this.itemsize = 4; this.kind = 'f'; break;
                    case 'f8': case 'float64': case 'float': this.itemsize = 8; this.kind = 'f'; break;
                    case 'c8': case 'complex64': this.itemsize = 8; this.kind = 'c'; break;
                    case 'c16': case 'complex128': case 'complex': this.itemsize = 16; this.kind = 'c'; break;
                    case 'M8': case 'M': this.itemsize = 8; this.kind = 'M'; break;
                    default:
                        if (obj.startsWith('V')) {
                            this.itemsize = parseInt(obj.substring(1), 10);
                            this.kind = 'V';
                        } else if (obj.startsWith('O')) {
                            this.itemsize = obj === 'O' ? 8 : parseInt(obj.substring(1), 10);
                            this.kind = 'O';
                        } else if (obj.startsWith('S')) {
                            this.itemsize = parseInt(obj.substring(1), 10);
                            this.kind = 'S';
                        } else if (obj.startsWith('U')) { // Unicode string
                            this.kind = 'U';
                            this.itemsize = 4 * parseInt(obj.substring(1), 10);
                        } else {
                            throw new python.Error(`Unsupported dtype '${obj}'.`);
                        }
                        break;
                }
                if (align) {
                    this.align = align;
                }
                if (copy) {
                    this.copy = copy;
                }
            }
            get str() {
                return (this.byteorder === '=' ? '<' : this.byteorder) + this.kind + this.itemsize.toString();
            }
            get name() {
                switch (this.kind) {
                    case 'V': return `void${this.itemsize === 0 ? '' : (this.itemsize * 8)}`;
                    case 'S': return `bytes${this.itemsize === 0 ? '' : (this.itemsize * 8)}`;
                    case 'U': return `str${this.itemsize === 0 ? '' : (this.itemsize * 8)}`;
                    case 'M': return 'datetime64';
                    case 'b': return 'bool';
                    default: return this.__name__;
                }
            }
            __setstate__(state) {
                switch (state.length) {
                    case 8:
                        [
                            this.version, this.byteorder, this.subarray, this.names,
                            this.fields, this.elsize, this.alignment, this.int_dtypeflags
                        ] = state;
                        break;
                    case 9:
                        [
                            this.version, this.byteorder, this.subarray, this.names,
                            this.fields, this.elsize, this.alignment, this.int_dtypeflags,
                            this.metadata
                        ] = state;
                        break;
                    default:
                        throw new python.Error(`Unsupported numpy.dtype setstate length '${state.length}'.`);
                }
            }
            get __name__() {
                switch (this.kind) {
                    case 'b':
                        switch (this.itemsize) {
                            case 1: return 'boolean';
                            default: throw new python.Error(`Unsupported boolean itemsize '${this.itemsize}'.`);
                        }
                    case 'i':
                        switch (this.itemsize) {
                            case 1: return 'int8';
                            case 2: return 'int16';
                            case 4: return 'int32';
                            case 8: return 'int64';
                            default: throw new python.Error(`Unsupported int itemsize '${this.itemsize}'.`);
                        }
                    case 'u':
                        switch (this.itemsize) {
                            case 1: return 'uint8';
                            case 2: return 'uint16';
                            case 4: return 'uint32';
                            case 8: return 'uint64';
                            default: throw new python.Error(`Unsupported uint itemsize '${this.itemsize}'.`);
                        }
                    case 'f':
                        switch (this.itemsize) {
                            case 2: return 'float16';
                            case 4: return 'float32';
                            case 8: return 'float64';
                            default: throw new python.Error(`Unsupported float itemsize '${this.itemsize}'.`);
                        }
                    case 'c':
                        switch (this.itemsize) {
                            case 8: return 'complex64';
                            case 16: return 'complex128';
                            default: throw new python.Error(`Unsupported complex itemsize '${this.itemsize}'.`);
                        }
                    case 'S':
                    case 'U':
                        return 'string';
                    case 'M':
                        return 'datetime';
                    case 'O':
                        return 'object';
                    case 'V':
                        return 'void';
                    default:
                        throw new python.Error(`Unsupported dtype kind '${this.kind}'.`);
                }
            }
        });
        this.registerType('numpy.generic', class {});
        this.registerType('numpy.inexact', class {});
        this.registerType('numpy.bool_', class extends numpy.generic {});
        this.registerType('numpy.number', class extends numpy.generic {});
        this.registerType('numpy.integer', class extends numpy.number {});
        this.registerType('numpy.floating', class extends numpy.inexact {});
        this.registerType('numpy.float16', class extends numpy.floating {});
        this.registerType('numpy.float32', class extends numpy.floating {});
        this.registerType('numpy.float64', class extends numpy.floating {});
        this.registerType('numpy.signedinteger', class extends numpy.integer {});
        this.registerType('numpy.int8', class extends numpy.signedinteger {});
        this.registerType('numpy.int16', class extends numpy.signedinteger {});
        this.registerType('numpy.int32', class extends numpy.signedinteger {});
        this.registerType('numpy.int64', class extends numpy.signedinteger {});
        this.registerType('numpy.unsignedinteger', class extends numpy.integer {});
        this.registerType('numpy.uint8', class extends numpy.unsignedinteger {});
        this.registerType('numpy.uint16', class extends numpy.unsignedinteger {});
        this.registerType('numpy.uint32', class extends numpy.unsignedinteger {});
        this.registerType('numpy.uint64', class extends numpy.unsignedinteger {});
        this.registerType('numpy.datetime64', class extends numpy.generic {});
        this.registerType('gensim.models.doc2vec.Doctag', class {});
        this.registerType('gensim.models.doc2vec.Doc2Vec', class {});
        this.registerType('gensim.models.doc2vec.Doc2VecTrainables', class {});
        this.registerType('gensim.models.doc2vec.Doc2VecVocab', class {});
        this.registerType('gensim.models.fasttext.FastText', class {});
        this.registerType('gensim.models.fasttext.FastTextTrainables', class {});
        this.registerType('gensim.models.fasttext.FastTextVocab', class {});
        this.registerType('gensim.models.fasttext.FastTextKeyedVectors', class {});
        this.registerType('gensim.models.keyedvectors.Doc2VecKeyedVectors', class {});
        this.registerType('gensim.models.keyedvectors.FastTextKeyedVectors', class {});
        this.registerType('gensim.models.keyedvectors.KeyedVectors', class {});
        this.registerType('gensim.models.keyedvectors.Vocab', class {});
        this.registerType('gensim.models.keyedvectors.Word2VecKeyedVectors', class {});
        this.registerType('gensim.models.phrases.Phrases', class {});
        this.registerType('gensim.models.tfidfmodel.TfidfModel', class {});
        this.registerType('gensim.models.word2vec.Vocab', class {});
        this.registerType('gensim.models.word2vec.Word2Vec', class {});
        this.registerType('gensim.models.word2vec.Word2VecTrainables', class {});
        this.registerType('gensim.models.word2vec.Word2VecVocab', class {});
        this.registerFunction('gensim.models.tfidfmodel.df2idf');
        this.registerFunction('gensim.utils.call_on_class_only', () => {
            throw new builtins.AttributeError('This method should be called on a class object.');
        });
        this.registerFunction('gensim.utils.identity');
        this.registerType('google3.learning.deepmind.research.nbr.pbl_jax.clean_jaxline.utils.optimizers.ScaleByLarsState', class {
            constructor(obj) {
                Object.assign(this, obj);
            }
        });
        this.registerType('joblib.numpy_pickle.NumpyArrayWrapper', class {

            __setstate__(state) {
                this.subclass = state.subclass;
                this.dtype = state.dtype;
                this.shape = state.shape;
                this.order = state.order;
                this.allow_mmap = state.allow_mmap;
            }
            __read__(unpickler) {
                if (this.dtype.__name__ === 'object') {
                    return unpickler.load();
                }
                const size = this.dtype.itemsize * this.shape.reduce((a, b) => a * b, 1);
                this.data = unpickler.read(size);
                return execution.invoke(this.subclass, [this.shape, this.dtype, this.data]);
            }
        });
        this.registerType('joblib.numpy_pickle.NDArrayWrapper', class {

            __setstate__(state) {
                this.subclass = state.subclass;
                this.filename = state.state;
                this.allow_mmap = state.allow_mmap;
            }
            __read__(/* unpickler */) {
                return this; // return execution.invoke(this.subclass, [ this.shape, this.dtype, this.data ]);
            }
        });
        sklearn.externals.joblib.numpy_pickle.NDArrayWrapper = joblib.numpy_pickle.NDArrayWrapper;
        sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper = joblib.numpy_pickle.NumpyArrayWrapper;
        this.registerType('keras.engine.sequential.Sequential', class {});
        this.registerType('lasagne.layers.conv.Conv2DLayer', class {});
        this.registerType('lasagne.layers.dense.DenseLayer', class {});
        this.registerType('lasagne.layers.input.InputLayer', class {});
        this.registerType('lasagne.layers.pool.MaxPool2DLayer', class {});
        this.registerType('lightgbm.sklearn.LGBMRegressor', class {});
        this.registerType('lightgbm.sklearn.LGBMClassifier', class {});
        this.registerType('lightgbm.basic.Booster', class {
            constructor() {
                this.average_output = false;
                this.models = [];
                this.loaded_parameter = '';
            }
            __setstate__(state) {
                if (typeof state.handle === 'string') {
                    this.LoadModelFromString(state.handle);
                    return;
                }
                Object.assign(this, state);
            }
            LoadModelFromString(model_str) {
                const lines = model_str.split('\n');
                const signature = lines.shift() || '?';
                if (signature.trim() !== 'tree') {
                    throw new python.Error(`Invalid signature '${signature.trim()}'.`);
                }
                // GBDT::LoadModelFromString() in https://github.com/microsoft/LightGBM/blob/master/src/boosting/gbdt_model_text.cpp
                const key_vals = new Map();
                while (lines.length > 0 && !lines[0].startsWith('Tree=')) {
                    const cur_line = lines.shift().trim();
                    if (cur_line.length > 0) {
                        const strs = cur_line.split('=');
                        if (strs.length === 1) {
                            key_vals.set(strs[0], '');
                        } else if (strs.length === 2) {
                            key_vals.set(strs[0], strs[1]);
                        } else if (strs.length > 2) {
                            if (strs[0] === "feature_names") {
                                key_vals.set(strs[0], cur_line.substring("feature_names=".length));
                            } else if (strs[0] === 'monotone_constraints') {
                                key_vals.set(strs[0], cur_line.substring('monotone_constraints='.length));
                            } else {
                                throw new python.Error(`Wrong line: ${cur_line.substring(0, Math.min(128, cur_line.length))}`);
                            }
                        }
                    }
                }
                const atoi = (key, value) => {
                    if (key_vals.has(key)) {
                        return parseInt(key_vals.get(key), 10);
                    }
                    if (value !== undefined) {
                        return value;
                    }
                    throw new python.Error(`Model file does not specify ${key}.`);
                };
                const list = (key, size) => {
                    if (key_vals.has(key)) {
                        const value = key_vals.get(key).split(' ');
                        if (value.length !== size) {
                            throw new python.Error(`Wrong size of ${key}.`);
                        }
                        return value;
                    }
                    throw new python.Error(`Model file does not contain ${key}.`);
                };
                this.version = key_vals.get('version') || '';
                this.num_class = atoi('num_class');
                this.num_tree_per_iteration = atoi('num_tree_per_iteration', this.num_class);
                this.label_index = atoi('label_index');
                this.max_feature_idx = atoi('max_feature_idx');
                if (key_vals.has('average_output')) {
                    this.average_output = true;
                }
                this.feature_names = list('feature_names', this.max_feature_idx + 1);
                this.feature_infos = list('feature_infos', this.max_feature_idx + 1);
                if (key_vals.has('monotone_constraints')) {
                    this.monotone_constraints = list('monotone_constraints', this.max_feature_idx + 1);
                }
                if (key_vals.has('objective')) {
                    this.objective = key_vals.get('objective');
                }
                let tree = null;
                while (lines.length > 0) {
                    const text = lines.shift();
                    const line = text.trim();
                    if (line.length === 0) {
                        continue;
                    }
                    if (line.startsWith('Tree=')) {
                        tree = { index: parseInt(line.split('=').pop(), 10) };
                        this.models.push(tree);
                        continue;
                    }
                    if (line === 'end of trees') {
                        break;
                    }
                    const param = line.split('=');
                    if (param.length !== 2) {
                        throw new python.Error(`Invalid property '${line}'.`);
                    }
                    const name = param[0].trim();
                    const value = param[1].trim();
                    tree[name] = value;
                }
                const ss = [];
                let is_inparameter = false;
                while (lines.length > 0) {
                    const text = lines.shift();
                    const line = text.trim();
                    if (line === 'parameters:') {
                        is_inparameter = true;
                        continue;
                    } else if (line === 'end of parameters') {
                        break;
                    } else if (is_inparameter) {
                        ss.push(line);
                    }
                }
                if (ss.length > 0) {
                    this.loaded_parameter = ss.join('\n');
                }
            }
        });
        this.registerFunction('megengine.functional.elemwise.clip', () => {});
        this.registerFunction('megengine.functional.elemwise.sqrt', () => {});
        this.registerFunction('megengine.functional.nn.conv2d', () => {});
        this.registerFunction('megengine.functional.nn.relu', () => {});
        this.registerFunction('megengine.functional.nn.sigmoid', () => {});
        this.registerFunction('megengine.functional.tensor.arange', () => {});
        this.registerFunction('megengine.functional.tensor.broadcast_to', () => {});
        this.registerFunction('megengine.functional.tensor.concat', () => {});
        this.registerFunction('megengine.functional.tensor.expand_dims', () => {});
        this.registerFunction('megengine.functional.tensor.flatten', () => {});
        this.registerFunction('megengine.functional.tensor.full', () => {});
        this.registerFunction('megengine.functional.tensor.reshape', () => {});
        this.registerFunction('megengine.functional.tensor.split', () => {});
        this.registerFunction('megengine.functional.tensor.stack', () => {});
        this.registerFunction('megengine.functional.tensor.transpose', () => {});
        this.registerFunction('megengine.functional.vision.interpolate', () => {});
        this.registerFunction('megengine.module.qat.module.QATModule._apply_fakequant_with_observer', () => {});
        this.registerType('megengine.core._imperative_rt.common.CompNode', class {});
        this.registerType('megengine.core._imperative_rt.ops.ElemwiseMultiType', class {});
        this.registerType('megengine.core._imperative_rt.ops.FakeQuant', class {});
        this.registerType('megengine.core._imperative_rt.ops.GetVarShape', class {});
        this.registerType('megengine.core._imperative_rt.ops.Resize', class {});
        this.registerType('megengine.core.ops._internal.param_defs.ConvolutionV0.Mode', class {});
        this.registerType('megengine.core.ops._internal.param_defs.Convolution.ComputeMode', class {});
        this.registerType('megengine.distributed.group.Group', class {});
        this.registerType('megengine.module.activation.ReLU', class {});
        this.registerType('megengine.module.activation.Softmax', class {});
        this.registerType('megengine.module.adaptive_pooling.AdaptiveAvgPool2d', class {});
        this.registerType('megengine.module.batchnorm.BatchNorm1d', class {});
        this.registerType('megengine.module.batchnorm.BatchNorm2d', class {});
        this.registerType('megengine.module.conv.Conv2d', class {});
        this.registerType('megengine.module.conv.ConvTranspose2d', class {});
        this.registerType('megengine.module.conv_bn.ConvBn2d', class {});
        this.registerType('megengine.module.dropout.Dropout', class {});
        this.registerType('megengine.module.identity.Identity', class {});
        this.registerType('megengine.module.linear.Linear', class {});
        this.registerType('megengine.module.module.Module', class {});
        this.registerType('megengine.module.normalization.InstanceNorm', class {});
        this.registerType('megengine.module.normalization.GroupNorm', class {});
        this.registerType('megengine.module.normalization.LayerNorm', class {});
        this.registerType('megengine.module.pooling.AvgPool2d', class {});
        this.registerType('megengine.module.pooling.MaxPool2d', class {});
        this.registerType('megengine.module.qat.concat.Concat', class {});
        this.registerType('megengine.module.qat.elemwise.Elemwise', class {});
        this.registerType('megengine.module.sequential.Sequential', class {});
        this.registerType('megengine.quantization.fake_quant.FakeQuantize', class {});
        this.registerType('megengine.quantization.fake_quant.TQT', class {});
        this.registerType('megengine.quantization.utils.QParams', class {});
        this.registerType('megengine.quantization.utils.QuantMode', class {});
        this.registerType('megengine.quantization.observer.ExponentialMovingAverageObserver', class {});
        this.registerType('megengine.quantization.observer.HistogramObserver', class {});
        this.registerType('megengine.quantization.observer.MinMaxObserver', class {});
        this.registerType('megengine.quantization.observer.PassiveObserver', class {});
        this.registerType('megengine.quantization.observer.SyncExponentialMovingAverageObserver', class {});
        this.registerType('megengine.quantization.observer.SyncMinMaxObserver', class {});
        this.registerType('megengine.traced_module.expr.Apply', class {});
        this.registerType('megengine.traced_module.expr.CallFunction', class {});
        this.registerType('megengine.traced_module.expr.CallMethod', class {});
        this.registerType('megengine.traced_module.expr.Constant', class {});
        this.registerType('megengine.traced_module.expr.GetAttr', class {});
        this.registerType('megengine.traced_module.expr.Input', class {});
        this.registerType('megengine.traced_module.fake_quant.FakeQuantize', class {});
        this.registerType('megengine.traced_module.node.ModuleNode', class {});
        this.registerType('megengine.traced_module.node.NodeMixin', class {});
        this.registerType('megengine.traced_module.node.TensorNode', class {});
        this.registerType('megengine.traced_module.pytree.ArgsIndex', class {});
        this.registerType('megengine.traced_module.serialization._ModuleState', class {});
        this.registerType('megengine.traced_module.traced_module.InternalGraph', class {});
        this.registerType('megengine.traced_module.traced_module.NameSpace', class {});
        this.registerType('megengine.traced_module.traced_module.TracedModule', class {});
        this.registerType('megengine.tensor.Parameter', class {
            constructor(data, dtype, device) {
                this.data = data;
                this.dtype = dtype;
                this.device = device;
            }
        });
        this.registerType('megengine.traced_module.pytree.TreeDef', class {
            toString() {
                let content = '';
                for (const child of this.children_defs) {
                    content += `${child},`;
                }
                if (typeof this.type === "string") {
                    return `${this.type.split(".").slice(-1)}(${content})`;
                }
                return `${this.type.__name__}(${content})`;
            }
        });
        this.registerType('megengine.traced_module.pytree.LeafDef', class {
            toString() {
                let content = '';
                if (this.const_val === null) {
                    content += '[';
                } else {
                    content += this.const_val;
                }
                for (const t of Object.values(this.type)) {
                    content += t.__name__;
                }
                content += ']';
                return content;
            }
        });
        this.registerType('megengine.tensor.Tensor', class {
            constructor(data, dtype, device) {
                this.data = data;
                this.dtype = dtype;
                this.device = device;
            }
        });
        this.registerType('megengine.core.tensor.dtype.QuantDtypeMeta', class {
            constructor(name, cname, np_dtype, qmin, qmax, is_signed) {
                this.name = name;
                this.cname = cname;
                this.np_dtype = np_dtype;
                this.qmin = qmin;
                this.qmax = qmax;
                this.is_signed = is_signed;
            }
        });
        this.registerType('nolearn.lasagne.base.BatchIterator', class {});
        this.registerType('nolearn.lasagne.base.Layers', class {});
        this.registerType('nolearn.lasagne.base.NeuralNet', class {});
        this.registerType('nolearn.lasagne.base.TrainSplit', class {});
        this.registerType('nolearn.lasagne.handlers.PrintLayerInfo', class {});
        this.registerType('nolearn.lasagne.handlers.PrintLog', class {});
        this.registerType('numpy.ndarray', class {
            constructor(shape, dtype, buffer, offset, strides, order) {
                this.shape = shape;
                this.dtype = dtype;
                this.data = buffer === undefined ? null : buffer;
                this.offset = offset === undefined ? 0 : offset;
                this._strides = strides === undefined ? null : strides;
                this.order = order === undefined ? null : order;
                this.flags = {};
                this._read();
            }
            static __new__(cls, shape, dtype, buffer, offset, strides, order) {
                return new cls(shape, dtype, buffer, offset, strides, order);
            }
            __setstate__(state) {
                [this.version, this.shape, this.dtype, this.flags.fn, this.data] = state;
                this._read();
            }
            flatten() {
                const size = this.shape.reduce((a, b) => a * b, 1);
                const value = execution.invoke('numpy.ndarray', [
                    [size], this.dtype, this.data, this.offset, this.strides, this.order
                ]);
                value.flags = this.flags;
                return value;
            }
            tobytes() {
                return this.data;
            }
            tolist() {
                if (this.shape.length < 0 || this.shape.length > 1) {
                    throw new python.Error(`Unsupported shape '${JSON.stringify(this.shape)}'.`);
                }
                const size = this.shape.reduce((a, b) => a * b, 1);
                const list = new Array(size);
                switch (this.dtype.kind) {
                    case 'U': {
                        const data = new Uint32Array(new Uint8Array(this.data).buffer);
                        const itemsize = this.dtype.itemsize >> 2;
                        let offset = 0;
                        for (let i = 0; i < size; i++) {
                            const buffer = data.subarray(offset, offset + itemsize);
                            const index = buffer.indexOf(0);
                            list[i] = Array.from(index >= 0 ? buffer.subarray(0, index) : buffer).map((c) => String.fromCodePoint(c)).join('');
                            offset += itemsize;
                        }
                        return list;
                    }
                    case 'S': {
                        const data = this.data;
                        const itemsize = this.dtype.itemsize;
                        const decoder = new TextDecoder('utf-8');
                        let offset = 0;
                        for (let i = 0; i < size; i++) {
                            const buffer = data.subarray(offset, offset + itemsize);
                            const index = buffer.indexOf(0);
                            list[i] = decoder.decode(index >= 0 ? buffer.subarray(0, index) : buffer);
                            offset += itemsize;
                        }
                        return list;
                    }
                    case 'O': {
                        return this.data;
                    }
                    default: {
                        throw new python.Error(`Type kind '${this.dtype.kind}' not implemented.`);
                    }
                }
            }
            get itemsize() {
                return this.dtype.itemsize;
            }
            get size() {
                return (this.shape || []).reduce((a, b) => a * b, 1);
            }
            get strides() {
                if (!this._strides) {
                    const shape = this.shape;
                    const strides = new Array(shape.length);
                    let stride = this.itemsize;
                    for (let i = shape.length - 1; i >= 0; i--) {
                        strides[i] = stride;
                        stride *= shape[i];
                    }
                    return strides;
                }
                return this._strides;
            }
            _read() {
                if (this.data) {
                    const length = this.dtype.itemsize * this.size;
                    if (typeof this.data === 'string') {
                        this.data = this._unescape(this.data, length);
                        if (this.data.length !== length) {
                            throw new python.Error('Invalid string array data size.');
                        }
                    } else if (this.data.length !== length) {
                        // throw new python.Error('Invalid array data size.');
                    }
                }
            }
            _unescape(token, size) {
                const length = token.length;
                const a = new Uint8Array(length);
                if (size && size === length) {
                    for (let p = 0; p < size; p++) {
                        a[p] = token.charCodeAt(p);
                    }
                    return a;
                }
                let i = 0;
                let o = 0;
                while (i < length) {
                    let c = token.charCodeAt(i++);
                    if (c !== 0x5C || i >= length) {
                        a[o++] = c;
                    } else {
                        c = token.charCodeAt(i++);
                        switch (c) {
                            case 0x27: a[o++] = 0x27; break; // '
                            case 0x5C: a[o++] = 0x5C; break; // \\
                            case 0x22: a[o++] = 0x22; break; // "
                            case 0x72: a[o++] = 0x0D; break; // \r
                            case 0x6E: a[o++] = 0x0A; break; // \n
                            case 0x74: a[o++] = 0x09; break; // \t
                            case 0x62: a[o++] = 0x08; break; // \b
                            case 0x58: // x
                            case 0x78: { // X
                                const xsi = i - 1;
                                const xso = o;
                                for (let xi = 0; xi < 2; xi++) {
                                    if (i >= length) {
                                        i = xsi;
                                        o = xso;
                                        a[o] = 0x5c;
                                        break;
                                    }
                                    let c = token.charCodeAt(i++);
                                    if (c >= 65 && c <= 70) {
                                        c -= 55;
                                    } else if (c >= 97 && c <= 102) {
                                        c -= 87;
                                    } else if (c >= 48 && c <= 57) {
                                        c -= 48;
                                    } else {
                                        c = -1;
                                    }
                                    if (c === -1) {
                                        i = xsi;
                                        o = xso;
                                        a[o] = 0x5c;
                                        break;
                                    }
                                    a[o] = a[o] << 4 | c;
                                }
                                o++;
                                break;
                            }
                            default:
                                if (c < 48 || c > 57) { // 0-9
                                    a[o++] = 0x5c;
                                    a[o++] = c;
                                } else {
                                    i--;
                                    const osi = i;
                                    const oso = o;
                                    for (let oi = 0; oi < 3; oi++) {
                                        if (i >= length) {
                                            i = osi;
                                            o = oso;
                                            a[o] = 0x5c;
                                            break;
                                        }
                                        const od = token.charCodeAt(i++);
                                        if (od < 48 || od > 57) {
                                            i = osi;
                                            o = oso;
                                            a[o] = 0x5c;
                                            break;
                                        }
                                        a[o] = a[o] << 3 | od - 48;
                                    }
                                    o++;
                                }
                                break;
                        }
                    }
                }
                return a.slice(0, o);
            }
        });
        this.registerType('numpy.matrix', class extends numpy.ndarray {
            static __new__(/* subtype, data, dtype, copy */) {
                throw new Error("'numpy.matrix.__new__' not implemented.");
            }
        });
        numpy.matrixlib.defmatrix.matrix = numpy.matrix;
        this.registerType('numpy.ma.core.MaskedArray', class extends numpy.ndarray {
            constructor(data /*, mask, dtype, copy, subok, ndmin, fill_value, keep_mask, hard_mask, shrink, order */) {
                super(data.shape, data.dtype, data.data);
            }
        });
        this.registerType('numpy.core.memmap.memmap', class extends numpy.ndarray {
        });
        this.registerType('pandas.core.frame.DataFrame', class {});
        this.registerFunction('pandas.core.indexes.base._new_Index', (cls, d) => {
            return new cls(d);
        });
        this.registerType('pandas.core.arrays.datetimes.DatetimeArray', class {});
        this.registerType('pandas.core.indexes.datetimes._new_DatetimeIndex', class {});
        this.registerType('pandas.core.indexes.datetimes.DatetimeIndex', class {});
        this.registerType('pandas.core.indexes.base.Index', class {});
        this.registerType('pandas.core.indexes.range.RangeIndex', class {});
        this.registerType('pandas.core.indexes.multi.MultiIndex', class {});
        this.registerType('pandas.core.indexes.numeric.Int64Index', class {});
        this.registerType('pandas.core.index.Int64Index', class {});
        pandas.core.index.Index = pandas.core.indexes.base.Index;
        pandas.core.index._new_Index = pandas.core.indexes.base._new_Index;
        this.registerFunction('pandas.core.internals.blocks.Block', class {
        });
        this.registerFunction('pandas.core.internals.blocks.NumpyBlock', class extends pandas.core.internals.blocks.Block {
        });
        this.registerFunction('pandas.core.internals.blocks.get_block_type', (/* dtype */) => {
            return pandas.core.internals.blocks.NumpyBlock;
        });
        this.registerFunction('pandas.core.internals.blocks.maybe_coerce_values', (values) => {
            return values;
        });
        this.registerFunction('pandas.core.internals.blocks.new_block', (values, placement, ndim, refs) => {
            const klass = execution.invoke('pandas.core.internals.blocks.get_block_type', [values.dtype]);
            return new klass(values, ndim, placement, refs);
        });
        this.registerType('pandas.core.internals.managers.SingleBlockManager', class {});
        this.registerType('pandas.core.internals.managers.BlockManager', class {});
        pandas.core.internals.BlockManager = pandas.core.internals.managers.BlockManager;
        this.registerType('pandas.core.series.Series', class {});
        this.registerFunction('pandas._libs.arrays.__pyx_unpickle_NDArrayBacked');
        this.registerFunction('pandas._libs.internals._unpickle_block', (values, placement, ndim) => {
            values = execution.invoke('pandas.core.internals.blocks.maybe_coerce_values', [values]);
            // if not isinstance(placement, BlockPlacement):
            //     placement = BlockPlacement(placement)
            return execution.invoke('pandas.core.internals.blocks.new_block', [values, placement, ndim]);
        });
        this.registerType('pandas._libs.tslibs.base.ABCTimestamp', class extends datetime.datetime {});
        this.registerType('pandas._libs.tslibs.offsets.BaseOffset', class {});
        this.registerType('pandas._libs.tslibs.offsets.SingleConstructorOffset', class extends pandas._libs.tslibs.offsets.BaseOffset {});
        this.registerType('pandas._libs.tslibs.offsets.Tick', class extends pandas._libs.tslibs.offsets.SingleConstructorOffset {});
        this.registerType('pandas._libs.tslibs.offsets.Day', class extends pandas._libs.tslibs.offsets.Tick {});
        this.registerType('pandas._libs.tslibs.offsets.Minute', class extends datetime.datetime {});
        this.registerFunction('pandas._libs.tslibs.timestamps._unpickle_timestamp');
        this.registerType('pandas._libs.tslibs.timestamps._Timestamp', class extends pandas._libs.tslibs.base.ABCTimestamp {});
        this.registerType('pandas._libs.tslibs.timestamps.Timestamp', class extends pandas._libs.tslibs.timestamps._Timestamp {});
        pandas._libs.tslib.Timestamp = pandas._libs.tslibs.timestamps.Timestamp;
        this.registerType('pathlib.Path', class {});
        this.registerType('pathlib.PosixPath', class {});
        this.registerType('pathlib.WindowsPath', class {});
        this.registerType('sklearn._loss.link.BaseLink', class {});
        this.registerType('sklearn._loss.link.IdentityLink', class extends sklearn._loss.link.BaseLink {});
        this.registerType('sklearn._loss.link.Interval', class {});
        this.registerType('sklearn._loss.link.MultinomialLogit', class extends sklearn._loss.link.BaseLink {});
        this.registerType('sklearn._loss._loss.CyLossFunction', class {});
        this.registerType('sklearn._loss._loss.CyHalfSquaredError', class extends sklearn._loss._loss.CyLossFunction {});
        this.registerFunction('sklearn._loss._loss.__pyx_unpickle_CyHalfSquaredError');
        this.registerType('sklearn._loss.loss.BaseLoss', class {});
        this.registerType('sklearn._loss.loss.HalfMultinomialLoss', class extends sklearn._loss.loss.BaseLoss {});
        this.registerType('sklearn.base.BaseEstimator', class {});
        this.registerType('sklearn.base.TransformerMixin', class {});
        this.registerType('sklearn.calibration._CalibratedClassifier', class {});
        this.registerType('sklearn.calibration._SigmoidCalibration', class {});
        this.registerType('sklearn.calibration.CalibratedClassifierCV', class {});
        this.registerType('sklearn.cluster._agglomerative.FeatureAgglomeration', class {});
        this.registerType('sklearn.cluster._dbscan.DBSCAN', class {});
        this.registerType('sklearn.cluster._kmeans.KMeans', class {});
        this.registerType('sklearn.cluster.k_means_.MiniBatchKMeans', class {});
        this.registerType('sklearn.compose._column_transformer.ColumnTransformer', class {});
        this.registerType('sklearn.compose._target.TransformedTargetRegressor', class {});
        this.registerType('sklearn.cross_decomposition._pls.PLSRegression', class {});
        this.registerType('sklearn.decomposition._fastica.FastICA', class {});
        this.registerType('sklearn.decomposition._pca.PCA', class {});
        this.registerType('sklearn.decomposition._truncated_svd.TruncatedSVD', class {});
        this.registerType('sklearn.decomposition.pca.PCA', class {});
        this.registerType('sklearn.decomposition.PCA', class {});
        this.registerType('sklearn.decomposition.truncated_svd.TruncatedSVD', class {});
        this.registerType('sklearn.discriminant_analysis.LinearDiscriminantAnalysis', class {});
        this.registerType('sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis', class {});
        this.registerType('sklearn.dummy.DummyClassifier', class {});
        this.registerType('sklearn.dummy.DummyRegressor', class {});
        this.registerType('sklearn.ensemble._bagging.BaggingClassifier', class {});
        this.registerType('sklearn.ensemble._bagging.BaggingRegressor', class {});
        this.registerType('sklearn.ensemble._forest.RandomForestClassifier', class {});
        this.registerType('sklearn.ensemble._forest.RandomForestRegressor', class {});
        this.registerType('sklearn.ensemble._forest.ExtraTreesClassifier', class {});
        this.registerType('sklearn.ensemble._forest.ExtraTreesRegressor', class {});
        this.registerType('sklearn.ensemble._gb_losses.BinomialDeviance', class {});
        this.registerType('sklearn.ensemble._gb_losses.ExponentialLoss', class {});
        this.registerType('sklearn.ensemble._gb_losses.LeastSquaresError', class {});
        this.registerType('sklearn.ensemble._gb_losses.MultinomialDeviance', class {});
        this.registerType('sklearn.ensemble._gb.GradientBoostingClassifier', class {});
        this.registerType('sklearn.ensemble._gb.GradientBoostingRegressor', class {});
        this.registerType('sklearn.ensemble._hist_gradient_boosting.binning._BinMapper', class {});
        this.registerType('sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor', class {});
        this.registerType('sklearn.ensemble._hist_gradient_boosting.loss.LeastSquares', class {});
        this.registerType('sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor', class {});
        this.registerType('sklearn.ensemble._iforest.IsolationForest', class {});
        this.registerType('sklearn.ensemble._stacking.StackingClassifier', class {});
        this.registerType('sklearn.ensemble._voting.VotingClassifier', class {});
        this.registerType('sklearn.ensemble._voting.VotingRegressor', class {});
        this.registerType('sklearn.ensemble._weight_boosting.AdaBoostClassifier', class {});
        this.registerType('sklearn.ensemble._weight_boosting.AdaBoostRegressor', class {});
        this.registerType('sklearn.ensemble.forest.RandomForestClassifier', class {});
        this.registerType('sklearn.ensemble.forest.RandomForestRegressor', class {});
        this.registerType('sklearn.ensemble.forest.ExtraTreesClassifier', class {});
        this.registerType('sklearn.ensemble.gradient_boosting.BinomialDeviance', class {});
        this.registerType('sklearn.ensemble.gradient_boosting.GradientBoostingClassifier', class {});
        this.registerType('sklearn.ensemble.gradient_boosting.LogOddsEstimator', class {});
        this.registerType('sklearn.ensemble.gradient_boosting.MultinomialDeviance', class {});
        this.registerType('sklearn.ensemble.gradient_boosting.PriorProbabilityEstimator', class {});
        this.registerType('sklearn.ensemble.weight_boosting.AdaBoostClassifier', class {});
        this.registerType('sklearn.feature_extraction._hashing.FeatureHasher', class {});
        this.registerType('sklearn.feature_extraction.text.CountVectorizer', class {});
        this.registerType('sklearn.feature_extraction.text.HashingVectorizer', class {});
        this.registerType('sklearn.feature_extraction.text.TfidfTransformer', class {});
        this.registerType('sklearn.feature_extraction.text.TfidfVectorizer', class {});
        this.registerType('sklearn.feature_selection._from_model.SelectFromModel', class {});
        this.registerFunction('sklearn.feature_selection._univariate_selection.chi2');
        this.registerType('sklearn.feature_selection._univariate_selection.GenericUnivariateSelect', class {});
        this.registerType('sklearn.feature_selection._univariate_selection.SelectKBest', class {});
        this.registerType('sklearn.feature_selection._univariate_selection.SelectPercentile', class {});
        this.registerType('sklearn.feature_selection._variance_threshold.VarianceThreshold', class {});
        this.registerType('sklearn.feature_selection.univariate_selection.SelectKBest', class {});
        this.registerType('sklearn.feature_selection.variance_threshold.VarianceThreshold', class {});
        this.registerType('sklearn.gaussian_process._gpc.GaussianProcessClassifier', class {});
        this.registerType('sklearn.gaussian_process._gpr.GaussianProcessRegressor', class {});
        this.registerType('sklearn.gaussian_process.gpc.GaussianProcessClassifier', class {});
        this.registerType('sklearn.gaussian_process.kernels.ConstantKernel', class {});
        this.registerType('sklearn.gaussian_process.kernels.DotProduct', class {});
        this.registerType('sklearn.gaussian_process.kernels.Product', class {});
        this.registerType('sklearn.gaussian_process.kernels.RBF', class {});
        this.registerType('sklearn.gaussian_process.kernels.Sum', class {});
        this.registerType('sklearn.gaussian_process.kernels.WhiteKernel', class {});
        this.registerType('sklearn.grid_search._CVScoreTuple', class {});
        this.registerType('sklearn.grid_search.GridSearchCV', class {});
        this.registerType('sklearn.impute._base.SimpleImputer', class {});
        this.registerType('sklearn.impute._iterative.IterativeImputer', class {});
        this.registerType('sklearn.impute.SimpleImputer', class {});
        this.registerType('sklearn.isotonic.IsotonicRegression', class {});
        this.registerType('sklearn.linear_model._base.LinearRegression', class {});
        this.registerType('sklearn.linear_model._bayes.BayesianRidge', class {});
        this.registerType('sklearn.linear_model._coordinate_descent.ElasticNetCV', class {});
        this.registerType('sklearn.linear_model._coordinate_descent.ElasticNet', class {});
        this.registerType('sklearn.linear_model._coordinate_descent.Lasso', class {});
        this.registerType('sklearn.linear_model._least_angle.LassoLarsCV', class {});
        this.registerType('sklearn.linear_model._logistic.LogisticRegression', class {});
        this.registerType('sklearn.linear_model._logistic.LogisticRegressionCV', class {});
        this.registerType('sklearn.linear_model._quantile.QuantileRegressor', class {});
        this.registerType('sklearn.linear_model._ridge.Ridge', class {});
        this.registerType('sklearn.linear_model._ridge.RidgeClassifier', class {});
        this.registerType('sklearn.linear_model._sgd_fast.Hinge', class {});
        this.registerType('sklearn.linear_model._sgd_fast.Log', class {});
        this.registerType('sklearn.linear_model._sgd_fast.ModifiedHuber', class {});
        this.registerType('sklearn.linear_model._sgd_fast.SquaredHinge', class {});
        this.registerType('sklearn.linear_model._stochastic_gradient.SGDClassifier', class {});
        this.registerType('sklearn.linear_model.base.LinearRegression', class {});
        this.registerType('sklearn.linear_model.sgd_fast.Hinge', class {});
        this.registerType('sklearn.linear_model.LogisticRegression', class {});
        this.registerType('sklearn.linear_model.logistic.LogisticRegression', class {});
        this.registerType('sklearn.linear_model.logistic.LogisticRegressionCV', class {});
        this.registerType('sklearn.linear_model.LassoLars​', class {});
        this.registerType('sklearn.linear_model.ridge.Ridge', class {});
        this.registerType('sklearn.linear_model.sgd_fast.Log', class {});
        this.registerType('sklearn.linear_model.stochastic_gradient.SGDClassifier', class {});
        this.registerType('sklearn.manifold._t_sne.TSNE', class {});
        this.registerType('sklearn.metrics._dist_metrics.EuclideanDistance', class {});
        this.registerType('sklearn.metrics._dist_metrics.EuclideanDistance64', class {});
        this.registerType('sklearn.metrics._dist_metrics.ManhattanDistance', class {});
        this.registerType('sklearn.metrics._scorer._PassthroughScorer', class {});
        this.registerType('sklearn.metrics._scorer._PredictScorer', class {});
        this.registerType('sklearn.metrics.scorer._PredictScorer', class {});
        this.registerType('sklearn.metrics._scorer._ThresholdScorer', class {});
        this.registerType('sklearn.mixture._bayesian_mixture.BayesianGaussianMixture', class {});
        this.registerType('sklearn.mixture._gaussian_mixture.GaussianMixture', class {});
        this.registerType('sklearn.model_selection._search.GridSearchCV', class {});
        this.registerType('sklearn.model_selection._search.RandomizedSearchCV', class {});
        this.registerType('sklearn.model_selection._split.KFold', class {});
        this.registerType('sklearn.model_selection._split.StratifiedKFold', class {});
        this.registerType('sklearn.model_selection._split.StratifiedShuffleSplit', class {});
        this.registerType('sklearn.multiclass.OneVsRestClassifier', class {});
        this.registerType('sklearn.multioutput.MultiOutputClassifier', class {});
        this.registerType('sklearn.multioutput.MultiOutputRegressor', class {});
        this.registerType('sklearn.naive_bayes.BernoulliNB', class {});
        this.registerType('sklearn.naive_bayes.ComplementNB', class {});
        this.registerType('sklearn.naive_bayes.GaussianNB', class {});
        this.registerType('sklearn.naive_bayes.MultinomialNB', class {});
        this.registerType('sklearn.neighbors.ball_tree.BallTree', class {});
        this.registerType('sklearn.neighbors._classification.KNeighborsClassifier', class {});
        this.registerType('sklearn.neighbors._dist_metrics.newObj', class {});
        this.registerType('sklearn.neighbors._dist_metrics.EuclideanDistance', class {});
        this.registerType('sklearn.neighbors._kd_tree.KDTree', class {});
        this.registerType('sklearn.neighbors._kd_tree.newObj', class {});
        this.registerType('sklearn.neighbors._regression.KNeighborsRegressor', class {});
        this.registerType('sklearn.neighbors._unsupervised.NearestNeighbors', class {});
        this.registerType('sklearn.neighbors.classification.KNeighborsClassifier', class {});
        this.registerType('sklearn.neighbors.dist_metrics.newObj', class {});
        this.registerType('sklearn.neighbors.dist_metrics.EuclideanDistance', class {});
        this.registerType('sklearn.neighbors.kd_tree.newObj', class {});
        this.registerType('sklearn.neighbors.kd_tree.KDTree', class {});
        this.registerType('sklearn.neighbors.KNeighborsClassifier', class {});
        this.registerType('sklearn.neighbors.KNeighborsRegressor', class {});
        this.registerType('sklearn.neighbors.regression.KNeighborsRegressor', class {});
        this.registerType('sklearn.neighbors.unsupervised.NearestNeighbors', class {});
        this.registerType('sklearn.neural_network._multilayer_perceptron.MLPClassifier', class {});
        this.registerType('sklearn.neural_network._multilayer_perceptron.MLPRegressor', class {});
        this.registerType('sklearn.neural_network._stochastic_optimizers.AdamOptimizer', class {});
        this.registerType('sklearn.neural_network._stochastic_optimizers.SGDOptimizer', class {});
        this.registerType('sklearn.neural_network.rbm.BernoulliRBM', class {});
        this.registerType('sklearn.neural_network.multilayer_perceptron.MLPClassifier', class {});
        this.registerType('sklearn.neural_network.multilayer_perceptron.MLPRegressor', class {});
        this.registerType('sklearn.neural_network.stochastic_gradient.SGDClassifier', class {});
        this.registerType('sklearn.pipeline.Pipeline', class {});
        this.registerType('sklearn.pipeline.FeatureUnion', class {});
        this.registerType('sklearn.preprocessing._data.MinMaxScaler', class {});
        this.registerType('sklearn.preprocessing._data.MaxAbsScaler', class {});
        this.registerType('sklearn.preprocessing._data.Normalizer', class {});
        this.registerType('sklearn.preprocessing._data.PolynomialFeatures', class {});
        this.registerType('sklearn.preprocessing._data.PowerTransformer', class {});
        this.registerType('sklearn.preprocessing._data.QuantileTransformer', class {});
        this.registerType('sklearn.preprocessing._data.RobustScaler', class {});
        this.registerType('sklearn.preprocessing._data.StandardScaler', class {});
        this.registerType('sklearn.preprocessing._discretization.KBinsDiscretizer', class {});
        this.registerType('sklearn.preprocessing._encoders.OneHotEncoder', class {});
        this.registerType('sklearn.preprocessing._encoders.OrdinalEncoder', class {});
        this.registerType('sklearn.preprocessing._function_transformer.FunctionTransformer', class {});
        this.registerType('sklearn.preprocessing._label.LabelBinarizer', class {});
        this.registerType('sklearn.preprocessing._label.LabelEncoder', class {});
        this.registerType('sklearn.preprocessing._label.MultiLabelBinarizer', class {});
        this.registerType('sklearn.preprocessing._polynomial.PolynomialFeatures', class {});
        this.registerType('sklearn.preprocessing.data.Binarizer', class {});
        this.registerType('sklearn.preprocessing.data.MaxAbsScaler', class {});
        this.registerType('sklearn.preprocessing.data.MinMaxScaler', class {});
        this.registerType('sklearn.preprocessing.data.Normalizer', class {});
        this.registerType('sklearn.preprocessing.data.OneHotEncoder', class {});
        this.registerType('sklearn.preprocessing.data.PolynomialFeatures', class {});
        this.registerType('sklearn.preprocessing.data.PowerTransformer', class {});
        this.registerType('sklearn.preprocessing.data.RobustScaler', class {});
        this.registerType('sklearn.preprocessing.data.QuantileTransformer', class {});
        this.registerType('sklearn.preprocessing.data.StandardScaler', class {});
        this.registerType('sklearn.preprocessing.imputation.Imputer', class {});
        this.registerType('sklearn.preprocessing.label.LabelBinarizer', class {});
        this.registerType('sklearn.preprocessing.label.LabelEncoder', class {});
        this.registerType('sklearn.preprocessing.label.MultiLabelBinarizer', class {});
        this.registerType('sklearn.svm._classes.LinearSVC', class {});
        this.registerType('sklearn.svm._classes.NuSVC', class {});
        this.registerType('sklearn.svm._classes.OneClassSVM', class {});
        this.registerType('sklearn.svm._classes.SVC', class {});
        this.registerType('sklearn.svm._classes.SVR', class {});
        this.registerType('sklearn.svm.classes.LinearSVC', class {});
        this.registerType('sklearn.svm.classes.OneClassSVM', class {});
        this.registerType('sklearn.svm.classes.SVC', class {});
        this.registerType('sklearn.svm.classes.SVR', class {});
        this.registerType('sklearn.tree._classes.DecisionTreeClassifier', class {});
        this.registerType('sklearn.tree._classes.DecisionTreeRegressor', class {});
        this.registerType('sklearn.tree._classes.ExtraTreeClassifier', class {});
        this.registerType('sklearn.tree._classes.ExtraTreeRegressor', class {});
        this.registerType('sklearn.tree._tree.Tree', class {
            constructor(n_features, n_classes, n_outputs) {
                this.n_features = n_features;
                this.n_classes = n_classes;
                this.n_outputs = n_outputs;
            }
            __setstate__(state) {
                this.max_depth = state.max_depth;
                this.node_count = state.node_count;
                this.nodes = state.nodes;
                this.values = state.values;
            }
        });
        this.registerType('sklearn.tree.tree.DecisionTreeClassifier', class {});
        this.registerType('sklearn.tree.tree.DecisionTreeRegressor', class {});
        this.registerType('sklearn.tree.tree.ExtraTreeClassifier', class {});
        this.registerType('sklearn.utils._bunch.Bunch', class {});
        this.registerType('sklearn.utils.deprecation.DeprecationDict', class {});
        this.registerType('pickle.Unpickler', class {
            constructor(data) {
                this._reader = data instanceof Uint8Array ? new python.BinaryReader(data) : new python.StreamReader(data);
                this.persistent_load = () => {
                    throw new python.Error('Unsupported persistent id.');
                };
            }
            load() {
                const reader = this._reader;
                const marker = [];
                let stack = [];
                const memo = {};
                let size = 0;
                while (reader.position < reader.length) {
                    const opcode = reader.byte();
                    // console.log((reader.position - 1).toString() + ' ' + Object.entries(OpCode).find(([, value]) => value === opcode)[0]);
                    // https://svn.python.org/projects/python/trunk/Lib/pickletools.py
                    // https://github.com/python/cpython/blob/master/Lib/pickle.py
                    switch (opcode) {
                        case 128: { // PROTO
                            const version = reader.byte();
                            if (version > 5) {
                                throw new python.Error(`Unsupported protocol version '${version}'.`);
                            }
                            break;
                        }
                        case 99: { // GLOBAL 'c'
                            const module = reader.line();
                            const name = reader.line();
                            stack.push(this.find_class(module, name));
                            break;
                        }
                        case 147: { // STACK_GLOBAL '\x93' (Protocol 4)
                            const name = stack.pop();
                            const module = stack.pop();
                            stack.push(this.find_class(module, name));
                            break;
                        }
                        case 111: { // OBJ 'o'
                            const args = stack;
                            const cls = args.pop();
                            stack = marker.pop();
                            const obj = this._instantiate(cls, args);
                            stack.push(obj);
                            break;
                        }
                        case 112 : { // PUT 'p'
                            const index = parseInt(reader.line(), 10);
                            memo[index] = stack[stack.length - 1];
                            size++;
                            break;
                        }
                        case 103: { // GET 'g'
                            const index = parseInt(reader.line(), 10);
                            stack.push(memo[index]);
                            break;
                        }
                        case 48: // POP '0'
                            stack.pop();
                            break;
                        case 49: // POP_MARK '1'
                            stack = marker.pop();
                            break;
                        case 50: // DUP '2'
                            stack.push(stack[stack.length - 1]);
                            break;
                        case 80: // PERSID 'P'
                            stack.push(this.persistent_load(reader.line()));
                            break;
                        case 81: // BINPERSID 'Q'
                            stack.push(this.persistent_load(stack.pop()));
                            break;
                        case 82: { // REDUCE 'R'
                            const args = stack.pop();
                            const func = stack.pop();
                            stack.push(this._reduce(func, args));
                            break;
                        }
                        case 129: { // NEWOBJ
                            const args = stack.pop();
                            const cls = stack.pop();
                            const obj = this._newobj(cls, args);
                            stack.push(obj);
                            break;
                        }
                        case 146: { // NEWOBJ_EX '\x92' (Protocol 4)
                            const kwargs = stack.pop();
                            const args = stack.pop();
                            const cls = stack.pop();
                            if (Object.entries(kwargs).length > 0) {
                                throw new python.Error("Unpickle 'NEWOBJ_EX' not implemented.");
                            }
                            const obj = this._newobj(cls, args);
                            stack.push(obj);
                            break;
                        }
                        case 104: // BINGET 'h'
                            stack.push(memo[reader.byte()]);
                            break;
                        case 105: { // INST 'i'
                            const module = reader.line();
                            const name = reader.line();
                            const args = stack;
                            const cls = `${module}.${name}`;
                            stack = marker.pop();
                            // cls = this.find_class(module, name)
                            const obj = this._instantiate(cls, args);
                            stack.push(obj);
                            break;
                        }
                        case 106: // LONG_BINGET 'j'
                            stack.push(memo[reader.uint32()]);
                            break;
                        case 113: // BINPUT 'q'
                            memo[reader.byte()] = stack[stack.length - 1];
                            size++;
                            break;
                        case 114: // LONG_BINPUT 'r'
                            memo[reader.uint32()] = stack[stack.length - 1];
                            size++;
                            break;
                        case 74: // BININT 'J'
                            stack.push(reader.int32());
                            break;
                        case 75: // BININT1 'K'
                            stack.push(reader.byte());
                            break;
                        case 76: // LONG 'L'
                            stack.push(parseInt(reader.line(), 10));
                            break;
                        case 77: // BININT2 'M'
                            stack.push(reader.uint16());
                            break;
                        case 66: // BINBYTES 'B' (Protocol 3)
                            stack.push(reader.read(reader.int32()));
                            break;
                        case 67: // SHORT_BINBYTES 'C' (Protocol 3)
                            stack.push(reader.read(reader.byte()));
                            break;
                        case 142: // BINBYTES8 '\x8e' (Protocol 4)
                            stack.push(reader.read(reader.int64().toNumber()));
                            break;
                        case 70: // FLOAT 'F'
                            stack.push(parseFloat(reader.line()));
                            break;
                        case 71: // BINFLOAT 'G'
                            stack.push(reader.float64());
                            break;
                        case 73: { // INT 'I'
                            const value = reader.line();
                            if (value === '01') {
                                stack.push(true);
                            } else if (value === '00') {
                                stack.push(false);
                            } else {
                                stack.push(parseInt(value, 10));
                            }
                            break;
                        }
                        case 93: // EMPTY_LIST ']'
                            stack.push(execution.invoke('builtins.list', []));
                            break;
                        case 41: // EMPTY_TUPLE ')'
                            stack.push([]);
                            break;
                        case 143: // EMPTY_SET '\x8f' (Protocol 4)
                            stack.push([]);
                            break;
                        case 144: { // ADDITEMS '\x90' (Protocol 4)
                            const items = stack;
                            stack = marker.pop();
                            const obj = stack[stack.length - 1];
                            for (let i = 0; i < items.length; i++) {
                                obj.push(items[i]);
                            }
                            break;
                        }
                        case 145: { // FROZENSET '\x91' (Protocol 4)
                            const items = stack;
                            stack = marker.pop();
                            stack.push(items);
                            break;
                        }
                        case 100: { // DICT 'd'
                            const items = stack;
                            stack = marker.pop();
                            const dict = {};
                            for (let i = 0; i < items.length; i += 2) {
                                dict[items[i]] = items[i + 1];
                            }
                            stack.push(dict);
                            break;
                        }
                        case 108: { // LIST 'l'
                            const items = stack;
                            stack = marker.pop();
                            stack.push(items);
                            break;
                        }
                        case 116: { // TUPLE 't'
                            const items = stack;
                            stack = marker.pop();
                            stack.push(items);
                            break;
                        }
                        case 133: { // TUPLE1 // '\x85'
                            stack.push([stack.pop()]);
                            break;
                        }
                        case 134: { // TUPLE2 '\x86'
                            const b = stack.pop();
                            const a = stack.pop();
                            stack.push([a, b]);
                            break;
                        }
                        case 135: { // TUPLE3 '\x87'
                            const c = stack.pop();
                            const b = stack.pop();
                            const a = stack.pop();
                            stack.push([a, b, c]);
                            break;
                        }
                        case 115: { // SETITEM 's'
                            const value = stack.pop();
                            const key = stack.pop();
                            const obj = stack[stack.length - 1];
                            if (obj.__setitem__) {
                                obj.__setitem__(key, value);
                            } else {
                                obj[key] = value;
                            }
                            break;
                        }
                        case 117: { // SETITEMS 'u'
                            const items = stack;
                            stack = marker.pop();
                            const obj = stack[stack.length - 1];
                            for (let i = 0; i < items.length; i += 2) {
                                if (obj.__setitem__) {
                                    obj.__setitem__(items[i], items[i + 1]);
                                } else {
                                    obj[items[i]] = items[i + 1];
                                }
                            }
                            break;
                        }
                        case 125: // EMPTY_DICT '}'
                            stack.push({});
                            break;
                        case 97: { // APPEND 'a'
                            const append = stack.pop();
                            stack[stack.length - 1].push(append);
                            break;
                        }
                        case 101: { // APPENDS 'e'
                            const appends = stack;
                            stack = marker.pop();
                            const list = stack[stack.length - 1];
                            list.push(...appends);
                            break;
                        }
                        case 83: { // STRING 'S'
                            const str = reader.line();
                            stack.push(str.substr(1, str.length - 2));
                            break;
                        }
                        case 84: // BINSTRING 'T'
                            stack.push(reader.string(reader.uint32()));
                            break;
                        case 85 : // SHORT_BINSTRING 'U'
                            stack.push(reader.string(reader.byte()));
                            break;
                        case 86: // UNICODE 'V'
                            stack.push(reader.line());
                            break;
                        case 88: // BINUNICODE 'X
                            stack.push(reader.string(reader.uint32(), 'utf-8'));
                            break;
                        case 140: // SHORT_BINUNICODE '\x8c' (Protocol 4)
                            stack.push(reader.string(reader.byte(), 'utf-8'));
                            break;
                        case 98: { // BUILD 'b'
                            const state = stack.pop();
                            let obj = stack.pop();
                            if (obj.__setstate__) {
                                if (obj.__setstate__.__call__) {
                                    obj.__setstate__.__call__([obj, state]);
                                } else {
                                    obj.__setstate__(state);
                                }
                            } else if (ArrayBuffer.isView(state) || Object(state) !== state) {
                                obj.__state__ = state;
                            } else if (obj instanceof Map) {
                                for (const key in state) {
                                    obj.set(key, state[key]);
                                }
                            } else {
                                Object.assign(obj, state);
                            }
                            if (obj.__read__) {
                                obj = obj.__read__(this);
                            }
                            stack.push(obj);
                            break;
                        }
                        case 40: // MARK '('
                            marker.push(stack);
                            stack = [];
                            break;
                        case 136: // NEWTRUE '\x88'
                            stack.push(true);
                            break;
                        case 137: // NEWFALSE '\x89'
                            stack.push(false);
                            break;
                        case 138: { // LONG1 '\x8a'
                            const data = reader.read(reader.byte());
                            let number = 0;
                            switch (data.length) {
                                case 0: number = 0; break;
                                /* eslint-disable prefer-destructuring */
                                case 1: number = data[0]; break;
                                /* eslint-enable prefer-destructuring */
                                case 2: number = data[1] << 8 | data[0]; break;
                                case 3: number = data[2] << 16 | data[1] << 8 | data[0]; break;
                                case 4: number = data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]; break;
                                case 5: number = data[4] * 0x100000000 + ((data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]) >>> 0); break;
                                default: number = Array.prototype.slice.call(data, 0); break;
                            }
                            stack.push(number);
                            break;
                        }
                        case 139: // LONG4 '\x8b'
                            // decode LONG4
                            stack.push(reader.read(reader.uint32()));
                            break;
                        case 148: // MEMOIZE '\x94' (Protocol 4)
                            memo[size++] = stack[stack.length - 1];
                            break;
                        case 149: // FRAME '\x95' (Protocol 4)
                            reader.read(8);
                            break;
                        case 150: { // BYTEARRAY8 '\x96' (Protocol 5)
                            stack.push(reader.read(reader.int64().toNumber()));
                            break;
                        }
                        case 78: // NONE 'N'
                            stack.push(null);
                            break;
                        case 46: // STOP '.'
                            return stack.pop();
                        case 141: // BINUNICODE8 '\x8d' (Protocol 4)
                        case 151: // NEXT_BUFFER '\x97' (Protocol 5)
                        case 152: // READONLY_BUFFER '\x98' (Protocol 5)
                        default:
                            throw new python.Error(`Unknown opcode ${opcode} at position ${(reader.position - 1)}.`);
                    }
                }
                throw new python.Error('Unexpected end of file.');
            }
            find_class(module, name) {
                execution.__import__(module);
                return execution.resolve(`${module}.${name}`);
            }
            _instantiate(cls, args) {
                return execution.invoke(cls, args);
            }
            _newobj(cls, args) {
                // cls.__new__(cls, args)
                return execution.invoke(cls, args);
            }
            _reduce(func, args) {
                return execution.invoke(func, args);
            }
            read(size) {
                return this._reader.read(size);
            }
            stream(size) {
                return this._reader.stream(size);
            }
        });
        this.registerType('random.Random', class {});
        this.registerType('re.Pattern', class {
            constructor(pattern, flags) {
                this.pattern = pattern;
                this.flags = flags;
            }
        });
        this.registerType('spacy._ml.PrecomputableAffine', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('spacy.syntax._parser_model.ParserModel', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('theano.compile.function_module._constructor_Function', class {});
        this.registerType('theano.compile.function_module._constructor_FunctionMaker', class {});
        this.registerType('theano.compile.function_module.Function', class {});
        this.registerType('theano.compile.function_module.Supervisor', class {});
        this.registerType('theano.compile.io.In', class {});
        this.registerType('theano.compile.io.SymbolicOutput', class {});
        this.registerType('theano.compile.mode.Mode', class {});
        this.registerType('theano.compile.ops.OutputGuard', class {});
        this.registerType('theano.compile.ops.Shape', class {});
        this.registerType('theano.compile.ops.Shape_i', class {});
        this.registerType('theano.gof.destroyhandler.DestroyHandler', class {});
        this.registerType('theano.gof.fg.FunctionGraph', class {});
        this.registerType('theano.gof.graph.Apply', class {});
        this.registerType('theano.gof.link.Container', class {});
        this.registerType('theano.gof.opt._metadict', class {});
        this.registerType('theano.gof.opt.ChangeTracker', class {});
        this.registerType('theano.gof.opt.MergeFeature', class {});
        this.registerType('theano.gof.optdb.Query', class {});
        this.registerType('theano.gof.toolbox.PreserveVariableAttributes', class {});
        this.registerType('theano.gof.toolbox.ReplaceValidate', class {});
        this.registerType('theano.gof.utils.scratchpad', class {});
        this.registerType('theano.misc.ordered_set.Link', class {});
        this.registerType('theano.misc.ordered_set.OrderedSet', class {});
        this.registerType('theano.sandbox.cuda.basic_ops.HostFromGpu', class {});
        this.registerType('theano.sandbox.cuda.type.CudaNdarray_unpickler', class {});
        this.registerType('theano.sandbox.cuda.type.CudaNdarrayType', class {});
        this.registerType('theano.sandbox.cuda.var.CudaNdarraySharedVariable', class {});
        this.registerType('theano.scalar.basic.Abs', class {});
        this.registerType('theano.scalar.basic.Add', class {});
        this.registerType('theano.scalar.basic.Cast', class {});
        this.registerType('theano.scalar.basic.Composite', class {});
        this.registerType('theano.scalar.basic.EQ', class {});
        this.registerType('theano.scalar.basic.GE', class {});
        this.registerType('theano.scalar.basic.Identity', class {});
        this.registerType('theano.scalar.basic.IntDiv', class {});
        this.registerType('theano.scalar.basic.Inv', class {});
        this.registerType('theano.scalar.basic.LE', class {});
        this.registerType('theano.scalar.basic.LT', class {});
        this.registerType('theano.scalar.basic.Mul', class {});
        this.registerType('theano.scalar.basic.Neg', class {});
        this.registerType('theano.scalar.basic.Scalar', class {});
        this.registerType('theano.scalar.basic.ScalarConstant', class {});
        this.registerType('theano.scalar.basic.ScalarVariable', class {});
        this.registerType('theano.scalar.basic.Second', class {});
        this.registerType('theano.scalar.basic.Sgn', class {});
        this.registerType('theano.scalar.basic.specific_out', class {});
        this.registerType('theano.scalar.basic.Sub', class {});
        this.registerType('theano.scalar.basic.Switch', class {});
        this.registerType('theano.scalar.basic.Tanh', class {});
        this.registerType('theano.scalar.basic.transfer_type', class {});
        this.registerType('theano.scalar.basic.TrueDiv', class {});
        this.registerType('theano.tensor.basic.Alloc', class {});
        this.registerType('theano.tensor.basic.Dot', class {});
        this.registerType('theano.tensor.basic.MaxAndArgmax', class {});
        this.registerType('theano.tensor.basic.Reshape', class {});
        this.registerType('theano.tensor.basic.ScalarFromTensor', class {});
        this.registerType('theano.tensor.blas.Dot22', class {});
        this.registerType('theano.tensor.blas.Dot22Scalar', class {});
        this.registerType('theano.tensor.blas.Gemm', class {});
        this.registerType('theano.tensor.elemwise.DimShuffle', class {});
        this.registerType('theano.tensor.elemwise.Elemwise', class {});
        this.registerType('theano.tensor.elemwise.Sum', class {});
        this.registerType('theano.tensor.nnet.abstract_conv.AbstractConv2d', class {});
        this.registerType('theano.tensor.nnet.abstract_conv.AbstractConv2d_gradInputs', class {});
        this.registerType('theano.tensor.nnet.abstract_conv.AbstractConv2d_gradWeights', class {});
        this.registerType('theano.tensor.nnet.corr.CorrMM', class {});
        this.registerType('theano.tensor.nnet.corr.CorrMM_gradInputs', class {});
        this.registerType('theano.tensor.nnet.corr.CorrMM_gradWeights', class {});
        this.registerType('theano.tensor.nnet.nnet.CrossentropyCategorical1Hot', class {});
        this.registerType('theano.tensor.nnet.nnet.CrossentropyCategorical1HotGrad', class {});
        this.registerType('theano.tensor.nnet.nnet.CrossentropySoftmax1HotWithBiasDx', class {});
        this.registerType('theano.tensor.nnet.nnet.CrossentropySoftmaxArgmax1HotWithBias', class {});
        this.registerType('theano.tensor.nnet.nnet.Softmax', class {});
        this.registerType('theano.tensor.nnet.nnet.SoftmaxGrad', class {});
        this.registerType('theano.tensor.nnet.nnet.SoftmaxWithBias', class {});
        this.registerType('theano.tensor.opt.MakeVector', class {});
        this.registerType('theano.tensor.opt.ShapeFeature', class {});
        this.registerType('theano.tensor.sharedvar.TensorSharedVariable', class {});
        this.registerType('theano.tensor.signal.pool.MaxPoolGrad', class {});
        this.registerType('theano.tensor.signal.pool.Pool', class {});
        this.registerType('theano.tensor.subtensor.Subtensor', class {});
        this.registerType('theano.tensor.type.TensorType', class {});
        this.registerType('theano.tensor.var.TensorConstant', class {});
        this.registerType('theano.tensor.var.TensorConstantSignature', class {});
        this.registerType('theano.tensor.var.TensorVariable', class {});
        this.registerType('thinc.describe.Biases', class {});
        this.registerType('thinc.describe.Dimension', class {});
        this.registerType('thinc.describe.Gradient', class {});
        this.registerType('thinc.describe.Weights', class {});
        this.registerType('thinc.describe.Synapses', class {});
        this.registerType('thinc.neural._classes.affine.Affine', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.convolution.ExtractWindow', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.feature_extracter.FeatureExtracter', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.feed_forward.FeedForward', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.function_layer.FunctionLayer', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.hash_embed.HashEmbed', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.layernorm.LayerNorm', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.maxout.Maxout', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.resnet.Residual', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural._classes.softmax.Softmax', class {
            __setstate__(state) {
                Object.assign(this, new pickle.Unpickler(state).load());
            }
        });
        this.registerType('thinc.neural.mem.Memory', class {
        });
        this.registerType('thinc.neural.ops.NumpyOps', class {
        });
        this.registerType('__main__.BYOLState', class {
            constructor(dict) {
                Object.assign(this, dict);
            }
        });
        this.registerType('types.GenericAlias', class {});
        this.registerType('types.SimpleNamespace', class {});
        this.register('types').ObjectType = builtins.object;
        this.register('types').ModuleType = builtins.module;
        this.register('types').MethodType = builtins.method;
        this.register('types').FunctionType = builtins.function;
        this.register('types').TypeType = builtins.type;
        this.register('types').CodeType = builtins.code;
        this.registerType('xgboost.compat.XGBoostLabelEncoder', class {});
        this.registerType('xgboost.core.Booster', class {});
        this.registerType('xgboost.sklearn.XGBClassifier', class {});
        this.registerType('xgboost.sklearn.XGBRegressor', class {});
        this.registerFunction('_codecs.encode', (obj, encoding) => {
            return execution.invoke('builtins.bytearray', [obj, encoding]);
        });
        this.registerType('builtins.bytearray', class extends Uint8Array {
            constructor(source, encoding /*, errors */) {
                source = builtins.bytes.__encode__(source, encoding);
                super(Number.isInteger(source) ? source : source.length);
                if (Array.isArray(source)) {
                    for (let i = 0; i < source.length; i++) {
                        this[i] = source;
                    }
                } else if (source instanceof Uint8Array) {
                    this.set(source, 0);
                } else if (typeof source === 'string') {
                    for (let i = 0; i < source.length; i++) {
                        this[i] = source.charCodeAt(i);
                    }
                }
            }
            static __encode__(source, encoding) {
                if (source === undefined) {
                    return 0;
                }
                if (Number.isInteger(source)) {
                    return source;
                }
                if (Array.isArray(source) || source instanceof Uint8Array) {
                    return source;
                }
                if (typeof source === 'string') {
                    switch (encoding) {
                        case 'latin1':
                        case 'latin-1':
                            return source;
                        case 'utf8':
                        case 'utf-8':
                            return new TextEncoder('utf-8').encode(source);
                        case undefined:
                            throw new python.Error('Unsupported string argument without an encoding.');
                        default:
                            throw new python.Error(`Unsupported encoding '${encoding}'.`);
                    }
                }
                throw new python.Error('Unsupported source.');
            }
        });
        this.registerType('builtins.bytes', class extends Uint8Array {
            constructor(source, encoding /*, errors */) {
                source = builtins.bytes.__encode__(source, encoding);
                super(Number.isInteger(source) ? source : source.length);
                if (Array.isArray(source)) {
                    for (let i = 0; i < source.length; i++) {
                        this[i] = source;
                    }
                } else if (source instanceof Uint8Array) {
                    this.set(source, 0);
                } else if (typeof source === 'string') {
                    for (let i = 0; i < source.length; i++) {
                        this[i] = source.charCodeAt(i);
                    }
                }
            }
            static __encode__(source, encoding) {
                if (source === undefined) {
                    return 0;
                }
                if (Number.isInteger(source)) {
                    return source;
                }
                if (Array.isArray(source) || source instanceof Uint8Array) {
                    return source;
                }
                if (typeof source === 'string') {
                    switch (encoding) {
                        case 'latin1':
                        case 'latin-1':
                            return source;
                        case 'utf8':
                        case 'utf-8':
                            return new TextEncoder('utf-8').encode(source);
                        case undefined:
                            throw new python.Error('Unsupported string argument without an encoding.');
                        default:
                            throw new python.Error(`Unsupported encoding '${encoding}'.`);
                    }
                }
                throw new python.Error('Unsupported source.');
            }
        });
        this.registerType('builtins.frozenset', class extends Set {
            constructor(iterable) {
                super();
                if (iterable) {
                    for (const item of iterable) {
                        this.add(item);
                    }
                }
            }
        });
        this.registerFunction('builtins.issubclass', (obj, type) => {
            const name = `${type.__module__}.${type.__name__}`;
            if (obj.__module__ && obj.__name__) {
                if (name === `${obj.__module__}.${obj.__name__}`) {
                    return true;
                }
            }
            if (obj.__bases__) {
                for (const base of obj.__bases__) {
                    if (builtins.issubclass(base, type)) {
                        return true;
                    }
                }
            }
            return false;
        });
        this.registerFunction('builtins.isinstance', (obj, type) => {
            return obj.__class__ ? builtins.issubclass(obj.__class__, type) : false;
        });
        this.registerFunction('builtins.hasattr', (obj, name) => {
            return Object.prototype.hasOwnProperty.call(obj, name);
        });
        this.registerFunction('builtins.getattr', (obj, name, defaultValue) => {
            if (Object.prototype.hasOwnProperty.call(obj, name)) {
                return obj[name];
            }
            return defaultValue;
        });
        this.registerFunction('builtins.setattr', (obj, name, value) => {
            obj[name] = value;
        });
        this.registerType('builtins.set', class extends Set {});
        this.registerType('builtins.slice', class {
            constructor(start, stop, step) {
                this.start = start;
                this.stop = stop;
                this.step = step;
            }
        });
        this.registerFunction('builtins.hash');
        this.registerFunction('cloudpickle.cloudpickle._builtin_type', (name) => {
            return name;
        });
        this.registerFunction('cloudpickle.cloudpickle._make_skeleton_class');
        this.registerFunction('cloudpickle.cloudpickle._make_function');
        this.registerFunction('cloudpickle.cloudpickle_fast._function_setstate');
        this.registerType('collections.Counter', class {});
        this.registerFunction('collections.defaultdict', (/* default_factory */) => {
            return {};
        });
        this.registerFunction('copy_reg._reconstructor', (cls, base, state) => {
            // copyreg._reconstructor in Python 3
            if (base === '__builtin__.object' || base === self._builtins.object) {
                return self.invoke(cls, []);
            } else if (base === '__builtin__.tuple' || base === self._builtins.tuple) {
                const obj = self.invoke(cls, []);
                for (let i = 0; i < state.length; i++) {
                    obj[i] = state[i];
                }
                return obj;
            }
            throw new python.Error(`Unsupported copy_reg._reconstructor base type '${base}'.`);
        });
        this.registerFunction('copy.deepcopy', (/* x */) => {
            throw new python.Error('Unsupported copy.deepcopy().');
        });
        this.registerFunction('dill._dill._create_array', (f, args, state, npdict) => {
            const array = f(...args);
            if (array.__setstate__) {
                array.__setstate__(state);
            }
            if (npdict) {
                throw new python.Error("'dill._dill._create_array::npdict' not implemented.");
            }
            return array;
        });
        this.registerFunction('dill._dill._create_cell', (/* args */) => {
            return function() {
            };
        });
        this.registerFunction('dill._dill._create_code', (args) => {
            return self.invoke('types.CodeType', [args]);
        });
        this.registerFunction('dill._dill._create_function', (/* fcode, fglobals, fname, fdefaults, fclosure, fdict, fkwdefaults */) => {
            return function() {
            };
        });
        this.registerFunction('dill._dill._create_namedtuple', (name, fieldnames, modulename /*, defaults */) => {
            const obj = execution.invoke('dill._dill._import_module', [`${modulename}.${name}`]);
            if (obj) {
                return obj;
            }
            return undefined;
        });
        this.registerFunction('dill._dill._create_type', (/* typeobj */) => {
            // return execution.invoke(typeobj, Array.from(arguments).slice(1));
            throw new python.Error("'dill._dill._create_type' not implemented.");
        });
        this.registerFunction('dill._dill._eval_repr');
        this.registerFunction('dill._dill._get_attr', (self, name) => {
            if (Object.prototype.hasOwnProperty.call(self, name)) {
                return self[name];
            }
            return undefined;
        });
        this.registerFunction('dill._dill._import_module', (import_name, safe) => {
            try {
                if (import_name.startsWith('__runtime__.')) {
                    return execution.module(import_name);
                } else if (import_name.indexOf('.') === -1) {
                    return execution.__import__(import_name);
                }
                return execution.resolve(import_name);
            } catch (error) {
                if (safe) {
                    return null;
                }
                throw error;
            }
        });
        this.registerFunction('dill._dill._load_type', (name) => {
            const _dill = self.register('dill._dill');
            if (!_dill._reverse_typemap) {
                _dill._reverse_typemap = new Map();
                for (const name of ['__builtin__', 'types']) {
                    const module = self.register(name);
                    for (const [name, obj] of Object.entries(module)) {
                        if (obj.__module__ === 'builtins' &&
                        obj.__class__ === builtins.type) {
                            _dill._reverse_typemap.set(name, obj);
                        }
                    }
                }
                _dill._reverse_typemap.set('PartialType', functools.partial);
                _dill._reverse_typemap.set('CellType', builtins.cell);
            }
            if (!_dill._reverse_typemap.has(name)) {
                throw new python.Error(`Unknown type name '${name}' in 'dill._dill._load_type'.`);
            }
            return _dill._reverse_typemap.get(name);
        });
        this.registerFunction('keras.saving.pickle_utils.deserialize_model_from_bytecode', (/* serialized_model */) => {
            return null; // throw new python.Error("'keras.saving.pickle_utils.deserialize_model_from_bytecode' not implemented.");
        });
        this.registerFunction('keras.src.saving.pickle_utils.deserialize_model_from_bytecode', keras.saving.pickle_utils.deserialize_model_from_bytecode);
        this.registerFunction('lasagne.nonlinearities.rectify');
        this.registerFunction('lasagne.nonlinearities.softmax');
        this.registerFunction('lasagne.objectives.categorical_crossentropy');
        this.registerFunction('lasagne.updates.nesterov_momentum');
        this.registerFunction('msgpack.unpackb', (packed, ext_hook) => {
            const BinaryReader = class {
                constructor(buffer, ext_hook) {
                    // https://github.com/msgpack/msgpack-javascript/blob/master/src/Decoder.ts
                    // https://github.com/msgpack/msgpack-python/blob/main/msgpack/_unpacker.pyx
                    this._buffer = buffer;
                    this._position = 0;
                    this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                    this._ext_hook = ext_hook;
                }
                value() {
                    const c = this._view.getUint8(this.skip(1));
                    if (c >= 0xe0) {
                        return c - 0x100;
                    }
                    if (c < 0xC0) {
                        if (c < 0x80) {
                            return c;
                        }
                        if (c < 0x90) {
                            return this.map(c - 0x80);
                        }
                        if (c < 0xa0) {
                            return this.array(c - 0x90);
                        }
                        return this.string(c - 0xa0);
                    }
                    switch (c) {
                        case 0xC0: return null;
                        case 0xC2: return false;
                        case 0xC3: return true;
                        case 0xC4: return this.read(this._view.getUint8(this.skip(1)));
                        case 0xC5: return this.read(this._view.getUint16(this.skip(2)));
                        case 0xC6: return this.read(this._view.getUint32(this.skip(4)));
                        case 0xC7: return this.extension(this._view.getUint8(this.skip(1)));
                        case 0xC8: return this.extension(this._view.getUint16(this.skip(2)));
                        case 0xC9: return this.extension(this._view.getUint32(this.skip(4)));
                        case 0xCA: return this._view.getFloat32(this.skip(4));
                        case 0xCB: return this._view.getFloat64(this.skip(8));
                        case 0xCC: return this._view.getUint8(this.skip(1));
                        case 0xCD: return this._view.getUint16(this.skip(2));
                        case 0xCE: return this._view.getUint32(this.skip(4));
                        case 0xCF: return this._view.getBitUint64(this.skip(8));
                        case 0xD0: return this._view.getInt8(this.skip(1));
                        case 0xD1: return this._view.getInt16(this.skip(2));
                        case 0xD2: return this._view.getInt32(this.skip(4));
                        case 0xD3: return this._view.getBigInt64(this.skip(8));
                        case 0xD4: return this.extension(1);
                        case 0xD5: return this.extension(2);
                        case 0xD6: return this.extension(4);
                        case 0xD7: return this.extension(8);
                        case 0xD8: return this.extension(16);
                        case 0xD9: return this.string(this._view.getUint8(this.skip(1)));
                        case 0xDA: return this.string(this._view.getUint16(this.skip(2)));
                        case 0xDB: return this.string(this._view.getUint32(this.skip(4)));
                        case 0xDC: return this.array(this._view.getUint16(this.skip(2)));
                        case 0xDD: return this.array(this._view.getUint32(this.skip(4)));
                        case 0xDE: return this.map(this._view.getUint16(this.skip(2)));
                        case 0xDF: return this.map(this._view.getUint32(this.skip(4)));
                        default: throw new python.Error(`Invalid code '${c}'.`);
                    }
                }
                map(size) {
                    const map = {};
                    for (let i = 0; i < size; i++) {
                        const key = this.value();
                        const value = this.value();
                        map[key] = value;
                    }
                    return map;
                }
                array(size) {
                    const array = new Array(size);
                    for (let i = 0; i < size; i++) {
                        array[i] = this.value();
                    }
                    return array;
                }
                extension(size) {
                    const code = this._view.getUint8(this.skip(1));
                    const data = this.read(size);
                    return this._ext_hook(code, data);
                }
                skip(offset) {
                    const position = this._position;
                    this._position += offset;
                    if (this._position > this._buffer.length) {
                        throw new python.Error(`Expected ${this._position - this._buffer.length} more bytes. The file might be corrupted. Unexpected end of file.`);
                    }
                    return position;
                }
                read(size) {
                    const data = this._buffer.subarray(this._position, this._position + size);
                    this._position += size;
                    return data;
                }
                string(size) {
                    const buffer = this.read(size);
                    this._decoder = this._decoder || new TextDecoder('utf8');
                    return this._decoder.decode(buffer);
                }
            };
            return new BinaryReader(packed, ext_hook).value();
        });
        this.registerFunction('nolearn.lasagne.base.objective');
        this.registerFunction('numpy.core._DType_reconstruct');
        this.registerFunction('numpy.core.multiarray._reconstruct', (subtype, shape, dtype) => {
            return numpy.ndarray.__new__(subtype, shape, dtype);
        });
        numpy.core._multiarray_umath._reconstruct = numpy.core.multiarray._reconstruct;
        this.registerFunction('numpy.core.multiarray.scalar', (dtype, rawData) => {
            let data = rawData;
            if (typeof rawData === 'string' || rawData instanceof String) {
                data = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; i++) {
                    data[i] = rawData.charCodeAt(i);
                }
            }
            switch (dtype.kind) {
                case 'b': {
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.itemsize) {
                        case 1: return view.getInt8(0) ? true : false;
                        default: throw new python.Error(`Unsupported scalar dtype boolean itemsize '${dtype.itemsize}'.`);
                    }
                }
                case 'f': {
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.itemsize) {
                        case 2: return view.getFloat16(0, dtype.byteorder === '<');
                        case 4: return view.getFloat32(0, dtype.byteorder === '<');
                        case 8: return view.getFloat64(0, dtype.byteorder === '<');
                        default: throw new python.Error(`Unsupported scalar dtype float itemsize '${dtype.itemsize}'.`);
                    }
                }
                case 'i': {
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.itemsize) {
                        case 1: return view.getInt8(0);
                        case 2: return view.getInt16(0, dtype.byteorder === '<');
                        case 4: return view.getInt32(0, dtype.byteorder === '<');
                        case 8: return view.getBigInt64(0, dtype.byteorder === '<');
                        default: throw new python.Error(`Unsupported scalar dtype int itemsize '${dtype.itemsize}'.`);
                    }
                }
                case 'u': {
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    switch (dtype.itemsize) {
                        case 1: return view.getUint8(0);
                        case 2: return view.getUint16(0, dtype.byteorder === '<');
                        case 4: return view.getUint32(0, dtype.byteorder === '<');
                        case 8: return view.getBigUint64(0, dtype.byteorder === '<');
                        default: throw new python.Error(`Unsupported scalar dtype uint itemsize '${dtype.itemsize}'.`);
                    }
                }
                case 'U': {
                    const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
                    const list = [];
                    for (let i = 0; i < dtype.itemsize; i += 4) {
                        list.push(String.fromCodePoint(view.getUint32(i, true)));
                    }
                    return list.join('');
                }
                default: {
                    throw new python.Error(`Unsupported scalar dtype kind '${dtype.kind}'.`);
                }
            }
        });
        this.registerFunction('numpy.core._multiarray_umath.scalar', (dtype, rawData) => {
            let data = rawData;
            if (typeof rawData === 'string') {
                data = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; i++) {
                    data[i] = rawData.charCodeAt(i);
                }
            }
            const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
            switch (dtype.__name__) {
                case 'uint8':
                    return dataView.getUint8(0);
                case 'float32':
                    return dataView.getFloat32(0, true);
                case 'float64':
                    return dataView.getFloat64(0, true);
                case 'int8':
                    return dataView.getInt8(0, true);
                case 'int16':
                    return dataView.getInt16(0, true);
                case 'int32':
                    return dataView.getInt32(0, true);
                case 'int64':
                    return dataView.getBigInt64(0, true);
                default:
                    throw new python.Error(`Unsupported scalar type '${dtype.__name__}'.`);
            }
        });
        this.registerFunction('numpy.load', (file) => {
            // https://github.com/numpy/numpy/blob/main/numpy/lib/format.py
            const signature = [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59];
            if (!file.read(6).every((v, i) => v === signature[i])) {
                throw new python.Error('Invalid signature.');
            }
            const version = file.read(2);
            const [major, minor] = version;
            if (major > 3) {
                throw new python.Error(`Invalid version '${[major, minor].join('.')}'.`);
            }
            const buffer = new Uint8Array([0, 0, 0, 0]);
            buffer.set(file.read(major >= 2 ? 4 : 2), 0);
            const header_length = buffer[3] << 24 | buffer[2] << 16 | buffer[1] << 8 | buffer[0];
            let header = file.read(header_length);
            const decoder = new TextDecoder(major >= 3 ? 'utf-8' : 'ascii');
            header = decoder.decode(header);
            header = JSON.parse(header.replace(/\(/,'[').replace(/\)/,']').replace('[,','[1,]').replace(',]',']').replace(/'/g, '"').replace(/:\s*False\s*,/,':false,').replace(/:\s*True\s*,/,':true,').replace(/,\s*\}/, ' }'));
            if (!header.descr || header.descr.length < 2) {
                throw new python.Error("Missing property 'descr'.");
            }
            if (!header.shape) {
                throw new python.Error("Missing property 'shape'.");
            }
            const shape = header.shape;
            const dtype = self.invoke('numpy.dtype', [header.descr.substring(1)]);
            dtype.byteorder = header.descr.substring(0, 1);
            let data = null;
            switch (dtype.byteorder) {
                case '|': {
                    data = file.read();
                    if (dtype.kind === 'O') {
                        const unpickler = execution.invoke('pickle.Unpickler', [data]);
                        return unpickler.load();
                    }
                    break;
                }
                case '>':
                case '<': {
                    if (header.descr.length !== 3 && header.descr[1] !== 'U') {
                        throw new python.Error(`Unsupported data type '${header.descr}'.`);
                    }
                    const count = shape.length === 0 ? 1 : shape.reduce((a, b) => a * b, 1);
                    data = file.read(dtype.itemsize * count);
                    break;
                }
                default: {
                    throw new python.Error(`Unsupported data type '${header.descr}'.`);
                }
            }
            if (header.fortran_order) {
                data = null;
            }
            return self.invoke('numpy.ndarray', [shape, dtype, data]);
        });
        this.registerFunction('numpy.save', (file, arr) => {
            const descr = arr.dtype.str;
            if (descr[0] !== '<' && descr[0] !== '>') {
                throw new python.Error(`Unsupported byte order '${descr}'.`);
            }
            if (descr.length !== 3 || (descr[1] !== 'f' && descr[1] !== 'i' && descr[1] !== 'u' && descr[1] !== 'c' && descr.substring(1) !== 'b1')) {
                throw new python.Error(`Unsupported data type '${descr}'.`);
            }
            let shape = '';
            switch (arr.shape.length) {
                case 0: shape = '()'; break;
                case 1: shape = `(${arr.shape[0]},)`; break;
                default: shape = `(${arr.shape.map((dimension) => dimension.toString()).join(', ')})`; break;
            }
            const properties = [
                `'descr': '${descr}'`,
                "'fortran_order': False",
                `'shape': ${shape}`
            ];
            let header = `{ ${properties.join(', ')} }`;
            header += `${' '.repeat(64 - ((header.length + 2 + 8 + 1) & 0x3f))}\n`;
            const encoder = new TextEncoder('ascii');
            file.write([0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59, 0x01, 0x00]); // '\\x93NUMPY' + version
            file.write([header.length & 0xff, (header.length >> 8) & 0xff]);
            file.write(encoder.encode(header));
            file.write(arr.tobytes());
        });
        this.registerFunction('numpy.asarray', (a, dtype) => {
            const encode = (context, data, dim) => {
                const size = context.shape[dim];
                const littleendian = context.littleendian;
                if (dim === context.shape.length - 1) {
                    for (let i = 0; i < size; i++) {
                        switch (context.dtype) {
                            case 'f2':
                                context.view.setFloat16(context.position, data[i], littleendian);
                                break;
                            case 'f4':
                                context.view.setFloat32(context.position, data[i], littleendian);
                                break;
                            case 'f8':
                                context.view.setFloat64(context.position, data[i], littleendian);
                                break;
                            case 'i1':
                                context.view.setInt8(context.position, data[i], littleendian);
                                break;
                            case 'i2':
                                context.view.setInt16(context.position, data[i], littleendian);
                                break;
                            case 'i4':
                                context.view.setInt32(context.position, data[i], littleendian);
                                break;
                            case 'i8':
                                context.view.setBigInt64(context.position, data[i], littleendian);
                                break;
                            case 'u1':
                                context.view.setUint8(context.position, data[i], littleendian);
                                break;
                            case 'u2':
                                context.view.setUint16(context.position, data[i], littleendian);
                                break;
                            case 'u4':
                                context.view.setUint32(context.position, data[i], littleendian);
                                break;
                            case 'u8':
                                context.view.setBigUint64(context.position, data[i], littleendian);
                                break;
                            case 'c8':
                                context.view.setComplex64(context.position, data[i], littleendian);
                                break;
                            case 'c16':
                                context.view.setComplex128(context.position, data[i], littleendian);
                                break;
                            case 'b1':
                                context.view.setInt8(context.position, data[i] ? 1 : 0);
                                break;
                            default:
                                throw new python.Error(`Unsupported tensor data type '${context.dtype}'.`);
                        }
                        context.position += context.itemsize;
                    }
                } else {
                    for (let j = 0; j < size; j++) {
                        encode(context, data[j], dim + 1);
                    }
                }
            };
            const array_size = (value) => {
                if (value.every((item) => Array.isArray(item))) {
                    const dims = value.map((item) => array_size(item));
                    const [dim] = dims;
                    for (let i = 1; i < dims.length; i++) {
                        if (dim.length === dims[i].length) {
                            if (!dims[i].every((value, i) => value === dim[i])) {
                                throw new python.Error('Invalid array shape.');
                            }
                        }
                    }
                    return [value.length].concat(dim);
                }
                return [value.length];
            };
            const shape = Array.isArray(a) ? array_size(a) : [];
            const size = dtype.itemsize * shape.reduce((a, b) => a * b, 1);
            const context = {
                position: 0,
                itemsize: dtype.itemsize,
                dtype: dtype.str.substring(1),
                littleendian: dtype.str[0],
                shape,
                data: new Uint8Array(size)
            };
            context.view = new DataView(context.data.buffer, context.data.byteOffset, size);
            encode(context, a, 0);
            return self.invoke('numpy.ndarray', [shape, dtype, context.data]);
        });
        this.registerFunction('numpy.mean');
        this.registerFunction('numpy.ma.core._mareconstruct', (subtype, baseclass, baseshape, basetype) => {
            const data = self.invoke(baseclass, [baseshape, basetype]);
            // = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
            const mask = self.invoke('numpy.ndarray', [baseshape, '']);
            return self.invoke(subtype, [data, mask, basetype]);
        });
        this.registerFunction('numpy.random.__RandomState_ctor', () => {
            return {};
        });
        this.registerFunction('numpy.random._pickle.__randomstate_ctor', () => {
            return {};
        });
        this.registerType('numpy.random.bit_generator.BitGenerator', class {});
        this.registerType('numpy.random._mt19937.MT19937', class extends numpy.random.bit_generator.BitGenerator {});
        this.registerType('numpy.random._pcg64.PCG64', class extends numpy.random.bit_generator.BitGenerator {});
        this.registerType('numpy.random._pcg64.PCG64DXSM', class extends numpy.random.bit_generator.BitGenerator {});
        this.registerType('numpy.random._philox.Philox', class extends numpy.random.bit_generator.BitGenerator {});
        this.registerType('numpy.random._sfc64.SFC64', class extends numpy.random.bit_generator.BitGenerator {});
        numpy.random._pickle.BitGenerators = {
            'MT19937': numpy.random._mt19937.MT19937,
            'PCG64': numpy.random._pcg64.PCG64,
            'PCG64DXSM': numpy.random._pcg64.PCG64DXSM,
            'Philox': numpy.random._philox.Philox,
            'SFC64': numpy.random._sfc64.SFC64,
        };
        this.registerType('numpy.random._generator.Generator', class {
            constructor(bit_generator) {
                this.bit_generator = bit_generator;
            }
        });
        this.registerFunction('numpy.random._pickle.__bit_generator_ctor', (bit_generator_name) => {
            bit_generator_name = bit_generator_name || 'MT19937';
            const bit_generator = numpy.random._pickle.BitGenerators[bit_generator_name];
            if (bit_generator) {
                return new bit_generator();
            }
            throw new python.Error(`Unknown bit generator '${bit_generator_name}'.`);
        });
        this.registerFunction('numpy.random._pickle.__generator_ctor', (bit_generator_name, bit_generator_ctor) => {
            bit_generator_ctor = bit_generator_ctor || numpy.random._pickle.__bit_generator_ctor;
            return new numpy.random._generator.Generator(bit_generator_ctor(bit_generator_name));
        });
        this.registerFunction('numpy.reshape');
        this.registerFunction('numpy.core.numeric._frombuffer', (/* buf, dtype, shape, order */) => {
            return {};
        });
        this.registerFunction('sklearn.feature_selection._univariate_selection.f_classif');
        this.registerFunction('sklearn.metrics.scorer._passthrough_scorer');
        this.registerFunction('sklearn.metrics._classification.accuracy_score');
        this.registerFunction('sklearn.metrics._classification.f1_score');
        this.registerFunction('sklearn.metrics._classification.precision_score');
        this.registerFunction('sklearn.metrics._classification.recall_score');
        this.registerFunction('sklearn.metrics._dist_metrics.newObj', (obj) => {
            return obj.__new__(obj);

        });
        this.registerFunction('sklearn.metrics._regression.mean_absolute_error');
        this.registerFunction('sklearn.metrics._regression.mean_squared_error');
        this.registerFunction('sklearn.metrics._regression.root_mean_squared_error');
        this.registerFunction('re._compile', (pattern, flags) => {
            return self.invoke('re.Pattern', [pattern, flags]);
        });
        this.registerFunction('srsly.cloudpickle.cloudpickle._builtin_type', (...args) => {
            return function() {
                return self.invoke(`types.${args[0]}`, args);
            };
        });
        this.registerFunction('theano.scalar.basic.same_out');
        this.registerFunction('theano.scalar.basic.same_out_nocomplex');
        this.registerFunction('theano.scalar.basic.upcast_out');
        this.registerFunction('theano.scalar.basic.upgrade_to_float');
        this.registerFunction('theano.tensor.nnet.conv2d');
        this.registerFunction('theano.tensor.type.values_eq_approx_remove_inf_nan');
        this.registerFunction('theano.tensor.type.values_eq_approx_remove_nan');
        this.registerType('torch.nn.modules.module.Module', class {
            constructor() {
                this._modules = execution.invoke('collections.OrderedDict', []);
                this._parameters = execution.invoke('collections.OrderedDict', []);
                this._buffers = execution.invoke('collections.OrderedDict', []);
            }
            __setattr__(name, value) {
                if (value instanceof torch.nn.modules.module.Module) {
                    this._modules.set(name, value);
                } else {
                    this[name] = value;
                }
            }
            __getattr__(name) {
                if (this._modules.has(name)) {
                    return this._modules.get(name);
                }
                return this[name];
            }
        });
        torch.nn.Module = torch.nn.modules.module.Module;
        torch.nn.modules.Module = torch.nn.modules.module.Module;
        this.registerType('torch.ao.quantization.fake_quantize.FakeQuantize', class {});
        this.registerType('torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize', class {});
        this.registerType('torch.ao.quantization.observer._PartialWrapper', class {});
        this.registerType('torch.ao.quantization.observer.HistogramObserver', class {});
        this.registerType('torch.ao.quantization.observer.MovingAverageMinMaxObserver', class {});
        this.registerType('torch.ao.quantization.observer.MovingAveragePerChannelMinMaxObserver', class {});
        this.registerType('torch.ao.quantization.observer.MinMaxObserver', class {});
        this.registerType('torch.ao.quantization.observer.PerChannelMinMaxObserver', class {});
        this.registerType('torch.ao.quantization.observer.PlaceholderObserver', class {});
        this.registerType('torch.ao.quantization.qconfig.QConfig', class {});
        this.registerType('torch.ao.quantization.qconfig.QConfigDynamic', class {});
        this.registerType('torch.ao.quantization.stubs.DeQuantStub', class {});
        this.registerType('torch.ao.quantization.stubs.QuantStub', class {});
        this.registerType('torch.ao.quantization.stubs.QuantWrapper', class {});
        this.registerFunction('torch.ao.quantization.fx.graph_module._save_packed_weight');
        this.registerFunction('torch.ao.quantization.fx._lower_to_native_backend._load_packed_weight');
        this.registerFunction('torch.ao.quantization.fx._lower_to_native_backend._save_packed_weight');
        this.registerFunction('torch.ao.quantization.quantization_mappings._get_special_act_post_process');
        this.registerFunction('torch.ao.quantization.quantization_mappings.get_default_dynamic_quant_module_mappings');
        this.registerFunction('torch.ao.quantization.quantization_mappings.get_default_qat_module_mappings');
        this.registerFunction('torch.ao.quantization.quantization_mappings.get_default_qconfig_propagation_list');
        this.registerFunction('torch.ao.quantization.quantization_mappings.get_default_static_quant_module_mappings');
        this.registerFunction('torch.ao.quantization.quantization_mappings.get_default_static_quant_reference_module_mappings');
        this.registerFunction('torch.ao.quantization.quantization_mappings.no_observer_set');
        this.registerFunction('torch.ao.quantization.quantization_mappings._has_special_act_post_process');
        this.registerFunction('torch.ao.quantization.utils.get_qparam_dict');
        this.registerFunction('torch.ao.quantization.utils.has_no_children_ignoring_parametrizations');
        this.registerType('torch.autograd.variable.Variable', class {});
        this.registerType('torch.backends.cudnn.rnn.Unserializable', class {});
        this.registerFunction('torch.distributed._shard.sharded_tensor.pre_load_state_dict_hook');
        this.registerFunction('torch.distributed._shard.sharded_tensor.state_dict_hook');
        this.registerType('torch.distributed.algorithms.join._JoinConfig', class {});
        this.registerType('torch.distributions.bernoulli.Bernoulli', class {});
        this.registerType('torch.distributions.beta.Beta', class {});
        this.registerType('torch.distributions.binomial.Binomial', class {});
        this.registerType('torch.distributions.categorical.Categorical', class {});
        this.registerType('torch.distributions.constraints._LowerCholesky', class {});
        this.registerType('torch.distributions.constraints._Real', class {});
        this.registerType('torch.distributions.dirichlet.Dirichlet', class {});
        this.registerType('torch.distributions.multivariate_normal.MultivariateNormal', class {});
        this.registerType('torch.distributions.normal.Normal', class {});
        this.registerType('torch.distributions.transforms._InverseTransform', class {});
        this.registerType('torch.distributions.transforms.AffineTransform', class {});
        this.registerType('torch.distributions.transforms.ComposeTransform', class {});
        this.registerType('torch.distributions.transforms.ExpTransform', class {});
        this.registerType('torch.distributions.transforms.LowerCholeskyTransform', class {});
        this.registerType('torch.distributions.uniform.Uniform', class {});
        this.registerType('torch.nn.backends.thnn._get_thnn_function_backend', class {});
        this.registerType('torch.nn.intrinsic.modules.fused._FusedModule', class {});
        this.registerType('torch.nn.intrinsic.modules.fused.ConvBnReLU2d', class {});
        this.registerType('torch.nn.intrinsic.modules.fused.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.modules.fused.BNReLU2d', class {});
        this.registerType('torch.nn.intrinsic.qat.modules.conv_fused.ConvBn2d', class {});
        this.registerType('torch.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d', class {});
        this.registerType('torch.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d', class {});
        this.registerType('torch.nn.intrinsic.quantized.modules.linear_relu.LinearReLU', class {});
        this.registerType('torch.nn.modules.activation.CELU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.ELU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.GELU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.GLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Hardtanh', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Hardshrink', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Hardsigmoid', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Hardswish', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.LeakyReLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.LogSigmoid', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.LogSoftmax', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Mish', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.MultiheadAttention', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.ReLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.ReLU6', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.PReLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.RReLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.SELU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Sigmoid', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.SiLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softmax', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softmax2d', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softmin', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softplus', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softshrink', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Softsign', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Tanh', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Tanhshrink', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.activation.Threshold', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.adaptive.AdaptiveLogSoftmaxWithLoss', class {});
        this.registerType('torch.nn.modules.batchnorm._NormBase', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.batchnorm._BatchNorm', class extends torch.nn.modules.batchnorm._NormBase {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm1d', class extends torch.nn.modules.batchnorm._BatchNorm {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm2d', class extends torch.nn.modules.batchnorm._BatchNorm {});
        this.registerType('torch.nn.modules.batchnorm.BatchNorm3d', class extends torch.nn.modules.batchnorm._BatchNorm {});
        this.registerType('torch.nn.modules.batchnorm.LazyBatchNorm1d', class {});
        this.registerType('torch.nn.modules.batchnorm.LazyBatchNorm2d', class {});
        this.registerType('torch.nn.modules.batchnorm.LazyBatchNorm3d', class {});
        this.registerType('torch.nn.modules.batchnorm.SyncBatchNorm', class {});
        this.registerType('torch.nn.modules.channelshuffle.ChannelShuffle', class {});
        this.registerType('torch.nn.modules.container.ModuleDict', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.container.ModuleList', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.container.ParameterDict', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.container.ParameterList', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.container.Sequential', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.conv._ConvNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.conv.Conv1d', class extends torch.nn.modules.conv._ConvNd {});
        this.registerType('torch.nn.modules.conv.Conv2d', class extends torch.nn.modules.conv._ConvNd {});
        this.registerType('torch.nn.modules.conv.Conv3d', class extends torch.nn.modules.conv._ConvNd {});
        this.registerType('torch.nn.modules.conv._ConvTransposeNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.conv.ConvTranspose1d', class extends torch.nn.modules.conv._ConvTransposeNd {});
        this.registerType('torch.nn.modules.conv.ConvTranspose2d', class extends torch.nn.modules.conv._ConvTransposeNd {});
        this.registerType('torch.nn.modules.conv.ConvTranspose3d', class extends torch.nn.modules.conv._ConvTransposeNd {});
        this.registerType('torch.nn.modules.conv.LazyConv1d', class {});
        this.registerType('torch.nn.modules.conv.LazyConv2d', class {});
        this.registerType('torch.nn.modules.conv.LazyConv3d', class {});
        this.registerType('torch.nn.modules.conv.LazyConvTranspose2d', class {});
        this.registerType('torch.nn.modules.distance.CosineSimilarity', class {});
        this.registerType('torch.nn.modules.dropout._DropoutNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.dropout.AlphaDropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout1d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout2d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout3d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.FeatureAlphaDropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.fold.Fold', class {});
        this.registerType('torch.nn.modules.fold.Unfold', class {});
        this.registerType('torch.nn.modules.flatten.Flatten', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.flatten.Unflatten', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm1d', class {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm2d', class {});
        this.registerType('torch.nn.modules.instancenorm.InstanceNorm3d', class {});
        this.registerType('torch.nn.modules.instancenorm.LazyInstanceNorm2d', class {});
        this.registerType('torch.nn.modules.linear._LinearWithBias', class {});
        this.registerType('torch.nn.modules.linear.Bilinear', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.linear.Identity', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.linear.LazyLinear', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.linear.Linear', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.linear.NonDynamicallyQuantizableLinear', class extends torch.nn.modules.linear.Linear {});
        this.registerType('torch.nn.modules.loss._Loss', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.loss._WeightedLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.BCELoss', class extends torch.nn.modules.loss._WeightedLoss {});
        this.registerType('torch.nn.modules.loss.BCEWithLogitsLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.CrossEntropyLoss', class extends torch.nn.modules.loss._WeightedLoss {});
        this.registerType('torch.nn.modules.loss.CosineEmbeddingLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.CTCLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.GaussianNLLLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.HuberLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.HingeEmbeddingLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.KLDivLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.L1Loss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.MarginRankingLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.MultiLabelMarginLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.MultiLabelSoftMarginLoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.MultiMarginLoss', class extends torch.nn.modules.loss._WeightedLoss {});
        this.registerType('torch.nn.modules.loss.MSELoss', class extends torch.nn.modules.loss._Loss {});
        this.registerType('torch.nn.modules.loss.NLLLoss', class extends torch.nn.modules.loss._WeightedLoss {});
        this.registerType('torch.nn.modules.loss.NLLLoss2d', class extends torch.nn.modules.loss.NLLLoss {});
        this.registerType('torch.nn.modules.loss.PoissonNLLLoss', class {});
        this.registerType('torch.nn.modules.loss.SmoothL1Loss', class {});
        this.registerType('torch.nn.modules.loss.SoftMarginLoss', class {});
        this.registerType('torch.nn.modules.loss.TripletMarginLoss', class {});
        this.registerType('torch.nn.modules.loss.TripletMarginWithDistanceLoss', class {});
        this.registerType('torch.nn.modules.module._IncompatibleKeys', class {});
        this.registerType('torch.nn.modules.module._WrappedHook', class {});
        this.registerType('torch.nn.modules.module.PatchForward', class {});
        this.registerType('torch.nn.modules.normalization.CrossMapLRN2d', class {});
        this.registerType('torch.nn.modules.normalization.GroupNorm', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.normalization.LayerNorm', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.normalization.LocalResponseNorm', class {});
        this.registerType('torch.nn.modules.padding._ReflectionPadNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.padding.ReflectionPad1d', class extends torch.nn.modules.padding._ReflectionPadNd {});
        this.registerType('torch.nn.modules.padding.ReflectionPad2d', class extends torch.nn.modules.padding._ReflectionPadNd {});
        this.registerType('torch.nn.modules.padding.ReflectionPad3d', class extends torch.nn.modules.padding._ReflectionPadNd {});
        this.registerType('torch.nn.modules.padding._ReplicationPadNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.padding.ReplicationPad1d', class extends torch.nn.modules.padding._ReplicationPadNd {});
        this.registerType('torch.nn.modules.padding.ReplicationPad2d', class extends torch.nn.modules.padding._ReplicationPadNd {});
        this.registerType('torch.nn.modules.padding.ReplicationPad3d', class extends torch.nn.modules.padding._ReplicationPadNd {});
        this.registerType('torch.nn.modules.padding.ZeroPad2d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad1d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad2d', class {});
        this.registerType('torch.nn.modules.padding.ConstantPad3d', class {});
        this.registerType('torch.nn.modules.pixelshuffle.PixelShuffle', class {});
        this.registerType('torch.nn.modules.pixelshuffle.PixelUnshuffle', class {});
        this.registerType('torch.nn.modules.pooling._AdaptiveAvgPoolNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool1d', class extends torch.nn.modules.pooling._AdaptiveAvgPoolNd {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool2d', class extends torch.nn.modules.pooling._AdaptiveAvgPoolNd {});
        this.registerType('torch.nn.modules.pooling.AdaptiveAvgPool3d', class extends torch.nn.modules.pooling._AdaptiveAvgPoolNd {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool1d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.AdaptiveMaxPool3d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool1d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool2d', class {});
        this.registerType('torch.nn.modules.pooling.AvgPool3d', class {});
        this.registerType('torch.nn.modules.pooling.FractionalMaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.LPPool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool1d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxPool3d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool1d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool2d', class {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool3d', class {});
        this.registerType('torch.nn.modules.rnn.GRU', class {});
        this.registerType('torch.nn.modules.rnn.GRUCell', class {});
        this.registerType('torch.nn.modules.rnn.LSTM', class {});
        this.registerType('torch.nn.modules.rnn.LSTMCell', class {});
        this.registerType('torch.nn.modules.rnn.RNNBase', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.rnn.RNN', class extends torch.nn.modules.rnn.RNNBase {});
        this.registerType('torch.nn.modules.rnn.RNNCellBase', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.rnn.RNNCell', class extends torch.nn.modules.rnn.RNNCellBase {});
        this.registerType('torch.nn.modules.sparse.Embedding', class {});
        this.registerType('torch.nn.modules.sparse.EmbeddingBag', class {});
        this.registerType('torch.nn.modules.transformer.Transformer', class {});
        this.registerType('torch.nn.modules.transformer.TransformerDecoder', class {});
        this.registerType('torch.nn.modules.transformer.TransformerDecoderLayer', class {});
        this.registerType('torch.nn.modules.transformer.TransformerEncoder', class {});
        this.registerType('torch.nn.modules.transformer.TransformerEncoderLayer', class {});
        this.registerType('torch.nn.modules.upsampling.Upsample', class {});
        this.registerType('torch.nn.modules.upsampling.UpsamplingBilinear2d', class {});
        this.registerType('torch.nn.modules.upsampling.UpsamplingNearest2d', class {});
        this.registerType('torch.nn.parallel.data_parallel.DataParallel', class {});
        this.registerType('torch.nn.parallel.distributed._DDPUnevenInputsConfig', class {});
        this.registerType('torch.nn.parallel.distributed.DistributedDataParallel', class {});
        this.registerType('torch.nn.qat.modules.conv.Conv2d', class {});
        this.registerType('torch.nn.qat.modules.linear.Linear', class {});
        this.registerType('torch.nn.quantized.modules.activation.ReLU', class {});
        this.registerType('torch.nn.quantized.modules.activation.LeakyReLU', class {});
        this.registerType('torch.nn.quantized.modules.activation.Softmax', class {});
        this.registerType('torch.nn.quantized.dynamic.modules.linear.Linear', class {});
        this.registerType('torch.nn.quantized.dynamic.modules.rnn.GRU', class {});
        this.registerType('torch.nn.quantized.dynamic.modules.rnn.LSTM', class {});
        this.registerType('torch.nn.quantized.dynamic.modules.rnn.LSTMCell', class {});
        this.registerType('torch.nn.quantized.dynamic.modules.rnn.PackedParameter', class {});
        this.registerType('torch.nn.quantized.modules.activation.ReLU6', class {});
        this.registerType('torch.nn.quantized.modules.batchnorm.BatchNorm2d', class {});
        this.registerType('torch.nn.quantized.modules.conv.Conv1d', class {});
        this.registerType('torch.nn.quantized.modules.conv.Conv2d', class {});
        this.registerType('torch.nn.quantized.modules.conv.ConvTranspose2d', class {});
        this.registerType('torch.nn.quantized.modules.DeQuantize', class {});
        this.registerType('torch.nn.quantized.modules.dropout.Dropout', class {});
        this.registerType('torch.nn.quantized.modules.embedding_ops.Embedding', class {});
        this.registerType('torch.nn.quantized.modules.embedding_ops.EmbeddingPackedParams', class {});
        this.registerType('torch.nn.quantized.modules.functional_modules.FloatFunctional', class {});
        this.registerType('torch.nn.quantized.modules.functional_modules.QFunctional', class {});
        this.registerType('torch.nn.quantized.modules.linear.Linear', class {});
        this.registerType('torch.nn.quantized.modules.linear.LinearPackedParams', class {});
        this.registerType('torch.nn.quantized.modules.normalization.InstanceNorm2d', class {});
        this.registerType('torch.nn.quantized.modules.normalization.GroupNorm', class extends torch.nn.modules.normalization.GroupNorm {});
        this.registerType('torch.nn.quantized.modules.normalization.LayerNorm', class extends torch.nn.modules.normalization.LayerNorm {});
        this.registerType('torch.nn.quantized.modules.Quantize', class {});
        this.registerType('torch.ao.nn.quantizable.modules.activation.MultiheadAttention', class extends torch.nn.modules.activation.MultiheadAttention {});
        this.registerType('torch.ao.nn.quantizable.modules.rnn._LSTMLayer', class {});
        this.registerType('torch.ao.nn.quantizable.modules.rnn._LSTMSingleLayer', class {});
        this.registerType('torch.ao.nn.quantizable.modules.rnn.LSTM', class {});
        this.registerType('torch.ao.nn.quantizable.modules.rnn.LSTMCell', class {});
        this.registerType('torch.ao.nn.quantized.modules.activation.MultiheadAttention', class extends torch.ao.nn.quantizable.modules.activation.MultiheadAttention {});
        this.registerType('torch.ao.nn.quantized.modules.activation.ReLU6', class extends torch.nn.modules.activation.ReLU {});
        this.registerType('torch.ao.nn.quantized.modules.activation.LeakyReLU', class extends torch.nn.modules.activation.LeakyReLU {});
        this.registerType('torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.batchnorm._BatchNorm',  class extends torch.nn.modules.batchnorm._BatchNorm {});
        this.registerType('torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d', class extends torch.ao.nn.quantized.modules.batchnorm._BatchNorm {});
        this.registerType('torch.ao.nn.quantized.modules.conv.Conv1d', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.conv.Conv2d', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.conv._ConvNd', class extends torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule {});
        this.registerType('torch.ao.nn.quantized.modules.conv._ConvTransposeNd', class extends torch.ao.nn.quantized.modules.conv._ConvNd {});
        this.registerType('torch.ao.nn.quantized.modules.conv.ConvTranspose2d', class extends torch.ao.nn.quantized.modules.conv._ConvTransposeNd {});
        this.registerType('torch.ao.nn.quantized.modules.Quantize', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.DeQuantize', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.dropout.Dropout', class extends torch.nn.modules.dropout.Dropout {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.FloatFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.QFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.FXFloatFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.linear.Linear', class extends torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule {});
        this.registerType('torch.ao.nn.quantized.modules.linear.LinearPackedParams', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.normalization.LayerNorm', class extends torch.nn.modules.normalization.LayerNorm {});
        this.registerType('torch.ao.nn.quantized.modules.rnn.LSTM', class {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.linear.Linear', class extends torch.ao.nn.quantized.modules.linear.Linear {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.PackedParameter', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.GRU', class extends torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase {});
        this.registerType('torch.ao.nn.quantized.reference.modules.conv.Conv1d', class {});
        this.registerType('torch.ao.nn.quantized.reference.modules.conv.Conv2d', class {});
        this.registerType('torch.ao.nn.quantized.reference.modules.linear.Linear', class {});
        this.registerType('torch.ao.nn.qat.modules.conv.Conv2d', class {});
        this.registerType('torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d', class extends torch.ao.nn.quantized.modules.conv.Conv2d {});
        this.registerType('torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU', class extends torch.ao.nn.quantized.modules.linear.Linear {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused._FusedModule', class extends torch.nn.modules.container.Sequential {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvBn2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvReLU1d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvReLU2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.LinearReLU', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.nn.utils.prune.CustomFromMask', class {});
        this.registerType('torch.nn.utils.prune.L1Unstructured', class {});
        this.registerType('torch.nn.utils.prune.LnStructured', class {});
        this.registerType('torch.nn.utils.prune.PruningContainer', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNorm', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormStateDictHook', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormLoadStateDictPreHook', class {});
        this.registerType('torch.nn.utils.weight_norm.WeightNorm', class {});
        this.registerFunction('torch.nn.utils.parametrize.type_before_parametrizations');
        this.registerType('torch.torch_version.TorchVersion', class extends String {});
        this.registerType('torch.optim.optimizer.Optimizer', class {});
        this.registerType('torch.optim.adam.Adam', class extends torch.optim.optimizer.Optimizer {});
        this.registerType('torch.optim.adamw.AdamW', class {});
        this.registerType('torch.optim.adagrad.Adagrad', class {});
        this.registerType('torch.optim.adadelta.Adadelta', class {});
        this.registerType('torch.optim.lbfgs.LBFGS', class {});
        this.registerType('torch.optim.lr_scheduler.CosineAnnealingLR', class {});
        this.registerType('torch.optim.lr_scheduler.CosineAnnealingWarmRestarts', class {});
        this.registerType('torch.optim.lr_scheduler.CyclicLR', class {});
        this.registerType('torch.optim.lr_scheduler.ExponentialLR', class {});
        this.registerType('torch.optim.lr_scheduler.LambdaLR', class {});
        this.registerType('torch.optim.lr_scheduler.LinearLR', class {});
        this.registerType('torch.optim.lr_scheduler.MultiStepLR', class {});
        this.registerType('torch.optim.lr_scheduler.OneCycleLR', class {});
        this.registerType('torch.optim.lr_scheduler.ReduceLROnPlateau', class {});
        this.registerType('torch.optim.lr_scheduler.StepLR', class {});
        this.registerType('torch.optim.optimizer._RequiredParameter', class {});
        this.registerType('torch.optim.radam.RAdam', class extends torch.optim.optimizer.Optimizer {});
        this.registerType('torch.optim.rmsprop.RMSprop', class {});
        this.registerType('torch.optim.sgd.SGD', class {});
        this.registerType('torch.optim.sparse_adam.SparseAdam', class {});
        this.registerType('torch.optim.swa_utils.SWALR', class {});
        torch.optim.RAdam = torch.optim.radam.RAdam;
        this.registerType('torch.quantization.fake_quantize.FakeQuantize', class {});
        this.registerType('torch.quantization.observer._PartialWrapper', class {});
        this.registerType('torch.quantization.observer.HistogramObserver', class {});
        this.registerType('torch.quantization.observer.MinMaxObserver', class {});
        this.registerType('torch.quantization.observer.MovingAverageMinMaxObserver', class {});
        this.registerType('torch.quantization.observer.MovingAveragePerChannelMinMaxObserver', class {});
        this.registerType('torch.quantization.qconfig.QConfig', class {});
        this.registerType('torch.quantization.stubs.DeQuantStub', class {});
        this.registerType('torch.quantization.stubs.QuantStub', class {});
        this.registerType('torch.utils._pytree.LeafSpec', class {});
        this.registerType('torch.utils._pytree.TreeSpec', class {});
        this.registerFunction('torch.utils.data._utils.collate.default_collate');
        this.registerType('torch.utils.data.dataloader._MultiProcessingDataLoaderIter', class {});
        this.registerType('torch.utils.data.dataloader.DataLoader', class {});
        this.registerType('torch.utils.data.dataset.Subset', class {});
        this.registerType('torch.utils.data.dataset.ConcatDataset', class {});
        this.registerType('torch.utils.data.dataset.TensorDataset', class {});
        this.registerType('torch.utils.data.sampler.BatchSampler', class {});
        this.registerType('torch.utils.data.sampler.RandomSampler', class {});
        this.registerType('torch.utils.data.sampler.SequentialSampler', class {});
        this.registerFunction('torchvision.datasets.folder.default_loader');
        this.registerType('torchvision.datasets.folder.ImageFolder', class {});
        this.registerType('torchvision.datasets.mnist.FashionMNIST', class {});
        this.registerType('torchvision.datasets.mnist.MNIST', class {});
        this.registerType('torchvision.datasets.video_utils.VideoClips', class {});
        this.registerType('torchvision.datasets.vision.StandardTransform', class {});
        this.registerType('torchvision.ops.deform_conv.DeformConv2d', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.FeaturePyramidNetwork', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.LastLevelMaxPool', class {});
        this.registerType('torchvision.ops.feature_pyramid_network.LastLevelP6P7', class {});
        this.registerType('torchvision.ops.misc.Conv2dNormActivation', class {});
        this.registerType('torchvision.ops.misc.ConvNormActivation', class {});
        this.registerType('torchvision.ops.misc.MLP', class extends torch.nn.modules.container.Sequential {});
        this.registerType('torchvision.ops.misc.ConvTranspose2d', class {});
        this.registerType('torchvision.ops.misc.FrozenBatchNorm2d', class {});
        this.registerType('torchvision.ops.misc.Permute', class {});
        this.registerType('torchvision.ops.misc.SqueezeExcitation', class {});
        this.registerType('torchvision.ops.poolers.LevelMapper', class {});
        this.registerType('torchvision.ops.poolers.MultiScaleRoIAlign', class {});
        this.registerType('torchvision.ops.stochastic_depth.StochasticDepth', class {});
        this.registerType('torchvision.models.alexnet.AlexNet', class {});
        this.registerType('torchvision.models.convnext.ConvNeXt', class {});
        this.registerType('torchvision.models.convnext.CNBlock', class {});
        this.registerType('torchvision.models.convnext.LayerNorm2d', class {});
        this.registerType('torchvision.models.densenet.DenseNet', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.models.densenet._DenseBlock', class extends torch.nn.modules.container.ModuleDict {});
        this.registerType('torchvision.models.densenet._DenseLayer', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.models.densenet._Transition', class extends torch.nn.modules.container.Sequential {});
        this.registerType('torchvision.models.detection._utils.BalancedPositiveNegativeSampler', class {});
        this.registerType('torchvision.models.detection._utils.BoxCoder', class {});
        this.registerType('torchvision.models.detection._utils.Matcher', class {});
        this.registerType('torchvision.models.detection._utils.SSDMatcher', class {});
        this.registerType('torchvision.models.detection.anchor_utils.AnchorGenerator', class {});
        this.registerType('torchvision.models.detection.anchor_utils.DefaultBoxGenerator', class {});
        this.registerType('torchvision.models.detection.backbone_utils.BackboneWithFPN', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.FasterRCNN', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.FastRCNNConvFCHead', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.FastRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.faster_rcnn.TwoMLPHead', class {});
        this.registerType('torchvision.models.detection.fcos.FCOS', class {});
        this.registerType('torchvision.models.detection.fcos.FCOSHead', class {});
        this.registerType('torchvision.models.detection.fcos.FCOSClassificationHead', class {});
        this.registerType('torchvision.models.detection.fcos.FCOSRegressionHead', class {});
        this.registerType('torchvision.models.detection._utils.BoxLinearCoder', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNN', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNNHeads', class {});
        this.registerType('torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNN', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNNHeads', class {});
        this.registerType('torchvision.models.detection.mask_rcnn.MaskRCNNPredictor', class {});
        this.registerType('torchvision.models.detection.retinanet.RetinaNet', class {});
        this.registerType('torchvision.models.detection.retinanet.RetinaNetClassificationHead', class {});
        this.registerType('torchvision.models.detection.retinanet.RetinaNetHead', class {});
        this.registerType('torchvision.models.detection.retinanet.RetinaNetRegressionHead', class {});
        this.registerType('torchvision.models.detection.roi_heads.RoIHeads', class {});
        this.registerType('torchvision.models.detection.rpn.AnchorGenerator', class {});
        this.registerType('torchvision.models.detection.rpn.RegionProposalNetwork', class {});
        this.registerType('torchvision.models.detection.rpn.RPNHead', class {});
        this.registerType('torchvision.models.detection.ssd.SSD', class {});
        this.registerType('torchvision.models.detection.ssd.SSDClassificationHead', class {});
        this.registerType('torchvision.models.detection.ssd.SSDHead', class {});
        this.registerType('torchvision.models.detection.ssd.SSDFeatureExtractorVGG', class {});
        this.registerType('torchvision.models.detection.ssd.SSDRegressionHead', class {});
        this.registerType('torchvision.models.detection.ssdlite.SSDLiteClassificationHead', class {});
        this.registerType('torchvision.models.detection.ssdlite.SSDLiteFeatureExtractorMobileNet', class {});
        this.registerType('torchvision.models.detection.ssdlite.SSDLiteHead', class {});
        this.registerType('torchvision.models.detection.ssdlite.SSDLiteRegressionHead', class {});
        this.registerType('torchvision.models.detection.transform.GeneralizedRCNNTransform', class {});
        this.registerType('torchvision.models.efficientnet.EfficientNet', class {});
        this.registerType('torchvision.models.efficientnet.FusedMBConv', class {});
        this.registerType('torchvision.models.efficientnet.MBConv', class {});
        this.registerType('torchvision.models.googlenet.BasicConv2d', class {});
        this.registerType('torchvision.models.googlenet.GoogLeNet', class {});
        this.registerType('torchvision.models.googlenet.Inception', class {});
        this.registerType('torchvision.models.googlenet.InceptionAux', class {});
        this.registerType('torchvision.models.inception.BasicConv2d', class {});
        this.registerType('torchvision.models.inception.Inception3', class {});
        this.registerType('torchvision.models.inception.InceptionAux', class {});
        this.registerType('torchvision.models.inception.InceptionA', class {});
        this.registerType('torchvision.models.inception.InceptionB', class {});
        this.registerType('torchvision.models.inception.InceptionC', class {});
        this.registerType('torchvision.models.inception.InceptionD', class {});
        this.registerType('torchvision.models.inception.InceptionE', class {});
        this.registerType('torchvision.models.mnasnet._InvertedResidual', class {});
        this.registerType('torchvision.models.mnasnet.MNASNet', class {});
        this.registerType('torchvision.models.maxvit.MaxVit', class {});
        this.registerType('torchvision.models.maxvit.MaxVitBlock', class {});
        this.registerType('torchvision.models.maxvit.MaxVitLayer', class {});
        this.registerType('torchvision.models.maxvit.MBConv', class {});
        this.registerType('torchvision.models.maxvit.PartitionAttentionLayer', class {});
        this.registerType('torchvision.models.maxvit.RelativePositionalMultiHeadAttention', class {});
        this.registerType('torchvision.models.maxvit.SwapAxes', class {});
        this.registerType('torchvision.models.maxvit.WindowDepartition', class {});
        this.registerType('torchvision.models.mobilenet.ConvBNReLU', class {});
        this.registerType('torchvision.models.mobilenet.MobileNetV2', class {});
        this.registerType('torchvision.models.mobilenet.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv2.ConvBNActivation', class {});
        this.registerType('torchvision.models.mobilenetv2.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv2.MobileNetV2', class {});
        this.registerType('torchvision.models.mobilenetv3.InvertedResidual', class {});
        this.registerType('torchvision.models.mobilenetv3.MobileNetV3', class {});
        this.registerType('torchvision.models.mobilenetv3.SqueezeExcitation', class {});
        this.registerType('torchvision.models.regnet.AnyStage', class extends torch.nn.modules.container.Sequential {});
        this.registerType('torchvision.models.regnet.BottleneckTransform', class {});
        this.registerType('torchvision.models.regnet.ResBottleneckBlock', class {});
        this.registerType('torchvision.models.regnet.RegNet', class {});
        this.registerType('torchvision.models.regnet.SimpleStemIN', class {});
        this.registerType('torchvision.models.resnet.Bottleneck', class {});
        this.registerType('torchvision.models.resnet.BasicBlock', class {});
        this.registerType('torchvision.models.quantization.mobilenet.QuantizableInvertedResidual', class {});
        this.registerType('torchvision.models.quantization.mobilenet.QuantizableMobileNetV2', class {});
        this.registerType('torchvision.models.quantization.mobilenetv2.QuantizableInvertedResidual', class {});
        this.registerType('torchvision.models.quantization.mobilenetv2.QuantizableMobileNetV2', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableBasicBlock', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableBottleneck', class {});
        this.registerType('torchvision.models.quantization.resnet.QuantizableResNet', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPP', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPPConv', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.ASPPPooling', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.DeepLabHead', class {});
        this.registerType('torchvision.models.segmentation.deeplabv3.DeepLabV3', class {});
        this.registerType('torchvision.models.segmentation.fcn.FCN', class {});
        this.registerType('torchvision.models.segmentation.fcn.FCNHead', class {});
        this.registerType('torchvision.models.segmentation.lraspp.LRASPPHead', class {});
        this.registerType('torchvision.models.shufflenetv2.ShuffleNetV2', class {});
        this.registerType('torchvision.models.shufflenetv2.InvertedResidual', class {});
        this.registerType('torchvision.models.squeezenet.Fire', class {});
        this.registerType('torchvision.models.squeezenet.SqueezeNet', class {});
        this.registerType('torchvision.models.resnet.ResNet', class {});
        this.registerType('torchvision.models.vgg.VGG', class {});
        this.registerType('torchvision.models.video.resnet.BasicBlock', class {});
        this.registerType('torchvision.models.video.resnet.BasicStem', class {});
        this.registerType('torchvision.models.video.resnet.Conv2Plus1D', class {});
        this.registerType('torchvision.models.video.resnet.Conv3DNoTemporal', class {});
        this.registerType('torchvision.models.video.resnet.Conv3DSimple', class {});
        this.registerType('torchvision.models.video.resnet.R2Plus1dStem', class {});
        this.registerType('torchvision.models.video.resnet.VideoResNet', class {});
        this.registerType('torchvision.models.vision_transformer.Encoder', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.models.vision_transformer.EncoderBlock', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.models.vision_transformer.MLPBlock', class extends torchvision.ops.misc.MLP {});
        this.registerType('torchvision.models.vision_transformer.VisionTransformer', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.models._utils.IntermediateLayerGetter', class {});
        this.registerType('torchvision.transforms._presets.ImageClassification', class {});
        this.registerType('torchvision.transforms.autoaugment.AutoAugment', class {});
        this.registerType('torchvision.transforms.functional.InterpolationMode', class {});
        this.registerType('torchvision.transforms.transforms.ColorJitter', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Compose', class {});
        this.registerType('torchvision.transforms.transforms.CenterCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Grayscale', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Lambda', class {});
        this.registerType('torchvision.transforms.transforms.Normalize', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomAffine', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomApply', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomHorizontalFlip', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomVerticalFlip', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomResizedCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomRotation', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Resize', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Scale', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.ToPILImage', class {});
        this.registerType('torchvision.transforms.transforms.ToTensor', class {});
        this.registerType('torchvision.transforms.v2._container.Compose', class {});
        this.registerType('torchvision.transforms.v2._misc.Normalize', class {});
        this.registerType('torchvision.transforms.v2._geometry.CenterCrop', class {});
        this.registerType('torchvision.transforms.v2._geometry.Resize', class {});
        this.registerFunction('torchvision.models.resnet.resnet18', () => {});
        this.registerFunction('torchvision.models.resnet.resnet34', () => {});
        this.registerFunction('torchvision.models.resnet.resnet50', () => {});
        this.registerFunction('torchvision.models.resnet.resnet101', () => {});
        this.registerFunction('torchvision.models.resnet.resnet152', () => {});
        this.registerFunction('torchvision.models.vision_transformer.vit_h_14', () => {});
        this.registerFunction('torchvision.ops.boxes.box_iou');
        this.registerFunction('builtins.annotate', (type, value) => {
            if (type === self._builtins.int) {
                return Number.isInteger(value) ? value : NaN;
            }
            if (type === self._builtins.float) {
                return typeof value === 'number' ? value : NaN;
            }
            if (type === self._builtins.number) {
                // if (pytorch.Utility.isTensor(value)) {
                //    value.resize_([]);
                // }
            }
            return value;
        });
        this.registerFunction('builtins.unchecked_cast', (type, value) => {
            return value;
        });
        this.registerFunction('builtins.uninitialized', (/* type */) => {
            return undefined;
        });
        this.registerFunction('ops.prim.data', (tensor) => {
            return tensor;
        });
        this.registerFunction('ops.prim.device', (tensor) => {
            return tensor.device;
        });
        this.registerFunction('ops.prim.dtype', (tensor) => {
            return tensor.dtype.scalar_type();
        });
        this.registerFunction('ops.prim.is_quantized', (tensor) => {
            return tensor.is_quantized;
        });
        this.registerFunction('ops.prim.is_cuda', (/* tensor */) => {
            return false;
        });
        this.registerFunction('ops.prim.is_nested', (tensor) => {
            return tensor.is_nested;
        });
        this.registerFunction('ops.prim.is_sparse', (tensor) => {
            return tensor.is_sparse;
        });
        this.registerFunction('ops.prim.unchecked_unwrap_optional', (value) => {
            return value;
        });
        this.registerFunction('ops.prim.NumToTensor', (value) => {
            const tensor = self.invoke('torch.Tensor', []);
            tensor.value = value;
            return tensor;
        });
        this.registerFunction('ops.prim.min', (...args) => {
            if (Array.isArray(args[0])) {
                return Math.min.apply(null, args[0]);
            }
            return Math.min.apply(null, args);
        });
        this.registerFunction('ops.prim.max', (...args) => {
            if (Array.isArray(args[0])) {
                return Math.max.apply(null, args[0]);
            }
            return Math.max.apply(null, args);
        });
        this.registerFunction('ops.prim.shape', (tensor) => {
            return tensor && tensor.size ? tensor.size() : undefined;
        });
        this.registerFunction('ops.quantized.conv_prepack', (weight, bias, stride, padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.conv1d_prepack', (weight, bias, stride, padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.conv2d_prepack', (weight, bias, stride, padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.conv3d_prepack', (weight, bias, stride, padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv3dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.conv_transpose1d_prepack', (weight, bias, stride, padding, output_padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.output_padding = output_padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.conv_transpose2d_prepack', (weight, bias, stride, padding, output_padding, dilation, groups) => {
            const params = self.invoke('__torch__.torch.classes.quantized.Conv2dPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            params.stride = stride;
            params.padding = padding;
            params.output_padding = output_padding;
            params.dilation = dilation;
            params.groups = groups;
            return params;
        });
        this.registerFunction('ops.quantized.linear_prepack', (weight, bias) => {
            const params = self.invoke('__torch__.torch.classes.quantized.LinearPackedParamsBase', []);
            params.weight = weight;
            params.bias = bias;
            return params;
        });
        this.registerFunction('ops.prim.RaiseException', (message) => {
            throw new python.Error(message);
        });
        this.registerFunction('builtins.range', (start, stop, step) => {
            if (stop === undefined && step === undefined) {
                if (Number.isInteger(start)) {
                    return Array(start).keys();
                }
                if (isNaN(start)) {
                    return [];
                }
            }
            throw new python.Error(`Unsupported range(${JSON.stringify(start)}, ${JSON.stringify(stop)}, ${JSON.stringify(step)})`);
        });
        this.registerFunction('torch._C._nn.gelu');
        this.registerFunction('torch._utils._rebuild_sparse_tensor', (layout, data) => {
            if (layout === torch.sparse_coo) {
                return self.invoke('torch._sparse_coo_tensor_unsafe', data);
            }
            throw new python.Error(`Unsupported sparse tensor layout '${layout ? layout.__str__() : ''}'.`);
        });
        this.registerFunction('torch._utils._rebuild_wrapper_subclass');
        this.registerFunction('torch.from_numpy', (obj) => {
            const dtypes = new Map([
                ['<f2', torch.float16],
                ['<f4', torch.float32],
                ['<f8', torch.float64],
                ['<i2', torch.int16],
                ['<i4', torch.int32],
                ['<i8', torch.int64],
            ]);
            if (!dtypes.has(obj.dtype.str)) {
                throw new python.Error(`Unsupported numpy.ndarray type '${obj.dtype.str}'.`);
            }
            const dtype = dtypes.get(obj.dtype.str);
            const strides = obj.strides.map((stride) => stride / obj.itemsize);
            const storage = execution.invoke('torch.storage._TypedStorage', [obj.size, dtype]);
            storage._set_cdata(obj.data);
            const tensor = execution.invoke('torch.Tensor', []);
            tensor.__setstate__([storage, 0, obj.shape, strides]);
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_device_tensor_from_numpy', (data, dtype, device, requires_grad) => {
            const tensor = execution.invoke('torch.from_numpy', [data]);
            // tensor = tensor.to(dtype, device)
            tensor.requires_grad = requires_grad;
            return tensor;
        });
        this.registerFunction('torch._sparse_coo_tensor_unsafe', (indices, values, size) => {
            const tensor = self.invoke('torch.Tensor', []);
            tensor._layout = torch.sparse_coo;
            tensor._indices = indices;
            tensor._values = values;
            tensor._shape = size;
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_tensor', (storage, storage_offset, size, stride) => {
            if (Array.isArray(storage) && storage.length === 5 && storage[0] === 'storage') {
                const [, storage_type, , ,size] = storage;
                storage = new storage_type(size);
            }
            const name = `${storage.__class__.__module__}.${storage.__class__.__name__.replace('Storage', 'Tensor')}`;
            const tensor = self.invoke(name, []);
            tensor.__setstate__([storage, storage_offset, size, stride]);
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_tensor_v2', (storage, storage_offset, size, stride, requires_grad, backward_hooks) => {
            const tensor = execution.invoke('torch._utils._rebuild_tensor', [storage, storage_offset, size, stride]);
            tensor.requires_grad = requires_grad;
            tensor.backward_hooks = backward_hooks;
            return tensor;
        });
        this.registerFunction('torch._utils._rebuild_parameter', (data, requires_grad, backward_hooks) => {
            const param = self.invoke('torch.nn.parameter.Parameter', [data, requires_grad]);
            param.backward_hooks = backward_hooks;
            return param;
        });
        this.registerFunction('torch._utils._rebuild_parameter_v2', (data, requires_grad, backward_hooks, state) => {
            const param = self.invoke('torch.nn.parameter.Parameter', [data, requires_grad]);
            param.backward_hooks = backward_hooks;
            execution.invoke('torch._utils._set_obj_state', [param, state]);
            return param;
        });
        this.registerFunction('torch._utils._rebuild_parameter_with_state', (data, requires_grad, backward_hooks, state) => {
            const _set_obj_state = (obj, state) => {
                const [dict_state, slots_state] = Array.isArray(state) ? state : [state, null];
                if (dict_state) {
                    for (const [k, v] of Object.entries(dict_state)) {
                        self.invoke('builtins.setattr', [obj, k, v]);
                    }
                }
                if (slots_state) {
                    for (const [k, v] of Object.entries(slots_state)) {
                        self.invoke('builtins.setattr', [obj, k, v]);
                    }
                }
            };
            const param = self.invoke('torch.nn.parameter.Parameter', [data, requires_grad]);
            param._backward_hooks = backward_hooks;
            _set_obj_state(param, state);
            return param;
        });
        this.registerFunction('torch._utils._rebuild_qtensor', (storage, storage_offset, size, stride, quantizer_params, requires_grad, backward_hooks) => {
            const tensor = execution.invoke('torch._utils._rebuild_tensor_v2', [storage, storage_offset, size, stride, requires_grad, backward_hooks]);
            tensor.quantizer_params = quantizer_params;
            return tensor;
        });
        this.registerFunction('torch._utils._set_obj_state', (obj, state) => {
            let dict_state = state;
            let slots_state = null;
            if (state instanceof self.builtins.tuple) {
                if (state.length !== 2) {
                    throw new python.Error(`Invalid serialized state: '${state}'.`);
                }
                [dict_state, slots_state] = state;
            }
            if (dict_state) {
                for (const [name, value] of Object.entries(dict_state)) {
                    execution.invoke('builtins.setattr', [obj, name, value]);
                }
            }
            if (slots_state) {
                for (const [name, value] of Object.entries(slots_state)) {
                    execution.invoke('builtins.setattr', [obj, name, value]);
                }
            }
            return obj;
        });
        this.registerFunction('torch._set_item', (dict, key, value) => {
            dict[key] = value;
        });
        this.registerFunction('torch._tensor._rebuild_from_type_v2', (func, new_type, args, state) => {
            let ret = func(...args);
            if (ret.__class__ !== new_type) {
                // ret = ret.as_subclass(new_type);
            }
            const setstate = execution.invoke('builtins.getattr', [ret.__class__, '__setstate__', torch.Tensor.__setstate__]);
            if (setstate === torch.Tensor.__setstate__) {
                ret = execution.invoke('torch._utils._set_obj_state', [ret, state]);
            } else {
                ret.__setstate__(state);
            }
            return ret;
        });
        this.registerFunction('torch.__and__', (left, right) => {
            return left && right;
        });
        this.registerFunction('torch.__contains__', (dict, key) => {
            return dict[key] !== undefined;
        });
        this.registerFunction('torch.__derive_index', (index, start, step) => {
            return start + index * step;
        });
        this.registerFunction('torch.__is__', (left, right) => {
            if (left === null && right === null) {
                return true;
            }
            if ((left !== null && right === null) || (left === null && right !== null)) {
                return false;
            }
            throw new python.Error("Unsupported 'torch.__is__' expression type.");
        });
        this.registerFunction('torch.__isnot__', (left, right) => {
            if (left === null && right === null) {
                return false;
            }
            if ((left !== null && right === null) || (left === null && right !== null)) {
                return true;
            }
            throw new python.Error("Unsupported 'torch.__isnot__' expression type.");
        });
        this.registerFunction('torch.__not__', (value) => {
            if (typeof value === 'boolean') {
                return !value;
            }
            throw new python.Error("Unsupported 'torch.__not__' expression type.");
        });
        this.registerFunction('torch.__range_length', (lo, hi, step) => {
            if (step === 0) {
                throw new python.Error('range() arg 3 must not be zero');
            }
            if (step > 0 && lo < hi) {
                return 1 + (hi - 1 - lo) / step;
            } else if (step < 0 && lo > hi) {
                return 1 + (lo - 1 - hi) / (0 - step);
            }
            return 0;
        });
        this.registerFunction('torch._nested_tensor_from_mask_left_aligned');
        this.registerFunction('torch._unwrap_optional', (value) => {
            return value;
        });
        this.registerFunction('torch.empty_strided', (/* size, stride, dtype, layout, device, pin_memory, requires_grad */) => {
            return null;
        });
        this.registerFunction('torch.add', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left * right;
            }
            if (Array.isArray(left) && Array.isArray(right)) {
                return left.concat(right);
            }
            if (typeof left === 'string' && typeof right === 'string') {
                return left + right;
            }
            throw new python.Error('Unsupported torch.add expression type.');
        });
        this.registerFunction('torch.append', (list, value) => {
            list.push(value);
            return value;
        });
        this.registerFunction('torch.clear', (value) => {
            if (Object(value) === value) {
                for (const key of Object.keys(value)) {
                    delete value[key];
                }
            }
        });
        this.registerFunction('torch.cosine_similarity');
        this.registerFunction('torch.extend', (list, value) => {
            list.push(...value);
        });
        this.registerFunction('torch.insert', (list, index, value) => {
            list.splice(index, 0, value);
            return value;
        });
        this.registerFunction('torch.replace', (value) => {
            return value;
        });
        this.registerFunction('torch.dict', (args) => {
            const obj = {};
            if (args) {
                if (Array.isArray(args)) {
                    for (const [key, value] of args) {
                        obj[key] = value;
                    }
                } else {
                    throw new python.Error("'torch.dict' arguments not supported.");
                }
            }
            return obj;
        });
        this.registerFunction('torch.dim', (tensor) => {
            if (tensor && tensor.size) {
                const size = tensor.size();
                if (size) {
                    return size.length;
                }
            }
            return NaN;
        });
        this.registerFunction('torch.numel', (tensor) => {
            if (tensor && tensor.size) {
                const size = tensor.size();
                if (size) {
                    return size.reduce((a, b) => a * b, 1);
                }
            }
            return NaN;
        });
        this.registerFunction('torch.eq', (left, right) => {
            if (typeof left === 'string' && typeof right === 'string') {
                return left === right;
            }
            if (typeof left === 'number' && typeof right === 'number') {
                if (isNaN(left) && isNaN(right)) {
                    return true;
                }
                return left === right;
            }
            if (left === undefined || right === undefined) {
                return true;
            }
            if (Array.isArray(left) && Array.isArray(right)) {
                return left.length === right.length && left.every((item, index) => item === right[index]);
            }
            throw new python.Error("Unsupported 'torch.eq' expression type.");
        });
        this.registerFunction('torch.floor', (value) => {
            return Math.floor(value);
        });
        this.registerFunction('torch.ceil', (value) => {
            return Math.ceil(value);
        });
        this.registerFunction('torch.floordiv', (left, right) => {
            return Math.floor(left / right);
        });
        this.registerFunction('torch.format', (...args) => {
            const list = args.shift().split(/({}D?)/);
            return list.map((text) => {
                if (text === '{}' || text === '{}D') {
                    const arg = args.shift();
                    if (Array.isArray(arg)) {
                        return `[${arg.map((item) => item.toString()).join(', ')}]`;
                    }
                    return arg ? arg.toString() : '?';
                }
                return text;
            }).join('');
        });
        this.registerFunction('torch.gt', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                if (!isNaN(left) && !isNaN(right)) {
                    return left > right;
                }
            }
            if (isNaN(left) && !isNaN(right)) {
                return true;
            }
            throw new python.Error("Unsupported 'torch.gt' expression type.");
        });
        this.registerFunction('torch.ge', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                if (!isNaN(left) && !isNaN(right)) {
                    return left > right;
                }
            }
            if (isNaN(left) && !isNaN(right)) {
                return true;
            }
            throw new python.Error("Unsupported 'torch.ge' expression type.");
        });
        this.registerFunction('torch.is_floating_point', (tensor) => {
            const type = tensor.dtype.scalar_type();
            return (type === 5 || type === 6 || type === 7);
        });
        this.registerFunction('torch.is_grad_enabled', () => {
            return false;
        });
        this.registerFunction('torch.isfinite');
        this.registerFunction('torch.set_grad_enabled', (/* value */) => {
        });
        this.registerFunction('torch.serialization._get_layout', (name) => {
            const value = name.startsWith('torch.') ? torch[name.split('.')[1]] : null;
            return value instanceof torch.layout ? value : null;
        });
        this.registerFunction('torch.storage._load_from_bytes', (b) => {
            return torch.load(b);
        });
        this.registerFunction('torch.jit._pickle.build_boollist', (data) => {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_doublelist', (data) => {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_intlist', (data) => {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_tensorlist', (data) => {
            return data;
        });
        this.registerFunction('torch.jit._pickle.build_tensor_from_id', (data) => {
            return self.builtins.CONSTANTS[`c${data}`];
        });
        this.registerFunction('torch.jit._pickle.restore_type_tag', (value /*, type_str */) => {
            return value;
        });
        this.registerFunction('torch.keys', (dict) => {
            return Object.keys(dict);
        });
        this.registerFunction('torch.len', (value) => {
            if (Array.isArray(value)) {
                return value.length;
            }
            if (value && value.shape && value.__len__) {
                return value.__len__();
            }
            return NaN;
        });
        this.registerFunction('torch.le', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                if (isNaN(left) || isNaN(right)) {
                    return false;
                }
                return left <= right;
            }
            if (left === undefined || right === undefined) {
                return true;
            }
            throw new python.Error("Unsupported 'torch.le' expression type.");
        });
        this.registerFunction('torch.list', (args) => {
            return args;
        });
        this.registerFunction('torch.list_with_default', (size /*, defaults */) => {
            return size;
        });
        this.registerType('torch.PyTorchFileReader', class {
            constructor(entries) {
                let prefix = 0;
                const paths = Array.from(entries.keys()).map((path) => path.replace(/\\/g, '/').split('/').reverse());
                for (let set = new Set(); set && paths.length > 0;) {
                    set = new Set(paths.map((path) => path.length > 1 ? path.pop() : null));
                    set = set.size > 1 || set.keys().next().value === null ? null : set;
                    prefix += set ? set.keys().next().value.length + 1 : 0;
                }
                this._records = new Map(Array.from(entries).map(([name, value]) => [name.substring(prefix), value]));
                this._version = '0';
                const stream = this.get_record('.data/version') || this.get_record('version') || null;
                if (stream) {
                    const decoder = new TextDecoder('utf-8');
                    const buffer = stream.peek();
                    const text = decoder.decode(buffer);
                    this._version = text.split('\n').shift().trim();
                }
            }
            has_record(name) {
                return this._records.has(name);
            }
            get_record(name) {
                return this._records.get(name);
            }
            get_all_records() {
                return Array.from(this._records.keys());
            }
            version() {
                return this._version;
            }
        });
        this.registerFunction('torch.load', (f) => {
            const legacy_load = (entries) => {
                const deserialized_objects = {};
                if (entries.has('storages')) {
                    const data = entries.get('storages');
                    const unpickler = execution.invoke('pickle.Unpickler', [data]);
                    const num_storages = unpickler.load();
                    for (let i = 0; i < num_storages; i++) {
                        const args = unpickler.load();
                        const [key, , storage_type] = args;
                        const obj = storage_type._new_with_file(unpickler);
                        deserialized_objects[key] = obj;
                    }
                    /*
                    let storage_views = unpickler.load();
                    for target_cdata, root_cdata, offset, size in storage_views:
                        root = deserialized_objects[root_cdata]
                        deserialized_objects[target_cdata] = root[offset:offset + size]
                    */
                }
                if (entries.has('tensors')) {
                    const data = entries.get('tensors');
                    const unpickler = execution.invoke('pickle.Unpickler', [data]);
                    const num_tensors = unpickler.load();
                    const int32 = (unpickler) => {
                        const buffer = unpickler.read(4);
                        return buffer[0] + (buffer[1] << 8) + (buffer[2] << 16) + (buffer[3] << 24);
                    };
                    const int64 = (unpickler) => {
                        const buffer = unpickler.read(8);
                        if (buffer[6] !== 0 && buffer[7] !== 0) {
                            throw new python.Error('Unsigned 64-bit value exceeds 32-bit range.');
                        }
                        return buffer[0] + (buffer[1] << 8) + (buffer[2] << 16) + (buffer[3] << 24) + (buffer[4] * 4294967296) + (buffer[5] * 1099511627776);
                    };
                    for (let i = 0; i < num_tensors; i++) {
                        const args = unpickler.load();
                        const [key, storage_id] = args;
                        const storage = deserialized_objects[storage_id];
                        const ndim = int32(unpickler);
                        unpickler.read(4);
                        const shape = Array.from(new Array(ndim)).map(() => int64(unpickler));
                        const stride = Array.from(new Array(ndim)).map(() => int64(unpickler));
                        const storage_offset = int64(unpickler);
                        const tensor = execution.invoke('torch._utils._rebuild_tensor', [storage, storage_offset, shape, stride]);
                        deserialized_objects[key] = tensor;
                    }
                }
                const data = entries.get('pickle');
                const unpickler = execution.invoke('pickle.Unpickler', [data]);
                unpickler.persistent_load = (saved_id) => deserialized_objects[saved_id];
                return unpickler.load();
            };
            const _legacy_load = () => {
                const unpickler = execution.invoke('pickle.Unpickler', [f]);
                unpickler.load(); // magic_number
                const protocol_version = unpickler.load();
                if (protocol_version !== 1001) {
                    throw new python.Error(`Unsupported protocol version '${protocol_version}'.`);
                }
                const sys_info = unpickler.load();
                if (sys_info.protocol_version !== 1001) {
                    throw new python.Error(`Unsupported protocol version '${sys_info.protocol_version}'.`);
                }
                if (sys_info.little_endian === false) {
                    throw new python.Error("Unsupported big-endian storage data.");
                }
                const module_source_map = new Map();
                const deserialized_objects = new Map();
                unpickler.persistent_load = (saved_id) => {
                    switch (saved_id[0]) {
                        case 'module': {
                            const [, module, ,source] = saved_id;
                            module_source_map.set(module, source);
                            return saved_id[1];
                        }
                        case 'storage': {
                            const [, storage_type, key, , size, view_metadata] = saved_id;
                            if (!deserialized_objects.has(key)) {
                                const obj = new storage_type(size);
                                deserialized_objects.set(key, obj);
                            }
                            if (view_metadata) {
                                const view_key = view_metadata.shift();
                                view_metadata.shift(); // view_offset
                                view_metadata.shift(); // view_size
                                if (!deserialized_objects.has(view_key)) {
                                    const view = null; // storage.slice(view_offset, view_offset + view_size);
                                    deserialized_objects.set(view_key, view);
                                }
                                return deserialized_objects.get(view_key);
                            }
                            return deserialized_objects.get(key);
                        }
                        default: {
                            throw new python.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
                        }
                    }
                };
                const obj = unpickler.load();
                const deserialized_storage_keys = unpickler.load();
                for (const deserialized_storage_key of deserialized_storage_keys) {
                    const storage = deserialized_objects.get(deserialized_storage_key);
                    storage._set_from_file(unpickler);
                }
                if (!obj) {
                    throw new python.Error('File format is not PyTorch.');
                }
                if (obj === 'None') {
                    throw new python.Error("File contains 'None' root object.");
                }
                return obj;
            };
            const _load = (entries) => {
                if (f.has('constant.pkl')) {
                    throw python.Error("TorchScript 'torch.load' not supported.");
                }
                const loaded_storages = new Map();
                const persistent_load = (saved_id) => {
                    switch (saved_id[0]) {
                        case 'storage': {
                            const [, storage_type, key, , numel] = saved_id;
                            if (!loaded_storages.has(key)) {
                                const storage = new storage_type(numel);
                                const name = `data/${key}`;
                                const stream = entries.get(name);
                                storage._set_cdata(stream);
                                loaded_storages.set(key, storage);
                            }
                            return loaded_storages.get(key);
                        }
                        default: {
                            throw new python.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
                        }
                    }
                };
                const data_file = entries.get('data.pkl');
                const unpickler = execution.invoke('pickle.Unpickler', [data_file]);
                unpickler.persistent_load = persistent_load;
                const result = unpickler.load();
                return result;
            };
            if (f instanceof Map) {
                const reader = new torch.PyTorchFileReader(f);
                const records = reader.get_all_records().map((name) => [name, reader.get_record(name)]);
                f = new Map(records);
                if (f.has('pickle')) {
                    return legacy_load(f);
                }
                if (f.has('data.pkl')) {
                    return _load(f);
                }
                throw new python.Error(`Unsupported 'torch.load' input '${JSON.stringify(Array.from(f.keys()))}'.`);
            }
            return _legacy_load(f);
        });
        this.registerFunction('torch.log10');
        this.registerFunction('torch.lt', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left < right;
            }
            throw new python.Error("Unsupported 'torch.lt' expression type.");
        });
        this.registerFunction('torch.mul', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left * right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            if (Array.isArray(left) && left.every((value) => typeof value === 'number') && typeof right === 'number') {
                return left.map((value) => value * right);
            }
            throw new python.Error("Unsupported 'torch.mul' expression type.");
        });
        this.registerFunction('torch.div', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left / right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new python.Error("Unsupported 'torch.div' expression type.");
        });
        this.registerFunction('torch.round', (value) => {
            if (typeof value === 'number') {
                return Math.round(value);
            }
            if (isNaN(value)) {
                return value;
            }
            throw new python.Error("Unsupported 'torch.round' expression type.");
        });
        this.registerFunction('torch.remainder', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left % right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new python.Error("Unsupported 'torch.remainder' expression type.");
        });
        this.registerFunction('torch.ne', (left, right) => {
            if (typeof left === 'boolean' && typeof right === 'boolean') {
                return left !== right;
            }
            if (typeof left === 'number' && typeof right === 'number') {
                if (isNaN(left) || isNaN(right)) {
                    return false;
                }
                return left !== right;
            }
            if (Array.isArray(left) && Array.isArray(right) && left.length === right.length) {
                return false;
            }
            if (typeof left === 'string' && typeof right === 'string') {
                return left !== right;
            }
            if (left === undefined || right === undefined) {
                return true;
            }
            throw new python.Error("Unsupported 'torch.ne' expression type.");
        });
        this.registerFunction('torch.neg', (value) => {
            if (typeof value === 'number') {
                return -value;
            }
            throw new python.Error("Unsupported 'torch.neg' expression type.");
        });
        this.registerFunction('torch.pow', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return Math.pow(left, right);
            }
            throw new python.Error("Unsupported 'torch.pow' expression type.");
        });
        this.registerFunction('torch.q_scale', (/* tensor */) => {
            return -1;
        });
        this.registerFunction('torch.t', (tensor) => {
            return tensor;
        });
        this.registerFunction('torch.size', (tensor, dim) => {
            if (tensor && tensor.size) {
                const size = tensor.size();
                if (Array.isArray(size)) {
                    if (dim === undefined) {
                        return size;
                    }
                    if (Number.isInteger(dim)) {
                        if (dim >= 0 && dim < size.length) {
                            return size[dim];
                        }
                        if (dim < 0 && -dim < size.length) {
                            return size[size.length + dim];
                        }
                    }
                    throw new python.Error(`Dimension out of range (expected to be in range of ${JSON.stringify(size)}, but got ${JSON.stringify(dim)}).`);
                }
            }
            if (Number.isInteger(dim)) {
                return NaN;
            }
            return [];
        });
        this.registerFunction('torch.sqrt', (x) => {
            return Math.sqrt(x);
        });
        this.registerFunction('torch.slice', (l, start, end, step) => {
            if (!Array.isArray(l)) {
                throw new python.Error('Slicing expected array');
            }
            step = step || 1;
            if (step !== 1) {
                throw new python.Error('Slicing only supports step=1');
            }
            start = Math.max(0, start >= 0 ? start : l.length + start);
            end = Math.min(l.length, end || Number.MAX_SAFE_INTEGER);
            return l.slice(start, end);
        });
        this.registerFunction('torch.sub', (left, right) => {
            if (typeof left === 'number' && typeof right === 'number') {
                return left - right;
            }
            throw new python.Error("Unsupported 'torch.sub' expression type.");
        });
        this.registerFunction('torch.functional.einsum');
        this.registerFunction('torch.functional.split');
        this.registerFunction('torch.nn.init.constant_');
        this.registerFunction('torch.nn.init.xavier_uniform_');
        this.registerFunction('torch.nn.functional.adaptive_avg_pool2d');
        this.registerFunction('torch.nn.functional.cross_entropy');
        this.registerFunction('torch.nn.functional.elu');
        this.registerFunction('torch.nn.functional.gelu');
        this.registerFunction('torch.nn.functional.hardsigmoid');
        this.registerFunction('torch.nn.functional.hardswish');
        this.registerFunction('torch.nn.functional.hardtanh');
        this.registerFunction('torch.nn.functional.interpolate');
        this.registerFunction('torch.nn.functional.leaky_relu');
        this.registerFunction('torch.nn.functional.linear');
        this.registerFunction('torch.nn.functional._max_pool2d');
        this.registerFunction('torch.nn.functional.max_pool2d_with_indices');
        this.registerFunction('torch.nn.functional.mse_loss');
        this.registerFunction('torch.nn.functional.pad');
        this.registerFunction('torch.nn.functional.relu');
        this.registerFunction('torch.nn.functional.relu6');
        this.registerFunction('torch.nn.functional.sigmoid');
        this.registerFunction('torch.nn.functional.silu');
        this.registerFunction('torch.nn.functional.softmax');
        this.registerFunction('torch.nn.functional.tanh');
        this.registerFunction('torch.values', (dict) => {
            return Object.values(dict);
        });
        this.registerFunction('torch.warn', () => {
        });
        this.registerType('torch._ops.OperatorBase', class {});
        this.registerType('torch._ops.HigherOrderOperator', class extends torch._ops.OperatorBase {});
        this.registerType('torch._ops.OpOverload', class extends torch._ops.OperatorBase {});
        this.registerType('torch.export.unflatten.UnflattenedModule', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.export.exported_program.ExportedProgram', class {});
        this.registerFunction('torch.export.unflatten');
        this.registerFunction('torch._export.exported_program._create_graph_module_for_export', (root, graph) => {
            return new torch.fx.graph_module.GraphModule(root, graph);
        });
        this.registerType('torch._export.serde.serialize.SerializedArtifact', class {
            constructor(exported_program, state_dict, constants) {
                this.exported_program = exported_program;
                this.state_dict = state_dict;
                this.constants = constants;
            }
        });
        this.registerType('torch.fx.experimental.symbolic_shapes.ShapeEnv', class {
            create_symintnode(/* sym, hint, source */) {
                return new torch.SymInt();
            }
        });
        this.registerFunction('torch.fx.graph_module._deserialize_graph_module', (/* forward, body */) => {
            return execution.invoke('torch.fx.graph_module.GraphModule', []);
        });
        this.registerFunction('torch.fx.graph_module._forward_from_src', (src, globals /*, co_fields */) => {
            globals = { ...globals };
            const context = new python.Execution.Context(globals, null);
            execution.exec(src, context);
            const forward_fn = globals.forward;
            delete globals.forward;
            return forward_fn;
        });
        this.registerFunction('torch.fx.graph_module.reduce_graph_module', (body, import_block) => {
            // https://github.com/pytorch/pytorch/blob/master/torch/fx/graph_module.py
            const fn_src = body._code || body.code;
            const forward = execution.invoke('torch.fx.graph_module._forward_from_src', [import_block + fn_src, {}]);
            return execution.invoke('torch.fx.graph_module._deserialize_graph_module', [forward, body]);
        });
        this.registerFunction('torch.fx.graph_module.reduce_package_graph_module', (importer, body, generated_module_name) => {
            const forward = importer.import_module(generated_module_name).forward;
            return execution.invoke('torch.fx.graph_module._deserialize_graph_module', [forward, body]);
        });
        this.registerType('torch.fx.graph.CodeGen', class {});
        this.registerType('torch.fx.graph._Namespace', class {
            constructor() {
                this._obj_to_name = new Map();
                this._unassociated_names = new Set();
                this._used_names = new Set();
                this._base_count = {};
            }
            create_name(candidate, obj) {
                if (obj && this._obj_to_name.has(obj)) {
                    return self._obj_to_name.get(obj);
                }
                candidate = candidate || '_unnamed';
                candidate = /^\d+$/.test(candidate) ? `_${candidate}` : candidate;
                candidate = candidate.replace(/[^0-9a-zA-Z_]+/, '_');
                const match = candidate.match(/(.*)_(\d+)$"/);
                let base = candidate;
                let num = null;
                if (match) {
                    [, base] = match;
                    num = parseInt(match[2], 10);
                }
                candidate = num ? `${base}_${num}` : base;
                if (!num) {
                    num = this._base_count[base] || 0;
                }
                while (this._used_names.has(candidate) || this._is_illegal_name(candidate, obj)) {
                    num += 1;
                    candidate = `${base}_${num}`;
                }
                this._used_names.add(candidate);
                this._base_count[base] = num;
                if (obj) {
                    this._obj_to_name[obj] = candidate;
                } else {
                    this._unassociated_names.add(candidate);
                }
                return candidate;
            }
            _is_illegal_name(/* name, obj */) {
                /*
                if name in keyword.kwlist:
                    return True
                if name in builtins.__dict__:
                    return obj is not builtins.__dict__[name]
                if name in _custom_builtins:
                    return obj is not _custom_builtins[name].obj
                */
                return false;
            }
            associate_name_with_obj() {

            }
        });
        this.registerType('torch.fx.node.Node', class {
            constructor(graph, name, op, target, args, kwargs, return_type) {
                this.graph = graph;
                this.name = name;
                this.op = op;
                this.target = target;
                this._input_nodes = new Map();
                this.users = new Map();
                this.type = return_type;
                this._prev = this;
                this._next = this;
                this._erased = false;
                this._repr_fn = null;
                this.meta = {};
            }
            prepend(x) {
                x._remove_from_list();
                const p = this._prev;
                p._next = x;
                x._prev = p;
                x._next = this;
                this._prev = x;
            }
            _remove_from_list() {
                const p = this._prev;
                const n = this._next;
                p._next = n;
                n._prev = p;
            }
        });
        torch.fx.Node = torch.fx.node.Node;
        this.registerType('torch.fx.graph.Graph', class {
            constructor() {
                this._root = new torch.fx.node.Node(self, '', 'root', '', [], {});
                this._used_names = new Map();
                this._len = 0;
                this._graph_namespace = new torch.fx.graph._Namespace();
                // this._owning_module = owning_module
                // this._tracer_cls = tracer_cls
                // this._tracer_extras = tracer_extras
                // this._codegen = CodeGen()
                // this._co_fields = {}
            }
            placeholder(name, type_expr /*, default_value */) {
                const args = []; // () if default_value is inspect.Signature.empty else (default_value,)
                return this.create_node('placeholder', name, args, type_expr);
            }
            create_node(op, target, args, kwargs, name, type_expr) {
                args = args || [];
                kwargs = kwargs || {};
                const candidate = name || this._target_to_str(target);
                name = this._graph_namespace.create_name(candidate, null);
                const n = new torch.fx.Node(this, name, op, target, args, kwargs, type_expr);
                this._graph_namespace.associate_name_with_obj(name, n);
                this._insert(n);
                this._len += 1;
                return n;
            }
            _insert(n) {
                this._root.prepend(n);
            }
            _target_to_str(target) {
                if (typeof target === 'string') {
                    if (target.startsWith('__') && target.endswith('__')) {
                        target = target.substring(2, target.length - 2);
                    }
                } else {
                    target = target.__name__;
                }
                return this._snake_case(target);
            }
            _snake_case(s) {
                const chars = [];
                let prev_lower = false;
                for (const c of s) {
                    const x = c.toLowerCase();
                    if (prev_lower && x !== c) {
                        chars.push('_');
                    } else {
                        prev_lower = true;
                    }
                    chars.push(x);
                }
                return chars.join('');
            }
        });
        torch.fx.Graph = torch.fx.graph.Graph;
        this.registerType('torch.fx.graph_module.GraphModule', class extends torch.nn.modules.module.Module {
            constructor(root, graph) {
                super();
                this.graph = graph;
            }
        });
        this.registerFunction('torch.fx._symbolic_trace.wrap', (fn_or_name) => {
            return fn_or_name;
        });
        this.registerType('torch.fx._symbolic_trace.Tracer', class {});
        this.registerFunction('torch._export.load', (f, expected_opset_version) => {
            const serialized_exported_program = f.get('serialized_exported_program.json');
            const serialized_state_dict = f.get('serialized_state_dict.pt');
            const serialized_constants = f.get('serialized_constants.pt');
            const artifact = new torch._export.serde.serialize.SerializedArtifact(serialized_exported_program, serialized_state_dict, serialized_constants);
            return torch._export.serde.serialize.deserialize(artifact, expected_opset_version);
        });
        this.registerFunction('torch._export.serde.serialize._dict_to_dataclass', (cls, data) => {
            if (data === null) {
                return data;
            }
            if (data.$type) {
                const res = {};
                res[data.$type] = data.$value;
                return res;
            }
            if (Array.isArray(data)) {
                for (let i = 0; i < data.length; i++) {
                    data[i] = torch._export.serde.serialize._dict_to_dataclass(null, data[i]);
                }
                return data;
            }
            if (data === Object(data)) {
                for (const key of Object.keys(data)) {
                    data[key] = torch._export.serde.serialize._dict_to_dataclass(null, data[key]);
                }
                return data;
            }
            return data;
        });
        this.registerFunction('torch._export.serde.serialize.deserialize', (artifact, expected_opset_version) => {
            artifact.exported_program = torch._export.serde.serialize._dict_to_dataclass(null, artifact.exported_program);
            return new torch._export.serde.serialize.ExportedProgramDeserializer(expected_opset_version).deserialize(artifact);
        });
        this.registerType('torch._export.serde.serialize.ExportedProgramDeserializer', class {
            constructor(expected_opset_version) {
                this.expected_opset_version = expected_opset_version;
            }
            deserialize(serialized_artifact) {
                const symbol_name_to_range = new Map(Object.entries(serialized_artifact.exported_program.range_constraints));
                /*
                symbol_name_to_range = {
                    k: symbolic_shapes.ValueRanges(_int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val))
                    for k, v in serialized_artifact.exported_program.range_constraints.items()
                }
                */
                const constants = serialized_artifact.constants ? torch.load(serialized_artifact.constants) : null;
                const tensor_constants = constants ? new Map(Object.entries(constants).filter(([, tensor]) => tensor instanceof torch.Tensor)) : null;
                const deserializer = new torch._export.serde.serialize.GraphModuleDeserializer();
                const res = deserializer.deserialize(serialized_artifact.exported_program.graph_module, symbol_name_to_range, constants);
                const range_constraints = null;
                /*
                range_constraints = self.deserialize_range_constraints(
                    symbol_name_to_range, res.names_to_symbols,
                )
                model_opset_version: Optional[Dict[str, int]] = serialized_artifact.exported_program.opset_version
                self._validate_model_opset_version(model_opset_version)
                upgrader = GraphModuleOpUpgrader(self.expected_opset_version, model_opset_version)
                */
                const state_dict = serialized_artifact.state_dict ? torch.load(serialized_artifact.state_dict) : null;
                const exported_program = new torch.export.exported_program.ExportedProgram(
                    res.graph_module, res.graph_module.graph, res.signature,
                    state_dict, range_constraints, res.module_call_graph, null,
                    null, // verifier=load_verifier(serialized_artifact.exported_program.dialect),
                    tensor_constants);
                return exported_program;
                // return upgrader.upgrade(exported_program)
            }
        });
        this.registerType('torch._export.serde.serialize.GraphModuleDeserializer', class {
            constructor() {
                this.serialized_name_to_node = new Map();
                this.serialized_name_to_meta = new Map();
                this.graph = new torch.fx.Graph();
                this.module = new torch.nn.Module();
                this._SYM_INT_OPS = new Set([
                    operator.mul, operator.add, operator.sub, operator.floordiv, operator.mod,
                    torch.sym_sqrt, torch.sym_int, torch.sym_ite, torch.sym_max, torch.sym_min, torch.sym_sqrt
                ]);
                this._SYM_BOOL_OPS = new Set([
                    operator.eq, operator.ne, operator.le, operator.ge, operator.lt, operator.gt,
                    torch.sym_not
                ]);
            }
            deserialize_graph_output(/* output */) {
                /*
                if (output.type === 'as_tensor') {
                    return self.serialized_name_to_node[output.as_tensor.name]
                }
                else if (output.type === 'as_sym_int') {
                    return self.serialized_name_to_node[output.as_sym_int.as_name]
                }
                elif output.type === 'as_sym_bool':
                    return self.serialized_name_to_node[output.as_sym_bool.as_name]
                else:
                    raise SerializeError(f'Unable to deserialize output node {output}')
                */
            }
            deserialize_graph(serialized_graph) {
                if (serialized_graph.constants) {
                    this.constants = new Map(Object.entries(serialized_graph.constants).map(([k, v]) => [k, torch.load(v)]));
                }
                for (const [name, tensor_value] of Object.entries(serialized_graph.tensor_values)) {
                    const meta_val = this.deserialize_tensor_meta(tensor_value.meta || tensor_value, this.fake_tensor_mode);
                    this.serialized_name_to_meta.set(name, meta_val);
                }
                for (const [name, sym_int_value] of Object.entries(serialized_graph.sym_int_values)) {
                    this.serialized_name_to_meta.set(name, this.deserialize_sym_int(sym_int_value));
                }
                for (const [name, sym_bool_value] in Object.entries(serialized_graph.sym_bool_values)) {
                    this.serialized_name_to_meta.set(name, this.deserialize_sym_bool(sym_bool_value));
                }
                for (const input of serialized_graph.inputs) {
                    const placeholder_node = this.graph.placeholder(input.as_tensor.name);
                    this.sync_fx_node(input.as_tensor.name, placeholder_node);
                }
                for (const serialized_node of serialized_graph.nodes) {
                    const target = this.deserialize_operator(serialized_node.target);
                    this.deserialize_node(serialized_node, target);
                }
                const outputs = [];
                for (const output of serialized_graph.outputs) {
                    outputs.push(this.deserialize_graph_output(output));
                }
            }
            deserialize_operator(serialized_target) {
                let target = null;
                if (serialized_target.startsWith('_operator')) {
                    target = operator;
                } else if (serialized_target.startsWith('torch')) {
                    target = torch;
                } else {
                    return serialized_target;
                }
                const serialized_target_names = serialized_target.split('.').reverse();
                serialized_target_names.pop();
                for (const name of serialized_target_names) {
                    target = target[name];
                    if (!target) {
                        return serialized_target;
                    }
                }
                return target;
            }
            deserialize_node(serialized_node, target) {
                let fx_node = null;
                if (this._SYM_BOOL_OPS.has(target) || this._SYM_INT_OPS.has(target)) {
                    /*
                    const name = serialized_node.outputs[0].value.as_name;
                    const args = self.deserialize_sym_op_inputs(serialized_node.inputs);
                    fx_node = self.graph.create_node("call_function", target, args, {}, name);
                    self.deserialize_sym_op_outputs(serialized_node, fx_node);
                    */
                } else if (builtins.isinstance(target, torch._ops.HigherOrderOperator)) {
                    // assert(len(serialized_node.outputs) === 1 && serialized_node.outputs[0].type in ('as_tensors', 'as_tensor')), 'Only single tensor output or list of tensor output is supported for higher order operators.')
                    const [output] = serialized_node.outputs;
                    const name = output.type === 'as_tensor' ? output.value.name : null;
                    const args = serialized_node.inputs.map((input) => this.deserialize_input(input.arg));
                    fx_node = this.graph.create_node('call_function', target, args, {}, name);
                    if (output.as_tensor !== null) {
                        this.sync_fx_node(name, fx_node);
                    }
                    if (output.as_tensors !== null) {
                        this.deserialize_multiple_outputs(serialized_node, fx_node);
                    }
                } else if (builtins.isinstance(target, torch._ops.OpOverload)) {
                    const name = this._is_single_tensor_return(target) ? serialized_node.outputs[0].as_tensor.name : null;
                    const [args, kwargs] = this.deserialize_inputs(target, serialized_node);
                    fx_node = self.graph.create_node('call_function', target, args, kwargs, name);
                    this.deserialize_outputs(serialized_node, fx_node);
                } else {
                    // throw new python.Error(`Unsupported target type '${target}'.`);
                }
                if (fx_node) {
                    Object.assign(fx_node.meta, this.deserialize_metadata(serialized_node.metadata));
                }
            }
            deserialize(serialized_graph_module, symbol_name_to_range, constants) {
                this.shape_env = new torch.fx.experimental.symbolic_shapes.ShapeEnv(/* assume_static_by_default = True */);
                /*
                this.fake_tensor_mode = FakeTensorMode(
                    allow_fallback_kernels=False,
                    allow_non_fake_inputs=True,
                    shape_env=self.shape_env,
                )
                */
                this.symbol_name_to_symbol = new Map();
                this.symbol_name_to_range = symbol_name_to_range || new Map();
                this.constants = constants || new Map();
                this.deserialize_graph(serialized_graph_module.graph);
                const sig = null; // self.deserialize_signature(serialized_graph_module.signature)
                const module_call_graph = null; // self.deserialize_module_call_graph(serialized_graph_module.module_call_graph)
                return {
                    graph_module: torch._export.exported_program._create_graph_module_for_export(this.module, this.graph),
                    signature: sig,
                    module_call_graph,
                    names_to_symbols: this.symbol_name_to_symbol
                };
            }
            sync_fx_node(name, fx_node) {
                if (this.serialized_name_to_node.has(name)) {
                    throw new python.Error(`Node ${name} has already been deserialized before.`);
                }
                this.serialized_name_to_node.set(name, fx_node);
                fx_node.meta.val = this.serialized_name_to_meta.get(name);
            }
            deserialize_sym_op_inputs(inputs) {
                return inputs.map((input) => this.deserialize_input(input.arg));
            }
            deserialize_inputs(target /* , serialized_node */) {
                const schema_args = target._schema.arguments;
                const actual_args = null;
                /*
                actual_args = {
                    input.name: self.deserialize_input(input.arg) for input in serialized_node.inputs
                }
                */
                const args = [];
                const kwargs = {};
                for (const schema_arg of schema_args) {
                    const is_positional = !schema_arg.has_default_value() && !schema_arg.kwarg_only;
                    if (is_positional) {
                        args.push(actual_args[schema_arg.name]);
                    } else if (schema_arg.name in actual_args) {
                        kwargs[schema_arg.name] = actual_args[schema_arg.name];
                    }
                }
                return [args, kwargs];
            }
            deserialize_input(/* inp */) {
                /*
                value = inp.value
                typ_ = inp.type
                if typ_ === 'as_none':
                    # None should converted as None, but is encoded as bool in serialized
                    # Convert serialized object to torch equivalent
                    return None
                elif typ_ === 'as_tensor':
                    return self.serialized_name_to_node[inp.as_tensor.name]
                elif typ_ === 'as_scalar_type':
                    return _SERIALIZE_TO_TORCH_DTYPE[inp.as_scalar_type]
                elif typ_ === 'as_memory_format':
                    return _SERIALIZE_TO_TORCH_MEMORY_FORMAT[inp.as_memory_format]
                elif typ_ === 'as_layout':
                    return _SERIALIZE_TO_TORCH_LAYOUT[inp.as_layout]
                elif typ_ === 'as_graph':
                    assert isinstance(value, GraphArgument)
                    with self.save_graph_module():
                        self.deserialize_graph(value.graph)
                        submodule = torch._export.exported_program._create_graph_module_for_export(self.module, self.graph)
                    self.module.register_module(value.name, submodule)
                    return self.graph.create_node(
                        'get_attr',
                        value.name,
                        name=value.name,
                    )
                elif typ_ === 'as_device':
                    return deserialize_device(inp.as_device)
                elif typ_ === 'as_int':
                    return inp.as_int
                elif typ_ === 'as_float':
                    return inp.as_float
                elif typ_ === 'as_bool':
                    return inp.as_bool
                elif typ_ === 'as_string':
                    return inp.as_string
                elif typ_ === 'as_sym_int':
                    return self.deserialize_sym_argument(inp.as_sym_int)
                elif typ_ === 'as_sym_bool':
                    return self.deserialize_sym_argument(inp.as_sym_bool)
                elif isinstance(value, list):
                    if len(value) === 0:
                        return []
                    elif isinstance(value[0], TensorArgument):
                        result = []
                        for arg in value:
                            result.append(self.serialized_name_to_node[arg.name])
                        return result
                    elif isinstance(value[0], (int, float, bool)):
                        # convert from serialized.python.types.List to python list
                        return list(value)
                    elif isinstance(value[0], (SymIntArgument, SymBoolArgument)):
                        return [self.deserialize_sym_argument(arg) for arg in value]
                    elif isinstance(value[0], OptionalTensorArgument):
                        def deserialize_optional_tensor_args(a):
                            if a.type === 'as_none':
                                return None
                            elif a.type === 'as_tensor':
                                return self.serialized_name_to_node[a.value]
                            else:
                                raise SerializeError(f'Unhandled argument {inp}')
                        return list(map(deserialize_optional_tensor_args, value))
                    else:
                        raise SerializeError(f'Unhandled argument {inp}')
                elif typ_ === 'as_custom_obj':
                    return self.constants[inp.as_custom_obj.name]
                else {
                    raise SerializeError(`Unhandled argument ${inp}.`);
                }
                */
            }
            deserialize_metadata(metadata) {
                const ret = {};
                const stack_trace = metadata.stack_trace;
                if (stack_trace) {
                    ret.stack_trace = stack_trace;
                }
                const deserialize_meta_func = (serialized_target) => {
                    let module = null;
                    let serialized_target_names = [];
                    if (serialized_target.startsWith('torch.nn')) {
                        module = torch.nn;
                        serialized_target_names = serialized_target.split('.').slice(1);
                    } else if (serialized_target.startsWith('torch')) {
                        module = torch;
                        serialized_target_names = serialized_target.split('.').slice(1);
                    } else {
                        return this.deserialize_operator(serialized_target);
                    }
                    let target = module;
                    for (const name of serialized_target_names) {
                        if (!builtins.hasattr(target, name)) {
                            return serialized_target;
                        }
                        target = builtins.getattr(target, name);
                    }
                    return target;
                };
                const nn_module_stack_str = metadata.nn_module_stack;
                if (nn_module_stack_str) {
                    const import_nn_module_stack = (key, path, ty) => {
                        return [key, [path, ty]];
                    };
                    const nn_module_stack = new Map(nn_module_stack_str.split(';').map((item) => import_nn_module_stack(...item.split(','))));
                    ret.nn_module_stack = nn_module_stack;
                }
                const source_fn_st_str = metadata.source_fn_stack;
                if (source_fn_st_str) {
                    const source_fn_st = [];
                    for (const source_fn_str of source_fn_st_str.split(';')) {
                        const [name, target_str] = source_fn_str.split(',');
                        source_fn_st.push([name, deserialize_meta_func(target_str)]);
                    }
                    ret.source_fn_stack = source_fn_st;
                }
                return ret;
            }
            deserialize_tensor_meta(tensor_meta) {
                const sizes = tensor_meta.sizes.map((val) => this.deserialize_sym_int(val));
                const strides = tensor_meta.strides.map((val) => this.deserialize_sym_int(val));
                const device = this.deserialize_device(tensor_meta.device);
                const dtype = null; // _SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype],
                return torch.empty_strided(sizes, strides, device, dtype);
            }
            deserialize_sym_int(s) {
                if (s.as_expr !== undefined && s.as_expr !== null) {
                    let sym = {};
                    if (this.symbol_name_to_symbol.has(s.as_expr.expr_str)) {
                        sym = this.symbol_name_to_symbol.get(s.as_expr.expr_str);
                    } else {
                        sym = {};
                        /*
                        sym = sympy.sympify(val.expr_str, locals=self.symbol_name_to_symbol)
                        if isinstance(sym, sympy.Symbol) {
                            self.symbol_name_to_symbol[val.expr_str] = sym
                            if vr := self.symbol_name_to_range.get(val.expr_str):
                                symbolic_shapes._constrain_symbol_range(
                                    self.shape_env,
                                    sym,
                                    compiler_min=vr.lower,  # type: ignore[arg-type]
                                    compiler_max=vr.upper,  # type: ignore[arg-type]
                                    runtime_min=vr.lower,  # type: ignore[arg-type]
                                    runtime_max=vr.upper  # type: ignore[arg-type]
                                )
                        }
                        */
                    }
                    const hint = s.as_expr.hint || null;
                    return this.shape_env.create_symintnode(sym, hint);
                } else if (s.as_int !== undefined && s.as_int !== null) {
                    return s.as_int;
                }
                throw new python.Error('SymInt has invalid field type.');
            }
            deserialize_device(d) {
                if (d.index !== undefined) {
                    return new torch.device(d.type, d.index);
                }
                return new torch.device(d.type);
            }
        });
        this.registerFunction('torch_utils.persistence._reconstruct_persistent_obj', (meta) => {
            const name = `_imported_module_${Math.floor(Math.random() * 10000)}`;
            const module = execution.invoke('types.ModuleType', [name]);
            execution.register('sys').modules.set(name, module);
            const context = new python.Execution.Context(module, null);
            execution.exec(meta.module_src, context);
            const obj = execution.invoke(`${name}.${meta.class_name}`, []);
            if (meta.state) {
                if (obj.__setstate__) {
                    obj.__setstate__(meta.state);
                } else {
                    Object.assign(obj, meta.state);
                }
            }
            return obj;
        });
        this.registerFunction('torch_utils.misc.assert_shape', (/* tensor, ref_shape */) => {});
        this.registerFunction('torch_utils.ops.conv2d_resample.conv2d_resample', (/* x, w, f, up, down, padding, groups, flip_weight, flip_filter */) => {});
        this.registerFunction('torch_utils.ops.upfirdn2d.setup_filter', (/* x, f, up, down, padding, flip_filter, gain, impl */) => {});
        this.registerFunction('torch_utils.ops.bias_act', (/* x, b, dim, act, alpha, gain, clamp, impl */) => {});
        this.registerFunction('torch_utils.ops.fma.fma', (/* a, b, c */) => {});
        this.registerType('torch.device', class {
            constructor(type, index) {
                this.type = type;
                if (index) {
                    this.index = index;
                }
            }
        });
        this.registerType('torch.dtype', class {
            constructor(scalar_type, name, itemsize) {
                this._scalar_type = scalar_type;
                this._name = name;
                this._itemsize = itemsize;
            }
            scalar_type() {
                return this._scalar_type;
            }
            itemsize() {
                return this._itemsize;
            }
            __reduce__() {
                return this._name;
            }
            __str__() {
                return `torch.${this._name}`;
            }
            toString() {
                return this.__str__();
            }
        });
        this.registerType('torch.layout', class {
            constructor(name) {
                this._name = name;
            }
            __str__() {
                return this._name;
            }
            toString() {
                return this.__str__();
            }
        });
        this.registerType('torch.qscheme', class {
            constructor(name) {
                this._name = name;
            }
            __str__() {
                return this._name;
            }
            toString() {
                return this.__str__();
            }
        });
        this.registerType('torch.utils.hooks.RemovableHandle', class {
            __setstate__(state) {
                [this.hooks_dict_ref, this.id] = state;
                this.hooks_dict_ref = this.hooks_dict_ref || new Map();
            }
        });
        this.registerType('torch.storage._StorageBase', class {
            constructor(size, dtype) {
                this._size = size;
                this._dtype = dtype;
                this._device = null;
            }
            get device() {
                return this._device;
            }
            get dtype() {
                return this._dtype;
            }
            element_size() {
                return this._dtype.element_size;
            }
            size() {
                return this._size;
            }
            get data() {
                return this._cdata;
            }
            _set_cdata(data) {
                const length = this.size() * this.dtype.itemsize();
                if (length !== data.length) {
                    throw new python.Error('Storage data size mismatch.');
                }
                this._cdata = data;
            }
            _set_from_file(unpickler) {
                const buffer = unpickler.read(8);
                const size = buffer.reverse().reduce((a, b) => (a * 256) + b, 0);
                if (size !== this.size()) {
                    throw new python.Error('Storage size mismatch.');
                }
                const itemsize = this.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                this._set_cdata(data);
            }
            static _new_with_file(unpickler) {
                const buffer = unpickler.read(8);
                const size = buffer.reverse().reduce((a, b) => (a * 256) + b, 0);
                const storage = new this(size);
                const itemsize = storage.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                storage._set_cdata(data);
                return storage;
            }
        });
        this.registerType('torch.storage._UntypedStorage', class extends torch.storage._StorageBase {
            constructor() {
                super();
                throw new python.Error('_UntypedStorage not implemented.');
            }
        });
        this.registerType('torch.storage._TypedStorage', class {
            constructor(...args) {
                if (args.length >= 2 && Number.isInteger(args[0]) && args[1] instanceof torch.dtype) {
                    if (args[3] instanceof torch.device) {
                        [this._size, this._dtype, , this._device] = args;
                    } else {
                        [this._size, this._dtype] = args;
                    }
                } else {
                    throw new python.Error(`Unsupported _TypedStorage arguments '${JSON.stringify(args)}'.`);
                }
            }
            get device() {
                return this._device;
            }
            get dtype() {
                return this._dtype;
            }
            element_size() {
                return this._dtype.element_size;
            }
            size() {
                return this._size;
            }
            get data() {
                return this._cdata;
            }
            _set_cdata(data) {
                const length = this.size() * this.dtype.itemsize();
                if (length !== data.length) {
                    throw new python.Error('Storage data size mismatch.');
                }
                this._cdata = data;
            }
            _set_from_file(unpickler) {
                const buffer = unpickler.read(8);
                const size = buffer.reverse().reduce((a, b) => (a * 256) + b, 0);
                if (size !== this.size()) {
                    throw new python.Error('Storage size mismatch.');
                }
                const itemsize = this.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                this._set_cdata(data);
            }
            static _new_with_file(unpickler) {
                const buffer = unpickler.read(8);
                const size = buffer.reverse().reduce((a, b) => (a * 256) + b, 0);
                const storage = new this(size);
                const itemsize = storage.dtype.itemsize();
                const data = unpickler.stream(itemsize * size);
                storage._set_cdata(data);
                return storage;
            }
        });
        this.registerType('torch.storage._LegacyStorage', class extends torch.storage._TypedStorage {
            constructor() {
                super();
                throw new python.Error('_LegacyStorage not implemented.');
            }
        });
        this.registerType('torch.BoolStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.bool);
            }
        });
        this.registerType('torch.ByteStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.uint8);
            }
        });
        this.registerType('torch.CharStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int8);
            }
        });
        this.registerType('torch.ShortStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int16);
            }
        });
        this.registerType('torch.IntStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int32);
            }
        });
        this.registerType('torch.LongStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.int64);
            }
        });
        this.registerType('torch.HalfStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float16);
            }
        });
        this.registerType('torch.FloatStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float32);
            }
        });
        this.registerType('torch.DoubleStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.float64);
            }
        });
        this.registerType('torch.ComplexHalfStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.complex32);
            }
        });
        this.registerType('torch.ComplexFloatStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.complex64);
            }
        });
        this.registerType('torch.ComplexDoubleStorage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.complex128);
            }
        });
        this.registerType('torch.QInt8Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.qint8);
            }
        });
        this.registerType('torch.QUInt8Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.quint8);
            }
        });
        this.registerType('torch.QInt32Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.qint32);
            }
        });
        this.registerType('torch.BFloat16Storage', class extends torch.storage._StorageBase {
            constructor(size) {
                super(size, torch.bfloat16);
            }
        });
        this.registerType('torch.Size', class extends Array {
            constructor(size) {
                super(size.length);
                for (let i = 0; i < size.length; i++) {
                    this[i] = size[i];
                }
            }
            __len__() {
                return this.length;
            }
        });
        this.registerType('torch.Tensor', class {
            constructor() {
                this._layout = torch.strided;
            }
            get device() {
                return this.storage().device;
            }
            get dtype() {
                if (this._layout === torch.sparse_coo) {
                    return this._values.dtype();
                }
                return this.storage().dtype;
            }
            get shape() {
                return this._shape;
            }
            get layout() {
                return this._layout;
            }
            get values() {
                if (this._layout === torch.sparse_coo) {
                    return this._values;
                }
                throw new python.Error(`Unsupported values in layout'${this._layout.__str__()}'.`);
            }
            get indices() {
                if (this._layout === torch.sparse_coo) {
                    return this._indices;
                }
                throw new python.Error(`Unsupported indices in layout'${this._indices.__str__()}'.`);
            }
            get is_quantized() {
                return this.__quantized__ === true;
            }
            get is_nested() {
                return this.__nested__ === true;
            }
            get is_sparse() {
                return this.layout !== torch.strided;
            }
            size() {
                return this._shape;
            }
            storage() {
                if (!this._storage) {
                    const name = this.__class__.__name__ === 'Tensor' ? 'FloatStorage' : this.__storage__.__name__.replace('Tensor', 'Storage');
                    this._storage = self.invoke(`${this.__class__.__module__}.${name}`, []);
                }
                return this._storage;
            }
            storage_offset() {
                return this._storage_offset;
            }
            stride() {
                return this._stride;
            }
            resize_(shape) {
                this._shape = shape;
            }
            __len__() {
                return this._shape[0];
            }
            __setstate__(state) {
                switch (state.length) {
                    case 3:
                        break;
                    case 4:
                        [this._storage, this._storage_offset, this._shape, this._stride] = state;
                        break;
                    case 5:
                        [this.data, ,this._backward_hooks, this.requires_grad] = state;
                        break;
                    default:
                        throw new python.Error(`Unsupported tensor state length '${state.length}'.`);
                }
            }
            __bool__() {
                return true;
            }
            __int__() {
                const storage = this.storage();
                if (storage && storage.dtype.__reduce__() === 'int64' && storage.data.length === 8) {
                    const buffer = storage.data.peek ? storage.data.peek() : storage.data;
                    const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                    return view.getBigInt64(0, true);
                }
                return NaN;
            }
            __float__() {
                const storage = this.storage();
                if (storage && storage.dtype.__reduce__() === 'float32') {
                    if (storage.size() !== undefined && storage.data.length === 4) {
                        const buffer = storage.data.peek ? storage.data.peek() : storage.data;
                        const view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
                        return view.getFloat32(0, true);
                    }
                }
                return NaN;
            }
            __str__() {
                return 'tensor(...)';
            }
        });
        this.registerType('torch.nn.parameter.Parameter', class extends torch.Tensor {
            constructor(data, requires_grad) {
                super();
                if (!data) {
                    data = self.invoke('torch.Tensor', [[]]);
                }
                this.data = data;
                this.requires_grad = requires_grad === undefined ? true : requires_grad;
            }
        });
        this.registerType('torch.nn.parameter.UninitializedParameter', class extends torch.nn.parameter.Parameter {
            constructor(requires_grad /*, device, dtype */) {
                super(undefined, requires_grad);
            }
        });
        this.registerType('torch.nn.parameter.UninitializedBuffer', class extends torch.Tensor {});
        this.registerType('torch.BoolTensor', class extends torch.Tensor {});
        this.registerType('torch.ByteTensor', class extends torch.Tensor {});
        this.registerType('torch.CharTensor', class extends torch.Tensor {});
        this.registerType('torch.ShortTensor', class extends torch.Tensor {});
        this.registerType('torch.IntTensor', class extends torch.Tensor {});
        this.registerType('torch.LongTensor', class extends torch.Tensor {});
        this.registerType('torch.HalfTensor', class extends torch.Tensor {});
        this.registerType('torch.FloatTensor', class extends torch.Tensor {});
        this.registerType('torch.DoubleTensor', class extends torch.Tensor {});
        this.registerType('torch.ComplexFloatTensor', class extends torch.Tensor {});
        this.registerType('torch.ComplexDoubleTensor', class extends torch.Tensor {});
        this.registerType('torch.QInt8Tensor', class extends torch.Tensor {});
        this.registerType('torch.QUInt8Tensor', class extends torch.Tensor {});
        this.registerType('torch.QInt32Tensor', class extends torch.Tensor {});
        this.registerType('torch.BFloat16Tensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.FloatTensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.DoubleTensor', class extends torch.Tensor {});
        this.registerType('torch.SymInt', class {
            constructor(node) {
                this.node = node;
            }
        });
        this.registerType('torch._C._TensorBase', class {});
        this.registerType('torch._C._VariableFunctionsClass', class {});
        this.register('torch.nn').Module = this.register('torch.nn.modules.module').Module;
        this.register('torch.optim').Adam = this.register('torch.optim.adam').Adam;
        this.register('torch.nn').ReLU = this.register('torch.nn.modules.activation').ReLU;
        this.register('sklearn.utils').Bunch = this.register('sklearn.utils._bunch').Bunch;
        /* eslint-disable no-multi-assign */
        torch.uint8 = torch.ByteStorage.dtype = new torch.dtype(0, 'uint8', 1);
        torch.int8 = torch.CharStorage.dtype = new torch.dtype(1, 'int8', 1);
        torch.int16 = torch.ShortStorage.dtype = new torch.dtype(2, 'int16', 2);
        torch.int32 = torch.IntStorage.dtype = new torch.dtype(3, 'int32', 4);
        torch.int64 = torch.LongStorage.dtype = new torch.dtype(4, 'int64', 8);
        torch.float16 = torch.HalfStorage.dtype = new torch.dtype(5, 'float16', 2);
        torch.float32 = torch.FloatStorage.dtype = new torch.dtype(6, 'float32', 4);
        torch.float64 = torch.DoubleStorage.dtype = new torch.dtype(7, 'float64', 8);
        torch.complex32 = torch.ComplexHalfStorage.dtype = new torch.dtype(8, 'complex32', 4);
        torch.complex64 = torch.ComplexFloatStorage.dtype = new torch.dtype(9, 'complex64', 8);
        torch.complex128 = torch.ComplexDoubleStorage.dtype = new torch.dtype(10, 'complex128', 16);
        torch.bool = torch.BoolStorage.dtype = new torch.dtype(11, 'boolean', 1);
        torch.qint8 = torch.QInt8Storage.dtype = new torch.dtype(12, 'qint8', 1);
        torch.quint8 = torch.QUInt8Storage.dtype = new torch.dtype(13, 'quint8', 1);
        torch.qint32 = torch.QInt32Storage.dtype = new torch.dtype(14, 'qint32', 4);
        torch.bfloat16 = torch.BFloat16Storage.dtype = new torch.dtype(15, 'bfloat16', 2);
        /* eslint-enable no-multi-assign */
        torch.quint4x2 = new torch.dtype(16, 'quint4x2');
        torch.strided = new torch.layout('torch.strided');
        torch.sparse_coo = new torch.layout('torch.sparse_coo');
        torch.sparse_csr = new torch.layout('torch.sparse_csr');
        torch.sparse_csc = new torch.layout('torch.sparse_csc');
        torch.sparse_bsr = new torch.layout('torch.sparse_bsr');
        torch.sparse_bsc = new torch.layout('torch.sparse_bsc');
        torch._mkldnn = new torch.layout('torch._mkldnn');
        torch.per_tensor_affine = new torch.qscheme('torch.per_tensor_affine');
        torch.per_channel_affine = new torch.qscheme('torch.per_channel_affine');
        torch.per_tensor_symmetric = new torch.qscheme('torch.per_tensor_symmetric');
        torch.per_channel_symmetric = new torch.qscheme('torch.per_channel_symmetric');
        torch.per_channel_affine_float_qparams = new torch.qscheme('torch.per_channel_affine_float_qparams');
        torch.inf = this.register('math').inf;
        this.registerFunction('fastcore.basics._using_attr');
        this.registerFunction('fastcore.imports.noop');
        this.registerType('fastcore.basics.fastuple', class {});
        this.registerType('fastcore.basics.GetAttr', class {});
        this.registerType('fastcore.dispatch._TypeDict', class {});
        this.registerType('fastcore.dispatch.TypeDispatch', class {});
        this.registerType('fastcore.foundation.L', class {});
        this.registerType('fastcore.transform.Pipeline', class {});
        this.registerType('fastcore.transform.Transform', class {});
        this.registerType('fastcore.transform.ItemTransform', class extends fastcore.transform.Transform {});
        this.registerType('fastai.callback.core.Callback', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.callback.core.TrainEvalCallback', class {});
        this.registerType('fastai.callback.fp16.AMPMode', class extends this._enum.Enum {});
        this.registerType('fastai.callback.fp16.MixedPrecision', class {});
        this.registerType('fastai.callback.hook._hook_inner', class {});
        this.registerType('fastai.callback.hook.Hook', class {});
        this.registerType('fastai.callback.hook.Hooks', class {});
        this.registerType('fastai.callback.progress.ProgressCallback', class {});
        this.registerType('fastai.callback.progress.ShowGraphCallback', class {});
        this.registerType('fastai.callback.tracker.EarlyStoppingCallback', class {});
        this.registerType('fastai.data.core.DataLoaders', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.data.core.Datasets', class {});
        this.registerType('fastai.data.load.DataLoader', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.data.core.TfmdDL', class extends fastai.data.load.DataLoader {});
        this.registerType('fastai.data.core.TfmdLists', class {});
        this.registerType('fastai.data.load._FakeLoader', class {});
        this.registerFunction('fastai.data.load._wif');
        this.registerType('fastai.data.transforms.Categorize', class {});
        this.registerType('fastai.data.transforms.Category', class {});
        this.registerType('fastai.data.transforms.CategoryMap', class {});
        this.registerType('fastai.data.transforms.ColReader', class {});
        this.registerType('fastai.data.transforms.IntToFloatTensor', class {});
        this.registerType('fastai.data.transforms.Normalize', class {});
        this.registerType('fastai.data.transforms.parent_label', class {});
        this.registerType('fastai.data.transforms.RegressionSetup', class {});
        this.registerType('fastai.data.transforms.ToTensor', class {});
        this.registerType('fastai.imports.noop', class {});
        this.registerType('fastai.layers.AdaptiveConcatPool2d', class {});
        this.registerType('fastai.layers.ConvLayer', class {});
        this.registerType('fastai.layers.Embedding', class {});
        this.registerType('fastai.layers.Flatten', class {});
        this.registerType('fastai.layers.LinBnDrop', class {});
        this.registerType('fastai.layers.MergeLayer', class {});
        this.registerType('fastai.layers.PixelShuffle_ICNR', class {});
        this.registerType('fastai.layers.ResBlock', class {});
        this.registerType('fastai.layers.SelfAttention', class {});
        this.registerType('fastai.layers.SigmoidRange', class {});
        this.registerType('fastai.layers.TimeDistributed', class {});
        this.registerType('fastai.layers.ToTensorBase', class {});
        this.registerType('fastai.learner.Metric', class {});
        this.registerType('fastai.learner.AvgLoss', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.AvgMetric', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.AvgSmoothLoss', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.CastToTensor', class extends fastai.callback.core.Callback {});
        this.registerType('fastai.learner.Dice', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.Learner', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.learner.Recorder', class {});
        this.registerType('fastai.losses.BaseLoss', class {});
        this.registerType('fastai.losses.CrossEntropyLossFlat', class {});
        this.registerType('fastai.metrics.AccumMetric', class extends fastai.learner.Metric {});
        this.registerFunction('fastai.metrics._rmse');
        this.registerFunction('fastai.metrics.accuracy');
        this.registerFunction('fastai.metrics.foreground_acc');
        this.registerFunction('fastai.metrics.mse');
        this.registerFunction('fastai.metrics.error_rate');
        this.registerType('fastai.optimizer._BaseOptimizer', class {});
        this.registerType('fastai.optimizer.Optimizer', class extends fastai.optimizer._BaseOptimizer {});
        this.registerFunction('fastai.optimizer.Adam');
        this.registerFunction('fastai.optimizer.adam_step');
        this.registerFunction('fastai.optimizer.average_grad');
        this.registerFunction('fastai.optimizer.weight_decay');
        this.registerType('fastai.tabular.core.Categorify', class {});
        this.registerType('fastai.tabular.core.FillMissing', class {});
        this.registerType('fastai.tabular.core.FillStrategy', class {});
        this.registerType('fastai.tabular.core.ReadTabBatch', class extends fastcore.transform.ItemTransform {});
        this.registerType('fastai.tabular.core.TabDataLoader', class extends fastai.data.core.TfmdDL {});
        this.registerType('fastai.tabular.data.TabularDataLoaders', class extends fastai.data.core.DataLoaders {});
        this.registerType('fastai.tabular.core.Tabular', class {});
        this.registerType('fastai.tabular.core.TabularPandas', class extends fastai.tabular.core.Tabular {});
        this.registerType('fastai.tabular.learner.TabularLearner', class extends fastai.learner.Learner {});
        this.registerType('fastai.tabular.model.TabularModel', class {});
        this.registerFunction('fastai.torch_core._fa_rebuild_tensor', (cls, ...args) => {
            const tensor = self.invoke('torch._utils._rebuild_tensor_v2', args);
            return self.invoke(cls, tensor);
        });
        this.registerType('fastai.torch_core.TensorBase', class extends torch.Tensor {
            constructor(x) {
                super();
                Object.assign(this, x);
            }
        });
        this.registerType('fastai.torch_core.TensorCategory', class {});
        this.registerType('fastai.torch_core.TensorImage', class {});
        this.registerType('fastai.torch_core.TensorMask', class {});
        this.registerFunction('fastai.torch_core.trainable_params');
        this.registerFunction('fastai.torch_core._rebuild_from_type', (func, type, args, dict) => {
            const tensor = self.invoke(type, [func(...args)]);
            Object.assign(tensor, dict);
            return tensor;
        });
        this.registerFunction('fastai.vision.augment.aug_transforms');
        this.registerType('fastai.vision.augment._BrightnessLogit', class {});
        this.registerType('fastai.vision.augment._ContrastLogit', class {});
        this.registerType('fastai.vision.augment._WarpCoord', class {});
        this.registerType('fastai.vision.augment.Brightness', class {});
        this.registerType('fastai.vision.augment.flip_mat', class {});
        this.registerType('fastai.vision.augment.Flip', class {});
        this.registerType('fastai.vision.augment.RandomResizedCropGPU', class {});
        this.registerType('fastai.vision.augment.Resize', class {});
        this.registerType('fastai.vision.augment.rotate_mat', class {});
        this.registerFunction('fastai.vision.augment.TensorImage.lighting');
        this.registerType('fastai.vision.augment.zoom_mat', class {});
        this.registerType('fastai.vision.core.PILImage', class {});
        this.registerType('fastai.vision.core.PILMask', class {});
        this.registerType('fastai.vision.core.AddMaskCodes', class {});
        this.registerFunction('fastai.vision.learner._resnet_split');
        this.registerFunction('fastai.vision.learner.default_split');
        this.registerType('fastai.learner._ConstantFunc', class {});
        this.registerType('fastai.vision.learner.TimmBody', class {});
        this.registerType('fastai.vision.models.unet.DynamicUnet', class {});
        this.registerType('fastai.vision.models.unet.ResizeToOrig', class {});
        this.registerType('fastai.vision.models.unet.UnetBlock', class {});
    }

    get builtins() {
        return this._builtins;
    }

    source(file) {
        return this._sources.has(file) ? this._sources.get(file) : null;
    }

    debug(/* file */) {
    }

    exec(code , context) {
        const reader = new python.Parser(code, '', null);
        const program = reader.parse();
        if (!program) {
            throw new python.Error("Module '?' parse error.");
        }
        this.block(program.body, context);
    }

    parse(file) {
        const buffer = this.source(file);
        if (buffer) {
            const debug = this.debug(file);
            const code = this._utf8Decoder.decode(buffer);
            const parser = new python.Parser(code, file, debug);
            const program = parser.parse();
            if (!program) {
                throw new python.Error(`Module '${file}' parse error.`);
            }
            return program;
        }
        return null;
    }

    import(name, current, level) {
        if (level) {
            let bits = current.split('.');
            if (bits.length < level) {
                throw new python.Error('Invalid relative import beyond top-level package.');
            }
            bits = bits.slice(0, bits.length - level);
            const base = bits.join('.');
            name = name ? [base, name].join('.') : base;
        }
        const index = name.lastIndexOf('.');
        let parent = null;
        let child = null;
        if (index > 0) {
            parent = name.substring(0, index);
            child = name.substring(index + 1);
            this.import(parent);
        }
        if (!this._modules.has(name)) {
            const module = this._registry.get(name) || new this._builtins.module(name);
            module.__package__ = name;
            this._modules.set(name, module);
            const path = name.split('.').join('/');
            module.__path__ = [path];
            const file = `${path}.py`;
            const program = this.parse(file);
            if (program) {
                module.__file__ = file;
                for (const [name, value] of Object.entries(this.builtins)) {
                    switch (name) {
                        case '__class__':
                        case '__package__':
                        case '__module__':
                        case '__name__':
                        case '__path__':
                        case '__file__':
                            break;
                        default:
                            module[name] = value;
                            break;
                    }
                }
                const context = new python.Execution.Context(module, null);
                if (name !== 'builtins') {
                    context.set('__builtins__', this._modules.get('builtins'));
                }
                this.block(program.body, context);
            }
            if (parent) {
                const parent_module = this._modules.get(parent);
                parent_module[child] = module;
            }
        }
        return this._modules.get(name);
    }

    __import__(name, globals, locals, fromlist, level) {
        let module = null;
        level = level || 0;
        if (level === 0) {
            module = this.import(name);
        } else {
            globals = globals || {};
            let current = globals.__package__;
            if (!current) {
                const spec = globals.__spec__;
                if (spec) {
                    current = spec.parent;
                } else {
                    const name = globals.__name__;
                    const bits = name.split('.');
                    bits.pop();
                    current = bits.join('.');
                }
            }
            module = this.import(name, current, level);
        }
        if (!fromlist) {
            if (level === 0) {
                return this.import(name.split('.')[0]);
            } else if (name) {
                throw new python.Error(`Unsupported relative import '${name}'.`);
                // cut_off = len(name) - len(name.partition('.')[0])
                // return sys.modules[module.__name__[:len(module.__name__)-cut_off]]
            }
        } else if (module.__path__) {
            const handle_fromlist = (module, fromlist, recursive) => {
                for (const name of fromlist) {
                    if (name === '*') {
                        if (!recursive && module.__all__) {
                            handle_fromlist(module, module.__all__, true);
                        }
                    } else if (!module[name]) {
                        this.import(`${module.__name__}.${name}`);
                    }
                }
                return module;
            };
            handle_fromlist(module, fromlist);
        }
        return module;
    }

    module(name) {
        return this._modules.get(name);
    }

    resolve(name) {
        const index = name.lastIndexOf('.');
        const memberName = index === -1 ? name : name.substring(index + 1, name.length);
        const moduleName = index === -1 ? '' : name.substring(0, index);
        const module = this.import(moduleName);
        let type = module ? module[memberName] : null;
        if (!type) {
            if (!this._unresolved.has(name)) {
                const moduleName = name.split('.').shift();
                if (this._registry.has(moduleName) && moduleName !== '__main__') {
                    this.emit('resolve', name);
                }
                const type = this._createType(name, class {});
                this._unresolved.set(name, type);
            }
            type = this._unresolved.get(name);
        }
        return type;
    }

    invoke(target, args) {
        if (typeof target === 'string') {
            target = this.resolve(target);
        }
        if (target) {
            if (target.__class__ === this._builtins.type) {
                if (target.prototype && target.prototype.__class__ === target) {
                    return Reflect.construct(target, args);
                }
                const obj = Object.create(target);
                if (obj.__init__ && typeof obj.__init__ === 'function') {
                    obj.__init__(...args);
                }
                return obj;
            } else if (target.__class__ === this._builtins.function) {
                if (target.__call__) {
                    return target.__call__(args);
                }
                return target(...args);
            }
        }
        throw new python.Error('Unsupported invoke target.');
    }

    call(target, name, args, context) {
        const callTarget = this.target(target, context);
        const callArguments = args.map((argument) => this.expression(argument, context));
        if (!callTarget || (name !== null && !callTarget[name])) {
            if (name === '__new__' && callArguments.length === 1 && callArguments[0] === callTarget) {
                name = null;
                callArguments.shift();
            } else {
                const format = (expression) => {
                    if (expression.type === 'id') {
                        return expression.value;
                    }
                    if (expression.type === '.') {
                        return `${format(expression.target)}.${format(expression.member)}`;
                    }
                    return null;
                };
                const targetName = `${format(target)}.${name}`;
                throw new python.Error(`Unknown function '${targetName}'.`);
            }
        }
        const func = name ? callTarget[name] : callTarget;
        if (func.__class__ === this._builtins.type) {
            if (func.prototype && func.prototype.__class__ === func) {
                return Reflect.construct(func, args);
            }
            const obj = Object.create(func);
            obj.__class__ = func;
            if (obj.__init__ && typeof obj.__init__ === 'function') {
                obj.__init__(...args);
            }
            return obj;
        }
        if (func.__class__ === this._builtins.function) {
            if (func.__call__) {
                return func.__call__(callArguments);
            }
        }
        if (func.__class__ === this._builtins.method) {
            if (func.__call__) {
                return func.__call__([callTarget].concat(callArguments));
            }
        }
        if (typeof func === 'function') {
            return func.apply(callTarget, callArguments);
        }
        throw new python.Error("Unsupported call expression.");
    }

    apply(method, args, context) {
        const locals = Array.prototype.slice.call(args);
        context = new python.Execution.Context(context.globals, {});
        for (const parameter of method.parameters) {
            let value = locals.shift();
            if (value === undefined && parameter.initializer) {
                value = this.expression(parameter.initializer, context);
            }
            context.set(parameter.name, value);
        }
        return this.block(method.body.statements, context);
    }

    block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            const statement = statements.shift();
            const value = this.statement(statement, context);
            if (value !== undefined) {
                return value;
            }
        }
        return undefined;
    }

    statement(statement, context) {
        switch (statement.type) {
            case 'pass': {
                break;
            }
            case 'return': {
                return this.expression(statement.expression, context);
            }
            case 'def': {
                const module = context.get('__name__');
                const self = this;
                const parent = context.get('__class__');
                const type = (parent === this._builtins.module) ? this._builtins.function : this._builtins.method;
                const func = {
                    __class__: type,
                    __globals__: context,
                    __module__: module,
                    __name__: statement.name,
                    __code__: statement,
                    __call__(args) {
                        return self.apply(this.__code__, args, this.__globals__);
                    }
                };
                context.set(statement.name, func);
                break;
            }
            case 'class': {
                const bases = statement.bases.map((arg) => this.expression(arg, context));
                if (bases.length > 1) {
                    throw new python.Error(`Unsupported multiple bases for class '${statement.name}'.`);
                }
                const base = bases.length === 1 ? bases[0] : null;
                const name = `${context.get('__name__')}.${statement.name}`;
                const value = this._createType(name, base ? class extends base {} : class {});
                value.__bases__ = bases;
                context.set(statement.name, value);
                this.block(statement.body.statements, new python.Execution.Context(context.globals, value.prototype));
                break;
            }
            case 'var': {
                context.set(statement.name, statement.initializer ? this.expression(statement.initializer, context) : undefined);
                break;
            }
            case '=': {
                this.expression(statement, context);
                break;
            }
            case 'if': {
                const condition = this.expression(statement.condition, context);
                if (condition === true || condition) {
                    const value = this.block(statement.then.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                    break;
                } else if (condition === false) {
                    if (statement.else) {
                        const value = this.block(statement.else.statements, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    break;
                }
                throw new python.Error("Unsupported condition.");
            }
            case 'for': {
                if (statement.target.length === 1 &&
                    statement.variable.length === 1 && statement.variable[0].type === 'id') {
                    const range = this.expression(statement.target[0], context);
                    const [variable] = statement.variable;
                    for (const current of range) {
                        this.statement({ type: '=', target: variable, expression: { type: 'number', value: current } }, context);
                        const value = this.block(statement.body.statements, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    break;
                }
                throw new python.Error("Unsupported 'for' statement.");
            }
            case 'while': {
                const condition = this.expression(statement.condition, context);
                if (condition) {
                    const value = this.block(statement.body.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                }
                break;
            }
            case 'with': {
                const items = [];
                for (const item of statement.item) {
                    items.push(this.expression(item.expression, context));
                }
                for (const item of items) {
                    if (item.__enter__ && item.__enter__.__call__) {
                        item.__enter__.__call__([item]);
                    }
                }
                const value = this.block(statement.body.statements, context);
                for (const item of items) {
                    if (item.__exit__ && item.__exit__.__call__) {
                        item.__exit__.__call__([item]);
                    }
                }
                if (value !== undefined) {
                    return value;
                }
                break;
            }
            case 'call': {
                this.expression(statement, context);
                break;
            }
            case 'import': {
                for (const alias of statement.names) {
                    let module = this.__import__(alias.name, context);
                    if (alias.asname) {
                        const bits = alias.name.split('.').reverse();
                        bits.pop();
                        while (bits.length > 0) {
                            module = module[bits.pop()];
                        }
                        context.set(alias.asname, module);
                    } else {
                        context.set(alias.name.split('.')[0], module);
                    }
                }
                break;
            }
            case 'import_from': {
                const fromlist = statement.names.map((name) => name.name);
                const module = this.__import__(statement.module, context.globals, context.locals, fromlist, statement.level);
                for (const entry of statement.names) {
                    const name = entry.name;
                    const asname = entry.asname ? entry.asname : null;
                    if (!module[name]) {
                        throw new python.Error(`Cannot import '${name}' from '${statement.module}'.`);
                    }
                    context.set(asname ? asname : name, module[name]);
                }
                break;
            }
            case 'string': {
                break;
            }
            default: {
                throw new python.Error(`Unsupported statement '${statement.type}'.`);
            }
        }
        return undefined;
    }

    expression(expression, context) {
        const self = context.get('self');
        switch (expression.type) {
            case '=': {
                const target = expression.target;
                if (target.type === 'id') {
                    context.set(target.value, this.expression(expression.expression, context));
                    return undefined;
                } else if (target.type === '[]') {
                    if (target.target.type === 'id' &&
                        target.arguments.type === 'list' &&
                        target.arguments.value.length === 1) {
                        const index = this.expression(target.arguments.value[0], context);
                        if (target.target.value === '__annotations__') {
                            context.set(target.target.value, context.get(target.target.value) || {});
                        }
                        const obj = context.get(target.target.value);
                        const value = this.expression(expression.expression, context);
                        if (obj instanceof Map) {
                            obj.set(index, value);
                        } else {
                            obj[index] = value;
                        }
                        return undefined;
                    }
                } else if (target.type === '.' &&
                    target.member.type === 'id') {
                    this.expression(target.target, context)[target.member.value] = this.expression(expression.expression, context);
                    return undefined;
                } else if (target.type === 'tuple') {
                    context.target.push(target.value);
                    const value = this.expression(expression.expression, context);
                    context.target.pop();
                    if  (target.value.every((item) => item.type === 'id')) {
                        if (target.value.length < value.length) {
                            throw new python.Error(`ValueError: too many values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        if (target.value.length > value.length) {
                            throw new python.Error(`ValueError: not enough values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        for (let i = 0; i < value.length; i++) {
                            context.set(target.value[i].value, value[i]);
                        }
                        return undefined;
                    }
                }
                break;
            }
            case 'list': {
                return expression.value.map((item) => this.expression(item, context));
            }
            case 'string': {
                return expression.value.substring(1, expression.value.length - 1);
            }
            case 'number': {
                return Number(expression.value);
            }
            case '[]': {
                if (expression.target.type === 'id' &&
                    expression.arguments.type === 'list' &&
                    expression.arguments.value.length === 1) {
                    if (context.get(expression.target.value)) {
                        const index = this.expression(expression.arguments.value[0], context);
                        const target = context.get(expression.target.value);
                        if (target instanceof Map) {
                            return target.get(index);
                        }
                        return target[index < 0 ? target.length + index : index];
                    }
                }
                const target = this.expression(expression.target, context);
                if (target && expression.arguments.type === 'list' &&
                    (target.__class__ === this._typing._TupleType ||
                     target.__class__ === this._typing._SpecialGenericAlias ||
                     target.__class__ === this._typing._SpecialForm)) {
                    const type = { ...target };
                    type.__args__ = expression.arguments.value.map((arg) => this.expression(arg, context));
                    return type;
                }
                if (expression.arguments.type === 'list' && expression.arguments.value.length === 1) {
                    const index = this.expression(expression.arguments.value[0], context);
                    if (target instanceof Map) {
                        return target.get(index);
                    }
                    return target[index < 0 ? target.length + index : index];
                }
                break;
            }
            case '.': {
                if (expression.member.type === 'id') {
                    const target = this.target(expression.target, context);
                    return target[expression.member.value];
                }
                throw new python.Error("Unsupported field expression.");
            }
            case 'call': {
                if (expression.target.type === '.') {
                    return this.call(expression.target.target, expression.target.member.value, expression.args, context);
                }
                return this.call(expression.target, null, expression.args, context);
            }
            case 'id': {
                switch (expression.value) {
                    case 'self': return self;
                    case 'None': return null;
                    case 'True': return true;
                    case 'False': return false;
                    default: {
                        const type = (value) => {
                            return value &&
                                (value.__class__ === this._builtins.type ||
                                 value.__class__ === this._typing._TupleType ||
                                 value.__class__ === this._typing._SpecialGenericAlias ||
                                 value.__class__ === this._typing._SpecialForm);
                        };
                        const builtin = this._builtins[expression.value];
                        if (type(builtin)) {
                            return builtin;
                        }
                        const value = context.get(expression.value);
                        if (value === undefined) {
                            const typing = this._typing[expression.value];
                            if (type(typing)) {
                                return typing;
                            }
                        }
                        return value;
                    }
                }
            }
            case 'tuple': {
                return expression.value.map((expression) => this.expression(expression, context));
            }
            case 'dict': {
                const dict = {};
                for (const pair of expression.value) {
                    if (pair.type !== 'pair') {
                        throw new python.Error(`Unsupported dict item type '${pair.type}'.`);
                    }
                    const key = this.expression(pair.key, context);
                    const value = this.expression(pair.value, context);
                    dict[key] = value;
                }
                return dict;
            }
            case 'unary': {
                switch (expression.op) {
                    case '-': {
                        return -this.expression(expression.operand, context);
                    }
                    default: {
                        throw new python.Error(`Unsupported unary expression '${expression.op}'.`);
                    }
                }
            }
            case 'binary': {
                switch (expression.op) {
                    case '==': {
                        return this.expression(expression.left, context) === this.expression(expression.right, context);
                    }
                    default: {
                        throw new python.Error(`Unsupported binary expression '${expression.op}'.`);
                    }
                }
            }
            default: {
                throw new python.Error(`Unsupported expression '${expression.type}'.`);
            }
        }
        return undefined;
    }

    target(expression, context) {
        let current = expression;
        let path = [];
        for (;;) {
            if (current.type === '.' && current.member && current.member.type === 'id') {
                path.push(current.member.value);
                current = current.target;
            } else if (current.type === 'id' && current.value !== 'self' && current.value !== 'CONSTANTS') {
                path.push(current.value);
                break;
            } else {
                path = null;
                break;
            }
        }
        if (path) {
            let target = null;
            for (let i = path.length - 1; i >= 0; i--) {
                target = target ? target[path[i]] : context.get(path[i]);
                if (!target) {
                    break;
                }
            }
            if (!target) {
                path.reverse();
                const name = path.join('.');
                const file = `${path.join('/')}.py`;
                if (this._sources.has(file)) {
                    target = this.import(name);
                } else {
                    target = this.resolve(name);
                }
            }
            return target;
        }
        return this.expression(expression, context);
    }

    add(name, source) {
        this._sources.set(name, source);
    }

    on(event, listener) {
        const value = this._events.get(event) || [];
        value.push(listener);
        this._events.set(event, value);
    }

    emit(event, ...args) {
        if (this._events.has(event)) {
            for (const callback of this._events.get(event)) {
                callback(this, ...args);
            }
        }
    }

    register(name, value) {
        if (!this._registry.has(name)) {
            value = value || new (this._registry.get('builtins').module)(name);
            this._registry.set(name, value);
            let current = name;
            for (;;) {
                const index = current.lastIndexOf('.');
                if (index === -1) {
                    break;
                }
                const child = current.substring(index + 1);
                current = current.substring(0, index);
                const parent = this.register(current);
                parent[child] = value;
                value = parent;
            }
        }
        return this._registry.get(name);
    }

    registerFunction(name, value) {
        const parts = name.split('.');
        if (!value) {
            value = () => {
                throw new python.Error(`'${name}' is not implemented.`);
            };
        }
        value.__class__ = this._builtins.function;
        value.__name__ = parts.pop();
        value.__module__ = parts.join('.');
        const module = this.register(value.__module__);
        if (module[name]) {
            throw new python.Error(`Function '${name}' is already registered.`);
        }
        module[value.__name__] = value;
        return value;
    }

    _createType(name, value) {
        const parts = name.split('.');
        value.__class__ = this._builtins.type;
        value.__name__ = parts.pop();
        value.__module__ = parts.join('.');
        value.prototype.__class__ = value;
        return value;
    }

    registerType(name, value) {
        value = this._createType(name, value);
        const parts = name.split('.');
        const memberName = parts.pop();
        const moduleName = parts.join('.');
        const module = this.register(moduleName);
        if (module[memberName]) {
            throw new python.Error(`Class '${memberName}' is already registered.`);
        }
        module[memberName] = value;
        return value;
    }
};

python.Execution.Context = class {

    constructor(globals, locals) {
        this.globals = globals;
        this.locals = locals;
    }

    set(name, value) {
        if (this.locals) {
            this.locals[name] = value;
        } else {
            this.globals[name] = value;
        }
    }

    get(name) {
        if (this.locals && name in this.locals) {
            return this.locals[name];
        }
        if (name in this.globals) {
            return this.globals[name];
        }
        return undefined;
    }

    get target() {
        this._target = this._target || [];
        return this._target;
    }
};

python.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._view = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._utf8Decoder = new TextDecoder('utf-8');
        this._asciiDecoder = new TextDecoder('ascii');
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        this._position = position >= 0 ? position : this._length + position;
        if (this._position > this._buffer.length) {
            throw new python.Error(`Expected ${this._position - this._buffer.length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new python.Error(`Expected ${this._position - this._buffer.length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    stream(length) {
        const buffer = this.read(length);
        return new python.BinaryReader(buffer);
    }

    peek(length) {
        const position = this._position;
        length = length === undefined ? this._length - this._position : length;
        this.skip(length);
        const end = this._position;
        this.skip(-length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, end);
    }

    read(length) {
        const position = this._position;
        length = length === undefined ? this._length - this._position : length;
        this.skip(length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._view.getUint8(position);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._view.getUint16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._view.getInt32(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._view.getUint32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._view.getBigInt64(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._view.getFloat64(position, false);
    }

    string(size, encoding) {
        const data = this.read(size);
        return (encoding === 'utf-8') ?
            this._utf8Decoder.decode(data) :
            this._asciiDecoder.decode(data);
    }

    line() {
        const index = this._buffer.indexOf(0x0A, this._position);
        if (index === -1) {
            throw new python.Error("Could not find end of line.");
        }
        const size = index - this._position;
        const text = this.string(size, 'ascii');
        this.skip(1);
        return text;
    }
};

python.StreamReader = class {

    constructor(stream) {
        this._stream = stream;
        this._length = stream.length;
        this._position = 0;
        this._utf8Decoder = new TextDecoder('utf-8');
        this._asciiDecoder = new TextDecoder('ascii');
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    seek(position) {
        this._stream.seek(position);
        this._position = this._stream.position;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new python.Error(`Expected ${this._position - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
    }

    stream(length) {
        this._stream.seek(this._position);
        this.skip(length);
        return this._stream.stream(length);
    }

    peek(length) {
        this._stream.seek(this._position);
        return this._stream.peek(length);
    }

    read(length) {
        this._stream.seek(this._position);
        this.skip(length);
        return this._stream.read(length);
    }

    byte() {
        const position = this._fill(1);
        return this._view.getUint8(position);
    }

    uint16() {
        const position = this._fill(2);
        return this._view.getUint16(position, true);
    }

    int32() {
        const position = this._fill(4);
        return this._view.getInt32(position, true);
    }

    uint32() {
        const position = this._fill(4);
        return this._view.getUint32(position, true);
    }

    int64() {
        const position = this._fill(8);
        return this._view.getBigInt64(position, true);
    }

    float64() {
        const position = this._fill(8);
        return this._view.getFloat64(position, true);
    }

    string(size, encoding) {
        const data = this.read(size);
        return (encoding === 'utf-8') ?
            this._utf8Decoder.decode(data) :
            this._asciiDecoder.decode(data);
    }

    line() {
        let position = this._fill(0);
        let index = this._buffer.indexOf(0x0A, position);
        if (index === -1) {
            const size = Math.min(0x1000000, this._stream.length - this._position);
            this._fill(size);
            this.skip(-size);
            position = this._fill(0);
            index = this._buffer.indexOf(0x0A, position);
            if (index === -1) {
                throw new python.Error("Could not find end of line.");
            }
        }
        const size = index - position;
        const text = this.string(size, 'ascii');
        this.skip(1);
        return text;
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error(`Expected ${this._position + length - this._length} more bytes. The file might be corrupted. Unexpected end of file.`);
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._stream.seek(this._offset);
            this._buffer = this._stream.read(Math.min(0x10000000, this._length - this._offset));
            this._view = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }
};

python.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Python Error';
    }
};

export const Execution = python.Execution;
