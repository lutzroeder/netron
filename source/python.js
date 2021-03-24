/* jshint esversion: 6 */

// Experimental Python parser

var python = python || {};

python.Parser = class {

    constructor(text, file, debug) {
        this._tokenizer = new python.Tokenizer(text, file);
        this._debug = debug;
        if (!python.Parser._precedence) {
            python.Parser._precedence = {
                'or': 2, 'and': 3, 'not' : 4,
                'in': 5, 'instanceof': 5, 'is': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '<>': 5, '==': 5, '!=': 5,
                '|': 6, '^' : 7, '&' : 8,
                '<<': 9, '>>': 9, '+': 10, '-': 10, '*': 11, '@': 11, '/': 11, '//': 11, '%': 11,
                // '+': 12, '-': 12,
                '~': 13, '**': 14
            };
        }
    }

    parse() {
        const node = this._node('program');
        node.body = [];
        while (!this._tokenizer.match('eof')) {
            const statement = this._parseStatement();
            if (statement) {
                node.body.push(statement);
                continue;
            }
            if (this._tokenizer.eat('\n') || this._tokenizer.eat(';') || this._tokenizer.peek().type == 'eof') {
                continue;
            }
            if (this._tokenizer.eat('indent') && this._tokenizer.peek().type == 'eof') {
                continue;
            }
            throw new python.Error('Unknown statement' + this._tokenizer.location());
        }
        return node;
    }

    _parseSuite() {
        const node = this._node('block');
        node.statements = [];
        let statement = null;
        if (this._tokenizer.eat('\n')) {
            if (this._tokenizer.eat('indent')) {
                while (!this._tokenizer.eat('eof') && !this._tokenizer.eat('dedent')) {
                    if (this._tokenizer.eat(';')) {
                        continue;
                    }
                    statement = this._parseStatement();
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
                    throw new python.Error('Empty statement' + this._tokenizer.location());
                }
            }
        }
        else if (!this._tokenizer.eat('eof')) {
            while (!this._tokenizer.match('\n') && !this._tokenizer.match('eof') && !this._tokenizer.match('dedent')) {
                if (this._tokenizer.eat(';')) {
                    continue;
                }
                statement = this._parseStatement();
                if (statement) {
                    node.statements.push(statement);
                    continue;
                }
                throw new python.Error('Empty statement' + this._tokenizer.location());
            }
            this._tokenizer.eat('\n');
        }

        return node;
    }

    _parseStatement() {

        let node = this._node();

        node = this._eat('id', 'break');
        if (node) {
            return node;
        }
        node = this._eat('id', 'continue');
        if (node) {
            return node;
        }
        node = this._eat('id', 'return');
        if (node) {
            node.expression = this._parseExpression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'raise');
        if (node) {
            node.exception = this._parseExpression(-1, [ 'from' ]);
            if (this._tokenizer.eat('id', 'from')) {
                node.from = this._parseExpression();
            }
            else if (this._tokenizer.eat(',')) {
                node.exception = [ node.exception ];
                node.exception.push(this._parseExpression());
                if (this._tokenizer.eat(',')) {
                    node.exception.push(this._parseExpression());
                }
            }
            return node;
        }
        node = this._eat('id', 'assert');
        if (node) {
            node.condition = this._parseExpression();
            while (this._tokenizer.eat(',')) {
                node.condition = { type: 'list', value: [ node.condition ] };
                node.condition.value.push(this._parseExpression());
            }
            return node;
        }
        node = this._eat('id', 'exec');
        if (node) {
            node.variable = this._parseExpression(-1, [ 'in' ]);
            if (this._tokenizer.eat('in')) {
                do {
                    node.target = node.target || [];
                    node.target.push(this._parseExpression(-1, [ 'in' ], false));
                }
                while (this._tokenizer.eat(','));
            }
            return node;
        }

        node = this._eat('id', 'global');
        if (node) {
            node.variable = [];
            do {
                node.variable.push(this._parseName());
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'nonlocal');
        if (node) {
            node.variable = [];
            do {
                node.variable.push(this._parseName());
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'import');
        if (node) {
            node.modules = [];
            do {
                const module = this._node('module');
                module.name = this._parseExpression(-1, [], false);
                if (this._tokenizer.eat('id', 'as')) {
                    module.as = this._parseExpression(-1, [], false);
                }
                node.modules.push(module);
            }
            while (this._tokenizer.eat(','));
            return node;
        }
        node = this._eat('id', 'from');
        if (node) {
            const dots = this._tokenizer.peek();
            if (dots && Array.from(dots.type).every((c) => c == '.')) {
                node.from = this._eat(dots.type);
                node.from.expression = this._parseExpression();
            }
            else {
                node.from = this._parseExpression();
            }
            this._tokenizer.expect('id', 'import');
            node.import = [];
            const close = this._tokenizer.eat('(');
            do {
                const symbol = this._node();
                symbol.symbol = this._parseExpression(-1, [], false);
                if (this._tokenizer.eat('id', 'as')) {
                    symbol.as = this._parseExpression(-1, [], false);
                }
                node.import.push(symbol);
            }
            while (this._tokenizer.eat(','));
            if (close) {
                this._tokenizer.expect(')');
            }
            return node;
        }
        node = this._eat('id', 'class');
        if (node) {
            node.name = this._parseName().value;
            if (this._tokenizer.peek().value === '(') {
                node.base = this._parseArguments();
            }
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            return node;
        }

        const async = this._eat('id', 'async');
        if (async &&
            !this._tokenizer.match('id', 'def') &&
            !this._tokenizer.match('id', 'with') &&
            !this._tokenizer.match('id', 'for')) {
            throw new python.Error("Expected 'def', 'with' or 'for'" + this._tokenizer.location());
        }

        node = this._eat('id', 'def');
        if (node) {
            if (async) {
                node.async = async;
            }
            node.name = this._parseName().value;
            this._tokenizer.expect('(');
            node.parameters = this._parseParameters(')');
            if (this._tokenizer.eat('->')) {
                node.returnType = this._parseType();
            }
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            return node;
        }
        node = this._eat('id', 'del');
        if (node) {
            node.expression = this._parseExpression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'print');
        if (node) {
            node.expression = this._parseExpression(-1, [], true);
            return node;
        }
        node = this._eat('id', 'if');
        if (node) {
            node.condition = this._parseExpression();
            this._tokenizer.expect(':');
            node.then = this._parseSuite();
            let current = node;
            this._tokenizer.eat('\n');
            while (this._tokenizer.eat('id', 'elif')) {
                current.else = this._node('if');
                current = current.else;
                current.condition = this._parseExpression();
                this._tokenizer.expect(':');
                current.then = this._parseSuite();
                this._tokenizer.eat('\n');
            }
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                current.else = this._parseSuite();
            }
            return node;
        }
        node = this._eat('id', 'while');
        if (node) {
            node.condition = this._parseExpression();
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                node.else = this._parseSuite();
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
            node.variable.push(this._parseExpression(-1, [ 'in' ]));
            while (this._tokenizer.eat(',')) {
                if (this._tokenizer.match('id', 'in')) {
                    node.variable.push({});
                    break;
                }
                node.variable.push(this._parseExpression(-1, [ 'in' ]));
            }
            this._tokenizer.expect('id', 'in');
            node.target = [];
            node.target.push(this._parseExpression());
            while (this._tokenizer.eat(',')) {
                if (this._tokenizer.match(':')) {
                    node.target.push({});
                    break;
                }
                node.target.push(this._parseExpression(-1, [ 'in' ]));
            }
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            if (this._tokenizer.eat('id', 'else')) {
                this._tokenizer.expect(':');
                node.else = this._parseSuite();
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
                item.expression = this._parseExpression();
                if (this._tokenizer.eat('id', 'as')) {
                    item.variable = this._parseExpression();
                }
                node.item.push(item);
            }
            while (this._tokenizer.eat(','));
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            return node;
        }
        node = this._eat('id', 'try');
        if (node) {
            this._tokenizer.expect(':');
            node.body = this._parseSuite();
            node.except = [];
            while (this._tokenizer.match('id', 'except')) {
                const except = this._node('except');
                this._tokenizer.expect('id', 'except');
                except.clause = [];
                except.clause.push(this._parseExpression());
                while (this._tokenizer.eat(',')) {
                    if (this._tokenizer.match(':') || this._tokenizer.match('as')) {
                        except.clause.push({});
                        break;
                    }
                    except.clause.push(this._parseExpression());
                }
                if (this._tokenizer.eat('id', 'as')) {
                    except.variable = this._parseExpression();
                }
                this._tokenizer.expect(':');
                except.body = this._parseSuite();
                node.except.push(except);
            }
            if (this._tokenizer.match('id', 'else')) {
                node.else = this._node('else');
                this._tokenizer.expect('id', 'else');
                this._tokenizer.expect(':');
                node.else.body = this._parseSuite();
            }
            if (this._tokenizer.match('id', 'finally')) {
                node.finally = this._node('finally');
                this._tokenizer.expect('id', 'finally');
                this._tokenizer.expect(':');
                node.finally.body = this._parseSuite();
            }
            return node;
        }

        if (this._tokenizer.match('@')) {
            node = this._node('decorator');
            this._tokenizer.expect('@');
            node.value = this._parseExpression();
            if (!node.value || (node.value.type !== 'call' && node.value.type !== 'id' && node.value.type !== '.')) {
                throw new python.Error('Invalid decorator' + this._tokenizer.location());
            }
            return node;
        }

        const expression = this._parseExpression(-1, [], true);
        if (expression) {
            if (expression.type == 'id' && this._tokenizer.eat(':')) {
                node = this._node('var');
                node.name = expression.value;
                node.location = expression.location;
                node.variableType = this._parseExpression(-1, [ '=' ]);
                if (this._tokenizer.eat('=')) {
                    node.initializer = this._parseExpression();
                }
                return node;
            }
            let statement = false;
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
                    statement = true;
                    break;
            }
            if (statement) {
                return expression;
            }
            throw new python.Error("Unhandled expression" + this._tokenizer.location());
        }

        return null;
    }

    _parseExpression(minPrecedence, terminal, tuple) {
        minPrecedence = minPrecedence || -1;
        const terminalSet = new Set(terminal);
        const stack = [];
        for (;;) {
            let node = this._node();
            const token = this._tokenizer.peek();
            if (stack.length == 1 && terminalSet.has(token.value)) {
                break;
            }
            const precedence = python.Parser._precedence[token.value];
            if (precedence) {
                if (precedence >= minPrecedence) {
                    this._tokenizer.read();
                    node.type = token.value;
                    if (token.type == 'id' && (token.value === 'in' || token.value === 'not')) {
                        if (token.value === 'in') {
                            node.type = 'in';
                        }
                        else if (this._tokenizer.eat('id', 'in')) {
                            node.type = 'not in';
                        }
                        else {
                            node.type = 'not';
                            node.expression = this._parseExpression(precedence, terminal, tuple === false ? false : true);
                            stack.push(node);
                            continue;
                        }
                    }
                    else if (token.value == '~') {
                        node.type = '~';
                        node.expression = this._parseExpression(precedence, terminal, tuple === false ? false : true);
                        stack.push(node);
                        continue;
                    }
                    else if (token.type == 'id' && token.value == 'is') {
                        if (this._tokenizer.eat('id', 'not')) {
                            node.type = 'is not';
                        }
                    }
                    node.left = stack.pop();
                    node.right = this._parseExpression(precedence, terminal, tuple === false ? false : true);
                    stack.push(node);
                    continue;
                }
            }
            if (this._tokenizer.eat(':=')) {
                node.type = ':=';
                node.target = stack.pop();
                node.expression = this._parseExpression(-1, terminal, tuple === false ? false : true);
                stack.push(node);
                continue;
            }
            if (this._tokenizer.eat('=')) {
                node.type = '=';
                node.target = stack.pop();
                node.expression = this._parseExpression(-1, terminal, tuple === false ? false : true);
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
                    node.expression = this._parseExpression(-1, terminal, true);
                    stack.push(node);
                    continue;
            }
            node = this._eat('id', 'if');
            if (node) {
                node.then = stack.pop();
                node.condition = this._parseExpression();
                this._tokenizer.expect('id', 'else');
                node.else = this._parseExpression();
                stack.push(node);
                continue;
            }
            while (this._tokenizer.match('id', 'for') || this._tokenizer.match('id', 'async')) {
                const async = this._eat('id', 'async');
                if (async && !this._tokenizer.match('id', 'for')) {
                    throw new python.Error("Expected 'for'" + this._tokenizer.location());
                }
                node = this._eat('id', 'for');
                if (node) {
                    if (async) {
                        node.async = async;
                    }
                    node.expression = stack.pop();
                    node.variable = this._parseExpression(-1, [ 'in' ], true);
                    this._tokenizer.expect('id', 'in');
                    node.target = this._parseExpression(-1, [ 'for', 'if' ], true);
                    while (this._tokenizer.eat('id', 'if')) {
                        node.condition = node.condition || [];
                        node.condition.push(this._parseExpression(-1, [ 'for', 'if' ]));
                    }
                    stack.push(node);
                }
            }
            node = this._eat('id', 'lambda');
            if (node) {
                node.parameters = this._parseParameters(':');
                node.body = this._parseExpression(-1, terminal, false);
                stack.push(node);
                continue;
            }
            node = this._eat('id', 'yield');
            if (node) {
                if (this._tokenizer.eat('id', 'from')) {
                    node.from = this._parseExpression(-1, [], true);
                }
                else {
                    node.expression = [];
                    do {
                        node.expression.push(this._parseExpression(-1, [], false));
                    }
                    while (this._tokenizer.eat(','));
                }
                stack.push(node);
                continue;
            }
            node = this._eat('id', 'await');
            if (node) {
                node.expression = this._parseExpression(minPrecedence, terminal, tuple);
                stack.push(node);
                continue;
            }
            node = this._eat('.');
            if (node) {
                this._tokenizer.eat('\n');
                node.target = stack.pop();
                node.member = this._parseName();
                stack.push(node);
                continue;
            }
            if (this._tokenizer.peek().value === '(') {
                if (stack.length == 0) {
                    node = this._node('tuple');
                    const args = this._parseArguments();
                    if (args.length == 1) {
                        stack.push(args[0]);
                    }
                    else {
                        node.value = args;
                        stack.push(node);
                    }
                }
                else {
                    node = this._node('call');
                    node.target = stack.pop();
                    node.arguments = this._parseArguments();
                    stack.push(node);
                }
                continue;
            }
            if (this._tokenizer.peek().value === '[') {
                if (stack.length == 0) {
                    stack.push(this._parseExpressions());
                }
                else {
                    node = this._node('[]');
                    node.target = stack.pop();
                    node.arguments = this._parseSlice();
                    stack.push(node);
                }
                continue;
            }
            if (this._tokenizer.peek().value == '{') {
                stack.push(this._parseDictOrSetMaker());
                continue;
            }
            node = this._node();
            const literal = this._parseLiteral();
            if (literal) {
                if (stack.length > 0 && literal.type == 'number' &&
                    (literal.value.startsWith('-') || literal.value.startsWith('+'))) {
                    node.type = literal.value.substring(0, 1);
                    literal.value = literal.value.substring(1);
                    node.left = stack.pop();
                    node.right = literal;
                    stack.push(node);
                }
                else if (stack.length == 1 && literal.type == 'string' && stack[0].type == 'string') {
                    stack[0].value += literal.value;
                }
                else {
                    if (literal.type === 'number') {
                        switch (literal.value) {
                            case 'inf': literal.value = Infinity; break;
                            case '-inf': literal.value = -Infinity; break;
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
            const identifier = this._parseName();
            if (identifier) {
                stack.push(identifier);
                continue;
            }

            if (tuple === true && stack.length == 1 && this._tokenizer.eat(',')) {
                if (stack[0].type === 'tuple') {
                    node = stack[0];
                }
                else {
                    node = this._node('tuple');
                    node.value = [ stack.pop() ];
                    stack.push(node);
                }
                // for, bar, = <expr>
                if (this._tokenizer.peek().value === '=') {
                    continue;
                }
                if (!this._tokenizer.match('=') && !terminalSet.has(this._tokenizer.peek().value)) {
                    const nextTerminal = terminal.slice(0).concat([ ',', '=' ]);
                    const expression = this._parseExpression(minPrecedence, nextTerminal, tuple);
                    if (expression) {
                        node.value.push(expression);
                        continue;
                    }
                }
                break;
            }
            break;
        }

        if (stack.length == 1) {
            return stack.pop();
        }
        if (stack.length != 0) {
            throw new python.Error('Unexpected expression' + this._tokenizer.location());
        }
        return null;
    }

    _parseDictOrSetMaker() {
        const list = [];
        this._tokenizer.expect('{');
        let dict = true;
        while (!this._tokenizer.eat('}')) {
            const item = this._parseExpression(-1, [], false);
            if (item == null) {
                throw new python.Error('Expected expression' + this._tokenizer.location());
            }
            if (!this._tokenizer.eat(':')) {
                dict = false;
            }
            if (dict) {
                const value = this._parseExpression(-1, [], false);
                if (value == null) {
                    throw new python.Error('Expected expression' + this._tokenizer.location());
                }
                list.push({ type: 'pair', key: item, value: value });
            }
            else {
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

    _parseExpressions() {
        const list = [];
        this._tokenizer.expect('[');
        while (!this._tokenizer.eat(']')) {
            const expression = this._parseExpression();
            if (expression == null) {
                throw new python.Error('Expected expression' + this._tokenizer.location());
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

    _parseSlice() {
        let node = { type: '::' };
        let list = [];
        const group = [ 'start', 'stop', 'step' ];
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
            if (this._tokenizer.peek().value != ']') {
                const expression = this._parseExpression();
                if (expression == null) {
                    throw new python.Error('Expected expression' + this._tokenizer.location());
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

    _parseName() {
        const token = this._tokenizer.peek();
        if (token.type == 'id' && !token.keyword) {
            this._tokenizer.read();
            return token;
        }
        return null;
    }

    _parseLiteral() {
        const token = this._tokenizer.peek();
        if (token.type == 'string' || token.type == 'number' || token.type == 'boolean') {
            this._tokenizer.read();
            return token;
        }
        return null;
    }

    _parseTypeArguments() {
        const list = [];
        this._tokenizer.expect('[');
        while (!this._tokenizer.eat(']')) {
            const type = this._parseType();
            if (type == null) {
                throw new python.Error('Expected type ' + this._tokenizer.location());
            }
            list.push(type);
            if (!this._tokenizer.eat(',')) {
                this._tokenizer.expect(']');
                break;
            }
        }
        return list;
    }

    _parseType() {
        const type = this._node();
        type.type = 'type';
        type.name = this._parseExpression(-1, [ '[', '=' ]);
        if (type.name) {
            if (this._tokenizer.peek().value === '[') {
                type.arguments = this._parseTypeArguments();
            }
            return type;
        }
        return null;
    }

    _parseParameter(terminal) {
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
        const identifier = this._parseName();
        if (identifier !== null) {
            node.name = identifier.value;
            if (terminal !== ':' && this._tokenizer.eat(':')) {
                node.parameterType = this._parseType();
            }
            if (this._tokenizer.eat('=')) {
                node.initializer = this._parseExpression();
            }
            return node;
        }
        return null;
    }

    _parseParameters(terminal) {
        const list = [];
        while (!this._tokenizer.eat(terminal)) {
            this._tokenizer.eat('\n');
            if (this._tokenizer.eat('(')) {
                list.push(this._parseParameters(')'));
            }
            else {
                list.push(this._parseParameter(terminal));
            }
            this._tokenizer.eat('\n');
            if (!this._tokenizer.eat(',')) {
                this._tokenizer.expect(terminal);
                break;
            }
        }
        return list;
    }

    _parseArguments() {
        const list = [];
        this._tokenizer.expect('(');
        while (!this._tokenizer.eat(')')) {
            if (this._tokenizer.eat('\n')) {
                continue;
            }
            const expression = this._parseExpression(-1, [], false);
            if (expression == null) {
                throw new python.Error('Expected expression ' + this._tokenizer.location());
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
            python.Tokenizer._whitespace = new RegExp('[\u1680\u180e\u2000-\u200a\u202f\u205f\u3000\ufeff]');
            const identifierStartChars = '\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0370-\u0374\u0376\u0377\u037a-\u037d\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u0527\u0531-\u0556\u0559\u0561-\u0587\u05d0-\u05ea\u05f0-\u05f2\u0620-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06e5\u06e6\u06ee\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u07f4\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828\u0840-\u0858\u08a0\u08a2-\u08ac\u0904-\u0939\u093d\u0950\u0958-\u0961\u0971-\u0977\u0979-\u097f\u0985-\u098c\u098f\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc\u09dd\u09df-\u09e1\u09f0\u09f1\u0a05-\u0a0a\u0a0f\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32\u0a33\u0a35\u0a36\u0a38\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0\u0ae1\u0b05-\u0b0c\u0b0f\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32\u0b33\u0b35-\u0b39\u0b3d\u0b5c\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99\u0b9a\u0b9c\u0b9e\u0b9f\u0ba3\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c33\u0c35-\u0c39\u0c3d\u0c58\u0c59\u0c60\u0c61\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0\u0ce1\u0cf1\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d60\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32\u0e33\u0e40-\u0e46\u0e81\u0e82\u0e84\u0e87\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa\u0eab\u0ead-\u0eb0\u0eb2\u0eb3\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065\u1066\u106e-\u1070\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u13a0-\u13f4\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f0\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1877\u1880-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191c\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19c1-\u19c7\u1a00-\u1a16\u1a20-\u1a54\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5\u1cf6\u1d00-\u1dbf\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u212f-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u2e2f\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312d\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fcc\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua61f\ua62a\ua62b\ua640-\ua66e\ua67f-\ua697\ua6a0-\ua6ef\ua717-\ua71f\ua722-\ua788\ua78b-\ua78e\ua790-\ua793\ua7a0-\ua7aa\ua7f8-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9cf\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa76\uaa7a\uaa80-\uaaaf\uaab1\uaab5\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uabc0-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40\ufb41\ufb43\ufb44\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff21-\uff3a\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc';
            const identifierChars = '\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u0620-\u0649\u0672-\u06d3\u06e7-\u06e8\u06fb-\u06fc\u0730-\u074a\u0800-\u0814\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0840-\u0857\u08e4-\u08fe\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\u0966-\u096f\u0981-\u0983\u09bc\u09be-\u09c4\u09c7\u09c8\u09d7\u09df-\u09e0\u0a01-\u0a03\u0a3c\u0a3e-\u0a42\u0a47\u0a48\u0a4b-\u0a4d\u0a51\u0a66-\u0a71\u0a75\u0a81-\u0a83\u0abc\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0ae6-\u0aef\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47\u0b48\u0b4b-\u0b4d\u0b56\u0b57\u0b5f-\u0b60\u0b66-\u0b6f\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7\u0be6-\u0bef\u0c01-\u0c03\u0c46-\u0c48\u0c4a-\u0c4d\u0c55\u0c56\u0c62-\u0c63\u0c66-\u0c6f\u0c82\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5\u0cd6\u0ce2-\u0ce3\u0ce6-\u0cef\u0d02\u0d03\u0d46-\u0d48\u0d57\u0d62-\u0d63\u0d66-\u0d6f\u0d82\u0d83\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2\u0df3\u0e34-\u0e3a\u0e40-\u0e45\u0e50-\u0e59\u0eb4-\u0eb9\u0ec8-\u0ecd\u0ed0-\u0ed9\u0f18\u0f19\u0f20-\u0f29\u0f35\u0f37\u0f39\u0f41-\u0f47\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u1000-\u1029\u1040-\u1049\u1067-\u106d\u1071-\u1074\u1082-\u108d\u108f-\u109d\u135d-\u135f\u170e-\u1710\u1720-\u1730\u1740-\u1750\u1772\u1773\u1780-\u17b2\u17dd\u17e0-\u17e9\u180b-\u180d\u1810-\u1819\u1920-\u192b\u1930-\u193b\u1951-\u196d\u19b0-\u19c0\u19c8-\u19c9\u19d0-\u19d9\u1a00-\u1a15\u1a20-\u1a53\u1a60-\u1a7c\u1a7f-\u1a89\u1a90-\u1a99\u1b46-\u1b4b\u1b50-\u1b59\u1b6b-\u1b73\u1bb0-\u1bb9\u1be6-\u1bf3\u1c00-\u1c22\u1c40-\u1c49\u1c5b-\u1c7d\u1cd0-\u1cd2\u1d00-\u1dbe\u1e01-\u1f15\u200c\u200d\u203f\u2040\u2054\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2d81-\u2d96\u2de0-\u2dff\u3021-\u3028\u3099\u309a\ua640-\ua66d\ua674-\ua67d\ua69f\ua6f0-\ua6f1\ua7f8-\ua800\ua806\ua80b\ua823-\ua827\ua880-\ua881\ua8b4-\ua8c4\ua8d0-\ua8d9\ua8f3-\ua8f7\ua900-\ua909\ua926-\ua92d\ua930-\ua945\ua980-\ua983\ua9b3-\ua9c0\uaa00-\uaa27\uaa40-\uaa41\uaa4c-\uaa4d\uaa50-\uaa59\uaa7b\uaae0-\uaae9\uaaf2-\uaaf3\uabc0-\uabe1\uabec\uabed\uabf0-\uabf9\ufb20-\ufb28\ufe00-\ufe0f\ufe20-\ufe26\ufe33\ufe34\ufe4d-\ufe4f\uff10-\uff19\uff3f';
            python.Tokenizer._identifierStart = new RegExp('[' + identifierStartChars + ']');
            /* eslint-disable */
            python.Tokenizer._identifierChar = new RegExp('[' + identifierStartChars + identifierChars + ']');
            /* eslint-enable */
        }
    }

    peek() {
        if (!this._cache) {
            this._token = this._tokenize(this._token);
            this._cache = true;
        }
        return this._token;
    }

    read() {
        if (!this._cache) {
            this._token = this._tokenize(this._token);
        }
        const next = this._position + this._token.value.length;
        while (this._position < next) {
            if (python.Tokenizer._isNewline(this._get(this._position))) {
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            }
            else {
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
            throw new python.Error("Unexpected '" + token.value + "' instead of '" + type + "'" + this.location());
        }
        if (value && token.value !== value) {
            throw new python.Error("Unexpected '" + token.value + "' instead of '" + value + "'" + this.location());
        }
        this.read();
    }

    location() {
        return ' at ' + this._file + ':' + (this._line + 1).toString() + ':' + (this._position - this._lineStart + 1).toString();
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
        switch(c) {
            case '\n':
            case '\r':
            case '\u2028': // 8232
            case '\u2029': // 8233
                return true;
        }
        return false;
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

    static _isDecimal(c) {
        return c >= '0' && c <= '9' || c === '_';
    }

    static _isHex(c) {
        return python.Tokenizer._isDecimal(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F') || c === '_';
    }

    static _isOctal(c) {
        return c >= '0' && c <= '7' || c === '_';
    }

    static _isBinary(c) {
        return c === '0' || c === '1' || c === '_';
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
            if (c == '#') {
                this._skipLine();
            }
            else if (python.Tokenizer._isSpace(c)) {
                this._position++;
            }
            else if (c == '\\') {
                // Explicit Line Continuation
                this._position++;
                if (python.Tokenizer._isNewline(this._get(this._position))) {
                    this._position = this._newLine(this._position);
                    this._lineStart = this._position;
                    this._line++;
                }
                else {
                    throw new python.Error("Unexpected '" + this._text[this._position] + "' after line continuation" + this.location());
                }
            }
            else if (this._brackets > 0 && python.Tokenizer._isNewline(c)) {
                // Implicit Line Continuation
                this._position = this._newLine(this._position);
                this._lineStart = this._position;
                this._line++;
            }
            else {
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

    _tokenize(token) {
        if (this._token.type !== '\n') {
            this._skipWhitespace();
        }
        if (this._token.type === 'dedent') {
            this._indentation.pop();
            this._outdent--;
            if (this._outdent > 0) {
                return { type: 'dedent', value: '' };
            }
        }
        if (token.type == '\n') {
            let indent = '';
            let i = this._position;
            while (i < this._text.length) {
                const c = this._text[i];
                if (python.Tokenizer._isSpace(c)) {
                    indent += c;
                    i++;
                }
                else if (python.Tokenizer._isNewline(c)) {
                    indent = '';
                    i = this._newLine(i);
                    this._position = i;
                    this._lineStart = i;
                    this._line++;
                }
                else if (c == '#') {
                    indent = '';
                    while (i < this._text.length && !python.Tokenizer._isNewline(this._text[i])) {
                        i++;
                    }
                    continue;
                }
                else {
                    break;
                }
            }
            let type = null;
            if (indent.length > 0) {
                const current = this._indentation.length > 0 ? this._indentation[this._indentation.length - 1] : '';
                if (indent.length > current.length) {
                    type = 'indent';
                    this._indentation.push(indent);
                }
                else if (indent.length > 0 && indent.length < current.length) {
                    type = 'dedent';
                    this._outdent = 0;
                    for (let j = this._indentation.length - 1; j >= 0 && indent.length < this._indentation[j].length; j--) {
                        this._outdent++;
                    }
                }
                else {
                    this._position += indent.length;
                }
            }
            else if (i >= this._text.length) {
                return { type: 'eof', value: '' };
            }
            else if (this._indentation.length > 0) {
                type = 'dedent';
                this._outdent = this._indentation.length;
            }

            switch (type) {
                case 'indent':
                case 'dedent':
                    return { type: type, value: indent };
            }
        }
        if (this._position >= this._text.length) {
            return { type: 'eof', value: '' };
        }
        const c = this._get(this._position);
        const string = this._string();
        if (string) {
            return string;
        }
        switch (c) {
            case '(':
            case '[':
            case '{':
                this._brackets++;
                return { type: c, value: c };
            case ')':
            case ']':
            case '}':
                if (this._brackets === 0) {
                    throw new python.Error("Unexpected '" + c + "'" + this.location);
                }
                this._brackets--;
                return { type: c, value: c };
            case ',':
            case ';':
            case '?':
                return { type: c, value: c };
            default: {
                const number = this._number();
                if (number) {
                    return number;
                }
                if (c === '.') {
                    let end = this._position + 1;
                    while (this._get(end) === '.') {
                        end++;
                    }
                    const text = this._text.substring(this._position, end);
                    return { type: text, value: text };
                }
                const identifier = this._identifier();
                if (identifier) {
                    return identifier;
                }
                const operator = this._operator();
                if (operator) {
                    return operator;
                }
                break;
            }
        }
        if (c === '.') {
            return { type: c, value: c };
        }
        if (c === '\\') {
            return { type: '\\', value: c };
        }
        if (python.Tokenizer._isNewline(c)) {
            return { type: '\n', value: this._text.substring(this._position, this._newLine(this._position)) };
        }
        throw new python.Error("Unexpected token '" + c + "'" + this.location());
    }

    _number() {
        let c = this._get(this._position);
        const sign = (c === '-' || c === '+') ? 1 : 0;
        let i = this._position + sign;
        c = this._get(i);
        if (c === '0') {
            let radix = 0;
            const n = this._get(i + 1);
            if ((n === 'x' || n === 'X') && python.Tokenizer._isHex(this._get(i + 2))) {
                i += 2;
                while (python.Tokenizer._isHex(this._get(i))) {
                    i += 1;
                }
                if (this._get(i) === 'l' || this._get(i) === 'L') {
                    i += 1;
                }
                radix = 16;
            }
            else if ((n === 'b' || n === 'B') && python.Tokenizer._isBinary(this._get(i + 2))) {
                i += 2;
                while (python.Tokenizer._isBinary(this._get(i))) {
                    i++;
                }
                radix = 2;
            }
            else if ((n === 'o' || n === 'O') && python.Tokenizer._isOctal(this._get(i + 2))) {
                i += 2;
                while (python.Tokenizer._isOctal(this._get(i))) {
                    i++;
                }
                radix = 8;
            }
            else if (n >= '0' && n <= '7') {
                i++;
                while (python.Tokenizer._isOctal(this._get(i))) {
                    i += 1;
                }
                if (this._get(i) === 'l' || this._get(i) === 'L') {
                    i += 1;
                }
                radix = 8;
            }
            if (radix > 0 && this._get(i) !== '.') {
                const radixText = this._text.substring(this._position, i);
                const radixParseText = radixText.indexOf('_') !== -1 ? radixText.split('_').join('') : radixText;
                if (!isNaN(parseInt(radixParseText, radix))) {
                    return { type: 'number', value: radixText };
                }
            }
        }
        i = this._position + sign;
        let decimal = false;
        if (this._get(i) >= '1' && this._get(i) <= '9') {
            while (python.Tokenizer._isDecimal(this._get(i))) {
                i++;
            }
            c = this._get(i).toLowerCase();
            decimal = c !== '.' && c !== 'e';
        }
        if (this._get(i) === '0') {
            i++;
            c = this._get(i).toLowerCase();
            decimal = !python.Tokenizer._isDecimal(c) && c !== '.' && c !== 'e' && c !== 'j';
        }
        if (decimal) {
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
            while (python.Tokenizer._isDecimal(this._get(i))) {
                i++;
            }
            if (this._get(i) === '.') {
                i++;
            }
            while (python.Tokenizer._isDecimal(this._get(i))) {
                i++;
            }
            if (i > (this._position + sign)) {
                if (this._get(i) === 'e' || this._get(i) === 'E') {
                    i++;
                    if (this._get(i) == '-' || this._get(i) == '+') {
                        i++;
                    }
                    if (!python.Tokenizer._isDecimal(this._get(i))) {
                        i = this._position;
                    }
                    else {
                        while (python.Tokenizer._isDecimal(this._get(i))) {
                            i++;
                        }
                    }
                }
                else {
                    while (python.Tokenizer._isDecimal(this._get(i))) {
                        i++;
                    }
                }
            }
            if (i > (this._position + sign)) {
                if (this._get(i) === 'j' || this._get(i) === 'J') {
                    return { type: 'number', value: this._text.substring(this._position, i + 1) };
                }
                const floatText = this._text.substring(this._position, i);
                const floatParseText = floatText.indexOf('_') != -1 ? floatText.split('_').join('') : floatText;
                if (!isNaN(parseFloat(floatParseText))) {
                    return { type: 'number', value: floatText };
                }
            }
        }
        i = this._position + sign;
        if (this._get(i) === 'i' && this._get(i + 1) === 'n' && this._get(i + 2) === 'f' && !python.Tokenizer._isIdentifierChar(this._get(i + 3))) {
            return { type: 'number', value: this._text.substring(this._position, i + 3) };
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
            return { type: 'id', value: text, keyword: python.Tokenizer._isKeyword(text) };
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
                if (c1 === '*') {
                    length = c2 === '=' ? 3 : 2;
                }
                else {
                    length = c1 === '=' ? 2 : 1;
                }
                break;
            case '/':
                if (c1 === '/') {
                    length = c2 === '=' ? 3 : 2;
                }
                else {
                    length = c1 === '=' ? 2 : 1;
                }
                break;
            case '<':
                if (c1 === '>') {
                    length = 2;
                }
                else if (c1 === '<') {
                    length = c2 === '=' ? 3 : 2;
                }
                else {
                    length = c1 === '=' ? 2 : 1;
                }
                break;
            case '>':
                if (c1 === '>') {
                    length = c2 === '=' ? 3 : 2;
                }
                else {
                    length = c1 === '=' ? 2 : 1;
                }
                break;
            case '@':
                length = c1 === '=' ? 2 : 1;
                break;
            case ':':
                length = c1 === '=' ? 2 : 1;
        }
        if (length > 0) {
            const text = this._text.substring(this._position, this._position + length);
            return { type: text, value: text };
        }
        return null;
    }

    _string() {
        let i = this._position;
        let prefix = -1;
        if (this._get(i) === "'" || this._get(i) === '"') {
            prefix = '';
        }
        else if (this._get(i + 1) === "'" || this._get(i + 1) === '"') {
            const c = this._get(i);
            switch (c.toLowerCase()) {
                case 'b':
                case 'f':
                case 'r':
                case 'u':
                    prefix = c;
                    break;
            }
        }
        else if (this._get(i + 2) === "'" || this._get(i + 2) === '"') {
            const cc = this._text.substr(this._position, 2);
            switch (cc.toLowerCase()) {
                case 'br':
                case 'fr':
                case 'rb':
                case 'rf':
                case 'ur':
                    prefix = cc;
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
            }
            i += count;
            if (count == 1) {
                while (i < this._text.length) {
                    if (this._text[i] === quote) {
                        return { type: 'string', value: this._text.substring(this._position, i + 1) };
                    }
                    else if (this._text[i] === '\\' &&
                             (this._get(i + 1) == quote || this._get(i + 1) == '\n' || this._get(i + 1) == '\\')) {
                        i += 2;
                    }
                    else if (this._text[i] === '\r' || this._text[i] === '\n') {
                        break;
                    }
                    else {
                        i++;
                    }
                }
            }
            else if (count == 3) {
                while (i < this._text.length) {
                    if (this._get(i) === quote && this._get(i + 1) === quote && this._get(i + 2) === quote) {
                        return { type: 'string', value: this._text.substring(this._position, i + 3) };
                    }
                    else if (this._get(i) === '\\' && this._get(i + 1) === quote) {
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

    static _isKeyword(value) {
        switch (value) {
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
                return true;
        }
        return false;
    }
};

python.Execution = class {

    constructor(sources, exceptionCallback) {
        const self = this;
        this._sources = sources || new Map();
        this._exceptionCallback = exceptionCallback;
        this._utf8Decoder = new TextDecoder('utf-8');
        this._packages = new Map();
        this._unknownNameMap = new Set();
        this._context = new python.Execution.Context();
        this._context.scope.builtins = {};
        this._context.scope.builtins.type = { __module__: 'builtins', __name__: 'type' };
        this._context.scope.builtins.type.__class__ = this._context.scope.builtins.type;
        this._context.scope.builtins.module = { __module__: 'builtins', __name__: 'module', __class__: this._context.scope.builtins.type };
        this._context.scope.builtins.module.__type__ = this._context.scope.builtins.module;
        this.registerModule('__builtin__');
        this.registerModule('_codecs');
        this.registerModule('argparse');
        this.registerModule('collections');
        this.registerModule('copy_reg');
        this.registerModule('cuml');
        this.registerModule('gensim');
        this.registerModule('joblib');
        this.registerModule('lightgbm');
        this.registerModule('numpy');
        this.registerModule('nolearn');
        this.registerModule('sklearn');
        this.registerModule('typing');
        this.registerModule('xgboost');
        const numpy = this._context.scope.numpy;
        const typing = this._context.scope.typing;
        this.registerType('builtins.function', class {});
        this.registerType('builtins.method', class {});
        this.registerType('builtins.dict', class {});
        this.registerType('builtins.list', class {});
        this.registerType('builtins.bool', class {});
        this.registerType('builtins.int', class {});
        this.registerType('builtins.float', class {});
        this.registerType('builtins.str', class {});
        this.registerType('builtins.tuple', class {});
        this.registerType('typing._Final', class {});
        this.registerType('typing._SpecialForm', class extends typing._Final {});
        this.registerType('typing._BaseGenericAlias', class extends typing._Final {});
        this.registerType('typing._SpecialGenericAlias', class extends typing._BaseGenericAlias {});
        this.registerType('typing._TupleType', class extends typing._SpecialGenericAlias {});
        typing.Optional = self.invoke('typing._SpecialForm', []);
        typing.List = self.invoke('typing._SpecialGenericAlias', []);
        typing.Dict = self.invoke('typing._SpecialGenericAlias', []);
        typing.Tuple = self.invoke('typing._TupleType', []);
        this.registerType('argparse.Namespace', class {
            constructor(args) {
                this.args = args;
            }
        });
        this.registerType('collections.deque', class {
            constructor(iterable) {
                let i = 0;
                for (const value of iterable) {
                    this[i++] = value;
                }
                this.length = i;
            }
        });
        this.registerType('numpy.core._multiarray_umath.scalar', class {
            constructor(dtype, rawData) {
                let data = rawData;
                if (typeof rawData === 'string') {
                    data = new Uint8Array(rawData.length);
                    for (let i = 0; i < rawData.length; i++) {
                        data[i] = rawData.charCodeAt(i);
                    }
                }
                const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
                switch (dtype.name) {
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
                        return dataView.getInt64(0, true);
                }
                throw new python.Error("Unknown scalar type '" + dtype.name + "'.");
            }
        });
        this.registerFunction('numpy.core.multiarray._reconstruct', function(subtype, shape, dtype) {
            return self.invoke(subtype, [ shape, dtype ]);
        });
        this.registerType('numpy.dtype', class {
            constructor(obj, align, copy) {
                this.align = align;
                this.copy = copy;
                switch (obj) {
                    case 'i1': this.name = 'int8'; this.itemsize = 1; break;
                    case 'i2': this.name = 'int16'; this.itemsize = 2; break;
                    case 'i4': this.name = 'int32'; this.itemsize = 4; break;
                    case 'i8': this.name = 'int64'; this.itemsize = 8; break;
                    case 'b1': this.name = 'int8'; this.itemsize = 1; break;
                    case 'u1': this.name = 'uint8'; this.itemsize = 1; break;
                    case 'u2': this.name = 'uint16'; this.itemsize = 2; break;
                    case 'u4': this.name = 'uint32'; this.itemsize = 4; break;
                    case 'u8': this.name = 'uint64'; this.itemsize = 8; break;
                    case 'f2': this.name = 'float16'; this.itemsize = 2; break;
                    case 'f4': this.name = 'float32'; this.itemsize = 4; break;
                    case 'f8': this.name = 'float64'; this.itemsize = 8; break;
                    case 'c8':  this.name = 'complex64'; this.itemsize = 8; break;
                    case 'c16': this.name = 'complex128'; this.itemsize = 16; break;
                    default:
                        if (obj.startsWith('V')) {
                            this.itemsize = Number(obj.substring(1));
                            this.name = 'void' + (this.itemsize * 8).toString();
                        }
                        else if (obj.startsWith('O')) {
                            this.itemsize = Number(obj.substring(1));
                            this.name = 'object';
                        }
                        else if (obj.startsWith('S')) {
                            this.itemsize = Number(obj.substring(1));
                            this.name = 'string';
                        }
                        else if (obj.startsWith('U')) {
                            this.itemsize = Number(obj.substring(1));
                            this.name = 'string';
                        }
                        else if (obj.startsWith('M')) {
                            this.itemsize = Number(obj.substring(1));
                            this.name = 'datetime';
                        }
                        else {
                            throw new python.Error("Unknown dtype '" + obj.toString() + "'.");
                        }
                        break;
                }
            }
            __setstate__(state) {
                switch (state.length) {
                    case 8:
                        this.version = state[0];
                        this.byteorder = state[1];
                        this.subarray = state[2];
                        this.names = state[3];
                        this.fields = state[4];
                        this.elsize = state[5];
                        this.alignment = state[6];
                        this.int_dtypeflags = state[7];
                        break;
                    case 9:
                        this.version = state[0];
                        this.byteorder = state[1];
                        this.subarray = state[2];
                        this.names = state[3];
                        this.fields = state[4];
                        this.elsize = state[5];
                        this.alignment = state[6];
                        this.int_dtypeflags = state[7];
                        this.metadata = state[8];
                        break;
                    default:
                        throw new python.Error("Unknown numpy.dtype setstate length '" + state.length.toString() + "'.");
                }
            }
        });
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
        this.registerType('gensim.models.keyedvectors.Vocab', class {});
        this.registerType('gensim.models.keyedvectors.Word2VecKeyedVectors', class {});
        this.registerType('gensim.models.phrases.Phrases', class {});
        this.registerType('gensim.models.tfidfmodel.TfidfModel', class {});
        this.registerType('gensim.models.word2vec.Vocab', class {});
        this.registerType('gensim.models.word2vec.Word2Vec', class {});
        this.registerType('gensim.models.word2vec.Word2VecTrainables', class {});
        this.registerType('gensim.models.word2vec.Word2VecVocab', class {});
        this.registerType('joblib.numpy_pickle.NumpyArrayWrapper', class {
            constructor(/* subtype, shape, dtype */) {
            }
            __setstate__(state) {
                this.subclass = state.subclass;
                this.dtype = state.dtype;
                this.shape = state.shape;
                this.order = state.order;
                this.allow_mmap = state.allow_mmap;
            }
            __read__(unpickler) {
                if (this.dtype.name == 'object') {
                    return unpickler.load((name, args) => self.invoke(name, args), null);
                }
                else {
                    const size = this.dtype.itemsize * this.shape.reduce((a, b) => a * b);
                    this.data = unpickler.read(size);
                }
                return self.invoke(this.subclass, [ this.shape, this.dtype, this.data ]);
            }
        });
        this.registerType('lightgbm.sklearn.LGBMRegressor', class {});
        this.registerType('lightgbm.sklearn.LGBMClassifier', class {});
        this.registerType('lightgbm.basic.Booster', class {});
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
                this.data = buffer !== undefined ? buffer : null;
                this.offset = offset !== undefined ? offset : 0;
                this.strides = strides !== undefined ? strides : null;
                this.order = offset !== undefined ? order : null;
            }
            __setstate__(state) {
                this.version = state[0];
                this.shape = state[1];
                this.dtype = state[2];
                this.isFortran = state[3];
                this.data = state[4];
            }
            __read__(unpickler) {
                const dims = this.shape && this.shape.length > 0 ? this.shape.reduce((a, b) => a * b) : 1;
                const size = this.dtype.itemsize * dims;
                if (typeof this.data == 'string') {
                    this.data = unpickler.unescape(this.data, size);
                    if (this.data.length != size) {
                        throw new python.Error('Invalid string array data size.');
                    }
                }
                else {
                    if (this.data.length != size) {
                        // throw new pytorch.Error('Invalid array data size.');
                    }
                }
                return this;
            }
        });
        this.registerType('numpy.ma.core.MaskedArray', class extends numpy.ndarray {
            constructor(data /*, mask, dtype, copy, subok, ndmin, fill_value, keep_mask, hard_mask, shrink, order */) {
                super(data.shape, data.dtype, data.data);
            }
        });
        this.registerType('pathlib.PosixPath', class {
            constructor() {
                this.path = Array.from(arguments).join('/');
            }
        });
        this.registerType('sklearn.calibration._CalibratedClassifier', class {});
        this.registerType('sklearn.calibration._SigmoidCalibration', class {});
        this.registerType('sklearn.calibration.CalibratedClassifierCV', class {});
        this.registerType('sklearn.compose._column_transformer.ColumnTransformer', class {});
        this.registerType('sklearn.compose._target.TransformedTargetRegressor', class {});
        this.registerType('sklearn.cluster._dbscan.DBSCAN', class {});
        this.registerType('sklearn.cluster._kmeans.KMeans', class {});
        this.registerType('sklearn.decomposition._pca.PCA', class {});
        this.registerType('sklearn.decomposition.PCA', class {});
        this.registerType('sklearn.decomposition.pca.PCA', class {});
        this.registerType('sklearn.decomposition._truncated_svd.TruncatedSVD', class {});
        this.registerType('sklearn.decomposition.truncated_svd.TruncatedSVD', class {});
        this.registerType('sklearn.discriminant_analysis.LinearDiscriminantAnalysis', class {});
        this.registerType('sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis', class {});
        this.registerType('sklearn.dummy.DummyClassifier', class {});
        this.registerType('sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper', class {
            constructor(/* subtype, shape, dtype */) {
            }
            __setstate__(state) {
                this.subclass = state.subclass;
                this.dtype = state.dtype;
                this.shape = state.shape;
                this.order = state.order;
                this.allow_mmap = state.allow_mmap;
            }
            __read__(unpickler) {
                if (this.dtype.name == 'object') {
                    return unpickler.load((name, args) => self.invoke(name, args), null);
                }
                else {
                    const size = this.dtype.itemsize * this.shape.reduce((a, b) => a * b);
                    this.data = unpickler.read(size);
                }
                return self.invoke(this.subclass, [ this.shape, this.dtype, this.data ]);
            }
        });
        this.registerType('sklearn.externals.joblib.numpy_pickle.NDArrayWrapper', class {});
        this.registerType('sklearn.ensemble._bagging.BaggingClassifier', class {});
        this.registerType('sklearn.ensemble._forest.RandomForestRegressor', class {});
        this.registerType('sklearn.ensemble._forest.RandomForestClassifier', class {});
        this.registerType('sklearn.ensemble._forest.ExtraTreesClassifier', class {});
        this.registerType('sklearn.ensemble._gb_losses.BinomialDeviance', class {});
        this.registerType('sklearn.ensemble._gb_losses.LeastSquaresError', class {});
        this.registerType('sklearn.ensemble._gb_losses.MultinomialDeviance', class {});
        this.registerType('sklearn.ensemble._gb.GradientBoostingClassifier', class {});
        this.registerType('sklearn.ensemble._gb.GradientBoostingRegressor', class {});
        this.registerType('sklearn.ensemble._iforest.IsolationForest', class {});
        this.registerType('sklearn.ensemble._voting.VotingClassifier', class {});
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
        this.registerType('sklearn.feature_selection._univariate_selection.SelectKBest', class {});
        this.registerType('sklearn.feature_selection._univariate_selection.SelectPercentile', class {});
        this.registerType('sklearn.feature_selection._variance_threshold.VarianceThreshold', class {});
        this.registerType('sklearn.feature_selection.univariate_selection.SelectKBest', class {});
        this.registerType('sklearn.feature_selection.variance_threshold.VarianceThreshold', class {});
        this.registerType('sklearn.gaussian_process.gpc.GaussianProcessClassifier', class {});
        this.registerType('sklearn.gaussian_process.kernels.ConstantKernel', class {});
        this.registerType('sklearn.gaussian_process.kernels.Product', class {});
        this.registerType('sklearn.gaussian_process.kernels.RBF', class {});
        this.registerType('sklearn.grid_search._CVScoreTuple', class {});
        this.registerType('sklearn.grid_search.GridSearchCV', class {});
        this.registerType('sklearn.impute._base.SimpleImputer', class {});
        this.registerType('sklearn.impute.SimpleImputer', class {});
        this.registerType('sklearn.isotonic.IsotonicRegression', class {});
        this.registerType('sklearn.linear_model._base.LinearRegression', class {});
        this.registerType('sklearn.linear_model._bayes.BayesianRidge', class {});
        this.registerType('sklearn.linear_model._coordinate_descent.ElasticNet', class {});
        this.registerType('sklearn.linear_model._logistic.LogisticRegression', class {});
        this.registerType('sklearn.linear_model._ridge.Ridge', class {});
        this.registerType('sklearn.linear_model._sgd_fast.Hinge', class {});
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
        this.registerType('sklearn.metrics._scorer._PredictScorer', class {});
        this.registerType('sklearn.metrics.scorer._PredictScorer', class {});
        this.registerType('sklearn.model_selection._search.GridSearchCV', class {});
        this.registerType('sklearn.model_selection._search.RandomizedSearchCV', class {});
        this.registerType('sklearn.model_selection._split.KFold', class {});
        this.registerType('sklearn.multiclass.OneVsRestClassifier', class {});
        this.registerType('sklearn.multioutput.MultiOutputRegressor', class {});
        this.registerType('sklearn.naive_bayes.BernoulliNB', class {});
        this.registerType('sklearn.naive_bayes.ComplementNB', class {});
        this.registerType('sklearn.naive_bayes.GaussianNB', class {});
        this.registerType('sklearn.naive_bayes.MultinomialNB', class {});
        this.registerType('sklearn.neighbors._classification.KNeighborsClassifier', class {});
        this.registerType('sklearn.neighbors._dist_metrics.newObj', class {});
        this.registerType('sklearn.neighbors._kd_tree.newObj', class {});
        this.registerType('sklearn.neighbors._regression.KNeighborsRegressor', class {});
        this.registerType('sklearn.neighbors.classification.KNeighborsClassifier', class {});
        this.registerType('sklearn.neighbors.dist_metrics.newObj', class {});
        this.registerType('sklearn.neighbors.kd_tree.newObj', class {});
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
        this.registerType('sklearn.preprocessing._data.QuantileTransformer', class {});
        this.registerType('sklearn.preprocessing._data.RobustScaler', class {});
        this.registerType('sklearn.preprocessing._data.StandardScaler', class {});
        this.registerType('sklearn.preprocessing._discretization.KBinsDiscretizer', class {});
        this.registerType('sklearn.preprocessing._encoders.OneHotEncoder', class {});
        this.registerType('sklearn.preprocessing._function_transformer.FunctionTransformer', class {});
        this.registerType('sklearn.preprocessing._label.LabelBinarizer', class {});
        this.registerType('sklearn.preprocessing._label.LabelEncoder', class {});
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
        this.registerType('sklearn.svm._classes.SVC', class {});
        this.registerType('sklearn.svm._classes.SVR', class {});
        this.registerType('sklearn.svm.classes.LinearSVC', class {});
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
        this.registerType('sklearn.utils.deprecation.DeprecationDict', class {});
        this.registerType('re.Pattern', function(pattern, flags) {
            this.pattern = pattern;
            this.flags = flags;
        });
        this.registerType('spacy._ml.PrecomputableAffine', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('spacy.syntax._parser_model.ParserModel', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.describe.Biases', class {
            __setstate__(state) {
                Object.assign(this, state);
            }
        });
        this.registerType('thinc.describe.Dimension', class {
            __setstate__(state) {
                Object.assign(this, state);
            }
        });
        this.registerType('thinc.describe.Gradient', class {
            __setstate__(state) {
                Object.assign(this, state);
            }
        });
        this.registerType('thinc.describe.Weights', class {
            __setstate__(state) {
                Object.assign(this, state);
            }
        });
        this.registerType('thinc.describe.Synapses', class {
            __setstate__(state) {
                Object.assign(this, state);
            }
        });
        this.registerType('thinc.neural._classes.affine.Affine', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.convolution.ExtractWindow', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.feature_extracter.FeatureExtracter', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.feed_forward.FeedForward', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.function_layer.FunctionLayer', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.hash_embed.HashEmbed', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.layernorm.LayerNorm', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.maxout.Maxout', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.resnet.Residual', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural._classes.softmax.Softmax', class {
            __setstate__(state) {
                Object.assign(this, new python.Unpickler(state).load((name, args) => self.invoke(name, args), null));
            }
        });
        this.registerType('thinc.neural.mem.Memory', class {
        });
        this.registerType('thinc.neural.ops.NumpyOps', class {
        });
        this.registerType('types.CodeType', class {
            constructor(/* args */) {
            }
        });
        this.registerType('types.MethodType', class {
            constructor(/* args */) {
            }
        });
        this.registerType('xgboost.compat.XGBoostLabelEncoder', class {});
        this.registerType('xgboost.core.Booster', class {});
        this.registerType('xgboost.sklearn.XGBClassifier', class {});
        this.registerType('xgboost.sklearn.XGBRegressor', class {});
        this.registerFunction('__builtin__.bytearray', function(source, encoding /*, errors */) {
            if (source) {
                if (encoding === 'latin-1') {
                    const array = new Uint8Array(source.length);
                    for (let i = 0; i < source.length; i++) {
                        array[i] = source.charCodeAt(i);
                    }
                    return array;
                }
                throw new python.Error("Unsupported bytearray encoding '" + JSON.stringify(encoding) + "'.");
            }
            return [];
        });
        this.registerFunction('__builtin__.bytes', function(source, encoding /*, errors */) {
            if (source) {
                if (encoding === 'latin-1') {
                    const array = new Uint8Array(source.length);
                    for (let i = 0; i < source.length; i++) {
                        array[i] = source.charCodeAt(i);
                    }
                    return array;
                }
                throw new python.Error("Unsupported bytearray encoding '" + JSON.stringify(encoding) + "'.");
            }
            return [];
        });
        this.registerFunction('__builtin__.set', function(iterable) {
            return iterable ? iterable : [];
        });
        this.registerFunction('__builtin__.frozenset', function(iterable) {
            return iterable ? iterable : [];
        });
        this.registerFunction('__builtin__.getattr', function(obj, name, defaultValue) {
            if (Object.prototype.hasOwnProperty.call(obj, name)) {
                return obj[name];
            }
            return defaultValue;
        });
        this.registerFunction('__builtin__.slice', function(start, stop , step) {
            return [ start, stop, step ];
        });
        this.registerFunction('__builtin__.type', function(obj) {
            return obj ? obj.__class__ : undefined;
        });
        this.registerFunction('_codecs.encode', function(obj /*, econding */) {
            return obj;
        });
        this.registerFunction('builtins.bytearray', function(data) {
            return { data: data };
        });
        this.registerFunction('builtins.getattr', function(obj, name, defaultValue) {
            if (Object.prototype.hasOwnProperty.call(obj, name)) {
                return obj[name];
            }
            return defaultValue;
        });
        this.registerFunction('builtins.set', function(iterable) {
            return iterable ? iterable : [];
        });
        this.registerFunction('builtins.slice', function(start, stop, step) {
            return { start: start, stop: stop, step: step };
        });
        this.registerFunction('cloudpickle.cloudpickle._builtin_type', function(name) {
            return name;
        });
        this.registerFunction('collections.Counter', function(/* iterable */) {
            return { __module__: 'collections', __name__: 'Counter' };
        });
        this.registerFunction('collections.OrderedDict', function(args) {
            const obj = new Map();
            obj.__setitem__ = function(key, value) {
                obj.set(key, value);
            };
            if (args) {
                for (const arg of args) {
                    obj.__setitem__(arg[0], arg[1]);
                }
            }
            return obj;
        });
        this.registerFunction('collections.defaultdict', function(/* default_factory */) {
            return {};
        });
        this.registerFunction('copy_reg._reconstructor', function(cls, base, state) {
            // copyreg._reconstructor in Python 3
            switch (base) {
                case '__builtin__.object': {
                    return self.invoke(cls, []);
                }
                case '__builtin__.tuple': {
                    const obj = self.invoke(cls, []);
                    for (let i = 0; i < state.length; i++) {
                        obj[i] = state[i];
                    }
                    return obj;
                }
            }
            throw new python.Error("Unknown copy_reg._reconstructor base type '" + base + "'.");
        });
        this.registerFunction('dill._dill._create_cell', function(/* args */) {
            return function() {
                // TODO
            };
        });
        this.registerFunction('dill._dill._create_code', function(args) {
            return self.invoke('types.CodeType', [ args ]);
        });
        this.registerFunction('dill._dill._create_function', function(/* fcode, fglobals, fname, fdefaults, fclosure, fdict, fkwdefaults */) {
            return function() {
                // TODO
            };
        });
        this.registerFunction('dill._dill._get_attr', function(self, name) {
            if (Object.prototype.hasOwnProperty.call(self, name)) {
                return self[name];
            }
            return undefined;
        });
        this.registerFunction('dill._dill._load_type', function(name) {
            return self.conext.getx('types.' + name);
        });
        this.registerFunction('getattr', function(obj, name, defaultValue) {
            if (Object.prototype.hasOwnProperty.call(obj, name)) {
                return obj[name];
            }
            return defaultValue;
        });
        this.registerFunction('numpy.core.multiarray.scalar', function(dtype, rawData) {
            let data = rawData;
            if (typeof rawData === 'string' || rawData instanceof String) {
                data = new Uint8Array(rawData.length);
                for (let i = 0; i < rawData.length; i++) {
                    data[i] = rawData.charCodeAt(i);
                }
            }
            const dataView = new DataView(data.buffer, data.byteOffset, data.byteLength);
            switch (dtype.name) {
                case 'float32':
                    return dataView.getFloat32(0, true);
                case 'float64':
                    return dataView.getFloat64(0, true);
                case 'uint8':
                    return dataView.getUint8(0, true);
                case 'int8':
                    return dataView.getInt8(0, true);
                case 'int16':
                    return dataView.getInt16(0, true);
                case 'int32':
                    return dataView.getInt32(0, true);
                case 'int64':
                    return dataView.getInt64(0, true);
            }
            throw new python.Error("Unknown scalar type '" + dtype.name + "'.");
        });
        this.registerFunction('numpy.ma.core._mareconstruct', function(subtype, baseclass, baseshape, basetype) {
            const data = self.invoke(baseclass, [ baseshape, basetype ]);
            // = ndarray.__new__(ndarray, baseshape, make_mask_descr(basetype))
            const mask = self.invoke('numpy.ndarray', [ baseshape, '' ]);
            return self.invoke(subtype, [ data, mask, basetype ]);
        });
        this.registerFunction('numpy.random.__RandomState_ctor', function() {
            return {};
        });
        this.registerFunction('numpy.random._pickle.__randomstate_ctor', function() {
            return {};
        });
        this.registerFunction('numpy.core.numeric._frombuffer', function(/* buf, dtype, shape, order */) {
            return {};
        });
        this.registerFunction('re._compile', function(pattern, flags) {
            return self.invoke('re.Pattern', [ pattern, flags ]);
        });
        this.registerFunction('srsly.cloudpickle.cloudpickle._builtin_type', function(name) {
            return function() {
                return self.invoke('types.' + name, arguments);
            };
        });
        this.registerType('cuml.common.array_descriptor.CumlArrayDescriptorMeta', class {});
        this.registerType('cuml.ensemble.randomforestclassifier.RandomForestClassifier', class {});
        this.registerType('cuml.raft.common.handle.Handle', class {
            __setstate__(state) {
                this._handle = state;
            }
        });
    }

    get context() {
        return this._context;
    }

    source(file) {
        return this._sources.has(file) ? this._sources.get(file) : null;
    }

    debug(/* file */) {
    }

    parse(file) {
        const buffer = this.source(file);
        if (buffer) {
            const debug = this.debug(file);
            const code = this._utf8Decoder.decode(buffer);
            const reader = new python.Parser(code, file, debug);
            const program = reader.parse();
            if (!program) {
                throw new python.Error("Module '" + file + "' parse error.");
            }
            return program;
        }
        return null;
    }

    package(name) {
        const index = name.lastIndexOf('.');
        if (index > 0) {
            this.package(name.substring(0, index));
        }
        if (!this._packages.has(name)) {
            const file = 'code/' + name.split('.').join('/') + '.py';
            const program = this.parse(file);
            if (program) {
                let globals = this._context.getx(name);
                if (globals === undefined) {
                    globals = {};
                    this._context.setx(name, globals);
                }
                globals.__class__ = this._context.scope.builtins.module;
                globals.__name__ = name;
                globals.__file__ = file;
                this._packages.set(name, globals);
                const context = this._context.push(globals);
                this.block(program.body, context);
            }
        }
        return this._packages.get(name);
    }

    type(name) {
        const type = this._context.getx(name);
        if (type !== undefined) {
            return type;
        }
        const parts = name.split('.');
        const className = parts.pop();
        const moduleName = parts.join('.');
        const module = this.package(moduleName);
        if (module) {
            return module[className];
        }
        return null;
    }

    invoke(name, args) {
        const target = this.type(name);
        if (target) {
            if (target.__class__ === this._context.scope.builtins.type) {
                if (target.prototype && target.prototype.__class__ === target) {
                    return Reflect.construct(target, args);
                }
                const obj = {};
                obj.__proto__ = target;
                obj.__class__ = target;
                if (obj.__init__ && typeof obj.__init__ === 'function') {
                    obj.__init__.apply(obj, args);
                }
                return obj;
            }
            else if (target.__class__ === this._context.scope.builtins.function) {
                if (target.__call__) {
                    return target.__call__(args);
                }
                else {
                    return target.apply(null, args);
                }
            }
        }
        this._raiseUnkownName(name);
        this.registerType(name, class {});
        return this.invoke(name, []);
    }

    call(target, name, args, context) {
        const callTarget = this._target(target, context);
        const callArguments = args.map((argument) => this.expression(argument, context));
        if (!callTarget || (name !== null && !callTarget[name])) {
            if (name === '__new__' && callArguments.length === 1 && callArguments[0] == callTarget) {
                name = null;
                callArguments.shift();
            }
            else {
                const targetName = python.Utility.target(target) + '.' + name;
                if (this.type(targetName)) {
                    return this.invoke(targetName, callArguments);
                }
                throw new python.Error("Unsupported function '" +  targetName + "'.");
            }
        }
        const func = name ? callTarget[name] : callTarget;
        if (func.__class__ === this._context.scope.builtins.type) {
            if (func.prototype && func.prototype.__class__ === func) {
                return Reflect.construct(func, args);
            }
            const obj = {};
            obj.__proto__ = func;
            obj.__class__ = func;
            if (obj.__init__ && typeof obj.__init__ === 'function') {
                obj.__init__.apply(obj, args);
            }
            return obj;
        }
        if (func.__class__ === this._context.scope.builtins.function) {
            if (func.__call__) {
                return func.__call__(callArguments);
            }
        }
        if (func.__class__ === this._context.scope.builtins.method) {
            if (func.__call__) {
                return func.__call__([ callTarget ].concat(callArguments));
            }
        }
        if (typeof func === 'function') {
            return func.apply(callTarget, callArguments);
        }
        throw new python.Error("Unsupported call expression.");
    }

    apply(method, args, context) {
        const locals = Array.prototype.slice.call(args);
        context = context.push();
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
                let type = null;
                if (parent === this._context.scope.builtins.type) {
                    type = this._context.scope.builtins.method;
                }
                else if (parent === this._context.scope.builtins.module) {
                    type = this._context.scope.builtins.function;
                }
                else {
                    throw new python.Error('Invalid function scope.');
                }
                const func = {
                    __class__: type,
                    __globals__: context,
                    __module__: module,
                    __name__: statement.name,
                    __code__: statement,
                    __call__: function(args) {
                        return self.apply(this.__code__, args, this.__globals__);
                    }
                };
                context.set(statement.name, func);
                break;
            }
            case 'class': {
                const scope = {
                    __class__:this._context.scope.builtins.type,
                    __module__: context.get('__name__'),
                    __name__: statement.name,
                };
                context.set(statement.name, scope);
                context = context.push(scope);
                this.block(statement.body.statements, context);
                context = context.pop();
                break;
            }
            case 'var': {
                context.set(statement.name, undefined);
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
                }
                else if (condition === false) {
                    const value = this.block(statement.else.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                    break;
                }
                throw new python.Error("Unknown condition.");
            }
            case 'for': {
                if (statement.target.length == 1 &&
                    statement.variable.length === 1 && statement.variable[0].type === 'id') {
                    const range = this.expression(statement.target[0], context);
                    const variable = statement.variable[0];
                    for (const current of range) {
                        this.statement({ type: '=', target: variable, expression: { type: 'number', value: current }}, context);
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
            case 'call': {
                this.expression(statement, context);
                break;
            }
            case 'import': {
                for (const module of statement.modules) {
                    const moduleName = python.Utility.target(module.name);
                    const globals = this.package(moduleName);
                    if (module.as) {
                        context.set(module.as, globals);
                    }
                }
                break;
            }
            default: {
                throw new python.Error("Unknown statement '" + statement.type + "'.");
            }
        }
    }


    expression(expression, context) {
        const self = context.getx('self');
        switch (expression.type) {
            case '=': {
                const target = expression.target;
                if (target.type === 'id') {
                    context.set(target.value, this.expression(expression.expression, context));
                    return;
                }
                else if (target.type === '[]') {
                    if (target.target.type === 'id' &&
                        target.arguments.type === 'list' &&
                        target.arguments.value.length === 1) {
                        const index = this.expression(target.arguments.value[0], context);
                        if (target.target.value === '__annotations__') {
                            context.set(target.target.value, context.get(target.target.value) || {});
                        }
                        context.get(target.target.value)[index] = this.expression(expression.expression, context);
                        return;
                    }
                }
                else if (target.type === '.' &&
                    target.member.type === 'id') {
                    this.expression(target.target, context)[target.member.value] = this.expression(expression.expression, context);
                    return;
                }
                else if (target.type === 'tuple') {
                    const value = this.expression(expression.expression, context);
                    if  (target.value.length == value.length && target.value.every((item) => item.type === 'id')) {
                        for (let i = 0; i < value.length; i++) {
                            context.set(target.value[i].value, value[i]);
                        }
                        return;
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
                        return target[index < 0 ? target.length + index : index];
                    }
                }
                const target = this.expression(expression.target, context);
                if (target && expression.arguments.type === 'list' &&
                    (target.__class__ === this.context.scope.typing._TupleType ||
                     target.__class__ === this.context.scope.typing._SpecialGenericAlias ||
                     target.__class__ === this.context.scope.typing._SpecialForm)) {
                    const type = Object.assign({}, target);
                    type.__args__ = expression.arguments.value.map((arg) => this.expression(arg, context));
                    return type;
                }
                if (expression.arguments.type === 'list' && expression.arguments.value.length === 1) {
                    const index = this.expression(expression.arguments.value[0], context);
                    return target[index < 0 ? target.length + index : index];
                }
                break;
            }
            case '.': {
                if (expression.member.type == 'id') {
                    const target = this._target(expression.target, context);
                    return target[expression.member.value];
                }
                throw new python.Error("Unsupported field expression.");
            }
            case 'call': {
                if (expression.target.type === 'id' && expression.target.value === 'annotate' && expression.arguments.length === 2) {
                    return this.expression(expression.arguments[1], context);
                }
                if (expression.target.type === 'id' && expression.target.value === 'unchecked_cast' && expression.arguments.length === 2) {
                    return this.expression(expression.arguments[1], context);
                }
                if (expression.target.type === '.') {
                    return this.call(expression.target.target, expression.target.member.value, expression.arguments, context);
                }
                return this.call(expression.target, null, expression.arguments, context);
            }
            case 'id': {
                switch (expression.value) {
                    case 'self': return self;
                    case 'None': return null;
                    case 'True': return true;
                    case 'False': return false;
                }
                const type =
                    this._context.scope.builtins[expression.value] ||
                    this._context.scope.typing[expression.value] ||
                    this._context.scope.torch[expression.value];
                if (type &&
                    (type.__class__ === this._context.scope.builtins.type ||
                     type.__class__ === this._context.scope.typing._TupleType ||
                     type.__class__ === this._context.scope.typing._SpecialGenericAlias ||
                     type.__class__ === this._context.scope.typing._SpecialForm)) {
                    return type;
                }
                return context.get(expression.value);
            }
            case 'tuple': {
                return expression.value.map((expression) => this.expression(expression, context));
            }
            case 'dict': {
                const dict = {};
                for (const pair of expression.value) {
                    if (pair.type !== 'pair') {
                        throw new python.Error("Unsupported dict item type '" + pair.type + "'.");
                    }
                    const key = this.expression(pair.key, context);
                    const value = this.expression(pair.value, context);
                    dict[key] = value;
                }
                return dict;
            }
        }
        throw new python.Error("Unknown expression '" + expression.type + "'.");
    }

    _target(expression, context) {
        let current = expression;
        let packageName = '';
        for (;;) {
            if (current.type === '.' && current.member && current.member.type === 'id') {
                packageName = '.' + current.member.value + packageName;
                current = current.target;
            }
            else if (current.type === 'id' && current.value !== 'self' && current.value !== 'CONSTANTS') {
                packageName = current.value + packageName;
                break;
            }
            else {
                packageName = null;
                break;
            }
        }
        if (packageName) {
            let target = context.getx(packageName);
            if (!target) {
                target = this.package(packageName);
                if (!target) {
                    target = context.getx(packageName);
                    if (!target) {
                        throw new python.Error("Failed to resolve module '" + packageName + "'.");
                    }
                }
            }
            return target;
        }
        return this.expression(expression, context);
    }

    registerFunction(name, callback) {
        if (this._context.getx(name)) {
            throw new python.Error("Function '" + name + "' is already registered.");
        }
        const parts = name.split('.');
        callback.__class__ = this._context.scope.builtins.function;
        callback.__name__ = parts.pop();
        callback.__module__ = parts.join('.');
        this._context.setx(name, callback);
    }

    registerType(name, type) {
        if (this._context.getx(name)) {
            throw new python.Error("Class '" + name + "' is already registered.");
        }
        const parts = name.split('.');
        type.__class__ = this._context.scope.builtins.type;
        type.__name__ = parts.pop();
        type.__module__ = parts.join('.');
        type.prototype.__class__ = type;
        this._context.setx(name, type);
    }

    registerModule(name) {
        this._context.scope[name] = { __name__: name, __class__: this._context.scope.builtins.module };
    }

    _raiseUnkownName(name) {
        if (name && !this._unknownNameMap.has(name)) {
            this._unknownNameMap.add(name);
            const module = name.split('.').shift();
            if (this._context.scope[module] && this._context.scope[module].__class__ == this._context.scope.builtins.module) {
                this._exceptionCallback(new python.Error("Unknown function '" + name + "'."), false);
            }
        }
    }
};

python.Execution.Context = class {

    constructor(parent, scope) {
        this._parent = parent || null;
        this._scope = scope || {};
    }

    push(scope) {
        return new python.Execution.Context(this, scope);
    }

    pop() {
        return this._parent;
    }

    get scope() {
        return this._scope;
    }

    set(name, value) {
        this._scope[name] = value;
    }

    get(name) {
        if (name in this._scope) {
            return this._scope[name];
        }
        if (this._parent) {
            return this._parent.get(name);
        }
        return undefined;
    }

    setx(name, value) {
        if (typeof name !== 'string' || !name.split) {
            throw new python.Error("Invalid name '" + JSON.stringify(name) + "'.");
        }
        const parts = name.split('.');
        if (parts.length == 1) {
            this.set(parts[0], value);
        }
        else {
            let parent = this.get(parts[0]);
            if (!parent) {
                parent = {};
                this.set(parts[0], parent);
            }
            parts.shift();
            while (parts.length > 1) {
                const part = parts.shift();
                parent[part] = parent[part] || {};
                parent = parent[part];
            }
            parent[parts[0]] = value;
        }
    }

    getx(name) {
        const parts = name.split('.');
        let value = this.get(parts[0]);
        if (value !== undefined) {
            parts.shift();
            while (parts.length > 0 && value[parts[0]]) {
                value = value[parts[0]];
                parts.shift();
            }
            if (parts.length === 0) {
                return value;
            }
        }
        return undefined;
    }
};

python.Utility = class {

    static target(expression) {
        if (expression.type == 'id') {
            return expression.value;
        }
        if (expression.type == '.') {
            return python.Utility.target(expression.target) + '.' + python.Utility.target(expression.member);
        }
        return null;
    }
};

python.Unpickler = class {

    constructor(buffer) {
        this._reader = buffer instanceof Uint8Array ? new python.Unpickler.BinaryReader(buffer) : new python.Unpickler.StreamReader(buffer);
    }

    load(function_call, persistent_load) {
        const reader = this._reader;
        const marker = [];
        let stack = [];
        const memo = new Map();
        const OpCode = python.Unpickler.OpCode;
        while (reader.position < reader.length) {
            const opcode = reader.byte();
            switch (opcode) {
                case OpCode.PROTO: {
                    const version = reader.byte();
                    if (version > 5) {
                        throw new python.Error("Unsupported protocol version '" + version + "'.");
                    }
                    break;
                }
                case OpCode.GLOBAL:
                    stack.push([ reader.line(), reader.line() ].join('.'));
                    break;
                case OpCode.STACK_GLOBAL:
                    stack.push([ stack.pop(), stack.pop() ].reverse().join('.'));
                    break;
                case OpCode.PUT: {
                    const index = parseInt(reader.line(), 10);
                    memo.set(index, stack[stack.length - 1]);
                    break;
                }
                case OpCode.OBJ: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(function_call(items.pop(), items));
                    break;
                }
                case OpCode.GET: {
                    const index = parseInt(reader.line(), 10);
                    stack.push(memo.get(index));
                    break;
                }
                case OpCode.POP:
                    stack.pop();
                    break;
                case OpCode.POP_MARK:
                    stack = marker.pop();
                    break;
                case OpCode.DUP:
                    stack.push(stack[stack.length-1]);
                    break;
                case OpCode.PERSID:
                    stack.push(persistent_load(reader.line()));
                    break;
                case OpCode.BINPERSID:
                    stack.push(persistent_load(stack.pop()));
                    break;
                case OpCode.REDUCE: {
                    const items = stack.pop();
                    const type = stack.pop();
                    stack.push(function_call(type, items));
                    break;
                }
                case OpCode.NEWOBJ: {
                    const items = stack.pop();
                    const type = stack.pop();
                    stack.push(function_call(type, items));
                    break;
                }
                case OpCode.BINGET:
                    stack.push(memo.get(reader.byte()));
                    break;
                case OpCode.LONG_BINGET:
                    stack.push(memo.get(reader.uint32()));
                    break;
                case OpCode.BINPUT:
                    memo.set(reader.byte(), stack[stack.length - 1]);
                    break;
                case OpCode.LONG_BINPUT:
                    memo.set(reader.uint32(), stack[stack.length - 1]);
                    break;
                case OpCode.BININT:
                    stack.push(reader.int32());
                    break;
                case OpCode.BININT1:
                    stack.push(reader.byte());
                    break;
                case OpCode.LONG:
                    stack.push(parseInt(reader.line(), 10));
                    break;
                case OpCode.BININT2:
                    stack.push(reader.uint16());
                    break;
                case OpCode.BINBYTES:
                    stack.push(reader.read(reader.int32()));
                    break;
                case OpCode.BINBYTES8:
                    stack.push(reader.read(reader.int64()));
                    break;
                case OpCode.SHORT_BINBYTES:
                    stack.push(reader.read(reader.byte()));
                    break;
                case OpCode.FLOAT:
                    stack.push(parseFloat(reader.line()));
                    break;
                case OpCode.BINFLOAT:
                    stack.push(reader.float64());
                    break;
                case OpCode.INT: {
                    const value = reader.line();
                    if (value == '01') {
                        stack.push(true);
                    }
                    else if (value == '00') {
                        stack.push(false);
                    }
                    else {
                        stack.push(parseInt(value, 10));
                    }
                    break;
                }
                case OpCode.EMPTY_LIST:
                    stack.push([]);
                    break;
                case OpCode.EMPTY_TUPLE:
                    stack.push([]);
                    break;
                case OpCode.EMPTY_SET:
                    stack.push([]);
                    break;
                case OpCode.ADDITEMS: {
                    const items = stack;
                    stack = marker.pop();
                    const obj = stack[stack.length - 1];
                    for (let i = 0; i < items.length; i++) {
                        obj.push(items[i]);
                    }
                    break;
                }
                case OpCode.DICT: {
                    const items = stack;
                    stack = marker.pop();
                    const dict = {};
                    for (let i = 0; i < items.length; i += 2) {
                        dict[items[i]] = items[i + 1];
                    }
                    stack.push(dict);
                    break;
                }
                case OpCode.LIST: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(items);
                    break;
                }
                case OpCode.TUPLE: {
                    const items = stack;
                    stack = marker.pop();
                    stack.push(items);
                    break;
                }
                case OpCode.SETITEM: {
                    const value = stack.pop();
                    const key = stack.pop();
                    const obj = stack[stack.length - 1];
                    if (obj.__setitem__) {
                        obj.__setitem__(key, value);
                    }
                    else {
                        obj[key] = value;
                    }
                    break;
                }
                case OpCode.SETITEMS: {
                    const items = stack;
                    stack = marker.pop();
                    const obj = stack[stack.length - 1];
                    for (let i = 0; i < items.length; i += 2) {
                        if (obj.__setitem__) {
                            obj.__setitem__(items[i], items[i + 1]);
                        }
                        else {
                            obj[items[i]] = items[i + 1];
                        }
                    }
                    break;
                }
                case OpCode.EMPTY_DICT:
                    stack.push({});
                    break;
                case OpCode.APPEND: {
                    const append = stack.pop();
                    stack[stack.length-1].push(append);
                    break;
                }
                case OpCode.APPENDS: {
                    const appends = stack;
                    stack = marker.pop();
                    const list = stack[stack.length - 1];
                    list.push.apply(list, appends);
                    break;
                }
                case OpCode.STRING: {
                    const str = reader.line();
                    stack.push(str.substr(1, str.length - 2));
                    break;
                }
                case OpCode.BINSTRING:
                    stack.push(reader.string(reader.uint32()));
                    break;
                case OpCode.SHORT_BINSTRING:
                    stack.push(reader.string(reader.byte()));
                    break;
                case OpCode.UNICODE:
                    stack.push(reader.line());
                    break;
                case OpCode.BINUNICODE:
                    stack.push(reader.string(reader.uint32(), 'utf-8'));
                    break;
                case OpCode.SHORT_BINUNICODE:
                    stack.push(reader.string(reader.byte(), 'utf-8'));
                    break;
                case OpCode.BUILD: {
                    const state = stack.pop();
                    let obj = stack.pop();
                    if (obj.__setstate__) {
                        if (obj.__setstate__.__call__) {
                            obj.__setstate__.__call__([ obj, state ]);
                        }
                        else {
                            obj.__setstate__(state);
                        }
                    }
                    else if (ArrayBuffer.isView(state) || Object(state) !== state) {
                        obj.__state__ = state;
                    }
                    else if (obj instanceof Map) {
                        for (const key in state) {
                            obj.set(key, state[key]);
                        }
                    }
                    else {
                        Object.assign(obj, state);
                    }
                    if (obj.__read__) {
                        obj = obj.__read__(this);
                    }
                    stack.push(obj);
                    break;
                }
                case OpCode.MARK:
                    marker.push(stack);
                    stack = [];
                    break;
                case OpCode.NEWTRUE:
                    stack.push(true);
                    break;
                case OpCode.NEWFALSE:
                    stack.push(false);
                    break;
                case OpCode.LONG1: {
                    const data = reader.read(reader.byte());
                    let number = 0;
                    switch (data.length) {
                        case 0: number = 0; break;
                        case 1: number = data[0]; break;
                        case 2: number = data[1] << 8 | data[0]; break;
                        case 3: number = data[2] << 16 | data[1] << 8 | data[0]; break;
                        case 4: number = data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]; break;
                        case 5: number = data[4] * 0x100000000 + ((data[3] << 24 | data[2] << 16 | data[1] << 8 | data[0]) >>> 0); break;
                        default: number = Array.prototype.slice.call(data, 0); break;
                    }
                    stack.push(number);
                    break;
                }
                case OpCode.LONG4:
                    // TODO decode LONG4
                    stack.push(reader.read(reader.uint32()));
                    break;
                case OpCode.TUPLE1:
                    stack.push([ stack.pop() ]);
                    break;
                case OpCode.TUPLE2: {
                    const b = stack.pop();
                    const a = stack.pop();
                    stack.push([ a, b ]);
                    break;
                }
                case OpCode.TUPLE3: {
                    const c = stack.pop();
                    const b = stack.pop();
                    const a = stack.pop();
                    stack.push([ a, b, c ]);
                    break;
                }
                case OpCode.MEMOIZE:
                    memo.set(memo.size, stack[stack.length - 1]);
                    break;
                case OpCode.FRAME:
                    reader.read(8);
                    break;
                case OpCode.BYTEARRAY8: {
                    stack.push(reader.read(reader.int64()));
                    break;
                }
                case OpCode.NONE:
                    stack.push(null);
                    break;
                case OpCode.STOP:
                    return stack.pop();
                default:
                    throw new python.Error("Unknown opcode '" + opcode + "'.");
            }
        }
        throw new python.Error('Unexpected end of file.');
    }

    read(size) {
        return this._reader.read(size);
    }

    stream(size) {
        return this._reader.stream(size);
    }

    int32() {
        return this._reader.int32();
    }

    int64() {
        return this._reader.int64();
    }

    unescape(token, size) {
        const length = token.length;
        const a = new Uint8Array(length);
        if (size && size == length) {
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
            }
            else {
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
                            let xd = token.charCodeAt(i++);
                            xd = xd >= 65 && xd <= 70 ? xd - 55 : xd >= 97 && xd <= 102 ? xd - 87 : xd >= 48 && xd <= 57 ? xd - 48 : -1;
                            if (xd === -1) {
                                i = xsi;
                                o = xso;
                                a[o] = 0x5c;
                                break;
                            }
                            a[o] = a[o] << 4 | xd;
                        }
                        o++;
                        break;
                    }
                    default:
                        if (c < 48 || c > 57) { // 0-9
                            a[o++] = 0x5c;
                            a[o++] = c;
                        }
                        else {
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
};

// https://svn.python.org/projects/python/trunk/Lib/pickletools.py
// https://github.com/python/cpython/blob/master/Lib/pickle.py
python.Unpickler.OpCode = {
    MARK: 40,              // '('
    EMPTY_TUPLE: 41,       // ')'
    STOP: 46,              // '.'
    POP: 48,               // '0'
    POP_MARK: 49,          // '1'
    DUP: 50,               // '2'
    BINBYTES: 66,          // 'B' (Protocol 3)
    SHORT_BINBYTES: 67,    // 'C' (Protocol 3)
    FLOAT: 70,             // 'F'
    BINFLOAT: 71,          // 'G'
    INT: 73,               // 'I'
    BININT: 74,            // 'J'
    BININT1: 75,           // 'K'
    LONG: 76,              // 'L'
    BININT2: 77,           // 'M'
    NONE: 78,              // 'N'
    PERSID: 80,            // 'P'
    BINPERSID: 81,         // 'Q'
    REDUCE: 82,            // 'R'
    STRING: 83,             // 'S'
    BINSTRING: 84,         // 'T'
    SHORT_BINSTRING: 85,   // 'U'
    UNICODE: 86,           // 'V'
    BINUNICODE: 88,        // 'X'
    EMPTY_LIST: 93,        // ']'
    APPEND: 97,            // 'a'
    BUILD: 98,             // 'b'
    GLOBAL: 99,            // 'c'
    DICT: 100,             // 'd'
    APPENDS: 101,          // 'e'
    GET: 103,              // 'g'
    BINGET: 104,           // 'h'
    LONG_BINGET: 106,      // 'j'
    LIST: 108,             // 'l'
    OBJ: 111,              // 'o'
    PUT: 112,              // 'p'
    BINPUT: 113,           // 'q'
    LONG_BINPUT: 114,      // 'r'
    SETITEM: 115,          // 's'
    TUPLE: 116,            // 't'
    SETITEMS: 117,         // 'u'
    EMPTY_DICT: 125,       // '}'
    PROTO: 128,
    NEWOBJ: 129,
    TUPLE1: 133,           // '\x85'
    TUPLE2: 134,           // '\x86'
    TUPLE3: 135,           // '\x87'
    NEWTRUE: 136,          // '\x88'
    NEWFALSE: 137,         // '\x89'
    LONG1: 138,            // '\x8a'
    LONG4: 139,            // '\x8b'
    SHORT_BINUNICODE: 140, // '\x8c' (Protocol 4)
    BINUNICODE8: 141,      // '\x8d' (Protocol 4)
    BINBYTES8: 142,        // '\x8e' (Protocol 4)
    EMPTY_SET: 143,        // '\x8f' (Protocol 4)
    ADDITEMS: 144,         // '\x90' (Protocol 4)
    FROZENSET: 145,        // '\x91' (Protocol 4)
    NEWOBJ_EX: 146,        // '\x92' (Protocol 4)
    STACK_GLOBAL: 147,     // '\x93' (Protocol 4)
    MEMOIZE: 148,          // '\x94' (Protocol 4)
    FRAME: 149,            // '\x95' (Protocol 4)
    BYTEARRAY8: 150,       // '\x96' (Protocol 5)
    NEXT_BUFFER: 151,      // '\x97' (Protocol 5)
    READONLY_BUFFER: 152   // '\x98' (Protocol 5)
};

python.Unpickler.BinaryReader = class {

    constructor(buffer) {
        this._buffer = buffer;
        this._length = buffer.length;
        this._position = 0;
        this._dataView = new DataView(buffer.buffer, buffer.byteOffset, buffer.byteLength);
        this._utf8Decoder = new TextDecoder('utf-8');
        this._asciiDecoder = new TextDecoder('ascii');
    }

    get position() {
        return this._position;
    }

    get length() {
        return this._length;
    }

    skip(offset) {
        this._position += offset;
        if (this._position > this._buffer.length) {
            throw new python.Error('Expected ' + (this._position - this._buffer.length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    stream(length) {
        const buffer = this.read(length);
        return new python.Unpickler.BinaryReader(buffer);
    }

    peek(length) {
        const position = this._position;
        length = length !== undefined ? length : this._length - this._position;
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
        length = length !== undefined ? length : this._length - this._position;
        this.skip(length);
        if (position === 0 && length === this._length) {
            return this._buffer;
        }
        return this._buffer.subarray(position, this._position);
    }

    byte() {
        const position = this._position;
        this.skip(1);
        return this._dataView.getUint8(position);
    }

    uint16() {
        const position = this._position;
        this.skip(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getUint32(position, true);
    }

    int64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getInt64(position, true).toNumber();
    }

    float32() {
        const position = this._position;
        this.skip(4);
        return this._dataView.getFloat32(position, true);
    }

    float64() {
        const position = this._position;
        this.skip(8);
        return this._dataView.getFloat64(position, true);
    }

    string(size, encoding) {
        const data = this.read(size);
        return (encoding == 'utf-8') ?
            this._utf8Decoder.decode(data) :
            this._asciiDecoder.decode(data);
    }

    line() {
        const index = this._buffer.indexOf(0x0A, this._position);
        if (index == -1) {
            throw new python.Error("Could not find end of line.");
        }
        const size = index - this._position;
        const text = this.string(size, 'ascii');
        this.skip(1);
        return text;
    }
};

python.Unpickler.StreamReader = class {

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

    skip(offset) {
        this._position += offset;
        if (this._position > this._length) {
            throw new python.Error('Expected ' + (this._position - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
    }

    stream(length) {
        this._stream.seek(this._position);
        this.skip(length);
        return this._stream.stream(length);
    }

    read(length) {
        this._stream.seek(this._position);
        this.skip(length);
        return this._stream.read(length);
    }

    byte() {
        const position = this._fill(1);
        return this._dataView.getUint8(position);
    }

    uint16() {
        const position = this._fill(2);
        return this._dataView.getUint16(position, true);
    }

    int32() {
        const position = this._fill(4);
        return this._dataView.getInt32(position, true);
    }

    uint32() {
        const position = this._fill(4);
        return this._dataView.getUint32(position, true);
    }

    int64() {
        const position = this._fill(8);
        return this._dataView.getInt64(position, true).toNumber();
    }

    float32() {
        const position = this._fill(4);
        return this._dataView.getFloat32(position, true);
    }

    float64() {
        const position = this._fill(8);
        return this._dataView.getFloat64(position, true);
    }

    string(size, encoding) {
        const data = this.read(size);
        return (encoding == 'utf-8') ?
            this._utf8Decoder.decode(data) :
            this._asciiDecoder.decode(data);
    }

    line() {
        const index = this._buffer.indexOf(0x0A, this._position);
        if (index == -1) {
            throw new python.Error("Could not find end of line.");
        }
        const size = index - this._position;
        const text = this.string(size, 'ascii');
        this.skip(1);
        return text;
    }

    _fill(length) {
        if (this._position + length > this._length) {
            throw new Error('Expected ' + (this._position + length - this._length) + ' more bytes. The file might be corrupted. Unexpected end of file.');
        }
        if (!this._buffer || this._position < this._offset || this._position + length > this._offset + this._buffer.length) {
            this._offset = this._position;
            this._stream.seek(this._offset);
            this._buffer = this._stream.read(Math.min(0x10000000, this._length - this._offset));
            this._dataView = new DataView(this._buffer.buffer, this._buffer.byteOffset, this._buffer.byteLength);
        }
        const position = this._position;
        this._position += length;
        return position - this._offset;
    }
};

python.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Python module.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Execution = python.Execution;
    module.exports.Unpickler = python.Unpickler;
}