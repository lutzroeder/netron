/* jshint esversion: 6 */
/* eslint "indent": [ "error", 4, { "SwitchCase": 1 } ] */

// Experimental Python parser

var python = python || {};

python.Parser = class {

    constructor(text, file) {
        this._tokenizer = new python.Tokenizer(text, file);
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
                    node.target.push(this._parseExpression(-1, [ 'in' ], false))
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
                let module = this._node('module');
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
                let symbol = this._node();
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
                let item = this._node();
                item.type = 'with_item'
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
                let except = this._node('except');
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
                this._tokenizer.expect('id', 'else')
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
            this._tokenizer.expect('@')
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
        let stack = [];
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
                        node.expression.push(this._parseExpression(-1, [], false))
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
                    let nextTerminal = terminal.slice(0).concat([ ',', '=' ]);
                    let expression = this._parseExpression(minPrecedence, nextTerminal, tuple);
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
        let list = [];
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
                list.push(item)
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
        let list = [];
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
                let expression = this._parseExpression();
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
        let list = [];
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
        let type = this._node();
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
        let node = this._node('parameter');
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
        let list = [];
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
        let list = [];
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
        let node = {};
        node.location = this._tokenizer.location();
        if (type) {
            node.type = type;
        }
        return node;
    }

    _eat(type, value) {
        if (this._tokenizer.match(type, value)) {
            let node = this._node(type === 'id' ? value : type);
            this._tokenizer.expect(type, value);
            return node;
        }
        return null;
    }
}

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
            let identifierStartChars = '\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0370-\u0374\u0376\u0377\u037a-\u037d\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u0527\u0531-\u0556\u0559\u0561-\u0587\u05d0-\u05ea\u05f0-\u05f2\u0620-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06e5\u06e6\u06ee\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u07f4\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828\u0840-\u0858\u08a0\u08a2-\u08ac\u0904-\u0939\u093d\u0950\u0958-\u0961\u0971-\u0977\u0979-\u097f\u0985-\u098c\u098f\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc\u09dd\u09df-\u09e1\u09f0\u09f1\u0a05-\u0a0a\u0a0f\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32\u0a33\u0a35\u0a36\u0a38\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0\u0ae1\u0b05-\u0b0c\u0b0f\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32\u0b33\u0b35-\u0b39\u0b3d\u0b5c\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99\u0b9a\u0b9c\u0b9e\u0b9f\u0ba3\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c33\u0c35-\u0c39\u0c3d\u0c58\u0c59\u0c60\u0c61\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0\u0ce1\u0cf1\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d60\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32\u0e33\u0e40-\u0e46\u0e81\u0e82\u0e84\u0e87\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa\u0eab\u0ead-\u0eb0\u0eb2\u0eb3\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065\u1066\u106e-\u1070\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u13a0-\u13f4\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f0\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1877\u1880-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191c\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19c1-\u19c7\u1a00-\u1a16\u1a20-\u1a54\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5\u1cf6\u1d00-\u1dbf\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u212f-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u2e2f\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312d\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fcc\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua61f\ua62a\ua62b\ua640-\ua66e\ua67f-\ua697\ua6a0-\ua6ef\ua717-\ua71f\ua722-\ua788\ua78b-\ua78e\ua790-\ua793\ua7a0-\ua7aa\ua7f8-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9cf\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa76\uaa7a\uaa80-\uaaaf\uaab1\uaab5\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uabc0-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40\ufb41\ufb43\ufb44\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff21-\uff3a\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc';
            let identifierChars = '\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u0620-\u0649\u0672-\u06d3\u06e7-\u06e8\u06fb-\u06fc\u0730-\u074a\u0800-\u0814\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0840-\u0857\u08e4-\u08fe\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\u0966-\u096f\u0981-\u0983\u09bc\u09be-\u09c4\u09c7\u09c8\u09d7\u09df-\u09e0\u0a01-\u0a03\u0a3c\u0a3e-\u0a42\u0a47\u0a48\u0a4b-\u0a4d\u0a51\u0a66-\u0a71\u0a75\u0a81-\u0a83\u0abc\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0ae6-\u0aef\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47\u0b48\u0b4b-\u0b4d\u0b56\u0b57\u0b5f-\u0b60\u0b66-\u0b6f\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7\u0be6-\u0bef\u0c01-\u0c03\u0c46-\u0c48\u0c4a-\u0c4d\u0c55\u0c56\u0c62-\u0c63\u0c66-\u0c6f\u0c82\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5\u0cd6\u0ce2-\u0ce3\u0ce6-\u0cef\u0d02\u0d03\u0d46-\u0d48\u0d57\u0d62-\u0d63\u0d66-\u0d6f\u0d82\u0d83\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2\u0df3\u0e34-\u0e3a\u0e40-\u0e45\u0e50-\u0e59\u0eb4-\u0eb9\u0ec8-\u0ecd\u0ed0-\u0ed9\u0f18\u0f19\u0f20-\u0f29\u0f35\u0f37\u0f39\u0f41-\u0f47\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u1000-\u1029\u1040-\u1049\u1067-\u106d\u1071-\u1074\u1082-\u108d\u108f-\u109d\u135d-\u135f\u170e-\u1710\u1720-\u1730\u1740-\u1750\u1772\u1773\u1780-\u17b2\u17dd\u17e0-\u17e9\u180b-\u180d\u1810-\u1819\u1920-\u192b\u1930-\u193b\u1951-\u196d\u19b0-\u19c0\u19c8-\u19c9\u19d0-\u19d9\u1a00-\u1a15\u1a20-\u1a53\u1a60-\u1a7c\u1a7f-\u1a89\u1a90-\u1a99\u1b46-\u1b4b\u1b50-\u1b59\u1b6b-\u1b73\u1bb0-\u1bb9\u1be6-\u1bf3\u1c00-\u1c22\u1c40-\u1c49\u1c5b-\u1c7d\u1cd0-\u1cd2\u1d00-\u1dbe\u1e01-\u1f15\u200c\u200d\u203f\u2040\u2054\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2d81-\u2d96\u2de0-\u2dff\u3021-\u3028\u3099\u309a\ua640-\ua66d\ua674-\ua67d\ua69f\ua6f0-\ua6f1\ua7f8-\ua800\ua806\ua80b\ua823-\ua827\ua880-\ua881\ua8b4-\ua8c4\ua8d0-\ua8d9\ua8f3-\ua8f7\ua900-\ua909\ua926-\ua92d\ua930-\ua945\ua980-\ua983\ua9b3-\ua9c0\uaa00-\uaa27\uaa40-\uaa41\uaa4c-\uaa4d\uaa50-\uaa59\uaa7b\uaae0-\uaae9\uaaf2-\uaaf3\uabc0-\uabe1\uabec\uabed\uabf0-\uabf9\ufb20-\ufb28\ufe00-\ufe0f\ufe20-\ufe26\ufe33\ufe34\ufe4d-\ufe4f\uff10-\uff19\uff3f';
            python.Tokenizer._identifierStart = new RegExp('[' + identifierStartChars + ']');
            python.Tokenizer._identifierChar = new RegExp('[' + identifierStartChars + identifierChars + ']');
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
                let  c = this._text[i];
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
                let current = this._indentation.length > 0 ? this._indentation[this._indentation.length - 1] : '';
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
            let n = this._get(i + 1);
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
                let floatParseText = floatText.indexOf('_') != -1 ? floatText.split('_').join('') : floatText;
                if (!isNaN(parseFloat(floatParseText))) {
                    return { type: 'number', value: floatText }
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
            let c = this._get(i);
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
            let cc = this._text.substr(this._position, 2);
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
}

python.Error = class extends Error {
    constructor(message) {
        super(message);
        this.name = 'Error loading Python module.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.Parser = python.Parser; 
}