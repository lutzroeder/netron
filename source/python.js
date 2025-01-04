
// Experimental Python Execution

const python = {};

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
                    if (items instanceof Map) {
                        items = Array.from(items);
                    } else if (!Array.isArray(items)) {
                        items = Object.entries(items);
                    }
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
            __delitem__(key) {
                this.delete(key);
            }
            get(key, defaultValue) {
                return super.get(key) || defaultValue;
            }
            setdefault(key, defaultValue) {
                if (this.has(key)) {
                    return this.get(key);
                }
                const value = defaultValue || null;
                this.set(key, value);
                return value;
            }
            pop(key) {
                if (this.__contains__(key)) {
                    const v = this.__getitem__(key);
                    this.__delitem__(key);
                    return v;
                }
                return null;
            }
            items() {
                return Array.from(this);
            }
            update(other) {
                for (const [key, value] of other) {
                    this.set(key, value);
                }
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
        this.builtins = builtins;
        this._registry.set('__builtin__', builtins);
        this.registerType('builtins.type', class {
            constructor(...args) {
                if (args.length === 1) {
                    const [obj] = args;
                    if (obj === null) {
                        /* eslint-disable no-constructor-return */
                        return builtins.NoneType;
                        /* eslint-enable no-constructor-return */
                    }
                    if (obj && obj.__class__) {
                        /* eslint-disable no-constructor-return */
                        return obj.__class__;
                        /* eslint-enable no-constructor-return */
                    }
                    throw new python.Error(`Unknown type '${obj}'`);
                }
                if (args.length === 3) {
                    const [name, bases, body] = args;
                    const cls = bases.length > 0 ? class extends bases[0] {} : class {};
                    execution.registerType(name, cls);
                    for (const [key, value] of body) {
                        cls[key] = value;
                    }
                    /* eslint-disable no-constructor-return */
                    return cls;
                    /* eslint-enable no-constructor-return */
                }
                throw new python.Error(`Invalid 'builtins.dict' argument count.`);
            }
        }).__class__ = builtins.type;
        this.registerType('builtins.module', module);
        this.registerType('builtins.method', class {});
        this.registerType('builtins.function', class {
            constructor(code, globals, name) {
                this.__code__ = code;
                this.__globals__ = globals;
                this.__name__ = name;
            }
        });
        this.registerType('builtins.classmethod', class {});
        this.registerType('builtins.code', class {});
        this.import('builtins');
        this.registerType('builtins.builtin_function_or_method', class {});
        const typing = this.register('typing');
        this.typing = typing;
        const operator = this.register('operator');
        this.register('_codecs');
        this.register('argparse');
        this._enum = this.register('enum');
        this.register('collections');
        const copy = this.register('copy');
        this.register('copy_reg');
        const ast = this.register('ast');
        this.ast = ast;
        this.register('cuml');
        const cloudpickle = this.register('cloudpickle');
        const datetime = this.register('datetime');
        this.register('gensim');
        const io = this.register('io');
        const joblib = this.register('joblib');
        const jax = this.register('jax');
        this.register('jax.numpy');
        this.register('jax._src.array');
        this.register('jax._src.device_array');
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
        this.register('numpy.core.multiarray');
        this.register('numpy.core._multiarray_umath');
        this.register('numpy.matrixlib.defmatrix');
        const pandas = this.register('pandas');
        this.register('pandas.indexes.base');
        this.register('pandas.indexes.range');
        this.register('pandas._libs.tslib');
        this.register('pandas._libs.internals');
        const pickle = this.register('pickle');
        const shap = this.register('shap');
        this.register('shap.explainers.linear');
        const sklearn = this.register('sklearn');
        this.register('sklearn.externals.joblib.numpy_pickle');
        const torch = this.register('torch');
        this.torch = torch;
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
        const sys = this.register('sys');
        sys.modules = this._modules;
        this.register('xgboost');
        this.registerType('ast.AST', class {});
        this.registerType('ast.mod', class extends ast.AST {});
        this.registerType('ast.expr', class extends ast.AST {});
        this.registerType('ast.unaryop', class extends ast.AST {});
        this.registerType('ast.binop', class extends ast.AST {});
        this.registerType('ast.operator', class extends ast.AST {});
        this.registerType('ast.boolop', class extends ast.AST {});
        this.registerType('ast.cmpop', class extends ast.AST {});
        this.registerType('ast.stmt', class extends ast.AST {});
        this.registerType('ast.excepthandler', class extends ast.AST {});
        this.registerType('ast.keyword', class extends ast.AST {
            constructor(arg, value) {
                super();
                this.arg = arg;
                this.value = value;
            }
        });
        this.registerType('ast.alias', class extends ast.AST {
            constructor(name, asname) {
                super();
                this.name = name;
                this.asname = asname;
            }
        });
        this.registerType('ast.Name', class extends ast.expr {
            constructor(id, ctx) {
                super();
                this.id = id;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.Constant', class extends ast.expr {
            constructor(value) {
                super();
                this.value = value;
            }
        });
        this.registerType('ast.Ellipsis', class extends ast.Constant {
            constructor() {
                super(builtins.ellipsis);
            }
        });
        this.registerType('ast.Starred', class extends ast.expr {
            constructor(value, ctx) {
                super();
                this.value = value;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.List', class extends ast.expr {
            constructor(elts, ctx) {
                super();
                this.elts = elts;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.Set', class extends ast.expr {
            constructor(elts) {
                super();
                this.elts = elts;
            }
        });
        this.registerType('ast.Tuple', class extends ast.expr {
            constructor(elts, ctx) {
                super();
                this.elts = elts;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.Dict', class extends ast.expr {
            constructor(keys, values) {
                super();
                this.keys = keys;
                this.values = values;
            }
        });
        this.registerType('ast.ListComp', class extends ast.expr {
            constructor(elt, generators) {
                super();
                this.elt = elt;
                this.generators = generators;
            }
        });
        this.registerType('ast.SetComp', class extends ast.expr {
            constructor(elt, generators) {
                super();
                this.elt = elt;
                this.generators = generators;
            }
        });
        this.registerType('ast.GeneratorExp', class extends ast.expr {
            constructor(elt, generators) {
                super();
                this.elt = elt;
                this.generators = generators;
            }
        });
        this.registerType('ast.DictComp', class extends ast.expr {
            constructor(key, value, generators) {
                super();
                this.key = key;
                this.value = value;
                this.generators = generators;
            }
        });
        this.registerType('ast.comprehension', class extends ast.AST {
            constructor(target, iter, ifs, is_async) {
                super();
                this.target = target;
                this.iter = iter;
                this.ifs = ifs;
                this.is_async = is_async;
            }
        });
        this.registerType('ast.Subscript', class extends ast.expr {
            constructor(value, slice, ctx) {
                super();
                this.value = value;
                this.slice = slice;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.UnaryOp', class extends ast.expr {
            constructor(op, operand) {
                super();
                this.op = op;
                this.operand = operand;
            }
        });
        this.registerType('ast.UAdd', class extends ast.unaryop {});
        this.registerType('ast.USub', class extends ast.unaryop {});
        this.registerType('ast.Not', class extends ast.unaryop {});
        this.registerType('ast.Invert', class extends ast.unaryop {});
        this.registerType('ast.BinOp', class extends ast.expr {
            constructor(left, op, right) {
                super();
                this.left = left;
                this.op = op;
                this.right = right;
            }
        });
        this.registerType('ast.Add', class extends ast.operator {});
        this.registerType('ast.Sub', class extends ast.operator {});
        this.registerType('ast.Mult', class extends ast.operator {});
        this.registerType('ast.Div', class extends ast.operator {});
        this.registerType('ast.FloorDiv', class extends ast.operator {});
        this.registerType('ast.Mod', class extends ast.operator {});
        this.registerType('ast.Pow', class extends ast.operator {});
        this.registerType('ast.LShift', class extends ast.operator {});
        this.registerType('ast.RShift', class extends ast.operator {});
        this.registerType('ast.BitOr', class extends ast.operator {});
        this.registerType('ast.BitXor', class extends ast.operator {});
        this.registerType('ast.BitAnd', class extends ast.operator {});
        this.registerType('ast.MatMult', class extends ast.operator {});
        this.registerType('ast.BoolOp', class extends ast.expr {
            constructor(op, values) {
                super();
                this.op = op;
                this.values = values;
            }
        });
        this.registerType('ast.And', class extends ast.boolop {});
        this.registerType('ast.Or', class extends ast.boolop {});
        this.registerType('ast.Compare', class extends ast.expr {
            constructor(left, ops, comparators) {
                super();
                this.left = left;
                this.ops = ops;
                this.comparators = comparators;
            }
        });
        this.registerType('ast.Eq', class extends ast.cmpop {});
        this.registerType('ast.NotEq', class extends ast.cmpop {});
        this.registerType('ast.Lt', class extends ast.cmpop {});
        this.registerType('ast.LtE', class extends ast.cmpop {});
        this.registerType('ast.Gt', class extends ast.cmpop {});
        this.registerType('ast.GtE', class extends ast.cmpop {});
        this.registerType('ast.Is', class extends ast.cmpop {});
        this.registerType('ast.IsNot', class extends ast.cmpop {});
        this.registerType('ast.In', class extends ast.cmpop {});
        this.registerType('ast.NotIn', class extends ast.cmpop {});
        this.registerType('ast.Call', class extends ast.expr {
            constructor(func, args, keywords) {
                super();
                this.func = func;
                this.args = args;
                this.keywords = keywords || [];
            }
        });
        this.registerType('ast.Attribute', class extends ast.expr {
            constructor(value, attr, ctx) {
                super();
                this.value = value;
                this.attr = attr;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.Lambda', class extends ast.expr {
            constructor(args, body) {
                super();
                this.args = args;
                this.body = body;
            }
        });
        this.registerType('ast.IfExp', class extends ast.expr {
            constructor(test, body, orelse) {
                super();
                this.test = test;
                this.body = body;
                this.orelse = orelse;
            }
        });
        this.registerType('ast.NamedExpr', class extends ast.expr {
            constructor(target, value) {
                super();
                this.target = target;
                this.value = value;
            }
        });
        this.registerType('ast.Yield', class extends ast.expr {
            constructor(value) {
                super();
                this.value = value;
            }
        });
        this.registerType('ast.YieldFrom', class extends ast.expr {
            constructor(value) {
                super();
                this.value = value;
            }
        });
        this.registerType('ast.Assign', class extends ast.stmt {
            constructor(targets, value, ctx) {
                super();
                this.targets = targets;
                this.value = value;
                if (ctx) {
                    this.ctx = ctx;
                }
            }
        });
        this.registerType('ast.AnnAssign', class extends ast.stmt {
            constructor(target, annotation, value, simple) {
                super();
                this.target = target;
                this.annotation = annotation;
                this.value = value;
                this.simple = simple;
            }
        });
        this.registerType('ast.AugAssign', class extends ast.stmt {
            constructor(target, op, value) {
                super();
                this.target = target;
                this.op = op;
                this.value = value;
            }
        });
        this.registerType('ast.If', class extends ast.stmt {
            constructor(test, body, orelse) {
                super();
                this.test = test;
                this.body = body;
                this.orelse = orelse;
            }
        });
        this.registerType('ast.For', class extends ast.stmt {
            constructor(target, iter, body, orelse /*, type_comment */) {
                super();
                this.target = target;
                this.iter = iter;
                this.body = body;
                this.orelse = orelse;
            }
        });
        this.registerType('ast.While', class extends ast.stmt {
            constructor(test, body, orelse /*, type_comment */) {
                super();
                this.test = test;
                this.body = body;
                this.orelse = orelse;
            }
        });
        this.registerType('ast.Del', class extends ast.stmt {
            constructor(targets) {
                super();
                this.targets = targets;
            }
        });
        this.registerType('ast.Return', class extends ast.stmt {
            constructor(value) {
                super();
                this.value = value;
            }
        });
        this.registerType('ast.Try', class extends ast.stmt {
            constructor(body, handlers, orelse, finalbody) {
                super();
                this.body = body;
                this.handlers = handlers;
                this.orelse = orelse;
                this.finalbody = finalbody;
            }
        });
        this.registerType('ast.ExceptHandler', class extends ast.excepthandler {
            constructor(type, name, body) {
                super();
                this.type_ = type;
                this.name = name;
                this.body = body;
            }
        });
        this.registerType('ast.ClassDef', class extends ast.stmt {
            constructor(name, bases, keywords, body, decorator_list, type_params) {
                super();
                this.name = name;
                this.bases = bases;
                this.keywords = keywords;
                this.body = body;
                this.decorator_list = decorator_list;
                this.type_params = type_params;
            }
        });
        this.registerType('ast.FunctionDef', class extends ast.stmt {
            constructor(name, args, body, decorator_list, returns, type_comment, type_params) {
                super();
                this.name = name;
                this.args = args;
                this.body = body;
                this.decorator_list = decorator_list;
                this.returns = returns;
                this.type_comment = type_comment;
                this.type_params = type_params;
            }
        });
        this.registerType('ast.arguments', class extends ast.AST {
            constructor(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults) {
                super();
                this.posonlyargs = posonlyargs;
                this.args = args;
                this.vararg = vararg;
                this.kwonlyargs = kwonlyargs;
                this.kw_defaults = kw_defaults;
                this.kwarg = kwarg;
                this.defaults = defaults;
            }
        });
        this.registerType('ast.arg', class extends ast.AST {
            constructor(arg, annotation, type_comment) {
                super();
                this.arg = arg;
                this.annotation = annotation;
                this.type_comment = type_comment;
            }
        });
        this.registerType('ast.Import', class extends ast.stmt {
            constructor(names) {
                super();
                this.names = names;
            }
        });
        this.registerType('ast.ImportFrom', class extends ast.stmt {
            constructor(module, names, level) {
                super();
                this.module = module;
                this.names = names;
                this.level = level;
            }
        });
        this.registerType('ast.Assert', class extends ast.stmt {
            constructor(test, msg) {
                super();
                this.test = test;
                this.msg = msg;
            }
        });
        this.registerType('ast.Raise', class extends ast.stmt {
            constructor(exc, cause) {
                super();
                this.exc = exc;
                this.cause = cause;
            }
        });
        this.registerType('ast.With', class extends ast.stmt {
            constructor(items, body, type_comment) {
                super();
                this.items = items;
                this.body = body;
                this.type_comment = type_comment;
            }
        });
        this.registerType('ast.withitem', class extends ast.AST {
            constructor(context_expr, optional_vars) {
                super();
                this.context_expr = context_expr;
                this.optional_vars = optional_vars;
            }
        });
        this.registerType('ast.Global', class extends ast.stmt {
            constructor(names) {
                super();
                this.names = names;
            }
        });
        this.registerType('ast.Nonlocal', class extends ast.stmt {
            constructor(names) {
                super();
                this.names = names;
            }
        });
        this.registerType('ast.Continue', class extends ast.stmt {});
        this.registerType('ast.Break', class extends ast.stmt {});
        this.registerType('ast.Pass', class extends ast.stmt {});
        this.registerType('ast.Await', class extends ast.stmt {
            constructor(value) {
                super();
                this.value = value;
            }
        });
        this.registerType('ast.Module', class extends ast.mod {
            constructor(body, type_ignores) {
                super();
                this.body = body;
                this.type_ignores = type_ignores;
            }
        });
        this.registerFunction('ast.parse', (source, filename, debug) => {
            const parser =  new ast._Parser(source, filename, debug);
            return parser.parse();
        });
        this.registerType('ast._Parser', class {
            constructor(text, file, debug) {
                this._tokenizer = new ast._Tokenizer(text, file);
                this._debug = debug;
                ast._Parser._precedence = ast._Parser._precedence || {
                    'or': 2, 'and': 3, 'not' : 4,
                    'in': 5, 'instanceof': 5, 'is': 5, '<': 5, '>': 5, '<=': 5, '>=': 5, '<>': 5, '==': 5, '!=': 5,
                    '|': 6, '^' : 7, '&' : 8,
                    '<<': 9, '>>': 9, '+': 10, '-': 10, '*': 11, '@': 11, '/': 11, '//': 11, '%': 11,
                    // '+': 12, '-': 12,
                    '~': 13, '**': 14
                };
            }
            parse() {
                const position = this._position();
                const body = [];
                while (!this._tokenizer.match('eof')) {
                    const statement = this._statement();
                    if (statement) {
                        body.push(statement);
                        continue;
                    }
                    if (this._tokenizer.eat('\n') || this._tokenizer.eat(';') || this._tokenizer.peek().type === 'eof') {
                        continue;
                    }
                    if (this._tokenizer.eat('indent') && this._tokenizer.peek().type === 'eof') {
                        continue;
                    }
                    throw new python.Error(`Unsupported statement ${this._location()}`);
                }
                const node = new ast.Module(body);
                this._mark(node, position);
                return node;
            }
            _suite() {
                const body = [];
                let statement = null;
                if (this._tokenizer.eat('\n')) {
                    if (this._tokenizer.eat('indent')) {
                        while (!this._tokenizer.eat('eof') && !this._tokenizer.eat('dedent')) {
                            if (this._tokenizer.eat(';')) {
                                continue;
                            }
                            statement = this._statement();
                            if (statement) {
                                body.push(statement);
                                continue;
                            }
                            if (this._tokenizer.eat('\n')) {
                                continue;
                            }
                            if (this._tokenizer.match('dedent') || this._tokenizer.match('eof')) {
                                continue;
                            }
                            throw new python.Error(`Empty statement ${this._location()}`);
                        }
                    }
                } else if (!this._tokenizer.eat('eof')) {
                    while (!this._tokenizer.match('\n') && !this._tokenizer.match('eof') && !this._tokenizer.match('dedent')) {
                        if (this._tokenizer.eat(';')) {
                            continue;
                        }
                        statement = this._statement();
                        if (statement) {
                            body.push(statement);
                            continue;
                        }
                        throw new python.Error(`Empty statement ${this._location()}`);
                    }
                    this._tokenizer.eat('\n');
                }
                return body;
            }
            _statement() {
                let node = null;
                let position = this._position();
                if (this._eat('id', 'break')) {
                    const node = new ast.Break();
                    return this._mark(node, position);
                }
                if (this._eat('id', 'continue')) {
                    const node = new ast.Continue();
                    return this._mark(node, position);
                }
                if (this._eat('id', 'return')) {
                    const value = this._expression(-1, [], true);
                    const node = new ast.Return(value);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'raise')) {
                    let exc = this._expression(-1, ['from']);
                    let cause = null;
                    if (this._tokenizer.eat('id', 'from')) {
                        cause = this._expression();
                    } else if (this._tokenizer.eat(',')) {
                        exc = [exc];
                        exc.push(this._expression());
                        if (this._tokenizer.eat(',')) {
                            exc.push(this._expression());
                        }
                    }
                    node = new ast.Raise(exc, cause);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'assert')) {
                    const test = this._expression(-1, [',']);
                    let msg = null;
                    if (this._tokenizer.eat(',')) {
                        msg = this._expression();
                    }
                    node = new ast.Assert(test, msg);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'global')) {
                    const names = [];
                    do {
                        const name = this._name(true);
                        names.push(name.id);
                    }
                    while (this._tokenizer.eat(','));
                    const node = new ast.Global(names);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'nonlocal')) {
                    const names = [];
                    do {
                        const name = this._name(true);
                        names.push(name.id);
                    }
                    while (this._tokenizer.eat(','));
                    const node = new ast.Nonlocal(names);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'import')) {
                    const names = [];
                    do {
                        const name = this._dottedName();
                        let asname = null;
                        if (this._tokenizer.eat('id', 'as')) {
                            asname = this._name(true).id;
                        }
                        const node = new ast.alias(name, asname);
                        names.push(node);
                    }
                    while (this._tokenizer.eat(','));
                    const node = new ast.Import(names);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'from')) {
                    let level = 0;
                    const dots = this._tokenizer.peek();
                    if (dots && Array.from(dots.type).every((c) => c === '.')) {
                        this._eat(dots.type);
                        level = Array.from(dots.type).length;
                    }
                    const module = this._dottedName();
                    this._tokenizer.expect('id', 'import');
                    const names = [];
                    const close = this._tokenizer.eat('(');
                    do {
                        const name = this._name(true).id;
                        let asname = null;
                        if (this._tokenizer.eat('id', 'as')) {
                            asname = this._name(true).id;
                        }
                        const node = new ast.alias(name, asname);
                        names.push(node);
                    }
                    while (this._tokenizer.eat(','));
                    if (close) {
                        this._tokenizer.expect(')');
                    }
                    const node = new ast.ImportFrom(module, names, level);
                    return this._mark(node, position);
                }
                const decorator_list = this._decorator();
                position = this._position();
                if (this._eat('id', 'class')) {
                    const name = this._name(true);
                    const bases = [];
                    if (this._tokenizer.eat('(')) {
                        while (!this._tokenizer.eat(')')) {
                            if (this._tokenizer.eat('\n')) {
                                continue;
                            }
                            const expression = this._expression(-1, [], false);
                            if (expression === null) {
                                throw new python.Error(`Expected expression ${this._location()}`);
                            }
                            bases.push(expression);
                            if (!this._tokenizer.eat(',')) {
                                this._tokenizer.eat('\n');
                                this._tokenizer.expect(')');
                                break;
                            }
                        }
                    }
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    const node = new ast.ClassDef(name.id, bases, null, body, decorator_list, null);
                    return this._mark(node, position);
                }
                const async = this._eat('id', 'async') !== null;
                if (async &&
                    !this._tokenizer.match('id', 'def') &&
                    !this._tokenizer.match('id', 'with') &&
                    !this._tokenizer.match('id', 'for')) {
                    throw new python.Error(`Expected 'def', 'with' or 'for' ${this._location()}`);
                }
                if (this._eat('id', 'def')) {
                    const name = this._name(true);
                    this._tokenizer.expect('(');
                    const args = this._arguments(')');
                    let returns = null;
                    if (this._tokenizer.eat('->')) {
                        returns = this._type();
                    }
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    const node = new ast.FunctionDef(name.id, args, body, decorator_list, returns, null, null);
                    if (async) {
                        node.async = async;
                    }
                    return this._mark(node, position);
                }
                if (decorator_list && decorator_list.length > 0) {
                    throw new python.Error('Unexpected decorator.');
                }
                if (this._eat('id', 'del')) {
                    const targets = this._expression(-1, [], true);
                    node = new ast.Del(targets);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'if')) {
                    const test = this._expression();
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    const node = new ast.If(test, body);
                    let current = node;
                    this._tokenizer.eat('\n');
                    while (this._tokenizer.eat('id', 'elif')) {
                        const test = this._expression();
                        this._tokenizer.expect(':');
                        const body = this._suite();
                        current.orelse = new ast.If(test, body);
                        current = current.orelse;
                        this._tokenizer.eat('\n');
                    }
                    if (this._tokenizer.eat('id', 'else')) {
                        this._tokenizer.expect(':');
                        current.orelse = this._suite();
                    }
                    return this._mark(node, position);
                }
                if (this._eat('id', 'while')) {
                    const test = this._expression();
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    let orelse = null;
                    if (this._tokenizer.eat('id', 'else')) {
                        this._tokenizer.expect(':');
                        orelse = this._suite();
                    }
                    const node = new ast.While(test, body, orelse);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'pass')) {
                    const node = new ast.Pass();
                    return this._mark(node, position);
                }
                if (this._eat('id', 'for')) {
                    let target = this._expression(-1, ['in']);
                    while (this._tokenizer.eat(',')) {
                        if (target instanceof ast.Tuple === false) {
                            target = new ast.Tuple([target]);
                        }
                        if (this._tokenizer.match('id', 'in')) {
                            target.elts.push({});
                            break;
                        }
                        target.elts.push(this._expression(-1, ['in']));
                    }
                    this._tokenizer.expect('id', 'in');
                    let iter = this._expression();
                    while (this._tokenizer.eat(',')) {
                        if (iter.type !== 'tuple') {
                            iter = new ast.Tuple([iter]);
                        }
                        if (this._tokenizer.match(':')) {
                            iter.elts.push({});
                            break;
                        }
                        iter.elts.push(this._expression(-1, ['in']));
                    }
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    let orelse = null;
                    if (this._tokenizer.eat('id', 'else')) {
                        this._tokenizer.expect(':');
                        orelse = this._suite();
                    }
                    const node = new ast.For(target, iter, body, orelse);
                    return this._mark(node, position);
                }
                if (this._eat('id', 'with')) {
                    const items = [];
                    do {
                        const context_expr = this._expression();
                        let optional_vars = null;
                        if (this._tokenizer.eat('id', 'as')) {
                            optional_vars = this._expression();
                        }
                        const node = new ast.withitem(context_expr, optional_vars);
                        items.push(node);
                    }
                    while (this._tokenizer.eat(','));
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    const node = new ast.With(items, body, null);
                    if (async) {
                        node.async = async;
                    }
                    return this._mark(node, position);
                }
                if (this._eat('id', 'try')) {
                    this._tokenizer.expect(':');
                    const body = this._suite();
                    const handlers = [];
                    let orelse = null;
                    let finalbody = null;
                    while (this._tokenizer.match('id', 'except')) {
                        this._tokenizer.expect('id', 'except');
                        const type = this._expression();
                        const name = this._tokenizer.eat('id', 'as') ? this._expression() : null;
                        this._tokenizer.expect(':');
                        const body = this._suite();
                        const except = new ast.ExceptHandler(type, name, body);
                        handlers.push(except);
                    }
                    if (this._tokenizer.match('id', 'else')) {
                        this._tokenizer.expect('id', 'else');
                        this._tokenizer.expect(':');
                        orelse = this._suite();
                    }
                    if (this._tokenizer.match('id', 'finally')) {
                        this._tokenizer.expect('id', 'finally');
                        this._tokenizer.expect(':');
                        finalbody = this._suite();
                    }
                    const node = new ast.Try(body, handlers, orelse, finalbody);
                    return this._mark(node, position);
                }
                const expr = this._expression(-1, [], true);
                if (expr) {
                    if (expr instanceof ast.Name && this._tokenizer.eat(':')) {
                        const position = this._position();
                        const annotation = this._expression(-1, ['=']);
                        let value = null;
                        if (this._tokenizer.eat('=')) {
                            value = this._expression();
                        }
                        node = new ast.AnnAssign(expr, annotation, value);
                        return this._mark(node, position);
                    }
                    switch (expr.__class__.__name__) {
                        case 'AnnAssign':
                        case 'Assert':
                        case 'Assign':
                        case 'Attribute':
                        case 'AugAssign':
                        case 'Await':
                        case 'BinOp':
                        case 'Call':
                        case 'Compare':
                        case 'Constant':
                        case 'Ellipsis':
                        case 'For':
                        case 'If':
                        case 'Lambda':
                        case 'List':
                        case 'Name':
                        case 'NamedExpr':
                        case 'Raise':
                        case 'Subscript':
                        case 'Tuple':
                        case 'Yield':
                            return expr;
                        default:
                            throw new python.Error(`Unhandled expression ${this._location()}`);
                    }
                }
                return null;
            }
            _expression(minPrecedence, terminal, tuple) {
                minPrecedence = minPrecedence || -1;
                const terminalSet = new Set(terminal);
                const stack = [];
                for (;;) {
                    let position = this._position();
                    let node = null;
                    const token = this._tokenizer.peek();
                    if (stack.length === 1 && terminalSet.has(token.value)) {
                        break;
                    }
                    const precedence = ast._Parser._precedence[token.value];
                    if (precedence) {
                        if (precedence >= minPrecedence) {
                            this._tokenizer.read();
                            if (token.value === 'not' && this._tokenizer.eat('id', 'in')) {
                                token.value = 'not in';
                            } else if (token.value === 'is' && this._tokenizer.eat('id', 'not')) {
                                token.value = 'is not';
                            }
                            if (stack.length > 0) {
                                let op = null;
                                switch (token.value) {
                                    case '+':  op = new ast.Add(); break;
                                    case '-':  op = new ast.Sub(); break;
                                    case '*':  op = new ast.Mult(); break;
                                    case '/':  op = new ast.Div(); break;
                                    case '//': op = new ast.FloorDiv(); break;
                                    case '**': op = new ast.Pow(); break;
                                    case '@':  op = new ast.MatMult(); break;
                                    case '&':  op = new ast.BitAnd(); break;
                                    case '^':  op = new ast.BitXor(); break;
                                    case '|':  op = new ast.BitOr(); break;
                                    case '%':  op = new ast.Mod(); break;
                                    case '>>': op = new ast.RShift(); break;
                                    case '<<': op = new ast.LShift(); break;
                                    default: break;
                                }
                                if (op) {
                                    const left = stack.pop();
                                    const right = this._expression(precedence, terminal, tuple === true ? true : false);
                                    node = new ast.BinOp(left, op, right);
                                } else {
                                    switch (token.value) {
                                        case '==': op = new ast.Eq(); break;
                                        case '!=': op = new ast.NotEq(); break;
                                        case '>=': op = new ast.GtE(); break;
                                        case '<=': op = new ast.LtE(); break;
                                        case '<':  op = new ast.Lt(); break;
                                        case '>':  op = new ast.Gt(); break;
                                        case 'is': op = new ast.Is(); break;
                                        case 'is not': op = new ast.IsNot(); break;
                                        case 'in': op = new ast.In(); break;
                                        case 'not in': op = new ast.NotIn(); break;
                                        default: break;
                                    }
                                    const left = stack.pop();
                                    const comparator = this._expression(precedence, terminal, tuple === true ? true : false);
                                    node = new ast.Compare(left, [op], [comparator]);
                                }
                            } else if (token.value === '*') {
                                const value =  this._expression(precedence, terminal, tuple === true ? true : false);
                                node = new ast.Starred(value);
                            } else if (token.value === '**') {
                                const value =  this._expression(precedence, terminal, tuple === true ? true : false);
                                node = new ast.keyword(null, value);
                            } else {
                                let op = null;
                                switch (token.value) {
                                    case '-': op = new ast.USub(); break;
                                    case '+': op = new ast.UAdd(); break;
                                    case '~': op = new ast.Invert(); break;
                                    case 'not': op = new ast.Not(); break;
                                    default: throw new python.Error(`Unsupported unary operator ${token.value} ${this._location()}`);
                                }
                                const operand =  this._expression(precedence, terminal, tuple === true ? true : false);
                                node = new ast.UnaryOp(op, operand);
                                node = this._mark(node, position);
                            }
                            stack.push(node);
                            continue;
                        }
                    }
                    if (this._tokenizer.eat(':=')) {
                        const target = stack.pop();
                        const value = this._expression(-1, terminal, tuple === false ? false : true);
                        const node = new ast.NamedExpr(target, value);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._tokenizer.eat('=')) {
                        const position = this._position();
                        const targets = stack.pop();
                        const value = this._expression(-1, terminal, tuple === false ? false : true);
                        const node = new ast.Assign([targets], value);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    let op = null;
                    switch (token.type) {
                        case '+=':  op = new ast.Add(); break;
                        case '-=':  op = new ast.Sub(); break;
                        case '**=': op = new ast.Pow(); break;
                        case '*=':  op = new ast.Mult(); break;
                        case '//=': op = new ast.FloorDiv(); break;
                        case '/=':  op = new ast.Div(); break;
                        case '&=':  op = new ast.BitAnd(); break;
                        case '%=':  op = new ast.Mod(); break;
                        case '^=':  op = new ast.BitXor(); break;
                        case '<<=': op = new ast.LShift(); break;
                        case '>>=': op = new ast.RShift(); break;
                        case '|=':  op = new ast.BitOr(); break;
                        case '@=':  op = new ast.MatMul(); break;
                        default: break;
                    }
                    if (op) {
                        this._tokenizer.expect(token.type);
                        const target = stack.pop();
                        const value = this._expression(-1, terminal, true);
                        const node = new ast.AugAssign(target, op, value);
                        stack.push(node);
                        continue;
                    }
                    position = this._position();
                    if (this._eat('id', 'if')) {
                        const body = stack.pop();
                        const test = this._expression();
                        this._tokenizer.expect('id', 'else');
                        const orelse = this._expression();
                        const node = new ast.IfExp(test, body, orelse);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._tokenizer.match('id', 'for') || this._tokenizer.match('id', 'async')) {
                        throw new python.Error('Not implemented.');
                    }
                    if (this._eat('id', 'lambda')) {
                        const args = this._arguments(':');
                        const body = this._expression(-1, terminal, false);
                        const node = new ast.Lambda(args, body);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._eat('id', 'yield')) {
                        if (this._tokenizer.eat('id', 'from')) {
                            const value = this._expression(-1, [], true);
                            node = new ast.YieldFrom(value);
                            stack.push(node);
                        } else {
                            const value = [];
                            do {
                                value.push(this._expression(-1, [], false));
                            }
                            while (this._tokenizer.eat(','));
                            node = new ast.Yield(value);
                            stack.push(node);
                        }
                        continue;
                    }
                    if (this._eat('id', 'await')) {
                        const value = this._expression(minPrecedence, terminal, tuple);
                        const node = new ast.Await(value);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._eat('.')) {
                        const value = stack.pop();
                        const attr = this._name().id;
                        const node = new ast.Attribute(value, attr);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._tokenizer.peek().type === '(') {
                        const position = this._position();
                        const args = [];
                        const keywords = [];
                        this._tokenizer.expect('(');
                        while (!this._tokenizer.eat(')')) {
                            if (this._tokenizer.eat('\n')) {
                                continue;
                            }
                            const expr = this._expression(-1, [], false);
                            if (expr === null) {
                                throw new python.Error(`Expected expression ${this._location()}`);
                            }
                            if (expr instanceof ast.Assign && expr.targets.length === 1) {
                                const [target] = expr.targets;
                                if (target instanceof ast.Name === false) {
                                    throw new python.Error(`Expected name ${this._location()}`);
                                }
                                const keyword = new ast.keyword(target.id, expr.value);
                                keywords.push(keyword);
                            } else {
                                args.push(expr);
                            }
                            if (!this._tokenizer.eat(',')) {
                                this._tokenizer.eat('\n');
                                this._tokenizer.expect(')');
                                break;
                            }
                        }
                        if (stack.length === 0 && keywords.length === 0) {
                            if (args.length === 1) {
                                [node] = args;
                            } else {
                                node = new ast.Tuple(args);
                                this._mark(node, position);
                            }
                        } else {
                            const func = stack.pop();
                            node = new ast.Call(func, args, keywords);
                            this._mark(node, position);
                        }
                        stack.push(node);
                        continue;
                    }
                    if (this._tokenizer.peek().type === '[') {
                        if (stack.length === 0) {
                            stack.push(this._expressions());
                        } else {
                            const value = stack.pop();
                            const elts = this._slice();
                            node = new ast.Subscript(value, elts);
                            stack.push(node);
                        }
                        continue;
                    }
                    if (this._tokenizer.peek().type === '{') {
                        const elts = [];
                        const keys = [];
                        const values = [];
                        this._tokenizer.expect('{');
                        let dict = true;
                        while (!this._tokenizer.eat('}')) {
                            const item = this._expression(-1, [], false);
                            if (item === null) {
                                throw new python.Error(`Expected expression ${this._location()}`);
                            }
                            if (!this._tokenizer.eat(':')) {
                                dict = false;
                            }
                            if (dict) {
                                const value = this._expression(-1, ['for'], false);
                                if (value === null) {
                                    throw new python.Error(`Expected expression ${this._location()}`);
                                }
                                if (this._eat('id', 'for')) {
                                    if (keys.length > 0 || values.length > 0 || elts.length > 0) {
                                        throw new python.Error(`Invalid list expression ${this._location()}`);
                                    }
                                    const target = this._expression(-1, ['in'], true);
                                    this._tokenizer.expect('id', 'in');
                                    const iter = this._expression(-1, ['for', 'if'], true);
                                    const ifs = [];
                                    while (this._tokenizer.eat('id', 'if')) {
                                        ifs.push(this._expression(-1, ['for', 'if']));
                                    }
                                    const comprehension = new ast.comprehension(target, iter, ifs /*, async */);
                                    const generators = [comprehension];
                                    this._tokenizer.expect('}');
                                    return new ast.DictComp(item, value, generators);
                                }
                                keys.push(item);
                                values.push(value);
                            } else {
                                elts.push(item);
                            }
                            this._tokenizer.eat(',');
                            this._tokenizer.eat('\n');
                            if (this._tokenizer.eat('}')) {
                                break;
                            }
                        }
                        if (keys.length !== values.length || (keys.length > 0 && elts.length > 0)) {
                            throw new python.Error(`Invalid set expression ${this._location()}`);
                        }
                        const node = elts.length > 0 ? new ast.Set(elts) : new ast.Dict(keys, values);
                        stack.push(node);
                        continue;
                    }
                    const literal = this._literal();
                    if (literal) {
                        if (stack.length > 0 && literal.type === 'number' && (literal.value.startsWith('-') || literal.value.startsWith('+'))) {
                            const op = literal.value < 0 ? new ast.Sub() : new ast.Add();
                            const left = stack.pop();
                            const right = new ast.Constant(Math.abs(literal.value));
                            node = new ast.BinOp(left, op, right);
                            stack.push(node);
                        } else if (stack.length === 1 && literal.type === 'string' && stack[0] instanceof ast.Constant && typeof stack[0].value === 'string') {
                            stack[0].value += literal.value.substring(1, literal.value.length - 1);
                        } else {
                            let value = literal.value;
                            if (literal.type === 'number') {
                                switch (value) {
                                    case 'inf': value = Infinity; break;
                                    case '-inf': value = -Infinity; break;
                                    default: value = Number(value); break;
                                }
                            } else if (literal.type === 'string') {
                                value = literal.value.substring(1, literal.value.length - 1);
                            } else {
                                throw new python.Error(`Invalid literal ${this._location()}`);
                            }
                            const node = new ast.Constant(value);
                            stack.push(node);
                        }
                        continue;
                    }
                    if (this._eat('id', 'False')) {
                        const node = new ast.Constant(false);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._eat('id', 'True')) {
                        const node = new ast.Constant(true);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._eat('id', 'None')) {
                        const node = new ast.Constant(null);
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    if (this._tokenizer.peek().keyword) {
                        break;
                    }
                    if (this._eat('...')) {
                        const node = new ast.Ellipsis();
                        this._mark(node, position);
                        stack.push(node);
                        continue;
                    }
                    const name = this._name();
                    if (name) {
                        stack.push(name);
                        continue;
                    }
                    if (tuple === true && stack.length === 1 && this._tokenizer.eat(',')) {
                        if (stack[0] instanceof ast.Tuple) {
                            [node] = stack;
                        } else {
                            const position = this._position();
                            const elts = [stack.pop()];
                            node = new ast.Tuple(elts);
                            this._mark(node, position);
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
                                node.elts.push(expression);
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
                    throw new python.Error(`Unexpected expression ${this._location()}`);
                }
                return null;
            }
            _decorator() {
                const list = [];
                while (this._tokenizer.eat('@')) {
                    const value = this._expression();
                    if (!value || (value instanceof ast.Call === false && value instanceof ast.Name === false && value instanceof ast.Attribute === false)) {
                        throw new python.Error(`Invalid decorator ${this._location()}`);
                    }
                    this._tokenizer.eat('\n');
                    list.push(value);
                }
                return list;
            }
            _expressions() {
                const elts = [];
                this._tokenizer.expect('[');
                while (!this._tokenizer.eat(']')) {
                    const expression = this._expression(-1, ['for']);
                    if (this._eat('id', 'for')) {
                        if (elts.length > 0) {
                            throw new python.Error(`Invalid list expression ${this._location()}`);
                        }
                        const target = this._expression(-1, ['in'], true);
                        this._tokenizer.expect('id', 'in');
                        const iter = this._expression(-1, ['for', 'if'], true);
                        const ifs = [];
                        while (this._tokenizer.eat('id', 'if')) {
                            ifs.push(this._expression(-1, ['for', 'if']));
                        }
                        const comprehension = new ast.comprehension(target, iter, ifs /*, async */);
                        const generators = [comprehension];
                        this._tokenizer.expect(']');
                        return new ast.ListComp(expression, generators);
                    }
                    if (expression === null) {
                        throw new python.Error(`Expected expression ${this._location()}`);
                    }
                    elts.push(expression);
                    this._tokenizer.eat(',');
                    while (this._tokenizer.eat('\n')) {
                        // continue
                    }
                    if (this._tokenizer.eat(']')) {
                        break;
                    }
                }
                return new ast.List(elts);
            }
            _slice() {
                let node = { type: '::' };
                let elts = [];
                const group = ['start', 'stop', 'step'];
                this._tokenizer.expect('[');
                while (!this._tokenizer.eat(']')) {
                    if (this._tokenizer.eat(':')) {
                        node[group.shift()] = new ast.List(elts);
                        elts = [];
                        continue;
                    }
                    if (this._tokenizer.eat(',')) {
                        // list.push({});
                        continue;
                    }
                    if (this._tokenizer.peek().type !== ']') {
                        const expression = this._expression();
                        if (expression === null) {
                            throw new python.Error(`Expected expression ${this._location()}`);
                        }
                        elts.push(expression);
                    }
                }
                if (elts.length > 0) {
                    node[group.shift()] = new ast.List(elts);
                }
                if (node.start && !node.stop && !node.step) {
                    node = node.start;
                }
                return node;
            }
            _name(required) {
                const token = this._tokenizer.peek();
                if (token.type === 'id' && !token.keyword) {
                    const position = this._position();
                    this._tokenizer.read();
                    const node = new ast.Name(token.value);
                    return this._mark(node, position);
                }
                if (required) {
                    throw new python.Error(`Invalid syntax ${this._location()}`);
                }
                return null;
            }
            _dottedName() {
                const list = [];
                do {
                    const name = this._name(true);
                    list.push(name.id);
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
                        throw new python.Error(`Expected type ${this._location()}`);
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
                const target = this._expression(-1, ['[', '=']);
                if (target) {
                    if (this._tokenizer.peek().value === '[') {
                        const elts = this._expressions();
                        return new ast.Subscript(target, elts);
                    }
                    return target;
                }
                return null;
            }
            _arguments(terminal) {
                let posonlyargs = [];
                let args = [];
                let vararg = null;
                const kwonlyargs = [];
                const kw_defaults = [];
                let kwarg = null;
                const defaults = [];
                let is_slash = false;
                let is_vararg = false; // '*'
                let is_kwarg = false; // '**'
                const read = (required) => {
                    const name = this._name(required);
                    if (name) {
                        const annotation = terminal !== ':' && this._tokenizer.eat(':') ? this._type() : null;
                        return new ast.arg(name.id, annotation, null);
                    }
                    return null;
                };
                while (!this._tokenizer.eat(terminal)) {
                    this._tokenizer.eat('\n');
                    if (this._tokenizer.eat('/')) {
                        if (is_slash || is_vararg || is_kwarg) {
                            throw new python.Error(`Invalid '/' in arguments ${this._location()}`);
                        }
                        is_slash = true;
                    } else if (this._tokenizer.eat('*')) {
                        if (is_vararg) {
                            throw new python.Error(`Multiple '*' arguments ${this._location()}`);
                        }
                        is_vararg = true;
                        const arg = read(false);
                        vararg = arg ? arg : vararg;
                    } else if (this._tokenizer.eat('**')) {
                        if (is_kwarg) {
                            throw new python.Error(`Multiple '**' arguments ${this._location()}`);
                        }
                        is_kwarg = true;
                        kwarg = read(true);
                    } else {
                        const arg = read(false);
                        if (!arg) {
                            this._tokenizer.expect(terminal);
                            break;
                        }
                        const default_value = this._tokenizer.eat('=') ? this._expression() : null;
                        if (!is_vararg && !is_kwarg) {
                            if (is_slash) {
                                args.push(arg);
                            } else {
                                posonlyargs.push(arg);
                            }
                            if (default_value !== null) {
                                defaults.push(default_value);
                            }
                        } else if (is_vararg && !is_kwarg) {
                            kwonlyargs.push(arg);
                            kw_defaults.push(default_value);
                        } else {
                            throw new python.Error(`Argument after '**' parameter ${this._location()}`);
                        }
                    }
                    this._tokenizer.eat('\n');
                    if (!this._tokenizer.eat(',')) {
                        this._tokenizer.expect(terminal);
                        break;
                    }
                }
                if (!is_slash) {
                    args = posonlyargs.concat(args);
                    posonlyargs = [];
                }
                return new ast.arguments(posonlyargs, args, vararg, kwonlyargs, kw_defaults, kwarg, defaults);
            }
            _eat(type, value) {
                if (this._tokenizer.match(type, value)) {
                    const position = this._position();
                    this._tokenizer.expect(type, value);
                    return position;
                }
                return null;
            }
            _mark(node, position) {
                node.location = position.location;
                node.filename = position.filename;
                node.lineno = position.lineno;
                node.col_offset = position.col_offset;
                node.end_lineno = this._tokenizer.lineno;
                node.end_col_offset = this._tokenizer.col_offset;
                return node;
            }
            _position() {
                return {
                    location: this._location(),
                    filename: this._tokenizer.filename,
                    lineno: this._tokenizer.lineno,
                    col_offset: this._tokenizer.col_offset
                };
            }
            _location() {
                return this._tokenizer.location();
            }
        });
        this.registerType('ast._Tokenizer', class {
            constructor(text, file) {
                this._text = text;
                this.filename = file;
                this.linepos = 0;
                this.lineno = 1;
                this._position = 0;
                this._token = { type: '', value: '' };
                this._brackets = 0;
                this._indentation = [];
                this._outdent = 0;
                if (!ast._Tokenizer._whitespace) {
                    ast._Tokenizer._whitespace = /[\u1680\u180e\u2000-\u200a\u202f\u205f\u3000\ufeff]/;
                    const identifierStartChars = '\xaa\xb5\xba\xc0-\xd6\xd8-\xf6\xf8-\u02c1\u02c6-\u02d1\u02e0-\u02e4\u02ec\u02ee\u0370-\u0374\u0376\u0377\u037a-\u037d\u0386\u0388-\u038a\u038c\u038e-\u03a1\u03a3-\u03f5\u03f7-\u0481\u048a-\u0527\u0531-\u0556\u0559\u0561-\u0587\u05d0-\u05ea\u05f0-\u05f2\u0620-\u064a\u066e\u066f\u0671-\u06d3\u06d5\u06e5\u06e6\u06ee\u06ef\u06fa-\u06fc\u06ff\u0710\u0712-\u072f\u074d-\u07a5\u07b1\u07ca-\u07ea\u07f4\u07f5\u07fa\u0800-\u0815\u081a\u0824\u0828\u0840-\u0858\u08a0\u08a2-\u08ac\u0904-\u0939\u093d\u0950\u0958-\u0961\u0971-\u0977\u0979-\u097f\u0985-\u098c\u098f\u0990\u0993-\u09a8\u09aa-\u09b0\u09b2\u09b6-\u09b9\u09bd\u09ce\u09dc\u09dd\u09df-\u09e1\u09f0\u09f1\u0a05-\u0a0a\u0a0f\u0a10\u0a13-\u0a28\u0a2a-\u0a30\u0a32\u0a33\u0a35\u0a36\u0a38\u0a39\u0a59-\u0a5c\u0a5e\u0a72-\u0a74\u0a85-\u0a8d\u0a8f-\u0a91\u0a93-\u0aa8\u0aaa-\u0ab0\u0ab2\u0ab3\u0ab5-\u0ab9\u0abd\u0ad0\u0ae0\u0ae1\u0b05-\u0b0c\u0b0f\u0b10\u0b13-\u0b28\u0b2a-\u0b30\u0b32\u0b33\u0b35-\u0b39\u0b3d\u0b5c\u0b5d\u0b5f-\u0b61\u0b71\u0b83\u0b85-\u0b8a\u0b8e-\u0b90\u0b92-\u0b95\u0b99\u0b9a\u0b9c\u0b9e\u0b9f\u0ba3\u0ba4\u0ba8-\u0baa\u0bae-\u0bb9\u0bd0\u0c05-\u0c0c\u0c0e-\u0c10\u0c12-\u0c28\u0c2a-\u0c33\u0c35-\u0c39\u0c3d\u0c58\u0c59\u0c60\u0c61\u0c85-\u0c8c\u0c8e-\u0c90\u0c92-\u0ca8\u0caa-\u0cb3\u0cb5-\u0cb9\u0cbd\u0cde\u0ce0\u0ce1\u0cf1\u0cf2\u0d05-\u0d0c\u0d0e-\u0d10\u0d12-\u0d3a\u0d3d\u0d4e\u0d60\u0d61\u0d7a-\u0d7f\u0d85-\u0d96\u0d9a-\u0db1\u0db3-\u0dbb\u0dbd\u0dc0-\u0dc6\u0e01-\u0e30\u0e32\u0e33\u0e40-\u0e46\u0e81\u0e82\u0e84\u0e87\u0e88\u0e8a\u0e8d\u0e94-\u0e97\u0e99-\u0e9f\u0ea1-\u0ea3\u0ea5\u0ea7\u0eaa\u0eab\u0ead-\u0eb0\u0eb2\u0eb3\u0ebd\u0ec0-\u0ec4\u0ec6\u0edc-\u0edf\u0f00\u0f40-\u0f47\u0f49-\u0f6c\u0f88-\u0f8c\u1000-\u102a\u103f\u1050-\u1055\u105a-\u105d\u1061\u1065\u1066\u106e-\u1070\u1075-\u1081\u108e\u10a0-\u10c5\u10c7\u10cd\u10d0-\u10fa\u10fc-\u1248\u124a-\u124d\u1250-\u1256\u1258\u125a-\u125d\u1260-\u1288\u128a-\u128d\u1290-\u12b0\u12b2-\u12b5\u12b8-\u12be\u12c0\u12c2-\u12c5\u12c8-\u12d6\u12d8-\u1310\u1312-\u1315\u1318-\u135a\u1380-\u138f\u13a0-\u13f4\u1401-\u166c\u166f-\u167f\u1681-\u169a\u16a0-\u16ea\u16ee-\u16f0\u1700-\u170c\u170e-\u1711\u1720-\u1731\u1740-\u1751\u1760-\u176c\u176e-\u1770\u1780-\u17b3\u17d7\u17dc\u1820-\u1877\u1880-\u18a8\u18aa\u18b0-\u18f5\u1900-\u191c\u1950-\u196d\u1970-\u1974\u1980-\u19ab\u19c1-\u19c7\u1a00-\u1a16\u1a20-\u1a54\u1aa7\u1b05-\u1b33\u1b45-\u1b4b\u1b83-\u1ba0\u1bae\u1baf\u1bba-\u1be5\u1c00-\u1c23\u1c4d-\u1c4f\u1c5a-\u1c7d\u1ce9-\u1cec\u1cee-\u1cf1\u1cf5\u1cf6\u1d00-\u1dbf\u1e00-\u1f15\u1f18-\u1f1d\u1f20-\u1f45\u1f48-\u1f4d\u1f50-\u1f57\u1f59\u1f5b\u1f5d\u1f5f-\u1f7d\u1f80-\u1fb4\u1fb6-\u1fbc\u1fbe\u1fc2-\u1fc4\u1fc6-\u1fcc\u1fd0-\u1fd3\u1fd6-\u1fdb\u1fe0-\u1fec\u1ff2-\u1ff4\u1ff6-\u1ffc\u2071\u207f\u2090-\u209c\u2102\u2107\u210a-\u2113\u2115\u2119-\u211d\u2124\u2126\u2128\u212a-\u212d\u212f-\u2139\u213c-\u213f\u2145-\u2149\u214e\u2160-\u2188\u2c00-\u2c2e\u2c30-\u2c5e\u2c60-\u2ce4\u2ceb-\u2cee\u2cf2\u2cf3\u2d00-\u2d25\u2d27\u2d2d\u2d30-\u2d67\u2d6f\u2d80-\u2d96\u2da0-\u2da6\u2da8-\u2dae\u2db0-\u2db6\u2db8-\u2dbe\u2dc0-\u2dc6\u2dc8-\u2dce\u2dd0-\u2dd6\u2dd8-\u2dde\u2e2f\u3005-\u3007\u3021-\u3029\u3031-\u3035\u3038-\u303c\u3041-\u3096\u309d-\u309f\u30a1-\u30fa\u30fc-\u30ff\u3105-\u312d\u3131-\u318e\u31a0-\u31ba\u31f0-\u31ff\u3400-\u4db5\u4e00-\u9fcc\ua000-\ua48c\ua4d0-\ua4fd\ua500-\ua60c\ua610-\ua61f\ua62a\ua62b\ua640-\ua66e\ua67f-\ua697\ua6a0-\ua6ef\ua717-\ua71f\ua722-\ua788\ua78b-\ua78e\ua790-\ua793\ua7a0-\ua7aa\ua7f8-\ua801\ua803-\ua805\ua807-\ua80a\ua80c-\ua822\ua840-\ua873\ua882-\ua8b3\ua8f2-\ua8f7\ua8fb\ua90a-\ua925\ua930-\ua946\ua960-\ua97c\ua984-\ua9b2\ua9cf\uaa00-\uaa28\uaa40-\uaa42\uaa44-\uaa4b\uaa60-\uaa76\uaa7a\uaa80-\uaaaf\uaab1\uaab5\uaab6\uaab9-\uaabd\uaac0\uaac2\uaadb-\uaadd\uaae0-\uaaea\uaaf2-\uaaf4\uab01-\uab06\uab09-\uab0e\uab11-\uab16\uab20-\uab26\uab28-\uab2e\uabc0-\uabe2\uac00-\ud7a3\ud7b0-\ud7c6\ud7cb-\ud7fb\uf900-\ufa6d\ufa70-\ufad9\ufb00-\ufb06\ufb13-\ufb17\ufb1d\ufb1f-\ufb28\ufb2a-\ufb36\ufb38-\ufb3c\ufb3e\ufb40\ufb41\ufb43\ufb44\ufb46-\ufbb1\ufbd3-\ufd3d\ufd50-\ufd8f\ufd92-\ufdc7\ufdf0-\ufdfb\ufe70-\ufe74\ufe76-\ufefc\uff21-\uff3a\uff41-\uff5a\uff66-\uffbe\uffc2-\uffc7\uffca-\uffcf\uffd2-\uffd7\uffda-\uffdc';
                    const identifierChars = '\u0300-\u036f\u0483-\u0487\u0591-\u05bd\u05bf\u05c1\u05c2\u05c4\u05c5\u05c7\u0610-\u061a\u0620-\u0649\u0672-\u06d3\u06e7-\u06e8\u06fb-\u06fc\u0730-\u074a\u0800-\u0814\u081b-\u0823\u0825-\u0827\u0829-\u082d\u0840-\u0857\u08e4-\u08fe\u0900-\u0903\u093a-\u093c\u093e-\u094f\u0951-\u0957\u0962-\u0963\u0966-\u096f\u0981-\u0983\u09bc\u09be-\u09c4\u09c7\u09c8\u09d7\u09df-\u09e0\u0a01-\u0a03\u0a3c\u0a3e-\u0a42\u0a47\u0a48\u0a4b-\u0a4d\u0a51\u0a66-\u0a71\u0a75\u0a81-\u0a83\u0abc\u0abe-\u0ac5\u0ac7-\u0ac9\u0acb-\u0acd\u0ae2-\u0ae3\u0ae6-\u0aef\u0b01-\u0b03\u0b3c\u0b3e-\u0b44\u0b47\u0b48\u0b4b-\u0b4d\u0b56\u0b57\u0b5f-\u0b60\u0b66-\u0b6f\u0b82\u0bbe-\u0bc2\u0bc6-\u0bc8\u0bca-\u0bcd\u0bd7\u0be6-\u0bef\u0c01-\u0c03\u0c46-\u0c48\u0c4a-\u0c4d\u0c55\u0c56\u0c62-\u0c63\u0c66-\u0c6f\u0c82\u0c83\u0cbc\u0cbe-\u0cc4\u0cc6-\u0cc8\u0cca-\u0ccd\u0cd5\u0cd6\u0ce2-\u0ce3\u0ce6-\u0cef\u0d02\u0d03\u0d46-\u0d48\u0d57\u0d62-\u0d63\u0d66-\u0d6f\u0d82\u0d83\u0dca\u0dcf-\u0dd4\u0dd6\u0dd8-\u0ddf\u0df2\u0df3\u0e34-\u0e3a\u0e40-\u0e45\u0e50-\u0e59\u0eb4-\u0eb9\u0ec8-\u0ecd\u0ed0-\u0ed9\u0f18\u0f19\u0f20-\u0f29\u0f35\u0f37\u0f39\u0f41-\u0f47\u0f71-\u0f84\u0f86-\u0f87\u0f8d-\u0f97\u0f99-\u0fbc\u0fc6\u1000-\u1029\u1040-\u1049\u1067-\u106d\u1071-\u1074\u1082-\u108d\u108f-\u109d\u135d-\u135f\u170e-\u1710\u1720-\u1730\u1740-\u1750\u1772\u1773\u1780-\u17b2\u17dd\u17e0-\u17e9\u180b-\u180d\u1810-\u1819\u1920-\u192b\u1930-\u193b\u1951-\u196d\u19b0-\u19c0\u19c8-\u19c9\u19d0-\u19d9\u1a00-\u1a15\u1a20-\u1a53\u1a60-\u1a7c\u1a7f-\u1a89\u1a90-\u1a99\u1b46-\u1b4b\u1b50-\u1b59\u1b6b-\u1b73\u1bb0-\u1bb9\u1be6-\u1bf3\u1c00-\u1c22\u1c40-\u1c49\u1c5b-\u1c7d\u1cd0-\u1cd2\u1d00-\u1dbe\u1e01-\u1f15\u200c\u200d\u203f\u2040\u2054\u20d0-\u20dc\u20e1\u20e5-\u20f0\u2d81-\u2d96\u2de0-\u2dff\u3021-\u3028\u3099\u309a\ua640-\ua66d\ua674-\ua67d\ua69f\ua6f0-\ua6f1\ua7f8-\ua800\ua806\ua80b\ua823-\ua827\ua880-\ua881\ua8b4-\ua8c4\ua8d0-\ua8d9\ua8f3-\ua8f7\ua900-\ua909\ua926-\ua92d\ua930-\ua945\ua980-\ua983\ua9b3-\ua9c0\uaa00-\uaa27\uaa40-\uaa41\uaa4c-\uaa4d\uaa50-\uaa59\uaa7b\uaae0-\uaae9\uaaf2-\uaaf3\uabc0-\uabe1\uabec\uabed\uabf0-\uabf9\ufb20-\ufb28\ufe00-\ufe0f\ufe20-\ufe26\ufe33\ufe34\ufe4d-\ufe4f\uff10-\uff19\uff3f';
                    ast._Tokenizer._identifierStart = new RegExp(`[${identifierStartChars}]`);
                    /* eslint-disable no-misleading-character-class */
                    ast._Tokenizer._identifierChar = new RegExp(`[${identifierStartChars}${identifierChars}]`);
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
                    if (ast._Tokenizer._isNewline(this._get(this._position))) {
                        this._position = this._newLine(this._position);
                        this.linepos = this._position;
                        this.lineno++;
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
                return `at ${this.filename}:${this.lineno}:${this.col_offset}.`;
            }
            get col_offset() {
                return this._position - this.linepos + 1;
            }
            static _isSpace(c) {
                if (c === ' ' || c === '\t' || c === '\v' || c === '\f' || c === '\xA0') {
                    return true;
                }
                if (c.charCodeAt(0) >= 0x1680) {
                    return ast._Tokenizer._whitespace.test(c);
                }
                return false;
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
                    return ast._Tokenizer._identifierStart.test(c);
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
                    return ast._Tokenizer._identifierChar.test(c);
                }
                return false;
            }
            _get(position) {
                return position >= this._text.length ? '\0' : this._text[position];
            }
            _skipLine() {
                while (this._position < this._text.length) {
                    if (ast._Tokenizer._isNewline(this._get(this._position))) {
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
                    } else if (ast._Tokenizer._isSpace(c)) {
                        this._position++;
                    } else if (c === '\\') {
                        // Explicit Line Continuation
                        this._position++;
                        if (ast._Tokenizer._isNewline(this._get(this._position))) {
                            this._position = this._newLine(this._position);
                            this.linepos = this._position;
                            this.lineno += 1;
                        } else {
                            throw new python.Error(`Unexpected '${this._text[this._position]}' after line continuation ${this.location()}`);
                        }
                    } else if (this._brackets > 0 && ast._Tokenizer._isNewline(c)) {
                        // Implicit Line Continuation
                        this._position = this._newLine(this._position);
                        this.linepos = this._position;
                        this.lineno += 1;
                    } else {
                        break;
                    }
                }
            }
            _newLine(position) {
                if ((this._get(position) === '\n' && this._get(position + 1) === '\r') || (this._get(position) === '\r' && this._get(position + 1) === '\n')) {
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
                        if (ast._Tokenizer._isSpace(c)) {
                            indent += c;
                            i++;
                        } else if (ast._Tokenizer._isNewline(c)) {
                            indent = '';
                            i = this._newLine(i);
                            this._position = i;
                            this.linepos = i;
                            this.lineno += 1;
                        } else if (c === '#') {
                            indent = '';
                            while (i < this._text.length && !ast._Tokenizer._isNewline(this._text[i])) {
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
                if (ast._Tokenizer._isNewline(c)) {
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
                if (ast._Tokenizer._isIdentifierStartChar(this._get(i))) {
                    i++;
                    while (ast._Tokenizer._isIdentifierChar(this._get(i))) {
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
                        case 'For':
                        case 'If':
                        case 'Import':
                        case 'in':
                        case 'is':
                        case 'not':
                        case 'or':
                            keyword = true;
                            break;
                        default:
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
                    case '+': case '&': case '|': case '^': case '=': case '!': case '%': case '~':
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
                    const cc = c.toLowerCase();
                    if (cc === 'b' || cc === 'f' || cc === 'r' || cc === 'u') {
                        prefix = c;
                    }
                } else if (this._get(i + 2) === "'" || this._get(i + 2) === '"') {
                    const c = this._text.substr(this._position, 2);
                    const cc = c.toLowerCase();
                    if (cc === 'br' || cc === 'fr' || cc === 'rb' || cc === 'rf' || cc === 'ur') {
                        prefix = c;
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
        });
        this.registerType('builtins.dict', dict);
        this.registerType('builtins.ellipsis', class {});
        this.registerType('builtins.cell', class {});
        this.registerType('builtins.list', class extends Array {});
        this.registerType('builtins.number', class {});
        this.registerFunction('builtins.__import__', (name, globals, locals, fromlist, level) => {
            return execution.__import__(name, globals, locals, fromlist, level);
        });
        this.registerType('builtins.bool', class extends Boolean {
            constructor(value) {
                if (value && value.__bool__) {
                    value = value.__bool__();
                } else if (value && value.__len__) {
                    value = value.__len__() > 0;
                } else {
                    value = value ? true : false;
                }
                super(value);
            }
        });
        this.registerType('builtins.int', class extends Number {
            constructor(value) {
                if (value && value.__int__) {
                    value = value.__int__();
                } else if (!Number.isInteger(value)) {
                    value = NaN;
                }
                super(value);
            }
        });
        this.registerType('builtins.float', class extends Number {
            constructor(value) {
                if (value && value.__float__) {
                    value = value.__float__();
                } else if (Number(value) !== value) {
                    value = NaN;
                }
                super(value);
            }
        });
        this.registerType('builtins.long', class extends Number {
            constructor(value) {
                if (value && value.__int__) {
                    value = value.__int__();
                } else if (!Number.isInteger(value)) {
                    value = NaN;
                }
                super(value);
            }
        });
        this.registerType('builtins.str', class extends String {
            constructor(value) {
                if (value && value.__str__) {
                    value = value.__str__();
                } else if (typeof value !== 'string') {
                    value = JSON.stringify(value);
                }
                super(value);
            }
        });
        this.registerType('builtins.complex', class {
            constructor(real, imaginary) {
                this.real = real;
                this.imag = imaginary;
            }
        });
        this.registerType('builtins.NoneType', class {});
        this.registerType('builtins.object', class {
            static __new__(cls, ...args) {
                return execution.invoke(cls, args);
            }
        });
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
        this.registerType('builtins.Warning', class {});
        this.registerType('builtins.FutureWarning', class extends builtins.Warning {});
        this.registerType('builtins.BaseException', class {});
        this.registerType('builtins.Exception', class extends builtins.BaseException {});
        this.registerType('builtins.AttributeError', class extends builtins.Exception {});
        this.registerType('builtins.SyntaxError', class extends builtins.Exception {});
        this.registerFunction('builtins.print', () => {});
        this.registerFunction('builtins.unicode');
        builtins.Ellipsis = new builtins.ellipsis();
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
        this.registerFunction('operator.getitem');
        this.registerFunction('operator.gt');
        this.registerFunction('operator.mul');
        this.registerFunction('operator.mod');
        this.registerFunction('operator.le');
        this.registerFunction('operator.lt');
        this.registerFunction('operator.ne');
        this.registerFunction('operator.floordiv');
        this.registerFunction('operator.sub');
        this.registerFunction('sys.path.append', () => {});
        this.registerFunction('sys.path.insert', () => {});
        this.registerType('argparse.Namespace', class {
            constructor(args) {
                this.args = args;
            }
        });
        this.registerType('catboost._catboost._CatBoost', class {
            _deserialize_model(/* serialized_model_str */) {
            }
        });
        this.registerType('catboost.core._CatBoostBase', class {
            constructor() {
                this._object = new catboost._catboost._CatBoost();
            }
            __setstate__(state) {
                for (const [key, value] of state) {
                    if (key === '__model') {
                        this._load_from_string(value);
                        continue;
                    }
                    this[key] = value;
                }
            }
            _load_from_string(dump_model_str) {
                this._deserialize_model(dump_model_str);
            }
            _deserialize_model(dump_model_str) {
                this._object._deserialize_model(dump_model_str);
            }
        });
        this.registerType('catboost.core.CatBoost', class extends catboost.core._CatBoostBase {
            load_model(/* blob */) {
                throw new python.Error("'catboost.core.CatBoostClassifier.load_model' not implemented.");
                // this._load_from_string(blob);
            }
        });
        this.registerType('catboost.core.CatBoostClassifier', class extends catboost.core.CatBoost {});
        this.registerType('catboost.core.CatBoostRegressor', class extends catboost.core.CatBoost {});
        catboost.CatBoostClassifier = catboost.core.CatBoostClassifier;
        catboost.CatBoostRegressor = catboost.core.CatBoostRegressor;
        catboost.CatBoost = catboost.core.CatBoost;
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
        this.registerType('cuml.internals.array.CumlArray', class {});
        this.registerType('cuml.internals.mem_type.MemoryType', class {});
        this.registerType('cuml.raft.common.handle.Handle', class {
            __setstate__(state) {
                this._handle = state;
            }
        });
        this.registerType('cuml.svm.svr.SVR', class {});
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
        this.registerType('hmmlearn.hmm.GMMHMM', class {});
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
        this.registerType('io.StringIO', class {
            constructor() {
                this._buf = [];
            }
            write(text) {
                this._buf.push(text);
            }
            toString() {
                return this._buf.join('');
            }
        });
        this.registerType('numpy.dtype', class {
            constructor(obj, align, copy) {
                if (typeof obj === 'string' && (obj.startsWith('<') || obj.startsWith('>') || obj.startsWith('|'))) {
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
                    case 'f1': case 'float8_e5m2': this.itemsize = 1; this.kind = 'f'; break;
                    case 'f2': case 'float16': this.itemsize = 2; this.kind = 'f'; break;
                    case 'f4': case 'float32': this.itemsize = 4; this.kind = 'f'; break;
                    case 'f8': case 'float64': case 'float': this.itemsize = 8; this.kind = 'f'; break;
                    case 'c8': case 'complex64': this.itemsize = 8; this.kind = 'c'; break;
                    case 'c16': case 'complex128': case 'complex': this.itemsize = 16; this.kind = 'c'; break;
                    case 'M8': case 'M': this.itemsize = 8; this.kind = 'M'; break;
                    case 'V': case 'void': this.itemsize = 0; this.kind = 'V'; break;
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
                        } else if (obj.startsWith('T')) {
                            this.kind = 'T';
                            this.itemsize = parseInt(obj.substring(1), 10);
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
                    case 'T': return `StringDType${this.itemsize === 0 ? '' : (this.itemsize * 8)}`;
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
                            case 1: return 'float8e5m2';
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
                    case 'T':
                        return 'string';
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
        this.registerType('numpy.dtypes.StringDType', class extends numpy.dtype {
            constructor() {
                super('|T16');
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
        this.registerType('gensim.models.keyedvectors.KeyedVectors', class {});
        this.registerType('gensim.models.keyedvectors.Vocab', class {});
        this.registerType('gensim.models.keyedvectors.Word2VecKeyedVectors', class {});
        this.registerType('gensim.models.ldamodel.LdaState', class {});
        this.registerType('gensim.models.ldamulticore.LdaMulticore', class {});
        this.registerFunction('gensim.models.phrases.original_scorer');
        this.registerType('gensim.models.phrases.Phraser', class {});
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
        this.registerType('joblib._store_backends.FileSystemStoreBackend', class {});
        this.registerType('joblib.memory.NotMemorizedFunc', class {});
        this.registerType('joblib.numpy_pickle.NumpyArrayWrapper', class {

            __read__(unpickler) {
                if (this.dtype.__name__ === 'object') {
                    return unpickler.load();
                }
                if (this.numpy_array_alignment_bytes) {
                    const [size] = unpickler.read(1);
                    unpickler.read(size);
                }
                if (this.order === 'F') {
                    throw new python.Error('Fortran order not implemented.');
                }
                const size = this.dtype.itemsize * this.shape.reduce((a, b) => a * b, 1);
                this.data = unpickler.read(size);
                return execution.invoke(this.subclass, [this.shape, this.dtype, this.data]);
            }
        });
        this.registerType('joblib.numpy_pickle.NDArrayWrapper', class {

            __setstate__(state) {
                this.subclass = state.get('subclass');
                this.filename = state.get('state');
                this.allow_mmap = state.get('allow_mmap');
            }
            __read__(/* unpickler */) {
                return this; // return execution.invoke(this.subclass, [ this.shape, this.dtype, this.data ]);
            }
        });
        sklearn.externals.joblib.numpy_pickle.NDArrayWrapper = joblib.numpy_pickle.NDArrayWrapper;
        sklearn.externals.joblib.numpy_pickle.NumpyArrayWrapper = joblib.numpy_pickle.NumpyArrayWrapper;
        this.registerType('keras.engine.sequential.Sequential', class {});
        this.registerType('keras.src.legacy.preprocessing.text.Tokenizer', class {});
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
                const model_str = state.get('_handle', state.get('handle', null));
                if (model_str) {
                    this.LoadModelFromString(model_str);
                    return;
                }
                for (const [key, value] of state) {
                    this[key] = value;
                }
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
        this.registerType('megengine.quantization.fake_quant.LSQ', class {});
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
                const value = new numpy.ndarray([size], this.dtype, this.data, this.offset, this.strides, this.order);
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
                    case 'V': {
                        const data = this.data;
                        const itemsize = this.dtype.itemsize;
                        let offset = 0;
                        for (let i = 0; i < size; i++) {
                            list[i] = data.slice(offset, offset + itemsize);
                            offset += itemsize;
                        }
                        return list;
                    }
                    case 'T': {
                        return this.data;
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
                throw new python.Error("'numpy.matrix.__new__' not implemented.");
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
        this.registerType('pandas.core.arrays.categorical.Categorical', class {});
        this.registerType('pandas.core.arrays.datetimes.DatetimeArray', class {});
        this.registerType('pandas.core.arrays.integer.IntegerArray', class {});
        this.registerType('pandas.core.frame.DataFrame', class {});
        this.registerFunction('pandas.core.indexes.base._new_Index', (cls, d) => {
            return new cls(d);
        });
        this.registerType('pandas.core.indexes.datetimes._new_DatetimeIndex', class {});
        this.registerType('pandas.core.indexes.datetimes.DatetimeIndex', class {});
        this.registerType('pandas.core.indexes.base.Index', class {});
        this.registerType('pandas.core.indexes.range.RangeIndex', class {});
        this.registerType('pandas.core.indexes.multi.MultiIndex', class {});
        this.registerType('pandas.core.indexes.numeric.Int64Index', class {});
        this.registerType('pandas.core.index.Int64Index', class {});
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
        pandas.indexes.base._new_Index = pandas.core.indexes.base._new_Index;
        pandas.indexes.base.Index = pandas.core.indexes.base.Index;
        pandas.indexes.range.RangeIndex = pandas.core.indexes.range.RangeIndex;
        pandas.core.index.Index = pandas.core.indexes.base.Index;
        pandas.core.index._new_Index = pandas.core.indexes.base._new_Index;
        pandas.core.internals.BlockManager = pandas.core.internals.managers.BlockManager;
        pandas._libs.tslib.Timestamp = pandas._libs.tslibs.timestamps.Timestamp;
        this.registerType('pathlib.Path', class {});
        this.registerType('pathlib.PosixPath', class {});
        this.registerType('pathlib.WindowsPath', class {});
        this.registerType('shap._serializable.Serializable', class {});
        this.registerType('shap.explainers._explainer.Explainer', class extends shap._serializable.Serializable {});
        this.registerType('shap.explainers._linear.LinearExplainer', class extends shap.explainers._explainer.Explainer {});
        shap.explainers.LinearExplainer = shap.explainers._linear.LinearExplainer;
        shap.explainers.linear.LinearExplainer = shap.explainers._linear.LinearExplainer;
        this.registerType('sklearn._loss.link.BaseLink', class {});
        this.registerType('sklearn._loss._loss.__pyx_unpickle_CyHalfBinomialLoss', class {});
        this.registerType('sklearn._loss._loss.__pyx_unpickle_CyHalfMultinomialLoss', class {});
        this.registerType('sklearn._loss._loss.CyLossFunction', class {});
        this.registerType('sklearn._loss._loss.CyHalfBinomialLoss', class {});
        this.registerType('sklearn._loss._loss.CyHalfMultinomialLoss', class {});
        this.registerType('sklearn._loss._loss.CyHalfSquaredError', class extends sklearn._loss._loss.CyLossFunction {});
        this.registerType('sklearn._loss.link.IdentityLink', class extends sklearn._loss.link.BaseLink {});
        this.registerType('sklearn._loss.link.Interval', class {});
        this.registerType('sklearn._loss.link.LogitLink', class {});
        this.registerType('sklearn._loss.link.MultinomialLogit', class extends sklearn._loss.link.BaseLink {});
        this.registerFunction('sklearn._loss._loss.__pyx_unpickle_CyHalfSquaredError');
        this.registerType('sklearn._loss.loss.BaseLoss', class {});
        this.registerType('sklearn._loss.loss.HalfBinomialLoss', class {});
        this.registerType('sklearn._loss.loss.HalfMultinomialLoss', class extends sklearn._loss.loss.BaseLoss {});
        this.registerType('sklearn._loss.loss.HalfSquaredError', class extends sklearn._loss.loss.BaseLoss {});
        this.registerType('sklearn.base.BaseEstimator', class {});
        this.registerType('sklearn.base.TransformerMixin', class {});
        this.registerType('sklearn.calibration._CalibratedClassifier', class {});
        this.registerType('sklearn.calibration._SigmoidCalibration', class {});
        this.registerType('sklearn.calibration.CalibratedClassifierCV', class {});
        this.registerType('sklearn.cluster._agglomerative.FeatureAgglomeration', class {});
        this.registerType('sklearn.cluster._dbscan.DBSCAN', class {});
        this.registerType('sklearn.cluster._kmeans.KMeans', class {});
        this.registerType('sklearn.cluster._kmeans.MiniBatchKMeans', class {});
        this.registerType('sklearn.cluster.k_means_.MiniBatchKMeans', class {});
        this.registerType('sklearn.compose._column_transformer._RemainderColsList', class {});
        this.registerType('sklearn.compose._column_transformer.ColumnTransformer', class {});
        this.registerType('sklearn.compose._column_transformer.make_column_selector', class {});
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
        this.registerType('sklearn.ensemble._gb_losses.LeastAbsoluteError', class {});
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
        this.registerType('sklearn.ensemble.voting_classifier.VotingClassifier', class {});
        this.registerType('sklearn.ensemble.weight_boosting.AdaBoostClassifier', class {});
        this.registerType('sklearn.feature_extraction._dict_vectorizer.DictVectorizer', class {});
        this.registerType('sklearn.feature_extraction._hashing.FeatureHasher', class {});
        this.registerType('sklearn.feature_extraction.text.CountVectorizer', class {});
        this.registerType('sklearn.feature_extraction.text.HashingVectorizer', class {});
        this.registerType('sklearn.feature_extraction.text.TfidfTransformer', class {});
        this.registerType('sklearn.feature_extraction.text.TfidfVectorizer', class {});
        this.registerType('sklearn.feature_selection._from_model.SelectFromModel', class {});
        this.registerFunction('sklearn.feature_selection._mutual_info.mutual_info_classif');
        this.registerFunction('sklearn.feature_selection._univariate_selection.chi2');
        this.registerType('sklearn.feature_selection._univariate_selection.GenericUnivariateSelect', class {});
        this.registerType('sklearn.feature_selection._univariate_selection.SelectKBest', class {});
        this.registerType('sklearn.feature_selection._univariate_selection.SelectPercentile', class {});
        this.registerType('sklearn.feature_selection._variance_threshold.VarianceThreshold', class {});
        this.registerType('sklearn.feature_selection._rfe.RFE', class {});
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
        this.registerType('sklearn.impute._iterative._ImputerTriplet', class {});
        this.registerType('sklearn.impute.SimpleImputer', class {});
        this.registerType('sklearn.isotonic.IsotonicRegression', class {});
        this.registerType('sklearn.kernel_ridge.KernelRidge', class {});
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
        this.registerType('sklearn.linear_model._ridge.RidgeClassifierCV', class {});
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
        this.registerType('sklearn.metrics._dist_metrics.DistanceMetric', class extends builtins.object {});
        this.registerType('sklearn.metrics._dist_metrics.DistanceMetric32', class extends sklearn.metrics._dist_metrics.DistanceMetric {});
        this.registerType('sklearn.metrics._dist_metrics.DistanceMetric64', class extends sklearn.metrics._dist_metrics.DistanceMetric {});
        this.registerType('sklearn.metrics._dist_metrics.EuclideanDistance', class extends sklearn.metrics._dist_metrics.DistanceMetric {});
        this.registerType('sklearn.metrics._dist_metrics.EuclideanDistance32', class extends sklearn.metrics._dist_metrics.DistanceMetric32 {});
        this.registerType('sklearn.metrics._dist_metrics.EuclideanDistance64', class extends sklearn.metrics._dist_metrics.DistanceMetric64 {});
        this.registerType('sklearn.metrics._dist_metrics.ManhattanDistance', class extends sklearn.metrics._dist_metrics.DistanceMetric {});
        this.registerType('sklearn.metrics._dist_metrics.ManhattanDistance64', class extends sklearn.metrics._dist_metrics.DistanceMetric64 {});
        this.registerType('sklearn.metrics._scorer._PassthroughScorer', class {});
        this.registerType('sklearn.metrics._scorer._PredictScorer', class {});
        this.registerType('sklearn.metrics.scorer._PredictScorer', class {});
        this.registerType('sklearn.metrics._scorer._ThresholdScorer', class {});
        this.registerType('sklearn.mixture._bayesian_mixture.BayesianGaussianMixture', class {});
        this.registerType('sklearn.mixture._gaussian_mixture.GaussianMixture', class {});
        this.registerType('sklearn.model_selection._search.GridSearchCV', class {});
        this.registerType('sklearn.model_selection._search.RandomizedSearchCV', class {});
        this.registerType('sklearn.model_selection._split.KFold', class {});
        this.registerType('sklearn.model_selection._split.RepeatedKFold', class {});
        this.registerType('sklearn.model_selection._split.StratifiedKFold', class {});
        this.registerType('sklearn.model_selection._split.StratifiedShuffleSplit', class {});
        this.registerType('sklearn.multiclass.OneVsRestClassifier', class {});
        this.registerType('sklearn.multioutput.ClassifierChain', class {});
        this.registerType('sklearn.multioutput.MultiOutputClassifier', class {});
        this.registerType('sklearn.multioutput.MultiOutputRegressor', class {});
        this.registerType('sklearn.naive_bayes.BernoulliNB', class {});
        this.registerType('sklearn.naive_bayes.ComplementNB', class {});
        this.registerType('sklearn.naive_bayes.GaussianNB', class {});
        this.registerType('sklearn.naive_bayes.MultinomialNB', class {});
        this.registerType('sklearn.neighbors.ball_tree.BallTree', class {});
        this.registerFunction('sklearn.neighbors.ball_tree.newObj', (obj) => {
            return obj.__new__(obj);
        });
        this.registerType('sklearn.neighbors._classification.KNeighborsClassifier', class {});
        this.registerFunction('sklearn.neighbors._dist_metrics.newObj');
        this.registerType('sklearn.neighbors._dist_metrics.EuclideanDistance', class {});
        this.registerType('sklearn.neighbors._kd_tree.BinaryTree64', class extends builtins.object {});
        this.registerType('sklearn.neighbors._kd_tree.KDTree64', class extends sklearn.neighbors._kd_tree.BinaryTree64 {});
        this.registerType('sklearn.neighbors._kd_tree.KDTree', class extends sklearn.neighbors._kd_tree.KDTree64 {});
        this.registerFunction('sklearn.neighbors._kd_tree.newObj', (obj) => {
            return obj.__new__(obj);
        });
        this.registerType('sklearn.neighbors._regression.KNeighborsRegressor', class {});
        this.registerType('sklearn.neighbors._unsupervised.NearestNeighbors', class {});
        this.registerType('sklearn.neighbors.classification.KNeighborsClassifier', class {});
        this.registerFunction('sklearn.neighbors.dist_metrics.newObj', (obj) => {
            return obj.__new__(obj);
        });
        this.registerType('sklearn.neighbors.dist_metrics.EuclideanDistance', class {});
        this.registerFunction('sklearn.neighbors.kd_tree.newObj', (obj) => {
            return obj.__new__(obj);
        });
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
        this.registerType('sklearn.random_projection.GaussianRandomProjection', class {});
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
                this.max_depth = state.get('max_depth');
                this.node_count = state.get('node_count');
                this.nodes = state.get('nodes');
                this.values = state.get('values');
            }
        });
        this.registerType('sklearn.tree.tree.DecisionTreeClassifier', class {});
        this.registerType('sklearn.tree.tree.DecisionTreeRegressor', class {});
        this.registerType('sklearn.tree.tree.ExtraTreeClassifier', class {});
        this.registerType('sklearn.utils._bunch.Bunch', class {});
        this.registerType('sklearn.utils._metadata_requests.MetadataRequest', class {});
        this.registerType('sklearn.utils._metadata_requests.MethodMetadataRequest', class {});
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
                    // console.log(`${(reader.position - 1).toString()} ${opcode}`);
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
                            stack.push(new builtins.list());
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
                            const dict = new builtins.dict();
                            for (let i = 0; i < items.length; i += 2) {
                                dict.__setitem__(items[i], items[i + 1]);
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
                            if (obj.__setitem__) {
                                for (let i = 0; i < items.length; i += 2) {
                                    obj.__setitem__(items[i], items[i + 1]);
                                }
                            } else {
                                for (let i = 0; i < items.length; i += 2) {
                                    obj[items[i]] = items[i + 1];
                                }
                            }
                            break;
                        }
                        case 125: // EMPTY_DICT '}'
                            stack.push(new builtins.dict());
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
                            } else if (obj instanceof Map && state instanceof Map) {
                                for (const [key, value] of state) {
                                    obj.set(key, value);
                                }
                            } else if (obj instanceof Map) {
                                for (const key in state) {
                                    obj.set(key, state[key]);
                                }
                            } else if (state instanceof Map) {
                                for (const [key, value] of state) {
                                    obj[key] = value;
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
        this.registerType('theano.scalar.basic.Pow', class {});
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
        const types = this.register('types');
        this.registerType('types.GenericAlias', class {});
        this.registerType('types.SimpleNamespace', class {});
        this.registerFunction('types.resolve_bases', (bases) => {
            return bases;
        });
        this.registerFunction('types.prepare_class', (name, bases, kwds) => {
            if (kwds) {
                kwds = new builtins.dict(kwds);
            } else {
                kwds = new builtins.dict();
            }
            let meta = null;
            if (kwds.__contains__('metaclass')) {
                meta = kwds.pop('metaclass');
            } else if (bases && bases.length > 0) {
                meta = builtins.type(bases[0]);
            } else {
                meta = builtins.type;
            }
            if (meta instanceof builtins.type) {
                meta = types._calculate_meta(meta, bases);
            }
            let ns = null;
            if (builtins.hasattr(meta, '__prepare__')) {
                // ns = meta.__prepare__(name, bases, **kwds)
            } else {
                ns = new builtins.dict();
            }
            return [meta, ns, kwds];
        });
        this.registerFunction('types._calculate_meta', (meta /*, bases*/) => {
            const winner = meta;
            return winner;
        });
        this.registerFunction('types.new_class', (name, bases, kwds, exec_body) => {
            const resolved_bases = types.resolve_bases(bases);
            const [meta, ns] = types.prepare_class(name, bases, kwds);
            if (exec_body) {
                exec_body(ns);
            }
            return new meta(name, resolved_bases, ns);
        });
        types.ObjectType = builtins.object;
        types.ModuleType = builtins.module;
        types.MethodType = builtins.method;
        types.FunctionType = builtins.function;
        types.TypeType = builtins.type;
        types.CodeType = builtins.code;
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
        this.registerFunction('builtins.exec');
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
            if (obj instanceof Map && obj.__contains__) {
                return obj.__contains__(name);
            }
            return Object.prototype.hasOwnProperty.call(obj, name);
        });
        this.registerFunction('builtins.getattr', (obj, name, defaultValue) => {
            if (Object.prototype.hasOwnProperty.call(obj, name)) {
                return obj[name];
            }
            if (obj && obj.__getattr__) {
                return obj.__getattr__(name);
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
        this.registerFunction('cloudpickle.cloudpickle._fill_function');

        this.registerType('cloudpickle.cloudpickle._empty_cell_value', class {});
        this.registerFunction('cloudpickle.cloudpickle._make_cell', (value) => {
            value = value || cloudpickle.cloudpickle._empty_cell_value;
            const cell = cloudpickle.cloudpickle._make_empty_cell();
            if (value !== cloudpickle.cloudpickle._empty_cell_value) {
                cell.cell_contents = value;
            }
            return cell;
        });
        this.registerFunction('cloudpickle.cloudpickle._make_function', (code, globals, name, argdefs, closure) => {
            // globals["__builtins__"] = __builtins__
            return new types.FunctionType(code, globals, name, argdefs, closure);
        });
        this.registerFunction('cloudpickle.cloudpickle._make_skel_func');
        cloudpickle.cloudpickle._DYNAMIC_CLASS_TRACKER_BY_ID = new builtins.dict();
        this.registerFunction('cloudpickle.cloudpickle._lookup_class_or_track', (class_tracker_id, class_def) => {
            if (class_tracker_id) {
                class_def = cloudpickle.cloudpickle._DYNAMIC_CLASS_TRACKER_BY_ID.setdefault(class_tracker_id, class_def);
            }
            return class_def;
        });
        this.registerFunction('cloudpickle.cloudpickle._make_skeleton_class', (type_constructor, name, bases, type_kwargs, class_tracker_id /*, extra */) => {
            // https://github.com/ray-project/ray/blob/5cd8967f1c0c16d3ae5fedb8449d0d25dd4f9f3e/python/ray/cloudpickle/cloudpickle.py#L523
            const kwds = { 'metaclass': type_constructor };
            const skeleton_class = types.new_class(name, bases, kwds, (ns) => ns.update(type_kwargs));
            return cloudpickle.cloudpickle._lookup_class_or_track(class_tracker_id, skeleton_class);
        });
        this.registerFunction('cloudpickle.cloudpickle._make_empty_cell', () => {
            return new builtins.cell();
        });
        this.registerFunction('cloudpickle.cloudpickle._class_setstate', (obj, state) => {
            [state] = state;
            let registry = null;
            for (const [attrname, attr] of state.items()) {
                if (attrname === '_abc_impl') {
                    registry = attr;
                } else {
                    builtins.setattr(obj, attrname, attr);
                }
            }
            if (sys.version_info >= (3, 13) && state.__contains__('__firstlineno__')) {
                obj.__firstlineno__ = state.get('__firstlineno__');
            }
            if (registry) {
                for (const subclass of registry) {
                    obj.register(subclass);
                }
            }
            return obj;
        });
        this.registerFunction('cloudpickle.cloudpickle._function_setstate', (obj, state) => {
            const [, slotstate] = state;
            [state] = state;
            // obj.__dict__.update(state)
            /* const obj_globals = */ slotstate.pop('__globals__');
            const obj_closure = slotstate.pop('__closure__');
            slotstate.pop('_cloudpickle_submodules');
            if (obj.__globals__) {
                // obj.__globals__.update(obj_globals);
                // obj.__globals__.__builtins__ = __builtins__;
            }
            if (obj_closure) {
                // let value = null;
                for (let i = 0; i < obj_closure.length; i++) {
                    // const cell = obj_closure[i];
                    try {
                        // value = cell.cell_contents;
                    } catch {
                        // cell is empty
                    }
                    // obj.__closure__[i].cell_contents = value;
                }
            }
            for (const [k, v] of slotstate.items()) {
                builtins.setattr(obj, k, v);
            }
        });
        this.registerFunction('cloudpickle.cloudpickle.subimport', (name) => {
            execution.__import__(name);
            return sys.modules.get(name);
        });
        this.registerFunction('cloudpickle.cloudpickle_fast._class_setstate');
        this.registerFunction('cloudpickle.cloudpickle_fast._function_setstate');
        const ray = this.register('ray');
        this.register('ray.cloudpickle.cloudpickle');
        ray.cloudpickle.cloudpickle._builtin_type = cloudpickle.cloudpickle._builtin_type;
        this.registerType('collections.Counter', class {});
        this.registerFunction('collections.defaultdict', (/* default_factory */) => {
            return {};
        });
        this.registerFunction('copy.deepcopy');
        this.registerFunction('copy_reg._reconstructor', (cls, base, state) => {
            // copyreg._reconstructor in Python 3
            if (base === '__builtin__.object' || base === builtins.object) {
                return self.invoke(cls, []);
            } else if (base === '__builtin__.tuple' || base === builtins.tuple) {
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
        this.registerFunction('dill._dill._create_type', (typeobj, ...args) => {
            const [name, bases, dict] = args;
            const type = class extends bases[0] {};
            const identifier = dict.__contains__('__module__') ? `${dict.__getitem__('__module__')}.${name}` : name;
            return self.registerType(identifier, Object.assign(type, dict));
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
                        if (obj.__module__ === 'builtins' && obj.__class__ === builtins.type) {
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
        this.registerFunction('dill._dill.loads');
        this.registerFunction('jax._src.array._reconstruct_array', (fun, args, arr_state, aval_state) => {
            const np_value = fun(...args);
            np_value.__setstate__(arr_state);
            const jnp_value = jax.device_put(np_value);
            jnp_value.aval = jnp_value.aval.update(aval_state);
            return jnp_value;
        });
        jax._src.device_array.reconstruct_device_array = jax._src.array._reconstruct_array;
        this.registerFunction('jax.device_put', (x) => {
            const aval = new jax._src.core.ShapedArray(x.shape, x.dtype);
            return new jax.Array(aval, x.data);
        });
        this.registerType('jax._src.core.AbstractValue', class {});
        this.registerType('jax._src.core.UnshapedArray',  class extends jax._src.core.AbstractValue {});
        this.registerType('jax._src.core.ShapedArray', class extends jax._src.core.UnshapedArray {
            constructor(shape, dtype, weak_type) {
                super();
                this.shape = shape;
                this.dtype = dtype;
                this.weak_type = weak_type || false;
            }
            update(dict) {
                const shape = dict.get('shape') || this.shape;
                const dtype = dict.get('dtype') || this.dtype;
                const weak_type = dict.get('weak_type') || this.weak_type;
                return new jax._src.core.ShapedArray(shape, dtype, weak_type);
            }
        });
        this.registerType('jax.Array', class {
            constructor(aval, data) {
                this.aval = aval;
                this.data = data;
            }
            get dtype() {
                return this.aval.dtype;
            }
            get shape() {
                return this.aval.shape;
            }
            tobytes() {
                return this.data;
            }
        });
        jax.numpy.ndarray = jax.Array;
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
        this.registerFunction('numpy.core._ufunc_reconstruct');
        this.registerFunction('numpy.core.numeric._frombuffer', (/* buf, dtype, shape, order */) => {
            return {};
        });
        this.registerFunction('numpy.core.multiarray._reconstruct', (subtype, shape, dtype) => {
            return numpy.ndarray.__new__(subtype, shape, dtype);
        });
        this.registerFunction('numpy._core.numeric._frombuffer');
        this.registerFunction('numpy._core._internal._convert_to_stringdtype_kwargs', () => {
            return new numpy.dtypes.StringDType();
        });
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
        this.registerFunction('numpy.core._multiarray_umath.cbrt');
        this.registerFunction('numpy.core._multiarray_umath.fmin');
        this.registerFunction('numpy.core._multiarray_umath.fmax');
        this.registerFunction('numpy.core._multiarray_umath.greater');
        this.registerFunction('numpy.core._multiarray_umath.less');
        this.registerFunction('numpy.core._multiarray_umath.log');
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
        this.registerFunction('numpy.core._multiarray_umath.sqrt');
        this.register('numpy._core.multiarray', numpy.core.multiarray);
        this.register('numpy._core._multiarray_umath', numpy.core._multiarray_umath);
        this.register('numpy._core._multiarray_umath', numpy.core._multiarray_umath);
        numpy._core._multiarray_umath._reconstruct = numpy.core.multiarray._reconstruct;
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
                    if (header.descr.length !== 3 && header.descr[1] !== 'U' && header.descr.substring(1) !== 'c16') {
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
            if ((descr.length !== 3 && descr.substring(1) !== 'c16') || (descr[1] !== 'f' && descr[1] !== 'i' && descr[1] !== 'u' && descr[1] !== 'c' && descr.substring(1) !== 'b1')) {
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
        this.registerFunction('numpy.amin');
        this.registerFunction('numpy.amax');
        this.registerFunction('numpy.std');
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
        this.registerFunction('numpy.max');
        this.registerFunction('numpy.mean');
        this.registerFunction('numpy.min');
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
        this.registerType('numpy.random.bit_generator.SeedSequence', class {});
        this.registerFunction('numpy.random.bit_generator.__pyx_unpickle_SeedSequence');
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
        this.registerFunction('sklearn.feature_selection._univariate_selection.f_classif');
        this.registerFunction('sklearn.feature_selection._univariate_selection.f_regression');
        this.registerFunction('sklearn.metrics.scorer._passthrough_scorer');
        this.registerFunction('sklearn.metrics._classification.accuracy_score');
        this.registerFunction('sklearn.metrics._classification.balanced_accuracy_score');
        this.registerFunction('sklearn.metrics._classification.f1_score');
        this.registerFunction('sklearn.metrics._classification.log_loss');
        this.registerFunction('sklearn.metrics._classification.precision_score');
        this.registerFunction('sklearn.metrics._classification.recall_score');
        this.registerFunction('sklearn.metrics._dist_metrics.newObj', (obj) => {
            return obj.__new__(obj);
        });
        this.registerFunction('sklearn.metrics._ranking.roc_auc_score');
        this.registerFunction('sklearn.metrics._regression.mean_absolute_error');
        this.registerFunction('sklearn.metrics._regression.mean_squared_error');
        this.registerFunction('sklearn.metrics._regression.r2_score');
        sklearn.metrics.regression = sklearn.metrics._regression;
        sklearn.metrics.r2_score = sklearn.metrics._regression.r2_score;
        this.registerFunction('sklearn.metrics._regression.root_mean_squared_error');
        this.registerFunction('sklearn.metrics._scorer._passthrough_scorer');
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
            __delattr__(name) {
                if (this._modules.has(name)) {
                    this._modules.delete(name);
                }
            }
            children() {
                return this._modules.values();
            }
            named_modules(memo, prefix, remove_duplicate) {
                memo = memo || new Set();
                prefix = prefix || '';
                const modules = new builtins.dict();
                if (!memo.has(this)) {
                    if (remove_duplicate) {
                        memo.add(this);
                    }
                    modules.set(prefix, this);
                    for (const [name, module] of this._modules.items()) {
                        if (module) {
                            const submodule_prefix = `${prefix}${(prefix ? '.' : '')}${name}`;
                            for (const [k, v] of module.named_modules(memo, submodule_prefix, remove_duplicate)) {
                                modules.set(k, v);
                            }
                        }
                    }
                }
                return modules;
            }
            named_children() {
                return this._modules;
            }
            parameters() {
                return this._parameters.values();
            }
            named_parameters(recurse) {
                if (recurse) {
                    throw new python.Error('Named parameters with recurse not implemented.');
                }
                return this._parameters;
            }
            buffers() {
                return this._buffers.values();
            }
            named_buffers(recurse) {
                if (recurse) {
                    throw new python.Error('Named parameters with recurse not implemented.');
                }
                return this._buffers;
            }
            _get_name() {
                return this.__class__.__name__;
            }
        });
        torch.nn.Module = torch.nn.modules.module.Module;
        torch.nn.modules.Module = torch.nn.modules.module.Module;
        this.registerType('torch._C._TensorBase', class extends builtins.object {});
        this.registerType('torch._C._TensorMeta', class extends builtins.type {});
        this.registerType('torch._C._VariableFunctionsClass', class extends builtins.object {});
        this.registerType('torch._C.OperatorRegistry', class {
            constructor() {
                this._operators = new Map();
            }
            registerOperator(op) {
                const key = op.schema().name;
                if (!this._operators.has(key)) {
                    this._operators.set(key, []);
                }
                this._operators.get(key).push(op);
            }
            getAllOperators() {
                const values = [];
                for (const [, ops] of this._operators) {
                    values.push(...ops);
                }
                return values;
            }
            getOperators(name) {
                return this._operators.get(name) || [];
            }
        });
        this.registerFunction('torch._C.getAllOperatorsFor', (name) => {
            return torch._C.getRegistry().getOperators(name);
        });
        this.registerType('torch._C.Operator', class {
            constructor(schema) {
                this._schema = schema;
            }
            schema() {
                return this._schema;
            }
            getOperation(/* node */) {
                return null;
            }
        });
        this.registerFunction('torch._C.getRegistry', () => {
            this._operators = this._operators || new torch._C.OperatorRegistry();
            return this._operators;
        });
        this.registerFunction('torch._C._get_schema', (op_name, overload_name) => {
            const operations = torch._C.getAllOperatorsFor(op_name);
            for (const op of operations) {
                if (op.schema().overload_name === overload_name) {
                    return op.schema();
                }
            }
            throw new python.Error(`Schema '${op_name}.${overload_name}' not found.`);
        });
        this.registerFunction('torch._C._jit_get_schemas_for_operator', (op_name) => {
            return torch._C.getAllOperatorsFor(op_name).map((op) => op.schema());
        });
        this.registerFunction('torch._C._jit_get_operation', (op_name) => {
            const sortedOps = torch._C.getAllOperatorsFor(op_name);
            if (sortedOps.length === 0) {
                return [null, null];
            }
            const overload_names = sortedOps.map((op) => op.schema().overload_name);
            return [{}, overload_names];
        });
        this.registerFunction('torch._C._get_operation_overload', (op_name, overload_name) => {
            const operations = torch._C.getAllOperatorsFor(op_name);
            for (const op of operations) {
                if (op.schema().overload_name === overload_name) {
                    return [{}, {}, null];
                }
            }
            return null;
        });
        this.registerType('torch._C.MatchedSchema', class {
            constructor(inputs, return_types, return_field_names, schema_name) {
                this.inputs = inputs;
                this.return_types = return_types;
                this.register_field_names = return_field_names;
                this.schema_name = schema_name;
            }
        });
        this.registerType('torch._C.Self', class {
        });
        this.registerType('torch._C.SimpleSelf', class extends torch._C.Self {
            constructor(classType) {
                super();
                this._classType = classType;
            }
            makeSugared(v) {
                v.setType(this._classType);
                return new torch._C.SimpleValue(v);
            }
            getClassType() {
                return this._classType;
            }
        });
        this.registerType('torch.jit.Function', class {
            isGraphFunction() {
                return false;
            }
            name() {
                return this.qualname().name();
            }
        });
        this.registerType('torch.jit.BuiltinOpFunction', class extends torch.jit.Function {
            constructor(qualname, schema) {
                super();
                this._name = qualname;
                this._schema = schema;
            }
            qualname() {
                return this._name;
            }
            getSchema() {
                return this._schema;
            }
        });
        this.registerFunction('torch._C.EliminateDeadCode', (/* graph */) => {

        });
        this.registerType('torch._C.ConstantPropagator', class {
            constructor(graph, aliasing_types, ignore_custom_classes) {
                this._graph = graph;
                this._aliasing_types = aliasing_types;
                this._ignore_custom_classes = ignore_custom_classes;
            }
            static NoAliasDb(graph) {
                return new torch._C.ConstantPropagator(graph, false, false);
            }
            run() {
                this.ConstantPropagation(this._graph.block());
                return this._made_change;
            }
            supportedNode() {
                return false; // not implemented.
            }
            ConstantPropagation(...args) {
                if (args[0] instanceof torch.Graph) {
                    //
                } else if (args[0] instanceof torch.Block) {
                    const [block] = args;
                    for (const n of block.nodes()) {
                        this.ConstantPropagation(n);
                    }
                } else if (args[0] instanceof torch.Node) {
                    const [n] = args;
                    const constant_inputs = n.inputs().every((v) => v.node().kind() === 'prim::Constant');
                    if (n.kind() === 'prim::If') {
                        throw new python.Error('Not implemented.');
                        /*
                        if (constant_inputs) {
                          inlineIf(n);
                        } else {
                          ConstantPropagation(n->blocks());
                          removeExtraIfOutputs(n);
                        }
                        */
                    } else if (n.kind() === 'prim::Loop') {
                        throw new python.Error('Not implemented.');
                        /*
                        if (loopWillNotRun(n)) {
                          removeLoopNode(n);
                        } else {
                          ConstantPropagation(n->blocks());
                          removeExtraLoopOutputs(n);
                        }
                        */
                    } else if (constant_inputs && this.supportedNode(n)) {
                        this.propagateNode(n);
                    } else {
                        // this.ConstantPropagation(n.blocks()); // not implemented
                    }
                } else {
                    throw new python.Error('Not implemented.');
                }
            }
        });
        this.registerFunction('torch._C.ConstantPropagationImmutableTypes', (graph) => {
            const cp = torch._C.ConstantPropagator.NoAliasDb(graph);
            const made_change = cp.run();
            if (made_change) {
                torch._C.EliminateDeadCode(graph);
            }
            return made_change;
        });
        this.registerType('torch._C.AliasDb', class {

        });
        this.registerFunction('torch._C.ConstantPooling', (...args) => {
            if (args[0] instanceof torch.Graph) {
                const [graph] = args;
                const aliasDb = new torch._C.AliasDb(graph);
                const constants = new Set();
                torch._C.ConstantPooling(graph.block(), constants, aliasDb);
            } else if (args[0] instanceof torch.Block) {
                const [block, constants, aliasDb] = args;
                for (const node of block.nodes()) {
                    // const it = node.next;
                    if (node.blocks().length > 0) {
                        for (const block of node.blocks()) {
                            torch._C.ConstantPooling(block, constants, aliasDb);
                        }
                        continue;
                    }
                    if (node.kind() !== 'prim::Constant') {
                        continue;
                    }
                    if (constants.has(node)) {
                        const existing = constants.get(node);
                        const old_ivalue = torch._C.toIValue(existing.output());
                        const new_ivalue = torch._C.toIValue(node.output());
                        const same_identity = (old_ivalue && new_ivalue && (old_ivalue.is(new_ivalue)));
                        if (!same_identity && !aliasDb.safeToChangeAliasingRelationship(node.outputs(), existing.outputs())) {
                            continue;
                        }
                        node.replaceAllUsesWith(existing);
                        node.destroy();
                        continue;
                    } else {
                        constants.add(node);
                    }
                    const [first_node] = node.owningGraph().block().nodes();
                    if (node !== first_node) {
                        node.moveBefore(first_node);
                    }
                }
            } else {
                throw new python.Error('Not implemented.');
            }
        });
        this.registerFunction('torch._C.preoptimizeGraph', (graph, disable_autocast) => {
            disable_autocast = disable_autocast || false;
            torch._C.Inline(graph);
            // torch._C.PeepholeOptimize(graph, true);
            torch._C.ConstantPropagationImmutableTypes(graph);
            if (!disable_autocast) {
                // torch._C.Autocast(graph);
            }
            torch._C.ConstantPooling(graph);
        });
        this.registerType('torch._C.GraphFunction', class extends torch.jit.Function {
            constructor(name, graph, function_creator, executor_execution_mode) {
                super();
                this._name = name;
                this._graph = graph;
                this._executor_execution_mode = executor_execution_mode;
                this._function_creator = function_creator;
                this._force_no_amp = false;
            }
            isGraphFunction() {
                return true;
            }
            qualname() {
                return this._name;
            }
            graph() {
                return this._graph;
            }
            optimized_graph() {
                const graph_ref = this._graph.copy();
                torch._C.preoptimizeGraph(graph_ref, this._force_no_amp);
                return graph_ref;
            }
            ensure_defined() {
                if (this._function_creator) {
                    const creator = this._function_creator;
                    this._function_creator = () => {
                        throw new python.Error('Recursive method call.');
                    };
                    creator(this);
                    this._function_creator = null;
                }
                this.check_single_output();
            }
            check_single_output() {
                if (this.graph().outputs().length !== 1) {
                    throw new python.Error('Graph must have a single output.');
                }
            }
            getSchema() {
                this._schema = this._schema || this.defaultSchemaFor(this);
                return this._schema;
            }
            setSchema(schema) {
                this._schema = schema;
            }
            num_inputs() {
                return this.graph().inputs().length;
            }
            unshapedType(type) {
                if (type.isSubtypeOf(torch.TensorType.get())) {
                    return torch.TensorType.get();
                }
                throw new python.Error('Not implemented.');
                /*
                at::ArrayRef<TypePtr> contained = type->containedTypes();
                if (contained.empty()) {
                    return type;
                }
                return type->withContained(fmap(type->containedTypes(), unshapedType));
                */
            }
            defaultSchemaFor(fn) {
                const args = [];
                const returns = [];
                const g = fn.graph();
                const num_inputs = fn.num_inputs();
                for (let i = 0; i < num_inputs; i++) {
                    const v = g.inputs()[i];
                    const name = v.hasDebugName() ? v.debugNameBase() : `argument_${i}`;
                    const argument = new torch.Argument(name, this.unshapedType(g.inputs()[i].type()));
                    args.push(argument);
                }
                const num_outputs = g.outputs().length;
                for (let i = 0; i < num_outputs; i++) {
                    const argument = new torch.Argument('', this.unshapedType(g.outputs()[i].type()));
                    returns.push(argument);
                }
                return new torch.FunctionSchema(fn.name(), '', args, returns);
            }
        });
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
        this.registerFunction('torch.ao.quantization.qconfig._activation_is_memoryless');
        this.registerFunction('torch.ao.quantization.qconfig._add_module_to_qconfig_obs_ctr');
        this.registerFunction('torch.ao.quantization.fx.graph_module._save_packed_weight');
        this.registerFunction('torch.ao.quantization.fx._lower_to_native_backend._load_packed_weight');
        this.registerFunction('torch.ao.quantization.fx._lower_to_native_backend._save_packed_weight');
        this.registerFunction('torch.ao.quantization.observer._is_activation_post_process');
        this.registerFunction('torch.ao.quantization.quantize._observer_forward_hook');
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
        this.registerFunction('torch.amp.grad_scaler._refresh_per_optimizer_state');
        this.registerType('torch.autograd.variable.Variable', class {});
        this.registerType('torch.backends.cudnn.rnn.Unserializable', class {});
        this.registerFunction('torch.distributed._shard.sharded_tensor.pre_load_state_dict_hook');
        this.registerFunction('torch.distributed._shard.sharded_tensor.state_dict_hook');
        this.registerType('torch.distributed.algorithms.join._JoinConfig', class {});
        this.registerFunction('torch.distributed._sharded_tensor.state_dict_hook');
        this.registerFunction('torch.distributed._sharded_tensor.pre_load_state_dict_hook');
        this.registerType('torch.distributed._tensor.api.DTensor', class extends torch._C._TensorMeta {});
        this.registerType('torch.distributed._tensor.placement_types.DTensorSpec', class {});
        this.registerType('torch.distributed._tensor.placement_types.Shard', class {});
        this.registerType('torch.distributed._tensor.placement_types.TensorMeta', class {});
        this.registerType('torch.distributed.device_mesh.DeviceMesh', class {});
        this.registerType('torch.distributions.bernoulli.Bernoulli', class {});
        this.registerType('torch.distributions.beta.Beta', class {});
        this.registerType('torch.distributions.binomial.Binomial', class {});
        this.registerType('torch.distributions.categorical.Categorical', class {});
        this.registerType('torch.distributions.constraints._LowerCholesky', class {});
        this.registerType('torch.distributions.constraints._Real', class {});
        this.registerType('torch.distributions.dirichlet.Dirichlet', class {});
        this.registerType('torch.distributions.mixture_same_family.MixtureSameFamily', class {});
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
        this.registerType('torch.nn.intrinsic.quantized.modules.bn_relu.BNReLU2d', class {});
        this.registerType('torch.nn.intrinsic.quantized.modules.conv_relu.ConvReLU1d', class {});
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
        this.registerType('torch.nn.modules.byted_batchnorm.BytedBatchNorm2d', class {});
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
        this.registerType('torch.nn.modules.distance.CosineSimilarity', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.distance.PairwiseDistance', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.dropout._DropoutNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.dropout.AlphaDropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout1d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout2d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.Dropout3d', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.dropout.FeatureAlphaDropout', class extends torch.nn.modules.dropout._DropoutNd {});
        this.registerType('torch.nn.modules.fold.Fold', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.fold.Unfold', class extends torch.nn.modules.module.Module {});
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
        this.registerType('torch.nn.modules.padding.ZeroPad1d', class {});
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
        this.registerType('torch.nn.modules.pooling.LPPool1d', class {});
        this.registerType('torch.nn.modules.pooling.LPPool2d', class {});
        this.registerType('torch.nn.modules.pooling._MaxPoolNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.pooling.MaxPool1d', class extends torch.nn.modules.pooling._MaxPoolNd {});
        this.registerType('torch.nn.modules.pooling.MaxPool2d', class extends torch.nn.modules.pooling._MaxPoolNd {});
        this.registerType('torch.nn.modules.pooling.MaxPool3d', class extends torch.nn.modules.pooling._MaxPoolNd {});
        this.registerType('torch.nn.modules.pooling._MaxUnpoolNd', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool1d', class extends torch.nn.modules.pooling._MaxUnpoolNd {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool2d', class extends torch.nn.modules.pooling._MaxUnpoolNd {});
        this.registerType('torch.nn.modules.pooling.MaxUnpool3d', class extends torch.nn.modules.pooling._MaxUnpoolNd {});
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
        this.registerType('torch.ao.nn.quantized.modules.activation.ELU', class extends torch.nn.modules.activation.ELU {});
        this.registerType('torch.ao.nn.quantized.modules.activation.Hardswish', class extends torch.nn.modules.activation.Hardswish {});
        this.registerType('torch.ao.nn.quantized.modules.activation.MultiheadAttention', class extends torch.ao.nn.quantizable.modules.activation.MultiheadAttention {});
        this.registerType('torch.ao.nn.quantized.modules.activation.PReLU', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.activation.ReLU6', class extends torch.nn.modules.activation.ReLU {});
        this.registerType('torch.ao.nn.quantized.modules.activation.LeakyReLU', class extends torch.nn.modules.activation.LeakyReLU {});
        this.registerType('torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.batchnorm._BatchNorm',  class extends torch.nn.modules.batchnorm._BatchNorm {});
        this.registerType('torch.ao.nn.quantized.modules.batchnorm.BatchNorm2d', class extends torch.ao.nn.quantized.modules.batchnorm._BatchNorm {});
        this.registerType('torch.ao.nn.quantized.modules.conv.Conv1d', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.conv.Conv2d', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.conv._ConvNd', class extends torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule {});
        this.registerType('torch.ao.nn.quantized.modules.conv._ConvTransposeNd', class extends torch.ao.nn.quantized.modules.conv._ConvNd {});
        this.registerType('torch.ao.nn.quantized.modules.conv.ConvTranspose1d', class extends torch.ao.nn.quantized.modules.conv._ConvTransposeNd {});
        this.registerType('torch.ao.nn.quantized.modules.conv.ConvTranspose2d', class extends torch.ao.nn.quantized.modules.conv._ConvTransposeNd {});
        this.registerType('torch.ao.nn.quantized.modules.Quantize', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.DeQuantize', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.dropout.Dropout', class extends torch.nn.modules.dropout.Dropout {});
        this.registerType('torch.ao.nn.quantized.modules.embedding_ops.Embedding', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.embedding_ops.EmbeddingPackedParams', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.FloatFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.QFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.functional_modules.FXFloatFunctional', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.linear.Linear', class extends torch.ao.nn.quantized.modules.utils.WeightedQuantizedModule {});
        this.registerType('torch.ao.nn.quantized.modules.linear.LinearPackedParams', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.modules.normalization.LayerNorm', class extends torch.nn.modules.normalization.LayerNorm {});
        this.registerType('torch.ao.nn.quantized.modules.normalization.InstanceNorm1d', class extends torch.nn.modules.instancenorm.InstanceNorm1d {});
        this.registerType('torch.ao.nn.quantized.modules.normalization.InstanceNorm2d', class extends torch.nn.modules.instancenorm.InstanceNorm2d {});
        this.registerType('torch.ao.nn.quantized.modules.normalization.InstanceNorm3d', class extends torch.nn.modules.instancenorm.InstanceNorm3d {});
        this.registerType('torch.ao.nn.quantized.modules.rnn.LSTM', class {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.linear.Linear', class extends torch.ao.nn.quantized.modules.linear.Linear {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.PackedParameter', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.GRU', class extends torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase {});
        this.registerType('torch.ao.nn.quantized.dynamic.modules.rnn.LSTM', class extends torch.ao.nn.quantized.dynamic.modules.rnn.RNNBase {});
        this.registerType('torch.ao.nn.quantized.reference.modules.conv.Conv1d', class {});
        this.registerType('torch.ao.nn.quantized.reference.modules.conv.Conv2d', class {});
        this.registerType('torch.ao.nn.quantized.reference.modules.linear.Linear', class {});
        this.registerType('torch.ao.nn.qat.modules.conv.Conv2d', class {});
        this.registerType('torch.ao.nn.qat.modules.linear.Linear', class {});
        this.registerType('torch.ao.nn.intrinsic.quantized.modules.conv_relu.ConvReLU2d', class extends torch.ao.nn.quantized.modules.conv.Conv2d {});
        this.registerType('torch.ao.nn.intrinsic.quantized.modules.linear_relu.LinearReLU', class extends torch.ao.nn.quantized.modules.linear.Linear {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused._FusedModule', class extends torch.nn.modules.container.Sequential {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvBn2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvReLU1d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvReLU2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.LinearReLU', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.modules.fused.ConvBnReLU2d', class extends torch.ao.nn.intrinsic.modules.fused._FusedModule {});
        this.registerType('torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d', class {});
        this.registerType('torch.nn.utils.prune.CustomFromMask', class {});
        this.registerType('torch.nn.utils.prune.L1Unstructured', class {});
        this.registerType('torch.nn.utils.prune.LnStructured', class {});
        this.registerType('torch.nn.utils.prune.PruningContainer', class {});
        this.registerType('torch.nn.utils.prune.RandomUnstructured', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNorm', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormStateDictHook', class {});
        this.registerType('torch.nn.utils.spectral_norm.SpectralNormLoadStateDictPreHook', class {});
        this.registerType('torch.nn.utils.weight_norm.WeightNorm', class {});
        this.registerFunction('torch.nn.utils.parametrize.type_before_parametrizations');
        this.registerType('torch.nn.utils.parametrize.ParametrizationList', class extends torch.nn.modules.container.ModuleList {});
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
        this.registerFunction('torch.quantization.fx.graph_module._save_packed_weight');
        this.registerType('torch.quantization.observer._PartialWrapper', class {});
        this.registerType('torch.quantization.observer.HistogramObserver', class {});
        this.registerType('torch.quantization.observer.MinMaxObserver', class {});
        this.registerType('torch.quantization.observer.MovingAverageMinMaxObserver', class {});
        this.registerType('torch.quantization.observer.MovingAveragePerChannelMinMaxObserver', class {});
        this.registerFunction('torch.quantization.observer._with_args');
        this.registerType('torch.quantization.qconfig.QConfig', class {});
        this.registerType('torch.quantization.stubs.DeQuantStub', class {});
        this.registerType('torch.quantization.stubs.QuantStub', class {});
        this.registerType('torch.utils._pytree.LeafSpec', class {});
        this.registerType('torch.utils._pytree.TreeSpec', class {});
        this.registerFunction('torch.utils._pytree.tree_map');
        this.registerFunction('torch.utils.checkpoint.checkpoint');
        this.registerType('torch.utils.data.dataloader._MultiProcessingDataLoaderIter', class {});
        this.registerType('torch.utils.data.dataloader.DataLoader', class {});
        this.registerFunction('torch.utils.data._utils.collate.default_collate');
        torch.utils.data.dataloader.default_collate = torch.utils.data._utils.collate.default_collate;
        this.registerType('torch.utils.data.dataset.Subset', class {});
        this.registerType('torch.utils.data.dataset.ConcatDataset', class {});
        this.registerType('torch.utils.data.dataset.TensorDataset', class {});
        this.registerType('torch.utils.data.sampler.BatchSampler', class {});
        this.registerType('torch.utils.data.sampler.RandomSampler', class {});
        this.registerType('torch.utils.data.sampler.SequentialSampler', class {});
        this.registerType('torch.utils.data.sampler.SubsetRandomSampler', class {});
        torch.nn.Sequential = torch.nn.modules.container.Sequential;
        this.registerType('torch.fx.experimental.symbolic_shapes.ShapeEnv', class {
            create_symintnode(/* sym, hint, source */) {
                return new torch.SymInt();
            }
        });
        this.registerType('torch.fx.proxy.TracerBase', class {
            constructor() {
                this.traced_func_name = 'forward';
            }
        });
        this.registerType('torch.fx._symbolic_trace.Tracer', class extends torch.fx.proxy.TracerBase {
            trace(root /*, concrete_args */) {
                let fn = null;
                if (root instanceof torch.nn.Module) {
                    // torch.fx._lazy_graph_module._LazyGraphModule.force_recompile(root)
                    this.root = root;
                    fn = builtins.getattr(new builtins.type(root), this.traced_func_name);
                    this.root_module_name = root._get_name();
                    this.submodule_paths = new builtins.dict(root.named_modules());
                } else {
                    this.root = new torch.nn.Module();
                    fn = root;
                }
                const tracer_cls = builtins.getattr(this, '__class__', null);
                this.graph = new torch.fx.graph.Graph(null, tracer_cls);
                if (builtins.hasattr(this, '__code__')) {
                    const code = fn.__code__;
                    this.graph._co_fields = {
                        co_name: code.co_name,
                        co_filename: code.co_filename,
                        co_firstlineno: code.co_firstlineno,
                    };
                }
                return this.graph;
            }
            is_leaf_module(m /*, module_qualified_name */) {
                return (m.__module__.startsWith('torch.nn') || m.__module__.startsWith('torch.ao.nn')) && m instanceof torch.nn.Sequential === false;
            }
        });
        this.registerType('torch.fx.experimental.proxy_tensor.PythonKeyTracer', class extends torch.fx._symbolic_trace.Tracer {});
        this.registerType('torch.fx.experimental.proxy_tensor._ModuleStackTracer', class extends torch.fx.experimental.proxy_tensor.PythonKeyTracer {});
        this.registerFunction('torch.fx._lazy_graph_module._make_graph_module', (...args) => {
            const graph_module_cls = args.pop() || torch.fx.graph_module.GraphModule;
            return new graph_module_cls(...args);
        });
        this.registerFunction('torch.fx.graph_module._deserialize_graph_module', (forward, body, graph_module_cls) => {
            let tracer_cls = body.get('_tracer_cls');
            if (!tracer_cls) {
                tracer_cls = torch.fx._symbolic_trace.Tracer;
            }
            const graphmodule_cls_name = body.get('_graphmodule_cls_name', 'GraphModule');
            const cls_tracer = tracer_cls;
            const KeepModules = class extends cls_tracer {
                is_leaf_module() {
                    return true;
                }
            };
            const com = new torch.fx.graph_module._CodeOnlyModule(body);
            const tracer_extras = body.get('_tracer_extras', new builtins.dict());
            const graph = new KeepModules().trace(com, tracer_extras);
            graph._tracer_cls = tracer_cls;
            const gm = torch.fx._lazy_graph_module._make_graph_module(com, graph, graphmodule_cls_name, graph_module_cls);
            for (const [k, v] of body.items()) {
                if (!builtins.hasattr(gm, k)) {
                    builtins.setattr(gm, k, v);
                }
            }
            return gm;
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
            let fn_src = null;
            if (body.has('_code')) {
                fn_src = body.get('_code');
            } else if (body.has('code')) {
                fn_src = body.get('code');
            } else {
                fn_src = body._code || body.code;
            }
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
                this._input_nodes = new builtins.dict();
                this.__update_args_kwargs(args, kwargs);
                this.users = new builtins.dict();
                this.type = return_type;
                this._prev = this;
                this._next = this;
                this._erased = false;
                this._repr_fn = null;
                this.meta = new builtins.dict();
            }
            get args() {
                return this._args;
            }
            get kwargs() {
                return this._kwargs;
            }
            get next() {
                return this._next;
            }
            prepend(x) {
                x._remove_from_list();
                const p = this._prev;
                [p._next, x._prev] = [x, p];
                [x._next, this._prev] = [this, x];
            }
            _remove_from_list() {
                const [p, n] = [this._prev, this._next];
                [p._next, n._prev] = [n, p];
            }
            __update_args_kwargs(new_args, new_kwargs) {
                const update_users_and_input_nodes = (n) => {
                    if (n instanceof torch.fx.node.Node) {
                        this._input_nodes.setdefault(n);
                        n.users.setdefault(this);
                    }
                    return n;
                };
                const map_aggregate = (a, fn) => {
                    if (a instanceof builtins.tuple) {
                        const t = new builtins.tuple(a.map((elem) => map_aggregate(elem, fn)));
                        if (!builtins.hasattr(a, '_fields')) {
                            return t;
                        }
                        throw new python.Error('Not implemented.');
                        // return type(a)(*t);
                    } else if (Array.isArray(a)) {
                        return a.map((elem) => map_aggregate(elem, fn));
                    } else if (a instanceof builtins.dict) {
                        const rv = new builtins.dict();
                        for (const [k, v] of a) {
                            rv.__setitem__(k, map_aggregate(v, fn));
                        }
                        return rv;
                    } else if (a instanceof builtins.slice) {
                        throw new python.Error('Not implemented.');
                        // return slice(map_aggregate(a.start, fn), map_aggregate(a.stop, fn), map_aggregate(a.step, fn))
                    }
                    return fn(a);
                };
                for (const old_use of this._input_nodes.keys()) {
                    old_use.users.pop(this);
                }
                // object.__setattr__(self, "_input_nodes", {})
                this._input_nodes = new builtins.dict();
                // object.__setattr__(self, "_args", map_aggregate(new_args, update_users_and_input_nodes))
                this._args = map_aggregate(new_args, update_users_and_input_nodes);
                // object.__setattr__(self, "_kwargs", map_aggregate(new_kwargs, update_users_and_input_nodes))
                this._kwargs = map_aggregate(new_kwargs, update_users_and_input_nodes);
            }
        });
        torch.fx.Node = torch.fx.node.Node;
        torch.fx.graph.Node = torch.fx.node.Node;
        this.registerType('torch.fx.graph.Graph', class {
            constructor(owning_module, tracer_cls, tracer_extras) {
                this._root = new torch.fx.node.Node(self, '', 'root', '', new builtins.list(), new builtins.dict());
                this._used_names = new Map();
                this._len = 0;
                this._graph_namespace = new torch.fx.graph._Namespace();
                this._owning_module = owning_module;
                this._tracer_cls = tracer_cls;
                this._tracer_extras = tracer_extras;
                // this._codegen = CodeGen()
                // this._co_fields = {}
            }
            get nodes() {
                const array = new Array(this._len);
                let node = this._root.next;
                for (let i = 0; node !== this._root; i++) {
                    array[i] = node;
                    node = node.next;
                }
                return array;
            }
            placeholder(name, type_expr /*, default_value */) {
                const args = []; // () if default_value is inspect.Signature.empty else (default_value,)
                const kwargs = new builtins.dict();
                return this.create_node('placeholder', name, args, kwargs, type_expr);
            }
            create_node(op, target, args, kwargs, name, type_expr) {
                args = args || new builtins.tuple();
                kwargs = kwargs || new builtins.dict();
                const candidate = name || this._target_to_str(target);
                name = this._graph_namespace.create_name(candidate, null);
                const n = new torch.fx.node.Node(this, name, op, target, args, kwargs, type_expr);
                this._graph_namespace.associate_name_with_obj(name, n);
                this._insert(n);
                this._len += 1;
                return n;
            }
            _insert(n) {
                this._root.prepend(n);
            }
            output(result, type_expr) {
                return this.create_node('output', 'output', new builtins.tuple(result), null, type_expr);
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
        this.registerType('torch.fx.graph_module._CodeOnlyModule', class extends torch.nn.modules.module.Module {
            constructor(body) {
                super();
                for (const [k, v] of body.items()) {
                    builtins.setattr(this, k, v);
                }
            }
        });
        this.registerType('torch.fx.graph_module.GraphModule', class extends torch.nn.modules.module.Module {
            constructor(root, graph, class_name) {
                super();
                this.__class__.__name__ = class_name || 'GraphModule';
                this.graph = graph;
            }
        });
        torch.fx.Graph = torch.fx.graph.Graph;
        torch.fx.GraphModule = torch.fx.graph_module.GraphModule;
        this.registerType('torch.fx.immutable_collections.immutable_dict', class extends builtins.dict {});
        this.registerFunction('torch.fx._symbolic_trace.wrap', (fn_or_name) => {
            return fn_or_name;
        });
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
        this.registerType('torchvision.ops.roi_align.RoIAlign', class {});
        this.registerType('torchvision.ops.stochastic_depth.StochasticDepth', class {});
        this.registerType('torchvision.models._api.Weights', class {});
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
        this.registerType('torchvision.models.efficientnet.EfficientNet_B3_Weights', class {});
        this.registerType('torchvision.models.efficientnet.FusedMBConv', class {});
        this.registerType('torchvision.models.efficientnet.MBConv', class {});
        this.registerType('torchvision.models.feature_extraction.LeafModuleAwareTracer', class extends torch.fx._symbolic_trace.Tracer {});
        this.registerType('torchvision.models.feature_extraction.NodePathTracer', class extends torchvision.models.feature_extraction.LeafModuleAwareTracer {});
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
        this.registerType('torchvision.models.quantization.mobilenetv3.QuantizableMobileNetV3', class {});
        this.registerType('torchvision.models.quantization.mobilenetv3.QuantizableInvertedResidual', class {});
        this.registerType('torchvision.models.quantization.mobilenetv3.QuantizableSqueezeExcitation', class {});
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
        this.registerType('torchvision.models.segmentation.lraspp.LRASPP', class {});
        this.registerType('torchvision.models.segmentation.lraspp.LRASPPHead', class {});
        this.registerType('torchvision.models.shufflenetv2.ShuffleNetV2', class {});
        this.registerType('torchvision.models.shufflenetv2.InvertedResidual', class {});
        this.registerType('torchvision.models.squeezenet.Fire', class {});
        this.registerType('torchvision.models.squeezenet.SqueezeNet', class {});
        this.registerType('torchvision.models.swin_transformer.PatchMerging', class {});
        this.registerType('torchvision.models.swin_transformer.PatchMergingV2', class {});
        this.registerType('torchvision.models.swin_transformer.ShiftedWindowAttention', class {});
        this.registerType('torchvision.models.swin_transformer.ShiftedWindowAttentionV2', class {});
        this.registerType('torchvision.models.swin_transformer.SwinTransformer', class {});
        this.registerType('torchvision.models.swin_transformer.SwinTransformerBlock', class {});
        this.registerType('torchvision.models.swin_transformer.SwinTransformerBlockV2', class {});
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
        this.registerType('torchvision.transforms._presets.ImageClassification', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.autoaugment.AutoAugment', class {});
        this.registerType('torchvision.transforms.autoaugment.AutoAugmentPolicy', class {});
        this.registerType('torchvision.transforms.autoaugment.AugMix', class {});
        this.registerType('torchvision.transforms.functional.InterpolationMode', class {});
        this.registerFunction('torchvision.transforms.functional.adjust_brightness');
        this.registerFunction('torchvision.transforms.functional.adjust_contrast');
        this.registerFunction('torchvision.transforms.functional.adjust_brightness');
        this.registerFunction('torchvision.transforms.functional.adjust_contrast');
        this.registerFunction('torchvision.transforms.functional.adjust_gamma');
        this.registerFunction('torchvision.transforms.functional.adjust_hue');
        this.registerFunction('torchvision.transforms.functional.adjust_saturation');
        this.registerType('torchvision.transforms.transforms.ColorJitter', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Compose', class {});
        this.registerType('torchvision.transforms.transforms.ConvertImageDtype', class {});
        this.registerType('torchvision.transforms.transforms.CenterCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.GaussianBlur', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Grayscale', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Lambda', class {});
        this.registerType('torchvision.transforms.transforms.Normalize', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.PILToTensor', class {});
        this.registerType('torchvision.transforms.transforms.RandomAffine', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomApply', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomErasing', class {});
        this.registerType('torchvision.transforms.transforms.RandomHorizontalFlip', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomVerticalFlip', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomResizedCrop', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.RandomRotation', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Resize', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.Scale', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.transforms.ToPILImage', class {});
        this.registerType('torchvision.transforms.transforms.ToTensor', class {});
        this.registerType('torchvision.transforms.v2._container.Compose', class {});
        this.registerType('torchvision.transforms.v2._misc.ConvertImageDtype', class {});
        this.registerType('torchvision.transforms.v2._misc.Normalize', class {});
        this.registerType('torchvision.transforms.v2._misc.ToDtype', class {});
        this.registerType('torchvision.transforms.v2._geometry.CenterCrop', class {});
        this.registerType('torchvision.transforms.v2._geometry.Resize', class {});
        this.registerType('torchvision.transforms.v2._geometry.Pad', class {});
        this.registerType('torchvision.transforms.v2._geometry.RandomCrop', class {});
        this.registerType('torchvision.transforms.v2._transform.Transform', class extends torch.nn.modules.module.Module {});
        this.registerType('torchvision.transforms.v2._type_conversion.ToImage', class extends torchvision.transforms.v2._transform.Transform {});
        this.registerType('torchvision.transforms.v2._type_conversion.PILToTensor', class {});
        this.registerFunction('torchvision.models.resnet.resnet18', () => {});
        this.registerFunction('torchvision.models.resnet.resnet34', () => {});
        this.registerFunction('torchvision.models.resnet.resnet50', () => {});
        this.registerFunction('torchvision.models.resnet.resnet101', () => {});
        this.registerFunction('torchvision.models.resnet.resnet152', () => {});
        this.registerFunction('torchvision.models.vision_transformer.vit_h_14', () => {});
        this.registerFunction('torchvision.ops.boxes.box_iou');
        this.registerFunction('builtins.annotate', (type, value) => {
            if (type === builtins.int) {
                return Number.isInteger(value) ? value : NaN;
            }
            if (type === builtins.float) {
                return typeof value === 'number' ? value : NaN;
            }
            if (type === builtins.number) {
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
        builtins.xrange = builtins.range;
        this.registerFunction('torch._C._nn.gelu');
        this.registerFunction('torch._C._nn.avg_pool2d');
        this.registerFunction('torch._C._nn.scaled_dot_product_attention');
        this.registerFunction('torch._C._nn.softplus');
        this.registerFunction('torch._native_multi_head_attention');
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
            const storage = execution.invoke('torch.storage.TypedStorage', [obj.size, dtype]);
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
        this.registerFunction('torch._utils._rebuild_meta_tensor_no_storage', (dtype, size, stride, requires_grad) => {
            return torch.empty_strided(size, stride, dtype, null, 'meta', false, requires_grad);
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
        this.registerFunction('torch._utils._rebuild_tensor_v3');
        this.registerFunction('torch._utils._rebuild_parameter', (data, requires_grad, backward_hooks) => {
            const param = new torch.nn.parameter.Parameter(data, requires_grad);
            param.backward_hooks = backward_hooks;
            return param;
        });
        this.registerFunction('torch._utils._rebuild_parameter_v2', (data, requires_grad, backward_hooks, state) => {
            const param = new torch.nn.parameter.Parameter(data, requires_grad);
            param.backward_hooks = backward_hooks;
            torch._utils._set_obj_state(param, state);
            return param;
        });
        this.registerFunction('torch._utils._rebuild_parameter_with_state', (data, requires_grad, backward_hooks, state) => {
            const _set_obj_state = (obj, state) => {
                const [dict_state, slots_state] = Array.isArray(state) ? state : [state, null];
                if (dict_state) {
                    for (const [k, v] of Object.entries(dict_state)) {
                        builtins.setattr(obj, k, v);
                    }
                }
                if (slots_state) {
                    for (const [k, v] of Object.entries(slots_state)) {
                        builtins.setattr(obj, k, v);
                    }
                }
            };
            const param = new torch.nn.parameter.Parameter(data, requires_grad);
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
                    builtins.setattr(obj, name, value);
                }
            }
            if (slots_state) {
                for (const [name, value] of Object.entries(slots_state)) {
                    builtins.setattr(obj, name, value);
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
            return builtins.hasattr(dict, key);
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
            if (Number.isInteger(value)) {
                value = Boolean(value);
            }
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
        this.registerFunction('torch.get_default_dtype', () => {
            torch._default_type = torch._default_type || torch.float32;
            return torch._default_type;
        });
        this.registerFunction('torch.set_default_dtype', (value) => {
            torch._default_type = value;
        });
        this.registerFunction('torch._prims_common.dtype_or_default', (value) => {
            return value || torch.get_default_dtype();
        });
        this.registerFunction('torch.empty_strided', (size, stride, dtype /*, layout, device, pin_memory, requires_grad */) => {
            const shape = size;
            dtype = torch._prims_common.dtype_or_default(dtype);
            size = shape.reduce((a, b) => a * b, 1);
            const storage = execution.invoke('torch.storage.TypedStorage', [size, dtype]);
            const tensor = execution.invoke('torch.Tensor', []);
            tensor.__setstate__([storage, 0, shape, stride]);
            return tensor;
        });
        this.registerFunction('torch.add', (left, right) => {
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left + right;
            }
            if (Array.isArray(left) && Array.isArray(right)) {
                return left.concat(right);
            }
            if (typeof left === 'string' && typeof right === 'string') {
                return left + right;
            }
            throw new python.Error('Unsupported torch.add expression type.');
        });
        this.registerFunction('torch.all', (input) => {
            if (Array.isArray(input) && input.length === 0) {
                return true;
            }
            throw new python.Error(`Unsupported 'torch.all' expression type.`);
        });
        this.registerFunction('torch.append', (list, value) => {
            list.push(value);
            return value;
        });
        this.registerFunction('torch.clear', (value) => {
            if (value instanceof torch.Value) {
                throw new python.Error('Invalid value.');
            }
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
        this.registerFunction('torch.replace', (value, oldvalue, newvalue /*, max */) => {
            return value.replace(oldvalue, newvalue);
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
        this.registerFunction('torch.strip', (self, chars) => {
            chars = chars || '\\n\\t\\f\\v';
            const regex = new RegExp(`[${chars}]`, 'g');
            return self.replace(regex, '');
        });
        this.registerFunction('torch.gt', (left, right) => {
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
        this.registerFunction('torch.is_autocast_enabled', () => {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
                this._version = 0;
                const stream = this.get_record('.data/version') || this.get_record('version') || null;
                if (stream) {
                    const decoder = new TextDecoder('utf-8');
                    const buffer = stream.peek();
                    const text = decoder.decode(buffer);
                    this._version = Number(text.split('\n').shift().trim());
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
                if (sys_info.get('protocol_version') !== 1001) {
                    throw new python.Error(`Unsupported protocol version '${sys_info.protocol_version}'.`);
                }
                if (sys_info.get('little_endian') === false) {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left < right;
            }
            throw new python.Error("Unsupported 'torch.lt' expression type.");
        });
        this.registerFunction('torch.mul', (left, right) => {
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left * right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            if (Array.isArray(left) && left.every((value) => typeof value === 'number' || value instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left.map((value) => value * right);
            }
            throw new python.Error("Unsupported 'torch.mul' expression type.");
        });
        this.registerFunction('torch.div', (left, right) => {
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left / right;
            }
            if (isNaN(left) || isNaN(right)) {
                return NaN;
            }
            throw new python.Error("Unsupported 'torch.div' expression type.");
        });
        this.registerFunction('torch.round', (value) => {
            if (typeof value === 'number' || value instanceof Number) {
                return Math.round(value);
            }
            if (isNaN(value)) {
                return value;
            }
            throw new python.Error("Unsupported 'torch.round' expression type.");
        });
        this.registerFunction('torch.remainder', (left, right) => {
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
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
            if ((typeof left === 'number' || left instanceof Number) && (typeof right === 'number' || right instanceof Number)) {
                return left - right;
            }
            throw new python.Error("Unsupported 'torch.sub' expression type.");
        });
        this.registerFunction('torch.sym_int');
        this.registerFunction('torch.sym_ite');
        this.registerFunction('torch.sym_max');
        this.registerFunction('torch.sym_min');
        this.registerFunction('torch.sym_not');
        this.registerFunction('torch.sym_sqrt');
        this.registerFunction('torch.sym_sqrt');
        this.registerFunction('torch.functional.einsum');
        this.registerFunction('torch.functional.norm');
        this.registerFunction('torch.functional.split');
        this.registerFunction('torch.nn.init.constant_');
        this.registerFunction('torch.nn.init.normal_');
        this.registerFunction('torch.nn.init.xavier_uniform_');
        this.registerFunction('torch.nn.functional.adaptive_avg_pool2d');
        this.registerFunction('torch.nn.functional.binary_cross_entropy');
        this.registerFunction('torch.nn.functional.binary_cross_entropy_with_logits');
        this.registerFunction('torch.nn.functional.cross_entropy');
        this.registerFunction('torch.nn.functional.elu');
        this.registerFunction('torch.nn.functional.gelu');
        this.registerFunction('torch.nn.functional.hardsigmoid');
        this.registerFunction('torch.nn.functional.hardswish');
        this.registerFunction('torch.nn.functional.hardtanh');
        this.registerFunction('torch.nn.functional.interpolate');
        this.registerFunction('torch.nn.functional.leaky_relu');
        this.registerFunction('torch.nn.functional.l1_loss');
        this.registerFunction('torch.nn.functional.linear');
        this.registerFunction('torch.nn.functional.log_softmax');
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
        this.registerType('torch._ops.OperatorBase', class {
            constructor() {
                this.functorch_table = {};
            }
        });
        this.registerType('torch._ops.HigherOrderOperator', class extends torch._ops.OperatorBase {
            constructor(name, cacheable) {
                super();
                this._name = name;
                this.__name__ = name;
                // _higher_order_ops[name] = this;
                this._ns = 'higher_order';
                this.__module__ = 'torch.ops.higher_order';
                this._cacheable = cacheable;
            }

        });
        this.registerType('torch.Type', class {
            constructor(kind, annotation_str) {
                this._kind = kind;
                if (annotation_str) {
                    this._annotation_str = annotation_str;
                }
            }
            static get(kind, annotation_str) {
                return new torch.Type(kind, annotation_str);
            }
            kind() {
                return this._kind;
            }
            get annotation_str() {
                return this._annotation_str;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isSubtypeOf(rhs) {
                if (rhs.kind() === 'AnyType' || this === rhs) {
                    return true;
                }
                if (rhs.kind() === 'OptionalType' && this.kind() !== 'OptionalType') {
                    return rhs.getElementType().equals(this);
                }
                if (rhs instanceof torch.UnionType) {
                    throw new python.Error('Not implemented.');
                }
                if (rhs instanceof torch._C.DynamicType) {
                    throw new python.Error('Not implemented.');
                }
                return this.equals(rhs);
            }
            containedTypes() {
                return [];
            }
            withContained(contained_types) {
                const current_contained = this.containedTypes();
                if (current_contained.length === 0 || current_contained.length !== contained_types.length) {
                    throw new python.Error('Invalid contained types.');
                }
                if (current_contained.length === contained_types.length && current_contained.every((x, index) => x.equals(contained_types[index]))) {
                    return this;
                }
                return this.createWithContained(contained_types);
            }
            isUnionType() {
                return false;
            }
            hasFreeVariables() {
                return false;
            }
            is_module() {
                return false;
            }
            expect(type) {
                if (this instanceof type === false) {
                    throw new python.Error(`Expected '${type.kind()}' but got '${this.kind()}'.`);
                }
                return this;
            }
            str() {
                if (this._kind === 'VarType' && this._annotation_str) {
                    return this._annotation_str;
                } else if (this._kind === 'ScalarTypeType') {
                    return 'ScalarType';
                } else if (this._kind === 'QSchemeType') {
                    return 'QScheme';
                } else if (this._kind) {
                    return this._kind;
                }
                throw new python.Error(`Not implemented '${this.kind()}'.`);
            }
            __str__() {
                return this.str();
            }
            toString() {
                return this.__str__();
            }
        });
        this.registerType('torch.ClassType', class extends torch.Type {
            constructor(qualified_name, cu, is_module) {
                super('ClassType', typeof qualified_name === 'string' ? qualified_name : qualified_name.qualifiedName());
                this._is_module = is_module;
                this._attributes = [];
                this._attributeTypes = [];
                this._methods = new Map();
                this._staticmethods = new Map();
                this._constants = new Map();
            }
            static create(qualifiedName, cu, is_module /*, doc_string, unresolved_class_attributes */) {
                return new torch.ClassType(qualifiedName, cu, is_module);
            }
            qualified_name() {
                return this.annotation_str;
            }
            name() {
                return this._qualified_name.split('.').pop();
            }
            is_module() {
                return this._is_module;
            }
            is_parameter(slot) {
                return this._attributes[slot].is_parameter === true;
            }
            is_buffer(slot) {
                return this._attributes[slot].is_buffer === true;
            }
            addMethod(func) {
                this._methods.set(func.name(), func);
            }
            findMethod(name) {
                return this._methods.get(name);
            }
            getMethod(name) {
                const method = this.findMethod(name);
                if (!method) {
                    throw new python.Error(`Method '${name}' not found on class '${this.str()}.`);
                }
                return method;
            }
            addStaticMethod(func) {
                this._staticmethods.set(func.name, func);
            }
            findStaticMethod(name) {
                return this._staticmethods.get(name);
            }
            numAttributes() {
                return this._attributes.length;
            }
            addAttribute(name, type, is_parameter, is_buffer) {
                is_parameter = is_parameter || false;
                is_buffer = is_buffer || false;
                const slot = this._attributes.length;
                this._attributes.push({ name, type, is_parameter, is_buffer });
                this._attributeTypes.push(type);
                return slot;
            }
            addOrCheckAttribute(name, ty, is_parameter, is_buffer) {
                is_parameter = is_parameter || false;
                is_buffer = is_buffer || false;
                const slot_idx = this.findAttributeSlot(name);
                if (slot_idx === null) {
                    return this.addAttribute(name, ty, is_parameter, is_buffer);
                }
                // TORCH_CHECK(is_parameter == this.is_parameter(*slot_idx), "Parameter field mismatch for the field '", name, "'");
                // const TypePtr& atype = getAttribute(*slot_idx);
                // TORCH_CHECK(ty.isSubtypeOf(*atype), ty.repr_str(), " is not compatible with the type ", atype.repr_str(), " for the field '", name, "'");
                return slot_idx;
            }
            findAttributeSlot(name) {
                for (let pos = 0; pos < this._attributes.length; pos++) {
                    if (name === this._attributes[pos].name) {
                        return pos;
                    }
                }
                return null;
            }
            findAttribute(name) {
                const slot = this.findAttributeSlot(name);
                if (slot !== null) {
                    return this._attributes[slot].type;
                }
                return null;
            }
            hasAttribute(name) {
                return this._attributes.find((attr) => attr.name === name);
            }
            getAttribute(arg) {
                const slot = Number.isInteger(arg) ? arg : this.findAttributeSlot(arg);
                return this._attributes[slot].type;
            }
            getAttributeName(slot) {
                return this._attributes[slot].name;
            }
            hasConstant(/* name */) {
            }
            methods() {
                throw new python.Error('Not implemented.');
            }
            addConstant(name, value) {
                this._constants.set(name, value);
            }
            containedTypes() {
                return this._attributeTypes;
            }
            str() {
                return this.qualified_name();
            }
        });
        this.registerType('torch.OptionalType', class extends torch.Type {
            constructor(elem) {
                super('OptionalType');
                this._contained = elem;
            }
            static create(elem) {
                return new torch.OptionalType(elem);
            }
            getElementType() {
                return this._contained;
            }
            equals(rhs) {
                return this.kind() === rhs.kind() && this.getElementType().equals(rhs.getElementType());
            }
            containedTypes() {
                return [this._contained];
            }
            isUnionType() {
                return true;
            }
            str() {
                return `${this.getElementType().str()}?`;
            }
            __str__() {
                return `Optional[${this.getElementType().__str__()}]`;
            }
        });
        this.registerType('torch.ListType', class extends torch.Type {
            constructor(elem) {
                super('ListType');
                this._elem = elem;
            }
            static create(elem) {
                return new torch.ListType(elem);
            }
            getElementType() {
                return this._elem;
            }
            equals(rhs) {
                return this.kind() === rhs.kind() && this.getElementType().equals(rhs.getElementType());
            }
            isSubtypeOf(rhs) {
                if (super.isSubtypeOf(rhs)) {
                    return true;
                }
                if (rhs.kind() === 'AnyListType') {
                    return true;
                }
                return false;
            }
            containedTypes() {
                return [this._elem];
            }
            hasFreeVariables() {
                return this.getElementType().hasFreeVariables();
            }
            str() {
                return `${this.getElementType().str()}[]`;
            }
            __str__() {
                return `List[${this.getElementType().__str__()}]`;
            }
        });
        this.registerType('torch.FutureType', class extends torch.Type {
            constructor(elem) {
                super('FutureType');
                this._elem = elem;
            }
            static get(elem) {
                return new torch.FutureType(elem);
            }
            getElementType() {
                return this._elem;
            }
            containedTypes() {
                throw new python.Error('Not implemented.');
            }
            str() {
                return `Future(${this.getElementType().str()})`;
            }
            __str__() {
                return `Future[${this.getElementType().__str__()}]`;
            }
        });
        this.registerType('torch.RRefType', class extends torch.Type {
            constructor(elem) {
                super('RRefType');
                this._elem = elem;
            }
            get(elem) {
                return new torch.RRefType(elem);
            }
            getElementType() {
                return this._elem;
            }
            containedTypes() {
                throw new python.Error('Not implemented.');
            }
            str() {
                return `RRef(${this.getElementType().str()})`;
            }
            __str__() {
                return `RRef[${this.getElementType().__str__()}]`;
            }
        });
        this.registerType('torch.AwaitType', class extends torch.Type {
            constructor(elem) {
                super('AwaitType');
                this._elem = elem;
            }
            static get(elem) {
                return new torch.AwaitType(elem);
            }
            getElementType() {
                return this._elem;
            }
            containedTypes() {
                throw new python.Error('Not implemented.');
            }
            str() {
                return `Await(${this.getElementType().str()})`;
            }
            __str__() {
                return `Await[${this.getElementType().__str__()}]`;
            }
        });
        this.registerType('torch.TupleType', class extends torch.Type {
            constructor(elements, annotation_str, schema) {
                super('TupleType', annotation_str);
                this._elements = elements;
                this._schema = schema;
            }
            static create(elements) {
                return new torch.TupleType(elements);
            }
            static createNamed(qualified_name, field_names, field_types /*, field_defaults */) {
                const args = [];
                for (let i = 0; i < field_names.length; i++) {
                    const arg = new torch.Argument(field_names[i], field_types[i], field_types[i]);
                    args.push(arg);
                }
                const schema = new torch.FunctionSchema(qualified_name, args);
                return new torch.TupleType(field_types, qualified_name, schema);
            }
            elements() {
                return this._elements;
            }
            containedTypes() {
                throw new python.Error('Not implemented.');
            }
            schema() {
                return this._schema;
            }
            str() {
                if (this._schema) {
                    return `NamedTuple(...)`;
                }
                return `(${this.elements().map((elem) => elem.str()).join(', ')})`;
            }
            __str__() {
                if (this.annotation_str) {
                    return this.annotation_str;
                }
                return `Tuple[${this.elements().map((elem) => elem.__str__()).join(', ')}]`;
            }
        });
        this.registerType('torch.AnyType', class extends torch.Type {
            constructor() {
                super('AnyType');
            }
            static get() {
                torch.AnyType.value = torch.AnyType.value || new torch.AnyType();
                return torch.AnyType.value;
            }
            str() {
                return 'Any';
            }
            __str__() {
                return 'Any';
            }
        });
        this.registerType('torch.NoneType', class extends torch.Type {
            constructor() {
                super('NoneType');
            }
            static get() {
                torch.NoneType.value = torch.NoneType.value || new torch.NoneType();
                return torch.NoneType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isSubtypeOf(/* rhs */) {
                return true;
            }
            str() {
                return 'NoneType';
            }
            __str__() {
                return 'NoneType';
            }
        });
        this.registerType('torch.TensorType', class extends torch.Type {
            constructor() {
                super('TensorType');
                this._is_inferred = false;
            }
            static get() {
                torch.TensorType.value = torch.TensorType.value || new torch.TensorType();
                return torch.TensorType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isInferredType() {
                return this._is_inferred;
            }
            str() {
                return 'Tensor';
            }
            __str__() {
                return 'Tensor';
            }
        });
        this.registerType('torch.NumberType', class extends torch.Type {
            constructor() {
                super('NumberType');
            }
            static get() {
                torch.NumberType.value = torch.NumberType.value || new torch.NumberType();
                return torch.NumberType.value;
            }
            str() {
                return 'Scalar';
            }
            __str__() {
                return 'number';
            }
        });
        this.registerType('torch.BoolType', class extends torch.Type {
            constructor() {
                super('BoolType');
            }
            static get() {
                torch.BoolType.value = torch.BoolType.value || new torch.BoolType();
                return torch.BoolType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            str() {
                return 'bool';
            }
            __str__() {
                return 'bool';
            }
        });
        this.registerType('torch.IntType', class extends torch.Type {
            constructor() {
                super('IntType');
            }
            static get() {
                torch.IntType.value = torch.IntType.value || new torch.IntType();
                return torch.IntType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isSubtypeOf(rhs) {
                return rhs.kind() === 'NumberType' || rhs.kind() === 'FloatType' || super.isSubtypeOf(rhs);
            }
            str() {
                return 'int';
            }
            __str__() {
                return 'int';
            }
        });
        this.registerType('torch.SymIntType', class extends torch.Type {
            constructor() {
                super('SymIntType');
            }
            static get() {
                torch.SymIntType.value = torch.SymIntType.value || new torch.SymIntType();
                return torch.SymIntType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            str() {
                return 'SymInt';
            }
            __str__() {
                return 'int';
            }
        });
        this.registerType('torch.FloatType', class extends torch.Type {
            constructor() {
                super('FloatType');
            }
            static get() {
                torch.FloatType.value = torch.FloatType.value || new torch.FloatType();
                return torch.FloatType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isSubtypeOf(rhs) {
                return this.kind() === 'NumberType' || super.isSubtypeOf(rhs);
            }
            str() {
                return 'float';
            }
            __str__() {
                return 'float';
            }
        });
        this.registerType('torch.StringType', class extends torch.Type {
            constructor() {
                super('StringType');
            }
            static get() {
                torch.StringType.value = torch.StringType.value || new torch.StringType();
                return torch.StringType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            str() {
                return 'str';
            }
            __str__() {
                return 'str';
            }
        });
        this.registerType('torch.ComplexType', class extends torch.Type {
            constructor() {
                super('ComplexType');
            }
            static get() {
                torch.ComplexType.value = torch.ComplexType.value || new torch.ComplexType();
                return torch.ComplexType.value;
            }
            equals(rhs) {
                return this.kind() === rhs.kind();
            }
            isSubtypeOf(rhs) {
                return this.kind() === 'NumberType' || super.isSubtypeOf(rhs);
            }
            str() {
                return 'complex';
            }
            __str__() {
                return 'complex';
            }
        });
        this.registerType('torch.DictType', class extends torch.Type {
            constructor(key, value) {
                super('DictType');
                this._key = key;
                this._value = value;
            }
            static create(key, value) {
                return new torch.DictType(key, value);
            }
            getKeyType() {
                return this._key;
            }
            getValueType() {
                return this._value;
            }
            hasFreeVariables() {
                return this.getKeyType().hasFreeVariables() || this.getValueType().hasFreeVariables();
            }
            containedTypes() {
                throw new python.Error('Not implemented.');
            }
            str() {
                return `Dict(${this.getKeyType().str()}, ${this.getValueType().str()})`;
            }
            __str__() {
                return `Dict(${this.getKeyType().__str__()}, ${this.getValueType().__str__()})`;
            }
        });
        this.registerType('torch.DeviceObjType', class extends torch.Type {
            constructor() {
                super('DeviceObjType');
            }
            static get() {
                torch.DeviceObjType.value ||= new torch.DeviceObjType();
                return torch.DeviceObjType.value;
            }
            str() {
                return 'Device';
            }
            __str__() {
                return 'Device';
            }
        });
        this.registerType('torch.StreamObjType', class extends torch.Type {
            constructor() {
                super('StreamObjType');
            }
            str() {
                return 'Stream';
            }
            __str__() {
                return 'Stream';
            }
        });
        this.registerType('torch._C._GeneratorType', class extends torch.Type {
            constructor() {
                super('GeneratorType');
            }
            static get() {
                torch._C._GeneratorType.value = torch._C._GeneratorType.value || new torch._C._GeneratorType();
                return torch._C._GeneratorType.value;
            }
            str() {
                return 'Generator';
            }
            __str__() {
                return 'Generator';
            }
        });
        this.registerType('torch.UnionType', class extends torch.Type {
            constructor() {
                super('UnionType');
            }
            isUnionType() {
                return true;
            }
        });
        this.registerType('torch.InterfaceType', class extends torch.Type {
            constructor() {
                super('InterfaceType');
            }
        });
        this.registerType('torch._C.DynamicType', class extends torch.Type {
            constructor() {
                super('DynamicType');
            }
        });
        this.registerType('torch._C.AliasInfo', class {
            constructor() {
                this.is_write = false;
                this.before_set = [];
                this.after_set = [];
                this.containedTypes = [];
            }
            addBeforeSet(value) {
                this.before_set.push(value);
            }
            addAfterSet(value) {
                this.after_set.push(value);
            }
            addContainedType(alias_info) {
                this.containedTypes.push(alias_info);
            }
            str() {
                const list = ['('];
                list.push(this.before_set.join('|'));
                if (this.after_set.length > 0) {
                    list.push(' -> ');
                    list.push(this.after_set.join('|'));
                }
                if (this.is_write) {
                    list.push('!');
                }
                list.push(')');
                return list.join('');
            }
        });
        this.registerType('torch._C.Lexer', class {
            constructor(buffer) {
                this.buffer = buffer;
                this.position = 0;
                this.value = '';
                this.next();
            }
            eat(kind) {
                if (this.kind !== kind) {
                    return null;
                }
                const value = this.value;
                this.next();
                return value;
            }
            expect(kind) {
                if (this.kind !== kind) {
                    throw new python.Error(`Unexpected '${this.kind}' instead of '${kind}'.`);
                }
                const value = this.value;
                this.next();
                return value;
            }
            whitespace(count) {
                if (this.kind !== ' ') {
                    if (count > this.value.length) {
                        throw new python.Error();
                    }
                    return false;
                }
                this.next();
                return true;
            }
            next() {
                this.position += this.value.length;
                let i = this.position;
                if (i >= this.buffer.length) {
                    this.kind = '\0';
                    this.value = '';
                } else if (this.buffer[i] === ' ') {
                    while (this.buffer[i] === ' ') {
                        i += 1;
                    }
                    this.kind = ' ';
                    this.value = this.buffer.slice(this.position, i);
                } else if (this.buffer[i] === '.' && this.buffer[i + 1] === '.' && this.buffer[i + 2] === '.') {
                    this.kind = '...';
                    this.value = '...';
                } else if (this.buffer[i] === '[' && this.buffer[i + 1] === ']') {
                    this.kind = '[]';
                    this.value = '[]';
                } else if (this.buffer[i] === '(' || this.buffer[i] === ')' || this.buffer[i] === ':' || this.buffer[i] === '.' || this.buffer[i] === '[' || this.buffer[i] === ']' || this.buffer[i] === ',' || this.buffer[i] === '=' || this.buffer[i] === '?' || this.buffer[i] === '!' || this.buffer[i] === '*' || this.buffer[i] === '|') {
                    this.kind = this.buffer[i];
                    this.value = this.buffer[i];
                } else if ((this.buffer[i] >= 'a' && this.buffer[i] <= 'z') || (this.buffer[i] >= 'A' && this.buffer[i] <= 'Z') || this.buffer[i] === '_') {
                    i += 1;
                    while (i < this.buffer.length && ((this.buffer[i] >= 'a' && this.buffer[i] <= 'z') || (this.buffer[i] >= 'A' && this.buffer[i] <= 'Z') || (this.buffer[i] >= '0' && this.buffer[i] <= '9') || this.buffer[i] === '_')) {
                        i += 1;
                    }
                    this.kind = 'id';
                    this.value = this.buffer.slice(this.position, i);
                } else if (this.buffer[i] === '-' && this.buffer[i + 1] === '>') {
                    this.kind = '->';
                    this.value = '->';
                } else if ((this.buffer[i] >= '0' && this.buffer[i] <= '9') || this.buffer[i] === '-') {
                    i += 1;
                    while (i < this.buffer.length && ((this.buffer[i] >= '0' && this.buffer[i] <= '9') || this.buffer[i] === '.' || this.buffer[i] === 'e' || this.buffer[i] === '-')) {
                        i += 1;
                    }
                    this.kind = '#';
                    this.value = this.buffer.slice(this.position, i);
                } else if (this.buffer[i] === "'" || this.buffer[i] === '"') {
                    const quote = this.buffer[i];
                    i += 1;
                    while (i < this.buffer.length && this.buffer[i] !== quote) {
                        i += (this.buffer[i] === '\\' && (this.buffer[i + 1] === "'" || this.buffer[i + 1] === '"' || this.buffer[i + 1] === '\\')) ? 2 : 1;
                    }
                    i += 1;
                    this.kind = 'string';
                    this.value = this.buffer.slice(this.position, i);
                } else {
                    throw new python.Error(`Unsupported token at '${this.position}'.`);
                }
            }
        });
        this.registerType('torch._C.SchemaTypeParser', class {
            constructor(L, complete_tensor_types, allow_typevars) {
                this.L = L;
                this.complete_tensor_types = complete_tensor_types;
                this._allow_typevars = allow_typevars;
            }
            parseType() {
                const r = this.parseFakeAndRealType();
                return { first: r[0], second: r[2] };
            }
            parseBaseType() {
                const L = this.L;
                const value = L.value;
                L.next();
                switch (value) {
                    case 'Tensor': return torch.TensorType.get();
                    case 'bool': return torch.BoolType.get();
                    case 'int': return torch.IntType.get();
                    case 'float': return torch.FloatType.get();
                    case 'complex': return torch.ComplexType.get();
                    case 'str': return torch.StringType.get();
                    case 'SymInt': return torch.SymIntType.get();
                    case 'Scalar': return torch.NumberType.get();
                    case 'ScalarType': return torch.Type.get('ScalarTypeType');
                    case 'Device': return torch.DeviceObjType.get();
                    case 'Layout': return torch.Type.get('Layout');
                    case 'MemoryFormat': return torch.Type.get('MemoryFormat');
                    case 'Generator': return torch._C._GeneratorType.get();
                    case 't': case 't1': case 't2': case 'tVal': return torch.Type.get('VarType', value);
                    case 'Any': return torch.AnyType.get();
                    case 'AnyEnumType': return torch.Type.get('AnyEnumType');
                    case 'Dimname': return torch.StringType.get();
                    case 'QScheme': return torch.Type.get('QSchemeType');
                    case 'Stream': return torch.StreamObjType.get();
                    case 'Storage': return torch.Type.get('Storage');
                    case 'AnyClassType': return torch.Type.get('AnyClassType');
                    case 'NoneType': return torch.NoneType.get();
                    default: throw new python.Error(`Unsupported type '${value}'.`);
                }
            }
            parseFakeAndRealType() {
                const L = this.L;
                let fake_value = null;
                let real_value = null;
                let alias_info = null;
                if (L.eat('(')) {
                    const types = [];
                    L.whitespace(0);
                    while (!L.eat(')')) {
                        const r = this.parseType();
                        types.push(r.first);
                        if (alias_info && r.second) {
                            alias_info.addContainedType(r.second);
                        }
                        L.whitespace(0);
                        L.eat(',');
                        L.whitespace(0);
                    }
                    real_value = torch.TupleType.create(types);
                    fake_value = real_value;
                } else if (L.value === 'Future') {
                    L.next();
                    L.expect('(');
                    const p = this.parseType();
                    const subtype = p.first;
                    // const subalias = p.second;
                    L.expect(')');
                    real_value = torch.FutureType.get(subtype);
                    fake_value = real_value;
                } else if (L.value === 'Await') {
                    L.next();
                    L.expect('(');
                    const p = this.parseType();
                    const subtype = p.first;
                    // const subalias = p.second;
                    L.expect(')');
                    real_value = torch.AwaitType.get(subtype);
                    fake_value = real_value;
                } else if (L.value === 'RRef') {
                    L.next();
                    L.expect('(');
                    const p = this.parseType();
                    const subtype = p.first;
                    // const subalias = p.second;
                    L.expect(')');
                    real_value = torch.RRefType.get(subtype);
                    fake_value = real_value;
                } else if (L.value === 'Tensor') {
                    L.next();
                    real_value = torch.TensorType.get();
                    fake_value = real_value;
                    alias_info = this.parseAliasAnnotation();
                } else if (L.value === 'Dict') {
                    L.next();
                    L.expect('(');
                    const key_type = this.parseType().first;
                    L.expect(',');
                    L.whitespace(0);
                    const value_type = this.parseType().first;
                    L.expect(')');
                    alias_info = this.parseAliasAnnotation();
                    real_value = torch.DictType.create(key_type, value_type);
                    fake_value = real_value;
                } else if (L.eat('Union')) {
                    L.next();
                    L.expect('(');
                    const types = [];
                    types.push(this.parseType().first);
                    while (L.cur().kind !== ')') {
                        L.expect(',');
                        types.push(this.parseType().first);
                    }
                    L.expect(')');
                    alias_info = this.parseAliasAnnotation();
                    real_value = new torch.UnionType(types);
                    fake_value = real_value;
                /* } else if (complete_tensor_types && L.cur().kind == TK_IDENT && parseTensorDType(L.cur().text())) {
                    fake_value = real_value = parseRefinedTensor();
                    alias_info = parseAliasAnnotation(); */
                } else if (L.value === "__torch__") {
                    let name = L.expect('id');
                    while (L.eat('.')) {
                        name = `${name}.${L.expect('id')}`;
                    }
                    real_value = torch.ClassType.create(name); // getCustomClass
                    fake_value = real_value;
                } else {
                    real_value = this.parseBaseType();
                    fake_value = real_value;
                    if (real_value.kind() === 'ScalarTypeType' ||
                        real_value.kind() === 'MemoryFormat' ||
                        real_value.kind() === 'Layout' ||
                        real_value.kind() === 'SymIntType') {
                        fake_value = torch.IntType.get();
                    }
                    alias_info = this.parseAliasAnnotation();
                }
                while (true) {
                    if (L.kind === '[]') {
                        L.expect('[]');
                        fake_value = torch.ListType.create(fake_value);
                        real_value = torch.ListType.create(real_value);
                        let container = this.parseAliasAnnotation();
                        if (alias_info) {
                            if (!container) {
                                container = new torch._C.AliasInfo();
                                container.is_write = alias_info.is_write;
                            }
                            container.addContainedType(alias_info);
                        }
                        alias_info = container;
                    } else if (L.eat('?')) {
                        fake_value = torch.OptionalType.create(fake_value);
                        real_value = torch.OptionalType.create(real_value);
                    } else {
                        break;
                    }
                }
                return [fake_value, real_value, alias_info];
            }
            parseAliasAnnotation() {
                const L = this.L;
                let alias_info = null;
                if (L.eat('(')) {
                    alias_info = new torch._C.AliasInfo();
                    do {
                        alias_info.addBeforeSet(L.value);
                        L.next();
                        if (L.eat('!')) {
                            alias_info.is_write = true;
                        }
                        L.whitespace(0);
                    }
                    while (L.eat('|'));
                    if (L.eat('->')) {
                        L.whitespace(0);
                        do {
                            alias_info.addAfterSet(L.value);
                            L.next();
                            L.whitespace(0);
                        }
                        while (L.eat('|'));
                    }
                    L.expect(')');
                }
                return alias_info;
            }
        });
        this.registerType('torch.Argument', class {
            constructor(...args) {
                // torch/aten/src/ATen/core/function_schema.h
                this.N = null;
                this.default_value = null;
                this.kwarg_only = false;
                this.alias_info = null;
                if (args.length === 2) {
                    [this.name, this.type] = args;
                    this.real_type = this.type;
                } else if (args.length === 3 && args[1] instanceof torch.Type && args[2] instanceof torch.Type) {
                    [this.name, this.type, this.real_type] = args;
                } else if (args.length === 6) {
                    [this.name, this.type, this.real_type, this.N, this.default_value, this.kwarg_only] = args;
                } else if (args.length === 7) {
                    [this.name, this.type, this.real_type, this.N, this.default_value, this.kwarg_only, this.alias_info] = args;
                } else {
                    throw new python.Error('Invalid arguments.');
                }
                const is_alias = this.alias_info && this.alias_info.is_write;
                this.is_out = this.kwarg_only && is_alias;
            }
            has_default_value() {
                return this.default_value !== undefined;
            }
            is_inferred_type() {
                if (this.type instanceof torch.TensorType) {
                    return this.type.isInferredType();
                }
                return false;
            }
            static parse(L, is_return, kwarg_only) {
                const type_parser = new torch._C.SchemaTypeParser(L);
                let [fake_type, real_type, alias_info] = type_parser.parseFakeAndRealType();
                L.whitespace(0);
                let N = null;
                if (L.eat('[')) {
                    fake_type = torch.ListType.create(fake_type);
                    real_type = torch.ListType.create(real_type);
                    if (L.kind === '#') {
                        N = Number(L.value);
                        L.next();
                    }
                    L.expect(']');
                    let container = type_parser.parseAliasAnnotation();
                    if (alias_info) {
                        if (!container) {
                            container = new torch._C.AliasInfo();
                            container.is_write = alias_info.is_write;
                        }
                        container.addContainedType(alias_info);
                    }
                    alias_info = container;
                    if (L.eat('?')) {
                        fake_type = torch.OptionalType.create(fake_type);
                        real_type = torch.OptionalType.create(real_type);
                    }
                }
                let name = null;
                /* eslint-disable no-undef-init */
                let default_value = undefined;
                /* eslint-enable no-undef-init */
                if (is_return) {
                    L.whitespace(0);
                    kwarg_only = false;
                    if (L.kind === 'id') {
                        name = L.expect('id');
                    }
                } else {
                    L.whitespace(1);
                    name = L.expect('id');
                    L.whitespace(0);
                    if (L.eat('=')) {
                        L.whitespace(0);
                        default_value = torch.Argument._parse_value(L);
                    }
                }
                return new torch.Argument(name, fake_type, real_type, N, default_value, kwarg_only, alias_info);
            }
            static _parse_value(L) {
                /* eslint-disable no-undef-init */
                let value = undefined;
                /* eslint-enable no-undef-init */
                if (L.kind === 'id') {
                    if (L.value === 'True' || L.value === 'False') {
                        value = L.value === 'True';
                    } else if (L.value === 'None') {
                        value = null;
                    } else if (L.value === 'Mean' || L.value === 'contiguous_format' || L.value === 'long') {
                        value = L.value;
                    } else if (typeof L.value === 'string') {
                        value = L.value;
                    } else if (typeof L.value === 'number') {
                        value = L.value;
                    } else {
                        throw new python.Error(`Unsupported default value '${L.value}'.`);
                    }
                } else if (L.kind === '#') {
                    value = Number(L.value);
                } else if (L.kind === 'string') {
                    value = L.value.slice(1, -1);
                } else if (L.kind === '[]') {
                    value = [];
                } else if (L.eat('[')) {
                    value = [];
                    if (!L.eat(']')) {
                        while (true) {
                            L.whitespace(0);
                            value.push(torch.Argument._parse_value(L));
                            L.whitespace(0);
                            if (!L.eat(',')) {
                                break;
                            }
                        }
                        L.expect(']');
                    }
                    return value;
                } else {
                    throw new python.Error(`Unsupported default value '${L.kind}'.`);
                }
                L.next();
                return value;
            }
            str() {
                const list = [];
                const type = this.real_type;
                const is_opt = type instanceof torch.OptionalType;
                const unopt_type = is_opt ? type.getElementType() : type;
                if (unopt_type instanceof torch.ListType) {
                    list.push(unopt_type.getElementType().str());
                    if (this.alias_info && this.alias_info.containedTypes.length > 0) {
                        list.push(this.alias_info.containedTypes[0].str());
                    }
                    list.push(this.N === null ? `[]` : `[${this.N}]`);
                } else {
                    list.push(unopt_type.str());
                }
                if (this.alias_info && this.alias_info.before_set.length > 0) {
                    list.push(this.alias_info.str());
                }
                if (is_opt) {
                    list.push('?');
                }
                if (this.name) {
                    list.push(' ');
                    list.push(this.name);
                }
                if (this.default_value !== undefined) {
                    const value = this.default_value;
                    if (value === null) {
                        list.push('=None');
                    } else if (typeof value === 'boolean') {
                        list.push('=');
                        list.push(value ? 'True' : 'False');
                    } else if (typeof value === 'string') {
                        list.push(`="${value}"`);
                    } else if (typeof value === 'number') {
                        list.push(`=${value}`);
                        if (Number.isInteger(value) && this.real_type instanceof torch.FloatType) {
                            list.push(`.`);
                        }
                    } else if (Array.isArray(value)) {
                        list.push(`=[${value.join(', ')}]`);
                    }
                }
                return list.join('');
            }
        });
        this.registerType('torch._C.SchemaParser', class {
            constructor(str, allow_typevars) {
                this.L = new torch._C.Lexer(str);
                this.type_parser = new torch._C.SchemaTypeParser(this.L, false, allow_typevars);
            }
            parseName() {
                const L = this.L;
                let name = L.expect('id').text();
                if (L.nextIf(':')) {
                    L.expect(':');
                    name = `${name}::${L.expect('ident').text()}`;
                }
                let overload_name = '';
                if (L.nextIf('.')) {
                    overload_name = L.expect('ident').text();
                }
                // const is_a_valid_overload_name = !((overload_name === "default") || (overload_name.rfind("__", 0) == 0));
                // TORCH_CHECK(is_a_valid_overload_name, overload_name, " is not a legal overload name for aten operators");
                return new torch._C.OperatorName(name, overload_name);
            }
            parseDeclaration() {
                const L = this.L;
                const name = this.parseName();
                if (L.cur().kind !== '(') {
                    return name;
                }
                throw new python.Error('Not implemented.');
            }
            parseExactlyOneDeclaration() {
                // const L = this.L;
                const result = this.parseDeclaration();
                // L.nextIf(TK_NEWLINE);
                // L.expect(TK_EOF);
                return result;
            }
            parseArgument() {
                throw new python.Error('Not implemented.');
            }
        });
        this.registerType('torch.FunctionSchema', class {
            constructor(name, overload_name, args, returns, is_vararg, is_varret) {
                const index = name.indexOf('(');
                if (index === -1) {
                    this._name = name;
                    this._overload_name = overload_name || '';
                    this._arguments = args || [];
                    this._returns = returns || [];
                    this._is_vararg = is_vararg || false;
                    this._is_varret = is_varret || false;
                } else {
                    const value = name.substring(0, index).trim();
                    const dot = value.indexOf('.');
                    if (dot === -1) {
                        this._name = value;
                        this._overload_name = '';
                    } else {
                        this._name = value.substring(0, dot);
                        this._overload_name = value.substring(dot + 1, value.length);
                    }
                    this._buffer = name.substring(index, name.length);
                }
            }
            static parse(schema) {
                return new torch.FunctionSchema(schema);
            }
            get name() {
                return this._name;
            }
            get overload_name() {
                return this._overload_name;
            }
            get arguments() {
                this._parse();
                return this._arguments;
            }
            get returns() {
                this._parse();
                return this._returns;
            }
            get is_vararg() {
                this._parse();
                return this._is_vararg;
            }
            get is_varret() {
                this._parse();
                return this._is_varret;
            }
            _parse() {
                if (this._buffer) {
                    const L = new torch._C.Lexer(this._buffer);
                    this._arguments = [];
                    this._is_vararg = false;
                    this._kwarg_only = false;
                    L.expect('(');
                    if (!L.eat(')')) {
                        while (true) {
                            L.whitespace(0);
                            if (this._is_vararg) {
                                throw new python.Error();
                            }
                            if (L.eat('*')) {
                                this._kwarg_only = true;
                            } else if (L.eat('...')) {
                                this._is_vararg = true;
                            } else {
                                const argument = torch.Argument.parse(L, false, this._kwarg_only);
                                this._arguments.push(argument);
                            }
                            L.whitespace(0);
                            if (!L.eat(',')) {
                                break;
                            }
                        }
                        L.expect(')');
                    }
                    L.whitespace(0);
                    L.expect('->');
                    L.whitespace(0);
                    this._returns = [];
                    this._is_varret = false;
                    if (L.eat('...')) {
                        this._is_varret = true;
                    } else if (L.eat('(')) {
                        L.whitespace(0);
                        if (!L.eat(')')) {
                            while (true) {
                                L.whitespace(0);
                                if (this._is_varret) {
                                    throw new python.Error();
                                }
                                if (L.eat('...')) {
                                    this._is_varret = true;
                                } else {
                                    const argument = torch.Argument.parse(L, true, false);
                                    this._returns.push(argument);
                                }
                                L.whitespace(0);
                                if (!L.eat(',')) {
                                    break;
                                }
                            }
                            L.expect(')');
                        }
                        L.whitespace(0);
                    } else {
                        this._returns.push(torch.Argument.parse(L, true, false));
                    }
                    delete this._buffer;
                }
            }
            __str__() {
                const list = [this.name];
                const overload_name = this.overload_name;
                if (overload_name !== '' && overload_name !== 'default') {
                    list.push(`.${this.overload_name}`);
                }
                list.push('(');
                let first = true;
                let kwarg_only = false;
                for (const argument of this.arguments) {
                    if (!first) {
                        list.push(', ');
                    }
                    if (argument.kwarg_only && !kwarg_only) {
                        list.push('*, ');
                        kwarg_only = true;
                    }
                    first = false;
                    list.push(argument.str());
                }
                if (this.is_vararg) {
                    if (!first) {
                        list.push(', ');
                    }
                    first = true;
                    list.push('...');
                }
                list.push(') -> ');
                const returns = this.returns;
                const braces = !this.is_varret &&
                   (returns.length !== 1 ||
                    returns[0].name ||
                    returns[0].real_type instanceof torch.TupleType ||
                    returns[0].real_type instanceof torch.ListType && returns[0].real_type.getElementType() instanceof torch.TupleType);
                if (braces) {
                    list.push('(');
                }
                first = true;
                for (const argument of this.returns) {
                    if (!first) {
                        list.push(', ');
                    }
                    first = false;
                    list.push(argument.str());
                }
                if (this.is_varret) {
                    if (!first) {
                        list.push(', ');
                    }
                    first = true;
                    list.push('...');
                }
                if (braces) {
                    list.push(')');
                }
                return list.join('');
            }
        });
        this.registerFunction('torch._C.string_to_type_lut', () => {
            if (!torch._C.string_to_type_lut.basePythonTypes) {
                const map = new Map();
                map.set('Tensor', torch.TensorType.get());
                map.set('int', torch.IntType.get());
                map.set('float', torch.FloatType.get());
                map.set('bool', torch.BoolType.get());
                map.set('complex', torch.ComplexType.get());
                map.set('str', torch.StringType.get());
                torch._C.string_to_type_lut.basePythonTypes = map;
            }
            return torch._C.string_to_type_lut.basePythonTypes;
        });
        this.registerType('torch.jit.ScriptTypeParser', class {
            constructor(resolver) {
                this._resolver = resolver;
            }
            parseSchemaFromDef(def, skip_self) {
                const name = def.name;
                const args = this.parseArgsFromDecl(def, skip_self);
                const returns = this.parseReturnFromDecl(def);
                return new torch.FunctionSchema(name, '', args, returns, false, false);
            }
            parseArgsFromDecl(decl, skip_self) {
                const retval = [];
                if (decl.args.posonlyargs.length > 0 || decl.args.kwonlyargs.length > 0) {
                    throw new python.Error('Unsupported function argument.');
                }
                const params = decl.args.args.slice();
                const kwonlyargs = new Set(Array.from(decl.args.kwonlyargs));
                const start = skip_self ? 1 : 0;
                for (let i = start; i < params.length; i++) {
                    const decl_arg = params[i];
                    const N = null;
                    const default_value = undefined;
                    const type = decl_arg.annotation ? this.parseTypeFromExpr(decl_arg.annotation) : null;
                    const arg = new torch.Argument(decl_arg.arg, type, type, N, default_value, kwonlyargs.has(decl_arg), null);
                    retval.push(arg);
                }
                return retval;
            }
            parseReturnFromDecl(decl) {
                if (!decl.returns) {
                    return [];
                }
                if (this.parseBroadcastList(decl.returns)) {
                    throw new python.Error('Broadcastable lists cannot appear as a return type.');
                }
                const parsed_type = this.parseTypeFromExpr(decl.returns);
                return [new torch.Argument('', parsed_type, parsed_type, null, undefined, false)];
            }
            parseTypeFromExpr(expr) {
                if (this._resolver) {
                    if (expr instanceof ast.Name) {
                        const type = this._resolver.resolveType(expr.id);
                        if (type) {
                            return type;
                        }
                    }
                }
                return this.parseTypeFromExprImpl(expr);
            }
            parseTypeFromExprImpl(expr) {
                if (expr instanceof ast.Subscript) {
                    const value_name = this.parseBaseTypeName(expr.value);
                    if (!value_name) {
                        throw new python.Error('Subscripted type must be a type identifier.');
                    }
                    return this.subscriptToType(value_name, expr);
                }
                const name = this.parseBaseTypeName(expr);
                if (name) {
                    const itr = torch._C.string_to_type_lut().get(name);
                    if (itr) {
                        return itr;
                    }
                    if (this._resolver) {
                        const typePtr = this._resolver.resolveType(name, expr);
                        if (typePtr) {
                            return typePtr;
                        }
                    }
                }
                return this._resolver._cu.execution.type(expr);
            }
            parseBaseTypeName(expr) {
                if (expr instanceof ast.Name) {
                    return expr.id;
                } else if (expr instanceof ast.Constant && expr.value === null) {
                    return 'None';
                } else if (expr instanceof ast.Attribute) {
                    const name = expr.attr;
                    const tensor_subtypes = new Set(['Tensor', 'LongTensor', 'FloatTensor', 'DoubleTensor', 'IntTensor', 'ShortTensor', 'HalfTensor', 'CharTensor', 'ByteTensor', 'BoolTensor']);
                    if (torch._C.isTorch(expr.value) && tensor_subtypes.has(name)) {
                        return name;
                    }
                    return torch._C.collectQualname(expr);
                }
                throw new python.Error('Unsupported type.');
            }
            parseBroadcastList(/* expr */) {
                return null;
            }
            parseType(str) {
                const expr = ast.parse(str);
                return this.parseTypeFromExpr(expr.body[0]);
            }
            subscriptToType(typeName, subscript) {
                if (typeName === 'Tuple' || typeName === 'tuple') {
                    /*
                    if (subscript.slice.elts.length === 1 && subscript.slice.elts[0].kind() === TK_TUPLE_LITERAL) {
                        const tup_literal = null; // TupleLiteral(subscript.subscript_exprs()[0]);
                        if (!tup_literal.inputs().empty()) {
                            throw new python.Error('Tuple literal in Tuple type annotation must not have any elements.');
                        }
                        return torch.TupleType.create({});
                    }
                    */
                    const subscript_expr_types = [];
                    for (const expr of subscript.slice.elts) {
                        subscript_expr_types.push(this.parseTypeFromExprImpl(expr));
                    }
                    return torch.TupleType.create(subscript_expr_types);
                } else if (typeName === 'List' || typeName === 'list') {
                    if (subscript.slice.elts.length !== 1) {
                        throw new python.Error('List type must have exactly one element type.');
                    }
                    const elem_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    return torch.ListType.create(elem_type);
                } else if (typeName === 'Optional') {
                    if (subscript.slice.elts.length !== 1) {
                        throw new python.Error('Optional type must have exactly one element type.');
                    }
                    const elem_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    return torch.OptionalType.create(elem_type);
                } else if (typeName === 'Union') {
                    const subscript_expr_types = [];
                    subscript_expr_types.reserve(subscript.subscript_exprs().size());
                    for (const expr of subscript.subscript_exprs()) {
                        subscript_expr_types.push(this.parseTypeFromExprImpl(expr));
                    }
                    return torch.UnionType.create(subscript_expr_types);
                } else if (typeName === 'Future' || typeName === 'torch.jit.Future') {
                    if (subscript.slice.elts.length !== 1) {
                        throw new python.Error('Future type must have exactly one element type.');
                    }
                    const elem_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    return torch.FutureType.create(elem_type);
                } else if (typeName === 'Await' || typeName === 'torch.jit._Await') {
                    if (subscript.slice.elts.length !== 1) {
                        throw new python.Error('Await type must have exactly one element type.');
                    }
                    const elem_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    return torch.AwaitType.create(elem_type);
                } else if (typeName === 'RRef') {
                    if (subscript.slice.elts.length !== 1) {
                        throw new python.Error('RRef type must have exactly one element type.');
                    }
                    const elem_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    return torch.RRefType.create(elem_type);
                } else if (typeName === 'Dict' || typeName === 'dict') {
                    if (subscript.slice.elts.length !== 2) {
                        throw new python.Error('Dict type must have exactly two element types.');
                    }
                    const key_type = this.parseTypeFromExprImpl(subscript.slice.elts[0]);
                    const value_type = this.parseTypeFromExprImpl(subscript.slice.elts[1]);
                    return torch.DictType.create(key_type, value_type);
                }
                throw new python.Error(`Unknown type constructor '${typeName}'.`);
            }
        });
        this.registerFunction('torch._C.isTorch', (expr) => {
            return expr instanceof ast.Name && expr.id === 'torch';
        });
        this.registerFunction('torch._C.collectQualname', (select) => {
            const base = select.value;
            if (base instanceof ast.Name) {
                return `${base.id}.${select.attr}`;
            }
            const basename = torch._C.collectQualname(base);
            return `${basename}.${select.attr}`;
        });
        this.registerType('torch._ops.OpOverload', class extends torch._ops.OperatorBase {
            constructor(overloadpacket, op, op_dk, schema, tags) {
                super();
                this._op = op;
                this._op_dk = op_dk;
                this._schema = schema;
                this._overloadpacket = overloadpacket;
                this._tags = tags;
                this._overloadname = schema.overload_name === '' ? 'default' : schema.overload_name;
                this._name = this._schema.name;
                this._name = schema.overload_name ? `${this._name}.${schema.overload_name}` : this._name;
                this.__name__ = `${this._schema.name.split('::')[1]}.${this._overloadname}`;
                this.__module__ = overloadpacket.__module__;
                op.__module__ = overloadpacket.__module__;
                this.__qualname__ = self._name;
                this.__annotations__ = {};
                // this._defined_in_python = this.__qualname__ in torch.library._defs
                let is_write = null;
                for (const a of this._schema.arguments) {
                    if (a.alias_info) {
                        is_write = is_write === null ? a.alias_info.is_write : a.alias_info.is_write || is_write;
                    }
                }
                this.is_view = is_write !== null && !is_write;
            }
            get name() {
                return this._name;
            }
        });
        this.registerType('torch._ops.OpOverloadPacket', class {
            constructor(qualified_op_name, op_name, op, overload_names) {
                this._qualified_op_name = qualified_op_name;
                this.__name__ = op_name;
                this._op = op;
                this._overload_names = overload_names;
                this._dir = [];
                this._has_torchbind_op_overload = this._schemas.some((schema) => this._has_script_object_arg(schema));
            }
            get _schemas() {
                return this._overload_names.map((overload_name) => torch._C._get_schema(this._qualified_op_name, overload_name));
            }
            __getattr__(key) {
                key = key === 'default' ? '' : key;
                const op_dk_tags = torch._C._get_operation_overload(this._qualified_op_name, key);
                const [op_, op_dk_, tags] = op_dk_tags;
                const schema = torch._C._get_schema(this._qualified_op_name, key);
                const overload = this._has_script_object_arg(schema) ?
                    new torch._ops.TorchBindOpOverload(this, op_, op_dk_, schema, tags) :
                    new torch._ops.OpOverload(this, op_, op_dk_, schema, tags);
                builtins.setattr(self, key, overload);
                this._dir.push(key);
                return overload;
            }
            _has_script_object_arg(/* schema */) {
                return false;
                // return any(isinstance(arg.type, torch.ClassType) for arg in schema.arguments)
            }
        });
        this.registerType('torch._ops._OpNamespace', class extends types.ModuleType {
            constructor(name) {
                super(`torch.ops.${name}`);
                this.name = name;
                this._dir = [];
            }
            __getattr__(op_name) {
                const namespace_name = this.name;
                const qualified_op_name = `${namespace_name}::${op_name}`;
                const module_name = `${this.__module__}.${namespace_name}`;
                let op = null;
                let overload_names = null;
                try {
                    [op, overload_names] = this._get_packet(qualified_op_name, module_name);
                } catch {
                    // continue regardless of error
                }
                if (!op) {
                    throw new python.Error(`Unknown operator type '${qualified_op_name}'.`);
                }
                op.__module__ = module_name;
                const opoverloadpacket = new torch._ops.OpOverloadPacket(qualified_op_name, op_name, op, overload_names);
                opoverloadpacket.__module__ = `${this.__module__}.${namespace_name}`;
                builtins.setattr(this, op_name, opoverloadpacket);
                this._dir.push(op_name);
                return opoverloadpacket;
            }
            _get_packet(qualname, op_module) {
                const [op, overload_names] = torch._C._jit_get_operation(qualname);
                if (op) {
                    // torch.jit._builtins._register_builtin(op, qualname);
                }
                op.__module__ = op_module;
                return [op, overload_names];
            }
        });
        this.registerType('torch.Graph', class {
            constructor() {
                this._next_unique = 1;
                this._unique_names = new Map();
                this._name_base_suffix = new Map();
                this._all_nodes = [];
                this._all_values = [];
                this._all_blocks = [];
                this._block = new torch.Block(this, null);
                this._insert_before = this.return_node();
            }
            create(kind, ...args) {
                let inputs = null;
                let num_outputs = 1;
                if (args.length === 2 && Array.isArray(args[0]) && typeof args[1] === 'number') {
                    [inputs, num_outputs] = args;
                } else if (args.length === 1) {
                    if (typeof args[0] === 'number') {
                        [num_outputs] = args;
                    } else if (Array.isArray(args[0])) {
                        [inputs] = args;
                    }
                }
                const n = new torch.Node(this, kind);
                if (inputs) {
                    for (const i of inputs) {
                        n.addInput(i);
                    }
                }
                for (let i = 0; i < num_outputs; i++) {
                    n.addOutput();
                }
                return n;
            }
            createClone(n, value_map, copy_blocks) {
                copy_blocks = copy_blocks === undefined ? true : copy_blocks;
                const r = n.allocNewInstance(this);
                for (const o of n.outputs()) {
                    r.addOutput().copyMetadata(o);
                }
                r.cloneFrom(n);
                for (const i of n.inputs()) {
                    r.addInput(value_map(i));
                }
                if (copy_blocks) {
                    for (const b of n.blocks()) {
                        r.addBlock().cloneFrom(b, value_map);
                    }
                }
                return r;
            }
            createNone() {
                const n = this.create('prim::Constant');
                n.output().setType(torch.NoneType.get());
                return n;
            }
            createUninitialized(typ) {
                const n = this.create('prim::Uninitialized');
                n.output().setType(typ);
                return n;
            }
            createList(contained_type, values) {
                const n = this.create('prim::ListConstruct', values);
                for (const v of values) {
                    if (!v.type().isSubtypeOf(contained_type)) {
                        throw new python.Error('Invalid list item.');
                    }
                }
                n.output().setType(torch.ListType.create(contained_type));
                return n;
            }
            createListUnpack(v, size) {
                const list_type = v.type().expect(torch.ListType);
                const elem_type = list_type.getElementType();
                const n = this.create('prim::ListUnpack', [v], 0);
                for (let i = 0; i < size; i++) {
                    n.addOutput().setType(elem_type);
                }
                return n;
            }
            createTuple(values, tuple_type) {
                if (!tuple_type) {
                    const types = values.map((v) => v.type());
                    tuple_type = torch.TupleType.create(types);
                }
                const n = this.create('prim::TupleConstruct', values);
                n.output().setType(tuple_type);
                return n;
            }
            createTupleUnpack(v) {
                const tt = v.type().expect(torch.TupleType);
                const n = this.create('prim::TupleUnpack', [v], 0);
                for (const element of tt.elements()) {
                    n.addOutput().setType(element);
                }
                return n;
            }
            createTupleIndex(tup, idx, output_type) {
                const n = this.create('prim::TupleIndex', [tup, idx]);
                n.output().setType(output_type);
                return n;
            }
            createDict(key_type, value_type, keys, values) {
                if (keys.length !== values.length) {
                    throw new python.Error('Invalid dictionary size.');
                }
                const n = this.create('prim::DictConstruct');
                const length = keys.length;
                for (let i = 0; i < length; i++) {
                    if (!keys[i].type().isSubtypeOf(key_type)) {
                        throw new python.Error('Invalid key.');
                    }
                    if (!values[i].type().isSubtypeOf(value_type)) {
                        throw new python.Error('Invalid value.');
                    }
                    n.addInput(keys[i]);
                    n.addInput(values[i]);
                }
                n.output().setType(torch.DictType.create(key_type, value_type));
                return n;
            }
            createObject(type) {
                const node = this.create('prim::CreateObject');
                node.output().setType(type);
                return node;
            }
            createIsInstance(v, types) {
                const n = this.create('prim::isinstance', [v], 1);
                n.tys_('types', types);
                n.output().setType(torch.BoolType.get());
                return n;
            }
            createSetAttr(obj, field, newValue) {
                const n = this.create('prim::SetAttr', [obj, newValue], 0);
                n.s_('name', field);
                return n;
            }
            createGetAttr(obj, field) {
                const n = this.create('prim::GetAttr', [obj]);
                n.s_('name', field);
                const classType = obj.type();
                const outputType = classType.getAttribute(field);
                n.output().setType(outputType);
                n.output().setDebugName(/^[0-9]+$/.test(field) ? `_${field}` : field);
                return n;
            }
            createLoad(name, type) {
                const n = this.create('prim::Load', [], 1);
                n.s_('name', name);
                n.output().setType(type);
                return n;
            }
            createStore(name, v) {
                const n = this.create('prim::Store', [v], 0);
                n.s_('name', name);
                return n;
            }
            inputs() {
                return this._block.inputs();
            }
            outputs() {
                return this._block.outputs();
            }
            nodes() {
                return this._block.nodes();
            }
            param_node() {
                return this._block.param_node();
            }
            return_node() {
                return this._block.return_node();
            }
            block() {
                return this._block;
            }
            addInput(name) {
                return this._block.addInput(name);
            }
            insertNode(node) {
                if (!this._insert_before.inBlockList()) {
                    throw new python.Error('Invalid insert point.');
                }
                return node.insertBefore(this._insert_before);
            }
            insertConstant(val, loc, scope) {
                return torch._C.insertConstant(this, val, loc, scope);
            }
            insertMethodCall(method_name, matched) {
                const result = this.insertNode(this.create('prim::CallMethod', matched.inputs)).s_('name', method_name).output().setType(matched.return_types[0]);
                return result;
            }
            insertUncheckedCast(v, type) {
                const n = this.create('prim::unchecked_cast', [v]);
                this.insertNode(n);
                n.output().setType(type);
                return n.output();
            }
            insertToList(v, type) {
                let dim = 0;
                let ptr = type;
                while (ptr instanceof torch.ListType) {
                    ptr = ptr.getElementType();
                    dim += 1;
                }
                let elem_ty = 0;
                if (ptr instanceof torch.IntType) {
                    elem_ty = 0;
                } else if (ptr instanceof torch.FloatType) {
                    elem_ty = 1;
                } else if (ptr instanceof torch.BoolType) {
                    elem_ty = 2;
                } else if (ptr instanceof torch.ComplexType) {
                    elem_ty = 3;
                } else {
                    throw new python.Error(`Unsupported list type '${type.kind()}'.`);
                }
                const dim_val = this.insertConstant(dim);
                const elem_ty_val = this.insertConstant(elem_ty);
                const n = this.insertNode(this.create('prim::tolist', [v, dim_val, elem_ty_val]));
                n.output().setType(type);
                return n.output();
            }
            insertPoint() {
                return this._insert_before;
            }
            setInsertPoint(node) {
                if (node instanceof torch.Block) {
                    node = node.return_node();
                }
                this._insert_before = node;
            }
            all_nodes() {
                return this._all_nodes;
            }
            all_blocks() {
                return this._all_blocks;
            }
            freeNode(n) {
                const index = this._all_nodes.indexOf(n);
                if (index !== -1) {
                    this._all_nodes.splice(index, 1);
                }
            }
            freeValue(v) {
                v.setDebugName('');
                const index = this._all_values.indexOf(v);
                if (index !== -1) {
                    this._all_values.splice(index, 1);
                }
            }
            freeBlock(b) {
                const index = this._all_blocks.indexOf(b);
                if (index !== -1) {
                    this._all_blocks.splice(index, 1);
                }
            }
            copy() {
                const new_g = new torch.Graph();
                new_g.cloneFrom(this);
                return new_g;
            }
            cloneFrom(src) {
                const env = (v) => {
                    throw new python.Error(`Use of value '${v.debugName()}' not in scope.`);
                };
                this.block().cloneFrom(src.block(), env);
            }
            set_op_version(version) {
                this._op_version = version;
            }
            get_op_version() {
                return this._op_version;
            }
            print(out, print_source_locations) {
                out.write('graph(');
                torch._C.const_value_list_with_types(out, this.inputs(), ',\n      ');
                out.write('):\n');
                const groups = [];
                for (const node of this.nodes()) {
                    node.print(out, 1, groups, print_source_locations);
                }
                out.write('  return (');
                torch._C.printValueRefs(out, this.outputs());
                out.write(')\n');
                for (let i = 0; i < groups.length; i++) {
                    const fg = groups[i];
                    out.write('with ');
                    out.write(fg.kind());
                    out.write(`_${i} = `);
                    out.write(fg.g('subsgraph'));
                }
                return out;
            }
            toString() {
                const out = new io.StringIO();
                this.print(out, true);
                return out.toString();
            }
        });
        this.registerType('torch.Block', class {
            constructor(graph, node) {
                this._graph = graph;
                this._input = graph.create('prim::Param', 0);
                this._output = graph.create('prim::Return', 0);
                this._owning_node = node;
                this._input.next = this._output;
                this._input.prev = this._output;
                this._output.next = this._input;
                this._output.prev = this._input;
                this._graph.all_blocks().push(this);
                this._output._owning_block = this;
                // output_.topo_position_ = kUpperBound;
                this._input._owning_block = this;
                // input_.topo_position_ = kLowerBound;
            }
            inputs() {
                return this._input.outputs();
            }
            outputs() {
                return this._output.inputs();
            }
            nodes() {
                const nodes = [];
                let current = this._input.next;
                do {
                    nodes.push(current);
                    current = current.next;
                } while (current !== this._input.prev);
                return nodes;
            }
            return_node() {
                return this._output;
            }
            param_node() {
                return this._input;
            }
            owningNode() {
                return this._owning_node;
            }
            owningGraph() {
                return this._graph;
            }
            addInput(name) {
                const value = this._input.addOutput();
                value.setDebugName(name || '');
                return value;
            }
            registerOutput(value) {
                this._output.addInput(value);
                return this.outputs().length - 1;
            }
            appendNode(n) {
                if (n._graph !== this._graph || n.inBlockList()) {
                    throw new python.Error('Node not in graph.');
                }
                n.insertBefore(this._output);
                return n;
            }
            cloneFrom(src, value_map) {
                const local_map = new Map();
                const env = (v) => {
                    if (local_map.has(v)) {
                        return local_map.get(v);
                    }
                    return value_map(v);
                };
                const graph = this.owningGraph();
                for (const input of src.inputs()) {
                    local_map.set(input, this.addInput().copyMetadata(input));
                }
                for (const node of src.nodes()) {
                    const new_node = this.appendNode(graph.createClone(node, env));
                    for (let i = 0; i < node.outputs().length; i++) {
                        const oo = node.outputs()[i];
                        const no = new_node.outputs()[i];
                        local_map.set(oo, no);
                        no.copyMetadata(oo);
                    }
                }
                for (const output of src.outputs()) {
                    this.registerOutput(env(output));
                }
            }
            eraseOutput(i) {
                this._output.removeInput(i);
            }
            destroy() {
                this._output.removeAllInputs();
                for (const n of this.nodes()) {
                    n.destroy();
                }
                this._output.destroy();
                this._input.destroy();
                this._graph.freeBlock(this);
            }
        });
        this.registerType('torch.Node', class {
            constructor(graph, kind) {
                this._graph = graph;
                this._kind = kind;
                this._values = new Map();
                this._inputs = [];
                this._outputs = [];
                this._blocks = [];
                this._graph.all_nodes().push(this);
                this._prev = null;
                this._next = null;
                this._source_range = null;
            }
            owningGraph() {
                return this._graph;
            }
            owningBlock() {
                return this._owning_block;
            }
            kind() {
                return this._kind;
            }
            schema() {
                if (this._op === undefined) {
                    this._op = null;
                    const index = this._kind.indexOf('.');
                    const name = index === -1 ? this._kind : this._kind.substring(0, index);
                    const overload_name = index === -1 ? '' : this._kind.substring(index + 1);
                    const candidates = torch._C.getAllOperatorsFor(name);
                    for (const candidate of candidates) {
                        if (candidate.schema().overload_name === overload_name) {
                            this._op = candidate;
                            break;
                        }
                    }
                }
                return this._op ? this._op.schema() : null;
            }
            matches(schema) {
                if (torch._C.isBlockListedSchema(schema)) {
                    return false;
                }
                if (this.kind() !== schema.name) {
                    return false;
                }
                const actuals = this.inputs();
                const formals = schema.arguments;
                if (actuals.length < formals.length) {
                    return false;
                }
                const type_env = new Map();
                for (let i = 0; i < formals.length; i++) {
                    let formal = formals[i].type;
                    const matched_type = torch._C.matchTypeVariables(formal, actuals[i].type(), type_env);
                    if (!matched_type.success()) {
                        return false;
                    }
                    const resolved = torch._C.tryEvalTypeVariables(formal, type_env);
                    if (resolved) {
                        formal = resolved;
                    }
                    if (!actuals[i].type().isSubtypeOf(formal)) {
                        return false;
                    }
                }
                if (!schema.is_vararg && actuals.length !== formals.length) {
                    return false;
                }
                return true;
            }
            maybeOperator() {
                if (!this._op) {
                    const candidates = torch._C.getAllOperatorsFor(this.kind());
                    for (const candidate of candidates) {
                        if (this.matches(candidate.schema())) {
                            this._op = candidate;
                            break;
                        }
                    }
                }
                return this._op;
            }
            getOperator() {
                const maybe = this.maybeOperator();
                if (maybe) {
                    return maybe;
                }
                throw new python.Error('Operator not found.');
            }
            getOperation() {
                return this.getOperator().getOperation(this);
            }
            inputs() {
                return this._inputs;
            }
            outputs() {
                return this._outputs;
            }
            input(i) {
                if (i === undefined && this._inputs.length !== 1) {
                    throw new python.Error('Node has multiple inputs.');
                }
                i = i || 0;
                return this._inputs[i];
            }
            output() {
                if (this._outputs.length !== 1) {
                    throw new python.Error('Node has multiple outputs.');
                }
                return this._outputs[0];
            }
            blocks() {
                return this._blocks;
            }
            addInput(value) {
                if (this._graph !== value.owningGraph()) {
                    throw new python.Error('Value not in graph.');
                }
                value.uses().push(new torch.Use(this, this._inputs.length));
                this._inputs.push(value);
                return value;
            }
            addOutput() {
                const value = new torch.Value(this);
                this._outputs.push(value);
                return value;
            }
            addBlock() {
                this._op = null;
                this._blocks.push(new torch.Block(this.owningGraph(), this));
                return this._blocks[this._blocks.length - 1];
            }
            get prev() {
                return this._prev;
            }
            set prev(value) {
                this._prev = value;
            }
            get next() {
                return this._next;
            }
            set next(value) {
                this._next = value;
            }
            insertBefore(n) {
                if (!n.inBlockList()) {
                    throw new python.Error('Node not in block.');
                }
                this.insertAfter(n.prev);
                return this;
            }
            insertAfter(n) {
                if (this.inBlockList() || !n.inBlockList() || !n.owningBlock()) {
                    throw new python.Error('Node not in block.');
                }
                if (n.kind() === 'prim::Return') {
                    throw new python.Error('Cannot insert after return.');
                }
                this._owning_block = n.owningBlock();
                const  next = n.next;
                n.next = this;
                this.prev = n;
                this.next = next;
                next.prev = this;
                // assignTopoPosition();
                return this;
            }
            allocNewInstance(g) {
                return new torch.Node(g, this.kind());
            }
            cloneFrom(s) {
                this._source_range = s._source_range;
                if (s._scope && !s._scope.isBlank()) {
                    this._scope = s._scope;
                }
                this.copyAttributes(s);
                this._callstack = s._callstack;
            }
            copyAttributes(rhs) {
                this._values = new Map(rhs._values);
                return this;
            }
            dropInput(i) {
                if (i >= this._inputs.length) {
                    throw new python.Error('Input index out of range.');
                }
                const input_node = this._inputs[i];
                const use_it = this.findUseForInput(i);
                input_node._uses.splice(use_it.offset, 1);
                this._inputs[i] = null;
                return input_node;
            }
            eraseOutput(i) {
                this._op = null;
                const v = this._outputs[i];
                this._outputs.splice(i, 1);
                this.owningGraph().freeValue(v);
            }
            eraseBlock(i) {
                this._op = null;
                const n = this._blocks[i];
                this._blocks.splice(i, 1);
                n.destroy();
            }
            findUseForInput(i) {
                const input_uses = this._inputs[i]._uses;
                for (const use_it of input_uses) {
                    if (use_it.user === this && use_it.offset === i) {
                        return use_it;
                    }
                }
                throw new python.Error('Input use not found.');
            }
            moveBefore(n) {
                this.removeFromList();
                this.insertBefore(n);
            }
            removeInput(i) {
                this._op = null;
                this.dropInput(i);
                for (let j = i + 1; j < this._inputs.length; j++) {
                    const it = this.findUseForInput(j);
                    it._offset--;
                }
                this._inputs.splice(i, 1);
            }
            removeAllInputs() {
                this._op = null;
                for (let i = 0; i < this._inputs.length; i++) {
                    this.dropInput(i);
                }
                this._inputs.splice(0, this._inputs.length);
            }
            inBlockList() {
                return this.next !== null;
            }
            removeFromList() {
                this._owning_block = null;
                const next = this.next;
                const prev = this.prev;
                prev.next = next;
                next.prev = prev;
                this.next = null;
                this.prev = null;
            }
            destroy() {
                while (this.outputs().length > 0) {
                    this.eraseOutput(this.outputs().length - 1);
                }
                while (this.blocks().length > 0) {
                    this.eraseBlock(this.blocks().length - 1);
                }
                this.removeAllInputs();
                if (this.inBlockList()) {
                    this.removeFromList();
                }
                this._graph.freeNode(this);
            }
            s_(name, value) {
                this._values.set(name, [value, 's']);
                return this;
            }
            s(name) {
                return this._values.get(name)[0];
            }
            ss_(name, value) {
                this._values.set(name, [value, 'ss']);
                return this;
            }
            ss(name) {
                return this._values.get(name)[0];
            }
            i_(name, value) {
                this._values.set(name, [value, 'i']);
                return this;
            }
            i(name) {
                return this._values.get(name)[0];
            }
            f_(name, value) {
                this._values.set(name, [value, 'f']);
                return this;
            }
            f(name) {
                return this._values.get(name)[0];
            }
            t_(name, value) {
                this._values.set(name, [value, 't']);
                return this;
            }
            t(name) {
                return this._values.get(name)[0];
            }
            tys_(name, value) {
                this._values.set(name, [value, 'tys']);
                return this;
            }
            tys(name) {
                return this._values.get(name)[0];
            }
            ival_(name, value) {
                this._values.set(name, [value, 'ival']);
                return this;
            }
            ival(name) {
                return this._values.get(name)[0];
            }
            hasAttribute(name) {
                return this._values.has(name);
            }
            hasAttributes() {
                return this._values.size > 0;
            }
            attributeNames() {
                return Array.from(this._values.keys());
            }
            kindOf(name) {
                return this._values.get(name)[1];
            }
            setSourceRange(r) {
                this._source_range = r instanceof ast.AST ? r.location || '' : r;
                return this;
            }
            sourceRange() {
                return this._source_range;
            }
            print_attributes(out, ignore_subgraph) {
                ignore_subgraph = ignore_subgraph || false;
                out.write('[');
                const names = this.attributeNames();
                for (let i = 0; i < names.length; i++) {
                    const name = names[i];
                    if (ignore_subgraph && name === 'subgraph') {
                        continue;
                    }
                    if (i > 0) {
                        out.write(', ');
                    }
                    out.write(`${name}=`);
                    out.write(this._values.get(name)[0]); // this.printAttrValue(out, name);
                }
                out.write(']');
            }
            print(out, level, groups, print_source_locations, print_attributes, print_scopes, print_body) {
                print_source_locations = print_source_locations === false ? false : true;
                print_attributes = print_attributes === false ? false : true;
                print_scopes = print_scopes === false ? false : true;
                print_body = print_body === false ? false : true;
                const outs = this.outputs();
                torch._C.indent(out, level);
                torch._C.const_value_list_with_types(out, outs, ', ');
                out.write(' = ');
                if (this.kind() === 'prim::PythonOp') {
                    throw new python.Error('Not implemented.');
                } else if (this.hasAttribute('subgraph') && groups) {
                    throw new python.Error('Not implemented.');
                } else {
                    out.write(this.kind());
                    if (print_attributes && this.hasAttributes()) {
                        this.print_attributes(out);
                    }
                }
                out.write('(');
                torch._C.printValueRefs(out, this.inputs());
                out.write(')');
                if (!print_body) {
                    return out;
                }
                out.write('\n');
                for (let i = 0; i < this.blocks().length; i++) {
                    const b = this.blocks()[i];
                    torch._C.indent(out, level + 1);
                    out.write(`block${i}(`);
                    torch._C.const_value_list_with_types(out, b.inputs());
                    out.write('):\n');
                    for (const nested of b.nodes()) {
                        nested.print(out, level + 2, groups);
                    }
                    torch._C.indent(out, level + 2);
                    out.write('-> (');
                    torch._C.printValueRefs(out, b.outputs());
                    out.write(')\n');
                }
                return out;
            }
        });
        this.registerType('torch.Value', class {
            constructor(node) {
                this._unique = node._graph._next_unique++;
                this._node = node;
                this._uses = [];
            }
            unique() {
                return this._unique;
            }
            node() {
                return this._node;
            }
            owningGraph() {
                return this._node.owningGraph();
            }
            uses() {
                return this._uses;
            }
            hasDebugName() {
                return this._unique_name;
            }
            setDebugName(name) {
                // if (!isValidName(name)) {
                //     throw std::runtime_error("Invalid name: '" + name + "'");
                // }
                const names = this.node().owningGraph()._unique_names;
                if (this.hasDebugName()) {
                    names.delete(this._unique_name);
                    this._unique_name = '';
                }
                if (!name) {
                    return this;
                }
                const old_owner_of_name = names.get(name);
                if (old_owner_of_name) {
                    let suffix = 1;
                    let name_base = name;
                    const last_dot_pos = name.lastIndexOf('.');
                    if (last_dot_pos !== -1) {
                        if (/^\d+$/.test(name.substring(last_dot_pos + 1))) {
                            suffix = Number(name.substring(last_dot_pos + 1));
                            name_base = name.substring(0, last_dot_pos);
                        }
                    }
                    const names_suffixes = this.node().owningGraph()._name_base_suffix;
                    if (names_suffixes.has(name_base)) {
                        suffix = Math.max(suffix, names_suffixes.get(name_base));
                    }
                    let replacement_name = null;
                    do {
                        replacement_name = `${name_base}.${suffix++}`;
                    } while (names.has(replacement_name));
                    names_suffixes.set(name_base, suffix);
                    old_owner_of_name.setDebugName(replacement_name);
                }
                names.set(name, this);
                this._unique_name = name;
                return this;
            }
            debugName() {
                if (this.hasDebugName()) {
                    return this._unique_name;
                }
                return this.unique().toString();
            }
            type() {
                return this._type;
            }
            setType(type) {
                if (type instanceof torch._C.DynamicType) {
                    type = type.fallback();
                }
                this._type = type;
                for (const use of this._uses) {
                    use.user._op = null;
                }
                return this;
            }
            set value(value) { // remove
                if (value instanceof torch.Value) {
                    throw new python.Error('Value cannot be a value.');
                }
                this._value = value;
            }
            get value() { // remove
                return this._value;
            }
            replaceFirstUseWith(newValue) {
                const [u] = this.uses();
                u.user._inputs[u.offset] = newValue;
                newValue._uses.push(u);
                this._uses.shift();
            }
            replaceAllUsesWith(newValue) {
                while (this.uses().length > 0) {
                    this.replaceFirstUseWith(newValue);
                }
            }
            copyMetadata(from) {
                this.setType(from.type());
                if (from.hasDebugName()) {
                    this.setDebugName(from.debugName());
                }
                return this;
            }
            toString() {
                const list = [];
                list.push(this.debugName());
                list.push(' : ');
                list.push(this.type().toString());
                return list.join('');
            }
        });
        this.registerType('torch.Use', class {
            constructor(node, offset) {
                this._node = node;
                this._offset = offset;
            }
            get user() {
                return this._node;
            }
            get offset() {
                return this._offset;
            }
        });
        this.registerType('torch._C.IValue', class {
            constructor(value) {
                this.value = value;
                this.tag = 'None';
                if (typeof value === 'boolean') {
                    this.tag = 'Bool';
                } else if (typeof value === 'string') {
                    this.tag = 'String';
                } else if (value instanceof torch.Tensor) {
                    this.tag = 'Tensor';
                } else if (value instanceof torch.ScriptObject) {
                    this.tag = 'Object';
                } else {
                    throw new python.Error('Unsupported type.');
                }
            }
            isObject() {
                return this.tag === 'Object';
            }
            toObject() {
                return this.value;
            }
            isTensor() {
                return this.tag === 'Tensor';
            }
            toTensor() {
                return this.value;
            }
            isInt() {
                return this.tag === 'Int';
            }
            toInt() {
                if (this.isInt()) {
                    return this.value;
                } else if (this.isSymInt()) {
                    return this.toSymInt().guard_int(/* __FILE__, __LINE__ */);
                }
                throw new python.Error('Expected int.');
            }
        });
        this.registerFunction('torch._C.indent', (out, level) => {
            for (let i = 0; i < level; i++) {
                out.write('  ');
            }
            return out;
        });
        this.registerFunction('torch._C.printValueRef', (out, n) => {
            out.write(`%${n.debugName()}`);
        });
        this.registerFunction('torch._C.printValueRefs', (out, nodes) => {
            for (let i = 0; i < nodes.length; i++) {
                const n = nodes[i];
                if (i > 0) {
                    out.write(', ');
                }
                torch._C.printValueRef(out, n);
            }
            return out;
        });
        this.registerFunction('torch._C.const_value_list_with_types', (out, values, delim) => {
            for (let i = 0; i < values.length; i++) {
                const n = values[i];
                if (i > 0) {
                    out.write(delim);
                }
                torch._C.printValueRef(out, n);
                out.write(' : ');
                out.write(n.type().toString());
            }
        });
        this.register('torch.jit._script');
        this.register('torch.jit._trace');
        this.registerType('torch._C.Source', class {
            constructor(text_view, filename) {
                this._text_view = text_view;
                this._filename = filename;
            }
            text_str() {
                return this._text_view;
            }
            filename() {
                return this._filename;
            }
        });
        this.registerType('torch._C.QualifiedName', class {
            constructor(...args) {
                let name = null;
                if (args.length === 1 && typeof args[0] === 'string') {
                    [name] = args;
                } else if (args.length === 1 && Array.isArray(args[0]) && args[0].every((arg) => typeof arg === 'string')) {
                    name = args[0].join('.');
                } else {
                    name = `${args[0].qualifiedName()}.${args[1]}`;
                }
                const index = name.lastIndexOf('.');
                this._qualifiedName = name;
                this._prefix = index === -1 ? '' : name.substring(0, index);
                this._name = index === -1 ? name : name.substring(index + 1);
            }
            qualifiedName() {
                return this._qualifiedName; // "foo.bar.baz"
            }
            prefix() {
                return this._prefix; // "foo.bar"
            }
            name() {
                return this._name; // "baz"
            }
            atoms() {
                return this._qualifiedName.split('.');
            }
        });
        this.registerType('torch._C.Resolver', class {
            resolveValue() {
                throw new python.Error('Not implemented.');
            }
            resolveType() {
                throw new python.Error('Not implemented.');
            }
        });
        this.registerType('torch._C.SourceImporter', class extends torch._C.Resolver {
            constructor(cu, constant_table, source_loader, version) {
                super();
                this._cu = cu;
                this._constant_table = constant_table;
                this._source_loader = source_loader;
                this._version = version;
                this._loaded_sources = new Set();
                this._to_be_defined = new Map();
                this._env = new Map([
                    ['torch', new torch._C.BuiltinModule('aten', version)],
                    ['ops', new torch._C.OpsValue(version)],
                    ['CONSTANTS', new torch._C.ConstantTableValue(constant_table)],
                    ['fork', torch._C.SpecialFormValue.create('prim::fork')],
                    ['awaitable', torch._C.SpecialFormValue.create('prim::awaitable')],
                    ['annotate', torch._C.SpecialFormValue.create('prim::annotate')],
                    ['unchecked_cast', torch._C.SpecialFormValue.create('prim::unchecked_cast')],
                    ['uninitialized', torch._C.SpecialFormValue.create('prim::Uninitialized')],
                ]);
            }
            loadType(name) {
                const type_parser = new torch.jit.ScriptTypeParser(this);
                return type_parser.parseType(name.qualifiedName());
            }
            resolveType(name) {
                name = new torch._C.QualifiedName(name);
                return this.findNamedType(name);
            }
            resolveValue(name, m, loc) {
                if (this._env.has(name)) {
                    return this._env.get(name);
                }
                const graph = m.graph();
                switch (name) {
                    case 'inf': return new torch._C.SimpleValue(graph.insertConstant('std::numeric_limits<double>::infinity()', loc));
                    case 'nan': return new torch._C.SimpleValue(graph.insertConstant('std::numeric_limits<double>::quiet_NaN()', loc));
                    case 'infj': return new torch._C.SimpleValue(graph.insertConstant('c10::complex<double>(0, std::numeric_limits<double>::infinity())', loc));
                    case 'nanj': return new torch._C.SimpleValue(graph.insertConstant('c10::complex<double>(0, std::numeric_limits<double>::quiet_NaN()', loc));
                    case '__torch__': return new torch._C.ClassNamespaceValue(new torch._C.QualifiedName(name), this);
                    default: return null;
                }
            }
            findNamedType(name) {
                // if (auto custom_class = getCustomClass(name.qualifiedName())) {
                //     return custom_class;
                // }
                this.parseSourceIfNeeded(name.prefix());
                const key = name.qualifiedName();
                const it = this._to_be_defined.get(key);
                if (it && it instanceof ast.ClassDef) {
                    this._to_be_defined.delete(key);
                    this.importNamedType(name.prefix(), it);
                }
                return this._cu.get_type(name);
            }
            importNamedType(qualifier, class_def) {
                const qualified_name = new torch._C.QualifiedName(`${qualifier}.${class_def.name}`);
                if (class_def.bases.length === 0) {
                    this.importClass(qualified_name, class_def, false);
                    return;
                }
                const superclass_name = class_def.bases[0].id;
                if (superclass_name === 'Module') {
                    this.importClass(qualified_name, class_def, true);
                } else if (superclass_name === 'NamedTuple') {
                    this.importNamedTuple(qualified_name, class_def);
                } else if (superclass_name === 'Interface') {
                    // this._cu.define_interface(qualified_name, class_def, shared_from_this(), is_module=false);
                } else if (superclass_name === 'ModuleInterface') {
                    // this._cu.define_interface(qualified_name, class_def, shared_from_this(), is_module=true);
                } else if (superclass_name === 'Enum') {
                    // importEnum(qualified_name, class_def);
                } else {
                    throw new python.Error('TorchScript does not support class inheritance.');
                }
            }
            importClass(qualified_classname, class_def, is_module) {
                if (qualified_classname.prefix().startsWith('__torch__.torch.classes')) {
                    return;
                }
                const parameter_names = new Set();
                const buffer_names = new Set();
                const methods = [];
                const method_resolvers = [];
                const attributes = [];
                const constants = [];
                const pre_hook_names = new Set();
                const pre_hook_def_map = new Map();
                const hook_names = new Set();
                const hook_def_map = new Map();
                const class_type = torch.ClassType.create(qualified_classname.qualifiedName(), this._cu, is_module);
                for (const stmt of class_def.body) {
                    if (stmt instanceof ast.Assign || stmt instanceof ast.AnnAssign) {
                        let target = null;
                        let annotation = null;
                        let value = null;
                        if (stmt instanceof ast.Assign) {
                            [target] = stmt.targets;
                            value = stmt.value;
                        } else {
                            target = stmt.target;
                            annotation = stmt.annotation;
                            value = stmt.value;
                        }
                        if (target instanceof ast.Name) {
                            const name = this._cu.execution.identifier(target);
                            switch (name) {
                                case '__annotations__': {
                                    continue;
                                }
                                case '__parameters__': {
                                    for (const elt of value.elts) {
                                        parameter_names.add(elt.value);
                                    }
                                    break;
                                }
                                case '__buffers__': {
                                    for (const elt of value.elts) {
                                        buffer_names.add(elt.value);
                                    }
                                    break;
                                }
                                case '__forward_pre_hooks__': {
                                    for (const elt of value.elts) {
                                        pre_hook_names.add(elt.value);
                                    }
                                    break;
                                }
                                case '__forward_hooks__': {
                                    for (const elt of value.elts) {
                                        hook_names.add(elt.value);
                                    }
                                    break;
                                }
                                default: {
                                    if (value) {
                                        constants.push({ name, value, annotation });
                                    } else {
                                        attributes.push({ name, value, annotation });
                                    }
                                    break;
                                }
                            }
                        } else if (target instanceof ast.Subscript && target.value instanceof ast.Name && target.value.id === '__annotations__') {
                            const name = target.slice.elts[0].value;
                            attributes.push({ name, value, annotation: stmt.value });
                            continue;
                        } else {
                            throw new python.Error('Unexpected statement kind in module metadata.');
                        }
                    } else if (stmt instanceof ast.FunctionDef) {
                        const def = stmt;
                        const def_name = def.name;
                        if (pre_hook_names.has(def_name)) {
                            pre_hook_def_map.set(def_name, def);
                        } else if (hook_names.has(def_name)) {
                            hook_def_map.set(def_name, def);
                        } else {
                            methods.push(def);
                            method_resolvers.push(this);
                        }
                    } else {
                        throw new python.Error('Unexpected statement kind in class body.');
                    }
                }
                for (const assign of attributes) {
                    const name = assign.name;
                    const annotation = this._cu.execution.type(assign.annotation, null);
                    const is_parameter = parameter_names.has(name);
                    const is_buffer = buffer_names.has(name);
                    class_type.addAttribute(name, annotation, is_parameter, is_buffer);
                }
                for (const constant of constants) {
                    class_type.addConstant(constant.name, constant.value);
                }
                this._cu.register_type(class_type);
                const self = new torch._C.SimpleSelf(class_type);
                this._cu.define(qualified_classname, [], [], methods, method_resolvers, self, false, this._version);
            }
            importNamedTuple(qualified_name, named_tuple_def) {
                const field_names = [];
                const field_types = [];
                const field_defaults = [];
                for (const stmt of named_tuple_def.body) {
                    if (stmt instanceof ast.AnnAssign === false) {
                        throw new python.Error('Unexpected statement in NamedTuple body.');
                    }
                    const target = this._cu.execution.identifier(stmt.target);
                    const annotation = this._cu.execution.type(stmt.annotation);
                    field_names.push(target);
                    field_types.push(annotation);
                }
                const tt = torch.TupleType.createNamed(qualified_name.qualifiedName(), field_names, field_types, field_defaults);
                this._cu.register_type(tt);
            }
            importFunction(qualifier, def) {
                const definitions = [def];
                const resolvers = [this];
                this._cu.define(new torch._C.QualifiedName(qualifier), /*properties=*/[], /*propResolvers=*/[], definitions, resolvers, null);
            }
            parseSourceIfNeeded(qualifier) {
                if (!qualifier || this._loaded_sources.has(qualifier)) {
                    return;
                }
                this._loaded_sources.add(qualifier);
                const src = this._source_loader(qualifier);
                if (!src) {
                    return;
                }
                const program = this._cu.execution.parse(src.filename(), src.text_str(), null);
                for (const stmt of program.body) {
                    if (stmt instanceof ast.ClassDef) {
                        const name = `${qualifier}.${stmt.name}`;
                        this._to_be_defined.set(name, stmt);
                    } else if (stmt instanceof ast.FunctionDef) {
                        const name = `${qualifier}.${stmt.name}`;
                        this._to_be_defined.set(name, stmt);
                    }
                }
            }
            findFunction(name) {
                this.parseSourceIfNeeded(name.prefix());
                const key = name.qualifiedName();
                const it = this._to_be_defined.get(key);
                if (it && it instanceof ast.FunctionDef) {
                    this._to_be_defined.delete(key);
                    this.importFunction(name.prefix(), it);
                }
                return this._cu.find_function(name);
            }
        });
        this.registerType('torch._C.FunctionResolver', class extends torch._C.Resolver {
            constructor(otherResolver, functionTable) {
                super();
                this._otherResolver = otherResolver;
                this._functionTable = functionTable;
            }
            resolveType(name, loc) {
                return this._otherResolver.resolveType(name, loc);
            }
        });
        this.registerType('torch._C.ScriptModuleDeserializer', class {
            constructor(cu, reader, pickle_dir_prefix, tensor_dir_prefix, storage_context) {
                this._compilation_unit = cu;
                this._reader = reader;
                this._storage_context = storage_context;
                this._code_prefix = !pickle_dir_prefix && !tensor_dir_prefix ? 'code/' : '.data/ts_code/code/';
                this._pickle_dir_prefix = pickle_dir_prefix || '';
                this._tensor_dir_prefix = tensor_dir_prefix || '';
                this._constant_table = [];
                const SourceLoader = (qualifier) => {
                    return this.findSourceInArchiveFromQualifier(this._reader, this._code_prefix, qualifier);
                };
                this._source_importer = new torch._C.SourceImporter(this._compilation_unit, this._constant_table, SourceLoader, reader.version());
            }
            deserialize() {
                const execution = this._compilation_unit.execution;
                const code_prefix = this._code_prefix;
                for (const name of this._reader.get_all_records()) {
                    if (name.startsWith(code_prefix) && name.endsWith('.py')) {
                        const file = name.substring(code_prefix.length);
                        const stream = this._reader.get_record(name);
                        const buffer = stream.peek();
                        execution.add(file, buffer);
                    }
                }
                const torch = execution.import('torch');
                execution.builtins.torch = torch;
                execution.builtins.Tensor = torch.Tensor;
                execution.builtins.ops = torch.ops;
                execution.builtins.inf = torch.inf;
                execution.builtins.CONSTANTS = {};
                execution._resolver = this._source_importer;
                const known_types = [
                    { name: '__torch__.torch.classes._nnapi.Compilation', methods: [
                        '__init__(__torch__.torch.classes._nnapi.Compilation self) -> NoneType',
                        'init(__torch__.torch.classes._nnapi.Compilation self, Tensor serialized_model_tensor, Tensor[] parameter_buffers) -> NoneType',
                        'init2(__torch__.torch.classes._nnapi.Compilation self, Tensor serialized_model_tensor, Tensor[] parameter_buffers, int compilation_preference, bool relax_f32_to_f16) -> NoneType',
                        'run(__torch__.torch.classes._nnapi.Compilation self, Tensor[] inputs, Tensor[] outputs) -> NoneType'
                    ] },
                    { name: '__torch__.torch.classes.quantized.Conv2dPackedParamsBase', attributes: 'Tensor weight, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups' },
                    { name: '__torch__.torch.classes.quantized.Conv3dPackedParamsBase', attributes: 'Tensor weight, Tensor bias, int[] stride, int[] padding, int[] dilation, int groups' },
                    { name: '__torch__.torch.classes.quantized.LinearPackedParamsBase', attributes: 'Tensor weight, Tensor? bias' },
                    { name: '__torch__.torch.classes.rnn.CellParamsBase', attributes: 'str type, Tensor[] tensors, float[] doubles, int[] longs, __torch__.torch.classes.quantized.LinearPackedParamsBase[] packed_params' },
                    { name: '__torch__.torch.classes.xnnpack.Conv2dOpContext', attributes: 'Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, int[] output_min, int[] output_max' },
                    { name: '__torch__.torch.classes.xnnpack.LinearOpContext', attributes: 'Tensor weight, Tensor bias, int[] output_min, int[] output_max' },
                    { name: '__torch__.torch.classes.xnnpack.TransposeConv2dOpContext', attributes: 'Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups, int[] output_min, int[] output_max' },
                ];
                for (const known_type of known_types) {
                    const prefix = new torch._C.QualifiedName(known_type.name);
                    const type = torch.ClassType.create(known_type.name, this._compilation_unit, false);
                    for (const known_method of known_type.methods || []) {
                        const schema = new torch.FunctionSchema(known_method);
                        const name = new torch._C.QualifiedName(prefix, schema.name);
                        const fn = new torch.jit.BuiltinOpFunction(name, schema);
                        type.addMethod(fn);
                    }
                    if (known_type.attributes) {
                        const schema = new torch.FunctionSchema(`(${known_type.attributes}) -> ()`);
                        for (const arg of schema.arguments) {
                            type.addAttribute(arg.name, arg.real_type);
                        }
                    }
                    this._compilation_unit.register_type(type);
                }
                if (this._reader.has_record('model.json')) {
                    return this.LEGACY_deserialize();
                }
                const constants = this.readArchive('constants');
                for (let i = 0; i < constants.length; i++) {
                    let val = constants[i];
                    if (val && val.__class__ && val.__class__.__module__.startsWith('__torch__.torch.classes.')) {
                        const type = this._source_importer.resolveType(`${val.__class__.__module__}.${val.__class__.__name__}`);
                        const obj = torch.ScriptObject.create(type);
                        obj._ivalue = val;
                        val = obj;
                    }
                    execution.builtins.CONSTANTS[`c${i}`] = val;
                    this._constant_table.push(val);
                }
                const obj = this.readArchive('data');
                const convertObject = (obj) => {
                    if (obj.__class__) {
                        const name = `${obj.__class__.__module__}.${obj.__class__.__name__}`;
                        const type = this._source_importer.loadType(new torch._C.QualifiedName(name));
                        const module = type.is_module() ? new torch.ScriptModule(type, this._compilation_unit) : new torch.ScriptObject(type);
                        for (let i = 0; i < type.numAttributes(); i++) {
                            const k = type.getAttributeName(i);
                            const t = type.getAttribute(i);
                            const v = obj[k];
                            if (t instanceof torch.ClassType) {
                                module.__setattr__(k, convertObject(v));
                            } else {
                                if (t instanceof torch.TensorType && v && v.__class__ && v instanceof torch.Tensor === false && v.__class__.__module__ === '__torch__.torch.classes.quantized') {
                                    const name = `${v.__class__.__module__}.${v.__class__.__name__}`;
                                    type._attributes[i].type = this._source_importer.resolveType(name);
                                }
                                module.__setattr__(k, obj[k]);
                            }
                        }
                        for (const [key, value] of Object.entries(Object.getPrototypeOf(obj))) {
                            if (value && value.__class__ === builtins.method) {
                                module[key] = value;
                            }
                        }
                        return module;
                    }
                    throw new python.Error('Module class not found.');
                };
                return convertObject(obj);
            }
            LEGACY_deserialize() {
                // https://github.com/pytorch/pytorch/blob/5e69e11d098a2cfccc8a59377c431e9c71cab9a8/torch/csrc/jit/serialization/import_legacy.cpp#L88
                const execution = this._compilation_unit.execution;
                const caffe2 = execution.proto.caffe2;
                const torch = execution.import('torch');
                const stream = this._reader.get_record('model.json');
                const buffer = stream.peek();
                const decoder = new TextDecoder('utf-8');
                const content = decoder.decode(buffer);
                const obj = JSON.parse(content);
                const model = execution.proto.torch.ModelDef.decodeJson(obj);
                const tensorTypeMap = new Map([
                    [caffe2.TensorProto.DataType.FLOAT, 'Float'],
                    [caffe2.TensorProto.DataType.FLOAT16, 'Half'],
                    [caffe2.TensorProto.DataType.DOUBLE, 'Double'],
                    [caffe2.TensorProto.DataType.INT8, 'Char'],
                    [caffe2.TensorProto.DataType.INT32, 'Int'],
                    [caffe2.TensorProto.DataType.INT64, 'Long']
                ]);
                const tensor_table = (model.tensors || []).map((constant) => {
                    const key = constant.data.key;
                    if (!tensorTypeMap.has(constant.data_type)) {
                        throw new python.Error(`Unsupported tensor data type '${constant.data_type}'.`);
                    }
                    const type = tensorTypeMap.get(constant.data_type);
                    const shape = constant.dims ? constant.dims.map((dim) => parseInt(dim, 10)) : null;
                    const strides = constant.strides ? constant.strides.map((dim) => parseInt(dim, 10)) : null;
                    const storage_type = execution.resolve(`torch.${type}Storage`);
                    const size = (shape || []).reduce((a, b) => a * b, 1);
                    const offset = parseInt(constant.offset, 10) || 0;
                    const storage = new storage_type(size);
                    const itemsize = storage.dtype.itemsize();
                    const stream = this._reader.get_record(key);
                    if (stream) {
                        const buffer = stream.peek();
                        const length = size * itemsize;
                        const data = buffer.slice(offset, offset + length);
                        storage._set_cdata(data);
                    }
                    const tensor = torch._utils._rebuild_tensor(storage, 0, shape, strides);
                    tensor.name = key;
                    return tensor;
                });
                execution.builtins.CONSTANTS = {};
                for (let i = 0; i < tensor_table.length; i++) {
                    execution.builtins.CONSTANTS[`c${i}`] = tensor_table[i];
                }
                const attributes = [];
                if (this._reader.has_record('attributes.pkl')) {
                    const stream = this._reader.get_record('attributes.pkl');
                    const buffer = stream.peek();
                    const unpickler = new pickle.Unpickler(buffer);
                    const obj = unpickler.load();
                    attributes.push(...obj);
                }
                this._LEGACY_moduleStack = ['__torch__'];
                const module_def = model.main_module;
                for (const tensor of tensor_table) {
                    this._constant_table.push(tensor);
                }
                return this.LEGACY_convertModule(module_def);
            }
            LEGACY_convertModule(module_def) {
                const atoms = new torch._C.QualifiedName(module_def.name).atoms();
                const numPushed = atoms.length;
                for (const atom of atoms) {
                    const sanitized = /^\d+$/.test(atom) ? `_${atom}` : atom;
                    this._LEGACY_moduleStack.push(sanitized);
                }
                const qn = new torch._C.QualifiedName(this._LEGACY_moduleStack);
                const module = new torch.ScriptModule(qn, this._compilation_unit);
                for (const sub_def of module_def.submodules || []) {
                    const submodule = this.LEGACY_convertModule(sub_def);
                    module.register_module(sub_def.name, submodule);
                }
                for (const param_def of module_def.parameters || []) {
                    const tensor = this._constant_table[Number(param_def.tensor_id)];
                    if (param_def.isBuffer) {
                        module.register_buffer(param_def.name, tensor);
                    } else {
                        module.register_parameter(param_def.name, tensor, false);
                    }
                }
                // const typeParser = new torch.jit.ScriptTypeParser(this._source_importer);
                for (const attr_def of module_def.attributes || []) {
                    if (module.hasattr(attr_def.name)) {
                        continue;
                    }
                    throw new python.Error('Not implemented.');
                    // IValue ivalue;
                    // if (attr_def.id() >= 0) {
                    //    ivalue = LEGACY_pickled_ivalues_.at(attr_def.id());
                    // }
                    // module.register_attribute(attr_def.name, typeParser.parseType(attr_def.type), ivalue);
                }
                if (module_def.torchscript_arena) {
                    const key = module_def.torchscript_arena.key;
                    const file = key.substring('code/'.length);
                    const name = file.replace(/\.py$/, '').split('/').join('.');
                    const code = execution.import(name);
                    if (code.forward.__class__ === execution.builtins.function) {
                        module.forward = code.forward;
                    }
                }
                /*
                std::shared_ptr<SourceRangeUnpickler> gen_ranges = null;
                if (module_def.has_torchscript_debug_arena()) {
                    const [data, size] = reader_->getRecord(module_def.torchscript_debug_arena().key());
                    gen_ranges = std::make_shared<ConcreteSourceRangeUnpickler>(std::move(data), size);
                }
                if (module_def.has_torchscript_arena()) {
                    const [data, size] =
                        reader_->getRecord(module_def.torchscript_arena().key());
                    std::string data_str(static_cast<const char*>(data.get()), size);
                    const src = std::make_shared<Source>(std::string(static_cast<const char*>(data.get()), size), module_def.torchscript_arena().key(), 1, std::move(gen_ranges));
                    source_importer_.LEGACY_import_methods(module, src);
                }
                if (module_def.has_get_state_attribute_id()) {
                    LEGACY_moduleSetState(module, LEGACY_pickled_ivalues_.at(module_def.get_state_attribute_id()));
                }
                const ClassTypePtr& module_type = module._ivalue()->type();
                for (size_t i = 0, N = module_type->numAttributes(); i < N; ++i) {
                    const IValue& v = module._ivalue()->getSlot(i);
                    if (module_type->getAttribute(i)->kind() != TypeKind::OptionalType) {
                        TORCH_CHECK(!v.isNone(), "The field '", module_type->getAttributeName(i), "' was left unitialized after __setstate__, but expected a ", "value of type '", v.type()->repr_str(), "'");
                    }
                }
                */
                for (let i = 0; i < numPushed; i++) {
                    this._LEGACY_moduleStack.pop();
                }
                return module;
            }
            readArchive(archive_name) {
                const type_resolver = (qn) => {
                    const cls = this._source_importer.loadType(qn);
                    return cls;
                };
                const ObjLoaderFunc = (/* type, ivalue */) => {
                };
                return this.readArchiveAndTensors(archive_name, this._pickle_dir_prefix, this._tensor_dir_prefix, type_resolver, ObjLoaderFunc, this._device, this._reader, null, this._storage_context);
            }
            readArchiveAndTensors(archive_name, pickle_prefix, tensor_prefix, type_resolver, obj_loader, device, stream_reader, type_parser, storage_context) {
                const picklename = `${pickle_prefix + archive_name}.pkl`;
                const stream = stream_reader.get_record(picklename);
                if (!stream) {
                    throw new python.Error(`File '${picklename}' is not found.`);
                }
                const buffer = stream.peek();
                const tensor_dir_path = tensor_prefix ? tensor_prefix : `${archive_name}/`;
                const read_record = (name) => {
                    const stream = stream_reader.get_record(tensor_dir_path + name);
                    return stream.length <= 0x40000 ? stream.peek() : stream;
                };
                const execution = this._compilation_unit.execution;
                const pickle = execution.__import__('pickle');
                const Unpickler = class extends pickle.Unpickler {
                    find_class(module, name) {
                        return super.find_class(module, name);
                    }
                };
                const unpickler = new Unpickler(buffer);
                unpickler.persistent_load = (saved_id) => {
                    if (saved_id[0] !== 'storage') {
                        throw new python.Error(`Unsupported persistent load type '${saved_id[0]}'.`);
                    }
                    const [, storage_type, key, , size] = saved_id;
                    if (storage_context && storage_context.has_storage(key)) {
                        return storage_context.get_storage(key);
                    }
                    const storage = new storage_type(size);
                    const storage_ptr = read_record(key);
                    storage._set_cdata(storage_ptr);
                    if (storage_context) {
                        storage_context.add_storage(key);
                    }
                    return storage;
                };
                return unpickler.load();
            }
            qualifierToArchivePath(qualifier, export_prefix) {
                return `${export_prefix}${qualifier.replace(/\./g, '/')}.py`;
            }
            findSourceInArchiveFromQualifier(reader, export_prefix, qualifier) {
                const path = this.qualifierToArchivePath(qualifier, export_prefix);
                if (!reader.has_record(path)) {
                    return null;
                }
                const data = reader.get_record(path);
                return new torch._C.Source(data.peek(), path);
            }
        });
        this.registerType('torch._C.WithInsertPoint', class {
            constructor(x) {
                const n = x instanceof torch.Block ? x.return_node() : x;
                this._prev = n.owningGraph().insertPoint();
                n.owningGraph().setInsertPoint(n);
            }
            dispose() {
                this._prev.owningGraph().setInsertPoint(this._prev);
            }
        });
        this.registerType('torch._C.Environment', class {
            constructor(method, resolver, b, next) {
                this.method = method;
                this.resolver = resolver;
                this.b = b;
                this.next = next;
                this.value_table = new Map();
                this.type_table = new Map();
            }
            block() {
                return this.b;
            }
            getSugaredVar(ident, range, required) {
                required = required || true;
                let retval = this.findInAnyFrame(ident);
                if (!retval) {
                    torch._C.Environment.globals = torch._C.Environment.globals || new Map([
                        ['tuple', torch._C.SpecialFormValue.create('prim::TupleConstruct')],
                        ['float', new torch._C.MagicMethod('__float__', new torch._C.CastValue(torch.FloatType.get(), 'aten::Float'))],
                        ['int', new torch._C.MagicMethod('__int__', new torch._C.CastValue(torch.IntType.get(), 'aten::Int'))],
                        ['bool', new torch._C.MagicMethod('__bool__', new torch._C.CastValue(torch.BoolType.get(), 'aten::Bool'))],
                        ["getattr", torch._C.SpecialFormValue.create('prim::GetAttr')],
                        ["hasattr", torch._C.SpecialFormValue.create('prim::HasAttr')],
                        ["isinstance", torch._C.SpecialFormValue.create('prim::isinstance')],
                        ['range', torch._C.SpecialFormValue.create('prim::range')],
                    ]);
                    if (torch._C.Environment.globals.has(ident)) {
                        retval = torch._C.Environment.globals.get(ident);
                    }
                }
                if (!retval) {
                    const type = this.resolver.resolveType(ident, range);
                    if (type instanceof torch.TupleType) {
                        retval = new torch.jit.NamedTupleConstructor(type);
                    }
                }
                if (!retval) {
                    retval = this.resolver.resolveValue(ident, this.method, range);
                }
                if (!retval) {
                    const type = this.resolver.resolveType(ident, range);
                    if (type instanceof torch.ClassType) {
                        retval = new torch.jit.ClassValue(type);
                    }
                }
                if (!retval && required) {
                    throw new python.Error(`The name '${ident}' is not defined.`);
                }
                return retval;
            }
            setVar(loc, name, value) {
                this.setSugaredVar(loc, name, new torch._C.SimpleValue(value), null);
            }
            setSugaredVar(loc, name, value, annotated_type) {
                let as_simple_value = torch._C.asSimple(value);
                if (as_simple_value && !as_simple_value.hasDebugName() && torch._C.meaningfulName(name) && as_simple_value.node().owningBlock() === this.block()) {
                    as_simple_value.setDebugName(name);
                }
                const parent = this.findInParentFrame(name);
                if (parent) {
                    if (annotated_type) {
                        throw new python.Error('Type already defined in an outer block.');
                    }
                    if (!as_simple_value) {
                        throw new python.Error('Only reassignments to first-class values are allowed.');
                    }
                    const simple_parent = torch._C.asSimple(parent);
                    if (!simple_parent) {
                        throw new python.Error('Only reassignments to first-class values are allowed.');
                    }
                    const parent_type = this.unshapedType(simple_parent.type());
                    as_simple_value = this.tryConvertToType(loc, this.b.owningGraph(), parent_type, as_simple_value, /*allow_conversions=*/true);
                    if (!as_simple_value.type().isSubtypeOf(parent_type)) {
                        throw new python.Error('Incompatible types.');
                    }
                    if (simple_parent.type().kind() === 'ListType' && as_simple_value.type().kind() === 'ListType') {
                        throw new python.Error('Invalid list type.');
                    }
                }
                if (as_simple_value) {
                    if (annotated_type && !as_simple_value.type().isSubtypeOf(annotated_type)) {
                        throw new python.Error('Invalid type.');
                    }
                    const value_store_type = annotated_type ? annotated_type : as_simple_value.type();
                    this.insertStore(name, loc, as_simple_value, value_store_type);
                } else {
                    this.value_table.set(name, value);
                }
            }
            findInThisFrame(name) {
                if (this.value_table.has(name)) {
                    return this.value_table.get(name);
                }
                if (this.type_table.has(name)) {
                    return this.insertLoad(name, this.type_table.get(name));
                }
                return null;
            }
            findInAnyFrame(name) {
                for (let runner = this; runner; runner = runner.next) {
                    const r = runner.findInThisFrame(name);
                    if (r) {
                        return r;
                    }
                }
                return null;
            }
            findInParentFrame(name) {
                return this.next ? this.next.findInAnyFrame(name) : null;
            }
            insertLoad(name, type) {
                const g = this.b.owningGraph();
                const load = g.insertNode(g.createLoad(name, type));
                if (torch._C.meaningfulName(name)) {
                    load.output().setDebugName(name);
                }
                return new torch._C.SimpleValue(load.output());
            }
            insertStore(name, loc, v, type) {
                const g = this.b.owningGraph();
                g.insertNode(g.createStore(name, v)).setSourceRange(loc);
                this.type_table.set(name, type);
            }
        });
        this.registerFunction('torch._C.RefinementSet', class {
        });
        this.registerFunction('torch._C.CondValue', class {
            constructor(...args) {
                if (args.length === 3) {
                    [this._value, this._refinements, this._static_if] = args;
                } else if (args.length === 4) {
                    const [g, loc, refinements, static_value] = args;
                    this._value = g.insertConstant(static_value, loc);
                    this._refinements = refinements;
                    this._static_if = static_value;
                } else {
                    throw new python.Error('Invalid number of arguments.');
                }
            }
            value() {
                return this._value;
            }
            staticIf() {
                return this._static_if;
            }
            refinements() {
                return this._refinements;
            }
        });
        this.registerFunction('torch._C.asSimple', (value) => {
            if (value instanceof torch._C.SimpleValue) {
                return value.getValue();
            }
            return null;
        });
        this.registerFunction('torch._C.meaningfulName', (name) => {
            if (name.length === 0 && name[0] === '$') {
                return false;
            }
            return name[0] !== '_' && !/[0-9]/.test(name.slice(1));
        });
        this.registerFunction('torch._C.materializeConstant', (val, graph, r, map) => {
            const existing_constant = map.get(val);
            if (existing_constant) {
                return existing_constant;
            }
            const guard = new torch._C.WithInsertPoint(graph.block().nodes()[0]);
            const new_constant = graph.insertConstant(val, r);
            map.set(val, new_constant);
            guard.dispose();
            return new_constant;
        });

        this.registerFunction('torch._C.getFullSchemaName', (schema) => {
            if (!schema.overload_name) {
                return `${schema.name}.${schema.overload_name}`;
            }
            return schema.name;
        });
        this.registerFunction('torch._C.insertGraph', (g, callee, inputs, value_map) => {
            const value_map_func = (v) => value_map.get(v);
            if (callee.inputs().length !== inputs.length) {
                throw new python.Error('Invalid number of inputs.');
            }
            for (let i = 0; i < inputs.length; i++) {
                value_map.set(callee.inputs()[i], inputs[i]);
            }
            for (const node of callee.nodes()) {
                const new_node = g.insertNode(g.createClone(node, value_map_func));
                for (let i = 0; i < node.outputs().length; i++) {
                    value_map.set(node.outputs()[i], new_node.outputs()[i]);
                }
            }
            const outputs = [];
            for (const output of callee.outputs()) {
                outputs.push(value_map_func(output));
            }
            return outputs;
        });
        this.registerFunction('torch._C.getAllBuiltinFunctionsFor', () => {
            return [];
        });
        this.registerFunction('torch._C.get_operator_version_map', () => {
            return new Map();
        });
        this.registerFunction('torch._C.varargsCanBeUsedAsList', (schema, arg_index, arg) => {
            const is_last_argument = arg_index + 1 === schema.arguments.length || schema.arguments[arg_index + 1].kwarg_only;
            let arg_type = arg.type;
            if (arg_type instanceof torch._C.DynamicType) {
                arg_type = arg_type.fallback();
            }
            const argument_is_list = arg_type.kind() === 'ListType';
            const typevar_list = argument_is_list && arg_type.getElementType().kind() === 'VarType';
            const arg_is_broadcasting_list = arg.N > 0;
            return is_last_argument && argument_is_list && !arg_is_broadcasting_list && !typevar_list;
        });
        this.registerFunction('torch._C.isBlockListedSchema', (schema) => {
            if ((schema.name === 'aten::view' && schema.overload_name === 'dtype') ||
                (schema.name === 'aten::max' && schema.overload_name === 'unary_out') ||
                (schema.name === 'aten::min' && schema.overload_name === 'unary_out')) {
                return true;
            }
            return false;
        });
        this.registerFunction('torch._C.unwrapOptional', (opt_type) => {
            if (opt_type instanceof torch._C.DynamicType) {
                return torch._C.unwrapOptional(opt_type.fallback());
            }
            if (opt_type instanceof torch.OptionalType) {
                return opt_type.getElementType();
            }
            return opt_type;
        });
        this.registerFunction('torch._C.isOpCurrentBasedOnUpgraderEntries', (upgraders_for_schema, current_version) => {
            const latest_update = upgraders_for_schema[upgraders_for_schema.length - 1].bumped_at_version;
            return current_version < latest_update;
        });
        this.registerFunction('torch._C.isOpSymbolCurrent', (name, current_version) => {
            const it = torch._C.get_operator_version_map().get(name);
            if (it) {
                return torch._C.isOpCurrentBasedOnUpgraderEntries(it, current_version);
            }
            return true;
        });
        this.registerFunction('torch._C.packOutputs', (g, values, field_names) => {
            if (values.length === 1) {
                return values[0];
            }
            let named_tuple = null;
            if (field_names) {
                const types = values.map((v) => v.type());
                named_tuple = torch.TupleType.createNamed(null, field_names.value(), types);
            }
            return g.insertNode(g.createTuple(values, named_tuple)).output();
        });
        this.registerFunction('torch._C.isIntOrFloatUsedAsList', (value, arg) => {
            const v_type = value.type();
            if (v_type !== torch.FloatType.get() && v_type !== torch.IntType.get()) {
                return false;
            }
            const arg_type = torch._C.unwrapOptional(arg.type);
            return arg_type instanceof torch.ListType && arg_type.getElementType() === v_type && arg.N;
        });
        this.registerFunction('torch._C.convertibleToList', (type, list_type_) => {
            const list_type = list_type_;
            if (list_type instanceof torch.ListType) {
                return false;
            }
            if (type.isSubtypeOf(list_type_)) {
                return true;
            }
            if (type instanceof torch.TupleType) {
                return type.elements().every((t) => t.isSubtypeOf(list_type.getElementType()));
            }
            return false;
        });
        this.registerFunction('torch._C.findInputWithName', (name, kwargs, is_aten) => {
            for (let i = 0; kwargs.length; i++) {
                if (is_aten && name === 'self' && kwargs[i].name() === 'input') {
                    return i;
                }
                if (kwargs[i].name() === name) {
                    return i;
                }
            }
            return null;
        });
        this.registerFunction('torch._C.tryCreateList', (elem_type, graph, loc, varargs, failure_messages, err, convert_tensor_to_num, type_env) => {
            const elem_arg = new torch.Argument('<varargs>', elem_type);
            const list_elements = [];
            for (const named_value of varargs) {
                const matched_value = torch._C.tryMatchArgument(/*arg=*/elem_arg, graph, loc, named_value, failure_messages, err, /*allow_conversions=*/convert_tensor_to_num, type_env);
                if (!matched_value) {
                    return null;
                }
                list_elements.push(matched_value);
            }
            return graph.insertNode(graph.createList(elem_type, list_elements)).output();
        });
        this.registerType('torch._C.MatchTypeReturn', class {
            constructor(reason) {
                this._reason = reason;
            }
            static Success() {
                return new torch._C.MatchTypeReturn(null);
            }
            success() {
                return this._reason === null;
            }
        });
        this.registerFunction('torch._C.matchTypeVariables', (formal, actual, type_env) => {
            if (!formal.hasFreeVariables()) {
                if (formal instanceof torch._C.DynamicType) {
                    return torch._C.matchTypeVariables(formal.fallback(), actual, type_env);
                }
                return torch._C.MatchTypeReturn.Success();
            }
            if (formal instanceof torch.VarType) {
                const it = type_env.get(formal.name);
                if (it === null) {
                    type_env.set(formal.name, actual);
                    return torch._C.MatchTypeReturn.Success();
                } else if (torch._C.unifyTypes(it, actual)) {
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match var.');
            } else if (formal instanceof torch.ListType) {
                if (actual instanceof torch.ListType) {
                    const innerMatch = torch._C.matchTypeVariables(formal.getElementType(), actual.getElementType(), type_env);
                    if (!innerMatch.success()) {
                        return innerMatch;
                    }
                    return torch._C.MatchTypeReturn.Success();
                } else if (actual instanceof torch.TupleType) {
                    const maybe_tuple_unified = torch._C.unifyTypeList(actual.elements(), '');
                    if (maybe_tuple_unified) {
                        return torch._C.matchTypeVariables(formal.getElementType(), maybe_tuple_unified, type_env);
                    }
                }
                return new torch._C.MatchTypeReturn('Cannot match list.');
            } else if (formal instanceof torch.TupleType) {
                if (actual instanceof torch.TupleType) {
                    if (formal.elements().length !== actual.elements().length) {
                        return torch._C.MatchTypeReturn('Cannot match tuples of mismatched size.');
                    }
                    for (let i = 0; i < formal.elements().length; i++) {
                        const result = torch._C.matchTypeVariables(formal.elements()[i], actual.elements()[i], type_env);
                        if (!result.success()) {
                            return result;
                        }
                    }
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match tuple.');
            } else if (formal instanceof torch.FutureType) {
                if (actual instanceof torch.FutureType) {
                    const innerMatch = torch._C.matchTypeVariables(formal.getElementType(), actual.getElementType(), type_env);
                    if (!innerMatch.success()) {
                        return innerMatch;
                    }
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match future.');
            } else if (formal instanceof torch.AwaitType) {
                if (actual instanceof torch.AwaitType) {
                    const innerMatch = torch._C.matchTypeVariables(formal.getElementType(), actual.getElementType(), type_env);
                    if (!innerMatch.success()) {
                        return innerMatch;
                    }
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match await.');
            } else if (formal instanceof torch.RRefType) {
                if (actual instanceof torch.RRefType) {
                    const innerMatch = torch._C.matchTypeVariables(formal.getElementType(), actual.getElementType(), type_env);
                    if (!innerMatch.success()) {
                        return innerMatch;
                    }
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match rref.');
            } else if (formal instanceof torch.OptionalType) {
                if (actual instanceof torch.OptionalType) {
                    const optionedMatch = torch._C.matchTypeVariables(formal.getElementType(), actual.getElementType(), type_env);
                    if (!optionedMatch.success()) {
                        return optionedMatch;
                    }
                } else if (!actual.isSubtypeOf(torch.NoneType.get())) {
                    return torch._C.matchTypeVariables(formal.getElementType(), actual, type_env);
                }
                return torch._C.MatchTypeReturn.Success();
            } else if (formal instanceof torch.DictType) {
                if (actual instanceof torch.DictType) {
                    const key_match = torch._C.matchTypeVariables(formal.getKeyType(), actual.getKeyType(), type_env);
                    if (!key_match.success()) {
                        return key_match;
                    }
                    const value_match = torch._C.matchTypeVariables(formal.getValueType(), actual.getValueType(), type_env);
                    if (!value_match.success()) {
                        return value_match;
                    }
                    return torch._C.MatchTypeReturn.Success();
                }
                return new torch._C.MatchTypeReturn('Cannot match dict.');
            }
            throw new python.Error('Unhandled free variable container.');
        });
        this.registerFunction('torch._C.tryMatchArgument', (arg, graph, loc, named_value, failure_messages, err, allow_conversions, type_env) => {
            let value = named_value.value(graph);
            if (torch._C.isIntOrFloatUsedAsList(value, arg)) {
                const repeated = Array(arg.N).fill(value);
                value = graph.insertNode(graph.createList(value.type(), repeated)).output();
            }
            const matched = torch._C.matchTypeVariables(arg.type, value.type(), type_env);
            if (!matched.success()) {
                if (failure_messages) {
                    throw new python.Error(`Could not match type ${value.type().repr_str()} to ${arg.type().repr_str()} in argument '${arg.name()}'.`);
                }
                return null;
            }
            const concrete_type = torch._C.tryEvalTypeVariables(arg.type, type_env);
            if (!concrete_type) {
                if (failure_messages) {
                    throw new python.Error(`Could not infer type for argument '${arg.name}'.`);
                }
                return null;
            }
            value = torch._C.tryConvertToType(loc, graph, concrete_type, value, allow_conversions);
            if (!value.type().isSubtypeOf(concrete_type)) {
                if (failure_messages) {
                    throw new python.Error(`Could not match type in argument '${arg.name()}'.`);
                }
                return null;
            }
            return value;
        });
        this.registerFunction('torch._C.tryConvertToType', (loc, graph, concrete_type, value, allow_conversions) => {
            // treat conversion to Optional[T] as conversions to T
            if (concrete_type instanceof torch.OptionalType) {
                const op = concrete_type;
                if (value.type().kind() !== 'OptionalType' && !value.type().isSubtypeOf(torch.NoneType.get())) {
                    return torch._C.tryConvertToType(loc, graph, op.getElementType(), value, allow_conversions);
                }
            }
            if (value.node().kind() === 'prim::EmptyListLiteral' && concrete_type instanceof torch.ListType) {
                value = graph.insertNode(graph.createList(concrete_type.getElementType(), [])).output();
            }
            if (value.type() instanceof torch.TupleType) {
                const value_tuple = value.type();
                if (torch._C.convertibleToList(value.type(), torch._C.unwrapOptional(concrete_type))) {
                    const unpacked = torch._C.createTupleUnpack(value);
                    const elem_type = torch._C.unwrapOptional(concrete_type).expect(torch.ListType).getElementType();
                    value = graph.insertNode(graph.createList(elem_type, unpacked)).output();
                }
                if (concrete_type instanceof torch.TupleType) {
                    const concrete_tuple = concrete_type;
                    if (!value_tuple.isSubtypeOf(concrete_tuple) &&
                        concrete_tuple.elements().length === value_tuple.elements().length) {
                        const unpacked = torch._C.createTupleUnpack(value);
                        const converted = [];
                        for (let i = 0; i < concrete_tuple.elements().length; i++) {
                            converted.push(torch._C.tryConvertToType(loc, graph, concrete_tuple.elements()[i], unpacked[i], allow_conversions));
                        }
                        value = graph.insertNode(graph.createTuple(converted)).output();
                    }
                }
            }
            if (allow_conversions) {
                const value_isa_tensor = value.type().isSubtypeOf(torch.TensorType.get());
                const value_equals_number = value.type() === torch.NumberType.get();
                const concrete_float = concrete_type === torch.FloatType.get();
                const concrete_complex = concrete_type === torch.ComplexType.get();
                const concrete_int = concrete_type === torch.IntType.get();
                const concrete_number = concrete_type === torch.NumberType.get();
                if (value_isa_tensor) {
                    if (concrete_float) {
                        value = graph.insert('aten::FloatImplicit', [value], {}, loc);
                    } else if (concrete_complex) {
                        value = graph.insert('aten::ComplexImplicit', [value], {}, loc);
                    } else if (concrete_int) {
                        value = graph.insert('aten::IntImplicit', [value], {}, loc);
                    } else if (concrete_number) {
                        value = graph.insert('aten::ScalarImplicit', [value], {}, loc);
                    }
                } else if (value_equals_number) {
                    if (concrete_float) {
                        value = graph.insert('aten::Float', [value], {}, loc);
                    } else if (concrete_complex) {
                        value = graph.insert('aten::Complex', [value], {}, loc);
                    } else if (concrete_int) {
                        value = graph.insert('aten::Int', [value], {}, loc);
                    }
                } else if (value.type() === torch.BoolType.get()) {
                    if (concrete_float) {
                        value = graph.insert('aten::Float', [value], {}, loc);
                    } else if (concrete_int || concrete_number) {
                        value = graph.insert('aten::Int', [value], {}, loc);
                    }
                }
                if (value.type().isSubtypeOf(torch.StringType.get()) && concrete_type.isSubtypeOf(torch.DeviceObjType.get())) {
                    return graph.insert('aten::device', [value], {}, loc);
                }
            }
            return value;
        });
        this.registerFunction('torch._C.tryEvalTypeVariables', (type, type_env) => {
            if (!type.hasFreeVariables()) {
                if (type instanceof torch._C.DynamicType) {
                    return torch._C.tryEvalTypeVariables(type.fallback(), type_env);
                }
                return type;
            }
            if (type instanceof torch.Type && type.kind() === 'VarType') {
                return type_env.get(type.annotation_str);
            }
            const contained = type.containedTypes();
            if (contained.length === 0) {
                return type;
            }
            const new_contained = [];
            for (const t of contained) {
                const r = torch._C.tryEvalTypeVariables(t, type_env);
                if (!r) {
                    return null;
                }
                new_contained.push(r);
            }
            return type.withContained(new_contained);
        });
        this.registerFunction('torch._C.tryMatchSchema', (schema, loc, graph, args, kwargs, self, failure_messages, allow_conversions) => {
            if (torch._C.isBlockListedSchema(schema)) {
                return null;
            }
            const err = null;
            const type_env = new Map();
            const positional_inputs = [];
            const used_kwarg = kwargs.map(() => false);
            const is_aten = schema.name.startsWith('aten::');
            let used_args = 0;
            for (let schema_i = 0; schema_i < schema.arguments.length; schema_i++) {
                const arg = schema.arguments[schema_i];
                let actual_named_value = null;
                if (arg.name === 'self' && self) {
                    actual_named_value = self;
                    self = null;
                } else if (!arg.kwarg_only && used_args < args.length) {
                    if (allow_conversions && torch._C.varargsCanBeUsedAsList(schema, schema_i, arg)) {
                        const value = args[used_args].value(graph);
                        const actual_type = value.type;
                        if (actual_type.kind !== 'ListType' && !torch._C.convertibleToList(actual_type, torch._C.unwrapOptional(arg.type))) {
                            const formal_type = torch._C.unwrapOptional(arg.type).expect(torch.ListType).getElementType();
                            const list = torch._C.tryCreateList(formal_type, graph, loc, args.slice(used_args), failure_messages, err, allow_conversions, type_env);
                            if (!list) {
                                return null;
                            }
                            used_args = args.length;
                            positional_inputs.push(list);
                            continue;
                        }
                    }
                    actual_named_value = args[used_args];
                    used_args++;
                } else {
                    const kwarg_idx = torch._C.findInputWithName(arg.name, kwargs, is_aten);
                    if (Number.isInteger(kwarg_idx)) {
                        const nv = kwargs[kwarg_idx];
                        if (used_kwarg[kwarg_idx]) {
                            if (failure_messages) {
                                throw new python.Error(`Argument '${nv.name()}' specified twice in schema.`);
                            }
                            return null;
                        }
                        used_kwarg[kwarg_idx] = true;
                        actual_named_value = nv;
                    } else if (arg.has_default_value()) {
                        actual_named_value = new torch._C.NamedValue(arg.default_value);
                    } else {
                        if (failure_messages) {
                            throw new python.Error(`Argument '${arg.name}' not provided.`);
                        }
                        return null;
                    }
                }
                const positional = torch._C.tryMatchArgument(arg, graph, loc, actual_named_value, failure_messages, err, allow_conversions, type_env);
                if (!positional) {
                    return null;
                }
                positional_inputs.push(positional);
            }
            if (self !== null) {
                if (failure_messages) {
                    throw new python.Error('Provided self argument not used in schema.');
                }
                return null;
            }
            if (schema.is_vararg) {
                for (; used_args < args.length; used_args++) {
                    positional_inputs.push(args[used_args].value(graph));
                }
            }
            if (used_args < args.length) {
                if (failure_messages) {
                    throw new python.Error('Too many positional arguments.');
                }
                return null;
            }
            for (let i = 0; i < kwargs.length; i++) {
                const nv = kwargs[i];
                if (!used_kwarg[i]) {
                    if (failure_messages) {
                        if (schema.argumentIndexWithName(nv.name())) {
                            throw new python.Error('Keyword argument specified twice.');
                        } else {
                            throw new python.Error('Keyword argument unknown.');
                        }
                    }
                    return null;
                }
            }
            const returns = schema.returns;
            const return_types = returns.map((r) => {
                const result = torch._C.tryEvalTypeVariables(r.type, type_env);
                if (!result) {
                    throw new python.Error('Unbound type variable.');
                }
                return result;
            });
            const return_has_field_names = returns.every((r) => !r.name);
            let return_field_names = null;
            if (return_has_field_names) {
                return_field_names = returns.map((r) => r.name);
            }
            const schema_name = torch._C.getFullSchemaName(schema);
            return new torch._C.MatchedSchema(positional_inputs, return_types, return_field_names, schema_name);
        });
        this.registerFunction('torch._C.matchSchema', (schema, loc, graph, args, kwargs, self) => {
            const result = torch._C.tryMatchSchema(schema, loc, graph, args, kwargs, self, null, true);
            if (result) {
                return result;
            }
            throw new python.Error('No matching schema found.');
        });
        this.registerFunction('torch._C.matchSchemas', (schemas, loc, graph, args, kwargs, self, render_errors) => {
            self = self || null;
            render_errors = render_errors || false;
            if (schemas.length === 0) {
                throw python.Error('No schemas found.');
            }
            if (schemas.length === 1) {
                return [0, torch._C.matchSchema(schemas[0], loc, graph, args, kwargs, self)];
            }
            for (const allow_conversions of [false, true]) {
                for (let i = 0; i < schemas.length; i++) {
                    const matched_schema = torch._C.tryMatchSchema(schemas[i], loc, graph, args, kwargs, self, null, allow_conversions);
                    if (matched_schema) {
                        return [i, matched_schema];
                    }
                }
            }
            if (!render_errors) {
                return torch._C.matchSchemas(schemas, loc, graph, args, kwargs, self, /*render_errors=*/true);
            }
            throw new python.Error('No matching schema found.');
        });
        this.registerFunction('torch._C.emitBuiltinCall', (loc, graph, name, args, kwargs, self) => {
            const variants = torch._C.getAllOperatorsFor(name);
            const builtin_functions = torch._C.getAllBuiltinFunctionsFor(name);
            const graph_version = graph.get_op_version();
            const schemas = [];
            const upgrader_schemas = [];
            for (const op of variants) {
                let found_upgrader = false;
                const op_name = torch._C.getFullSchemaName(op.schema());
                if (Number.isInteger(graph_version)) {
                    const version_entry = torch._C.get_operator_version_map().get(op_name);
                    if (version_entry) {
                        const old_schema_entry = torch._C.findUpgrader(version_entry.second, graph_version.value());
                        if (old_schema_entry.has_value()) {
                            const old_schema = torch._C.parseSchema(old_schema_entry.value().old_schema);
                            upgrader_schemas.push(old_schema);
                            found_upgrader = true;
                        } else if (!torch._C.isOpCurrentBasedOnUpgraderEntries(version_entry.second, graph_version.value())) {
                            throw new python.Error('Valid upgrader must be present.');
                        }
                    }
                }
                if (!found_upgrader) {
                    schemas.push(op.schema());
                }
            }
            if (variants.length === 0) {
                const oldSchemas = torch._C.loadPossibleHistoricOps(name.toQualString(), graph_version);
                upgrader_schemas.reserve(oldSchemas.size());
                for (const old_schema_entry of oldSchemas) {
                    const old_schema = torch._C.parseSchema(old_schema_entry);
                    upgrader_schemas.push(old_schema);
                }
            }
            for (const schema of upgrader_schemas) {
                schemas.push(schema);
            }
            for (const method of builtin_functions) {
                method.ensure_defined();
                schemas.push(method.getSchema());
            }
            if (schemas.length === 0) {
                const user_function_name = name.toQualString();
                throw new python.Error(`Unknown built-in function '${user_function_name}'.`);
            }
            const matched = torch._C.matchSchemas(schemas, loc, graph, args, kwargs, self);
            if (matched[0] < variants.length + upgrader_schemas.length) {
                return torch._C.emitBuiltinNode(matched[1], loc, graph, name, graph_version);
            }
            const fn = builtin_functions[matched.first - variants.size()];
            return torch._C.insertGraph(graph, torch._C.toGraphFunction(fn).graph(), matched.second.inputs, new Map())[0];
        });
        this.registerFunction('torch._C.emitBuiltinNode', (matched_schema, loc, graph, name, version) => {
            const n = graph.insertNode(graph.create(name, matched_schema.inputs, 0)).setSourceRange(loc);
            for (const ret of matched_schema.return_types) {
                n.addOutput().setType(ret);
            }
            if (!Number.isInteger(version) || torch._C.isOpSymbolCurrent(matched_schema.schema_name, version)) {
                n.getOperation();
            } else {
                n.setHistoricSchemaName(matched_schema.schema_name);
            }
            return torch._C.packOutputs(graph, n.outputs(), matched_schema.return_field_names);
        });
        this.registerFunction('torch._C.unshapedType', (type) => {
            if (type.isSubtypeOf(torch.TensorType.get())) {
                return torch.TensorType.get();
            }
            const contained = type.containedTypes();
            if (contained.length === 0) {
                return type;
            }
            return type.withContained(contained.map((t) => torch._C.unshapedType(t)));
        });
        this.registerFunction('torch._C.unifyTypesImpl', (t1, t2, default_to_union, type_hint) => {
            default_to_union = default_to_union || false;
            type_hint = type_hint || null;
            if (t1.isSubtypeOf(t2)) {
                return t2;
            } else if (t2.isSubtypeOf(t1)) {
                return t1;
            }
            if (t1.kind() === 'TensorType' && t2.kind() === 'TensorType') {
                return t1.merge(t2);
            }
            if (t1.isSubtypeOf(torch.NoneType.get()) && !t2.isSubtypeOf(torch.NoneType.get())) {
                return torch.OptionalType.create(t2);
            } else if (t2.isSubtypeOf(torch.NoneType.get()) && !t1.isSubtypeOf(torch.NoneType.get())) {
                return torch.OptionalType.create(t1);
            }
            if (t1 instanceof torch.OptionalType) {
                const elem = torch._C.unifyTypes(t1.getElementType(), t2);
                if (elem) {
                    return torch.OptionalType.create(elem);
                }
            } else if (t2 instanceof torch.OptionalType) {
                const elem = torch._C.unifyTypes(t2.getElementType(), t1);
                if (elem) {
                    return torch.OptionalType.create(elem);
                }
            }
            if (t1 instanceof torch.TupleType && t2 instanceof torch.TupleType) {
                if (t1.elements().size() !== t2.elements().size()) {
                    return null;
                }
                const elements = [];
                for (let i = 0; i < t1.elements().length; i++) {
                    const elem = torch._C.unifyTypes(t1.elements()[i], t2.elements()[i], default_to_union);
                    if (elem) {
                        elements.push(elem);
                    } else {
                        return null;
                    }
                }
                return torch.TupleType.create(elements);
            }
            if (t1 instanceof torch.FutureType && t2 instanceof torch.FutureType) {
                const elem = torch._C.unifyTypes(t1.getElementType(), t2.getElementType());
                if (elem) {
                    return torch.FutureType.create(elem);
                }
            }
            const t1_unshaped = torch._C.unshapedType(t1);
            const t2_unshaped = torch._C.unshapedType(t2);
            if (t1_unshaped.isSubtypeOf(t2_unshaped)) {
                return t2_unshaped;
            } else if (t2_unshaped.isSubtypeOf(t1_unshaped)) {
                return t1_unshaped;
            }
            if (type_hint && t1.isSubtypeOf(type_hint) && t2.isSubtypeOf(type_hint)) {
                return type_hint;
            }
            return null;
        });
        this.registerFunction('torch._C.unifyTypes', (t1, t2, default_to_union, type_hint) => {
            const unified = torch._C.unifyTypesImpl(t1, t2, default_to_union, type_hint);
            if (default_to_union && !unified) {
                return torch.UnionType.create([t1, t2]);
            }
            return unified;
        });
        this.registerFunction('torch._C.unifyTypeList', (elements, why_not, default_to_union, type_hint) => {
            if (elements.length === 0) {
                return null;
            }
            let [ret_type] = elements;
            for (let i = 1; i < elements.length && ret_type; i++) {
                const maybe_unified = torch._C.unifyTypes(ret_type, elements[i], default_to_union, type_hint);
                if (!maybe_unified) {
                    return null;
                }
                ret_type = maybe_unified;
            }
            return ret_type;
        });

        this.registerFunction('torch._C.insertableTensor', (ten) => {
            return !ten.requires_grad() && ten.has_storage() && !ten.is_nested();
        });
        this.registerFunction('torch._C.insertConstant', (g, val, loc, scope) => {
            const value = torch._C.tryInsertConstant(g, val, loc, scope);
            if (value !== undefined) {
                return value;
            }
            throw new python.Error('Unsupported value kind.');
        });
        this.registerFunction('torch._C.tryInsertConstant', (g, val, loc, scope) => {
            const ivalue = false;
            if (ivalue) {
                val = new torch._C.IValue(val); // remove
                const n = g.create('prim::Constant');
                if (val.isTensor()) {
                    const ref = val.toTensor();
                    if (!torch._C.insertableTensor(val.toTensor())) {
                        n.destroy();
                        return null;
                    }
                    if (!ref.defined()) {
                        n.destroy();
                        return g.insertNode(g.createNone()).output();
                    }
                    // TORCH_INTERNAL_ASSERT(!ref.requires_grad());
                    n.output().inferTypeFrom(ref); // note: before t_ because of std::move(ref)
                    n.t_('value', ref);
                } else if (val.isInt()) {
                    n.i_('value', val.toInt());
                    n.output().setType(torch.IntType.get());
                } else if (val.isDouble()) {
                    n.f_('value', val.toDouble());
                    n.output().setType(torch.FloatType.get());
                } else if (val.isComplexDouble()) {
                    n.c_('value', val.toComplexDouble());
                    n.output().setType(torch.ComplexType.get());
                } else if (val.isBool()) {
                    n.i_('value', val.toBool());
                    n.output().setType(torch.BoolType.get());
                } else if (val.isList()) {
                    const fast_path_list = val.isBoolList() || val.isIntList() || val.isDoubleList();
                    if (fast_path_list || torch._C.insertableIValue(val)) {
                        n.ival_('value', val);
                        n.output().setType(val.type());
                    } else {
                        n.destroy();
                        return null;
                    }
                } else if (val.isString()) {
                    n.s_('value', val.toStringRef());
                    n.output().setType(torch.StringType.get());
                } else if (val.isDevice()) {
                    n.s_('value', val.toDevice().str());
                    n.output().setType(torch.DeviceObjType.get());
                } else if (val.isGenerator()) {
                    n.ival_('value', val.toGenerator());
                    n.output().setType(torch.GeneratorType.get());
                } else if (val.isStream()) {
                    n.ival_('value', val);
                    n.output().setType(torch.StreamObjType.get());
                } else if (val.isNone()) {
                    n.output().setType(torch.NoneType.get());
                } else if (val.isTuple()) {
                    if (torch._C.insertableIValue(val)) {
                        n.ival_('value', val);
                        n.output().setType(val.type());
                    } else {
                        n.destroy();
                        return null;
                    }
                } else if (val.isObject()) {
                    const ref = val.toObjectRef();
                    // see: [Constant Object Weak CompilationUnit Reference]
                    if (!ref.type().is_module() && (ref.is_weak_compilation_ref() || ref.is_empty_strong_compilation_ref())) {
                        n.ival_('value', val);
                        n.output().setType(val.type());
                    } else {
                        n.destroy();
                        return null;
                    }
                } else if ((val.isGenericDict() && torch._C.insertableIValue(val)) || (val.isEnum())) {
                    n.ival_('value', val);
                    n.output().setType(val.type());
                } else {
                    n.destroy();
                    return null;
                }
                if (loc) {
                    n.setSourceRange(loc);
                }
                if (scope) {
                    n.setScope(scope);
                }
                return g.insertNode(n).output();
            }
            const n = g.create('prim::Constant');
            let type = null;
            if (val === null) {
                n.ival_('value', val);
                type = torch.NoneType.get();
            } else if (typeof val === 'string') {
                n.s_('value', val);
                type = torch.StringType.get();
            } else if (Array.isArray(val) && val.every((item) => typeof item === 'string')) {
                n.ss_('value', val);
                type = torch.ListType.create(torch.StringType.get());
            } else if (typeof val === 'boolean') {
                n.i_('value', val === true ? 1 : 0);
                type = torch.BoolType.get();
            } else if (Number.isInteger(val)) {
                n.i_('value', val);
                type = torch.IntType.get();
            } else if (typeof val === 'number') {
                n.f_('value', val);
                type = torch.FloatType.get();
            } else if (val instanceof torch.Tensor) {
                n.t_('value', val);
                type = torch.TensorType.get();
            } else if (val instanceof torch.ScriptObject) {
                n.ival_('value', val);
                type = val.type();
            } else {
                throw new python.Error(`Unsupported value type '${typeof value}'.`);
            }
            if (type) {
                n.output().setType(type);
            }
            if (loc) {
                n.setSourceRange(loc);
            }
            if (scope) {
                n.setScope(scope);
            }
            return g.insertNode(n).output();
        });
        this.registerFunction('torch._C.toIValue', (v) => {
            if (v.node().kind() !== 'prim::Constant' || v.type().kind() === 'FunctionType') {
                return null;
            }
            const node = v.node();
            const type = v.type();
            if (type.isSubtypeOf(torch.TensorType.get())) {
                return node.t('value');
            } else if (type.isSubtypeOf(torch.BoolType.get())) {
                return Boolean(node.i('value'));
            } else if (type.isSubtypeOf(torch.NumberType.get()) && node.kindOf('value') === 'i') {
                return node.i('value');
            } else if (type.isSubtypeOf(torch.NumberType.get()) && node.kindOf('value') === 'f') {
                return node.f('value');
            } else if (type.isSubtypeOf(torch.NumberType.get()) && node.kindOf('value') === 'c') {
                return node.c('value');
            } else if (type instanceof torch.ListType && node.kindOf('value') === 'ival') {
                const list = node.ival('value');
                // TORCH_INTERNAL_ASSERT(list.isList());
                return list;
            } else if (type instanceof torch.DictType && node.kindOf('value') === 'ival') {
                const dict = node.ival('value');
                // TORCH_INTERNAL_ASSERT(dict.isGenericDict());
                return dict;
            } else if (type instanceof torch.TupleType && node.kindOf('value') === 'ival') {
                const tup = node.ival('value');
                // TORCH_INTERNAL_ASSERT(tup.isTuple());
                return tup;
            } else if (type === torch.StringType.get()) {
                const s = node.s('value');
                return s;
            } else if (type === torch.DeviceObjType.get()) {
                throw new python.Error('Not implemented.');
                // const d = c10::Device(node.s('value'));
                // return d;
            } else if (type === torch.GeneratorType.get()) {
                throw new python.Error('Not implemented.');
                // const generator = node.ival('value').toGenerator();
                // return generator;
            } else if (type === torch.StreamObjType.get()) {
                throw new python.Error('Not implemented.');
                // const s = node.ival('value').toStream();
                // return s;
            } else if (node.mustBeNone()) {
                throw new python.Error('Not implemented.');
                // return IValue();
            } else if (type.kind() === 'EnumType') {
                const enum_val = node.ival('value');
                return enum_val;
            } else if (type instanceof torch.ClassType && !type.is_module()) {
                const class_val = node.ival('value');
                return class_val;
            }
            throw new python.Error('Unsupported constant literal.');
        });
        this.registerType('torch._C.NamedValue', class {
            constructor(...args) {
                if (args.length === 1) {
                    if (args[0] instanceof torch.Value) {
                        [this._value] = args;
                    } else {
                        [this._ivalue] = args;
                    }
                } else if (args.length === 3 && typeof args[1] === 'string' && args[2] instanceof torch.Value) {
                    [this._loc, this._name, this._value] = args;
                } else {
                    throw new python.Error('Invalid argument.');
                }
            }
            name() {
                return this._name;
            }
            value(g) {
                if (!this._value) {
                    return torch._C.insertConstant(g, this._ivalue);
                }
                return this._value;
            }
        });
        this.registerType('torch._C.SugaredValue', class {
        });
        this.registerType('torch._C.SimpleValue', class extends torch._C.SugaredValue {
            constructor(value) {
                super();
                this._value = value;
            }
            asValue(/* range, m */) {
                return this._value;
            }
            getValue() {
                return this._value;
            }
            asTuple(loc, m, size_hint) {
                const make_simple_value = (v) => new torch._C.SimpleValue(v);
                if (this._value.type() instanceof torch.TupleType) {
                    const outputs = torch._C.createTupleUnpack(this._value);
                    return outputs.map((v) => make_simple_value(v));
                } else if (this._value.type() instanceof torch.ListType) {
                    if (!size_hint) {
                        throw new python.Error('Cannot statically infer the expected size of a list in this context.');
                    }
                    const graph = this._value.owningGraph();
                    const unpack = graph.insertNode(graph.createListUnpack(this._value, size_hint));
                    return unpack.outputs().map((v) => make_simple_value(v));
                } else if (this._value.type().kind() === 'AnyTupleType') {
                    throw new python.Error('Provided tuple is not fully defined including its element types.');
                }
                throw new python.Error(`Cannot use '${this._value.type().toString()}' as tuple.`);
            }
            attr(loc, m, field) {
                if (this._value.type().isSubtypeOf(torch.TensorType.get())) {
                    if (torch._C.builtin_cast_method_to_scalar_type().has(field)) {
                        return new torch._C.TensorCastValue(torch._C.builtin_cast_method_to_scalar_type().get(field), new torch._C.NamedValue(loc, 'self', this._value));
                    }
                }
                if (this._value.type() instanceof torch.TupleType) {
                    throw new python.Error('Not implemented.');
                }
                if (this._value.type() instanceof torch.AwaitType) {
                    throw new python.Error('Not implemented.');
                }
                if (this._value.type() instanceof torch.ClassType) {
                    const classType = this._value.type();
                    if (classType.findMethod(field)) {
                        return new torch._C.MethodValue(this.getValue(), [field]);
                    }
                    if (classType.hasAttribute(field)) {
                        const g = m.graph();
                        const n = g.insertNode(g.createGetAttr(this._value, field));
                        return new torch._C.SimpleValue(n.output());
                    }
                    const prop = classType.getProperty(field);
                    if (prop) {
                        return new torch._C.MethodValue(this._value, [prop.getter.name()]).call(loc, m, {}, {}, /*n_binders=*/1);
                    }
                }
                if (this._value.type() instanceof torch.InterfaceType) {
                    throw new python.Error('Not implemented.');
                }
                throw new python.Error('Not implemented.');
            }
        });
        this.registerType('torch._C.MethodValue', class extends torch._C.SugaredValue {
            constructor(self, method_names) {
                super();
                this._self = self;
                this._method_names = method_names;
            }
            call(loc, f, args, kwargs /*, n_binders */) {
                const argsWithSelf = [new torch._C.NamedValue(this._self), ...args];
                const schemas = [];
                for (const method_name of this._method_names) {
                    const type = this._self.type();
                    if (type instanceof torch.ClassType) {
                        const class_type = type;
                        const method = class_type.getMethod(method_name);
                        method.ensure_defined();
                        schemas.push(method.getSchema());
                    } else if (type instanceof torch.InterfaceType) {
                        const interface_type = type;
                        schemas.push(interface_type.getMethod(method_name));
                    } else {
                        throw new python.Error('Method constructed that is not a class or interface.');
                    }
                }
                const match = torch._C.matchSchemas(schemas, loc, f.graph(), argsWithSelf, kwargs);
                const output = f.graph().insertMethodCall(this._method_names[match[0]], match[1]);
                output.node().setSourceRange(loc);
                return new torch._C.SimpleValue(output);
            }
        });
        this.registerType('torch._C.SpecialFormValue', class extends torch._C.SugaredValue {
            constructor(form) {
                super();
                this._form = form;
            }
            form() {
                return this._form;
            }
            static create(form) {
                return new torch._C.SpecialFormValue(form);
            }
        });
        this.registerType('torch._C.BuiltinFunction', class extends torch._C.SugaredValue {
            constructor(symbol, self) {
                super();
                this.symbol = symbol;
                this.self = self;
            }
            call(loc, m, args, kwargs /*, n_binders */) {
                return new torch._C.SimpleValue(torch._C.emitBuiltinCall(loc, m.graph(), this.symbol, args, kwargs, this.self));
            }
        });
        this.registerType('torch._C.BuiltinModule', class extends torch._C.SugaredValue {
            constructor(name, version) {
                super();
                this.name = name;
                this.version = version || null;
            }
            attr(loc, m, field) {
                if (field === 'autograd') {
                    return new torch._C.BuiltinModule('aten', this.version);
                }
                const sym = `${this.name}::${field}`;
                return new torch._C.BuiltinFunction(sym, null);
            }
        });
        this.registerType('torch._C.OpsValue', class extends torch._C.SugaredValue {
            constructor(version) {
                super();
                this._version = version;
            }
            attr(loc, m, field) {
                return new torch._C.BuiltinModule(field, this._version);
            }
        });
        this.registerType('torch._C.ConstantTableValue', class extends torch._C.SugaredValue {
            constructor(constants) {
                super();
                this._constants = constants;
                this.non_holding_object_cache = new Map();
            }
            attr(loc, m, field) {
                const offset = parseInt(field.substring(1), 10);
                if (!Number.isInteger(offset)) {
                    throw new python.Error(`Invalid constant identifier '${field}.`);
                }
                if (offset < 0 || offset >= this._constants.length) {
                    throw new python.Error('Invalid constant index.');
                }
                const ivalue = new torch._C.IValue(this._constants[offset]); // remove IValue
                let value = null;
                if (ivalue.isObject() && !ivalue.toObject().is_weak_compilation_ref()) {
                    const obj = ivalue.toObject();
                    if (!this.non_holding_object_cache.has(obj)) {
                        this.non_holding_object_cache.set(obj, obj.copy_to_weak_compilation_ref());
                    }
                    value = m.graph().insertConstant(this.non_holding_object_cache[obj], loc);
                } else {
                    value = m.graph().insertConstant(this._constants[offset], loc);
                }
                value.setType(torch._C.unshapedType(value.type()));
                return new torch._C.SimpleValue(value);
            }
        });
        this.registerType('torch._C.CastValue', class extends torch._C.BuiltinFunction {
            constructor(type, method) {
                super(method, null);
                this._type = type;
            }
            call(loc, m, args, kwargs, n_binders) {
                if (args.length === 1 && kwargs.length === 0) {
                    const len_op = new torch._C.BuiltinFunction('aten::len', null);
                    const gt_op = new torch._C.BuiltinFunction('aten::gt', null);
                    const zero = m.graph().insertConstant(0);
                    const v = args[0].value(m.graph());
                    if (v.type().isSubtypeOf(this._type)) {
                        return new torch._C.SimpleValue(v);
                    } else if (this._type === torch.BoolType.get() && (v.type().isSubtypeOf(torch.AnyListType.get()) || v.type().isSubtypeOf(torch.StringType.get()) || v.type() instanceof torch.DictType)) {
                        const len = len_op.call(loc, m, [v], [], 1);
                        return gt_op.call(loc, m, [len.asValue(loc, m), zero], [], 1);
                    }
                }
                return super.call(loc, m, args, kwargs, n_binders);
            }
        });
        this.registerType('torch._C.MagicMethod', class extends torch._C.SugaredValue {
            constructor(desugared_name, base) {
                super();
                this._base_value = base;
                this._desugared_name = desugared_name;
            }
            call(loc, m, args, kwargs, n_binders) {
                if (args.length > 0) {
                    const self = args[0].value(m.graph());
                    if (self.type() instanceof torch.ClassType) {
                        return new torch._C.SimpleValue(self)
                            .attr(loc, m, this._desugared_name)
                            .call(loc, m, args.slice(1), kwargs, n_binders);
                    }
                }
                if (!this._base_value) {
                    throw new python.Error('Invalid magic method.');
                }
                return this._base_value.call(loc, m, args, kwargs, n_binders);
            }
        });
        this.registerType('torch._C.RangeValue', class extends torch._C.SugaredValue {
            constructor(loc, m, inputs, static_len) {
                super();
                static_len = static_len || null;
                if (inputs.length === 0 || inputs.length > 3 || !inputs.every((value) => value.type() instanceof torch.IntType)) {
                    throw new python.Error('Invalid range inputs.');
                }
                const g = m.graph();
                if (inputs.length === 1) {
                    [this._end] = inputs;
                    this._start = g.insertConstant(0, loc);
                    this._step = g.insertConstant(1, loc);
                    this._has_only_end = true;
                } else {
                    [this._start, this._end] = inputs;
                    this._step = inputs.length === 3 ? inputs[2] : g.insertConstant(1, loc);
                    this._has_only_end = false;
                }
                this._static_len = static_len;
            }
        });
        this.registerType('torch._C.ClassNamespaceValue', class extends torch._C.SugaredValue {
            constructor(name, si) {
                super();
                this._basename = name;
                this._si = si;
            }
            attr(loc, m, name) {
                const fullName = new torch._C.QualifiedName(this._basename, name);
                const serializable_type = this._si.findNamedType(fullName);
                if (serializable_type) {
                    if (serializable_type instanceof torch.ClassType) {
                        return new torch._C.ClassValue(serializable_type);
                    } else if (serializable_type instanceof torch.TupleType) {
                        return new torch._C.NamedTupleConstructor(serializable_type);
                    } else if (serializable_type instanceof torch.EnumType) {
                        return new torch._C.SugaredEnumClass(serializable_type);
                    }
                }
                const fn = this._si.findFunction(fullName);
                if (fn) {
                    return new torch._C.FunctionValue(fn);
                }
                return new torch._C.ClassNamespaceValue(fullName, this._si);
            }
        });
        this.registerType('torch.package.PackageImporter', class {
            constructor(reader) {
                this.zip_reader = reader;
            }
            load_pickle(module, resource) {
                const name = `${module.replace(/\./, '/')}/${resource}`;
                const stream = this.zip_reader.get_record(name);
                const loaded_reduces = new Map();
                this.storage_context = new torch._C.DeserializationStorageContext();
                const unpickler = new pickle.Unpickler(stream);
                unpickler.persistent_load = (saved_id) => {
                    switch (saved_id[0]) {
                        case 'storage': {
                            const [, storage_type, key, , size] = saved_id;
                            if (!this.storage_context.has_storage(key)) {
                                const storage = new storage_type(size);
                                const stream = this.zip_reader.get_record(`.data/${key}.storage`);
                                const buffer = stream.peek();
                                storage._set_cdata(buffer);
                                this.storage_context.add_storage(key, storage);
                            }
                            return this.storage_context.get_storage(key);
                        }
                        case 'reduce_package': {
                            if (saved_id.length === 2) {
                                const [, func, args] = saved_id;
                                return execution.invoke(func, args);
                            }
                            const [, reduce_id, func, args] = saved_id;
                            if (!loaded_reduces.has(reduce_id)) {
                                const value = execution.invoke(func, [this].concat(args));
                                loaded_reduces.set(reduce_id, value);
                            }
                            return loaded_reduces.get(reduce_id);
                        }
                        default: {
                            throw new python.Error(`Unknown package typename '${saved_id[0]}'.`);
                        }
                    }
                };
                const obj = unpickler.load();
                this.storage_context = null;
                return obj;
            }
            import_module(name) {
                return execution.import(name);
            }
        });
        this.registerFunction('torch.jit.load', (file, map_location, extra_files) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C.import_ir_module(cu, file, map_location, extra_files);
            const module = torch.jit._script.wrap_cpp_module(cpp_module);
            module.forward = cpp_module.forward; // remove
            return module;
        });
        this.registerFunction('torch._C.import_ir_module', function(cu, reader, ...args) {
            switch (arguments.length) {
                case 4: {
                    const [device, extra_files] = args;
                    const deserializer = new torch._C.ScriptModuleDeserializer(cu, reader);
                    return deserializer.deserialize(device, extra_files);
                }
                case 5: {
                    const [storage_context, device, ts_id] = args;
                    const deserializer = new torch._C.ScriptModuleDeserializer(cu, reader, `.data/ts_code/${ts_id}/`, '.data/', storage_context);
                    return deserializer.deserialize(device, null);
                }
                default: {
                    throw new python.Error("Invalid 'torch._C.import_ir_module' signature.");
                }
            }

        });
        this.registerFunction('torch._C._import_ir_module_from_package', (cu, reader, storage_context, map_location, ts_id) => {
            return torch._C.import_ir_module(cu, reader, storage_context, null, ts_id);
        });
        this.registerFunction('torch._C.tryToGraphFunction', (value) => {
            if (value instanceof torch.Node) {
                const node = value;
                if (node.kind() === 'prim::CallFunction') {
                    throw new python.Error('Not implemented.');
                }
                if (node.kind() === 'prim::CallMethod') {
                    const name = node.s('name');
                    const class_type = node.input(0).type();
                    if (class_type) {
                        const fn = class_type.getMethod(name);
                        return torch._C.tryToGraphFunction(fn);
                    }
                }
                return null;
            } else if (value instanceof torch.jit.Function) {
                const fn = value;
                if (!fn.isGraphFunction()) {
                    return null;
                }
                return fn;
            }
            throw new python.Error('Not implemented.');
        });
        this.registerType('torch._C.ModuleInstanceInfo', class {
            constructor(module_type, instance_name) {
                this._module_type = module_type;
                this._instance_name = instance_name;
            }
        });
        this.registerFunction('torch._C.createTupleUnpack', (v) => {
            if (v.node().kind() === 'prim::TupleConstruct') {
                return v.node().inputs();
            }
            const g = v.owningGraph();
            return g.insertNode(g.createTupleUnpack(v)).outputs();
        });
        this.registerFunction('torch._C.inlineCallStackOfNode', (/* new_node, new_cs_entriesm, callee, to_replace, m_info */) => {
            /*
            const new_node_cs = new_node.callstack();
            const raw_callstack_ptr = new_node_cs ? new_node_cs : nullptr;
            if (!new_cs_entries.has(raw_callstack_ptr)) {
                if (new_node_cs) {
                    new_cs_entries.set(raw_callstack_ptr, c10::make_intrusive<InlinedCallStack>(*new_node_cs, callee, to_replace.sourceRange(), m_info));
                } else {
                    new_cs_entries.set(raw_callstack_ptr, c10::make_intrusive<InlinedCallStack>(callee, to_replace.sourceRange(), m_info);
                }
            }
            new_node.setCallStack(new_cs_entries.at(raw_callstack_ptr));
            for (const block of new_node.blocks()) {
                torch._C.inlineCallStackOfBlock(block, new_cs_entries, callee, to_replace, m_info);
            }
            */
        });
        this.registerFunction('torch._C.inlineCallTo', (to_replace, callee, callee_graph) => {
            if (callee_graph === undefined || typeof callee_graph === 'boolean') {
                callee_graph = callee_graph === undefined ? true : callee_graph;
                callee_graph = callee_graph ? callee.optimized_graph() : callee.graph();
            }
            const guard = new torch._C.WithInsertPoint(to_replace);
            const value_map = new Map();
            const new_outputs = torch._C.insertGraph(to_replace.owningGraph(), callee_graph, to_replace.inputs(), value_map);
            const new_callstack_entries = new Map();
            let module_instance_info = null;
            if (to_replace.kind() === 'prim::CallMethod') {
                const class_type_ptr = to_replace.input(0).type();
                if (to_replace.input(0).node().kind() === 'prim::GetAttr') {
                    module_instance_info = new torch._C.ModuleInstanceInfo(class_type_ptr, to_replace.input(0).node().s('name'));
                } else if (!to_replace.owningGraph().inputs().empty() && to_replace.input(0) === to_replace.owningGraph().inputs()[0]) {
                    module_instance_info = new torch._C.ModuleInstanceInfo(class_type_ptr, 'SELF');
                } else {
                    module_instance_info = new torch._C.ModuleInstanceInfo(class_type_ptr, 'INSTANCE_NAME_UNKNOWN');
                }
            }
            const updated_nodes = new Set();
            for (const kv of value_map) {
                const is_graph_input = callee_graph.inputs().indexOf(kv[0]);
                if (is_graph_input === -1) {
                    continue;
                }
                const new_node = kv[1].node();
                if (updated_nodes.has(new_node)) {
                    continue;
                }
                updated_nodes.add(new_node);
                torch._C.inlineCallStackOfNode(new_node, new_callstack_entries, callee, to_replace, module_instance_info);
            }
            const old_outputs = to_replace.outputs();
            // AT_ASSERT(new_outputs.size() == old_outputs.size());
            for (let i = 0; i < old_outputs.length; i++) {
                if (old_outputs[i].hasDebugName()) {
                    new_outputs[i].setDebugName(old_outputs[i].debugName());
                }
                old_outputs[i].replaceAllUsesWith(new_outputs[i]);
            }
            to_replace.destroy();
            guard.dispose();
            return new_outputs;
        });
        this.registerFunction('torch._C.inlineCalls', (block) => {
            for (const cur of block.nodes()) {
                switch (cur.kind()) {
                    case 'prim::CallFunction': {
                        const graphFunction = torch._C.tryToGraphFunction(cur);
                        if (graphFunction) {
                            const function_constant = cur.input(0).node();
                            // const fun_type = function_constant.output().type().expect(torch.FunctionType);
                            cur.removeInput(0);
                            let g = null;
                            const fallback = function_constant.hasAttribute('fallback');
                            if (fallback && graphFunction.get_executor().isOptimized()) {
                                const exec_plans = graphFunction.get_executor().getDebugState().execution_plans;
                                if (!exec_plans.empty()) {
                                    g = exec_plans.begin().second.graph;
                                    torch._C.Inline(g);
                                }
                            }
                            if (g === null) {
                                g = graphFunction.optimized_graph();
                            }
                            torch._C.inlineCallTo(cur, graphFunction, g);
                        }
                        break;
                    }
                    case 'prim::CallMethod': {
                        const graphFunction = torch._C.tryToGraphFunction(cur);
                        torch._C.inlineCallTo(cur, graphFunction);
                        break;
                    }
                    default: {
                        for (const b of cur.blocks()) {
                            torch._C.inlineCalls(b);
                        }
                    }
                }
            }
        });
        this.registerFunction('torch._C.Inline', (graph) => {
            torch._C.inlineCalls(graph.block());
        });
        this.registerFunction('torch._C._jit_pass_inline', (graph) => {
            torch._C.Inline(graph);
        });
        this.registerFunction('torch.jit._script.unpackage_script_module', (importer, script_module_id) => {
            const cu = new torch.jit.CompilationUnit();
            cu.execution = execution;
            const cpp_module = torch._C._import_ir_module_from_package(cu, importer.zip_reader, importer.storage_context, importer.last_map_location, script_module_id);
            return torch.jit._script.wrap_cpp_module(cpp_module);
        });
        this.registerFunction('torch.jit._script.wrap_cpp_module', (cpp_module) => {
            const init_fn = (script_module) => {
                for (const [name, module] of new torch.ModuleDict(script_module._c).items()) {
                    script_module.__setattr__(name, torch.jit._script.wrap_cpp_module(module));
                }
            };
            return torch.jit._script.RecursiveScriptModule._construct(cpp_module, init_fn);
        });
        this.registerType('torch._C.DeserializationStorageContext', class extends Map {
            has_storage(name) {
                return this.has(name);
            }
            get_storage(name) {
                return this.get(name);
            }
            add_storage(name, storage) {
                return this.set(name, storage);
            }
        });
        this.registerType('torch.ScriptFunction', class {
            constructor(name, graph /*, function_creator */) {
                this._name = name;
                this._graph = graph;
            }
        });
        this.registerType('torch.ScriptMethod', class {
            constructor(owner, value) {
                this._owner = owner;
                this._function = value;
            }
            get name() {
                return this._function.name();
            }
            get owner() {
                return this._owner;
            }
            __call__(/* args, kwargs */) {
                throw new python.Error();
            }
            get graph() {
                return this._function.graph();
            }
            get schema() {
                // return this.function().getSchema();
                throw new python.Error();
            }
            get code() {
                throw new python.Error();
            }
            get code_with_constants() {
                throw new python.Error();
            }
        });
        this.registerType('torch.ScriptObject', class {
            constructor(type) {
                this._typ = type;
                this._ivalue = {};
            }
            static create(type) {
                if (type.is_module()) {
                    return new torch.ScriptModule(type);
                }
                return new torch.ScriptObject(type);
            }
            type() {
                return this._typ;
            }
            _type() {
                return this._typ; // torch.ClassType
            }
            _get_method(name) {
                for (const fn of this._type.methods()) {
                    if (name === fn.name) {
                        return new torch.ScriptMethod(this /* _value() */, fn);
                    }
                }
                return null;
            }
            _has_method(/* name */) {
                throw new python.Error();
            }
            __setattr__(name, value) {
                // if (this._type.hasContant(name))
                this._ivalue[name] = value;
            }
            __getattr__(name) {
                return this._ivalue[name];
            }
            hasattr(name) {
                return this._typ.hasAttribute(name) || this._typ.hasConstant(name);
            }
            getattr(name) {
                return this.__getattr__(name);
            }
            _properties() {
                throw new python.Error();
            }
            is_weak_compilation_ref() {
                return true; // not implemented
            }
        });
        this.registerType('torch.ScriptModule', class extends torch.ScriptObject {
            constructor(...args) {
                if (args[0] instanceof torch._C.QualifiedName && args[1] instanceof torch.jit.CompilationUnit) {
                    const [class_name, cu, shouldMangle] = args;
                    super(...torch.ScriptModule.create_module_object(class_name, cu, shouldMangle));
                } else {
                    super(...args);
                }
            }
            get qualified_name() {
                return this.type().qualified_name();
            }
            get code_with_constants() {
                const const_map = {};
                const_map.const_mapping = new Map(Object.entries(execution.builtins.CONSTANTS));
                return [null, const_map];
            }
            get graph() {
                if (!this._graph) {
                    if (execution.to_ir) {
                        const fn = this._typ.getMethod('forward');
                        this._graph = fn.graph();
                    } else {
                        const isTensor = (obj) => {
                            const name = obj && obj.__class__ ? obj.__class__.__module__ : null;
                            switch (name) {
                                case 'torch':
                                case 'torch.cuda':
                                    return obj.__class__.__name__.endsWith('Tensor');
                                case 'torch.nn.parameter':
                                    return obj.__class__.__name__ === 'Parameter';
                                default:
                                    return false;
                            }
                        };
                        if (!this.forward) {
                            return null;
                        }
                        const args = [];
                        if (this.forward.__code__ && this.forward.__code__.args) {
                            const params = this.forward.__code__.args.args;
                            for (let i = 0; i < params.length; i++) {
                                const arg = params[i];
                                const value = execution.graph.addInput(arg.arg);
                                if (i === 0 && arg.arg === 'self' && !arg.annotation) {
                                    value.setType(this.type());
                                } else {
                                    value.setType(execution.type(arg.annotation));
                                }
                                if (isTensor(value)) {
                                    value.__variable__ = arg.name;
                                    value.__origin__ = 'graph-input';
                                }
                                args.push(value);
                            }
                        }
                        execution.purge = new Set();
                        const result = this.forward.__call__(args);
                        const queue = Array.from(execution.purge);
                        const visited = new Set();
                        while (queue.length > 0) {
                            const node = queue.shift();
                            if (visited.has(node)) {
                                continue;
                            }
                            visited.add(node);
                            if (node.outputs().every((output) => output.uses().length === 0)) {
                                for (const input of node.inputs()) {
                                    queue.push(input.node());
                                }
                                node.destroy();
                            }
                        }
                        if (Array.isArray(result)) {
                            for (const output of result) {
                                if (isTensor(output)) {
                                    const value = execution.variable(output);
                                    execution.graph.return_node().addInput(value);
                                }
                            }
                        } else if (isTensor(result)) {
                            const value = execution.variable(result);
                            execution.graph.return_node().addInput(value);
                        } else if (result instanceof torch.Value) {
                            execution.graph.return_node().addInput(result);
                        } else if (Object(result) === result) {
                            for (const key of Object.keys(result)) {
                                const item = result[key];
                                if (Array.isArray(item)) {
                                    for (const output of item) {
                                        if (isTensor(output)) {
                                            const value = execution.variable(output);
                                            execution.graph.return_node().addInput(value);
                                        }
                                    }
                                } else if (isTensor(item)) {
                                    const value = execution.variable(item);
                                    execution.graph.return_node().addInput(value);
                                }
                            }
                        }
                        this._graph = execution.graph;
                    }
                }
                return this._graph;
            }
            static create_module_object(class_name, cu, shouldMangle) {
                shouldMangle = shouldMangle || false;
                if (!class_name.prefix()) {
                    class_name = new torch._C.QualifiedName('__torch__', class_name.name());
                }
                if (shouldMangle && cu.get_class(class_name)) {
                    class_name = cu.mangle(class_name);
                }
                const cls = torch.ClassType.create(class_name, cu, true);
                cu.register_type(cls);
                return [cls, cu];
            }
            register_module(name, module) {
                this.type().addOrCheckAttribute(name, module.type());
                this.__setattr__(name, module); // _ivalue()->setAttr(name, module._ivalue());
            }
            register_buffer(name, v) {
                this.type().addOrCheckAttribute(name, torch.TensorType.get(), false, true);
                this.__setattr__(name, v); // _ivalue()->setAttr(name, std::move(v));
            }
            register_parameter(name, v, is_buffer) {
                this.type().addOrCheckAttribute(name, torch.TensorType.get(), !is_buffer, is_buffer);
                this.__setattr__(name, v); // _ivalue()->setAttr(name, std::move(v));
            }
            register_attribute(name, t, v, is_param, is_buffer) {
                this.type().addOrCheckAttribute(name, t, is_param, is_buffer);
                // _ivalue()->setAttr(name, v);
            }
        });
        this.registerType('torch.ModuleDict', class {
            constructor(mod) {
                this._module = mod;
            }
            items() {
                const result = new Map();
                const type = this._module.type();
                for (let i = 0; i < type.numAttributes(); i++) {
                    const k = type.getAttributeName(i);
                    const t = type.getAttribute(i);
                    if (t && t.is_module()) {
                        result.set(k, this._module.__getattr__(k));
                    }
                }
                return result;
            }
        });
        this.registerType('torch.ParameterDict', class {
            constructor(mod) {
                this._module = mod;
            }
            items() {
                const result = new Map();
                const type = this._module.type();
                for (let i = 0; i < type.numAttributes(); i++) {
                    if (type.is_parameter(i)) {
                        const k = type.getAttributeName(i);
                        const v = this._module.__getattr__(k);
                        if (v instanceof torch.Tensor) {
                            result.set(k, v);
                        }
                    }
                }
                return result;
            }
        });
        this.registerType('torch.BufferDict', class {
            constructor(mod) {
                this._module = mod;
            }
            items() {
                const result = new Map();
                const type = this._module.type();
                for (let i = 0; i < type.numAttributes(); i++) {
                    if (type.is_buffer(i)) {
                        const t = type.getAttribute(i);
                        if (t.isSubtypeOf(torch.TensorType.get())) {
                            const k = type.getAttributeName(i);
                            const v = this._module.__getattr__(k);
                            result.set(k, v);
                        }
                    }
                }
                return result;
            }
        });
        this.registerType('torch.jit.to_ir', class {
            constructor(def, _resolver, self, method) {
                this.method = method;
                this.graph = method.graph();
                this.resolver = _resolver;
                this.integral_constants = new Map();
                this.fp_constants = new Map();
                this.exit_blocks = new Set();
                this._typeParser = new torch.jit.ScriptTypeParser(this.resolver);
                this.environment_stack = null;
                this._def_stack = [];
                this._temp_name_count = 0;
                this.pushFrame(this.graph.block(), true);
                if (self && def && def.args.args.length === 0) {
                    throw new python.Error('Method must have a self argument.');
                }
                method.setSchema(this.emitDef(def, self, this.graph.block()));
                // torch._C.ReplaceOldOperatorsWithUpgraders(this.graph);
                torch._C.ConvertToSSA(this.graph);
                // torch._C.CanonicalizeModifiedLoops(this.graph);
                torch._C.NormalizeOps(this.graph.block());
                torch._C.runCleanupPasses(this.graph);
            }
            pushFrame(b, starts_def) {
                starts_def = starts_def || false;
                if (starts_def) {
                    this._def_stack.push({});
                }
                this.environment_stack = new torch._C.Environment(this.method, this.resolver, b, this.environment_stack);
            }
            popFrame(ends_def) {
                const old_frame = this.environment_stack;
                this.environment_stack = this.environment_stack.next;
                if (ends_def) {
                    this._def_stack.pop();
                }
                return old_frame;
            }
            emitDef(def, self, block) {
                const schema = this._typeParser.parseSchemaFromDef(def, self !== null);
                if (schema.returns.length === 1) {
                    this._def_stack[this._def_stack.length - 1]._declared_return_type = schema.returns[0].type;
                }
                const args = this.emitFormalArguments(def, self, schema, block);
                if (execution.to_ir) {
                    this.emitStatements(def.body);
                    this.handleMaybeNoReturn(def, block);
                }
                const returns = [this.emitOutput(def, schema, block)];
                return new torch.FunctionSchema(def.name, '', args, returns);
            }
            emitFormalArguments(def, self, schema, block) {
                const args = [];
                const params = def.args.args;
                const expected_annotation_size = self ? def.args.args.length - 1 : def.args.args.length;
                if (schema.arguments.length !== expected_annotation_size) {
                    throw new python.Error('Invalid formal arguments.');
                }
                let it = 0;
                if (self) {
                    const name = params[it].arg;
                    const new_input = block.addInput().setDebugName(name);
                    this.environment_stack.setSugaredVar(it, name, self.makeSugared(new_input), null);
                    args.push(new torch.Argument(name, new_input.type()));
                    it++;
                }
                const shouldDeriveType = this.shouldDeriveSetStateType(def, schema);
                let arg_annotation_idx = 0;
                for (; it < params.length; it++) {
                    const name = params[it].arg;
                    const new_input = block.addInput();
                    if (torch._C.meaningfulName(name)) {
                        new_input.setDebugName(name);
                    }
                    let arg = schema.arguments[arg_annotation_idx++];
                    if (shouldDeriveType) {
                        if (schema.arguments.length === 1) {
                            throw new python.Error('Invalid schema.');
                        }
                        const inferredStateType = this.getTypeForSetStateArg(def, self);
                        arg = arg.cloneWithType(inferredStateType);
                    }
                    args.push(arg);
                    new_input.setType(arg.type);
                    this.environment_stack.setVar(params[it], name, new_input);
                }
                return args;
            }
            emitOutput(range, schema, block) {
                const ret_type = this._def_stack[this._def_stack.length - 1]._merged_return_type;
                const placeholder_return = this.graph.insertNode(this.graph.createUninitialized(ret_type)).output();
                block.registerOutput(placeholder_return);
                return new torch.Argument('', this._def_stack[this._def_stack.length - 1]._merged_return_type);
            }
            emitStatements(stmts) {
                for (let i = 0; i < stmts.length; i++) {
                    const stmt = stmts[i];
                    if (stmt instanceof ast.If) {
                        this.emitIf(stmt);
                    } else if (stmt instanceof ast.Assign) {
                        this.emitAssignment(stmt);
                    } else if (stmt instanceof ast.Return) {
                        this.emitReturn(stmt);
                    } else {
                        throw new python.Error(`Unrecognized statement kind '${stmt.__class__.__name__}'.`);
                    }
                }
            }
            emitIf(stmt) {
                const cond_value = this.emitCondExpr(stmt.test);
                this.emitIfElseBlocks(stmt, cond_value, stmt.body, stmt.orelse);
            }
            emitCondExpr(expr) {
                /*
                switch (expr.kind()) {
                    case TK_AND:
                    case TK_OR: {
                      auto binop = BinOp(expr);
                      return emitShortCircuitLogical(
                          binop.range(), binop.lhs(), binop.rhs(), expr.kind() == TK_OR);
                    }
                    case TK_NOT: {
                      CondValue v = emitCondExpr(Expr(expr.tree()->trees()[0]));
                      Value* result = emitBuiltinCall(
                          expr.range(), *graph, aten::__not__, {v.value()}, {});
                      std::optional<bool> static_if;
                      if (v.staticIf()) {
                        static_if = !*v.staticIf();
                      }
                      return CondValue(result, v.refinements().Not(), static_if);
                    } break;
                    case TK_IS:
                    case TK_ISNOT: {
                      // meta programming on AST for is/is not cases and emit branches base on
                      auto cond_op = BinOp(expr);
                      Value* lhs_val = emitExpr(cond_op.lhs());
                      Value* rhs_val = emitExpr(cond_op.rhs());
                      auto lhs_none = canBeNone(lhs_val);
                      auto rhs_none = canBeNone(rhs_val);
                      // Dispatch logic (A: ALWAYS, N: NEVER, M: MAYBE):
                      // AA, -> statically IS always holds, IS_NOT never holds
                      // AN , NA-> statically IS_NOT always holds, IS never holds
                      // MA, MM, MN, NM, NN, AM -> cannot prove anything statically
                      bool its_is = expr.kind() == TK_IS;
                      if (lhs_none == ALWAYS && rhs_none == ALWAYS) {
                        return CondValue(*graph, expr.range(), its_is, {});
                      } else if (
                          (lhs_none == ALWAYS && rhs_none == NEVER) ||
                          (lhs_none == NEVER && rhs_none == ALWAYS)) {
                        // lhs_val/rhs_val with A/M: only emit never_none_branch
                        return CondValue(*graph, expr.range(), !its_is, {});
                      } else {
                        auto kind = getNodeKind(expr.kind(), expr.get()->trees().size());
                        Value* cond_value = emitBuiltinCall(
                            expr.get()->range(),
                            *method.graph(),
                            kind,
                            {lhs_val, rhs_val},
                            {});
                        auto refinements = RefinementSet(findIsNoneRefinements(
                            cond_op.lhs(), lhs_val, cond_op.rhs(), rhs_val, expr.kind()));
                        return CondValue(cond_value, refinements, null);
                      }
                    } break;
                */
                if (expr instanceof ast.Call) {
                    throw new python.Error('Not implemented.');
                    /*
                        auto apply = Apply(expr);
                        auto callee = Apply(expr).callee();
                        if (callee.kind() == TK_VAR) {
                          if (Var(callee).name().name() == "isinstance") {
                            checkApplyNumInputs(apply, 2);
                            return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
                          }
                          if (Var(callee).name().name() == "hasattr") {
                            checkApplyNumInputs(apply, 2);
                            return emitHasAttr(apply.inputs()[0], apply.inputs()[1]);
                          }
                        }
                        auto sv = emitSugaredExpr(apply.callee(), 1);
                        auto loc = apply.callee().range();
                        if (auto special_form = dynamic_cast<SpecialFormValue*>(sv.get())) {
                          if (special_form->form() == prim::isinstance) {
                            checkApplyNumInputs(apply, 2);
                            return emitIsInstance(apply.inputs()[0], apply.inputs()[1]);
                          }
                        }
                    */
                }
                const expr_out = this.emitToBool(expr, this.emitExpr(expr));
                let static_if = null;
                const kind = expr_out.node().kind();
                if (kind === 'aten::is_scripting') {
                    static_if = true;
                } else if (kind === 'aten::has_torch_function') {
                    static_if = false;
                }
                const maybe_ivalue = torch._C.toIValue(expr_out);
                if (maybe_ivalue) {
                    static_if = maybe_ivalue.toBool();
                }
                return new torch._C.CondValue(expr_out, new torch._C.RefinementSet({}), static_if);
            }
            emitIfElseBlocks(loc, cond_value, trueBranch, falseBranch) {
                if (cond_value.staticIf() !== null) {
                    if (cond_value.staticIf()) {
                        this.insertRefinements(loc, cond_value.refinements());
                        this.emitStatements(trueBranch);
                    } else {
                        this.insertRefinements(loc, cond_value.refinements().Not());
                        this.emitStatements(falseBranch);
                    }
                    return;
                }
                const n = this.graph.insertNode(this.create('prim::If', loc, 0));
                n.addInput(cond_value.value());
                const true_block = n.addBlock();
                const false_block = n.addBlock();
                /* const save_true = */ this.emitSingleIfBranch(true_block, trueBranch, cond_value.refinements());
                /* const save_false = */ this.emitSingleIfBranch(false_block, falseBranch, cond_value.refinements().Not());
                const true_exits = this.exit_blocks.has(true_block);
                const false_exits = this.exit_blocks.has(false_block);
                if (true_exits && false_exits) {
                    this.exit_blocks.add(n.owningBlock());
                }
                /*
                const mutated_variables = new Set();
                for (const v of save_true.definedVariables()) {
                    const insert = new torch._C.WithInsertPoint(false_block);
                    if (save_false.findInAnyFrame(v) || false_exits) {
                        mutated_variables.insert(v);
                    } else {
                        if (reportSourceLocation(loc.source().size())) {
                            this.environment_stack.setVariableTypeError(v, [=]() -> std::string {
                            error << v << " is not defined in the false branch";
                            return error.what();
                            });
                        } else {
                            this.environment_stack.setVariableTypeError(v, [=]() -> std::string {
                            std::stringstream ss;
                            ss << v << " is not defined in the false branch. "
                                << "The source info is eliminated due to the source file is too large. "
                                << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                                << "as env var";
                            return ss.str();
                            });
                        }
                    }
                    insert.dispose();
                }
                for (const v of save_false.definedVariables()) {
                  {
                    const insert = new torch._C.WithInsertPoint(true_block);
                    if (save_true.findInAnyFrame(v) || true_exits) {
                      mutated_variables.insert(v);
                    } else {
                      if (reportSourceLocation(loc.source().size())) {
                        ErrorReport error(loc);
                        environment_stack.setVariableTypeError(v, [=]() -> std::string {
                          error << v << " is not defined in the true branch";
                          return error.what();
                        });
                      } else {
                        environment_stack.setVariableTypeError(v, [=]() -> std::string {
                          std::stringstream ss;
                          ss << v << " is not defined in the false branch. "
                             << "The source info is eliminated due to the source file is too large. "
                             << "To get it back, please set PYTORCH_JIT_ENABLE_LARGE_SOURCE_LOCATION=1 "
                             << "as env var";
                          return ss.str();
                        });
                      }
                    }
                  }
                }
                for (const x of mutated_variables) {
                    let tv = null;
                    let fv = null;
                    {
                        const insert = new torch._C.WithInsertPoint(true_block);
                        if (!true_exits) {
                            tv = save_true.getVar(x, loc);
                        }
                        insert.dispose();
                    }
                    {
                        const insert = new torch._C.WithInsertPoint(false_block);
                        if (!false_exits) {
                            fv = save_false.getVar(x, loc);
                        }
                        insert.dispose();
                    }
                    if (true_exits && false_exits) {
                        continue;
                    } else if (true_exits) {
                        tv = graph.createUninitialized(fv.type())
                                .insertBefore(true_block.return_node())
                                .output();
                        graph.createStore(x, tv).insertBefore(true_block.return_node());
                    } else if (false_exits) {
                        fv = graph.createUninitialized(tv.type())
                                .insertBefore(false_block.return_node())
                                .output();
                        graph.createStore(x, fv).insertBefore(false_block.return_node());
                    }
                    const maybe_sugared_x = this.environment_stack.findInAnyFrame(x);
                    const full_type = null;
                    if (maybe_sugared_x) {
                        Value* maybe_simple = asSimple(maybe_sugared_x);
                        if (maybe_simple) {
                            full_type = maybe_simple.type();
                        }
                    }
                    const default_to_union = full_type &&
                        (full_type.kind() == UnionType::Kind ||
                        full_type.kind() == OptionalType::Kind ||
                        full_type.kind() == NumberType::Kind);
                    auto unified = unifyTypes(tv.type(), fv.type(), default_to_union=default_to_union);
                    if (!unified) {
                        ErrorReport error(loc);
                        error << "Type mismatch: " << x << " is set to type "
                            << tv.type().repr_str() << " in the true branch"
                            << " and type " << fv.type().repr_str()
                            << " in the false branch";
                        if (save_true.findInParentFrame(x) ||
                            save_false.findInParentFrame(x)) {
                        throw ErrorReport(error);
                        } else {
                        environment_stack.setVariableTypeError(
                            x, [=]() . std::string { return error.what(); });
                        continue;
                        }
                    }
                    this.environment_stack.setType(x, unified);
                }
                */
                throw new python.Error('Not implemented.');
            }
            emitSingleIfBranch(b, branch, refinements) {
                this.pushFrame(b);
                const guard = new torch._C.WithInsertPoint(b);
                this.insertRefinements(branch, refinements);
                this.emitStatements(branch);
                const frame = this.popFrame();
                guard.dispose();
                return frame;
            }
            emitToBool(loc, v) {
                let out = null;
                const bool_cast = this.environment_stack.getSugaredVar("bool", loc);
                out = torch._C.asSimple(bool_cast.call(loc, this.method, [new torch._C.NamedValue(v)], [], 0));
                if (!out) {
                    throw new python.Error('Could not cast value to bool.');
                }
                if (!out.type().isSubtypeOf(torch.BoolType.get())) {
                    throw new python.Error('Expected a bool expression for condition.');
                }
                return out;
            }
            emitAssignment(stmt) {
                if (stmt.targets.length === 1) {
                    return this.emitSingleAssignment(stmt);
                }
                if (stmt.targets.length <= 1) {
                    throw new python.Error('Invalid assignment.');
                }
                throw new python.Error('Not implemented.');
                /*
                const tmp_name = this.createTempName("$tmp_assign_");
                this.environment_stack.setSugaredVar(stmt.value, tmp_name, this.emitSugaredExpr(stmt.value, 1), annotated_type=null);
                const ident = new ast.Name(tmp_name);
                for (const expr of lhs_list) {
                    const assign = new ast.Assign(targets, value, ctx);
                    this.emitSingleAssignment(Assign.create(stmt,
                        List<Expr>.create(expr.range(), [expr]),
                        Maybe<Expr>::create(stmt.rhs().range(), ident),
                            Maybe<Expr>::create(stmt.range())));
                }
                */
            }
            emitSingleAssignment(stmt) {
                const rhs = stmt.value;
                const [lhs] = stmt.targets;
                if (lhs instanceof ast.Name) {
                    const type = null;
                    const rhs_sugared_val = this.emitSugaredExpr(rhs, 1, type);
                    // BC HACK
                    this.environment_stack.setSugaredVar(stmt, lhs.id, rhs_sugared_val, /*annotated_type=*/type);
                } else if (lhs instanceof ast.Tuple) {
                    this.emitTupleAssign(lhs, rhs);
                } else {
                    throw new python.Error('Unexpected expression on left-hand side of assignment.');
                }
            }
            emitTupleAssign(...args) {
                if (args.length === 2) {
                    const [tl, rhs] = args;
                    let n_binders = tl.elts.length;
                    const starred_unpack = this.validateAssignLhsExpr(tl.elts, tl);
                    if (starred_unpack) {
                        n_binders--;
                    }
                    const output = this.emitSugaredExpr(rhs, n_binders);
                    this.emitTupleAssign(tl, output, rhs, n_binders, starred_unpack);
                } else if (args.length === 5) {
                    const [tl, rhs_output, rhs_loc, n_binders, starred_unpack] = args;
                    const outputs = rhs_output.asTuple(rhs_loc, this.method, starred_unpack ? null : n_binders);
                    if (outputs.length < n_binders) {
                        throw new python.Error('Not enough values to unpack.');
                    }
                    if (outputs.length > n_binders && !starred_unpack) {
                        throw new python.Error('Too many values to unpack.');
                    }
                    this.emitExprsAssign(tl.elts, outputs, rhs_loc, n_binders);
                } else {
                    throw new python.Error('Not implemented.');
                }
            }
            emitExprsAssign(lhs_exprs, outputs /*, rhs_loc, n_binders */) {
                let i = 0;
                for (const assignee of lhs_exprs) {
                    if (assignee instanceof ast.Subscript) {
                        throw new python.Error('Not implemented.');
                        /*
                        this.emitSubscriptAssign(
                            rhs_loc,
                            Subscript(assignee),
                            NamedValue(rhs_loc, outputs.at(i).asValue(rhs_loc, method)));
                        i++;
                        */
                    } else if (assignee instanceof ast.Name) {
                        this.environment_stack.setSugaredVar(assignee, assignee.id, outputs[i], /*annotated_type=*/null);
                        i++;
                    } else if (assignee instanceof ast.Starred) {
                        throw new python.Error('Not implemented.');
                        /*
                        auto var = Starred(assignee).expr();
                        if (var.kind() != TK_VAR) {
                        throw(
                            ErrorReport(var) << "Cannot pack a tuple into a non-variable");
                        }
                        size_t n_matched = outputs.size() - n_binders;
                        ArrayRef<std::shared_ptr<SugaredValue>> outputs_ref = outputs;
                        auto values = fmap(
                            outputs_ref.slice(i, n_matched),
                            [&](const std::shared_ptr<SugaredValue>& v) {
                            return v.asValue(assignee.range(), method);
                            });
                        auto tup = graph.insertNode(graph.createTuple(values)).output();
                        environment_stack.setVar(var.range(), Var(var).name().name(), tup);
                        i += n_matched;
                        */
                    } else if (assignee instanceof ast.Tuple) {
                        throw new python.Error('Not implemented.');
                        /*
                        // recursively emit tuple assignments on tuple literal input
                        TupleLiteral sub_tl = TupleLiteral(assignee);
                        size_t sub_n_binders = sub_tl.inputs().size();
                        bool sub_starred_unpack =
                            validateAssignLhsExpr(sub_tl.inputs(), sub_tl.range());
                        if (sub_starred_unpack)
                        sub_n_binders--;
                        emitTupleAssign(
                            sub_tl,
                            outputs.at(i),
                            rhs_loc,
                            sub_n_binders,
                            sub_starred_unpack);
                        i++;
                        */
                    } else if (assignee instanceof ast.Attribute) {
                        throw new python.Error('Not implemented.');
                        /*
                        emitSelectAssign(assignee, outputs.at(i), rhs_loc);
                        i++;
                        */
                    } else {
                        throw new python.Error('Unexpected expression on left-hand side of assignment.');
                    }
                }
            }
            emitReturn(stmt) {
                let declared_return_type = this._def_stack[this._def_stack.length - 1]._declared_return_type_;
                let actual_return = this.emitExpr(stmt.value, declared_return_type);
                if (declared_return_type) {
                    if (!(actual_return.type().isSubtypeOf(torch.TensorType.get()) && actual_return.type().isSubtypeOf(torch.NoneType.get()))) {
                        actual_return = this.tryConvertToType(stmt, this.graph, declared_return_type, actual_return, /*allow_conversions=*/true);
                    }
                    if (!actual_return.type().isSubtypeOf(declared_return_type)) {
                        throw new python.Error(`Invalid return type.`);
                    }
                } else {
                    declared_return_type = this._def_stack[this._def_stack.length - 1]._merged_return_type;
                    if (!declared_return_type) {
                        declared_return_type = actual_return.type();
                    }
                    const merged_return_type = torch._C.unifyTypes(declared_return_type, actual_return.type());
                    if (!merged_return_type) {
                        throw new python.Error(`Invalid return type.`);
                    }
                    declared_return_type = merged_return_type;
                }
                this._def_stack[this._def_stack.length - 1]._merged_return_type = declared_return_type;
                if (declared_return_type === torch.AnyType.get() && actual_return.type() !== torch.AnyType.get()) {
                    actual_return = this.graph.insertUncheckedCast(actual_return, declared_return_type);
                }
                this.graph.insertNode(this.graph.create('prim::ReturnStmt', [actual_return], 0));
                this.exit_blocks.add(this.environment_stack.block());
            }
            getNamedValues(trees, maybe_unpack) {
                const values = [];
                for (const tree of trees) {
                    if (maybe_unpack && tree instanceof ast.Starred) {
                        throw new python.Error('Starred argument not supported.');
                    } else {
                        values.push(new torch._C.NamedValue(this.emitExpr(tree)));
                    }
                }
                return values;
            }
            getValues(trees, maybe_unpack) {
                return this.getNamedValues(trees, maybe_unpack).map((value) => value.value(this.graph));
            }
            emitExpr(tree, type_hint) {
                type_hint = type_hint || null;
                let out_val = this.emitSugaredExpr(tree, 1, type_hint).asValue(tree, this.method);
                if (type_hint === torch.AnyType.get() && out_val.type() !== torch.AnyType.get()) {
                    out_val = this.graph.insertUncheckedCast(out_val, type_hint);
                }
                return out_val;
            }
            emitSugaredExpr(tree, n_binders, type_hint) {
                if (tree instanceof ast.Name) { // TK_VAR
                    return this.environment_stack.getSugaredVar(tree.id);
                } else if (tree instanceof ast.Attribute) {
                    const sv = this.emitSugaredExpr(tree.value, 1);
                    return sv.attr(tree, this.method, tree.attr);
                } else if (tree instanceof ast.Call) { // TK_APPLY
                    return this.emitApplyExpr(tree, n_binders, type_hint);
                } if (tree instanceof ast.Subscript) {
                    throw new python.Error('Not implemented.');
                }
                return new torch._C.SimpleValue(this.emitSimpleExpr(tree, type_hint));
            }
            emitApplyExpr(apply, n_binders, type_hint) {
                type_hint = type_hint || null;
                const sv = this.emitSugaredExpr(apply.func, 1);
                const loc = apply.func;
                if (sv instanceof torch._C.SpecialFormValue) {
                    return this.emitApplySpecialForm(sv.form(), apply, sv, type_hint);
                }
                const args = this.getNamedValues(apply.args, true);
                const kwargs = this.emitAttributes(apply.keywords);
                return sv.call(loc, this.method, args, kwargs, n_binders);
            }
            emitAttributes(attributes) {
                return attributes.map((attr) => new torch._C.NamedValue(attr, attr.arg, this.emitExpr(attr.value)));
            }
            emitApplySpecialForm(form, apply, sv /*, type_hint */) {
                switch (form) {
                    case 'prim::fork': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::awaitable': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::annotate': {
                        this.checkApplyNumInputs(apply, 2);
                        const type = this._typeParser.parseTypeFromExpr(apply.args[0]);
                        let expr = torch._C.tryConvertToType(apply, this.graph, type, this.emitExpr(apply.args[1], type), /*allow_conversions=*/true);
                        if (!expr.type().isSubtypeOf(type)) {
                            throw new python.Error('Invalid expression type.');
                        }
                        if ((type instanceof torch.OptionalType || (type instanceof torch.UnionType && type.expect(torch.UnionType).canHoldType(torch.NoneType.get()))) && expr.type().isSubtypeOf(torch.NoneType.get())) {
                            const none = this.graph.createNone();
                            none.output().setType(type);
                            this.graph.insertNode(none);
                            expr = none.output();
                        }
                        return new torch._C.SimpleValue(expr);
                    }
                    case 'prim::unchecked_cast': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::GetAttr': {
                        this.checkApplyNumInputsRange(apply, 2, 3);
                        const obj = this.emitSugaredExpr(apply.args[0], 1);
                        if (apply.args[1] instanceof ast.Constant === false || typeof apply.args[1].value !== 'string') {
                            throw new python.Error('Invalid argument.');
                        }
                        const name = apply.args[1].value;
                        if (apply.args.length === 2) {
                            return obj.attr(apply, this.method, name);
                        } else if (obj.hasAttr(apply, this.method, name)) {
                            return obj.attr(apply, this.method, name);
                        }
                        return this.emitSugaredExpr(apply.inputs()[2], 1);
                    }
                    case 'prim::Uninitialized': {
                        this.checkApplyNumInputs(apply, 1);
                        const type = this._typeParser.parseTypeFromExpr(apply.args[0]);
                        const out = this.graph.insertNode(this.graph.createUninitialized(type)).setSourceRange(apply);
                        return new torch._C.SimpleValue(out.output());
                    }
                    case 'prim::TupleConstruct': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::isinstance': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::tolist': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::HasAttr': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::CreateObject': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::range': {
                        const input_vals = this.getValues(apply.args, true);
                        return new torch._C.RangeValue(apply, this.method, input_vals);
                    }
                    case 'prim::enumerate': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::zip': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::list': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'prim::dict': {
                        throw new python.Error('Not implemented.');
                    }
                    case 'aten::index': {
                        throw new python.Error('Not implemented.');
                    }
                    default: {
                        throw new python.Error(`Unsupported special form '${sv.from()}'.`);
                    }
                }
            }
            emitSimpleExpr(tree, type_hint) {
                if (tree instanceof ast.Constant) {
                    if (tree.value === true) {
                        return this.graph.insertConstant(true, tree);
                    } else if (tree.value === false) {
                        return this.graph.insertConstant(false, tree);
                    } else if (tree.value === null) {
                        return this.graph.insertConstant(null, tree); // IValue()
                    } else if (typeof tree.value === 'string') {
                        return this.emitStringLiteral(tree);
                    }
                    return this.emitConst(tree);
                } else if (tree instanceof ast.List) {
                    return this.emitListLiteral(tree, type_hint);
                } else if (tree instanceof ast.Tuple) {
                    const values = this.getValues(tree.elts, /*maybe_unpack=*/true);
                    return this.graph.insertNode(this.graph.createTuple(values)).output();
                }
                throw new python.Error(`Simple expression '${tree.__class__.__name__}' not implemented.`);
            }
            emitStringLiteral(c) {
                return torch._C.insertConstant(this.graph, c.value, c);
            }
            emitConst(c) {
                if (Number.isInteger(c.value)) {
                    return torch._C.materializeConstant(c.value, this.graph, c, this.integral_constants);
                } else if (typeof c.value === 'number') {
                    return torch._C.materializeConstant(c.value, this.graph, c, this.fp_constants);
                }
                throw new python.Error(`Unsupported constant type.`);
            }
            emitListLiteral(ll, type_hint) {
                type_hint = type_hint || null;
                const values = this.getValues(ll.elts, true);
                if (values.length === 0 && type_hint === null) {
                    throw new python.Error('Not implemented.');
                }
                let inferred_elem_type = torch.TensorType.get();
                const refined_type_hint = type_hint;
                const annotated_union_type = refined_type_hint && refined_type_hint.isUnionType() ? refined_type_hint : null;
                const all_candidates = [];
                if (refined_type_hint) {
                    throw new python.Error('Not implemented.');
                }
                if (values.length !== 0) {
                    const types = values.map((v) => v.type());
                    const elem_type_hint = refined_type_hint && refined_type_hint.kind() === 'ListType' ? refined_type_hint.getElementType() : null;
                    const unified_elem_type = torch._C.unifyTypeList(types, null /*nowhere*/, /*default_to_union=*/true, elem_type_hint);
                    if (!refined_type_hint && unified_elem_type.kind() === 'UnionType') {
                        throw new python.Error('Not implemented.');
                    }
                    if (all_candidates.length === 0 && refined_type_hint && !unified_elem_type.isSubtypeOf(inferred_elem_type)) {
                        throw new python.Error('Not implemented.');
                    }
                    if (all_candidates.length !== 0) {
                        this.refineAndSetListTypeHintFromCandidatesVector(all_candidates, type_hint, refined_type_hint, unified_elem_type, ll);
                        inferred_elem_type = refined_type_hint.expect(torch.ListType).getElementType();
                    }
                    if (!refined_type_hint) {
                        inferred_elem_type = unified_elem_type;
                    }
                }
                let result = this.graph.insertNode(this.graph.createList(inferred_elem_type, values));
                if (annotated_union_type) {
                    const n = this.graph.insertNode(this.graph.create('prim::unchecked_cast', [result.output()]));
                    n.output().setType(annotated_union_type);
                    result = n;
                }
                return result.output();
            }
            create(kind, loc, n_outputs) {
                return this.graph.create(kind, n_outputs).setSourceRange(loc);
            }
            insertRefinements(loc, ref) {
                for (const r of ref.activeRefinements()) {
                    const v = this.environment_stack.getVar(r.identifier(), loc);
                    const new_v = this.graph.insertUncheckedCast(v, r.type());
                    this.environment_stack.setVar(loc, r.identifier(), new_v);
                }
            }
            shouldDeriveSetStateType(def, schema) {
                const noTypeAnnotations = schema.arguments.every((arg) => arg.is_inferred_type());
                const shouldInfer = def.name === '__setstate__' && noTypeAnnotations;
                if (!shouldInfer) {
                    return false;
                }
                if (def.name !== '__setstate__' && def.args.args.length !== 2) {
                    throw new python.Error(`Invalid '__setstate' method.`);
                }
                return true;
            }
            checkApplyNumInputs(apply, expected_inputs) {
                if (apply.args.length !== expected_inputs) {
                    throw new python.Error('Invalid number of arguments.');
                }
                if (apply.keywords.length > 0) {
                    throw new python.Error('Invalid number of keyword arguments.');
                }
            }
            checkApplyNumInputsRange(apply, min_expected_inputs, max_expected_inputs) {
                const position_arg_size = apply.args.length;
                if (position_arg_size < min_expected_inputs || position_arg_size > max_expected_inputs) {
                    throw new python.Error('Invalid number of arguments.');
                }
                if (apply.keywords.length > 0) {
                    throw new python.Error('Invalid number of keyword arguments.');
                }
            }
            validateAssignLhsExpr(lhs /*, r */) {
                let num_normal_assign = 0;
                let num_starred = 0;
                for (const assignee of lhs) {
                    if (assignee instanceof ast.Name || assignee instanceof ast.Subscript || assignee instanceof ast.Tuple || assignee instanceof ast.Attribute) {
                        num_normal_assign++;
                    } else if (assignee instanceof ast.Starred) {
                        num_starred++;
                    } else {
                        throw new python.Error('Assignment must be a variable, subscript, or starred expression.');
                    }
                }
                if (num_starred > 1) {
                    throw new python.Error('Only one starred expression is allowed.');
                }
                if (num_starred > 0 && num_normal_assign === 0) {
                    throw new python.Error('Invalid starred expression.');
                }
                return num_starred;
            }
            createTempName(prefix) {
                return `${prefix}${this._temp_name_count++}`;
            }
            handleMaybeNoReturn(def, block) {
                const decl_ret = this._def_stack[this._def_stack.length - 1]._declared_return_type;
                if (this.exit_blocks.size === 0) {
                    if (decl_ret && decl_ret !== torch.NoneType.get()) {
                        throw new python.Error('Function was not annotated as having type None, but does not return along all paths.');
                    }
                    const b = new torch._C.WithInsertPoint(block.nodes()[-1]);
                    // this.emitReturn(Return::create(def.range(), Expr(Compound::create(TK_NONE, def.range(), {}))));
                    b.dispose();
                    throw new Error();
                } else if (this._def_stack[this._def_stack.length - 1]._merged_return_type === null) {
                    this._def_stack[this._def_stack.length - 1]._merged_return_type = decl_ret === null ? torch.NoneType.get() : decl_ret;
                }
            }
        });
        this.registerType('torch.jit.CompilationUnit', class {
            constructor() {
                this._functions = new Map();
                this._classes = new Map();
            }
            register_type(namedType) {
                this._classes.set(namedType.annotation_str, namedType);
            }
            register_function(fn) {
                this._functions.set(fn.name(), fn);
                return fn;
            }
            define(...args) {
                const [prefix] = args;
                if (Array.isArray(args[1])) {
                    const [, /* properties */, /* propResolvers */, definitions, defResolvers, self, shouldMangle, operator_set_version] = args;
                    const function_table = new Map();
                    const functions = [];
                    const record_function = (fn) => {
                        function_table.set(fn.name(), fn);
                        functions.push(fn);
                        this.register_function(fn);
                    };
                    // properties
                    for (let i = 0; i < definitions.length; i++) {
                        const fn = this.define(prefix, definitions[i], defResolvers[i], self, function_table, shouldMangle, 'method', operator_set_version);
                        record_function(fn);
                    }
                    for (const [name, fn] of function_table) {
                        if (name === '__init__') {
                            fn.ensure_defined();
                        }
                    }
                    for (const fn of functions) {
                        fn.ensure_defined();
                    }
                    return functions;
                } else if (args[1] instanceof ast.FunctionDef) {
                    const [, def, resolver, self, function_table, shouldMangle, type, operator_set_version] = args;
                    let _resolver = resolver;
                    if (!self) {
                        _resolver = new torch._C.FunctionResolver(resolver, function_table);
                    }
                    const creator = (method) => {
                        // let call_name = method.qualname().name();
                        // if (self) {
                        //    const atoms = method.qualname().atoms();
                        //    // TORCH_INTERNAL_ASSERT(atoms.size() >= 2);
                        //    call_name = `${atoms.at(atoms.size() - 2)}.${atoms.at(atoms.size() - 1)}`;
                        // }
                        // this.call(call_name, def.range());
                        return new torch.jit.to_ir(def, _resolver, self, method);
                    };
                    const name = prefix ? new torch._C.QualifiedName(prefix, def.name) : new torch._C.QualifiedName(def.name);
                    const graph = new torch.Graph();
                    graph.set_op_version(operator_set_version);
                    const fn = new torch._C.GraphFunction(name, graph, creator);
                    fn.__ast__ = def;
                    if (shouldMangle && this.find_function(name)) {
                        // name = mangle(name);
                    }
                    if (self) {
                        if (type === 'hook') {
                            self.getClassType().addForwardHook(fn);
                        } else if (type === 'prehook') {
                            self.getClassType().addPreHook(fn);
                        } else {
                            self.getClassType().addMethod(fn);
                        }
                    }
                    return fn;
                }
                throw new python.Error('Invalid arguments.');
            }
            get_type(name) {
                return this._classes.get(name.qualifiedName());
            }
            get_class(name) {
                return this.get_type(name);
            }
            find_function(name) {
                const key = name.qualifiedName();
                return this._functions.get(key);
            }
        });
        this.registerFunction('torch._C.ConvertToSSA', (graph) => {
            const ctrl = new torch._C.ControlFlowLoadStores();
            ctrl.run(graph);
            const exit_vars = new torch._C.LoopContinuations();
            exit_vars.run(graph);
            torch._C.InlineLoopCondition(graph);
            const erase_loads_stores = new torch._C.EraseLoadStores();
            erase_loads_stores.run(graph);
            torch._C.TransformExits(graph);
        });
        this.registerType('torch._C.MiniEnvironment', class {
            constructor(b, next) {
                this.next = next || null;
                this.table = new Map();
            }
            setVar(name, value) {
                this.table.set(name, value);
            }
            findInThisFrame(name) {
                if (this.table.has(name)) {
                    return this.table.get(name);
                }
                return null;
            }
            findInAnyFrame(name) {
                for (let runner = this; runner; runner = runner.next) {
                    const r = runner.findInThisFrame(name);
                    if (r) {
                        return r;
                    }
                }
                return null;
            }
        });
        this.registerType('torch._C.ValueEnvironment', class extends torch._C.MiniEnvironment {
        });
        this.registerType('torch._C.TypeEnvironment', class extends torch._C.MiniEnvironment {
        });
        this.registerType('torch._C.ControlFlowLoadStores', class {
            pushFrame(b) {
                this.environment_stack = new torch._C.TypeEnvironment(b, this.environment_stack);
            }
            popFrame() {
                const old_frame = this.environment_stack;
                this.environment_stack = this.environment_stack.next;
                return old_frame;
            }
            addControlFlowLoadStores(block) {
                this.pushFrame(block);
                for (const n of block.nodes()) {
                    switch (n.kind()) {
                        case 'prim::If': {
                            this.addIfLoadStores(n);
                            break;
                        }
                        case 'prim::Loop': {
                            this.addLoopLoadStores(n);
                            break;
                        }
                        case 'prim::Closure': {
                            for (const b of n.blocks()) {
                                this.addControlFlowLoadStores(b);
                            }
                            break;
                        }
                        case 'prim::Store': {
                            this.environment_stack.setVar(n.s('name'), n.input().type());
                            break;
                        }
                        case 'prim::ComprehensionScope': {
                            this.addControlFlowLoadStores(n.blocks()[0]);
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                }
                return this.popFrame();
            }
            run(graph) {
                this.addControlFlowLoadStores(graph.block());
            }
        });
        this.registerType('torch._C.LoopContinuations', class {
            run(/* graph */) {
            }
        });
        this.registerFunction('torch._C.InlineLoopCondition', (/* graph */) => {
        });
        this.registerType('torch._C.EraseLoadStores', class {
            pushFrame(b) {
                this.environment_stack = new torch._C.ValueEnvironment(b, this.environment_stack);
            }
            popFrame() {
                const old_frame = this.environment_stack;
                this.environment_stack = this.environment_stack.next;
                return old_frame;
            }
            eraseBlockLoadStores(block) {
                this.pushFrame(block);
                for (const n of block.nodes()) {
                    switch (n.kind()) {
                        case 'prim::Store': {
                            this.environment_stack.setVar(n.s('name'), n.input());
                            n.destroy();
                            break;
                        }
                        case 'prim::Load': {
                            const name = n.s('name');
                            const value = this.environment_stack.findInAnyFrame(name);
                            if (!value) {
                                throw new python.Error(`Undefined variable '${name}'.`);
                            }
                            n.output().replaceAllUsesWith(value);
                            n.destroy();
                            break;
                        }
                        case 'prim::ComprehensionScope': {
                            const [body] = n.blocks();
                            this.eraseBlockLoadStores(body);
                            for (const body_node of body.nodes()) {
                                body_node.moveBefore(n);
                            }
                            n.destroy();
                            break;
                        }
                        default: {
                            for (const b of n.blocks()) {
                                this.eraseBlockLoadStores(b);
                            }
                            break;
                        }
                    }
                }
                this.popFrame();
            }
            run(graph) {
                this.eraseBlockLoadStores(graph.block());
            }
        });
        this.registerFunction('torch._C.convertEnterExitNodesToWithBlocks', (/* graph */) => {
        });
        this.registerFunction('torch._C.inlineConsecutiveIfs', (/* graph */) => {
        });
        this.registerType('torch._C.ExitPair', class {
            constructor(exit_v, exit_val_ref) {
                const exit_vals = [];
                for (const v of exit_val_ref) {
                    exit_vals.push(v);
                }
                if (exit_v.type() !== torch.BoolType.get()) {
                    throw new python.Error('Invalid exit value type.');
                }
                this.first = exit_v;
                this.second = exit_vals;
            }
            hasExited() {
                return this.first;
            }
            exitValues() {
                return this.second;
            }
        });
        this.registerType('torch._C.ExitTransformer', class {
            constructor(graph) {
                this._graph = graph;
                this._target_block = null;
                this._unit_values = new Map();
                const guard = new torch._C.WithInsertPoint(this._graph.block().nodes()[0]);
                this._true_val = this._graph.insertConstant(true);
                this._false_val = this._graph.insertConstant(false);
                this._throws_val = this.getUnitValue(torch.BoolType.get());
                guard.dispose();
            }
            getUnitValue(type) {
                const maybe_val = this._unit_values.get(type);
                if (maybe_val) {
                    return maybe_val;
                }
                const unit = this._graph.createUninitialized(type).insertAfter(this._graph.param_node()).output();
                this._unit_values.set(type, unit);
                return unit;
            }
            transformReturnStmts() {
                this._current_exit_kind = 'prim::ReturnStmt';
                this.transformExits(this._graph.block());
            }
            transformLoopContinuations() {
                this._current_exit_kind = 'prim::LoopContinuation';
                this.transformExits(this._graph.block());
            }
            destroyNodeAfterExit(n) {
                for (const output of n.outputs()) {
                    if (output.uses().length > 0) {
                        output.replaceAllUsesWith(this.getUnitValue(output.type()));
                    }
                }
                n.destroy();
            }
            deleteAfterExitNodes(block, iter) {
                const nodes = block.nodes();
                if (iter === nodes[nodes.length - 1]) {
                    return;
                }
                const insert = new torch._C.WithInsertPoint(block.nodes()[0]);
                for (const it of nodes.reverse()) {
                    if (it === iter) {
                        break;
                    }
                    if (it !== block.return_node()) {
                        this.destroyNodeAfterExit(it);
                    }
                }
                this.destroyNodeAfterExit(iter);
                insert.dispose();
            }
            updateTargetBlock(block) {
                if (torch._C.ExitTransformer.owningNodeKind(block) === 'prim::Loop' && this._current_exit_kind === 'prim::LoopContinuation') {
                    this._target_block = block;
                } else if (torch._C.ExitTransformer.isGraphOrClosureBlock(block) && this._current_exit_kind === 'prim::ReturnStmt') {
                    this._target_block = block;
                }
            }
            transformExits(block) {
                const prev_target_block = this._target_block;
                this.updateTargetBlock(block);
                let exit_pair = this.constructWontExitPair();
                for (const node of block.nodes()) {
                    const it = node.next;
                    switch (node.kind()) {
                        case 'prim::RaiseException': {
                            exit_pair = this.constructThrowsExitPair();
                            break;
                        }
                        case 'prim::ReturnStmt':
                        case 'prim::LoopContinuation': {
                            if (node.kind() === this._current_exit_kind) {
                                exit_pair = this.constructWillExitPair(node.inputs());
                                node.destroy();
                            }
                            break;
                        }
                        case 'prim::If': {
                            exit_pair = this.transformIf(node);
                            break;
                        }
                        case 'prim::With': {
                            exit_pair = this.transformWith(node);
                            break;
                        }
                        case 'prim::Closure': {
                            this.transformExits(node.blocks()[0]);
                            break;
                        }
                        case 'prim::Loop': {
                            exit_pair = this.transformLoop(node);
                            break;
                        }
                        default: {
                            break;
                        }
                    }
                    const status = this.getExitStatus(exit_pair);
                    if (status === 'WILL' || status === 'THROWS') {
                        this.deleteAfterExitNodes(block, it);
                        break;
                    }
                    if (status === 'MIGHT') {
                        throw new python.Error('Not implemented.');
                        // const nodes = block.nodes();
                        // if (node === nodes[nodes.length - 1]) {
                        //     exit_pair = this.guardBlockNodes(block, exit_pair, it);
                        // }
                        // break;
                    }
                }
                if (this._target_block === block) {
                    if (this.getExitStatus(exit_pair) === 'MIGHT') {
                        const new_if = this._graph.create('prim::If', 0).insertBefore(block.return_node());
                        new_if.addBlock();
                        new_if.addBlock();
                        new_if.addInput(exit_pair.hasExited());
                        torch._C.ExistTransformer.addIfOutputs(new_if, exit_pair.exitValues(), block.outputs());
                        torch._C.ExistTransformer.replaceBlockOutputs(block, new_if.soutputs());
                    } else if (this.getExitStatus(exit_pair) === 'WILL') {
                        torch._C.ExitTransformer.replaceBlockOutputs(block, exit_pair.exitValues());
                    }
                    exit_pair = this.constructWontExitPair();
                }
                this._target_block = prev_target_block;
                return exit_pair;
            }
            constructWontExitPair() {
                return new torch._C.ExitPair(this._false_val, []);
            }
            constructWillExitPair(exit_val_ref) {
                return new torch._C.ExitPair(this._true_val, exit_val_ref);
            }
            getExitStatus(exit_pair) {
                const exit_v = exit_pair.hasExited();
                if (exit_v === this._true_val) {
                    return 'WILL';
                } else if (exit_v === this._false_val) {
                    return 'WONT';
                } else if (exit_v === this._throws_val) {
                    return 'THROWS';
                }
                return 'MIGHT';
            }
            static owningNodeKind(block) {
                if (block.owningNode()) {
                    return block.owningNode().kind();
                }
                return null;
            }
            static isGraphOrClosureBlock(block) {
                return block.owningNode() === null || torch._C.ExistTransformer.owningNodeKind(block) === 'prim::Closure';
            }
            static removeOutputs(b) {
                while (b.outputs().length > 0) {
                    b.eraseOutput(0);
                }
            }
            static registerBlockOutputs(b, outs) {
                for (const out of outs) {
                    b.registerOutput(out);
                }
            }
            static replaceBlockOutputs(b, outs) {
                torch._C.ExitTransformer.removeOutputs(b);
                torch._C.ExitTransformer.registerBlockOutputs(b, outs);
            }
        });
        this.registerFunction('torch._C.convertWithBlocksToEnterExitNodes', (/* graph */) => {
        });
        this.registerFunction('torch._C.TransformExits', (graph) => {
            torch._C.convertEnterExitNodesToWithBlocks(graph);
            const e_loop = new torch._C.ExitTransformer(graph);
            e_loop.transformLoopContinuations();
            const e_ret = new torch._C.ExitTransformer(graph);
            e_ret.transformReturnStmts();
            torch._C.inlineConsecutiveIfs(graph.block());
            torch._C.convertWithBlocksToEnterExitNodes(graph);
        });
        this.registerFunction('torch._C.normalizeRSub', (/* iter */) => {
        });
        this.registerFunction('torch._C.normalizeOpAliases', (/* iter */) => {
        });
        this.registerFunction('torch._C.normalizeIsBool', (/* iter */) => {
        });
        this.registerFunction('torch._C.NormalizeOps', (block) => {
            for (const it of block.nodes()) {
                for (const sub of it.blocks()) {
                    torch._C.NormalizeOps(sub);
                }
                if (torch._C.normalizeRSub(it)) {
                    continue;
                }
                if (torch._C.normalizeOpAliases(it)) {
                    continue;
                }
                if (torch._C.normalizeIsBool(it)) {
                    continue;
                }
            }
        });
        this.registerFunction('torch._C.getInlineEverythingMode', () => {
            return false;
        });
        this.registerFunction('torch._C.runCleanupPasses', (to_clean) => {
            /*
            torch._C.liftClosures(to_clean);
            torch._C.inlineForkedClosures(to_clean);
            */
            if (torch._C.getInlineEverythingMode()) {
                torch._C.Inline(to_clean);
            }
            /*
            torch._C.eraseListLiterals(to_clean);
            torch._C.LowerSimpleTuples(to_clean);
            torch._C.ConstantPropagationImmutableTypes(to_clean);
            torch._C.ConstantPooling(to_clean);
            torch._C.CanonicalizeOutputs(to_clean);
            torch._C.AnnotateWarns(to_clean);
            */
        });
        this.registerType('torch.jit._script.ScriptModule', class extends torch.nn.modules.module.Module {});
        this.registerType('torch.jit._trace.TracedModule', class extends torch.jit._script.ScriptModule {});
        this.registerType('torch.jit._trace.TopLevelTracedModule', class extends torch.jit._trace.TracedModule {});
        this.registerType('torch.jit._script.RecursiveScriptModule', class extends torch.jit._script.ScriptModule {
            constructor(cpp_module) {
                super();
                this._initializing = true;
                this._c = cpp_module;
            }
            static _construct(cpp_module, init_fn) {
                const script_module = new torch.jit._script.RecursiveScriptModule(cpp_module);
                init_fn(script_module);
                torch.jit._script.RecursiveScriptModule._finalize_scriptmodule(script_module);
                return script_module;
            }
            static _finalize_scriptmodule(script_module) {
                script_module._parameters = new torch.ParameterDict(script_module._c).items();
                script_module._buffers = new torch.BufferDict(script_module._c).items();
                // script_module._modules = OrderedModuleDict(script_module._c, script_module._modules)
                script_module._initializing = false;
            }
            get graph() {
                // return this._c._get_method("forward").graph;
                return this._c.graph;
            }
            get code_with_constants() {
                // return this.forward.code_with_constants;
                return this._c.code_with_constants;
            }
            __setattr__(name, value) {
                if (this._initializing) {
                    super.__setattr__(name, value);
                } else if (this._modules.has(name)) {
                    this._modules.set(name, value);
                } else if (this._c.hasattr(name)) {
                    this._c.setattr(name, value);
                } else {
                    //
                }
            }
            __getattr__(name) {
                if (this._initializing) {
                    return super.__getattr__(name);
                }
                if (this._modules.has(name)) {
                    return this._modules.get(name);
                }
                if (this._c.hasattr(name)) {
                    return this._c.getattr(name);
                }
                if (this._c._has_method(name)) {
                    //
                }
                return super.__getattr__(name);
            }
        });
        torch.jit.ScriptModule = torch.jit._script.ScriptModule;
        torch.jit.RecursiveScriptModule = torch.jit._script.RecursiveScriptModule;
        torch.jit.TopLevelTracedModule = torch.jit._trace.TopLevelTracedModule;
        torch.CompilationUnit = torch.jit.CompilationUnit;
        torch._C.CompilationUnit = torch.jit.CompilationUnit;
        torch._C.ScriptModule = torch.ScriptModule;
        torch._C.ClassType = torch.ClassType;
        this.registerType('torch._C.FlatBuffersLoader', class {
            constructor(cu) {
                this._cu = cu;
                const torch = cu.execution.__import__('torch');
                this._torch = torch;
                const dtypes = Array.from(new Set(Object.values(torch).filter((obj) => obj instanceof torch.dtype)));
                this._dtypes = new Map(dtypes.map((dtype) => [dtype.scalar_type(), dtype]));
                this._ivalue_parsers = new Map();
                this._ivalue_parsers.set(torch.mobile.serialization.Int, (ivalue) => ivalue.val.int_val);
                this._ivalue_parsers.set(torch.mobile.serialization.Bool, (ivalue) => ivalue.val.bool_val);
                this._ivalue_parsers.set(torch.mobile.serialization.Double, (ivalue) => ivalue.val.double_val);
                this._ivalue_parsers.set(torch.mobile.serialization.TensorMetadata, (ivalue) => this.parseTensor(ivalue));
                this._ivalue_parsers.set(torch.mobile.serialization.Object, (ivalue) => this.parseObject(ivalue));
            }
            parseModule(module) {
                this._module = module;
                this._all_functions = new Map();
                this._all_ivalues = new Array(module.ivalues.length);
                this._all_types = new Array(module.object_types.length);
                const mobile_ivalue_size = module.mobile_ivalue_size ? module.mobile_ivalue_size : module.ivalues.length;
                for (let i = 0; i < mobile_ivalue_size; i++) {
                    this.parseAndPopulate(i, module.ivalues[i]);
                }
                const m = this._all_ivalues[module.state_obj];
                for (const [name, value] of this._all_functions) {
                    const class_index = module.ivalues[name].val.class_type;
                    const class_type = this._all_types[class_index];
                    if (value) {
                        class_type.addMethod(value);
                    }
                }
                m._min_operator_version = module.operator_version;
                m._bytecode_version = module.bytecode_version;
                return m;
            }
            parseAndPopulate(i, ivalue) {
                if (ivalue.val instanceof torch.mobile.serialization.Function) {
                    this._all_functions.set(i, this.parseFunction(ivalue.val));
                } else {
                    this._all_ivalues[i] = this.parseIValue(ivalue);
                }
            }
            parseFunction(/* val */) {
                return null;
            }
            parseIValue(ivalue) {
                if (ivalue.val) {
                    const callback = this._ivalue_parsers.get(ivalue.val.constructor);
                    return callback(ivalue);
                }
                return null;
            }
            parseTensor(ivalue) {
                return this.parseTensorFromMetadata(ivalue.val);
            }
            parseTensorFromMetadata(metadata) {
                if (metadata.quantized_schema) {
                    throw new torch.Error('Quantized schema not implemented.');
                }
                const index = metadata.storage_location_index;
                const data = this._module.storage_data[index].data;
                const dtype = this._dtypes.get(metadata.scalar_type);
                const size = data.length / dtype.itemsize();
                const storage = this._cu.execution.invoke('torch.storage.TypedStorage', [size, dtype]);
                storage._set_cdata(data);
                const tensor = this._cu.execution.invoke('torch.Tensor', []);
                const shape = Array.from(metadata.sizes);
                const stride = Array.from(metadata.strides);
                tensor.__setstate__([storage, metadata.storage_offset, shape, stride]);
                return tensor;
            }
            parseObject(ivalue) {
                const object = ivalue.val;
                const obj_type = this._module.object_types[object.type_index];
                const cls = this.getOrCreateClassTypeForObject(object);
                switch (obj_type.type) {
                    case torch.mobile.serialization.TypeType.CLASS_WITH_FIELD: {
                        const torch = this._torch;
                        const obj = torch.ScriptObject.create(cls);
                        for (let i = 0; i < object.attrs.length; i++) {
                            const attr_name = obj_type.attr_names[i];
                            const val = this._all_ivalues[object.attrs[i]];
                            obj.__setattr__(attr_name, val);
                        }
                        return obj;
                    }
                    case torch.mobile.serialization.TypeType.CUSTOM_CLASS:
                    case torch.mobile.serialization.TypeType.CLASS_WITH_SETSTATE:
                    default: {
                        throw new python.Error(`Unknown object type type '${obj_type.type}'.`);
                    }
                }
            }
            getOrCreateClassTypeForObject(object) {
                let cls = this._all_types[object.type_index];
                const obj_type = this._module.object_types[object.type_index];
                if (!cls) {
                    const name = obj_type.type_name;
                    if (name.startsWith('__torch__') || name.startsWith('torch.jit')) {
                        cls = this._cu.get_class(new torch._C.QualifiedName(name));
                        if (!cls) {
                            const torch = this._torch;
                            cls = torch.ClassType.create(name, this._cu, true);
                            this._cu.register_type(cls);
                        }
                    } else {
                        // cls = c10::parseType(qn_str).cast<ClassType>();
                    }
                    this._all_types[object.type_index] = cls;
                    if (obj_type.type === torch.mobile.serialization.TypeType.CLASS_WITH_FIELD) {
                        for (let i = 0; i < object.attrs.length; i++) {
                            // const val = this._all_ivalues[object.attrs[i]];
                            cls.addAttribute(obj_type.attr_names[i] /*, null val.type(c10::DynamicType) */);
                        }
                    }
                }
                return cls;
            }
        });
        this.registerType('torch.export.UnflattenedModule', class extends torch.nn.modules.module.Module {
            constructor(export_module, flat_args_adapter) {
                super();
                const export_graph = copy.deepcopy(export_module.graph);
                self.graph_signature = copy.deepcopy(export_module.graph_signature);
                this.graph = torch.fx.Graph();
                this.graph.owning_module = this;
                this.module_call_graph = copy.deepcopy(export_module.module_call_graph);
                this.flat_args_adapter = flat_args_adapter;
                this.adapted = false;
                // this._run_with_interpreter = RUN_WITH_INTERPRETER
                this._inplace_buffer_mutations(export_graph, this.graph_signature);
            }
        });
        this.registerType('torch.export.graph_signature.ExportGraphSignature', class {
            constructor(input_specs, output_specs) {
                this.input_specs = input_specs;
                this.output_specs = output_specs;
            }
            user_inputs() {
                const user_inputs = [];
                for (const s of this.input_specs) {
                    if (s.kind !== torch.export.graph_signature.InputKind.USER_INPUT) {
                        continue;
                    }
                    if (s.arg instanceof torch.export.graph_signature.TensorArgument ||
                        s.arg instanceof torch.export.graph_signature.SymIntArgument ||
                        s.arg instanceof torch.export.graph_signature.CustomObjArgument) {
                        user_inputs.push(s.arg.name);
                    } else if (s.arg instanceof torch.export.graph_signature.ConstantArgument) {
                        user_inputs.push(s.arg.value);
                    } else {
                        throw new python.Error(`Unsupported user input '${s.arg}'.`);
                    }
                }
                return user_inputs;
            }
            user_outputs() {
                const user_outputs = [];
                for (const s of this.output_specs) {
                    if (s.kind !== torch.export.graph_signature.OutputKind.USER_OUTPUT) {
                        continue;
                    }
                    if (s.arg instanceof torch.export.graph_signature.TensorArgument ||
                        s.arg instanceof torch.export.graph_signature.SymIntArgument ||
                        s.arg instanceof torch.export.graph_signature.CustomObjArgument) {
                        user_outputs.push(s.arg.name);
                    } else if (s.arg instanceof torch.export.graph_signature.ConstantArgument) {
                        user_outputs.push(s.arg.value);
                    } else {
                        throw new python.Error(`Unsupported user output '${s.arg}'.`);
                    }
                }
                return user_outputs;
            }
            inputs_to_parameters() {
                return new Map(this.input_specs
                    .filter((s) => s.kind === torch.export.graph_signature.InputKind.PARAMETER && s.arg instanceof torch.export.graph_signature.TensorArgument && typeof s.target === 'string')
                    .map((s) => [s.arg.name, s.target]));
            }
            inputs_to_buffers() {
                return new Map(this.input_specs
                    .filter((s) => s.kind === torch.export.graph_signature.InputKind.BUFFER && s.arg instanceof torch.export.graph_signature.TensorArgument && typeof s.target === 'string')
                    .map((s) => [s.arg.name, s.target]));
            }
            inputs_to_lifted_tensor_constants() {
                return new Map(this.input_specs
                    .filter((s) => s.kind === torch.export.graph_signature.InputKind.CONSTANT_TENSOR && s.arg instanceof torch.export.graph_signature.TensorArgument && typeof s.target === 'string')
                    .map((s) => [s.arg.name, s.target]));
            }
        });
        torch.export.graph_signature.InputKind = {
            USER_INPUT: 0,
            PARAMETER: 1,
            BUFFER: 2,
            CONSTANT_TENSOR: 3,
            CUSTOM_OBJ: 4,
            TOKEN: 5
        };
        this.registerType('torch.export.graph_signature.InputSpec', class {
            constructor(kind, arg, target, persistent) {
                this.kind = kind;
                this.arg = arg;
                this.target = target;
                this.persistent = persistent || null;
            }
        });
        torch.export.graph_signature.OutputKind = {
            USER_OUTPUT: 0,
            LOSS_OUTPUT: 1,
            BUFFER_MUTATION: 2,
            GRADIENT_TO_PARAMETER: 3,
            GRADIENT_TO_USER_INPUT: 4,
            USER_INPUT_MUTATION: 5,
            TOKEN: 6
        };
        this.registerType('torch.export.graph_signature.OutputSpec', class {
            constructor(kind, arg, target) {
                this.kind = kind;
                this.arg = arg;
                this.target = target;
            }
        });
        this.registerType('torch.export.graph_signature.ConstantArgument', class {
            constructor(name, value) {
                this.name = name;
                this.value = value; // Union[int, float, bool, str, None]
            }
        });
        this.registerType('torch.export.graph_signature.TensorArgument', class {
            constructor(name) {
                this.name = name;
            }
        });
        this.registerType('torch.export.graph_signature.SymIntArgument', class {
            constructor(name) {
                this.name = name;
            }
        });
        this.registerType('torch.export.graph_signature.CustomObjArgument', class {
            constructor(name, class_fqn, fake_val) {
                this.name = name;
                this.class_fqn = class_fqn;
                this.fake_val = fake_val;
            }
        });
        this.registerType('torch.export.exported_program.ExportedProgram', class {
            constructor(root, graph, graph_signature, state_dict, range_constraints, module_call_graph, example_inputs, verifier, tensor_constants, constants) {
                // graph._codegen = torch.fx.graph.CodeGen()
                this._graph_module = this._create_graph_module_for_export(root, graph);
                if (root instanceof torch.fx.GraphModule) {
                    // this._graph_module.meta.update(root.meta);
                }
                this._graph_signature = graph_signature;
                this._state_dict = state_dict;
                this._range_constraints = range_constraints;
                this._module_call_graph = module_call_graph;
                this._example_inputs = example_inputs;
                this._constants = tensor_constants || constants || {};
            }
            _create_graph_module_for_export(root, graph) {
                let gm = null;
                try {
                    gm = new torch.fx.GraphModule(root, graph);
                } catch {
                    const gm = new torch.fx.GraphModule(root, torch.fx.Graph());
                    gm._graph = graph;
                }
                return gm;
            }
            get graph_module() {
                return this._graph_module;
            }
            get graph() {
                return this._graph_module.graph;
            }
            get graph_signature() {
                return this._graph_signature;
            }
            get state_dict() {
                return this._state_dict;
            }
            get constants() {
                return this._constants;
            }
        });
        this.registerType('torch.export.exported_program.ModuleCallEntry', class {});
        this.registerType('torch.export.exported_program.ModuleCallSignature', class {});
        this.registerFunction('torch.export.unflatten', (module, flat_args_adapter) => {
            module = torch.export._remove_effect_tokens(module);
            return new torch.export.UnflattenedModule(module, flat_args_adapter);
        });
        this.registerFunction('torch._export.exported_program._create_graph_module_for_export', (root, graph) => {
            return new torch.fx.graph_module.GraphModule(root, graph);
        });
        this.registerType('torch._export.serde.serialize.SerializedArtifact', class {
            constructor(exported_program, state_dict, constants, example_inputs) {
                this.exported_program = exported_program;
                this.state_dict = state_dict;
                this.constants = constants;
                this.example_inputs = example_inputs;
            }
        });
        torch._export.serde.serialize._SYM_INT_OPS = new Set([
            operator.mul, operator.add, operator.sub, operator.floordiv, operator.mod,
            torch.sym_sqrt, torch.sym_int, torch.sym_ite, torch.sym_max, torch.sym_min, torch.sym_sqrt
        ]);
        torch._export.serde.serialize._SYM_BOOL_OPS = new Set([
            operator.eq, operator.ne, operator.le, operator.ge, operator.lt, operator.gt,
            torch.sym_not
        ]);
        this.registerType('torch._export.serde.union._Union', class {
            constructor(obj) {
                if (obj.$type) {
                    this.type = obj.$type;
                    this[obj.$type] = obj.$value;
                    delete obj.$type;
                    delete obj.$value;
                } else {
                    let entries = Object.entries(obj);
                    if (entries.length > 1) {
                        entries = entries.filter(([, value]) => value !== null);
                    }
                    if (entries.length !== 1) {
                        throw new Error();
                    }
                    const [entry] = entries;
                    const [type, value] = entry;
                    this.type = type;
                    this[type] = value;
                }
            }
            get value() {
                return this[this.type];
            }
        });
        this.registerType('torch._export.serde.schema.NamedArgument', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
                this.name = obj.name;
            }
        });
        this.registerType('torch._export.serde.schema.Argument', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'as_int' || this.type === 'as_ints' ||
                    this.type === 'as_float' || this.type === 'as_floats' ||
                    this.type === 'as_bool' || this.type === 'as_bools' ||
                    this.type === 'as_string' || this.type === 'as_strings' ||
                    this.type === 'as_scalar_type' || this.type === 'as_device' ||
                    this.type === 'as_memory_format' || this.type === 'as_layout') {
                    // continue
                } else if (this.type === 'as_none') {
                    this.as_none = null;
                } else if (this.type === 'as_tensor') {
                    this.as_tensor = new torch._export.serde.schema.TensorArgument(this.as_tensor);
                } else if (this.type === 'as_tensors') {
                    this.as_tensors = this.as_tensors.map((item) => new torch._export.serde.schema.TensorArgument(item));
                } else if (this.type === 'as_sym_int') {
                    this.as_sym_int = new torch._export.serde.schema.SymIntArgument(this.as_sym_int);
                } else if (this.type === 'as_sym_ints') {
                    this.as_sym_ints = this.as_sym_ints.map((item) => new torch._export.serde.schema.SymIntArgument(item));
                } else if (this.type === 'as_optional_tensors') {
                    this.as_optional_tensors = this.as_optional_tensors.map((item) => new torch._export.serde.schema.OptionalTensorArgument(item));
                } else {
                    throw new python.Error(`Unsupported argument '${this.type}'.`);
                }
                /*
                as_tensors: List[TensorArgument]
                as_string: str
                as_strings: List[str]
                as_sym_int: SymIntArgument
                as_sym_ints: List[SymIntArgument]
                as_scalar_type: ScalarType
                as_memory_format: MemoryFormat
                as_layout: Layout
                as_bools: List[bool]
                as_sym_bool: SymBoolArgument
                as_sym_bools: List[SymBoolArgument]
                as_graph: GraphArgument
                as_optional_tensors: List[OptionalTensorArgument]
                as_custom_obj: CustomObjArgument
                */
            }
        });
        this.registerType('torch._export.serde.schema.Node', class {
            constructor(obj) {
                this.target = obj.target;
                this.inputs = obj.inputs.map((input) => new torch._export.serde.schema.NamedArgument(input));
                this.outputs = obj.outputs.map((output) => new torch._export.serde.schema.Argument(output));
                this.metadata = new Map(Object.entries(obj.metadata));
            }
        });
        torch._export.serde.schema.ScalarType = {
            UNKNOWN: 0,
            BYTE: 1,
            CHAR: 2,
            SHORT: 3,
            INT: 4,
            LONG: 5,
            HALF: 6,
            FLOAT: 7,
            DOUBLE: 8,
            COMPLEXHALF: 9,
            COMPLEXFLOAT: 10,
            COMPLEXDOUBLE: 11,
            BOOL: 12,
            BFLOAT16: 13
        };
        torch._export.serde.schema.Layout = {
            Unknown: 0,
            SparseCoo: 1,
            SparseCsr: 2,
            SparseCsc: 3,
            SparseBsr: 4,
            SparseBsc: 5,
            _mkldnn: 6,
            Strided: 7
        };
        torch._export.serde.schema.MemoryFormat = {
            Unknown: 0,
            ContiguousFormat: 1,
            ChannelsLast: 2,
            ChannelsLast3d: 3,
            PreserveFormat: 4,
        };
        this.registerType('torch._export.serde.schema.Device', class {
            constructor(obj) {
                Object.assign(this, { ...obj });
            }
        });
        this.registerType('torch._export.serde.schema.TensorMeta', class {
            constructor(obj) {
                obj = obj.meta || obj;
                this.dtype = obj.dtype;
                this.sizes = obj.sizes.map((size) => new torch._export.serde.schema.SymInt(size));
                this.requires_grad = obj.requires_grad;
                this.device = obj.device;
                this.strides = obj.strides.map((stride) => new torch._export.serde.schema.SymInt(stride));
                this.storage_offset = new torch._export.serde.schema.SymInt(Number.isInteger(obj.storage_offset) ? { as_int: obj.storage_offset } : obj.storage_offset);
                this.layout = obj.layout;
            }
        });
        this.registerType('torch._export.serde.schema.Graph', class {
            constructor(obj) {
                this.inputs = obj.inputs.map((input) => new torch._export.serde.schema.Argument(input));
                this.outputs = obj.outputs.map((output) => new torch._export.serde.schema.Argument(output));
                this.nodes = obj.nodes.map((node) => new torch._export.serde.schema.Node(node));
                this.tensor_values = new Map(Object.entries(obj.tensor_values).map(([key, value]) => [key, new torch._export.serde.schema.TensorMeta(value)]));
                this.sym_int_values = new Map(Object.entries(obj.sym_int_values).map(([key, value]) => [key, new torch._export.serde.schema.SymInt(value)]));
                this.sym_bool_values = new Map(Object.entries(obj.sym_bool_values).map(([key, value]) => [key, new torch._export.serde.schema.SymBool(value)]));
                this.is_single_tensor_return = obj.is_single_tensor_return;
                this.custom_obj_values = new Map(Object.entries(obj.custom_obj_values || {}).map(([key, value]) => [key, new torch._export.serde.schema.CustomObjArgument(value)]));
                if (obj.contants) {
                    // this.constants = new Map(Object.entries(serialized_graph.constants).map(([k, v]) => [k, torch.load(v)]));
                    // graph_signature -> input_specs -> tensor_constant
                }
            }
        });
        this.registerType('torch._export.serde.schema.ModuleCallSignature', class {
            constructor(obj) {
                Object.assign(this, { ...obj });
                this.inputs = this.inputs.map((item) => new torch._export.serde.schema.Argument(item));
                this.outputs = this.outputs.map((item) => new torch._export.serde.schema.Argument(item));
            }
        });
        this.registerType('torch._export.serde.schema.ModuleCallEntry', class {
            constructor(obj) {
                Object.assign(this, { ...obj });
                this.signature = this.signature ? new torch._export.serde.schema.ModuleCallSignature(this.signature) : null;
            }
        });
        this.registerType('torch._export.serde.schema.GraphModule', class {
            constructor(obj) {
                this.graph = new torch._export.serde.schema.Graph(obj.graph);
                this.signature = new torch._export.serde.schema.GraphSignature(obj.signature);
                this.module_call_graph = obj.module_call_graph.map((item) => new torch._export.serde.schema.ModuleCallEntry(item));
                this.metadata = new Map(Object.entries(obj.metadata || {}));
            }
        });
        this.registerType('torch._export.serde.schema.ExportedProgram', class {
            constructor(obj) {
                Object.assign(this, { ...obj });
                this.graph_module = new torch._export.serde.schema.GraphModule(obj.graph_module);
            }
        });
        this.registerType('torch._export.serde.schema.SymExprHint', class extends torch._export.serde.union._Union {});
        this.registerType('torch._export.serde.schema.SymExpr', class {
            constructor(obj) {
                this.expr_str = obj.expr_str;
                this.hint = obj.hint ? new torch._export.serde.schema.SymExprHint(obj.hint) : null;
            }
        });
        this.registerType('torch._export.serde.schema.SymInt', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'as_int') {
                    // continue
                } else if (this.type === 'as_expr') {
                    this.as_expr = new torch._export.serde.schema.SymExpr(this.as_expr);
                } else {
                    throw new python.Error(`Unsupported symbolic int '${this.type}'.`);
                }
            }
        });
        this.registerType('torch._export.serde.schema.SymIntArgument', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                Object.assign(this, { ...obj });
            }
        });
        this.registerType('torch._export.serde.schema.SymBool', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'as_bool') {
                    // continue
                } else if (this.type === 'as_expr') {
                    this.as_expr = new torch._export.serde.schema.SymExpr(this.as_expr);
                } else {
                    throw new python.Error(`Unsupported symbolic bool '${this.type}'.`);
                }
            }
        });
        this.registerType('torch._export.serde.schema.SymBoolArgument', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                Object.assign(this, { ...obj });
            }
        });
        this.registerType('torch._export.serde.schema.GraphSignature', class {
            constructor(obj) {
                this.input_specs = [];
                if (Array.isArray(obj.input_specs)) {
                    this.input_specs = obj.input_specs.map((input_spec) => new torch._export.serde.schema.InputSpec(input_spec));
                }
                if (Array.isArray(obj.user_inputs)) {
                    for (const user_input of obj.user_inputs) {
                        this.input_specs.push(new torch._export.serde.schema.InputSpec({ user_input: { arg: { as_string: user_input } } }));
                    }
                }
                if (obj.inputs_to_parameters) {
                    for (const [input, parameter_name] of Object.entries(obj.inputs_to_parameters)) {
                        this.input_specs.push(new torch._export.serde.schema.InputSpec({ parameter: { arg: { name: input }, parameter_name } }));
                    }
                }
                this.output_specs = [];
                if (Array.isArray(obj.output_specs)) {
                    this.output_specs = obj.output_specs.map((output_spec) => new torch._export.serde.schema.OutputSpec(output_spec));
                }
            }
        });
        this.registerType('torch._export.serde.schema.UserInputSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
            }
        });
        this.registerType('torch._export.serde.schema.InputToParameterSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.TensorArgument(obj.arg);
                this.parameter_name = obj.parameter_name;
            }
        });
        this.registerType('torch._export.serde.schema.InputToBufferSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.TensorArgument(obj.arg);
                this.buffer_name = obj.buffer_name;
            }
        });
        this.registerType('torch._export.serde.schema.InputToTensorConstantSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.TensorArgument(obj.arg);
                this.tensor_constant_name = obj.tensor_constant_name;
            }
        });
        this.registerType('torch._export.serde.schema.InputToConstantInputSpec', class {
            constructor(obj) {
                this.name = obj.name;
                this.value = new torch._export.serde.schema.ConstantValue(obj.value);
            }
        });
        this.registerType('torch._export.serde.schema.ConstantValue', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'as_int' || this.type === 'as_float' || this.type === 'as_bool' || this.type === 'as_string' || this.type === 'as_strings') {
                    // continue
                } else if (this.type === 'as_none') {
                    this.as_none = null;
                } else {
                    throw new python.Error(`Unsupported constant value type '${this.type}'.`);
                }
            }
        });
        this.registerType('torch._export.serde.schema.InputSpec', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'user_input') {
                    this.user_input = new torch._export.serde.schema.UserInputSpec(this.user_input);
                } else if (this.type === 'parameter') {
                    this.parameter = new torch._export.serde.schema.InputToParameterSpec(this.parameter);
                } else if (this.type === 'buffer') {
                    this.buffer = new torch._export.serde.schema.InputToBufferSpec(this.buffer);
                } else if (this.type === 'tensor_constant') {
                    this.tensor_constant = new torch._export.serde.schema.InputToTensorConstantSpec(this.tensor_constant);
                } else if (this.type === 'constant_input') {
                    this.constant_input = new torch._export.serde.schema.InputToConstantInputSpec(this.constant_input);
                } else {
                    throw new python.Error(`Unsupported input spec type '${this.type}'.`);
                }
                /*
                custom_obj: InputToCustomObjSpec
                token: InputTokenSpec
                */
            }
        });
        this.registerType('torch._export.serde.schema.UserOutputSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
            }
        });
        this.registerType('torch._export.serde.schema.BufferMutationSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
                this.buffer_name = obj.buffer_name;
            }
        });
        this.registerType('torch._export.serde.schema.GradientToParameterSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
                this.parameter_name = obj.parameter_name;
            }
        });
        this.registerType('torch._export.serde.schema.GradientToUserInputSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
                this.user_input_name = obj.user_input_name;
            }
        });
        this.registerType('torch._export.serde.schema.UserInputMutationSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.Argument(obj.arg);
                this.user_input_name = obj.user_input_name;
            }
        });
        this.registerType('torch._export.serde.schema.OutputTokenSpec', class {
            constructor(obj) {
                this.arg = new torch._export.serde.schema.TokenArgument(obj.arg);
            }
        });
        this.registerType('torch._export.serde.schema.OutputSpec', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'user_output') {
                    this.user_output = new torch._export.serde.schema.UserOutputSpec(this.user_output);
                } else if (this.type === 'loss_output') {
                    this.loss_output = new torch._export.serde.schema.LossOutputSpec(this.loss_output);
                } else if (this.type === 'buffer_mutation') {
                    this.buffer_mutation = new torch._export.serde.schema.BufferMutationSpec(this.buffer_mutation);
                } else if (this.type === 'gradient_to_parameter') {
                    this.gradient_to_parameter = new torch._export.serde.schema.GradientToParameterSpec(this.gradient_to_parameter);
                } else if (this.type === 'gradient_to_user_input') {
                    this.gradient_to_user_input = new torch._export.serde.schema.GradientToUserInputSpec(this.gradient_to_user_input);
                } else if (this.type === 'user_input_mutation') {
                    this.user_input_mutation = new torch._export.serde.schema.UserInputMutationSpec(this.user_input_mutation);
                } else if (this.type === 'token') {
                    this.token = new torch._export.serde.schema.OutputTokenSpec(this.token);
                }
            }
        });
        this.registerType('torch._export.serde.schema.TensorArgument', class {
            constructor(obj) {
                this.name = obj.name;
            }
        });
        this.registerType('torch._export.serde.schema.TokenArgument', class {
            constructor(obj) {
                this.name = obj.name;
            }
        });
        this.registerType('torch._export.serde.schema.OptionalTensorArgument', class extends torch._export.serde.union._Union {
            constructor(obj) {
                super(obj);
                if (this.type === 'as_tensor') {
                    this.as_tensor = new torch._export.serde.schema.TensorArgument({ name: this.as_tensor });
                } else if (this.type === 'as_none') {
                    this.as_none = null;
                } else {
                    throw new python.Error(`Unsupported optional tensor argument '${this.type}'.`);
                }
            }
        });
        this.registerFunction('torch._export.load', (f, expected_opset_version) => {
            const serialized_exported_program = f.get('serialized_exported_program.json');
            const serialized_state_dict = f.get('serialized_state_dict.pt');
            const serialized_constants = f.get('serialized_constants.pt');
            const serialized_example_inputs = f.get('serialized_example_inputs.pt');
            const artifact = new torch._export.serde.serialize.SerializedArtifact(serialized_exported_program, serialized_state_dict, serialized_constants, serialized_example_inputs);
            return torch._export.serde.serialize.deserialize(artifact, expected_opset_version);
        });
        this.registerFunction('torch._export.serde.serialize._dict_to_dataclass', (cls, data) => {
            if (data === null) {
                return data;
            }
            if (cls) {
                return new cls(data);
            }
            throw new python.Error(`Unsupported data class '${cls.__name__}'.`);
        });
        this.registerFunction('torch._export.serde.serialize.deserialize', (artifact, expected_opset_version) => {
            const serialized_exported_program = torch._export.serde.serialize._dict_to_dataclass(torch._export.serde.schema.ExportedProgram, artifact.exported_program);
            return new torch._export.serde.serialize.ExportedProgramDeserializer(expected_opset_version).deserialize(serialized_exported_program, artifact.state_dict, artifact.constants, artifact.example_inputs);
        });
        this.registerType('torch._export.serde.serialize.ExportedProgramDeserializer', class {
            constructor(expected_opset_version) {
                this.expected_opset_version = expected_opset_version;
            }
            deserialize(exported_program, state_dict, constants, example_inputs) {
                const symbol_name_to_range = new Map(Object.entries(exported_program.range_constraints));
                /*
                symbol_name_to_range = {
                    k: symbolic_shapes.ValueRanges(_int_to_sympy_int(v.min_val), _int_to_sympy_int(v.max_val))
                    for k, v in exported_program.range_constraints.items()
                }
                */
                const deserializer = new torch._export.serde.serialize.GraphModuleDeserializer();
                const res = deserializer.deserialize(
                    exported_program.graph_module,
                    state_dict,
                    constants,
                    example_inputs,
                    symbol_name_to_range);
                const range_constraints = null;
                /*
                range_constraints = self.deserialize_range_constraints(
                    symbol_name_to_range, res.names_to_symbols,
                )
                model_opset_version: Optional[Dict[str, int]] = serialized_artifact.exported_program.opset_version
                self._validate_model_opset_version(model_opset_version)
                upgrader = GraphModuleOpUpgrader(self.expected_opset_version, model_opset_version)
                */
                return new torch.export.exported_program.ExportedProgram(
                    res.graph_module, res.graph_module.graph, res.signature,
                    res.state_dict, range_constraints, res.module_call_graph, res.example_inputs,
                    null, // verifier=load_verifier(serialized_artifact.exported_program.dialect),
                    res.constants);
                // return upgrader.upgrade(exported_program)
            }
        });
        this.registerFunction('torch._export.serde.serialize.deserialize_torch_artifact', (serialized) => {
            if (!serialized) {
                return new builtins.dict();
            }
            const artifact = torch.load(serialized);
            return artifact;
        });
        this.registerType('torch._export.serde.serialize.GraphModuleDeserializer', class {
            constructor() {
                this.serialized_name_to_node = new Map();
                this.serialized_name_to_meta = new Map();
                this.graph = new torch.fx.Graph();
                this.module = new torch.nn.Module();
            }
            deserialize_graph_output(output) {
                if (output.type === 'as_tensor') {
                    return this.serialized_name_to_node.get(output.as_tensor.name);
                } else if (output.type === 'as_sym_int') {
                    return this.serialized_name_to_node.get(output.as_sym_int.as_name);
                } else if (output.type === 'as_sym_bool') {
                    return this.serialized_name_to_node.get(output.as_sym_bool.as_name);
                } else if (output.type === 'as_int') {
                    return this.serialized_name_to_node.get(output.as_int.as_name);
                } else if (output.type === 'as_none') {
                    return this.serialized_name_to_node.get(output.as_sym_bool.as_name);
                }
                throw new python.Error(`Unsupported graph node ${output.type}.`);
            }
            deserialize_graph(serialized_graph) {
                for (const [name, tensor_value] of serialized_graph.tensor_values) {
                    const meta_val = this.deserialize_tensor_meta(tensor_value.meta || tensor_value, this.fake_tensor_mode);
                    this.serialized_name_to_meta.set(name, meta_val);
                }
                for (const [name, sym_int_value] of serialized_graph.sym_int_values) {
                    this.serialized_name_to_meta.set(name, this.deserialize_sym_int(sym_int_value));
                }
                for (const [name, sym_bool_value] of serialized_graph.sym_bool_values) {
                    this.serialized_name_to_meta.set(name, this.deserialize_sym_bool(sym_bool_value));
                }
                for (const [name, script_obj_meta] of serialized_graph.custom_obj_values) {
                    this.serialized_name_to_meta.set(name, this.deserialize_script_obj_meta(script_obj_meta));
                }
                for (let i = 0; i < serialized_graph.inputs.length; i++) {
                    const input = serialized_graph.inputs[i];
                    if (input.type === 'as_tensor' || input.type === 'as_sym_int' || input.type === 'as_custom_obj') {
                        const node_name = input.value.name;
                        const placeholder_node = this.graph.placeholder(node_name);
                        placeholder_node.name = node_name;
                        this.sync_fx_node(node_name, placeholder_node);
                    } else if (input.type === 'as_int' || input.type === 'as_float' || input.type === 'as_bool' || input.type === 'as_none' || input.type === 'as_string') {
                        const node_name = this.signature.input_specs[i].arg.name;
                        const placeholder_node = this.graph.placeholder(node_name);
                        placeholder_node.meta.set('val', this.deserialize_input(input));
                    } else {
                        throw new python.Error(`Invalid input ${input.type}.`);
                    }
                }
                for (const serialized_node of serialized_graph.nodes) {
                    const target = this.deserialize_operator(serialized_node.target);
                    this.deserialize_node(serialized_node, target);
                }
                let outputs = [];
                for (const output of serialized_graph.outputs) {
                    outputs.push(this.deserialize_graph_output(output));
                }
                if (serialized_graph.is_single_tensor_return) {
                    [outputs] = outputs;
                } else {
                    outputs = new builtins.tuple(outputs);
                }
                const output_node = this.graph.output(outputs);
                if (serialized_graph.is_single_tensor_return) {
                    output_node.meta.set("val", output_node.args[0].meta.get('val'));
                } else {
                    /* output_node.meta["val"] = tuple(
                        arg.meta["val"] if isinstance(arg, torch.fx.Node) else arg
                        for arg in output_node.args[0]
                    ) */
                }
                return self.graph;
            }
            deserialize_operator(serialized_target) {
                let module = null;
                let serialized_target_names = null;
                if (serialized_target.startsWith('_operator')) {
                    module = operator;
                    serialized_target_names = serialized_target.split(".").slice(1);
                } else if (serialized_target.startsWith('torch')) {
                    module = torch;
                    serialized_target_names = serialized_target.split(".").slice(1);
                } else if (serialized_target.startsWith('#')) {
                    return self.deserialize_extension_operator(serialized_target);
                } else {
                    return serialized_target;
                }
                let target = module;
                for (const name of serialized_target_names) {
                    target = builtins.getattr(target, name);
                    if (!target) {
                        return serialized_target;
                    }
                }
                return target;
            }
            deserialize_node(serialized_node, target) {
                let fx_node = null;
                if (torch._export.serde.serialize._SYM_BOOL_OPS.has(target) || torch._export.serde.serialize._SYM_INT_OPS.has(target)) {
                    const name = serialized_node.outputs[0].value.as_name;
                    const args = this.deserialize_sym_op_inputs(serialized_node.inputs);
                    fx_node = this.graph.create_node('call_function', target, args, null, name);
                    this.deserialize_sym_op_outputs(serialized_node, fx_node);
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
                    fx_node = this.graph.create_node('call_function', target, args, kwargs, name);
                    this.deserialize_outputs(serialized_node, fx_node);
                } else {
                    throw new python.Error(`Unsupported node target type '${target}'.`);
                }
                fx_node.meta.update(this.deserialize_metadata(serialized_node.metadata));
                if (fx_node.op !== 'placeholder' && fx_node.op !== 'output' && !fx_node.meta.has('nn_module_stack')) {
                    fx_node.meta.set('nn_module_stack', new builtins.dict());
                }
            }
            deserialize_input_spec(i) {
                if (i.type === 'user_input') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.USER_INPUT,
                        this.deserialize_argument_spec(i.user_input.arg),
                        null);
                } else if (i.type === 'parameter') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.PARAMETER,
                        new torch.export.graph_signature.TensorArgument(i.parameter.arg.name),
                        i.parameter.parameter_name,
                    );
                } else if (i.type === 'buffer') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.BUFFER,
                        new torch.export.graph_signature.TensorArgument(i.buffer.arg.name),
                        i.buffer.buffer_name,
                        i.buffer.persistent,
                    );
                } else if (i.type === 'tensor_constant') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.CONSTANT_TENSOR,
                        new torch.export.graph_signature.TensorArgument(i.tensor_constant.arg.name),
                        i.tensor_constant.tensor_constant_name);
                } else if (i.type === 'custom_obj') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.CUSTOM_OBJ,
                        new torch.export.graph_signature.CustomObjArgument(i.custom_obj.arg.name, i.custom_obj.arg.class_fqn),
                        i.custom_obj.custom_obj_name);
                } else if (i.type === 'token') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.TOKEN,
                        new torch.export.graph_signature.TokenArgument(i.token.arg.name),
                        null);
                } else if (i.type === 'constant_input') {
                    return new torch.export.graph_signature.InputSpec(
                        torch.export.graph_signature.InputKind.USER_INPUT,
                        new torch.export.graph_signature.ConstantArgument(i.constant_input.name, this.deserialize_constant_input(i.constant_input.value)),
                        null);
                }
                throw new python.Error(`Unknown input spec ${i}`);
            }
            deserialize_constant_input(inp) {
                if (inp.type === 'as_int') {
                    return inp.as_int;
                } else if (inp.type === 'as_float') {
                    return inp.as_float;
                } else if (inp.type === 'as_string') {
                    return inp.as_string;
                } else if (inp.type === 'as_bool') {
                    return inp.as_bool;
                } else if (inp.type === 'as_none') {
                    return null;
                }
                throw new python.Error(`Unhandled constant argument ${inp} to deserialize.`);
            }
            deserialize_output_spec(o) {
                if (o.type === 'user_output') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.USER_OUTPUT,
                        this.deserialize_argument_spec(o.user_output.arg),
                        null);
                } else if (o.type === 'loss_output') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.LOSS_OUTPUT,
                        new torch.export.graph_signature.TensorArgument(o.loss_output.arg.name),
                        null);
                } else if (o.type === 'buffer_mutation') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.BUFFER_MUTATION,
                        new torch.export.graph_signature.TensorArgument(o.buffer_mutation.arg.name),
                        o.buffer_mutation.buffer_name);
                } else if (o.type === 'gradient_to_parameter') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.GRADIENT_TO_PARAMETER,
                        new torch.export.graph_signature.TensorArgument(o.gradient_to_parameter.arg.name),
                        o.gradient_to_parameter.parameter_name);
                } else if (o.type === 'gradient_to_user_input') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.GRADIENT_TO_USER_INPUT,
                        new torch.export.graph_signature.TensorArgument(o.gradient_to_user_input.arg.name),
                        o.gradient_to_user_input.user_input_name);
                } else if (o.type === 'user_input_mutation') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.USER_INPUT_MUTATION,
                        new torch.export.graph_signature.TensorArgument(o.user_input_mutation.arg.name),
                        o.user_input_mutation.user_input_name);
                } else if (o.type === 'token') {
                    return new torch.export.graph_signature.OutputSpec(
                        torch.export.graph_signature.OutputKind.TOKEN,
                        new torch.export.graph_signature.TokenArgument(o.token.arg.name),
                        null);
                }
                throw new python.Error(`Unknown output spec ${o}.`);
            }
            deserialize_signature(sig) {
                return new torch.export.graph_signature.ExportGraphSignature(
                    sig.input_specs.map((i) => this.deserialize_input_spec(i)),
                    sig.output_specs.map((o) => this.deserialize_output_spec(o)));
            }
            deserialize(serialized_graph_module, serialized_state_dict, constants, example_inputs, symbol_name_to_range) {
                this.shape_env = new torch.fx.experimental.symbolic_shapes.ShapeEnv(/* assume_static_by_default = True */);
                /*
                this.fake_tensor_mode = FakeTensorMode(
                    allow_fallback_kernels=False,
                    allow_non_fake_inputs=True,
                    shape_env=this.shape_env,
                )
                */
                this.symbol_name_to_symbol = new Map();
                this.constants = torch._export.serde.serialize.deserialize_torch_artifact(constants);
                this.signature = this.deserialize_signature(serialized_graph_module.signature);
                this.symbol_name_to_range = symbol_name_to_range || new Map();
                /*
                    if symbol_name_to_range:
                    for k, vr in symbol_name_to_range.items():
                        lower = int(vr.lower)
                        if vr.upper >= 2:  # max is >= 2, not sym bool range
                            lower = max(2, lower)
                        this.symbol_name_to_range[k] = symbolic_shapes.ValueRanges(_int_to_sympy_int(lower), vr.upper)
                    */
                this.example_inputs = null;
                if (example_inputs && example_inputs.length > 0) {
                    torch._export.serde.serialize.deserialize_torch_artifact(example_inputs);
                }
                this.deserialize_graph(serialized_graph_module.graph);
                const module_call_graph = null; // this.deserialize_module_call_graph(serialized_graph_module.module_call_graph)
                return {
                    graph_module: torch._export.exported_program._create_graph_module_for_export(this.module, this.graph),
                    signature: this.signature,
                    module_call_graph,
                    names_to_symbols: this.symbol_name_to_symbol,
                    state_dict: torch._export.serde.serialize.deserialize_torch_artifact(serialized_state_dict),
                    constants: this.constants,
                    example_inputs: this.example_inputs,
                };
            }
            sync_fx_node(name, fx_node) {
                if (this.serialized_name_to_node.has(name)) {
                    throw new python.Error(`Node ${name} has already been deserialized before.`);
                }
                this.serialized_name_to_node.set(name, fx_node);
                fx_node.meta.set('val', this.serialized_name_to_meta.get(name));
            }
            deserialize_sym_op_inputs(inputs) {
                return inputs.map((input) => this.deserialize_input(input.arg));
            }
            deserialize_inputs(target, serialized_node) {
                const schema_args = this._get_schema_from_target(target).arguments;
                const actual_args = new Map(serialized_node.inputs.map((input) => [input.name, this.deserialize_input(input.arg)]));
                const args = new builtins.list();
                const kwargs = new builtins.dict();
                for (const schema_arg of schema_args) {
                    const is_positional = !schema_arg.has_default_value() && !schema_arg.kwarg_only;
                    if (is_positional) {
                        args.push(actual_args.get(schema_arg.name));
                    } else if (actual_args.has(schema_arg.name)) {
                        kwargs.set(schema_arg.name, actual_args.get(schema_arg.name));
                    }
                }
                return [args, kwargs];
            }
            deserialize_input(inp) {
                const value = inp.value;
                const typ_ = inp.type;
                if (typ_ === 'as_none') {
                    return null;
                } else if (typ_ === 'as_tensor') {
                    return this.serialized_name_to_node.get(inp.as_tensor.name);
                } else if (typ_ === 'as_scalar_type') {
                    return torch._export.serde.serialize._SERIALIZE_TO_TORCH_DTYPE[inp.as_scalar_type];
                } else if (typ_ === 'as_memory_format') {
                    return torch._export.serde.serialize._SERIALIZE_TO_TORCH_MEMORY_FORMAT[inp.as_memory_format];
                } else if (typ_ === 'as_layout') {
                    return torch._export.serde.serialize._SERIALIZE_TO_TORCH_LAYOUT[inp.as_layout];
                } else if (typ_ === 'as_graph') {
                    /* assert isinstance(value, GraphArgument)
                    with this.save_graph_module():
                        this.deserialize_graph(value.graph)
                        submodule = ep._create_graph_module_for_export(this.module, this.graph)
                    this.module.register_module(value.name, submodule)
                    return this.graph.create_node(
                        'get_attr',
                        value.name,
                        name=value.name,
                    )*/
                } else if (typ_ === 'as_device') {
                    return this.deserialize_device(inp.as_device);
                } else if (typ_ === 'as_int') {
                    return inp.as_int;
                } else if (typ_ === 'as_float') {
                    return inp.as_float;
                } else if (typ_ === 'as_bool') {
                    return inp.as_bool;
                } else if (typ_ === 'as_string') {
                    return inp.as_string;
                } else if (typ_ === 'as_sym_int') {
                    return this.deserialize_sym_argument(inp.as_sym_int);
                } else if (typ_ === 'as_sym_bool') {
                    return this.deserialize_sym_argument(inp.as_sym_bool);
                } else if (Array.isArray(value)) {
                    if (value.length === 0) {
                        return [];
                    } else if (typ_ === 'as_tensors') {
                        const result = [];
                        for (const arg of value) {
                            result.push(this.serialized_name_to_node.get(arg.name));
                        }
                        return result;
                    } else if (typ_ === 'as_ints' || typ_ === 'as_floats' || typ_ === 'as_bools' || typ_ ===  'as_strings') {
                        return Array.from(value);
                    } else if (typ_ === 'as_sym_ints' || typ_ === 'as_sym_bools') {
                        return value.map((arg) => this.deserialize_sym_argument(arg));
                    } else if (typ_ === 'as_optional_tensors') {
                        const deserialize_optional_tensor_args = (a) => {
                            if (a.type === 'as_none') {
                                return null;
                            } else if (a.type === 'as_tensor') {
                                return this.serialized_name_to_node.get(a.value.name);
                            }
                            throw new python.Error(`Unsupported argument '${typ_}'.`);
                        };
                        return value.map((item) => deserialize_optional_tensor_args(item));
                    }
                    throw new python.Error(`Unsupported argument '${typ_}'.`);
                } else if (typ_ === 'as_custom_obj') {
                    if (this.serialized_name_to_node.has(inp.as_custom_obj.name)) {
                        return this.serialized_name_to_node.get(inp.as_custom_obj.name);
                    }
                    return this.constants[inp.as_custom_obj.name];
                } else if (typ_ === 'as_operator') {
                    return this.deserialize_operator(inp.as_operator);
                }
                throw new python.Error(`Unsupported argument '${typ_}'.`);
            }
            deserialize_sym_argument(sym_arg) {
                if (sym_arg instanceof torch._export.serde.schema.SymIntArgument) {
                    if (sym_arg.type === 'as_int') {
                        return sym_arg.as_int;
                    } else if (sym_arg.type === 'as_name') {
                        return this.serialized_name_to_node.get(sym_arg.as_name);
                    }
                } else if (sym_arg instanceof torch._export.serde.schema.SymBoolArgument) {
                    if (sym_arg.type === 'as_bool') {
                        return sym_arg.as_bool;
                    } else if (sym_arg.type === 'as_name') {
                        return self.serialized_name_to_node.get(sym_arg.as_name);
                    }
                }
                throw new python.Error(`Unsupported symbolic argument type '${sym_arg.type}`);
            }
            deserialize_sym_op_outputs(serialized_node, fx_node) {
                this.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node);
            }
            deserialize_outputs(serialized_node, fx_node) {
                if (serialized_node.outputs.length === 0) {
                    return;
                }
                if (serialized_node.outputs.length === 1 &&
                    serialized_node.outputs[0].type === 'as_tensor') {
                    this.sync_fx_node(serialized_node.outputs[0].as_tensor.name, fx_node);
                    return;
                } else if (serialized_node.outputs.length === 1 &&
                     (serialized_node.outputs[0].value instanceof torch._export.serde.schema.SymIntArgument ||
                      serialized_node.outputs[0].value instanceof torch._export.serde.schema.SymBoolArgument)) {
                    this.sync_fx_node(serialized_node.outputs[0].value.as_name, fx_node);
                    return;
                }
                this.deserialize_multiple_outputs(serialized_node, fx_node);
            }
            deserialize_multiple_outputs(serialized_node, fx_node) {
                const deserialized_metadata = this.deserialize_metadata(serialized_node.metadata);
                const generate_getitem = (meta_val, fx_node, arg, idx) => {
                    let name = '';
                    if (arg instanceof torch._export.serde.schema.TensorArgument) {
                        name = arg.name;
                    } else if (arg instanceof torch._export.serde.schema.SymIntArgument) {
                        name = arg.as_name;
                    } else {
                        throw new python.Error(`Unsupported argument type '${arg}'.`);
                    }
                    const individual_output = this.graph.create_node(
                        'call_function',
                        operator.getitem,
                        new builtins.tuple([fx_node, idx]),
                        null,
                        name,
                    );
                    this.sync_fx_node(name, individual_output);
                    meta_val.push(this.serialized_name_to_meta.get(name));
                    individual_output.meta.update(deserialized_metadata);
                };
                const generate_getitems = (meta_val, fx_node, args) => {
                    for (let idx = 0; idx < args.length; idx++) {
                        let arg = args[idx];
                        if (arg instanceof torch._export.serde.schema.Argument) {
                            arg = arg.value;
                        }
                        if (arg instanceof torch._export.serde.schema.TensorArgument || arg instanceof torch._export.serde.schema.SymIntArgument) {
                            generate_getitem(meta_val, fx_node, arg, idx);
                        } else if (Array.isArray(arg)) { // arg instanceof (list, tuple))
                            const list_output = this.graph.create_node(
                                'call_function',
                                operator.getitem,
                                (fx_node, idx),
                            );
                            meta_val.append([]);
                            generate_getitems(meta_val[-1], list_output, arg);
                            list_output.meta.update(deserialized_metadata);
                            list_output.meta.set('val', meta_val[-1]);
                        } else {
                            throw new python.Error(`Unsupported node output type: '${arg}'.`);
                        }
                    }
                };
                const meta_val = [];
                if (serialized_node.outputs.length === 1) {
                    // assert isinstance(serialized_node.outputs[0].value, list)
                    // assert isinstance(serialized_node.outputs[0].value[0], TensorArgument)
                    generate_getitems(meta_val, fx_node, serialized_node.outputs[0].as_tensors);
                } else {
                    generate_getitems(meta_val, fx_node, serialized_node.outputs);
                }
                fx_node.meta.set('val', new builtins.tuple(meta_val));
                this.serialized_name_to_node.set(fx_node.name, fx_node);
            }
            deserialize_metadata(metadata) {
                const ret = new builtins.dict();
                const stack_trace = metadata.get('stack_trace');
                if (stack_trace) {
                    ret.set('stack_trace', stack_trace);
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
                const nn_module_stack_str = metadata.get('nn_module_stack');
                if (nn_module_stack_str) {
                    const import_nn_module_stack = (key, path, ty) => {
                        return [key, [path, ty]];
                    };
                    const nn_module_stack = new Map(nn_module_stack_str.split(';').map((item) => import_nn_module_stack(...item.split(','))));
                    ret.set('nn_module_stack', nn_module_stack);
                }
                const source_fn_st_str = metadata.get('source_fn_stack');
                if (source_fn_st_str) {
                    const source_fn_st = [];
                    for (const source_fn_str of source_fn_st_str.split(';')) {
                        const [name, target_str] = source_fn_str.split(',');
                        source_fn_st.push([name, deserialize_meta_func(target_str)]);
                    }
                    ret.set('source_fn_stack', source_fn_st);
                }
                const torch_fn = metadata.get('torch_fn');
                if (torch_fn) {
                    ret.set('torch_fn', new builtins.tuple(torch_fn.split(';')));
                }
                const custom_str = metadata.get('custom');
                if (custom_str) {
                    ret.set('custom', JSON.parse(custom_str));
                }
                return ret;
            }
            deserialize_argument_spec(x) {
                if (x.type === 'as_tensor') {
                    return new torch.export.graph_signature.TensorArgument(x.as_tensor.name);
                } else if (x.type === 'as_sym_int') {
                    return new torch.export.graph_signature.SymIntArgument(x.as_sym_int.as_name);
                } else if (x.type === 'as_custom_obj') {
                    return new torch.export.graph_signature.ConstantArgument(x.as_custom_obj.name, this.deserialize_input(x));
                }
                return new torch.export.graph_signature.ConstantArgument('', this.deserialize_input(x));
            }
            deserialize_tensor_meta(tensor_meta) {
                const sizes = tensor_meta.sizes.map((val) => this.deserialize_sym_int(val));
                const strides = tensor_meta.strides.map((val) => this.deserialize_sym_int(val));
                const device = this.deserialize_device(tensor_meta.device);
                const dtype = torch._export.serde.serialize._SERIALIZE_TO_TORCH_DTYPE[tensor_meta.dtype];
                return torch.empty_strided(sizes, strides, dtype, null, device);
            }
            deserialize_sym_int(s) {
                if (s.as_expr !== undefined && s.as_expr !== null) {
                    let sym = {};
                    if (this.symbol_name_to_symbol.has(s.as_expr.expr_str)) {
                        sym = this.symbol_name_to_symbol.get(s.as_expr.expr_str);
                    } else {
                        sym = {};
                        /*
                        sym = sympy.sympify(val.expr_str, locals=this.symbol_name_to_symbol)
                        if isinstance(sym, sympy.Symbol) {
                            this.symbol_name_to_symbol[val.expr_str] = sym
                            if vr := this.symbol_name_to_range.get(val.expr_str):
                                symbolic_shapes._constrain_symbol_range(
                                    this.shape_env,
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
                    if (hint && (hint.$type === 'as_int' || hint.as_int !== undefined)) {
                        return this.deserialize_sym_int(hint);
                    }
                    return this.shape_env.create_symintnode(sym, hint);
                } else if (s.as_int !== undefined && s.as_int !== null) {
                    return s.as_int;
                } else if (s.$type === 'as_int') {
                    return s.$value;
                }
                throw new python.Error('SymInt has invalid field type.');
            }
            deserialize_device(d) {
                if (d.index === null) {
                    return new torch.device(d.type);
                }
                return new torch.device(d.type, d.index);
            }
            _get_schema_from_target(target) {
                if (target instanceof torch._ops.OpOverload) {
                    return target._schema;
                }
                throw new python.Error(`Unsupported schema '${target.name}'.`);
            }
            _is_single_tensor_return(target) {
                const schema = this._get_schema_from_target(target);
                const returns = schema.returns;
                return returns.length === 1 && returns[0].real_type instanceof torch.TensorType;
            }
        });
        this.registerType('torch._export.verifier.Verifier', class {});
        this.registerType('torch._dynamo.convert_frame.CatchErrorsWrapper', class {});
        this.registerType('torch._dynamo.convert_frame.ConvertFrameAssert', class {});
        this.registerType('torch._dynamo.convert_frame.ConvertFrame', class {});
        this.registerType('torch._dynamo.eval_frame._TorchDynamoContext', class {});
        this.registerType('torch._dynamo.eval_frame.OptimizedModule', class extends torch.nn.modules.module.Module {});
        this.registerType('torch._dynamo.eval_frame.OptimizeContext', class extends torch._dynamo.eval_frame._TorchDynamoContext {});
        this.registerType('torch._dynamo.hooks.Hooks', class {});
        this.registerType('torch._dynamo.repro.after_dynamo.WrapBackendDebug', class {});
        this.registerType('torch._TorchCompileInductorWrapper', class {});
        this.registerFunction('torch._inductor.compile_fx.compile_fx');
        this.registerFunction('torch_utils.persistence._reconstruct_persistent_obj', (meta) => {
            const name = `_imported_module_${Math.floor(Math.random() * 10000)}`;
            const module = new types.ModuleType(name);
            execution.register('sys').modules.set(name, module);
            const context = new python.Execution.Context(module, null);
            execution.exec(meta.get('module_src'), context);
            const obj = execution.invoke(`${name}.${meta.get('class_name')}`, []);
            const state = meta.get('state');
            if (state) {
                if (obj.__setstate__) {
                    obj.__setstate__(state);
                } else {
                    for (const [key, value] of state) {
                        obj[key] = value;
                    }
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
                this.index = index ? index : null;
            }
            __str__() {
                return this.index === null ? this.type : `${this.type}:${this.index}`;
            }
            toString() {
                const index = this.index === null ? '' : `, index=${this.index}`;
                return `device(type='${this.type}'${index})`;
            }
        });
        this.registerType('torch.memory_format', class {
            constructor(name) {
                this.name = name;
            }
            __str__() {
                return `torch.${this.name}`;
            }
            toString() {
                return this.__str__();
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
                return `torch.${this._name}`;
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
        this.registerType('torch.storage.UntypedStorage', class extends torch.storage._StorageBase {
            constructor() {
                super();
                throw new python.Error('UntypedStorage not implemented.');
            }
        });
        this.registerType('torch.storage.TypedStorage', class {
            constructor(...args) {
                if (args.length >= 2 && Number.isInteger(args[0]) && args[1] instanceof torch.dtype) {
                    if (args[3] instanceof torch.device) {
                        [this._size, this._dtype, , this._device] = args;
                    } else {
                        [this._size, this._dtype] = args;
                    }
                } else {
                    throw new python.Error(`Unsupported TypedStorage arguments '${JSON.stringify(args)}'.`);
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
        this.registerType('torch.storage._LegacyStorage', class extends torch.storage.TypedStorage {
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
                this.data = data || new torch.Tensor([]);
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
        this.registerType('torch.cuda._CudaLegacyStorage', class extends torch.storage._LegacyStorage {});
        this.registerType('torch.cuda.FloatStorage', class extends torch.cuda._CudaLegacyStorage {});
        this.registerType('torch.cuda.FloatTensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.DoubleStorage', class extends torch.cuda._CudaLegacyStorage {});
        this.registerType('torch.cuda.DoubleTensor', class extends torch.Tensor {});
        this.registerType('torch.cuda.amp.grad_scaler.GradScaler', class {});
        this.registerFunction('torch.cuda.amp.grad_scaler._refresh_per_optimizer_state');
        this.registerType('torch.SymBool', class {
            constructor(node) {
                this.node = node;
            }
        });
        this.registerType('torch.SymInt', class {
            constructor(node) {
                this.node = node;
            }
        });
        this.register('torch.nn').Module = this.register('torch.nn.modules.module').Module;
        this.register('torch.optim').Adam = this.register('torch.optim.adam').Adam;
        this.register('torch.nn').ReLU = this.register('torch.nn.modules.activation').ReLU;
        this.register('sklearn.utils').Bunch = this.register('sklearn.utils._bunch').Bunch;
        /* eslint-disable no-multi-assign */
        // https://github.com/pytorch/pytorch/blob/main/c10/core/ScalarType.h
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
        torch.quint4x2 = new torch.dtype(16, 'quint4x2');
        torch.quint2x4 = new torch.dtype(17, 'quint2x4');
        torch.bits1x8 = new torch.dtype(18, 'bits1x8');
        torch.bits2x4 = new torch.dtype(19, 'bits2x4');
        torch.bits2x4 = new torch.dtype(20, 'bits2x4');
        torch.bits8 = new torch.dtype(21, 'bits8');
        torch.bits16 = new torch.dtype(22, 'bits16');
        torch.float8_e5m2 = new torch.dtype(23, 'float8_e5m2', 1);
        torch.float8_e5m2fnuz = new torch.dtype(24, 'float8_e5m2fnuz', 1);
        torch.float8_e4m3fn = new torch.dtype(25, 'float8_e4m3fn', 1);
        torch.float8_e4m3fnuz = new torch.dtype(26, 'float8_e4m3fnuz', 1);
        torch.uint16 = new torch.dtype(27, 'uint16', 2);
        torch.uint32 = new torch.dtype(28, 'uint32', 4);
        torch.uint64 = new torch.dtype(29, 'uint64', 8);
        torch._export.serde.serialize._SERIALIZE_TO_TORCH_DTYPE = Object.fromEntries([
            ['uint8', 'BYTE'],
            ['int8', 'CHAR'], ['int16', 'SHORT'], ['int32', 'INT'], ['int64', 'LONG'],
            ['float16', 'HALF'], ['float32', 'FLOAT'], ['float64', 'DOUBLE'],
            ['complex32', 'COMPLEXHALF'], ['complex64', 'COMPLEXFLOAT'], ['complex128', 'COMPLEXDOUBLE'],
            ['bool', 'BOOL'],
            ['bfloat16', 'BFLOAT16']
        ].map(([key, value]) => [torch._export.serde.schema.ScalarType[value], torch[key]]));
        torch.contiguous_format = new torch.memory_format('contiguous_format');
        torch.channels_last = new torch.memory_format('channels_last');
        torch.channels_last_3d = new torch.memory_format('channels_last_3d');
        torch.preserve_format = new torch.memory_format('preserve_format');
        torch._export.serde.serialize._SERIALIZE_TO_TORCH_MEMORY_FORMAT = Object.fromEntries([
            ['contiguous_format', 'ContiguousFormat'],
            ['channels_last', 'ChannelsLast'],
            ['channels_last_3d', 'ChannelsLast3d'],
            ['preserve_format', 'PreserveFormat']
        ].map(([key, value]) => [torch._export.serde.schema.MemoryFormat[value], torch[key]]));
        /* eslint-enable no-multi-assign */
        torch.strided = new torch.layout('strided');
        torch.sparse_coo = new torch.layout('sparse_coo');
        torch.sparse_csr = new torch.layout('sparse_csr');
        torch.sparse_csc = new torch.layout('sparse_csc');
        torch.sparse_bsr = new torch.layout('sparse_bsr');
        torch.sparse_bsc = new torch.layout('sparse_bsc');
        torch._mkldnn = new torch.layout('_mkldnn');
        torch._export.serde.serialize._SERIALIZE_TO_TORCH_LAYOUT = Object.fromEntries([
            ['sparse_coo', 'SparseCoo'],
            ['sparse_csr', 'SparseCsr'],
            ['sparse_csc', 'SparseCsc'],
            ['sparse_bsr', 'SparseBsr'],
            ['sparse_bsc', 'SparseBsc'],
            ['_mkldnn', '_mkldnn'],
            ['strided', 'Strided'],
        ].map(([key, value]) => [torch._export.serde.schema.Layout[value], torch[key]]));
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
        this.registerType('fastcore.transform.Pipeline', class extends builtins.object {});
        this.registerType('fastcore.transform.Transform', class extends builtins.object {});
        this.registerType('fastcore.transform.DisplayedTransform', class extends fastcore.transform.Transform {});
        this.registerType('fastcore.transform.ItemTransform', class extends fastcore.transform.Transform {});
        this.registerType('fastai.basic_train.Learner', class {});
        this.registerType('fastai.basic_train.Recorder', class {});
        this.registerFunction('fastai.torch_core._fa_rebuild_tensor', (cls, ...args) => {
            const tensor = torch._utils._rebuild_tensor_v2(...args);
            return self.invoke(cls, tensor);
        });
        this.registerFunction('fastai.torch_core.trainable_params');
        this.registerFunction('fastai.torch_core._rebuild_from_type', (func, type, args, dict) => {
            const tensor = self.invoke(type, [func(...args)]);
            Object.assign(tensor, dict);
            return tensor;
        });
        this.registerType('fastai.torch_core.Module', class extends torch.nn.modules.module.Module {});
        this.registerType('fastai.torch_core.TensorBase', class extends torch.Tensor {
            constructor(x) {
                super();
                Object.assign(this, x);
            }
        });
        this.registerType('fastai.torch_core.TensorCategory', class extends fastai.torch_core.TensorBase {});
        this.registerType('fastai.torch_core.TensorImageBase', class extends fastai.torch_core.TensorBase {});
        this.registerType('fastai.torch_core.TensorImage', class extends fastai.torch_core.TensorImageBase {});
        this.registerType('fastai.torch_core.TensorMask', class extends fastai.torch_core.TensorImageBase {});
        this.registerType('fastai.torch_core.TensorMultiCategory', class extends fastai.torch_core.TensorCategory {});
        this.registerFunction('fastai.torch_core.uniform');
        this.registerType('fastai.callback.core.Callback', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.callback.core.TrainEvalCallback', class extends fastai.callback.core.Callback {});
        this.registerType('fastai.callback.fp16.AMPMode', class extends this._enum.Enum {});
        this.registerType('fastai.callback.fp16.MixedPrecision', class {});
        this.registerFunction('fastai.callback.hook._hook_inner');
        this.registerType('fastai.callback.hook.Hook', class extends builtins.object {});
        this.registerType('fastai.callback.hook.Hooks', class extends builtins.object {});
        this.registerType('fastai.callback.mixup.MixHandler', class extends fastai.callback.core.Callback {});
        this.registerType('fastai.callback.mixup.CutMix', class extends fastai.callback.mixup.MixHandler {});
        this.registerType('fastai.callback.progress.ProgressCallback', class {});
        this.registerType('fastai.callback.progress.ShowGraphCallback', class {});
        this.registerType('fastai.callback.tracker.EarlyStoppingCallback', class {});
        this.registerType('fastai.callback.tracker.TrackerCallback', class {});
        this.registerType('fastai.callback.tracker.SaveModelCallback', class extends fastai.callback.tracker.TrackerCallback {});
        this.registerType('fastai.data.core.DataLoaders', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.data.core.Datasets', class {});
        this.registerType('fastai.data.load.DataLoader', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.data.core.FilteredBase', class {});
        this.registerType('fastai.data.core.TfmdDL', class extends fastai.data.load.DataLoader {});
        this.registerType('fastai.data.core.TfmdLists', class {});
        this.registerType('fastai.data.load._FakeLoader', class {});
        this.registerFunction('fastai.data.load._wif');
        this.registerType('fastai.data.transforms.Categorize', class {});
        this.registerType('fastai.data.transforms.Category', class {});
        this.registerType('fastai.data.transforms.CategoryMap', class {});
        this.registerType('fastai.data.transforms.ColReader', class {});
        this.registerType('fastai.data.transforms.IntToFloatTensor', class {});
        this.registerType('fastai.data.transforms.MultiCategorize', class {});
        this.registerType('fastai.data.transforms.Normalize', class {});
        this.registerType('fastai.data.transforms.parent_label', class {});
        this.registerType('fastai.data.transforms.OneHotEncode', class {});
        this.registerType('fastai.data.transforms.RegressionSetup', class {});
        this.registerType('fastai.data.transforms.ToTensor', class {});
        this.registerType('fastai.data_block.CategoryList', class {});
        this.registerType('fastai.data_block.CategoryProcessor', class {});
        this.registerType('fastai.imports.noop', class {});
        this.registerType('fastai.layers.AdaptiveConcatPool2d', class {});
        this.registerType('fastai.layers.ConvLayer', class {});
        this.registerType('fastai.layers.Embedding', class {});
        this.registerType('fastai.layers.Flatten', class {});
        this.registerType('fastai.layers.FlattenedLoss', class {});
        this.registerType('fastai.layers.LinBnDrop', class {});
        this.registerType('fastai.layers.MergeLayer', class {});
        this.registerType('fastai.layers.PixelShuffle_ICNR', class {});
        this.registerType('fastai.layers.ResBlock', class {});
        this.registerType('fastai.layers.SelfAttention', class {});
        this.registerType('fastai.layers.SigmoidRange', class {});
        this.registerType('fastai.layers.TimeDistributed', class {});
        this.registerType('fastai.layers.ToTensorBase', class {});
        this.registerType('fastai.learner._ConstantFunc', class {});
        this.registerType('fastai.learner.Metric', class {});
        this.registerType('fastai.learner.AvgLoss', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.AvgMetric', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.AvgSmoothLoss', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.CastToTensor', class extends fastai.callback.core.Callback {});
        this.registerType('fastai.learner.Dice', class extends fastai.learner.Metric {});
        this.registerType('fastai.learner.Learner', class extends fastcore.basics.GetAttr {});
        this.registerType('fastai.learner.Recorder', class {});
        this.registerType('fastai.losses.BaseLoss', class {});
        this.registerType('fastai.losses.BCEWithLogitsLossFlat', class {});
        this.registerType('fastai.losses.CrossEntropyLossFlat', class extends fastai.losses.BaseLoss {});
        this.registerType('fastai.losses.FocalLoss', class extends fastai.torch_core.Module {});
        this.registerType('fastai.losses.FocalLossFlat', class extends fastai.losses.BaseLoss {});
        this.registerType('fastai.losses.LabelSmoothingCrossEntropy', class extends fastai.torch_core.Module {});
        this.registerType('fastai.metrics.AccumMetric', class extends fastai.learner.Metric {});
        this.registerType('fastai.metrics.Dice', class {});
        this.registerType('fastai.metrics.JaccardCoeff', class {});
        this.registerFunction('fastai.metrics._rmse');
        this.registerFunction('fastai.metrics.accuracy');
        this.registerFunction('fastai.metrics.accuracy_multi');
        this.registerFunction('fastai.metrics.foreground_acc');
        this.registerFunction('fastai.metrics.mse');
        this.registerFunction('fastai.metrics.error_rate');
        this.registerType('fastai.optimizer._BaseOptimizer', class {});
        this.registerType('fastai.optimizer.Optimizer', class extends fastai.optimizer._BaseOptimizer {});
        this.registerFunction('fastai.optimizer.Adam');
        this.registerFunction('fastai.optimizer.adam_step');
        this.registerFunction('fastai.optimizer.average_grad');
        this.registerFunction('fastai.optimizer.average_sqr_grad');
        this.registerFunction('fastai.optimizer.RAdam');
        this.registerFunction('fastai.optimizer.step_stat');
        this.registerFunction('fastai.optimizer.weight_decay');
        this.registerType('fastai.tabular.core.Categorify', class {});
        this.registerType('fastai.tabular.core.FillMissing', class {});
        this.registerType('fastai.tabular.core.FillStrategy', class {});
        this.registerType('fastai.tabular.core.ReadTabBatch', class extends fastcore.transform.ItemTransform {});
        this.registerType('fastai.tabular.core.TabDataLoader', class extends fastai.data.core.TfmdDL {});
        this.registerType('fastai.tabular.data.TabularDataLoaders', class extends fastai.data.core.DataLoaders {});
        this.registerType('fastai.tabular.core.Tabular', class {});
        this.registerType('fastai.tabular.core.TabularPandas', class extends fastai.tabular.core.Tabular {});
        this.registerType('fastai.tabular.core.TabWeightedDL', class {});
        this.registerType('fastai.tabular.learner.TabularLearner', class extends fastai.learner.Learner {});
        this.registerType('fastai.tabular.model.TabularModel', class {});
        this.registerFunction('fastai.vision.augment.aug_transforms');
        this.registerFunction('fastai.vision.augment.dihedral_mat');
        this.registerType('fastai.vision.augment._BrightnessLogit', class {});
        this.registerType('fastai.vision.augment._ContrastLogit', class {});
        this.registerType('fastai.vision.augment._WarpCoord', class {});
        this.registerType('fastai.vision.augment.RandTransform', class extends fastcore.transform.DisplayedTransform {});
        this.registerType('fastai.vision.augment.AffineCoordTfm', class extends fastai.vision.augment.RandTransform {});
        this.registerType('fastai.vision.augment.Brightness', class {});
        this.registerType('fastai.vision.augment.flip_mat', class {});
        this.registerType('fastai.vision.augment.Flip', class {});
        this.registerType('fastai.vision.augment.RandomResizedCropGPU', class {});
        this.registerType('fastai.vision.augment.Resize', class {});
        this.registerType('fastai.vision.augment.rotate_mat', class {});
        this.registerFunction('fastai.vision.augment.TensorImage.lighting');
        this.registerType('fastai.vision.augment.Warp', class extends fastai.vision.augment.AffineCoordTfm {});
        this.registerType('fastai.vision.augment.zoom_mat', class {});
        this.registerType('fastai.vision.core.PILImage', class {});
        this.registerType('fastai.vision.core.PILMask', class {});
        this.registerType('fastai.vision.core.AddMaskCodes', class {});
        this.registerType('fastai.vision.data.ImageList', class {});
        this.registerType('fastai.vision.image.Image', class {});
        this.registerType('fastai.vision.image.RandTransform', class {});
        this.registerType('fastai.vision.image.TfmCrop', class {});
        this.registerFunction('fastai.vision.learner._resnet_split');
        this.registerFunction('fastai.vision.learner.default_split');
        this.registerFunction('fastai.vision.learner.default_split');
        this.registerType('fastai.vision.learner.TimmBody', class {});
        this.registerType('fastai.vision.models.unet.DynamicUnet', class {});
        this.registerType('fastai.vision.models.unet.ResizeToOrig', class {});
        this.registerType('fastai.vision.models.unet.UnetBlock', class {});
        this.registerType('fastai.vision.models.xresnet.XResNet', class {});
        this.registerFunction('fastai.vision.transform._crop_pad');
    }

    exec(code , context) {
        const ast = this.ast;
        const program = ast.parse(code, '', null);
        if (!program) {
            throw new python.Error("Module '?' parse error.");
        }
        this.block(program.body, context);
    }

    debug(/* file */) {
    }

    source(file) {
        if (this._sources.has(file)) {
            return this._sources.get(file);
        }
        return null;
    }

    read(file) {
        const buffer = this.source(file);
        if (buffer) {
            const debug = this.debug(file);
            return this.parse(file, buffer, debug);
        }
        return null;
    }

    parse(filename, buffer, debug) {
        const ast = this.ast;
        const source = this._utf8Decoder.decode(buffer);
        const program = ast.parse(source, filename, debug);
        if (!program) {
            throw new python.Error(`Module '${filename}' parse error.`);
        }
        return program;
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
            const module = this._registry.get(name) || new this.builtins.module(name);
            module.__package__ = name;
            this._modules.set(name, module);
            const path = name.split('.').join('/');
            module.__path__ = [path];
            const file = `${path}.py`;
            const program = this.read(file);
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
        const builtins = this.builtins;
        if (typeof target === 'string') {
            target = this.resolve(target);
        }
        if (target) {
            if (target.__class__ === builtins.type) {
                if (target.prototype && target.prototype.__class__ === target) {
                    return Reflect.construct(target, args);
                }
                const obj = Object.create(target);
                if (obj.__init__ && typeof obj.__init__ === 'function') {
                    obj.__init__(...args);
                }
                return obj;
            } else if (target.__class__ === builtins.function) {
                if (target.__call__) {
                    return target.__call__(args);
                }
                return target(...args);
            }
        }
        throw new python.Error('Unsupported invoke target.');
    }

    call(target, name, args, keywords, context) {
        const builtins = this.builtins;
        const callTarget = this.target(target, context);
        const callArguments = args.map((arg) => this.expression(arg, context));
        if (!callTarget || (name !== null && !callTarget[name])) {
            if (name === '__new__' && callArguments.length === 1 && callArguments[0] === callTarget) {
                name = null;
                callArguments.shift();
            } else {
                const targetName = `${this.identifier(target)}.${name}`;
                throw new python.Error(`Unknown function '${targetName}'.`);
            }
        }
        const func = name ? callTarget[name] : callTarget;
        if (func.__class__ === builtins.type) {
            if (func.prototype && func.prototype.__class__ === func) {
                return Reflect.construct(func, callArguments);
            }
            const obj = Object.create(func);
            obj.__class__ = func;
            if (obj.__init__ && typeof obj.__init__ === 'function') {
                obj.__init__(...args);
            }
            return obj;
        }
        if (func.__class__ === builtins.function) {
            if (func.__call__) {
                return func.__call__(callArguments);
            }
        }
        if (func.__class__ === builtins.method) {
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
        args = method.args.posonlyargs.concat(method.args.args);
        const default_pos = args.length - method.args.defaults.length;
        for (let i = 0; i < method.args.args.length; i++) {
            const arg = method.args.args[i];
            let value = null;
            if (locals.length > 0) {
                value = locals.shift();
            } else if (i >= default_pos) {
                value = this.expression(method.args.defaults[i - default_pos], context);
            } else {
                throw new python.Error('Missing required positional argument.');
            }
            context.set(arg.arg, value);
        }
        return this.block(method.body, context);
    }

    block(statements, context) {
        statements = Array.prototype.slice.call(statements);
        while (statements.length > 0) {
            const stmt = statements.shift();
            const value = this.statement(stmt, context);
            if (value !== undefined) {
                return value;
            }
        }
        return undefined;
    }

    statement(stmt, context) {
        const ast = this.ast;
        const builtins = this.builtins;
        switch (stmt.__class__.__name__) {
            case 'Pass': {
                break;
            }
            case 'Constant': {
                break;
            }
            case 'Return': {
                return this.expression(stmt.value, context);
            }
            case 'FunctionDef': {
                const module = context.get('__name__');
                const self = this;
                const parent = context.get('__class__');
                const type = (parent === builtins.module) ? builtins.function : builtins.method;
                const func = {
                    __class__: type,
                    __globals__: context,
                    __module__: module,
                    __name__: stmt.name,
                    __code__: stmt,
                    __call__(args) {
                        return self.apply(this.__code__, args, this.__globals__);
                    }
                };
                context.set(stmt.name, func);
                break;
            }
            case 'ClassDef': {
                const bases = stmt.bases.map((base) => this.expression(base, context));
                if (bases.length > 1) {
                    throw new python.Error(`Unsupported multiple bases for class '${stmt.name}'.`);
                }
                const base = bases.length === 1 ? bases[0] : null;
                const name = `${context.get('__name__')}.${stmt.name}`;
                const value = this._createType(name, base ? class extends base {} : class {});
                value.__bases__ = bases;
                context.set(stmt.name, value);
                this.block(stmt.body, new python.Execution.Context(context.globals, value.prototype));
                break;
            }
            case 'AnnAssign': {
                const target = this.identifier(stmt.target, context);
                context.set(target, stmt.value ? this.expression(stmt.value, context) : undefined);
                break;
            }
            case 'Assign': {
                this.expression(stmt, context);
                break;
            }
            case 'If': {
                const test = this.expression(stmt.test, context);
                if (test === true || test) {
                    const value = this.block(stmt.body.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                    break;
                } else if (test === false) {
                    if (stmt.orelse) {
                        const value = this.block(stmt.orelse.statements, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    break;
                }
                throw new python.Error("Unsupported condition.");
            }
            case 'For': {
                if (stmt.target instanceof ast.Name && stmt.iter instanceof ast.Tuple === false) {
                    const range = this.expression(stmt.iter, context);
                    const variable = stmt.target;
                    for (const current of range) {
                        this.statement({ type: '=', target: variable, expression: { type: 'number', value: current } }, context);
                        const value = this.block(stmt.body.statements, context);
                        if (value !== undefined) {
                            return value;
                        }
                    }
                    break;
                }
                throw new python.Error("Unsupported 'for' statement.");
            }
            case 'While': {
                const test = this.expression(stmt.test, context);
                if (test) {
                    const value = this.block(stmt.body.statements, context);
                    if (value !== undefined) {
                        return value;
                    }
                }
                break;
            }
            case 'With': {
                const items = [];
                for (const item of stmt.items) {
                    items.push(this.expression(item.context_expr, context));
                }
                for (const item of items) {
                    if (item.__enter__ && item.__enter__.__call__) {
                        item.__enter__.__call__([item]);
                    }
                }
                const value = this.block(stmt.body, context);
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
            case 'Call': {
                this.expression(stmt, context);
                break;
            }
            case 'Import': {
                for (const alias of stmt.names) {
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
            case 'ImportFrom': {
                const fromlist = stmt.names.map((name) => name.name);
                const module = this.__import__(stmt.module, context.globals, context.locals, fromlist, stmt.level);
                for (const entry of stmt.names) {
                    const name = entry.name;
                    const asname = entry.asname ? entry.asname : null;
                    if (!module[name]) {
                        throw new python.Error(`Cannot import '${name}' from '${stmt.module}'.`);
                    }
                    context.set(asname ? asname : name, module[name]);
                }
                break;
            }
            default: {
                throw new python.Error(`Unsupported statement '${stmt.type}'.`);
            }
        }
        return undefined;
    }

    expression(expr, context) {
        const ast = this.ast;
        const builtins = this.builtins;
        const typing = this.typing;
        const self = context.get('self');
        switch (expr.__class__.__name__) {
            case 'Assign': {
                const [target] = expr.targets;
                if (target instanceof ast.Name) {
                    const value = this.expression(expr.value, context);
                    context.set(target.id, value);
                    return undefined;
                } else if (target instanceof ast.Subscript) {
                    if (target.value instanceof ast.Name &&
                        target.slice instanceof ast.List &&
                        target.slice.elts.length === 1) {
                        const index = this.expression(target.slice.elts[0], context);
                        const id = target.value.id;
                        if (id === '__annotations__') {
                            context.set(id, context.get(id) || {});
                        }
                        const obj = context.get(id);
                        const value = this.expression(expr.value, context);
                        if (obj instanceof Map) {
                            obj.set(index, value);
                        } else {
                            obj[index] = value;
                        }
                        return undefined;
                    }
                } else if (target instanceof ast.Attribute) {
                    const obj = this.expression(target.value, context);
                    const value = this.expression(expr.value, context);
                    obj[target.attr] = value;
                    return undefined;
                } else if (target instanceof ast.Tuple) {
                    context.target.push(target.elts);
                    const value = this.expression(expr.value, context);
                    context.target.pop();
                    if  (target.elts.every((elt) => elt instanceof ast.Name)) {
                        if (target.elts.length < value.length) {
                            throw new python.Error(`ValueError: too many values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        if (target.elts.length > value.length) {
                            throw new python.Error(`ValueError: not enough values to unpack (expected ${target.value.length}, actual ${value.length}).`);
                        }
                        for (let i = 0; i < value.length; i++) {
                            context.set(target.elts[i].id, value[i]);
                        }
                        return undefined;
                    }
                }
                break;
            }
            case 'List': {
                return expr.elts.map((expr) => this.expression(expr, context));
            }
            case 'Constant': {
                return expr.value;
            }
            case 'Subscript': {
                if (expr.value instanceof ast.Name &&
                    expr.slice instanceof ast.List &&
                    expr.slice.elts.length === 1) {
                    const id = expr.value.id;
                    if (context.get(id)) {
                        const index = this.expression(expr.slice.elts[0], context);
                        const target = context.get(id);
                        if (target instanceof Map) {
                            return target.get(index);
                        }
                        return target[index < 0 ? target.length + index : index];
                    }
                }
                const value = this.expression(expr.value, context);
                if (value && expr.slice instanceof ast.List &&
                    (value.__class__ === typing._TupleType ||
                     value.__class__ === typing._SpecialGenericAlias ||
                     value.__class__ === typing._SpecialForm)) {
                    const type = { ...value };
                    type.__args__ = expr.slice.elts.map((arg) => this.expression(arg, context));
                    return type;
                }
                if (expr.slice instanceof ast.List && expr.slice.elts.length === 1) {
                    const index = this.expression(expr.slice.elts[0], context);
                    if (value instanceof Map) {
                        return value.get(index);
                    }
                    return value[index < 0 ? value.length + index : index];
                }
                break;
            }
            case 'Attribute': {
                const value = this.target(expr.value, context);
                return value[expr.attr];
            }
            case 'Call': {
                const func = expr.func;
                if (func instanceof ast.Attribute) {
                    return this.call(func.value, func.attr, expr.args, expr.keywords, context, expr.location);
                }
                return this.call(func, null, expr.args, expr.keywords, context);
            }
            case 'Name': {
                const id = expr.id;
                if (id === 'self') {
                    return self;
                }
                const type = (value) => {
                    return value &&
                        (value.__class__ === builtins.type ||
                            value.__class__ === typing._TupleType ||
                            value.__class__ === typing._SpecialGenericAlias ||
                            value.__class__ === typing._SpecialForm);
                };
                const builtin = builtins[id];
                if (type(builtin)) {
                    return builtin;
                }
                const value = context.get(id);
                if (value === undefined) {
                    const value = typing[id];
                    if (type(value)) {
                        return value;
                    }
                }
                return value;
            }
            case 'Tuple': {
                return expr.elts.map((expr) => this.expression(expr, context));
            }
            case 'Dict': {
                const dict = {};
                for (let i = 0; i < expr.keys.length; i++) {
                    const key = this.expression(expr.keys[i], context);
                    const value = this.expression(expr.values[i], context);
                    dict[key] = value;
                }
                return dict;
            }
            case 'UnaryOp': {
                if (expr.op instanceof ast.USub) {
                    return -this.expression(expr.operand, context);
                }
                throw new python.Error(`Unsupported unary expression '${expr.op}'.`);
            }
            case 'binary': {
                switch (expr.op) {
                    case '==': {
                        return this.expression(expr.left, context) === this.expression(expr.right, context);
                    }
                    default: {
                        throw new python.Error(`Unsupported binary expression '${expr.op}'.`);
                    }
                }
            }
            default: {
                throw new python.Error(`Unsupported expression '${expr.type}'.`);
            }
        }
        return undefined;
    }

    identifier(expr) {
        const ast = this.ast;
        if (expr instanceof ast.Name) {
            return expr.id;
        }
        if (expr instanceof ast.Attribute) {
            return `${this.identifier(expr.value)}.${expr.attr}`;
        }
        return null;
    }

    target(expr, context) {
        const ast = this.ast;
        let current = expr;
        let path = [];
        for (;;) {
            if (current instanceof ast.Attribute) {
                path.push(current.attr);
                current = current.value;
            } else if (current instanceof ast.Name && current.id !== 'self' && current.id !== 'CONSTANTS') {
                path.push(current.id);
                break;
            } else {
                path = null;
                break;
            }
        }
        if (path) {
            let target = null;
            for (let i = path.length - 1; i >= 0; i--) {
                const name = path[i];
                if (target) {
                    target = target.__getattr__ ? target.__getattr__(name) : target[name];
                } else {
                    target = context.get(name);
                }
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
        return this.expression(expr, context);
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
                if (!value.__module__) {
                    value.__module__ = current;
                }
                const parent = this.register(current);
                parent[child] = value;
                value = parent;
            }
        }
        return this._registry.get(name);
    }

    registerFunction(name, value) {
        const builtins = this.builtins;
        const index = name.lastIndexOf('.');
        if (!value) {
            value = () => {
                throw new python.Error(`'${name}' is not implemented.`);
            };
        }
        value.__class__ = builtins.function;
        value.__name__ = index === -1 ? name : name.substring(index + 1);
        value.__module__ = index === -1 ? '' : name.substring(0, index);
        const module = this.register(value.__module__);
        if (module[name]) {
            throw new python.Error(`Function '${name}' is already registered.`);
        }
        module[value.__name__] = value;
        return value;
    }

    _createType(name, value) {
        const builtins = this.builtins;
        const index = name.lastIndexOf('.');
        value.__class__ = builtins.type;
        value.__name__ = index === -1 ? name : name.substring(index + 1);
        value.__module__ = index === -1 ? '' : name.substring(0, index);
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
        return this._view.getFloat64(position, false);
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
            const size = Math.min(0x20000000, this._stream.length - this._position);
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
            const size = Math.max(length, Math.min(0x10000000, this._length - this._offset));
            this._buffer = this._stream.read(size);
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
