// Experimental
// contributor @tucan9389

const mlir = {};

// Import text utilities and MLIR adapter for improved MLIR parsing
import * as text from './text.js';
import { mlir as mlirAdapter } from './mlir-json-adapter.js';

mlir.ModelFactory = class {

    async match(context) {
        const stream = context.stream;
        if (stream && stream.length > 4) {
            const buffer = stream.peek(4);
            const signature = String.fromCharCode.apply(null, buffer);
            if (signature === 'ML\xEFR') {
                return context.set('mlir.binary');
            }
        }
        try {
            const reader = await context.read('text', 0x10000);
            for (let line = reader.read('\n'); line !== undefined; line = reader.read('\n')) {
                if (/module\s+(\w+\s+)?{/.test(line) || /tensor<\w+>/.test(line) || /func\s*@\w+/.test(line)) {
                    return context.set('mlir.text');
                }
            }
        } catch {
            // continue regardless of error
        }
        return null;
    }

    async open(context) {
        switch (context.type) {
            case 'mlir.text': {
                const decoder = await context.read('text.decoder');
                const parser = new mlir.Parser(decoder);
                const obj = await parser.read();
                const metadata = new mlir.Metadata();
                return new mlir.Model(metadata, obj);
            }
            case 'mlir.binary': {
                const reader = await context.read('binary');
                const parser = new mlir.BytecodeReader(reader);
                parser.read();
                throw new mlir.Error('File contains unsupported MLIR bytecode data.');
            }
            default: {
                throw new mlir.Error(`Unsupported MLIR format '${context.type}'.`);
            }
        }
    }
};

mlir.Model = class {

    constructor(metadata, obj) {
        this.format = 'MLIR';
        this.modules = [];
        this.metadata = [];
        for (const op of obj.operations) {
            if (op.name.endsWith('.func')) {
                const graph = new mlir.Graph(metadata, op);
                this.modules.push(graph);
            }
            if (op.name.endsWith('.module')) {
                for (const region of op.regions) {
                    for (const block of region.blocks) {
                        for (const op of block.operations) {
                            if (op.name.endsWith('.func')) {
                                const graph = new mlir.Graph(metadata, op);
                                this.modules.push(graph);
                            }
                        }
                    }
                }
            }
        }
        if (obj.definitions) {
            for (const attribute of obj.definitions) {
                const value = typeof attribute.value === 'string' ? attribute.value : JSON.stringify(attribute.value);
                const metadata = new mlir.Argument(attribute.name, value, 'attribute');
                this.metadata.push(metadata);
            }
        }
    }
};

mlir.Graph = class {

    constructor(metadata, func) {
        const attr = Object.fromEntries(func.attributes.map((attr) => [attr.name, attr.value]));
        this.name = attr.sym_name || '';
        this.type = func.name === 'func' || func.name.endsWith('.func') ? 'function' : '';
        this.description = func.name;
        this.inputs = [];
        this.outputs = [];
        this.nodes = [];
        // inputs of function
        const function_type = attr.function_type;
        if (function_type && function_type.inputs) {
            for (let i = 0; i < function_type.inputs.length; i++) {
                const input = function_type.inputs[i];
                const name = input.name || i.toString();
                const type = input.type ? mlir.Utility.valueType(input.type) : null;
                const value = new mlir.Value(input.value, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.inputs.push(argument);
            }
        }
        // outputs of function
        if (function_type && function_type.results) {
            for (let i = 0; i < function_type.results.length; i++) {
                const output = function_type.results[i];
                const name = output.name || i.toString();
                const type = output.type ? mlir.Utility.valueType(output.type) : null;
                const value = new mlir.Value(output.value, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
        // operations
        // args is map of edges. args will be converted to mlir.Arguemnts.
        const values = new Map();
        values.map = (name) => {
            if (!values.has(name)) {
                values.set(name, { name, to: [], from: [] });
            }
            return values.get(name);
        };
        // operations - setup arguments
        const operations = [];
        for (const region of func.regions) {
            for (const block of region.blocks) {
                for (const op of block.operations) {
                    const operation = {
                        type: op.kind || op.name,
                        identifier: op.name,
                        attributes: op.attributes,
                        operands: [],
                        results: [],
                        delete: false,
                    };
                    const operands = op.operands || [];
                    for (let i = 0; i < operands.length; i++) {
                        const input = op.operands[i];
                        if (input.value instanceof Uint8Array) {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value,
                                type: input.type
                            });
                        } else if (Number.isInteger(input.value)) {
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: input.value,
                                type: 'int64'
                            });
                        } else {
                            const arg = values.map(input.value);
                            operation.operands.push({
                                name: input.name || i.toString(),
                                value: arg
                            });
                        }
                    }
                    const results = op.results || [];
                    for (let i = 0; i < results.length; i++) {
                        const output = op.results[i];
                        const arg = values.map(output.value);
                        operation.results.push({
                            name: output.name || i.toString(),
                            value: arg
                        });
                        arg.type = output.type;
                    }
                    operations.push(operation);
                }
            }
        }
        // operations - connect arguments
        for (const operation of operations) {
            for (let i = 0; i < operation.operands.length; i++) {
                const input = operation.operands[i];
                if (input.value && typeof input.value === 'object' && input.value.to && input.value.from) {
                    input.value.to.push([operation, i]);
                }
            }
            for (let i = 0; i < operation.results.length; i++) {
                const output = operation.results[i];
                if (output.value && typeof output.value === 'object' && output.value.to && output.value.from) {
                    output.value.from.push([operation, i]);
                }
            }
        }
        // operations - convert to mlir.Node
        for (const op of operations) {
            if (!op.delete) {
                const node = new mlir.Node(metadata, op, values);
                this.nodes.push(node);
            }
        }
    }
};

mlir.Argument = class {

    constructor(name, value, type) {
        this.name = name;
        this.value = value;
        this.type = type || null;
    }
};

mlir.Value = class {

    constructor(name, type, description, initializer) {
        this.name = name;
        this.type = type || null;
        this.description = description || '';
        this.initializer = initializer || null;
    }
};

mlir.Node = class {

    constructor(metadata, op, values) {
        this.type = metadata.type(op.type);
        this.name = op.identifier;
        this.description = op.type;
        this.inputs = [];
        this.outputs = [];
        this.attributes = [];
        // inputs
        for (let i = 0; i < op.operands.length; i++) {
            const input = op.operands[i];
            if (input.value && typeof input.value === 'object' && input.value.to && input.value.from) {
                const name = input.name || i.toString();
                const type = mlir.Utility.valueType(input.value.type);
                const value = new mlir.Value(input.value.name, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.inputs.push(argument);
            } else {
                const name = input.name || i.toString();
                const type = input.type;
                const value = new mlir.Value('', type, '', input.value);
                const argument = new mlir.Argument(name, [value]);
                this.inputs.push(argument);
            }
        }
        // outputs
        for (let i = 0; i < op.results.length; i++) {
            const output = op.results[i];
            if (output.value && typeof output.value === 'object' && output.value.to && output.value.from) {
                const name = output.name || i.toString();
                const type = mlir.Utility.valueType(output.value.type);
                const value = new mlir.Value(output.value.name, type, '', null);
                const argument = new mlir.Argument(name, [value]);
                this.outputs.push(argument);
            }
        }
        // attributes
        if (op.attributes) {
            for (let i = 0; i < op.attributes.length; i++) {
                const attr = op.attributes[i];
                const name = attr.name || i.toString();
                let type = attr.type;
                let value = attr.value;
                if (type && type.startsWith('tensor<')) {
                    value = new mlir.Tensor(mlir.Utility.valueType(type), value);
                    type = 'tensor';
                }
                const attribute = new mlir.Argument(name, value, type || 'attribute');
                this.attributes.push(attribute);
            }
        }
    }
};

mlir.Tensor = class {

    constructor(type, data) {
        this.type = type;
        this.values = data;
        this.encoding = data instanceof Uint8Array ? '<' : '|';
    }
};

mlir.TensorType = class {

    constructor(dataType, shape) {
        this.dataType = mlir.Utility.dataType(dataType); // string
        this.shape = shape || new mlir.TensorShape([]);  // mlir.TensorShape
    }

    toString() {
        return this.dataType + this.shape.toString();
    }
};

mlir.TensorShape = class {

    constructor(dimensions) {
        this.dimensions = dimensions;
    }

    toString() {
        if (!this.dimensions || this.dimensions.length === 0) {
            return '';
        }
        return `[${this.dimensions.map((dimension) => dimension.toString()).join(',')}]`;
    }
};

mlir.Parser = class {

    constructor(decoder) {
        this._decoder = decoder;
        this._parserModule = null;
    }

    async read() {
        return this.parse();
    }

    async parse() {
        try {
            // Download and initialize the parser module if not already done
            if (!this._parserModule) {
                await this._loadMlirParser();
            }
            
            // Get the MLIR text content from the decoder
            let mlirText = '';
            
            // Read all characters from decoder
            let char;
            while ((char = this._decoder.decode()) !== undefined) {
                mlirText += char;
            }
            
            // Use mlir-js-parser to parse the MLIR text
            const result = this._parserModule.parseMlirJson(mlirText);
            
            if (!result.ok) {
                throw new mlir.Error(`MLIR parsing failed: ${result.error}`);
            }
            
            // Use adapter to convert to Netron format
            const converted = mlirAdapter.JsonAdapter.convert(result.json);
            
            return converted;
        } catch (error) {
            throw new mlir.Error(`Failed to parse MLIR: ${error.message}`);
        }
    }

    async _loadMlirParser() {
        try {
            // Try to load local files first (for development with local setup)
            try {
                const { createParserModule } = await import('./mlir-bindings.js');
                const moduleFactory = (await import('./mlir_parser.js')).default;
                
                const moduleFactoryWithConfig = () => moduleFactory({
                    locateFile: (path) => {
                        if (path.endsWith('.wasm')) {
                            return './' + path;
                        }
                        return path;
                    }
                });
                
                this._parserModule = await createParserModule(moduleFactoryWithConfig);
                console.log('MLIR parser loaded from local files');
                return;
            } catch (localError) {
                console.log('Local MLIR parser files not found, trying GitHub Releases...');
            }
            
            // Fallback to GitHub Releases (development only)
            const isLocalDev = typeof window !== 'undefined' && (
                window.location.hostname === 'localhost' ||
                window.location.hostname === '127.0.0.1'
            );
            if (isLocalDev) {
                const baseUrl = 'https://github.com/tucan9389/mlir-js-parser/releases/download/v0.1/';
                try {
                    const bindingsModule = await import(baseUrl + 'bindings.js');
                    const createParserModule = bindingsModule.createParserModule;
                    const parserModule = await import(baseUrl + 'mlir_parser.js');
                    const moduleFactory = parserModule.default;
                    const moduleFactoryWithConfig = () => moduleFactory({
                        locateFile: (path) => path.endsWith('.wasm') ? baseUrl + path : path
                    });
                    this._parserModule = await createParserModule(moduleFactoryWithConfig);
                    console.log('MLIR parser loaded from GitHub Releases (dev)');
                    return;
                } catch (githubError) {
                    console.error('Failed to load MLIR parser from GitHub Releases:', githubError);
                    throw new Error('MLIR parser could not be loaded from GitHub Releases');
                }
            }
            
        } catch (error) {
            throw new mlir.Error(`MLIR parser not available. For MLIR support, ensure network access or run 'node tools/update-mlir-parser.js' for local setup. Error: ${error.message}`);
        }
    }
};

mlir.BytecodeReader = class {

    constructor(reader) {
        this._reader = reader;
    }

    read() {
        const signature = this._reader.read(4);
        if (signature[0] !== 0x4D || signature[1] !== 0x4C || signature[2] !== 0xEF || signature[3] !== 0x52) {
            throw new mlir.Error('Invalid bytecode signature.');
        }
        const version = this._reader.read(4);
        if (version[0] !== 0x00 || version[1] !== 0x00 || version[2] !== 0x00 || version[3] !== 0x00) {
            throw new mlir.Error('Invalid bytecode version.');
        }
        const producer = this._readStringRef();
        throw new mlir.Error(`Unsupported MLIR bytecode producer '${producer}'.`);
    }

    _readStringRef() {
        let result = '';
        while (true) {
            const value = this._reader.read(1)[0];
            if (value === 0x00) {
                break;
            }
            result += String.fromCharCode(value);
        }
        return result;
    }
};

mlir.Utility = class {

    static dataType(value) {
        switch (value) {
            case 'f16': return 'float16';
            case 'f32': return 'float32';
            case 'f64': return 'float64';
            case 'i1': return 'boolean';
            case 'i8': return 'int8';
            case 'i16': return 'int16';
            case 'i32': return 'int32';
            case 'i64': return 'int64';
            case 'ui8': return 'uint8';
            case 'ui16': return 'uint16';
            case 'ui32': return 'uint32';
            case 'ui64': return 'uint64';
            default: return value;
        }
    }

    static valueType(value) {
        if (!value || typeof value !== 'string') {
            return value; // Return as-is if not a valid string
        }
        const index = value.indexOf('<');
        if (index !== -1) {
            const spec = value.substring(index + 1, value.length - 1);
            const shape = [];
            const index2 = spec.indexOf('x');
            if (index2 !== -1) {
                const size = spec.substring(0, index2);
                const parts = size.split('x');
                for (const part of parts) {
                    const dimension = parseInt(part.trim(), 10);
                    shape.push(dimension);
                }
            }
            const type = spec.substring(spec.lastIndexOf('x') + 1);
            const dataType = mlir.Utility.dataType(type);
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (value.startsWith('tensor<') && value.endsWith('>')) {
            const spec = value.substring(7, value.length - 1);
            const shape = [];
            const parts = spec.split('x');
            for (let i = 0; i < parts.length - 1; i++) {
                const dimension = parseInt(parts[i].trim(), 10);
                shape.push(dimension);
            }
            const dataType = parts[parts.length - 1];
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        if (value.startsWith('!torch.vtensor<') && value.endsWith('>')) {
            const spec = value.substring(15, value.length - 1);
            const index = spec.lastIndexOf(',');
            const shape = JSON.parse(spec.substring(0, index));
            const dataType = spec.substring(index + 1);
            return new mlir.TensorType(dataType, new mlir.TensorShape(shape));
        }
        return value;
    }
};

mlir.Metadata = class {

    constructor() {
        this._types = new Map();
        this.register('stablehlo.reshape', 'Shape');
        this.register('asuka.split', 'Tensor');
        this.register('stablehlo.transpose', 'Transform');
        this.register('toy.transpose', 'Transform');
        this.register('asuka.softmax', 'Activation');
        this.register('stablehlo.slice', 'Tensor');
    }

    register(name, category) {
        this._types.set(name, { name, category });
    }

    type(name) {
        return this._types.get(name) || { name };
    }
};

mlir.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading MLIR model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports !== 'undefined') {
    module.exports.mlir = mlir;
}

// ESM export for modern imports
export const ModelFactory = mlir.ModelFactory;