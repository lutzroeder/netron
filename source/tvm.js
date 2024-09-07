
const tvm = {};

tvm.ModelFactory = class {

    match(context) {
        const identifier = context.identifier;
        const extension = identifier.split('.').pop().toLowerCase();
        if (extension === 'json') {
            const obj = context.peek('json');
            if (obj && Array.isArray(obj.nodes) && Array.isArray(obj.arg_nodes) && Array.isArray(obj.heads) &&
                obj.nodes.every((node) => node && (node.op === 'null' || node.op === 'tvm_op'))) {
                context.type = 'tvm.json';
                context.target = obj;
                return;
            }
        }
        const stream = context.stream;
        const signature = [0xB7, 0x9C, 0x04, 0x05, 0x4F, 0x8D, 0xE5, 0xF7];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            context.type = 'tvm.params';
        }
    }

    filter(context, type) {
        return context.type !== 'tvm.json' || type !== 'tvm.params';
    }

    async open(context) {
        switch (context.type) {
            case 'tvm.json':
                throw new tvm.Error(`Unsupported TVN model.`);
            case 'tvm.params':
                // https://github.com/apache/tvm/blob/main/src/runtime/file_utils.cc#L184
                throw new tvm.Error(`Invalid file content. File contains TVN NDArray data.`);
            default:
                throw new tvm.Error(`Unsupported TVN format '${context.type}'.`);
        }
    }
};

tvm.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading TVM model.';
    }
};

export const ModelFactory = tvm.ModelFactory;
