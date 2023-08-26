
var nnc = {};

nnc.ModelFactory = class {

    match(context) {
        const stream = context.stream;
        const signature = [ 0xC0, 0x0F, 0x00, 0x00, 0x45, 0x4E, 0x4E, 0x43 ];
        if (stream && signature.length <= stream.length && stream.peek(signature.length).every((value, index) => value === signature[index])) {
            return 'nnc';
        }
        return '';
    }

    async open(/* context, target */) {
        throw new nnc.Error('File contains undocumented NNC data.');
    }
};

nnc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading NNC model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = nnc.ModelFactory;
}


