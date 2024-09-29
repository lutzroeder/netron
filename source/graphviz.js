
const graphviz = {};

graphviz.ModelFactory = class {

    match(context) {
        const reader = context.read('text', 0x10000);
        if (reader) {
            try {
                const line = reader.read('\n');
                if (line === undefined) {
                    return;
                }
                if (line.indexOf('digraph') !== -1) {
                    context.type = 'graphviz.dot';
                }
            } catch {
                // continue regardless of error
            }
        }
    }

    async open(/* context */) {
        throw new graphviz.Error('Invalid file content. File contains Graphviz data.');
    }
};

graphviz.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading Graphviz model.';
    }
};

export const ModelFactory = graphviz.ModelFactory;
