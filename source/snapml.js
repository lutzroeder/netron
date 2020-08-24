/* jshint esversion: 6 */

var snapml = snapml || {};

snapml.ModelFactory = class {

    match(context) {
        const extension = context.identifier.split('.').pop().toLowerCase();
        if (extension == 'dnn') {
            const tags = context.tags('pb');
            if (tags.get(2) == 0 && tags.get(4) == 0 && tags.get(10) == 2) {
                return true;
            }
        }
        return false;
    }

    open(context, host) {
        throw new snapml.Error("File contains undocumented SnapML data in '" + context.identifier + "'.");
    }
};

snapml.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading SnapML model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = snapml.ModelFactory;
}
