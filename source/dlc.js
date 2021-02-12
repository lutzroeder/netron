/* jshint esversion: 6 */

var dlc = dlc || {};

dlc.ModelFactory = class {

    match(context) {
        const entries = context.entries('zip');
        if (entries.find((entry) => entry.name === 'model')) {
            return true;
        }
        return false;
    }

    open(/* context */) {
        return Promise.resolve().then(() => {
            throw new dlc.Error("File contains undocumented DLC data.");
        });
    }
};

dlc.Error = class extends Error {

    constructor(message) {
        super(message);
        this.name = 'Error loading DLC model.';
    }
};

if (typeof module !== 'undefined' && typeof module.exports === 'object') {
    module.exports.ModelFactory = dlc.ModelFactory;
}
