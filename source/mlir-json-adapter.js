// MLIR JSON Adapter - Convert mlir-js-parser output to Netron's expected format

const mlir = {};

mlir.JsonAdapter = class {
    
    /**
     * Convert mlir-js-parser JSON output to Netron's internal format
     * @param {Object} mlirJson - Output from mlir-js-parser.parseMlirJson()
     * @returns {Object} - Netron compatible object with operations and definitions
     */
    static convert(mlirJson) {
        if (!mlirJson || mlirJson.name !== 'builtin.module') {
            throw new Error('Expected builtin.module at root');
        }
        
        const result = {
            operations: [],
            definitions: []
        };
        
        // Extract top-level operations from the module
        this._convertRegions(mlirJson.regions, result.operations);
        
        // Extract definitions from module attributes (if any)
        if (mlirJson.attributes) {
            for (const [name, value] of Object.entries(mlirJson.attributes)) {
                result.definitions.push({ name, value });
            }
        }
        
        return result;
    }
    
    /**
     * Convert regions to operations list
     * @param {Array} regions - MLIR regions from JSON
     * @param {Array} operations - Target operations array
     */
    static _convertRegions(regions, operations) {
        for (const region of regions) {
            for (const block of region.blocks) {
                for (const op of block.operations) {
                    const operation = this._convertOperation(op);
                    operations.push(operation);
                }
            }
        }
    }
    
    /**
     * Convert single operation from mlir-js-parser JSON to Netron format
     * @param {Object} jsonOp - Operation from mlir-js-parser JSON
     * @returns {Object} - Netron compatible operation
     */
    static _convertOperation(jsonOp) {
        const operation = {
            name: jsonOp.name,
            kind: this._extractKind(jsonOp.name),
            attributes: this._convertAttributes(jsonOp.attributes),
            operands: this._convertOperands(jsonOp.operands),
            results: this._convertResults(jsonOp.results),
            regions: []
        };
        
        // Convert nested regions
        if (jsonOp.regions && jsonOp.regions.length > 0) {
            for (const region of jsonOp.regions) {
                const convertedRegion = {
                    blocks: []
                };
                for (const block of region.blocks) {
                    const convertedBlock = {
                        arguments: this._convertBlockArguments(block.arguments),
                        operations: []
                    };
                    for (const nestedOp of block.operations) {
                        convertedBlock.operations.push(this._convertOperation(nestedOp));
                    }
                    convertedRegion.blocks.push(convertedBlock);
                }
                operation.regions.push(convertedRegion);
            }
        }
        
        return operation;
    }
    
    /**
     * Extract operation kind from full name (e.g., "func.func" -> "func")
     * @param {string} name - Full operation name
     * @returns {string} - Operation kind
     */
    static _extractKind(name) {
        if (name.startsWith('torch.')) {
            const parts = name.split('.');
            if (parts[1] === 'aten' || parts[1] === 'prim') {
                return parts[2] || parts[1];
            }
            return parts[1] || name;
        }
        
        const lastDot = name.lastIndexOf('.');
        return lastDot !== -1 ? name.substring(lastDot + 1) : name;
    }
    
    /**
     * Convert attributes from mlir-js-parser format to Netron format
     * @param {Object} jsonAttributes - Attributes from JSON
     * @returns {Array} - Array of attribute objects
     */
    static _convertAttributes(jsonAttributes) {
        const attributes = [];
        for (const [name, value] of Object.entries(jsonAttributes)) {
            let convertedValue = value;
            let type = this._inferAttributeType(value);
            
            // Special handling for function_type attribute
            if (name === 'function_type' && typeof value === 'string') {
                convertedValue = this._parseFunctionType(value);
                type = 'function_type';
            }
            
            attributes.push({
                name,
                value: convertedValue,
                type
            });
        }
        return attributes;
    }
    
    /**
     * Convert operands from mlir-js-parser format
     * @param {Array} jsonOperands - Operands from JSON
     * @returns {Array} - Array of operand objects
     */
    static _convertOperands(jsonOperands) {
        return jsonOperands.map((operand, index) => ({
            name: operand.name || index.toString(),
            value: operand.value || `%${index}`, // SSA value name
            type: operand.type
        }));
    }
    
    /**
     * Convert results from mlir-js-parser format
     * @param {Array} jsonResults - Results from JSON
     * @returns {Array} - Array of result objects
     */
    static _convertResults(jsonResults) {
        return jsonResults.map((result, index) => ({
            name: result.name || index.toString(),
            value: result.value || `%${index}`, // SSA value name
            type: result.type
        }));
    }
    
    /**
     * Convert block arguments
     * @param {Array} jsonArguments - Block arguments from JSON
     * @returns {Array} - Array of argument objects
     */
    static _convertBlockArguments(jsonArguments) {
        return jsonArguments.map((arg, index) => ({
            name: arg.name || index.toString(),
            type: arg.type,
            value: arg.value || `%arg${index}`
        }));
    }
    
    /**
     * Infer attribute type from value
     * @param {*} value - Attribute value
     * @returns {string} - Inferred type
     */
    static _inferAttributeType(value) {
        if (typeof value === 'string') {
            // Check if it looks like a number
            if (/^\d+$/.test(value)) return 'int64';
            if (/^\d*\.\d*$/.test(value)) return 'float32';
            return 'string';
        }
        if (typeof value === 'boolean') return 'boolean';
        if (Array.isArray(value)) return 'array';
        if (typeof value === 'object') return 'dictionary';
        return 'attribute';
    }
    
    /**
     * Extract function type information for compatibility with existing Netron code
     * This is needed for functions that have special handling in mlir.Graph constructor
     * @param {Object} funcOp - Function operation from JSON
     * @returns {Object} - Function type information
     */
    static extractFunctionType(funcOp) {
        // Look for function_type attribute
        const functionTypeAttr = funcOp.attributes.function_type;
        
        if (functionTypeAttr) {
            return functionTypeAttr;
        }
        
        // Fallback: try to infer from operation structure
        return {
            inputs: funcOp.operands || [],
            results: funcOp.results || []
        };
    }
    
    /**
     * Parse MLIR function type string into inputs/results format
     * Converts "(i32, f32) -> (tensor<4xf32>)" to {inputs: [...], results: [...]}
     * @param {string} functionTypeStr - Function type string from MLIR
     * @returns {Object} - Object with inputs and results arrays
     */
    static _parseFunctionType(functionTypeStr) {
        try {
            // Basic parsing for function type string like "(i32) -> i32" or "(i32, f32) -> (tensor<4xf32>)"
            const match = functionTypeStr.match(/^\(([^)]*)\)\s*->\s*(.+)$/);
            
            if (!match) {
                // Fallback for malformed strings
                return { inputs: [], results: [] };
            }
            
            const inputsStr = match[1].trim();
            const resultsStr = match[2].trim();
            
            // Parse inputs
            const inputs = [];
            if (inputsStr) {
                const inputTypes = inputsStr.split(',').map(s => s.trim()).filter(s => s);
                for (let i = 0; i < inputTypes.length; i++) {
                    inputs.push({
                        name: i.toString(),
                        type: inputTypes[i],
                        value: `%arg${i}`
                    });
                }
            }
            
            // Parse results
            const results = [];
            let resultTypes = [];
            
            if (resultsStr.startsWith('(') && resultsStr.endsWith(')')) {
                // Multiple results: "(type1, type2)"
                const innerResults = resultsStr.slice(1, -1).trim();
                if (innerResults) {
                    resultTypes = innerResults.split(',').map(s => s.trim()).filter(s => s);
                }
            } else {
                // Single result: "type"
                resultTypes = [resultsStr];
            }
            
            for (let i = 0; i < resultTypes.length; i++) {
                results.push({
                    name: i.toString(),
                    type: resultTypes[i],
                    value: `%${i}`
                });
            }
            
            return { inputs, results };
            
        } catch (error) {
            // Fallback on parsing error
            console.warn('Failed to parse function type:', functionTypeStr, error);
            return { inputs: [], results: [] };
        }
    }
};

// Export for use in mlir.js
export { mlir };