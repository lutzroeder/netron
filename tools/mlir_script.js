
import * as fs from 'fs/promises';
import * as path from 'path';
import * as tablegen from './tablegen.js';
import * as url from 'url';

class Operator {

    constructor(def) {
        this.def = def;
        let opInfo = null;
        for (const parent of this.def.parents) {
            const parentClass = this.def.parser.classes.get(parent.name);
            if (parentClass) {
                opInfo = this._findOpParent(parentClass, parent.args, {});
                if (opInfo) {
                    break;
                }
            }
        }
        this.dialectName = opInfo?.dialect || null;
        this.opName = opInfo?.mnemonic || null;
    }

    getDialectName() {
        return this.dialectName || '';
    }

    getOperationName() {
        return this.dialectName && this.opName ? `${this.dialectName}.${this.opName}` : null;
    }

    _findOpParent(parentClass, parentArgs, substitutions) {
        const subs = { ...substitutions };
        if (parentClass.templateArgs && parentArgs) {
            for (let i = 0; i < Math.min(parentClass.templateArgs.length, parentArgs.length); i++) {
                const paramName = parentClass.templateArgs[i].name;
                const argValue = parentArgs[i];
                subs[paramName] = (typeof argValue === 'string' && substitutions[argValue])
                    ? substitutions[argValue] : argValue;
            }
        }
        if (parentClass.name === 'Op' && parentArgs.length >= 2) {
            let [dialectArg, mnemonicArg] = parentArgs;
            if (typeof dialectArg === 'string' && subs[dialectArg]) {
                dialectArg = subs[dialectArg];
            }
            if (typeof mnemonicArg === 'string' && subs[mnemonicArg]) {
                mnemonicArg = subs[mnemonicArg];
            }
            let dialectName = null;
            if (typeof dialectArg === 'string') {
                const dialectDef = this.def.parser.defs.get(dialectArg) || this.def.parser.classes.get(dialectArg);
                if (dialectDef) {
                    dialectName = dialectDef.getValueAsString('name');
                }
            }
            const mnemonic = typeof mnemonicArg === 'string' ? mnemonicArg.replace(/^"|"$/g, '') : null;
            if (dialectName && mnemonic) {
                return { dialect: dialectName, mnemonic };
            }
        }
        for (const grandparent of parentClass.parents) {
            const grandparentClass = this.def.parser.classes.get(grandparent.name);
            if (grandparentClass) {
                const resolvedArgs = grandparent.args.map((arg) =>
                    (typeof arg === 'string' && subs[arg]) ? subs[arg] : arg
                );
                const result = this._findOpParent(grandparentClass, resolvedArgs, subs);
                if (result) {
                    return result;
                }
            }
        }
        return null;
    }
}

const access = async (path) => {
    try {
        await fs.access(path);
        return true;
    } catch {
        return false;
    }
};

const main = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const source = path.join(dirname, '..', 'third_party', 'source');
    const paths = [
        path.join(source, 'llvm-project', 'mlir', 'include'),
        path.join(source, 'stablehlo'),
        path.join(source, 'onnx-mlir'),
        path.join(source, 'torch-mlir', 'include'),
        path.join(source, 'tensorflow'),
        path.join(source, 'mlir-hlo'),
        path.join(source, 'iree', 'compiler', 'src')
    ];
    const dialects = [
        'stablehlo/dialect/StablehloOps.td',
        'stablehlo/dialect/ChloOps.td',
        'mlir/Dialect/Affine/IR/AffineOps.td',
        'mlir/Dialect/Func/IR/FuncOps.td',
        'mlir/Dialect/Linalg/IR/LinalgOps.td',
        'mlir/Dialect/MemRef/IR/MemRefOps.td',
        'mlir/Dialect/Quant/IR/QuantOps.td',
        'mlir/Dialect/Tensor/IR/TensorOps.td',
        'mlir/Dialect/Tosa/IR/TosaOps.td',
        'mlir/Dialect/Vector/IR/VectorOps.td',
        'mlir/Dialect/IRDL/IR/IRDLOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVStructureOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVControlFlowOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVArithmeticOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVLogicalOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVBitOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVCastOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVCompositeOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVAtomicOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVBarrierOps.td',
        'mlir/Dialect/SPIRV/IR/SPIRVGroupOps.td',
        'src/Dialect/ONNX/ONNX.td',
        'src/Dialect/ONNX/ONNXOps.td.inc',
        'src/Dialect/ONNX/AdditionalONNXOps.td',
        'torch-mlir/Dialect/Torch/IR/TorchOps.td',
        'tensorflow/compiler/mlir/lite/ir/tfl_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td',
        'mhlo/IR/hlo_ops.td',
        'iree/compiler/Dialect/HAL/IR/HALOps.td',
        'iree/compiler/Dialect/Flow/IR/FlowOps.td',
    ];
    const file = path.join(dirname, '..', 'source', 'mlir-metadata.json');
    const operations = new Map();
    const exists = await access(file);
    if (exists) {
        const content = await fs.readFile(file, 'utf-8');
        const json = JSON.parse(content);
        for (const op of json) {
            if (op.name) {
                operations.set(op.name, op);
            }
        }
    }
    const parser = new tablegen.Reader();
    await parser.parse(dialects, paths);
    for (const [, def] of parser.defs) {
        const op = new Operator(def);
        const operationName = op.getOperationName();
        if (!operationName) {
            continue;
        }
        const metadata = {
            name: operationName,
            dialect: op.getDialectName()
        };
        const summary = def.resolveField('summary');
        if (summary && summary.value) {
            metadata.summary = summary.value.value;
        }
        const description = def.resolveField('description');
        if (description && description.value) {
            metadata.description = description.value.value;
        }
        const argsField = def.resolveField('arguments');
        if (argsField && argsField.value && argsField.value.type === 'dag') {
            const dag = argsField.value.value;
            if (dag.operator === 'ins') {
                metadata.inputs = [];
                metadata.attributes = [];
                for (const operand of dag.operands) {
                    if (!operand.value || !operand.name) {
                        continue;
                    }
                    let typeName = '';
                    if (operand.value.type === 'def') {
                        typeName = operand.value.value;
                    } else {
                        // Try to extract from other value types
                        typeName = String(operand.value.value);
                    }
                    if (typeName.includes('Attr')) {
                        metadata.attributes.push({
                            name: operand.name,
                            type: typeName
                        });
                    } else {
                        metadata.inputs.push({
                            name: operand.name,
                            type: typeName
                        });
                    }
                }
            }
        }
        const resultsField = def.resolveField('results');
        if (resultsField && resultsField.value && resultsField.value.type === 'dag') {
            const dag = resultsField.value.value;
            if (dag.operator === 'outs') {
                metadata.outputs = [];
                for (const operand of dag.operands) {
                    if (!operand.value || !operand.name) {
                        continue;
                    }
                    let typeName = '';
                    if (operand.value.type === 'def') {
                        typeName = operand.value.value;
                    } else {
                        typeName = String(operand.value.value);
                    }
                    metadata.outputs.push({
                        name: operand.name,
                        type: typeName
                    });
                }
            }
        }
        const assemblyFormatField = def.resolveField('assemblyFormat');
        if (assemblyFormatField && assemblyFormatField.value) {
            metadata.assemblyFormat = assemblyFormatField.value.value;
        }
        const regionsField = def.resolveField('regions');
        if (regionsField) {
            metadata.hasRegions = true;
        }
        const operation = {};
        if (metadata.name) {
            operation.name = metadata.name;
        }
        if (metadata.category) {
            operation.category = metadata.category;
        }
        if (metadata.summary) {
            let summary = metadata.summary.trim();
            summary = summary.replace(/^"|"$/g, '');
            if (summary) {
                operation.summary = summary;
            }
        }
        if (metadata.description) {
            let desc = metadata.description.trim();
            desc = desc.replace(/^\[\{\s*|\s*\}\]$/g, '');
            desc = desc.trim();
            if (desc) {
                operation.description = desc;
            }
        }
        if (metadata.inputs && metadata.inputs.length > 0) {
            operation.inputs = metadata.inputs;
        }
        if (metadata.outputs && metadata.outputs.length > 0) {
            operation.outputs = metadata.outputs;
        }
        if (metadata.attributes && metadata.attributes.length > 0) {
            operation.attributes = metadata.attributes;
        }
        if (metadata.assemblyFormat) {
            let format = metadata.assemblyFormat.trim();
            format = format.replace(/^\[\{\s*|\s*\}\]$/g, '');
            if (format) {
                operation.assemblyFormat = format;
            }
        }
        if (Object.keys(operation).length > 1) {
            if (!operation.category) {
                const name = operation.name.replace(/^(stablehlo|chlo|affine|linalg|memref|quant|vector|tosa|tfl|tf|onnx|torch)\./, '');
                if (['reshape', 'broadcast_in_dim', 'dynamic_reshape', 'Reshape', 'Shape', 'Size', 'ConstantOfShape'].includes(name)) {
                    operation.category = 'Shape';
                } else if (['transpose', 'reverse', 'pad', 'Transpose', 'Pad'].includes(name)) {
                    operation.category = 'Transform';
                } else if (['slice', 'dynamic_slice', 'gather', 'scatter', 'Slice', 'Gather', 'Scatter'].includes(name)) {
                    operation.category = 'Tensor';
                } else if (['tanh', 'Sigmoid', 'Tanh', 'Relu', 'Softmax', 'softmax', 'sigmoid', 'relu'].includes(name)) {
                    operation.category = 'Activation';
                } else if (['convolution', 'Conv', 'matmul', 'batch_matmul', 'conv2d', 'conv3d', 'fully_connected', 'conv_2d'].includes(name)) {
                    operation.category = 'Layer';
                }
            }
            operations.set(operationName, operation);
        }
    }
    const sorted = Array.from(operations.values()).sort((a, b) => a.name.localeCompare(b.name));
    const output = JSON.stringify(sorted, null, 2);
    const formatted = output.replace(/\{\s+"name":\s+"([^"]+)",\s+"type":\s+"([^"]+)"\s+\}/g, '{ "name": "$1", "type": "$2" }');
    await fs.writeFile(file, formatted, 'utf-8');
};

await main();
