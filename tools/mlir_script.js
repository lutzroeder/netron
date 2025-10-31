
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
    const source = path.join(dirname, '..', 'third_party', 'source', 'mlir');
    const paths = [
        path.join(source, 'llvm-project', 'mlir', 'include'),
        path.join(source, 'llvm-project', 'mlir', 'examples', 'toy', 'Ch7', 'include'),
        path.join(source, 'stablehlo'),
        path.join(source, 'onnx-mlir'),
        path.join(source, 'torch-mlir', 'include'),
        path.join(source, 'mlir-hlo', 'include'),
        path.join(source, 'iree', 'compiler', 'src'),
        path.join(source, 'FlashTensor', 'include'),
        path.join(source, 'tpu-mlir', 'include'),
        path.join(source, 'tensorflow'),
        path.join(source, 'plaidml'),
        path.join(source, 'mlir-dace', 'include'),
        path.join(source, 'lltz', 'mlir', 'dialect', 'include', 'Michelson'),
    ];
    const dialects = [
        'mlir/IR/BuiltinAttributeInterfaces.td',
        'mlir/IR/BuiltinTypeInterfaces.td',
        'mlir/IR/BuiltinLocationAttributes.td',
        'mlir/IR/BuiltinDialect.td',
        'mlir/IR/BuiltinOps.td',
        'mlir/IR/BuiltinDialectBytecode.td',
        'mlir/IR/BuiltinAttributes.td',
        'mlir/IR/BuiltinTypes.td',
        'mlir/Dialect/Async/IR/AsyncOps.td',
        'mlir/Dialect/Affine/IR/AffineOps.td',
        'mlir/Dialect/Affine/IR/AffineOps.td',
        'mlir/Dialect/Arith/IR/ArithOps.td',
        'mlir/Dialect/ControlFlow/IR/ControlFlowOps.td',
        'mlir/Dialect/Func/IR/FuncOps.td',
        'mlir/Dialect/GPU/IR/GPUOps.td',
        'mlir/Dialect/SCF/IR/SCFOps.td',
        'mlir/Dialect/Linalg/IR/LinalgOps.td',
        'mlir/Dialect/MemRef/IR/MemRefOps.td',
        'mlir/Dialect/Bufferization/IR/BufferizationOps.td',
        'mlir/Dialect/Quant/IR/QuantOps.td',
        'mlir/Dialect/Shape/IR/ShapeOps.td',
        'mlir/Dialect/SparseTensor/IR/SparseTensorOps.td',
        'mlir/Dialect/Tensor/IR/TensorOps.td',
        'mlir/Dialect/Tosa/IR/TosaOps.td',
        'mlir/Dialect/Vector/IR/VectorOps.td',
        'mlir/Dialect/X86Vector/X86Vector.td',
        'mlir/Dialect/XeGPU/IR/XeGPUOps.td',
        'mlir/Dialect/Transform/IR/TransformOps.td',
        'mlir/Dialect/WasmSSA/IR/WasmSSAOps.td',
        'mlir/Dialect/IRDL/IR/IRDLOps.td',
        'mlir/Dialect/LLVMIR/LLVMOps.td',
        'mlir/Dialect/Math/IR/MathOps.td',
        'mlir/Dialect/MLProgram/IR/MLProgramOps.td',
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
        'mlir/Dialect/EmitC/IR/EmitC.td',
        'mlir/Dialect/Complex/IR/ComplexOps.td',
        'mlir/Dialect/Index/IR/IndexOps.td',
        'mlir/Dialect/PDL/IR/PDLOps.td',
        'mlir/Dialect/Ptr/IR/PtrOps.td',
        'mlir/Dialect/UB/IR/UBOps.td',
        'mlir/Dialect/AMDGPU/IR/AMDGPU.td',
        'mlir/Dialect/NVGPU/IR/NVGPUOps.td',
        'mlir/Dialect/Shard/IR/ShardOps.td',
        'mlir/Dialect/AMX/AMX.td',
        'mlir/Dialect/SMT/IR/SMTOps.td',
        'mlir/Dialect/SMT/IR/SMTArrayOps.td',
        'mlir/Dialect/SMT/IR/SMTBitVectorOps.td',
        'mlir/Dialect/SMT/IR/SMTIntOps.td',
        'toy/Ops.td',
        'stablehlo/dialect/StablehloOps.td',
        'stablehlo/dialect/ChloOps.td',
        'stablehlo/dialect/VhloOps.td',
        'stablehlo/reference/InterpreterOps.td',
        'src/Dialect/ONNX/ONNX.td',
        'src/Dialect/ONNX/ONNXOps.td.inc',
        'src/Dialect/ONNX/AdditionalONNXOps.td',
        'src/Dialect/Krnl/Krnl.td',
        'torch-mlir/Dialect/Torch/IR/TorchOps.td',
        'torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.td',
        'torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.td',
        'tensorflow/compiler/mlir/lite/ir/tfl_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/tf_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/tf_device_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/tf_executor_ops.td',
        'tensorflow/compiler/mlir/tools/kernel_gen/ir/tf_framework_ops.td',
        'tensorflow/compiler/mlir/tfr/ir/tfr_ops.td',
        'tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback.td',
        'mlir-hlo/Dialect/mhlo/IR/hlo_ops.td',
        'iree/compiler/Dialect/HAL/IR/HALOps.td',
        'iree/compiler/Dialect/Flow/IR/FlowOps.td',
        'iree/compiler/Dialect/Stream/IR/StreamOps.td',
        'iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td',
        'iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td',
        'iree/compiler/Dialect/Util/IR/UtilOps.td',
        'iree/compiler/Dialect/VM/IR/VMOps.td',
        'iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.td',
        'iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.td',
        'iree/compiler/Dialect/Encoding/IR/EncodingOps.td',
        'asuka/Dialect/Asuka/IR/AsukaOps.td',
        'tpu_mlir/Dialect/Top/IR/TopOps.td',
        'tpu_mlir/Dialect/Tpu/IR/TpuOps.td',
        'pmlc/dialect/tile/ir/ops.td',
        'pmlc/dialect/stdx/ir/ops.td',
        'SDFG/Dialect/Ops.td',
        'MichelsonOps.td',
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
        const args = def.resolveField('arguments');
        if (args && args.value && args.value.type === 'dag') {
            const dag = args.value.value;
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
        const results = def.resolveField('results');
        if (results && results.value && results.value.type === 'dag') {
            const dag = results.value.value;
            if (dag.operator === 'outs') {
                metadata.outputs = [];
                for (const operand of dag.operands) {
                    if (!operand.value || !operand.name) {
                        continue;
                    }
                    if (operand.value.type !== 'def') {
                        throw new Error('Unexpected result operand value type');
                    }
                    const type = operand.value.value;
                    metadata.outputs.push({ name: operand.name, type });
                }
            }
        }
        const successors = def.resolveField('successors');
        if (successors && successors.value && successors.value.type === 'dag') {
            const dag = successors.value.value;
            if (dag.operator === 'successor') {
                metadata.successors = [];
                for (const operand of dag.operands) {
                    if (!operand.name) {
                        continue;
                    }
                    metadata.successors.push({ name: operand.name });
                }
            }
        }
        const assemblyFormat = def.resolveField('assemblyFormat');
        if (assemblyFormat && assemblyFormat.value) {
            metadata.assemblyFormat = assemblyFormat.value.value;
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
        if (metadata.successors && metadata.successors.length > 0) {
            operation.successors = metadata.successors;
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
                const name = operation.name.replace(/^(stablehlo|chlo|affine|linalg|memref|quant|vector|tosa|tfl|tf|onnx|torch\.aten|gpu)\./, '');
                if (['reshape', 'broadcast_in_dim', 'dynamic_reshape', 'Reshape', 'Shape', 'Size', 'ConstantOfShape'].indexOf(name) !== -1) {
                    operation.category = 'Shape';
                } else if (['transpose', 'reverse', 'pad', 'Transpose', 'Pad'].indexOf(name) !== -1) {
                    operation.category = 'Transform';
                } else if (['slice', 'dynamic_slice', 'gather', 'scatter', 'Slice', 'Gather', 'Scatter', 'concatenate'].indexOf(name) !== -1) {
                    operation.category = 'Tensor';
                } else if (['tanh', 'Sigmoid', 'Tanh', 'Relu', 'Softmax', 'softmax', 'sigmoid', 'relu'].indexOf(name) !== -1) {
                    operation.category = 'Activation';
                } else if (['convolution', 'Conv', 'matmul', 'batch_matmul', 'conv2d', 'conv3d', 'fully_connected', 'conv_2d'].indexOf(name) !== -1) {
                    operation.category = 'Layer';
                } else if (['batch_norm_inference'].includes(name)) {
                    operation.category = 'Normalization';
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
