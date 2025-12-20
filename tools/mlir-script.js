
import * as child_process from 'child_process';
import * as fs from 'fs/promises';
import * as os from 'os';
import * as path from 'path';
import * as process from 'process';
import * as readline from 'readline';
import * as tablegen from './tablegen.js';
import * as url from 'url';

const write = (message) => {
    if (process.stdout.write) {
        process.stdout.write(message);
    }
};

const writeLine = (message) => {
    write(message + os.EOL);
};

class Operator {

    constructor(def) {
        this.def = def;
        this.opName = def.getValueAsString('opName');
        const dialectDef = this.def.getValueAsDef('opDialect');
        if (dialectDef) {
            this.dialectName = dialectDef.getValueAsString('name');
        }
    }

    getDialectName() {
        return this.dialectName || '';
    }

    getOperationName() {
        return this.dialectName && this.opName ? `${this.dialectName}.${this.opName}` : null;
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

const schema = async () => {
    const dirname = path.dirname(url.fileURLToPath(import.meta.url));
    const source = path.join(dirname, '..', 'third_party', 'source', 'mlir');
    const paths = [
        path.join(source),
        path.join(source, 'llvm-project'),
        path.join(source, 'llvm-project', 'mlir', 'include'),
        path.join(source, 'llvm-project', 'mlir', 'test', 'lib', 'Dialect', 'Test'),
        path.join(source, 'llvm-project', 'mlir', 'test', 'lib', 'Dialect', 'Transform'),
        path.join(source, 'llvm-project', 'mlir', 'test', 'lib', 'Transforms'),
        path.join(source, 'llvm-project', 'mlir', 'include', 'mlir', 'Dialect', 'ArmNeon'),
        path.join(source, 'llvm-project', 'mlir', 'include', 'mlir', 'Dialect', 'ArmSME', 'IR'),
        path.join(source, 'llvm-project', 'mlir', 'include', 'mlir', 'Dialect', 'ArmSVE', 'IR'),
        path.join(source, 'llvm-project', 'mlir', 'examples', 'standalone', 'include'),
        path.join(source, 'llvm-project', 'mlir', 'examples', 'toy', 'Ch7', 'include'),
        path.join(source, 'llvm-project', 'mlir', 'examples', 'transform', 'Ch2', 'include'),
        path.join(source, 'llvm-project', 'mlir', 'examples', 'transform', 'Ch3', 'include'),
        path.join(source, 'stablehlo'),
        path.join(source, 'shardy'),
        path.join(source, 'xla', 'xla', 'mlir_hlo'),
        path.join(source, 'onnx-mlir'),
        path.join(source, 'torch-mlir', 'include'),
        path.join(source, 'triton', 'include'),
        path.join(source, 'triton', 'third_party'),
        path.join(source, 'triton', 'third_party', 'amd', 'include', 'Dialect', 'TritonAMDGPU', 'IR'),
        path.join(source, 'iree', 'compiler', 'src'),
        path.join(source, 'iree', 'compiler', 'src', 'iree', 'compiler', 'Codegen', 'Dialect', 'PCF', 'IR'),
        path.join(source, 'iree', 'compiler', 'src', 'iree', 'compiler', 'Modules', 'IO', 'Parameters', 'IR'),
        path.join(source, 'iree', 'llvm-external-projects', 'iree-dialects', 'include'),
        path.join(source, 'FlashTensor', 'include'),
        path.join(source, 'tpu-mlir', 'include'),
        path.join(source, 'tensorflow'),
        path.join(source, 'tensorflow', 'tensorflow', 'compiler', 'mlir', 'tfrt', 'ir'),
        path.join(source, 'runtime', 'include'),
        path.join(source, 'plaidml'),
        path.join(source, 'plaidml', 'pmlc', 'dialect', 'pxa', 'ir'),
        path.join(source, 'mlir-dace', 'include'),
        path.join(source, 'lltz', 'mlir', 'dialect', 'include', 'Michelson'),
        path.join(source, 'lagrad', 'include', 'LAGrad'),
        path.join(source, 'TensorRT-Incubator', 'mlir-tensorrt', 'tensorrt', 'include'),
        path.join(source, 'TensorRT-Incubator', 'mlir-tensorrt', 'executor', 'include'),
        path.join(source, 'TensorRT-Incubator', 'mlir-tensorrt', 'compiler', 'include'),
        path.join(source, 'TensorRT-Incubator', 'mlir-tensorrt', 'kernel', 'include'),
        path.join(source, 'TensorRT-Incubator', 'mlir-tensorrt', 'common', 'include'),
        path.join(source, 'triton', 'third_party', 'nvidia', 'include'),
        path.join(source, 'triton', 'third_party', 'nvidia', 'include', 'Dialect', 'NVGPU', 'IR'),
        path.join(source, 'triton', 'third_party', 'nvidia', 'include', 'Dialect', 'NVWS', 'IR'),
        path.join(source, '_', 'llvm-project', 'mlir', 'include'),
        path.join(source, 'clangir'),
        path.join(source, 'clangir', 'clang', 'include'),
        path.join(source, 'rocMLIR'),
        path.join(source, 'rocMLIR', 'mlir', 'include'),
    ];
    const dialects = [
        'mlir/include/mlir/IR/BuiltinAttributeInterfaces.td',
        'mlir/include/mlir/IR/BuiltinTypeInterfaces.td',
        'mlir/include/mlir/IR/BuiltinLocationAttributes.td',
        'mlir/include/mlir/IR/BuiltinDialect.td',
        'mlir/include/mlir/IR/BuiltinOps.td',
        'mlir/include/mlir/IR/BuiltinDialectBytecode.td',
        'mlir/include/mlir/IR/BuiltinAttributes.td',
        'mlir/include/mlir/IR/BuiltinTypes.td',
        'mlir/include/mlir/Dialect/Affine/IR/AffineOps.td',
        'mlir/include/mlir/Dialect/Affine/IR/AffineOps.td',
        'mlir/include/mlir/Dialect/Affine/TransformOps/AffineTransformOps.td',
        'mlir/include/mlir/Dialect/AMDGPU/IR/AMDGPU.td',
        'mlir/include/mlir/Dialect/AMX/AMX.td',
        'mlir/include/mlir/Dialect/Arith/IR/ArithOps.td',
        'mlir/include/mlir/Dialect/ArmNeon/ArmNeon.td',
        'mlir/include/mlir/Dialect/ArmNeon/TransformOps/ArmNeonVectorTransformOps.td',
        'mlir/include/mlir/Dialect/ArmSME/IR/ArmSMEOps.td',
        'mlir/include/mlir/Dialect/ArmSVE/IR/ArmSVE.td',
        'mlir/include/mlir/Dialect/ArmSVE/TransformOps/ArmSVEVectorTransformOps.td',
        'mlir/include/mlir/Dialect/Async/IR/AsyncOps.td',
        'mlir/include/mlir/Dialect/Bufferization/IR/BufferizationOps.td',
        'mlir/include/mlir/Dialect/Bufferization/TransformOps/BufferizationTransformOps.td',
        'mlir/include/mlir/Dialect/Complex/IR/ComplexOps.td',
        'mlir/include/mlir/Dialect/ControlFlow/IR/ControlFlowOps.td',
        'mlir/include/mlir/Dialect/DLTI/TransformOps/DLTITransformOps.td',
        'mlir/include/mlir/Dialect/EmitC/IR/EmitC.td',
        'mlir/include/mlir/Dialect/Func/IR/FuncOps.td',
        'mlir/include/mlir/Dialect/Func/TransformOps/FuncTransformOps.td',
        'mlir/include/mlir/Dialect/GPU/IR/GPUOps.td',
        'mlir/include/mlir/Dialect/GPU/TransformOps/GPUTransformOps.td',
        'mlir/include/mlir/Dialect/Index/IR/IndexOps.td',
        'mlir/include/mlir/Dialect/IRDL/IR/IRDLOps.td',
        'mlir/include/mlir/Dialect/Linalg/IR/LinalgOps.td',
        'mlir/include/mlir/Dialect/Linalg/IR/LinalgRelayoutOps.td',
        'mlir/include/mlir/Dialect/Linalg/IR/LinalgStructuredOps.td',
        'mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgMatchOps.td',
        'mlir/include/mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.td',
        'mlir/include/mlir/Dialect/LLVMIR/LLVMIntrinsicOps.td',
        'mlir/include/mlir/Dialect/LLVMIR/LLVMOps.td',
        'mlir/include/mlir/Dialect/LLVMIR/NVVMOps.td',
        'mlir/include/mlir/Dialect/LLVMIR/ROCDLOps.td',
        'mlir/include/mlir/Dialect/LLVMIR/XeVMOps.td',
        'mlir/include/mlir/Dialect/Math/IR/MathOps.td',
        'mlir/include/mlir/Dialect/MemRef/IR/MemRefOps.td',
        'mlir/include/mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.td',
        'mlir/include/mlir/Dialect/MLProgram/IR/MLProgramOps.td',
        'mlir/include/mlir/Dialect/MPI/IR/MPIOps.td',
        'mlir/include/mlir/Dialect/NVGPU/IR/NVGPUOps.td',
        'mlir/include/mlir/Dialect/NVGPU/TransformOps/NVGPUTransformOps.td',
        'mlir/include/mlir/Dialect/OpenACC/OpenACCOps.td',
        'mlir/include/mlir/Dialect/OpenMP/OpenMPOps.td',
        'mlir/include/mlir/Dialect/PDL/IR/PDLOps.td',
        'mlir/include/mlir/Dialect/PDLInterp/IR/PDLInterpOps.td',
        'mlir/include/mlir/Dialect/Ptr/IR/PtrOps.td',
        'mlir/include/mlir/Dialect/Quant/IR/QuantOps.td',
        'mlir/include/mlir/Dialect/SCF/IR/SCFOps.td',
        'mlir/include/mlir/Dialect/SCF/TransformOps/SCFTransformOps.td',
        'mlir/include/mlir/Dialect/Shape/IR/ShapeOps.td',
        'mlir/include/mlir/Dialect/Shard/IR/ShardOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTArrayOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTBitVectorOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTIntOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTOps.td',
        'mlir/include/mlir/Dialect/SparseTensor/IR/SparseTensorOps.td',
        'mlir/include/mlir/Dialect/SparseTensor/TransformOps/SparseTensorTransformOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVArithmeticOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVAtomicOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVBarrierOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVBitOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVCastOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVCLOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVCompositeOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVControlFlowOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVCooperativeMatrixOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVGLOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVGraphOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVGroupOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVImageOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVIntegerDotProductOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVIntelExtOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVLogicalOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVMatrixOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVMemoryOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVMeshOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVMiscOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVNonUniformOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVPrimitiveOps.td',
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVStructureOps.td',
        'mlir/include/mlir/Dialect/Tensor/IR/TensorOps.td',
        'mlir/include/mlir/Dialect/Tensor/TransformOps/TensorTransformOps.td',
        'mlir/include/mlir/Dialect/Tosa/IR/TosaOps.td',
        'mlir/include/mlir/Dialect/Transform/DebugExtension/DebugExtensionOps.td',
        'mlir/include/mlir/Dialect/Transform/IR/TransformOps.td',
        'mlir/include/mlir/Dialect/Transform/IRDLExtension/IRDLExtensionOps.td',
        'mlir/include/mlir/Dialect/Transform/LoopExtension/LoopExtensionOps.td',
        'mlir/include/mlir/Dialect/Transform/PDLExtension/PDLExtensionOps.td',
        'mlir/include/mlir/Dialect/Transform/SMTExtension/SMTExtensionOps.td',
        'mlir/include/mlir/Dialect/Transform/TuneExtension/TuneExtensionOps.td',
        'mlir/include/mlir/Dialect/UB/IR/UBOps.td',
        'mlir/include/mlir/Dialect/Vector/IR/VectorOps.td',
        'mlir/include/mlir/Dialect/Vector/TransformOps/VectorTransformOps.td',
        'mlir/include/mlir/Dialect/WasmSSA/IR/WasmSSAOps.td',
        'mlir/include/mlir/Dialect/X86Vector/X86Vector.td',
        'mlir/include/mlir/Dialect/XeGPU/IR/XeGPUOps.td',
        'mlir/include/mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.td',
        'mlir/examples/toy/Ch7/include/toy/Ops.td',
        'mlir/examples/transform/Ch2/include/MyExtension.td',
        'mlir/examples/transform/Ch3/include/MyExtension.td',
        'stablehlo/dialect/StablehloOps.td',
        'stablehlo/dialect/ChloOps.td',
        'stablehlo/dialect/VhloOps.td',
        'stablehlo/reference/InterpreterOps.td',
        'stablehlo/tests/CheckOps.td',
        'shardy/dialect/sdy/ir/ops.td',
        'shardy/dialect/mpmd/ir/ops.td',
        'mhlo/IR/hlo_ops.td',
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
        'tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_async.td',
        'tensorflow/compiler/mlir/tfrt/ir/tfrt_fallback_sync.td',
        'tensorflow/compiler/mlir/tfrt/ir/mlrt/tf_mlrt_ops.td',
        'tensorflow/compiler/mlir/tensorflow/ir/host_runtime/tfrt_ops.td',
        'tensorflow/compiler/mlir/tfrt/runtime_fallback/runtime_fallback_ops.td',
        'tensorflow/compiler/mlir/tfrt/ir/mlrt/mlrt_ops.td',
        'tfrt/core_runtime/opdefs/core_runtime.td',
        'tfrt/core_runtime/opdefs/sync/core_runtime.td',
        'tfrt/basic_kernels/opdefs/basic_kernels.td',
        'tfrt/test_kernels/opdefs/test_kernels.td',
        'tfrt/tensor/opdefs/tensor.td',
        'tfrt/tensor/opdefs/dense_host_tensor.td',
        'tfrt/tensor/opdefs/tensor_shape.td',
        'mlir/test/lib/Dialect/Test/TestOps.td',
        'mlir/test/lib/Dialect/Test/TestOpsSyntax.td',
        'mlir/test/lib/Dialect/Transform/TestTransformDialectExtension.td',
        'mlir/test/lib/Transforms/TestTransformsOps.td',
        'iree/compiler/Dialect/HAL/IR/HALOps.td',
        'iree/compiler/Dialect/HAL/IR/HALTypes.td',
        'iree/compiler/Modules/HAL/Loader/IR/HALLoaderOps.td',
        'iree/compiler/Modules/HAL/Inline/IR/HALInlineOps.td',
        'iree/compiler/Modules/IO/Parameters/IR/IOParametersOps.td',
        'iree/compiler/Dialect/Flow/IR/FlowOps.td',
        'iree/compiler/Dialect/Stream/IR/StreamOps.td',
        'iree/compiler/Dialect/Util/TransformOps/UtilTransformOps.td',
        'iree/compiler/Codegen/Dialect/GPU/TransformExtensions/IREEGPUExtensionsOps.td',
        'iree/compiler/Codegen/Common/TransformExtensions/CommonExtensionsOps.td',
        'iree/compiler/Codegen/LLVMGPU/TransformExtensions/LLVMGPUExtensionsOps.td',
        'iree/compiler/Dialect/Flow/TransformExtensions/FlowExtensionsOps.td',
        'iree/compiler/Dialect/LinalgExt/TransformExtensions/LinalgExtExtensionsOps.td',
        'iree/compiler/Preprocessing/TransformExtensions/PreprocessingExtensionsOps.td',
        'iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtOps.td',
        'iree/compiler/Codegen/Dialect/PCF/IR/PCFOps.td',
        'iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUOps.td',
        'iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenOps.td',
        'iree/compiler/Codegen/Dialect/Codegen/IR/UKernelOps.td',
        'iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.td',
        'iree/compiler/Dialect/LinalgExt/IR/LinalgExtPureOps.td',
        'iree/compiler/Dialect/TensorExt/IR/TensorExtOps.td',
        'iree/compiler/Dialect/Util/IR/UtilOps.td',
        'iree/compiler/Dialect/VM/IR/VMOps.td',
        'iree/compiler/Dialect/VMVX/IR/VMVXOps.td',
        'iree/compiler/Dialect/Encoding/IR/EncodingOps.td',
        'iree/compiler/src/iree/compiler/Modules/Check/IR/CheckOps.td',
        'iree-dialects/Dialect/LinalgTransform/StructuredTransformOpsExt.td',
        'asuka/Dialect/Asuka/IR/AsukaOps.td',
        'tpu_mlir/Dialect/Top/IR/TopOps.td',
        'tpu_mlir/Dialect/Tpu/IR/TpuOps.td',
        'pmlc/dialect/tile/ir/ops.td',
        'pmlc/dialect/stdx/ir/ops.td',
        'pmlc/dialect/pxa/ir/ops.td',
        'pmlc/dialect/linalgx/ir/ops.td',
        'pmlc/dialect/xsmm/ir/ops.td',
        'SDFG/Dialect/Ops.td',
        'lltz/mlir/dialect/include/Michelson/MichelsonOps.td',
        'triton/Dialect/Triton/IR/TritonOps.td',
        'triton/Dialect/TritonGPU/IR/TritonGPUOps.td',
        'triton/Dialect/Gluon/IR/GluonOps.td',
        'triton/Dialect/TritonNvidiaGPU/IR/TritonNvidiaGPUOps.td',
        'triton/third_party/nvidia/include/Dialect/NVWS/IR/NVWSOps.td',
        'amd/include/Dialect/TritonAMDGPU/IR/TritonAMDGPUOps.td',
        'proton/Dialect/include/Dialect/Proton/IR/ProtonOps.td',
        'proton/Dialect/include/Dialect/ProtonGPU/IR/ProtonGPUOps.td',
        'lagrad/include/LAGrad/LAGradOps.td',
        'mlir-tensorrt-dialect/TensorRT/IR/TensorRTOps.td',
        'mlir-executor/Executor/IR/ExecutorOps.td',
        'mlir-tensorrt/Dialect/CUDA/IR/CUDAOps.td',
        'mlir-tensorrt/Dialect/TensorRTRuntime/IR/TensorRTRuntimeOps.td',
        'mlir-tensorrt/Dialect/Plan/IR/PlanOps.td',
        'mlir-kernel/Kernel/IR/Ops.td',
        'Dialect/NVGPU/IR/NVGPUOps.td',
        'Standalone/StandaloneOps.td',
        'clang/include/clang/CIR/Dialect/IR/CIROps.td',
        'mlir/include/mlir/Dialect/MIGraphX/IR/MIGraphX.td',
    ];
    const file = path.join(dirname, '..', 'source', 'mlir-metadata.json');
    const operations = new Map();
    const exists = await access(file);
    if (exists) {
        const content = await fs.readFile(file, 'utf-8');
        const json = JSON.parse(content);
        for (const op of json) {
            if (op.name.endsWith('.') || op.name.includes('..') || op.name.includes('#')) {
                throw new Error(`Invalid operation name '${op.name}'.`);
            }
            if (op.name && !op.name.endsWith('.')) {
                operations.set(op.name, op);
            }
        }
    }
    let count = 0;
    const parser = new tablegen.Reader();
    await parser.parse(dialects, paths);
    for (const def of parser.defs) {
        const op = new Operator(def);
        let operationName = op.getOperationName();
        if (!operationName) {
            continue;
        }
        if (operationName.endsWith('.') || operationName.includes('..') || operationName.includes('#')) {
            throw new Error(`Invalid operation name '${operationName}'.`);
        }
        // Workaround: Handle conflicting dialects from stablehlo and iree
        if (operationName.startsWith('check.')) {
            if (def.location.file.includes('stablehlo')) {
                operationName = operationName.replace(/^check./, 'check.<stablehlo>.');
            } else if (def.location.file.includes('iree')) {
                operationName = operationName.replace(/^check./, 'check.<iree>.');
            }
        }
        const operation = {
            name: operationName
        };
        if (operations.has(operationName)) {
            const existing = operations.get(operationName);
            if (existing.category) {
                operation.category = existing.category;
            }
        }
        let args = def.getValueAsDag('arguments');
        if (!args || !args.operands || args.operands.length === 0) {
            // Try to get from parent Arguments class
            for (const parent of def.parents) {
                if (parent.name === 'Arguments' && parent.args && parent.args.length > 0) {
                    const [dagValue] = parent.args;
                    if (dagValue && dagValue.type === 'dag') {
                        args = dagValue.value;
                    }
                    break;
                }
            }
        }
        const name = operation.name.replace(/^(asuka|stablehlo|chlo|affine|linalg|memref|quant|vector|tosa|tfl|tf|onnx|torch\.aten|gpu)\./, '');
        if (['reshape', 'broadcast_in_dim', 'dynamic_reshape', 'Reshape', 'Shape', 'Size', 'ConstantOfShape'].indexOf(name) !== -1) {
            operation.category = 'Shape';
        } else if (['transpose', 'reverse', 'pad', 'Transpose', 'Pad'].indexOf(name) !== -1) {
            operation.category = 'Transform';
        } else if (['slice', 'split', 'dynamic_slice', 'gather', 'scatter', 'Slice', 'Gather', 'Scatter', 'concatenate'].indexOf(name) !== -1) {
            operation.category = 'Tensor';
        } else if (['tanh', 'Sigmoid', 'Tanh', 'Relu', 'Softmax', 'softmax', 'sigmoid', 'relu'].indexOf(name) !== -1) {
            operation.category = 'Activation';
        } else if (['convolution', 'Conv', 'conv2d', 'conv3d', 'fully_connected', 'conv_2d'].indexOf(name) !== -1) {
            operation.category = 'Layer';
        } else if (['batch_norm_inference'].includes(name)) {
            operation.category = 'Normalization';
        }
        if (def.getValue('summary')) {
            const summary = def.getValueAsString('summary').trim();
            if (summary) {
                operation.summary = summary;
            }
        }
        if (def.getValue('description')) {
            const description = def.getValueAsString('description');
            if (description) {
                operation.description = description;
            }
        }
        // Convert TableGen value to constraint string
        const toConstraintString = (value) => {
            if (!value) {
                return null;
            }
            if (value.type === 'def') {
                const defName = value.value;
                // Check if this is an enum attribute or property and add enum cases
                const attrDef = parser.getDef(defName) || parser.getClass(defName);
                // Resolve type aliases to their base forms for proper parser dispatch
                if (attrDef) {
                    for (const parent of attrDef.parents) {
                        // ConfinedAttr<BaseAttr, Constraints> -> ConfinedAttr<BaseAttr, Constraints>
                        if (parent.name === 'ConfinedAttr' && parent.args && parent.args.length > 0) {
                            const args = parent.args.map((arg) => toConstraintString(arg)).filter((x) => x !== null);
                            return `ConfinedAttr<${args.join(', ')}>`;
                        }
                        // TypedArrayAttrBase<ElementType, ...> -> TypedArrayAttrBase<ElementType>
                        if (parent.name === 'TypedArrayAttrBase' && parent.args && parent.args.length > 0) {
                            const innerType = toConstraintString(parent.args[0]);
                            return innerType ? `TypedArrayAttrBase<${innerType}>` : 'ArrayAttr';
                        }
                        // Variadic<Type> -> Variadic<Type>
                        if (parent.name === 'Variadic' && parent.args && parent.args.length > 0) {
                            const innerType = toConstraintString(parent.args[0]);
                            return innerType ? `Variadic<${innerType}>` : 'Variadic';
                        }
                        // Optional<Type> -> Optional<Type>
                        if (parent.name === 'Optional' && parent.args && parent.args.length > 0) {
                            const innerType = toConstraintString(parent.args[0]);
                            return innerType ? `Optional<${innerType}>` : 'Optional';
                        }
                    }
                }
                if (attrDef && (attrDef.isEnumAttr() || attrDef.isEnumProp())) {
                    const cases = attrDef.getEnumCases();
                    if (cases && cases.length > 0) {
                        return `${defName}{${cases.join('|')}}`;
                    }
                }
                return defName;
            }
            if (value.type === 'string' || value.type === 'code') {
                return value.value;
            }
            if (value.type === 'int') {
                return String(value.value);
            }
            if (value.type === 'list') {
                const items = value.value.map((item) => toConstraintString(item)).filter((x) => x !== null);
                return items.length > 0 ? `[${items.join(', ')}]` : null;
            }
            if (value.type === 'dag' && value.value) {
                const dag = value.value;
                // Unwrap Arg/Res wrappers - they just hold the type constraint plus metadata
                // The first operand is the actual type constraint
                if ((dag.operator === 'Arg' || dag.operator === 'Res') && dag.operands.length > 0) {
                    return toConstraintString(dag.operands[0].value);
                }
                const args = dag.operands.map((op) => toConstraintString(op.value)).filter((x) => x !== null);
                if (args.length > 0) {
                    return `${dag.operator}<${args.join(', ')}>`;
                }
                return dag.operator;
            }
            return null;
        };
        const attributes = [];
        const inputs = [];
        const outputs = [];
        if (args && args.operator === 'ins') {
            for (const operand of args.operands) {
                if (operand.value && operand.name) {
                    const type = toConstraintString(operand.value);
                    // Check if this is an actual attribute/property constraint by looking up the def
                    // and checking if it inherits from Attr or Property class (matches LLVM reference)
                    const checkIsAttr = (record, visited = new Set()) => {
                        if (!record || visited.has(record.name)) {
                            return false;
                        }
                        visited.add(record.name);
                        // Check for attribute base classes - both constraint and bytecode encoding types
                        if (record.name === 'Attr' || record.name === 'AttributeKind' || record.name === 'DialectAttribute') {
                            return true;
                        }
                        // Check for property base classes - properties are also attributes (compile-time metadata)
                        if (record.name === 'Property' || record.name === 'PropConstraint' || record.name === 'EnumProp') {
                            return true;
                        }
                        for (const parent of record.parents) {
                            const parentClass = parser.getClass(parent.name);
                            if (parentClass && checkIsAttr(parentClass, visited)) {
                                return true;
                            }
                        }
                        return false;
                    };
                    let isAttribute = false;
                    if (operand.value) {
                        if (operand.value.type === 'def') {
                            // Simple def reference like StrAttr
                            const constraintDef = parser.getDef(operand.value.value) || parser.getClass(operand.value.value);
                            if (constraintDef) {
                                isAttribute = checkIsAttr(constraintDef);
                            }
                        } else if (operand.value.type === 'dag' && operand.value.value) {
                            // DAG constraint like OptionalAttr<StrAttr>
                            const dag = operand.value.value;
                            // Check the operator (e.g., OptionalAttr, DefaultValuedAttr)
                            const operatorDef = parser.getDef(dag.operator) || parser.getClass(dag.operator);
                            if (operatorDef && checkIsAttr(operatorDef)) {
                                isAttribute = true;
                            } else if (dag.operands && dag.operands.length > 0) {
                                // Check the first operand (the wrapped type)
                                const innerValue = dag.operands[0].value;
                                if (innerValue && innerValue.type === 'def') {
                                    const innerDef = parser.getDef(innerValue.value) || parser.getClass(innerValue.value);
                                    if (innerDef && checkIsAttr(innerDef)) {
                                        isAttribute = true;
                                    }
                                }
                            }
                        }
                    }
                    if (isAttribute) {
                        attributes.push({ name: operand.name, type });
                    } else {
                        inputs.push({ name: operand.name, type });
                    }
                }
            }
        }
        // If no attributes were found from 'arguments', try to extract from 'builders'
        // Some ops (like krnl.entry_point) define attributes only via builders
        if (attributes.length === 0) {
            const buildersField = def.getValue('builders');
            if (buildersField && buildersField.value && buildersField.value.type === 'list') {
                const buildersList = buildersField.value.value;
                if (buildersList && buildersList.length > 0) {
                    // Try to get attribute name mappings from extraClassDeclaration
                    const attrNameMap = new Map();
                    const extraDecl = def.getValueAsString('extraClassDeclaration');
                    if (extraDecl) {
                        // Match patterns like: static StringRef getXxxAttrName() { return "yyy"; }
                        const matches = extraDecl.matchAll(/get(\w+)AttrName\(\)\s*\{\s*return\s+"(\w+)"/g);
                        for (const match of matches) {
                            // Map parameter name (camelCase) to actual attribute name
                            const paramName = match[1].charAt(0).toLowerCase() + match[1].slice(1);
                            attrNameMap.set(paramName, match[2]);
                        }
                    }
                    // Parse first OpBuilder<(ins ...)> to extract attributes
                    for (const builder of buildersList) {
                        if (builder.type === 'dag' && builder.value && builder.value.operator === 'OpBuilder') {
                            const builderOperands = builder.value.operands;
                            if (builderOperands && builderOperands.length > 0) {
                                const [insArg] = builderOperands;
                                if (insArg && insArg.value && insArg.value.type === 'dag' && insArg.value.value.operator === 'ins') {
                                    const insDag = insArg.value.value;
                                    for (const param of insDag.operands) {
                                        // Builder parameters have string types like "SymbolRefAttr":$name
                                        if (param.value && param.value.type === 'string' && param.name) {
                                            const cppType = param.value.value;
                                            // Only include Attr types (not Value or other C++ types)
                                            if (cppType.endsWith('Attr') || cppType.includes('Attr:')) {
                                                // Strip trailing Attr suffix variations and get base attr name
                                                const attrType = cppType.replace(/::$/, '').replace(/:.*$/, '');
                                                // Use the mapped name if available, otherwise derive from param name
                                                let attrName = param.name;
                                                // Remove common suffixes like 'Attr' from the parameter name
                                                const cleanParamName = attrName.replace(/Attr$/, '');
                                                if (attrNameMap.has(cleanParamName)) {
                                                    attrName = attrNameMap.get(cleanParamName);
                                                } else if (attrNameMap.has(attrName)) {
                                                    attrName = attrNameMap.get(attrName);
                                                } else {
                                                    attrName = cleanParamName;
                                                }
                                                attributes.push({ name: attrName, type: attrType });
                                            }
                                        }
                                    }
                                    break; // Only process first builder
                                }
                            }
                        }
                    }
                }
            }
        }
        let results = def.getValueAsDag('results');
        if (!results || !results.operands || results.operands.length === 0) {
            // Try to get from parent Results class
            for (const parent of def.parents) {
                if (parent.name === 'Results' && parent.args && parent.args.length > 0) {
                    const [dagValue] = parent.args;
                    if (dagValue && dagValue.type === 'dag') {
                        results = dagValue.value;
                    }
                    break;
                }
            }
        }
        if (results && results.operator === 'outs') {
            for (const operand of results.operands) {
                if (operand.value && operand.name) {
                    const type = toConstraintString(operand.value);
                    outputs.push({ name: operand.name, type });
                }
            }
        }
        if (inputs.length > 0) {
            operation.inputs = inputs;
        }
        if (outputs.length > 0) {
            operation.outputs = outputs;
        }
        if (attributes.length > 0) {
            operation.attributes = attributes;
        }
        const successors = def.getValueAsDag('successors');
        if (successors && successors.operator === 'successor') {
            const list = [];
            for (const operand of successors.operands) {
                if (operand.name) {
                    list.push({ name: operand.name });
                }
            }
            if (list.length > 0) {
                operation.successors = list;
            }
        }
        const regions = def.getValueAsDag('regions');
        if (regions && regions.operator === 'region') {
            const list = [];
            for (const operand of regions.operands) {
                if (operand.name) {
                    const type = toConstraintString(operand.value);
                    list.push({ name: operand.name, type });
                }
            }
            if (list.length > 0) {
                operation.regions = list;
            }
        }
        if (def.getValue('assemblyFormat')) {
            const assemblyFormat = def.getValueAsString('assemblyFormat');
            if (assemblyFormat) {
                operation.assemblyFormat = assemblyFormat.trim();
            }
        }
        if (def.getValue('hasCustomAssemblyFormat') && def.getValueAsBit('hasCustomAssemblyFormat')) {
            operation.hasCustomAssemblyFormat = true;
        }
        if (def.getValue('parser')) {
            operation.parser = def.getValueAsString('parser');
        }
        // Extract defaultDialect from OpAsmOpInterface
        for (const parent of def.parents) {
            const possibleTraitArgs = parent.args && parent.args.length >= 2 ? [parent.args[1], parent.args[2]] : [];
            for (const traitsArg of possibleTraitArgs) {
                if (traitsArg && traitsArg.type === 'list' && traitsArg.value) {
                    for (const trait of traitsArg.value) {
                        const traitName = trait.type === 'def' ? trait.value : null;
                        const traitDag = trait.type === 'dag' && trait.value?.operator ? trait.value.operator : null;
                        if (traitName === 'OpAsmOpInterface' || traitDag === 'DeclareOpInterfaceMethods') {
                            if (traitDag === 'DeclareOpInterfaceMethods' && trait.value?.operands) {
                                if (trait.value.operands.some((operand) => operand.value && operand.value.type === 'list' && operand.value.value.some((method) => method.type === 'string' && method.value === 'getDefaultDialect'))) {
                                    const [dialectName] = operationName.split('.');
                                    operation.defaultDialect = dialectName;
                                    break;
                                }
                            }
                            const extraClass = def.getValueAsString('extraClassDeclaration');
                            if (extraClass) {
                                const match = extraClass.match(/getDefaultDialect\(\)\s*\{\s*return\s+"(\w+)"/);
                                if (match) {
                                    [, operation.defaultDialect] = match;
                                    break;
                                }
                            }
                        }
                    }
                    if (operation.defaultDialect) {
                        break;
                    }
                }
            }
            if (operation.defaultDialect) {
                break;
            }
        }
        if (Object.keys(operation).length > 1) {
            operations.set(operationName, operation);
            count++;
        }
    }
    const sorted = Array.from(operations.values()).sort((a, b) => a.name.localeCompare(b.name));
    const output = JSON.stringify(sorted, null, 2);
    const formatted = output.replace(/\{\s+"name":\s+"([^"]+)",\s+"type":\s+"([^"]+)"\s+\}/g, '{ "name": "$1", "type": "$2" }');
    await fs.writeFile(file, formatted, 'utf-8');
    if (count < 6300) {
        throw new Error(`Unexpected operation count '${count}'.`);
    }
};

const test = async (pattern) => {
    pattern = pattern || './third_party/source/mlir/**/*.mlir';
    const errorTotals = new Map();
    const filesByError = new Map();
    const fileErrorDetails = new Map();
    let currentFile = null;
    const validFiles = new Set();
    const invalidFiles = new Set([
        'third_party/source/mlir/stablehlo/stablehlo/tests/ops_stablehlo.mlir',
        'third_party/source/mlir/stablehlo/stablehlo/tests/print_types_invalid.mlir',
        'third_party/source/mlir/stablehlo/stablehlo/tests/vhlo/invalid_vhlo_future.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/tensorflow/tests/tf_executor_ops_invalid.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Quant/parse-uniform-invalid.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/map.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/simple_sdfg.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/symbol.mlir',
    ]);
    return new Promise((resolve, reject) => {
        const cmd = 'node';
        const args = ['--max-old-space-size=8192', './test/models.js', 'continue', pattern];
        const process = child_process.spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        const stdout = readline.createInterface({ input: process.stdout, crlfDelay: Infinity });
        const stderr = readline.createInterface({ input: process.stderr, crlfDelay: Infinity });
        const processLine = (line) => {
            writeLine(line);
            const stripped = line.trim();
            if (!stripped) {
                return;
            }
            if (stripped.startsWith('third_party/')) {
                currentFile = stripped;
                if (stripped.toLowerCase().includes('invalid') || stripped.startsWith('third_party/source/mlir/mlir-dace/design')) {
                    invalidFiles.add(currentFile);
                } else {
                    validFiles.add(currentFile);
                }
                return;
            }
            if (currentFile && invalidFiles.has(currentFile)) {
                return;
            }
            if (currentFile && !invalidFiles.has(currentFile)) {
                // Skip summary lines (e.g., "123 / 456 = 78.9%")
                if (/^\s*\d+\s*\/\s*\d+\s*=\s*[\d.]+%\s*$/.test(stripped)) {
                    return;
                }
                // Normalize error message
                const key = stripped.split(' at ', 1)[0].trim().replace(/\.$/, '').trim();
                if (key) {
                    errorTotals.set(key, (errorTotals.get(key) || 0) + 1);
                    if (!filesByError.has(key)) {
                        filesByError.set(key, new Map());
                    }
                    const fileCounts = filesByError.get(key);
                    fileCounts.set(currentFile, (fileCounts.get(currentFile) || 0) + 1);
                    if (!fileErrorDetails.has(key)) {
                        fileErrorDetails.set(key, new Map());
                    }
                    const details = fileErrorDetails.get(key);
                    if (!details.has(currentFile)) {
                        details.set(currentFile, []);
                    }
                    details.get(currentFile).push(stripped);
                }
            }
        };
        stdout.on('line', processLine);
        stderr.on('line', processLine);
        process.on('error', (error) => {
            reject(new Error(`Failed to start process: ${error.message}`));
        });
        process.on('close', (/* code */) => {
            const totalValid = validFiles.size;
            const filesWithErrors = new Set();
            for (const [, fileCounts] of filesByError) {
                for (const file of fileCounts.keys()) {
                    filesWithErrors.add(file);
                }
            }
            writeLine('');
            writeLine('-'.repeat(75));
            if (errorTotals.size > 0) {
                const sortedErrors = Array.from(errorTotals.entries()).sort((a, b) => b[1] - a[1]).slice(0, 100);
                for (const [err, cnt] of sortedErrors) {
                    const fileCounts = filesByError.get(err);
                    const topFiles = Array.from(fileCounts.entries()).sort((a, b) => b[1] - a[1]).slice(0, 100);
                    writeLine(`${cnt}  |  ${err}`);
                    for (const [file,] of topFiles) {
                        writeLine(`  ${file}`);
                        const details = fileErrorDetails.get(err).get(file);
                        for (const specificError of details) {
                            writeLine(`    ${specificError}`);
                        }
                    }
                    writeLine('');
                }
            }
            if (totalValid > 0) {
                const succeeded = totalValid - filesWithErrors.size;
                const percentage = (succeeded * 100.0) / totalValid;
                writeLine(`  ${succeeded} / ${totalValid} =  ${percentage.toPrecision(6)}%  - skipped ${invalidFiles.size} files`);
            } else {
                writeLine('  No valid files processed');
            }
            writeLine();
            resolve();
        });
    });
};

const main = async () => {
    const command = process.argv.length >= 3 ? process.argv[2] : 'schema';
    switch (command) {
        case 'test': {
            writeLine(process.argv);
            await test(process.argv.slice(3).join(' '));
            break;
        }
        case 'schema':
        default: {
            await schema();
            break;
        }
    }
};

await main();
