
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
        path.join(source, 'llvm-project', 'mlir', 'examples', 'transform', 'Ch4', 'include'),
        path.join(source, 'stablehlo'),
        path.join(source, 'shardy'),
        path.join(source, 'xla', 'xla', 'mlir_hlo'),
        path.join(source, 'xla'),
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
        path.join(source, 'tensorflow', 'tensorflow', 'core', 'ir'),
        path.join(source, 'tensorflow', 'tensorflow', 'compiler', 'mlir', 'tfrt', 'ir'),
        path.join(source, 'xla', 'xla', 'backends', 'gpu', 'codegen', 'triton', 'ir'),
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
        path.join(source, 'clangir'),
        path.join(source, 'clangir', 'clang', 'include'),
        path.join(source, 'rocMLIR'),
        path.join(source, 'rocMLIR', 'mlir', 'include'),
        path.join(source, 'ensemble-compilation', 'lib', 'Dialect', 'Ensemble'),
        path.join(source, 'mlir-tutorial', 'lib', 'Dialect', 'Poly'),
        path.join(source, 'mlir-tutorial', 'lib', 'Dialect', 'Noisy'),
        path.join(source, '_', 'llvm-project', 'mlir', 'include'),
        path.join(source, '_', 'mlir-hlo'),
    ];
    const dialects = [
        'pmlc/dialect/tile/ir/ops.td',
        'pmlc/dialect/stdx/ir/ops.td',
        'pmlc/dialect/pxa/ir/ops.td',
        'pmlc/dialect/linalgx/ir/ops.td',
        'pmlc/dialect/xsmm/ir/ops.td',
        'pmlc/dialect/layer/ir/ops.td',
        'mlir/include/mlir/IR/BuiltinAttributeInterfaces.td',
        'mlir/include/mlir/IR/BuiltinTypeInterfaces.td',
        'mlir/include/mlir/IR/BuiltinLocationAttributes.td',
        'mlir/include/mlir/IR/BuiltinDialect.td',
        'mlir/include/mlir/IR/BuiltinOps.td',
        'mlir/include/mlir/IR/BuiltinDialectBytecode.td',
        'mlir/include/mlir/IR/BuiltinAttributes.td',
        'mlir/include/mlir/IR/BuiltinTypes.td',
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
        'mlir/include/mlir/Dialect/SMT/IR/SMTOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTArrayOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTBitVectorOps.td',
        'mlir/include/mlir/Dialect/SMT/IR/SMTIntOps.td',
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
        'mlir/include/mlir/Dialect/SPIRV/IR/SPIRVTosaOps.td',
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
        'mlir/include/mlir/Dialect/X86Vector/TransformOps/X86VectorTransformOps.td',
        'mlir/include/mlir/Dialect/XeGPU/IR/XeGPUOps.td',
        'mlir/include/mlir/Dialect/XeGPU/TransformOps/XeGPUTransformOps.td',
        'mlir/examples/toy/Ch7/include/toy/Ops.td',
        'mlir/examples/transform/Ch2/include/MyExtension.td',
        'mlir/examples/transform/Ch3/include/MyExtension.td',
        'mlir/examples/transform/Ch4/include/MyExtension.td',
        'stablehlo/dialect/StablehloOps.td',
        'stablehlo/dialect/ChloOps.td',
        'stablehlo/dialect/VhloOps.td',
        'stablehlo/reference/InterpreterOps.td',
        'stablehlo/tests/CheckOps.td',
        'shardy/dialect/sdy/ir/ops.td',
        'shardy/dialect/mpmd/ir/ops.td',
        'mhlo/IR/hlo_ops.td',
        'thlo/IR/thlo_ops.td',
        'src/Dialect/ONNX/ONNX.td',
        'src/Dialect/ONNX/ONNXOps.td.inc',
        'src/Dialect/ONNX/AdditionalONNXOps.td',
        'src/Dialect/Krnl/Krnl.td',
        'torch-mlir/Dialect/Torch/IR/TorchOps.td',
        'torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.td',
        'torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.td',
        'tensorflow/core/ir/ops.td',
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
        'tfrt/tensor/opdefs/coo_host_tensor.td',
        'tfrt/tensor/opdefs/tensor_shape.td',
        'mlir/test/lib/Dialect/Test/TestOps.td',
        'mlir/test/lib/Dialect/Test/TestOpsSyntax.td',
        'mlir/test/lib/Dialect/Transform/TestTransformDialectExtension.td',
        'mlir/test/lib/Interfaces/TilingInterface/TestTilingInterfaceTransformOps.td',
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
        'SDFG/Dialect/Ops.td',
        'lltz/mlir/dialect/include/Michelson/MichelsonOps.td',
        'triton/Dialect/Triton/IR/TritonOps.td',
        'triton/Dialect/TritonGPU/IR/TritonGPUOps.td',
        'triton/Dialect/TritonInstrument/IR/TritonInstrumentOps.td',
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
        'mlir-kernel/Kernel/TransformOps/KernelTransformOps.td',
        'Dialect/NVGPU/IR/NVGPUOps.td',
        'Standalone/StandaloneOps.td',
        'clang/include/clang/CIR/Dialect/IR/CIROps.td',
        'mlir/include/mlir/Dialect/MIGraphX/IR/MIGraphX.td',
        'xla/backends/cpu/codegen/emitters/ir/xla_cpu_ops.td',
        'xla/backends/gpu/codegen/emitters/ir/xla_gpu_ops.td',
        'xla/backends/gpu/codegen/triton/ir/triton_xla_ops.td',
        'xla/codegen/emitters/ir/xla_ops.td',
        'xla/codegen/xtile/ir/xtile_ops.td',
        'xla/python/ifrt/ir/ifrt_ops.td',
        'xla/python/ifrt/ir/vifrt_ops.td',
        'xla/xla/mlir/framework/ir/xla_framework_ops.td',
        'ensemble-compilation/lib/Dialect/Ensemble/EnsembleOps.td',
        'mlir-tutorial/lib/Dialect/Poly/PolyOps.td',
        'NoisyOps.td',
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
        if (['reshape', 'broadcast_in_dim', 'dynamic_reshape', 'Reshape', 'Shape', 'Size', 'ConstantOfShape'].includes(name)) {
            operation.category = 'Shape';
        } else if (['transpose', 'reverse', 'pad', 'Transpose', 'Pad'].includes(name)) {
            operation.category = 'Transform';
        } else if (['slice', 'split', 'dynamic_slice', 'gather', 'scatter', 'Slice', 'Gather', 'Scatter', 'concat', 'concatenate'].includes(name)) {
            operation.category = 'Tensor';
        } else if (['tanh', 'Sigmoid', 'Tanh', 'Relu', 'Softmax', 'softmax', 'sigmoid', 'relu', 'clamp'].includes(name)) {
            operation.category = 'Activation';
        } else if (['convolution', 'Conv', 'conv2d', 'conv3d', 'fully_connected', 'conv_2d'].includes(name)) {
            operation.category = 'Layer';
        } else if (['max_pool2d', 'MaxPoolSingleOut'].includes(name)) {
            operation.category = 'Pool';
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
        const operands = [];
        const results = [];
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
                        // Check for enum attribute base classes (e.g., AtomicBinOp, AtomicOrdering)
                        // Inheritance chain: LLVM_EnumAttr -> I64EnumAttr -> IntEnumAttr -> EnumAttrInfo
                        if (record.name === 'EnumAttrInfo' || record.name === 'IntEnumAttr' ||
                            record.name === 'I32EnumAttr' || record.name === 'I64EnumAttr' ||
                            record.name === 'IntEnumAttrBase' || record.name === 'SignlessIntegerAttrBase' ||
                            record.name === 'BitEnumAttr' || record.name === 'I32BitEnumAttr' ||
                            record.name === 'I64BitEnumAttr') {
                            return true;
                        }
                        for (const parent of record.parents || []) {
                            const parentClass = parser.getClass(parent.name);
                            if (parentClass && checkIsAttr(parentClass, visited)) {
                                return true;
                            }
                            // Fallback: if parent class not found, check if the parent name itself
                            // matches known enum attribute base classes (handles cases where the
                            // base class definition wasn't parsed from included files)
                            if (!parentClass && parent.name) {
                                if (parent.name === 'I64EnumAttr' || parent.name === 'I32EnumAttr' ||
                                    parent.name === 'IntEnumAttr' || parent.name === 'EnumAttrInfo' ||
                                    parent.name === 'BitEnumAttr' || parent.name === 'I32BitEnumAttr' ||
                                    parent.name === 'I64BitEnumAttr' || parent.name.endsWith('EnumAttr')) {
                                    return true;
                                }
                            }
                        }
                        return false;
                    };
                    // Recursively check if a value represents an attribute type
                    const checkValueIsAttr = (value) => {
                        if (!value) {
                            return false;
                        }
                        if (value.type === 'def') {
                            // Simple def reference like StrAttr
                            const constraintDef = parser.getDef(value.value) || parser.getClass(value.value);
                            if (constraintDef) {
                                return checkIsAttr(constraintDef);
                            }
                            // Fallback: if definition not found but name ends with "Attr", treat as attribute
                            // This handles cases like CancellationConstructTypeAttr which are generated/external
                            if (typeof value.value === 'string' && value.value.endsWith('Attr')) {
                                return true;
                            }
                            // Also check for enum predicates (like IntPredicate, CmpIPredicate) which inherit from
                            // I64EnumAttr but don't end with "Attr" - look for the Predicate suffix pattern
                            if (typeof value.value === 'string' && value.value.endsWith('Predicate')) {
                                return true;
                            }
                            return false;
                        }
                        if (value.type === 'dag' && value.value) {
                            // DAG constraint like OptionalAttr<StrAttr> or Arg<OptionalAttr<ArrayAttr>, ...>
                            const dag = value.value;
                            // Check the operator (e.g., OptionalAttr, DefaultValuedAttr)
                            const operatorDef = parser.getDef(dag.operator) || parser.getClass(dag.operator);
                            if (operatorDef && checkIsAttr(operatorDef)) {
                                return true;
                            }
                            // Fallback: if operator name ends with "Attr", treat as attribute
                            if (typeof dag.operator === 'string' && dag.operator.endsWith('Attr')) {
                                return true;
                            }
                            // For wrappers like Arg<...>, check the first operand recursively
                            if (dag.operands && dag.operands.length > 0) {
                                return checkValueIsAttr(dag.operands[0].value);
                            }
                        }
                        return false;
                    };
                    const isAttribute = checkValueIsAttr(operand.value);
                    if (isAttribute) {
                        attributes.push({ name: operand.name, type });
                    } else {
                        operands.push({ name: operand.name, type });
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
        let resultsDag = def.getValueAsDag('results');
        if (!resultsDag || !resultsDag.operands || resultsDag.operands.length === 0) {
            // Try to get from parent Results class
            for (const parent of def.parents) {
                if (parent.name === 'Results' && parent.args && parent.args.length > 0) {
                    const [dagValue] = parent.args;
                    if (dagValue && dagValue.type === 'dag') {
                        resultsDag = dagValue.value;
                    }
                    break;
                }
            }
        }
        if (resultsDag && resultsDag.operator === 'outs') {
            for (let i = 0; i < resultsDag.operands.length; i++) {
                const operand = resultsDag.operands[i];
                if (operand.value) {
                    const type = toConstraintString(operand.value);
                    // Use operand name if present, otherwise generate default name
                    const name = operand.name || (resultsDag.operands.length === 1 ? 'result' : `result${i}`);
                    results.push({ name, type });
                }
            }
        }
        if (operands.length > 0) {
            operation.operands = operands;
        }
        if (results.length > 0) {
            operation.results = results;
        }
        if (attributes.length > 0) {
            operation.attributes = attributes;
        }
        const successors = def.getValueAsDag('successors');
        if (successors && successors.operator === 'successor') {
            const list = [];
            for (const operand of successors.operands) {
                if (operand.name) {
                    const type = toConstraintString(operand.value);
                    list.push({ name: operand.name, type });
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
        // Extract traits (isolation, type inference, operand segments, and interface traits)
        const traits = [];
        const extractTraitsFromList = (traitsArg) => {
            if (!traitsArg) {
                return;
            }
            // Handle concat expressions: traits # [list of traits]
            if (traitsArg.type === 'concat' && Array.isArray(traitsArg.value)) {
                for (const element of traitsArg.value) {
                    extractTraitsFromList(element);
                }
                return;
            }
            // Handle !listconcat expressions (stored as bang type)
            if (traitsArg.type === 'bang' && traitsArg.value) {
                const bangOp = traitsArg.value.operator || traitsArg.value.op;
                if (bangOp === 'listconcat' && traitsArg.value.args) {
                    for (const arg of traitsArg.value.args) {
                        extractTraitsFromList(arg);
                    }
                    return;
                }
            }
            // Handle arrays directly (evaluated listconcat results)
            if (Array.isArray(traitsArg)) {
                for (const element of traitsArg) {
                    extractTraitsFromList({ type: 'list', value: [element] });
                }
                return;
            }
            if (traitsArg.type === 'list' && traitsArg.value) {
                for (const trait of traitsArg.value) {
                    const traitName = trait.type === 'def' ? trait.value : null;
                    const traitDag = trait.type === 'dag' && trait.value && trait.value.operator ? trait.value.operator : null;
                    // Extract AllTypesMatch traits as string - parsed on demand in getOperation
                    if (traitDag === 'AllTypesMatch' && trait.value && trait.value.operands) {
                        const [namesOperand] = trait.value.operands;
                        if (namesOperand && namesOperand.value && namesOperand.value.type === 'list') {
                            const names = namesOperand.value.value.filter((v) => v.type === 'string').map((v) => v.value);
                            if (names.length > 0) {
                                const traitType = `AllTypesMatch<[${names.map((n) => `'${n}'`).join(', ')}]>`;
                                if (traits.every((t) => t.type !== traitType)) {
                                    traits.push({ type: traitType });
                                }
                            }
                        }
                    }
                    // Extract TypesMatchWith traits as string - parsed on demand in getOperation
                    if (traitDag === 'TypesMatchWith' && trait.value && trait.value.operands) {
                        const operands = trait.value.operands;
                        if (operands.length >= 4) {
                            const getStringValue = (op) => op?.value?.value || op?.value;
                            const from = getStringValue(operands[1]);
                            const to = getStringValue(operands[2]);
                            const transformer = getStringValue(operands[3]);
                            if (from && to && transformer) {
                                const traitType = `TypesMatchWith<'${from}', '${to}', '${transformer}'>`;
                                if (traits.every((t) => t.type !== traitType)) {
                                    traits.push({ type: traitType });
                                }
                            }
                        }
                    }
                    // Handle classes that inherit from TypesMatchWith (e.g., PointeeTypeMatchTrait)
                    if (traitDag && traitDag !== 'TypesMatchWith' && trait.value && trait.value.operands) {
                        const traitClass = parser.getClass(traitDag);
                        if (traitClass) {
                            // Check if this class inherits from TypesMatchWith
                            for (const classParent of traitClass.parents || []) {
                                if (classParent.name === 'TypesMatchWith' && classParent.args && classParent.args.length >= 4) {
                                    // Build template bindings from trait operands to class template args
                                    const bindings = new Map();
                                    for (let i = 0; i < traitClass.templateArgs.length && i < trait.value.operands.length; i++) {
                                        const paramName = traitClass.templateArgs[i].name;
                                        const argValue = trait.value.operands[i];
                                        if (argValue && argValue.value) {
                                            bindings.set(paramName, argValue.value.type === 'string' ? argValue.value.value : argValue.value);
                                        }
                                    }
                                    // Extract from, to, transformer from TypesMatchWith parent args (indices 1, 2, 3)
                                    const resolveArg = (arg) => {
                                        if (!arg) {
                                            return null;
                                        }
                                        if (arg.type === 'string' || arg.type === 'code') {
                                            return arg.value;
                                        }
                                        if (arg.type === 'def' && typeof arg.value === 'string') {
                                            // Check if it's a template parameter reference
                                            return bindings.has(arg.value) ? bindings.get(arg.value) : arg.value;
                                        }
                                        return null;
                                    };
                                    const from = resolveArg(classParent.args[1]);
                                    const to = resolveArg(classParent.args[2]);
                                    const transformer = resolveArg(classParent.args[3]);
                                    if (from && to && transformer) {
                                        const traitType = `TypesMatchWith<'${from}', '${to}', '${transformer}'>`;
                                        if (traits.every((t) => t.type !== traitType)) {
                                            traits.push({ type: traitType });
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if ((traitName === 'AttrSizedOperandSegments' || traitDag === 'AttrSizedOperandSegments') && traits.every((t) => t.type !== 'AttrSizedOperandSegments')) {
                        traits.push({ type: 'AttrSizedOperandSegments' });
                    }
                    if ((traitName === 'SameOperandsAndResultType' || traitDag === 'SameOperandsAndResultType') && traits.every((t) => t.type !== 'SameOperandsAndResultType')) {
                        traits.push({ type: 'SameOperandsAndResultType' });
                    }
                    if (traitName === 'TensorRTInferTensorResultTypes' && traits.every((t) => t.type !== 'SameOperandsAndResultType')) {
                        traits.push({ type: 'SameOperandsAndResultType' });
                    }
                    if (traitName === 'IsolatedFromAbove' && traits.every((trait) => trait.type !== 'IsolatedFromAbove')) {
                        traits.push({ type: 'IsolatedFromAbove' });
                    }
                    if ((traitName === 'FirstAttrDerivedResultType' || traitDag === 'FirstAttrDerivedResultType') && traits.every((t) => t.type !== 'FirstAttrDerivedResultType')) {
                        traits.push({ type: 'FirstAttrDerivedResultType' });
                    }
                    if ((traitName === 'InferTypeOpInterface' || traitName === 'InferTypeOpAdaptor') &&
                        traits.every((trait) => trait.type !== 'InferTypeOpInterface')) {
                        traits.push({ type: 'InferTypeOpInterface' });
                    }
                    if (traitName === 'OpAsmOpInterface' || traitDag === 'DeclareOpInterfaceMethods') {
                        if (traitDag === 'DeclareOpInterfaceMethods' && trait.value && trait.value.operands) {
                            if (trait.value.operands.some((operand) => operand.value && operand.value.type === 'list' && operand.value.value.some((method) => method.type === 'string' && method.value === 'getDefaultDialect'))) {
                                const [dialectName] = operationName.split('.');
                                operation.defaultDialect = dialectName;
                            }
                        }
                        if (!operation.defaultDialect) {
                            const extraClass = def.getValueAsString('extraClassDeclaration');
                            if (extraClass) {
                                const match = extraClass.match(/getDefaultDialect\(\)\s*\{\s*return\s+"(\w+)"/);
                                if (match) {
                                    [, operation.defaultDialect] = match;
                                }
                            }
                        }
                    }
                }
            }
        };
        // Recursively extract traits from parent classes
        const extractTraitsFromParents = (parents, visited = new Set()) => {
            for (const parent of parents) {
                if (visited.has(parent.name)) {
                    continue;
                }
                visited.add(parent.name);
                // Extract traits from parent args (check all args as traits can be at different positions)
                if (parent.args) {
                    for (const traitsArg of parent.args) {
                        extractTraitsFromList(traitsArg);
                    }
                }
                // Recursively look at parent class definition
                const parentClass = parser.getClass(parent.name);
                if (parentClass && parentClass.parents) {
                    // Also extract traits from the parent class's own parent args
                    // This handles cases like Linalg_RelayoutOp which defines TypesMatchWith
                    // in its own inheritance from Op, not in args passed by children
                    for (const classParent of parentClass.parents) {
                        // Look for Op parent which typically has traits in args[2]
                        if (classParent.name === 'Op' && classParent.args && classParent.args.length >= 3) {
                            extractTraitsFromList(classParent.args[2]);
                        }
                        // Also check args that might contain traits (e.g., listconcat results)
                        if (classParent.args) {
                            for (const arg of classParent.args) {
                                extractTraitsFromList(arg);
                            }
                        }
                    }
                    extractTraitsFromParents(parentClass.parents, visited);
                }
            }
        };
        extractTraitsFromParents(def.parents);
        if (traits.length > 0) {
            operation.traits = traits;
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
        if (Object.keys(operation).length > 1) {
            operations.set(operationName, operation);
            count++;
        }
    }
    const sorted = Array.from(operations.values()).sort((a, b) => a.name.localeCompare(b.name));
    const output = JSON.stringify(sorted, null, 2);
    let formatted = output.replace(/\{\s+"name":\s+"([^"]+)",\s+"type":\s+"((?:[^"\\]|\\.)*)"\s+\}/g, '{ "name": "$1", "type": "$2" }');
    formatted = formatted.replace(/\{\s+"type":\s+"((?:[^"\\]|\\.)*)"\s+\}/g, '{ "type": "$1" }');
    await fs.writeFile(file, formatted, 'utf-8');
    if (count < 6300) {
        throw new Error(`Unexpected operation count '${count}'.`);
    }
};

const test = async (pattern) => {
    pattern = pattern || './**/*.mlir';
    const errorTotals = new Map();
    let currentFile = null;
    const fileErrors = new Map(); // file -> [error lines]
    const allFiles = new Set();
    await new Promise((resolve, reject) => {
        const cmd = 'node';
        const args = ['--max-old-space-size=8192', './test/models.js', 'continue', pattern];
        const proc = child_process.spawn(cmd, args, { stdio: ['ignore', 'pipe', 'pipe'] });
        const stdout = readline.createInterface({ input: proc.stdout, crlfDelay: Infinity });
        const stderr = readline.createInterface({ input: proc.stderr, crlfDelay: Infinity });
        const processLine = (line) => {
            writeLine(line);
            const stripped = line.trim();
            if (!stripped) {
                return;
            }
            if (stripped.startsWith('third_party/')) {
                currentFile = stripped;
                allFiles.add(currentFile);
                return;
            }
            if (currentFile) {
                if (/^\s*\d+\s*\/\s*\d+\s*=\s*[\d.]+%\s*$/.test(stripped)) {
                    return;
                }
                if (!fileErrors.has(currentFile)) {
                    fileErrors.set(currentFile, []);
                }
                fileErrors.get(currentFile).push(stripped);
            }
        };
        stdout.on('line', processLine);
        stderr.on('line', processLine);
        proc.on('error', (error) => reject(new Error(`Failed to start process: ${error.message}`)));
        proc.on('close', resolve);
    });
    const validFiles = new Set();
    const invalidFiles = new Set([
        'third_party/source/mlir/ensemble-compilation/tests/benchmarks/quantum_volume.mlir',
        'third_party/source/mlir/ensemble-compilation/tests/ensemble_gate_distribution.mlir',
        'third_party/source/mlir/iree/compiler/src/iree/compiler/Codegen/Dialect/Codegen/IR/test/lowering_config_attr.mlir',
        'third_party/source/mlir/iree/samples/compiler_plugins/simple_io_sample/test/print.mlir',
        'third_party/source/mlir/lltz/mlir/dialect/irdl/michelson.irdl.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Builtin/ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/IRDL/regions-ops.irdl.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/IRDL/testd.irdl.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/IRDL/variadics.irdl.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Linalg/tile-to-forall.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/LLVMIR/func.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/LLVMIR/global.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Quant/parse-uniform-invalid.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SMT/bitvector-errors.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/barrier-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/composite-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/control-flow-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/gl-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/khr-cooperative-matrix-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/logical-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/memory-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/misc-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/ocl-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/structure-ops.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/SPIRV/IR/types.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Tosa/level_check.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Tosa/verifier.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Dialect/Transform/test-pass-application.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/Examples/transform-opt/syntax-error.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/attribute.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/dynamic.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/invalid-unregistered.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/parser.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/parser-string-literal-comment.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/IR/zero_whitespace.mlir',
        'third_party/source/mlir/llvm-project/mlir/test/mlir-tblgen/attr-or-type-format.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/map.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/simple_sdfg.mlir',
        'third_party/source/mlir/mlir-dace/design/mlir/symbol.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Converter/toSDFG/llvm/load.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Converter/toSDFG/llvm/store.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Dialect/consume/too_many_params.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Dialect/memlet/explicit_tile.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Dialect/state/missing_identifier.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Dialect/state/missing_region.mlir',
        'third_party/source/mlir/mlir-dace/test/SDFG/Dialect/tasklet/missing_return_type.mlir',
        'third_party/source/mlir/runtime/mlir_tests/bef_executor/tutorial.mlir',
        'third_party/source/mlir/runtime/mlir_tests/core_runtime/basic_ops.mlir',
        'third_party/source/mlir/shardy/shardy/dialect/mpmd/ir/test/memory_kind_parse_and_print.mlir',
        'third_party/source/mlir/stablehlo/stablehlo/tests/ops_stablehlo.mlir',
        'third_party/source/mlir/stablehlo/stablehlo/tests/print_types_invalid.mlir',
        'third_party/source/mlir/stablehlo/stablehlo/tests/vhlo/invalid_vhlo_future.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_tf_drq.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_uniform_quantized.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_xla_weight_only.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/tensorflow/tests/compile_mlir_util/serialized-mlir-module-str-attr.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/tensorflow/tests/tf_executor_ops_invalid.mlir',
        'third_party/source/mlir/tensorflow/tensorflow/compiler/mlir/tfr/tests/ops.mlir',
        'third_party/source/mlir/xla/xla/hlo/translate/hlo_to_mhlo/tests/import_bounded_dynamism_stablehlo.mlir',
        'third_party/source/mlir/xla/xla/mlir_hlo/tests/Dialect/mhlo/ops.mlir',
        'third_party/source/mlir/xla/xla/mlir_hlo/tests/Dialect/mhlo/verifier_reduce_op.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_tf_drq.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_uniform_quantized.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library_xla_weight_only.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/quantization/tensorflow/passes/quantized_function_library.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/tensorflow/tests/compile_mlir_util/serialized-mlir-module-str-attr.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/tensorflow/tests/tf_executor_ops_invalid.mlir',
        'third_party/source/tensorflow/tensorflow/compiler/mlir/tfr/tests/ops.mlir',
        'third_party/source/tensorflow/third_party/xla/xla/hlo/translate/hlo_to_mhlo/tests/import_bounded_dynamism_stablehlo.mlir',
        'third_party/source/tensorflow/third_party/xla/xla/mlir_hlo/tests/Dialect/mhlo/ops.mlir',
        'third_party/source/tensorflow/third_party/xla/xla/mlir_hlo/tests/Dialect/mhlo/verifier_reduce_op.mlir',
        'third_party/test/mlir/sample.mlir',
    ]);
    const readRunHeader = async (filePath) => {
        const handle = await fs.open(filePath, 'r');
        const buffer = Buffer.alloc(256);
        await handle.read(buffer, 0, 256, 0);
        await handle.close();
        const content = buffer.toString('utf-8').split('\n')[0];
        return content.startsWith('// RUN:') ? content : null;
    };
    for (const file of allFiles) {
        if (file.toLowerCase().includes('invalid')) {
            invalidFiles.add(file);
        } else if (file.startsWith('third_party/source/mlir/mlir-dace/design')) {
            invalidFiles.add(file);
        } else {
            // eslint-disable-next-line no-await-in-loop
            const run = await readRunHeader(file);
            if (run?.includes('mlir-translate --import-wasm')) {
                invalidFiles.add(file);
            } else {
                validFiles.add(file);
            }
        }
    }
    const filesByError = new Map();
    const fileErrorDetails = new Map();
    for (const [file, errors] of fileErrors) {
        if (invalidFiles.has(file)) {
            continue;
        }
        for (const error of errors) {
            const key = error.split(' at ', 1)[0].trim().replace(/\.$/, '').trim();
            if (key) {
                errorTotals.set(key, (errorTotals.get(key) || 0) + 1);
                if (!filesByError.has(key)) {
                    filesByError.set(key, new Map());
                }
                filesByError.get(key).set(file, (filesByError.get(key).get(file) || 0) + 1);
                if (!fileErrorDetails.has(key)) {
                    fileErrorDetails.set(key, new Map());
                }
                if (!fileErrorDetails.get(key).has(file)) {
                    fileErrorDetails.get(key).set(file, []);
                }
                fileErrorDetails.get(key).get(file).push(error);
            }
        }
    }
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
    writeLine('');
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
