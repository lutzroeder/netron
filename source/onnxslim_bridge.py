"""OnnxSlim integration bridge for Netron.

This module provides an HTTP API to modify ONNX models using OnnxSlim.
It can be used as a standalone server or integrated with Netron's Python server.
"""

import json
import logging
import os
import sys
import tempfile
import traceback
from typing import Any

# Remove the script's directory from sys.path to avoid shadowing the onnx module
# Netron has an onnx.py that would conflict with the actual onnx package
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir in sys.path:
    sys.path.remove(_script_dir)

logger = logging.getLogger(__name__)


def get_onnxslim_version() -> str:
    """Get OnnxSlim version."""
    try:
        from onnxslim.version import __version__
        return __version__
    except ImportError:
        return "unknown"


def get_available_operations() -> dict:
    """Return available OnnxSlim operations and their descriptions."""
    return {
        "slim": {
            "description": "Optimize ONNX model (constant folding, dead node elimination, etc.)",
            "parameters": {
                "no_shape_infer": {"type": "boolean", "default": False, "description": "Disable shape inference"},
                "skip_optimizations": {
                    "type": "array",
                    "items": ["constant_folding", "graph_fusion", "dead_node_elimination", "subexpression_elimination", "weight_tying"],
                    "default": [],
                    "description": "Optimizations to skip"
                },
                "size_threshold": {"type": "integer", "default": None, "description": "Skip folding constants larger than this (bytes)"}
            }
        },
        "convert_dtype": {
            "description": "Convert model data type",
            "parameters": {
                "dtype": {"type": "string", "enum": ["fp16", "fp32"], "description": "Target data type"}
            }
        },
        "modify_inputs": {
            "description": "Modify model inputs",
            "parameters": {
                "inputs": {"type": "array", "description": "Input modifications in format 'name:dtype'"},
                "input_shapes": {"type": "array", "description": "Input shape modifications in format 'name:dim1,dim2,...'"}
            }
        },
        "modify_outputs": {
            "description": "Modify model outputs",
            "parameters": {
                "outputs": {"type": "array", "description": "Output modifications in format 'name:dtype'"}
            }
        },
        "inspect": {
            "description": "Get model information without modifying",
            "parameters": {}
        }
    }


def slim_model(input_path: str | None = None, output_path: str | None = None,
               model_data: bytes | None = None, model_name: str = "model", **kwargs) -> dict:
    """Optimize an ONNX model using OnnxSlim.

    Args:
        input_path: Path to input ONNX model (optional if model_data provided)
        output_path: Path to save optimized model (optional, will use temp file if not provided)
        model_data: Raw model bytes (alternative to input_path for in-memory models)
        model_name: Name for the model (used when model_data is provided)
        **kwargs: Additional arguments passed to onnxslim.slim()

    Returns:
        dict with status, output_path, and model info
    """
    try:
        from onnxslim import slim
        from onnxslim.utils import summarize_model
        import onnx

        # Load original model
        if model_data:
            original_model = onnx.load_from_string(model_data)
            base_name = os.path.splitext(model_name)[0] if model_name else "model"
        elif input_path:
            original_model = onnx.load(input_path)
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            return {"status": "error", "error": "No input_path or model_data provided"}

        original_info = summarize_model(original_model, model_name or base_name)

        # Generate output path if not provided
        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"{base_name}_slimmed.onnx"
            )

        # Run slim - pass model object for in-memory, path for file
        if model_data:
            slim(original_model, output_path, **kwargs)
        else:
            slim(input_path, output_path, **kwargs)

        # Load optimized model for info
        optimized_model = onnx.load(output_path)
        optimized_info = summarize_model(optimized_model, os.path.basename(output_path))

        return {
            "status": "success",
            "output_path": output_path,
            "original": _model_info_to_dict(original_info),
            "optimized": _model_info_to_dict(optimized_info)
        }

    except Exception as e:
        logger.error(f"Error slimming model: {e}")
        logger.error(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def convert_dtype(input_path: str | None = None, dtype: str = "fp16", output_path: str | None = None,
                  model_data: bytes | None = None, model_name: str = "model") -> dict:
    """Convert model data type.

    Args:
        input_path: Path to input ONNX model (optional if model_data provided)
        dtype: Target data type ('fp16' or 'fp32')
        output_path: Path to save converted model
        model_data: Raw model bytes (alternative to input_path)
        model_name: Name for the model (used when model_data is provided)

    Returns:
        dict with status and output_path
    """
    try:
        from onnxslim import slim
        import onnx

        if model_data:
            model = onnx.load_from_string(model_data)
            base_name = os.path.splitext(model_name)[0] if model_name else "model"
        elif input_path:
            model = None  # Will use path directly
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            return {"status": "error", "error": "No input_path or model_data provided"}

        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"{base_name}_{dtype}.onnx"
            )

        if model_data:
            slim(model, output_path, dtype=dtype, skip_optimizations=[
                "constant_folding", "graph_fusion", "dead_node_elimination",
                "subexpression_elimination", "weight_tying"
            ])
        else:
            slim(input_path, output_path, dtype=dtype, skip_optimizations=[
                "constant_folding", "graph_fusion", "dead_node_elimination",
                "subexpression_elimination", "weight_tying"
            ])

        return {
            "status": "success",
            "output_path": output_path,
            "dtype": dtype
        }

    except Exception as e:
        logger.error(f"Error converting dtype: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def modify_inputs(input_path: str | None = None, inputs: list | None = None,
                  input_shapes: list | None = None, output_path: str | None = None,
                  model_data: bytes | None = None, model_name: str = "model") -> dict:
    """Modify model inputs.

    Args:
        input_path: Path to input ONNX model (optional if model_data provided)
        inputs: List of input modifications in format 'name:dtype'
        input_shapes: List of shape modifications in format 'name:dim1,dim2,...'
        output_path: Path to save modified model
        model_data: Raw model bytes (alternative to input_path)
        model_name: Name for the model (used when model_data is provided)

    Returns:
        dict with status and output_path
    """
    try:
        from onnxslim import slim
        import onnx

        if model_data:
            model = onnx.load_from_string(model_data)
            base_name = os.path.splitext(model_name)[0] if model_name else "model"
        elif input_path:
            model = None
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            return {"status": "error", "error": "No input_path or model_data provided"}

        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"{base_name}_modified.onnx"
            )

        if model_data:
            slim(model, output_path,
                 inputs=inputs,
                 input_shapes=input_shapes,
                 skip_optimizations=[
                     "constant_folding", "graph_fusion", "dead_node_elimination",
                     "subexpression_elimination", "weight_tying"
                 ])
        else:
            slim(input_path, output_path,
                 inputs=inputs,
                 input_shapes=input_shapes,
                 skip_optimizations=[
                     "constant_folding", "graph_fusion", "dead_node_elimination",
                     "subexpression_elimination", "weight_tying"
                 ])

        return {
            "status": "success",
            "output_path": output_path
        }

    except Exception as e:
        logger.error(f"Error modifying inputs: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def modify_outputs(input_path: str | None = None, outputs: list = None, output_path: str | None = None,
                   model_data: bytes | None = None, model_name: str = "model") -> dict:
    """Modify model outputs.

    Args:
        input_path: Path to input ONNX model (optional if model_data provided)
        outputs: List of output modifications in format 'name:dtype'
        output_path: Path to save modified model
        model_data: Raw model bytes (alternative to input_path)
        model_name: Name for the model (used when model_data is provided)

    Returns:
        dict with status and output_path
    """
    try:
        from onnxslim import slim
        import onnx

        if model_data:
            model = onnx.load_from_string(model_data)
            base_name = os.path.splitext(model_name)[0] if model_name else "model"
        elif input_path:
            model = None
            base_name = os.path.splitext(os.path.basename(input_path))[0]
        else:
            return {"status": "error", "error": "No input_path or model_data provided"}

        if output_path is None:
            output_path = os.path.join(
                tempfile.gettempdir(),
                f"{base_name}_modified.onnx"
            )

        if model_data:
            slim(model, output_path,
                 outputs=outputs,
                 skip_optimizations=[
                     "constant_folding", "graph_fusion", "dead_node_elimination",
                     "subexpression_elimination", "weight_tying"
                 ])
        else:
            slim(input_path, output_path,
                 outputs=outputs,
                 skip_optimizations=[
                     "constant_folding", "graph_fusion", "dead_node_elimination",
                     "subexpression_elimination", "weight_tying"
                 ])

        return {
            "status": "success",
            "output_path": output_path
        }

    except Exception as e:
        logger.error(f"Error modifying outputs: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def inspect_model(input_path: str | None = None, model_data: bytes | None = None,
                  model_name: str = "model") -> dict:
    """Get detailed model information.

    Args:
        input_path: Path to ONNX model (optional if model_data provided)
        model_data: Raw model bytes (alternative to input_path)
        model_name: Name for the model (used when model_data is provided)

    Returns:
        dict with model information
    """
    try:
        import onnx
        from onnxslim.utils import summarize_model

        if model_data:
            model = onnx.load_from_string(model_data)
            name = model_name or "model"
        elif input_path:
            model = onnx.load(input_path)
            name = os.path.basename(input_path)
        else:
            return {"status": "error", "error": "No input_path or model_data provided"}

        info = summarize_model(model, name)

        return {
            "status": "success",
            "info": _model_info_to_dict(info)
        }

    except Exception as e:
        logger.error(f"Error inspecting model: {e}")
        return {
            "status": "error",
            "error": str(e)
        }


def _to_serializable(value: Any) -> Any:
    """Convert numpy types to JSON-serializable Python types."""
    if hasattr(value, 'item'):  # numpy scalar
        return value.item()
    elif isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    elif isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    return value


def _model_info_to_dict(info: Any) -> dict:
    """Convert model info object to serializable dict."""
    result = {}

    if hasattr(info, "name"):
        result["name"] = info.name
    if hasattr(info, "op_type_counts"):
        # Convert op_type_counts values (may be numpy ints) to native Python ints
        op_counts = {}
        if info.op_type_counts:
            for k, v in info.op_type_counts.items():
                op_counts[k] = v.item() if hasattr(v, 'item') else int(v)
        result["op_type_counts"] = op_counts
    if hasattr(info, "op_set"):
        result["op_set"] = _to_serializable(info.op_set)
    if hasattr(info, "model_size"):
        result["model_size"] = _to_serializable(info.model_size)
    if hasattr(info, "input_info"):
        result["inputs"] = [_tensor_info_to_dict(t) for t in info.input_info] if info.input_info else []
    if hasattr(info, "output_info"):
        result["outputs"] = [_tensor_info_to_dict(t) for t in info.output_info] if info.output_info else []

    return result


def _tensor_info_to_dict(tensor_info: Any) -> dict:
    """Convert tensor info to serializable dict."""
    result = {}
    if hasattr(tensor_info, "name"):
        result["name"] = tensor_info.name
    if hasattr(tensor_info, "dtype"):
        result["dtype"] = str(tensor_info.dtype)
    if hasattr(tensor_info, "shape"):
        # Convert numpy int64 to native Python int for JSON serialization
        shape = []
        if tensor_info.shape:
            for dim in tensor_info.shape:
                if hasattr(dim, 'item'):  # numpy type
                    shape.append(dim.item())
                elif isinstance(dim, (int, str)):
                    shape.append(dim)
                else:
                    shape.append(int(dim) if isinstance(dim, (float,)) else str(dim))
        result["shape"] = shape
    return result


def handle_request(request_data: dict) -> dict:
    """Handle an API request.

    Args:
        request_data: dict with 'operation', 'input_path' or 'model_data', and operation-specific parameters

    Returns:
        dict with operation result
    """
    operation = request_data.get("operation")
    input_path = request_data.get("input_path")
    output_path = request_data.get("output_path")
    model_data = request_data.get("model_data")  # bytes for in-memory model
    model_name = request_data.get("model_name", "model")

    if not operation:
        return {"status": "error", "error": "Missing 'operation' field"}

    if operation == "get_operations":
        return {
            "status": "success",
            "operations": get_available_operations(),
            "version": get_onnxslim_version()
        }

    # Check if we have either input_path or model_data
    if not input_path and not model_data:
        return {"status": "error", "error": "Missing 'input_path' or 'model_data' field"}

    # If input_path provided, verify it exists (skip if we have model_data)
    if input_path and not model_data and not os.path.exists(input_path):
        return {"status": "error", "error": f"Input file not found: {input_path}"}

    if operation == "slim":
        return slim_model(
            input_path=input_path,
            output_path=output_path,
            model_data=model_data,
            model_name=model_name,
            no_shape_infer=request_data.get("no_shape_infer", False),
            skip_optimizations=request_data.get("skip_optimizations"),
            size_threshold=request_data.get("size_threshold")
        )

    elif operation == "convert_dtype":
        dtype = request_data.get("dtype")
        if not dtype:
            return {"status": "error", "error": "Missing 'dtype' parameter"}
        return convert_dtype(
            input_path=input_path,
            dtype=dtype,
            output_path=output_path,
            model_data=model_data,
            model_name=model_name
        )

    elif operation == "modify_inputs":
        return modify_inputs(
            input_path=input_path,
            inputs=request_data.get("inputs"),
            input_shapes=request_data.get("input_shapes"),
            output_path=output_path,
            model_data=model_data,
            model_name=model_name
        )

    elif operation == "modify_outputs":
        outputs = request_data.get("outputs")
        if not outputs:
            return {"status": "error", "error": "Missing 'outputs' parameter"}
        return modify_outputs(
            input_path=input_path,
            outputs=outputs,
            output_path=output_path,
            model_data=model_data,
            model_name=model_name
        )

    elif operation == "inspect":
        return inspect_model(
            input_path=input_path,
            model_data=model_data,
            model_name=model_name
        )

    else:
        return {"status": "error", "error": f"Unknown operation: {operation}"}


# CLI interface for testing
if __name__ == "__main__":
    import argparse
    import base64

    parser = argparse.ArgumentParser(description="OnnxSlim bridge for Netron")
    parser.add_argument("--operation", "-o",
                        choices=["slim", "convert_dtype", "modify_inputs", "modify_outputs", "inspect", "get_operations"],
                        help="Operation to perform")
    parser.add_argument("--input", "-i", help="Input ONNX model path")
    parser.add_argument("--output", help="Output ONNX model path")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], help="Target dtype for convert_dtype")
    parser.add_argument("--no-shape-infer", action="store_true", help="Disable shape inference")
    parser.add_argument("--skip-optimizations", nargs="+", help="Optimizations to skip")
    parser.add_argument("--inputs", nargs="+", help="Input modifications")
    parser.add_argument("--input-shapes", nargs="+", help="Input shape modifications")
    parser.add_argument("--outputs", nargs="+", help="Output modifications")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--stdin", action="store_true", help="Read JSON request from stdin")

    args = parser.parse_args()

    if args.stdin:
        # Read JSON request from stdin (supports in-memory model data)
        try:
            input_data = sys.stdin.read()
            request = json.loads(input_data)

            # Decode base64 model data if present
            if request.get("model_data_base64"):
                request["model_data"] = base64.b64decode(request["model_data_base64"])
                del request["model_data_base64"]

            # Suppress onnxslim's console output by redirecting stdout/stderr during processing
            import io
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()
            try:
                result = handle_request(request)
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            print(json.dumps(result))
        except Exception as e:
            print(json.dumps({"status": "error", "error": str(e)}))
            sys.exit(1)
    else:
        if not args.operation:
            parser.error("--operation is required when not using --stdin")

        request = {
            "operation": args.operation,
            "input_path": args.input,
            "output_path": args.output,
            "dtype": args.dtype,
            "no_shape_infer": args.no_shape_infer,
            "skip_optimizations": args.skip_optimizations,
            "inputs": args.inputs,
            "input_shapes": args.input_shapes,
            "outputs": args.outputs
        }

        result = handle_request(request)

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            if result.get("status") == "success":
                print(f"Success!")
                if "output_path" in result:
                    print(f"Output: {result['output_path']}")
                if "info" in result:
                    print(f"Model info: {json.dumps(result['info'], indent=2)}")
            else:
                print(f"Error: {result.get('error')}")
                sys.exit(1)
