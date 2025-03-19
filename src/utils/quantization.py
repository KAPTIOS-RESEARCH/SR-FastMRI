import torch, os, logging, time
import numpy as np
from torch import nn
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat
from onnxruntime import InferenceSession
from typing import Sequence, Any, Tuple, List
from onnxruntime.quantization import quant_pre_process
from src.data.utils.fastMRI.transforms import SRSample

def get_layers_to_fuse(model: nn.Module, prefix: str = "") -> list:
    """Recursively retrieves a list of groupable layers for fusion.

    Args:
        model (nn.Module): The model to quantize.
        prefix (str): The prefix for the module names (for nested modules).

    Returns:
        fused_layers (list): The list of fused layer pairs.
    """
    fused_layers = []
    prev_layer = None
    prev_name = None

    for module_name, module in model.named_children():
        # Build the full name of the current module
        full_name = f"{prefix}.{module_name}" if prefix else module_name
        layer_name = module.__class__.__name__

        # Recursively process submodules
        if len(list(module.children())) > 0:
            fused_layers.extend(get_layers_to_fuse(module, full_name))
        
        # Check for fusable patterns
        if prev_layer:
            # Conv2d or Linear followed by an Activation
            if prev_layer in {"Conv2d", "Linear"} and layer_name in {"ReLU", "LeakyReLU", "Sigmoid", "Tanh"}:
                fused_layers.append([prev_name, full_name])
                prev_layer, prev_name = None, None

            # Conv2d followed by Normalization
            elif prev_layer == "Conv2d" and layer_name in {"BatchNorm2d", "LayerNorm", "InstanceNorm2d"}:
                prev_layer, prev_name = layer_name, full_name

            # Normalization followed by Activation
            elif prev_layer in {"BatchNorm2d", "LayerNorm", "InstanceNorm2d"} and layer_name in {"ReLU", "LeakyReLU", "Sigmoid", "Tanh"}:
                fused_layers.append([prev_name, full_name])
                prev_layer, prev_name = None, None

            else:
                prev_layer, prev_name = layer_name, full_name
        else:
            prev_layer, prev_name = layer_name, full_name

    return fused_layers


def fuse_layers(model: nn.Module):
    """Fuse layers of a model for quantization

    Args:
        model (nn.Module): The original model
    Returns:
        fused_model (nn.Module): The model with fused layers
    """
    fused_layers = get_layers_to_fuse(model)
    return torch.quantization.fuse_modules(
        model,
        fused_layers
    )
    
def quantize_onnx_model(model_path: str, quantized_model_path: str, calibration_dataset: CalibrationDataReader):
    """Run uint8 quantization on an onnx model.

    Args:
        model_path (str): Path to the .onnx model file
        quantized_model_path (str): Path to save the quantized .onnx model file
        calibration_dataset (CalibrationDataReader): The calibration dataset to retrieve the input data dict for ONNXinferenceSession
    """
    quant_pre_process(model_path, model_path)
    quantize_static(
        model_path,
        quantized_model_path,
        calibration_dataset,
        quant_format=QuantFormat.QDQ
    )

    logging.info(f"Original ONNX model size (MB): {os.path.getsize(model_path) / (1024 * 1024):.2f}")
    logging.info(f"Quantized ONNX model size (MB): {os.path.getsize(quantized_model_path) / (1024 * 1024):.2f}")
    logging.info(f"Quantized model saved to {quantized_model_path}")

def get_session_performance(
    session: InferenceSession, 
    data: List[SRSample], 
):
    """Run inference of a batch to retrieve the model accuracy and time performances.

    Args:
        session (InferenceSession): The model onnxruntime session
        data (List[SRSample]): The batch SRSample

    Returns:
        tuple[float, float]: Average inference time in milliseconds, accuracy
    """
    times = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    for sample in data:
        x = x.numpy()
        x = np.expand_dims(x, axis=0)
        start_time = time.perf_counter()
        session.run([output_name], {input_name: x})[0]
        times.append(time.perf_counter() - start_time)

    avg_time_ms = np.mean(times) * 1000
    return avg_time_ms

def quantized_session_performance_benchmark(
        original_model_path: str,
        quantized_model_path: str,
        data_sample: Tuple[torch.Tensor, torch.Tensor], 
        device_providers: Sequence[str | tuple[str, dict[Any, Any]]] | None = None,
    ):
    """Compares the performance of a model with its quantized version.

    Args:
        original_model_path (str): The original model .onnx file path
        quantized_model_path (str): The quantized model .onnx file path
        x (torch.Tensor): The input data
        y (torch.Tensor): The target data
        runs (int, optional): The number of inference runs to perform. Defaults to 50.
    """
    non_quantized_session = InferenceSession(original_model_path, providers=device_providers)
    quantized_session = InferenceSession(quantized_model_path, providers=device_providers)

    t1 = get_session_performance(non_quantized_session, data_sample)
    t2 = get_session_performance(quantized_session, data_sample)

    print(f"🟢 Non-Quantized Model Avg Inference Time: {t1:.2f} ms")
    print(f"🔥 Quantized Model Avg Inference Time: {t2:.2f} ms")
    print(f"🚀 Speedup: {t1 / t1:.2f}x faster with quantization")
