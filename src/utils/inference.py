import time
import numpy as np
import torch, logging
from onnxruntime import InferenceSession
from tqdm import tqdm
import matplotlib.pyplot as plt

def run_session_inference(session: InferenceSession, dataset: torch.utils.data.Dataset):
    """Run inference of an onnx session on all samples in the given dataset

    Args:
        session (InferenceSession): The onnx session
        dataset (torch.utils.data.Dataset): The PyTorch dataset

    Returns:
        tuple: The average time of inference in seconds, the average PSNR
    """
    times = []
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    with tqdm(total=len(dataset), desc=f'Running inference on {len(dataset)} samples ...') as pbar:
        for idx, sample in enumerate(dataset):
            x, _ = sample.image, sample.target
            x = x.numpy()
            x = np.expand_dims(x, axis=0)
            start_time = time.perf_counter()
            session.run([output_name], {input_name: x})[0]
            times.append(time.perf_counter() - start_time)
            pbar.update(1)
    avg_time_sec = np.mean(times)
    return avg_time_sec