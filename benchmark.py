import logging
from argparse import ArgumentParser
from onnxruntime import InferenceSession
from src.data.sets.super_resolution import FastMRISuperResolutionDataset
from src.utils.inference import run_session_inference
from torch.utils.data import Subset

if __name__ == "__main__":

    project_name = "SRFastMRI"

    logging.basicConfig(
        level=logging.INFO,
        format=f'%(asctime)s - {project_name} Inference Benchmark - %(levelname)s - %(message)s'
    )

    parser = ArgumentParser()
    parser.add_argument("--fastMRI_data_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--quantized_model_path", type=str, required=True)
    parser.add_argument("--n_data_samples", type=int, default=50)
    args = parser.parse_args()
    non_quantized_session = InferenceSession(args.model_path)
    quantized_session = InferenceSession(args.quantized_model_path)
    logging.info('Inference sessions loaded ✅')
    dataset = FastMRISuperResolutionDataset(args.fastMRI_data_path)
    dataset = Subset(dataset, range(args.n_data_samples))
    
    t1, psnr1 = run_session_inference(non_quantized_session, dataset)
    t2, psnr2 = run_session_inference(quantized_session, dataset)

    logging.info(f"🟢 Non-Quantized Model | Avg Inference Time & PSNR : {t1:.2f} ms; {psnr1:.2f}")
    logging.info(f"🔥 Quantized Model | Avg Inference Time & PSNR : {t2:.2f} ms; {psnr2:.2f}")
    logging.info(f"🚀 Speedup: {t1 / t2:.2f}x faster with quantization")

