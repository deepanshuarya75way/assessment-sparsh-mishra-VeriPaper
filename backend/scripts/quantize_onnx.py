#!/usr/bin/env python3
"""Export the fine-tuned DistilBERT detector to ONNX and quantize to int8.

Produces backend/models/ai_detector_transformer/model_quantized.onnx (~30-60 MB),
which runs on CPU with onnxruntime at a fraction of the memory cost of PyTorch
so the whole app fits Render's 512 MB free-tier limit.
"""
import subprocess
import sys
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "ai_detector_transformer"


def main() -> int:
    model_dir = MODEL_DIR
    fp32_onnx = model_dir / "model.onnx"
    q8_onnx = model_dir / "model_quantized.onnx"

    if not q8_onnx.exists():
        from optimum.onnxruntime import ORTQuantizer
        from optimum.exporters.onnx import main_export
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_dir)
        # Export for sequence classification with fixed input shapes into a
        # temp staging dir so the quantizer reads from a clean source.
        import tempfile

        staging = Path(tempfile.mkdtemp(prefix="onnx_staging_"))
        try:
            main_export(
                model_name_or_path=str(model_dir),
                output=str(fp32_onnx),
                task="sequence-classification",
                batch_size=1,
                sequence_length=256,
            )
            from optimum.onnxruntime.configuration import AutoQuantizationConfig

            quantizer = ORTQuantizer.from_pretrained(str(fp32_onnx))
            qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
            quantized = quantizer.quantize(
                quantization_config=qconfig,
                save_dir=str(model_dir),
                file_suffix="quantized",
            )
            # Normalize the output name for the serving path
            final = model_dir / "model_quantized.onnx"
            if quantized != final:
                (final.parent / quantized.name).rename(final)
        finally:
            fp32_onnx.unlink(missing_ok=True)
            for child in staging.iterdir():
                child.unlink(missing_ok=True)
            staging.rmdir()

    size_mb = q8_onnx.stat().st_size / 1e6
    print(f"Quantized ONNX model ready: {q8_onnx} ({size_mb:.1f} MB)")

    # Quick sanity inference to confirm validity
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(str(q8_onnx))
    inputs = {i.name: np.zeros((1, 256), dtype=np.int64) for i in sess.get_inputs()}
    out = sess.run(None, inputs)
    print(f"ONNX session OK, output shape: {out[0].shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
