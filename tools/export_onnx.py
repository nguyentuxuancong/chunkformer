import argparse
import os

import numpy as np
import onnxruntime as ort
import torch

import onnx
from chunkformer.chunkformer_model import ChunkFormerModel


class EncoderONNXWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, xs, xs_lens, chunk_size, left_context_size, right_context_size):
        return self.encoder.forward_encoder(
            xs,
            xs_lens,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
            export=True,
        )


def get_encoder_from_inference_dir(inference_model_dir):
    model = ChunkFormerModel.from_pretrained(inference_model_dir)
    encoder = model.get_encoder()
    encoder.eval()
    return encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export ChunkFormerEncoder to ONNX.")
    parser.add_argument(
        "inference_model_dir", type=str, help="Path to the inference model directory."
    )
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for encoder export.")
    parser.add_argument(
        "--left_context_size", type=int, default=128, help="Left context size for encoder export."
    )
    parser.add_argument(
        "--right_context_size", type=int, default=128, help="Right context size for encoder export."
    )
    args = parser.parse_args()

    inference_model_dir = args.inference_model_dir
    encoder = get_encoder_from_inference_dir(inference_model_dir)
    encoder_onnx = EncoderONNXWrapper(encoder)
    encoder_onnx.eval()

    batch_size = 1
    seq_len = 500 * 8 + 7
    input_dim = 80
    xs = torch.randn(batch_size, seq_len, input_dim)
    xs_lens = torch.full((batch_size,), seq_len, dtype=torch.long)
    chunk_size = args.chunk_size
    left_context_size = args.left_context_size
    right_context_size = args.right_context_size

    onnx_dir = os.path.join(inference_model_dir, "onnx")
    os.makedirs(onnx_dir, exist_ok=True)
    onnx_path = os.path.join(onnx_dir, "chunkformer_encoder.onnx")

    # call the encoder
    with torch.no_grad():
        encoder_out, encoder_mask = encoder_onnx(
            xs,
            xs_lens,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
        print(
            f"Encoder output shape: {encoder_out.shape}, Encoder mask shape: {encoder_mask.shape}"
        )

    onnx_model = torch.onnx.export(
        encoder_onnx,
        (xs, xs_lens, chunk_size, left_context_size, right_context_size),
        onnx_path,
        input_names=["xs", "xs_lens", "chunk_size", "left_context_size", "right_context_size"],
        output_names=["encoder_out", "encoder_mask"],
        opset_version=17,
        dynamic_axes={
            "xs": {0: "batch_size", 1: "seq_len"},
            "xs_lens": {0: "batch_size"},
            "encoder_out": {0: "batch_size", 1: "out_seq_len"},
            "encoder_mask": {0: "batch_size", 2: "out_seq_len"},
        },
        # verbose=True,
        dynamo=False,
        report=False,
        profile=False,
        verbose=1,
    )
    print(f"Exported ChunkFormerEncoder to {onnx_path} using forward_encoder.")

    # Verify the ONNX model
    onnx_model = onnx.load(onnx_path)  # type: ignore[attr-defined]
    onnx.checker.check_model(onnx_model)  # type: ignore[attr-defined]
    print("ONNX model is valid.")

    # Test inference with ONNX Runtime
    ort_session = ort.InferenceSession(onnx_path)
    ort_inputs = {
        "xs": xs.numpy(),
        "xs_lens": xs_lens.numpy(),
        "chunk_size": np.array(chunk_size, dtype=np.int64),
        "left_context_size": np.array(left_context_size, dtype=np.int64),
        "right_context_size": np.array(right_context_size, dtype=np.int64),
    }
    ort_outs = ort_session.run(None, ort_inputs)
    print("ONNX Runtime inference successful.")

    # Compare with PyTorch output
    with torch.no_grad():
        torch_out, torch_mask = encoder_onnx(
            xs,
            xs_lens,
            chunk_size=chunk_size,
            left_context_size=left_context_size,
            right_context_size=right_context_size,
        )
    print("PyTorch inference successful.")

    # Compare the results
    np.testing.assert_allclose(torch_out.numpy(), ort_outs[0], rtol=1e-05, atol=5e-06)
    assert (torch_mask.numpy() == ort_outs[1]).all(), "Encoder masks do not match!"
    print("The outputs from PyTorch and ONNX Runtime match!")

    # Compute the difference
    diff_out = np.abs(torch_out.numpy() - ort_outs[0])
    print(f"Max difference in encoder_out: {diff_out.max()}")
