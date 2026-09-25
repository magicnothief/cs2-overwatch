"""Turn a trained scorer bundle into what the app ships: ONNX weights plus a JSON.

Training keeps PyTorch; the app does not. The network is 136 KB, and PyTorch is
1.2 GB before its CUDA libraries, for a model that scores a whole match on a CPU
in well under a second. So the trained bundle is exported once:

    scorer.onnx   the FlickCNN with a sigmoid on the end: windows in, a
                  probability per window out (input "windows", shape
                  (n, channels, ticks); output "score", shape (n,))
    scorer.json   everything else in the bundle the app reads: architecture,
                  the frozen clean reference, when it was trained

Only training and the tests import this module (it needs torch and onnx).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from overwatch.layers.l3_behavior.models import FlickCNN

#: Fields of the bundle that travel with the ONNX file.
SHIPPED = ("config", "reference", "trained_at", "cv_label")


class _Probability(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, windows: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(windows))


def model_from(bundle: dict) -> FlickCNN:
    config = bundle["config"]
    model = FlickCNN(
        in_channels=config["in_channels"],
        width=config["width"],
        pooling=config["pooling"],
    )
    model.load_state_dict(bundle["state_dict"])
    return model.eval()


def export_scorer(bundle: dict, out: str | Path) -> Path:
    """Write <out>.onnx and <out>.json from a bundle; returns the .onnx path.

    Checks the exported graph against PyTorch on random windows before returning.
    """
    import onnxruntime as ort

    out = Path(out).with_suffix(".onnx")
    out.parent.mkdir(parents=True, exist_ok=True)
    config = bundle["config"]
    ticks = config["pre"] + config["post"] + 1
    model = _Probability(model_from(bundle)).eval()
    sample = torch.randn(4, config["in_channels"], ticks)
    torch.onnx.export(
        model,
        (sample,),
        out,
        input_names=["windows"],
        output_names=["score"],
        dynamic_shapes={"windows": {0: torch.export.Dim("n", min=1)}},
        external_data=False,
        verbose=False,
    )
    meta = {k: bundle[k] for k in SHIPPED if k in bundle}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")

    probe = torch.randn(33, config["in_channels"], ticks)
    with torch.no_grad():
        expected = model(probe).numpy()
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"windows": probe.numpy()})[0]
    if not np.allclose(got, expected, atol=1e-5):
        msg = f"ONNX export differs from PyTorch by {np.abs(got - expected).max():.2g}"
        raise RuntimeError(msg)
    return out


__all__ = ["SHIPPED", "export_scorer", "model_from"]
