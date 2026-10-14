"""Export NRMS-lite's user encoder to ONNX.

Inputs ``hist`` float32 ``[B, HISTORY_LEN, EMBED_DIM]`` and ``mask`` bool
``[B, HISTORY_LEN]`` (right-aligned, as built by ``firstread_core.features``);
output ``user_vec`` float32 ``[B, EMBED_DIM]``. Only the batch dimension is
dynamic, so the API can micro-batch concurrent requests into one call.

The score temperature (``model.scale``) is not part of the graph: ranking only
needs the dot product, and the API reads ``score_scale`` from the metadata when
it needs scores on the training logit scale.
"""

import warnings
from pathlib import Path

import onnx
import torch

from firstread_core.embedding import EMBED_DIM, EMBED_MODEL, EMBED_MODEL_REVISION
from firstread_core.features import HISTORY_LEN
from firstread_ml.models.nrms_lite import NRMSLite

OPSET = 18
INPUTS = ("hist", "mask")
OUTPUT = "user_vec"
MAX_BATCH = 4096


def export_user_encoder(
    model: NRMSLite, path: Path, metadata: dict[str, str] | None = None
) -> Path:
    """Write ``model`` (in eval mode) to ``path`` and return it."""
    model = model.cpu().eval()
    hist = torch.zeros(2, HISTORY_LEN, EMBED_DIM)
    mask = torch.zeros(2, HISTORY_LEN, dtype=torch.bool)
    mask[1, -3:] = True  # one empty and one partial row, so neither path is pruned
    batch = torch.export.Dim("batch", min=1, max=MAX_BATCH)
    with warnings.catch_warnings():
        # Both inputs share the one "batch" Dim on purpose; torch warns about the alias.
        warnings.filterwarnings("ignore", message=".*shares the same shape constraints.*")
        program = torch.onnx.export(
            model,
            (hist, mask),
            dynamo=True,
            opset_version=OPSET,
            input_names=list(INPUTS),
            output_names=[OUTPUT],
            dynamic_shapes={"hist": {0: batch}, "mask": {0: batch}},
        )
    if program is None:
        raise RuntimeError("torch.onnx.export returned no program")
    path.parent.mkdir(parents=True, exist_ok=True)
    program.save(str(path))
    meta = {
        "history_len": str(HISTORY_LEN),
        "embed_dim": str(EMBED_DIM),
        "embed_model": EMBED_MODEL,
        "embed_model_revision": EMBED_MODEL_REVISION,
        "score_scale": f"{float(model.scale.detach()):.6f}",
    } | (metadata or {})
    proto = onnx.load(str(path))
    onnx.helper.set_model_props(proto, meta)
    onnx.checker.check_model(proto)
    onnx.save(proto, str(path))
    return path
