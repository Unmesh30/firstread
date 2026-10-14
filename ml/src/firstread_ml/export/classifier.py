"""Export the category classifier (logistic regression on embeddings) to ONNX.

Input ``embedding`` float32 ``[B, EMBED_DIM]``; outputs ``label`` int64 ``[B]``
(an index into ``labels.json``) and ``probabilities`` float32 ``[B, n_labels]``.
``zipmap=False`` keeps probabilities a plain tensor instead of a list of dicts.
"""

from pathlib import Path
from typing import Any

import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType

from firstread_core.embedding import EMBED_DIM

INPUT = "embedding"
OUTPUTS = ("label", "probabilities")
OPSET = 18


def export_classifier(clf: Any, path: Path, metadata: dict[str, str] | None = None) -> Path:
    proto = to_onnx(
        clf,
        initial_types=[(INPUT, FloatTensorType([None, EMBED_DIM]))],
        options={id(clf): {"zipmap": False}},
        target_opset={"": OPSET, "ai.onnx.ml": 3},
        final_types=None,
    )
    for out, name in zip(proto.graph.output, OUTPUTS, strict=True):
        _rename(proto, out.name, name)
    onnx.helper.set_model_props(proto, metadata or {})
    onnx.checker.check_model(proto)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(proto, str(path))
    return path


def _rename(proto: onnx.ModelProto, old: str, new: str) -> None:
    if old == new:
        return
    for node in proto.graph.node:
        node.output[:] = [new if o == old else o for o in node.output]
        node.input[:] = [new if i == old else i for i in node.input]
    for out in proto.graph.output:
        if out.name == old:
            out.name = new
