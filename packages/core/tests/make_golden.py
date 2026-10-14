"""Regenerate the golden embedding vectors (run on purpose, never in CI).

    uv run python packages/core/tests/make_golden.py

Only rerun this when the embedding model is changed deliberately, and record why.
"""

import json
import platform
from pathlib import Path

from firstread_core.embedding import (
    EMBED_MODEL,
    EMBED_MODEL_REVISION,
    EMBED_MODEL_SOURCE,
    FASTEMBED_VERSION,
    Embedder,
)
from firstread_core.text import article_text

# Our own sentences (not MIND data). Each is embedded through article_text.
SENTENCES = [
    ("Central bank holds interest rates steady", "Officials signalled cuts may come later."),
    ("Underdog team wins the championship in overtime", ""),
    ("New battery chemistry doubles electric car range?", "Researchers report lab results."),
]

OUT = Path(__file__).parent / "data" / "golden_bge_small_en_v1_5.json"


def main() -> None:
    texts = [article_text(t, s) for t, s in SENTENCES]
    vecs = Embedder().embed(texts)
    OUT.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "source": EMBED_MODEL_SOURCE,
                "revision": EMBED_MODEL_REVISION,
                "fastembed_version": FASTEMBED_VERSION,
                "platform": f"{platform.system()} {platform.machine()}",
                "items": [
                    {"title": t, "summary": s, "text": x, "vector": [float(v) for v in vec]}
                    for (t, s), x, vec in zip(SENTENCES, texts, vecs, strict=True)
                ],
            },
            indent=1,
        )
        + "\n"
    )
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
