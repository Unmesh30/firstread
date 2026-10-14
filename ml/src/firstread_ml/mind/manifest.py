"""Pinned MIND files on the Hugging Face mirror.

The official Azure links have returned PublicAccessNotPermitted since July 2024.
The Recommenders project hosts the official zips at
https://huggingface.co/datasets/Recommenders/MIND. We pin the repo revision and
each file's SHA256 (from the HF API, checked 2026-10-13) and verify after download.
MINDlarge_test is not offered: it has no labels.
"""

from dataclasses import dataclass

HF_REPO = "Recommenders/MIND"
HF_REVISION = "2115068f457b577f93a268fffe22373e45727d89"
LICENSE_URL = "https://github.com/msnews/MIND/blob/master/MSR%20License_Data.pdf"


@dataclass(frozen=True)
class MindFile:
    name: str  # e.g. "MINDsmall_train"; the zip is f"{name}.zip"
    sha256: str
    size: int

    @property
    def filename(self) -> str:
        return f"{self.name}.zip"


FILES = {
    f.name: f
    for f in (
        MindFile(
            "MINDsmall_train",
            "6ef97a271580b98ccfc4301ada55cc639423cb0576a78b8dcfcf74a4dbcc3194",
            52994575,
        ),
        MindFile(
            "MINDsmall_dev",
            "d6ce515dcaa6b6d47ddf0a326eebc8a31b84735ae410285c9882ca2a06eec669",
            30948560,
        ),
        MindFile(
            "MINDlarge_train",
            "7e7da0ecc45db1d8769eebf92301675e7af443b0d5266ac12b6e102bb8399233",
            531360717,
        ),
        MindFile(
            "MINDlarge_dev",
            "67b0437b48b6baee5655b6063a56d29edbcc92308fc0140555b0ae27263cf9e9",
            103592887,
        ),
    )
}


def split_name(variant: str, split: str) -> str:
    """``("small", "train") -> "MINDsmall_train"``."""
    return f"MIND{variant}_{split}"


# Variants we can download. ("synthetic" exists only as locally generated test data.)
VARIANTS = {"small": ("train", "dev"), "large": ("train", "dev")}
