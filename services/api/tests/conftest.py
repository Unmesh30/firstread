from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from firstread_api.config import Settings
from firstread_api.main import create_app
from firstread_api.testing import Signer, make_settings


@pytest.fixture
def signer() -> Signer:
    return Signer()


@pytest.fixture
def settings(tmp_path: Path, signer: Signer) -> Settings:
    return make_settings(auth_local_jwks_file=signer.write_jwks(tmp_path / "jwks.json"),
                         readyz_timeout_s=0.2)  # fmt: skip


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c
