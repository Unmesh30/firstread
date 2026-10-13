from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from firstread_api.config import Settings
from firstread_api.main import create_app


@pytest.fixture
def settings() -> Settings:
    return Settings(app_env="test", readyz_timeout_s=0.2)


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as c:
        yield c
