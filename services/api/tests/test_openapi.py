"""The committed OpenAPI spec is the contract P4's generated client is built from."""

from firstread_api import cli

EXPECTED_OPERATIONS = {
    ("get", "/healthz"): "getHealth",
    ("get", "/readyz"): "getReady",
    ("get", "/v1/topics"): "listTopics",
    ("get", "/v1/me"): "getMe",
    ("put", "/v1/me/topics"): "setMyTopics",
    ("delete", "/v1/me"): "deleteMe",
    ("post", "/v1/events"): "postEvents",
    ("get", "/v1/feed"): "getFeed",
    ("get", "/v1/articles/{article_id}/why"): "getWhy",
    ("get", "/v1/demo/personas"): "listPersonas",
    ("post", "/v1/demo/personas/{persona_id}/activate"): "activatePersona",
    ("get", "/v1/stats"): "getStats",
}


def test_committed_spec_matches_the_app() -> None:
    assert cli.SPEC_PATH.read_text() == cli.render(cli.spec()), (
        "services/api/openapi.json is stale: run `uv run firstread-api openapi`"
    )


def test_operation_ids_are_stable() -> None:
    paths = cli.spec()["paths"]
    got = {(m, p): op["operationId"] for p, ops in paths.items() for m, op in ops.items()}
    assert got == EXPECTED_OPERATIONS
    assert "/metrics" not in paths  # internal


def test_errors_are_documented_with_the_shared_schema() -> None:
    doc = cli.spec()
    me = doc["paths"]["/v1/me"]["get"]["responses"]
    assert me["401"]["content"]["application/json"]["schema"]["$ref"].endswith("/ErrorResponse")
    assert "ErrorResponse" in doc["components"]["schemas"]
