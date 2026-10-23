"""Request and response models. P4 generates its client from the OpenAPI spec, so field
names and operationIds are part of the contract (``services/api/openapi.json``)."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from firstread_core.topics import TOPIC_IDS

MIN_TOPICS = 3  # SPEC section 2: readers pick at least 3 topics
MAX_EVENTS = 100
MAX_CLOCK_SKEW = timedelta(hours=24)

EventType = Literal["impression", "click", "like", "hide", "dwell"]
NEEDS_ARTICLE = frozenset({"impression", "click", "like", "hide", "dwell"})


class Envelope(BaseModel):
    """Every JSON response carries these (SPEC section 8)."""

    request_id: str
    model_version: str


class Topic(BaseModel):
    id: str
    name: str


class TopicsResponse(Envelope):
    topics: list[Topic]


class TopicsUpdate(BaseModel):
    topics: list[str] = Field(min_length=MIN_TOPICS, max_length=len(TOPIC_IDS))

    @field_validator("topics")
    @classmethod
    def _known_and_unique(cls, v: list[str]) -> list[str]:
        unknown = sorted(set(v) - set(TOPIC_IDS))
        if unknown:
            raise ValueError(f"unknown topics: {unknown}")
        if len(set(v)) != len(v):
            raise ValueError("topics must be unique")
        return v


class HistoryItem(BaseModel):
    article_id: uuid.UUID
    title: str
    source_name: str
    category: str
    clicked_at: datetime


class MeResponse(Envelope):
    user_id: uuid.UUID
    is_anonymous: bool
    topics: list[str]
    hidden_sources: list[str]
    active_persona: str | None
    history_count: int
    recent_history: list[HistoryItem]


class EventIn(BaseModel):
    event_id: uuid.UUID
    type: EventType
    article_id: uuid.UUID | None = None
    position: int | None = Field(default=None, ge=0, le=999)
    dwell_ms: int | None = Field(default=None, ge=0, le=3_600_000)
    scope: Literal["article", "source"] | None = None
    request_id: str | None = Field(default=None, max_length=64)
    client_ts: datetime | None = None

    @model_validator(mode="after")
    def _per_type(self) -> EventIn:
        if self.type in NEEDS_ARTICLE and self.article_id is None:
            raise ValueError(f"{self.type} needs article_id")
        if self.type == "dwell" and self.dwell_ms is None:
            raise ValueError("dwell needs dwell_ms")
        if self.type != "dwell" and self.dwell_ms is not None:
            raise ValueError("dwell_ms is only for dwell events")
        if self.scope is not None and self.type != "hide":
            raise ValueError("scope is only for hide events")
        if self.client_ts is not None:
            ts = self.client_ts if self.client_ts.tzinfo else self.client_ts.replace(tzinfo=UTC)
            if abs(datetime.now(UTC) - ts) > MAX_CLOCK_SKEW:
                raise ValueError("client_ts is more than 24 h from server time")
        return self


class EventsBatch(BaseModel):
    events: list[EventIn] = Field(min_length=1, max_length=MAX_EVENTS)

    @field_validator("events")
    @classmethod
    def _unique_ids(cls, v: list[EventIn]) -> list[EventIn]:
        if len({e.event_id for e in v}) != len(v):
            raise ValueError("event_id must be unique within a batch")
        return v


class RejectedEvent(BaseModel):
    event_id: uuid.UUID
    code: Literal["unknown_article"]


class EventsResult(Envelope):
    accepted: list[uuid.UUID]
    duplicate: list[uuid.UUID]
    rejected: list[RejectedEvent]
