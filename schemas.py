"""HTTP request and response contracts for the ingestion API."""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class TelemetryBatch(BaseModel):
    session_id: Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9:_-]+$")]
    reaction_times_ms: list[Annotated[float, Field(gt=0, le=5_000)]]
    movement_speeds: list[Annotated[float, Field(ge=0, le=100)]]
    click_timestamps_ms: list[Annotated[float, Field(ge=0)]]
    aim_movements: list[bool]

    @field_validator("click_timestamps_ms")
    @classmethod
    def timestamps_must_be_increasing(cls, values: list[float]) -> list[float]:
        if len(values) < 2 or any(right <= left for left, right in zip(values, values[1:])):
            raise ValueError("must contain at least two strictly increasing timestamps")
        return values

    @field_validator("reaction_times_ms", "movement_speeds", "aim_movements")
    @classmethod
    def streams_must_not_be_empty(cls, values: list[object]) -> list[object]:
        if not values:
            raise ValueError("must not be empty")
        return values


class QueuedResponse(BaseModel):
    session_id: str
    task_id: str
    status: str = "queued"
