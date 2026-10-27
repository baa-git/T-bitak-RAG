"""Shared base class for domain models."""

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Immutable value object: domain data is created once and never mutated in place."""

    model_config = ConfigDict(frozen=True)
