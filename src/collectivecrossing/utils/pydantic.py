"""Pydantic utilities for the collective crossing environment."""

from pydantic import BaseModel, ConfigDict


class ConfigClass(BaseModel):
    """A class that configures the pydantic model with comprehensive settings."""

    model_config = ConfigDict(
        # Field validation behavior
        extra="forbid",  # Forbid extra fields
        validate_assignment=True,  # Validate fields when assigned
        validate_default=True,  # Validate default values
        arbitrary_types_allowed=False,  # Don't allow arbitrary types
        # Serialization settings
        use_enum_values=True,  # Use enum values instead of enum objects
        populate_by_name=True,  # Allow population by field name
        # Performance settings
        frozen=True,  # Model is immutable
        # Error handling
        loc_by_alias=True,  # Use field aliases in error locations
        # Field behavior
        validate_by_name=True,  # Allow population by field name
        # Custom JSON encoders (example)
        json_encoders={
            # Add custom encoders here if needed
        },
        # Extra schema properties
        json_schema_extra={"examples": [{"example_field": "example_value"}]},
    )
