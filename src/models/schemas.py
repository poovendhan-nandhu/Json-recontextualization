"""Pydantic models for API requests and responses."""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


# API Models for Scenario Re-Contextualization
class TransformRequest(BaseModel):
    """Request model for transformation endpoint."""
    
    input_json: dict = Field(..., description="The input JSON to transform")
    selected_scenario: int = Field(..., description="Target scenario index")
    locked_fields: Optional[list[str]] = Field(
        None,
        description="Optional list of locked field paths (uses default if not provided)"
    )


class ValidationReport(BaseModel):
    """Validation report model."""
    
    schema_pass: bool
    locked_fields_compliance: bool
    locked_field_hashes: dict[str, str]
    changed_paths: list[str]
    scenario_consistency_score: float
    old_scenario_keywords_found: list[dict]
    runtime_ms: int
    retries: int
    openai_stats: dict
    final_status: Literal["OK", "FAIL"]


class TransformResponse(BaseModel):
    """Response model for transformation endpoint."""
    
    output_json: dict
    validation_report: ValidationReport
    execution_time_ms: int


class ValidateOnlyRequest(BaseModel):
    """Request for validation-only endpoint."""
    
    original_json: dict
    transformed_json: dict
    locked_fields: Optional[list[str]] = None


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    version: str
    openai_connected: bool


# Original schema models (preserved)
class NodeSchema(BaseModel):
    id: str
    name: str
    type: str
    properties: dict

class WorkflowSchema(BaseModel):
    id: str
    name: str
    nodes: List[NodeSchema]
    connections: List[dict]

class CreateNodeRequest(BaseModel):
    name: str
    type: str
    properties: dict

class CreateWorkflowRequest(BaseModel):
    name: str
    nodes: List[CreateNodeRequest]
    connections: List[dict]

class WorkflowResponse(BaseModel):
    id: str
    name: str
    nodes: List[NodeSchema]
    connections: List[dict]
