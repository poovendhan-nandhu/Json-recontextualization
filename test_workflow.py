"""Test the scenario re-contextualization workflow."""
import json
import pytest
from src.graph.workflow import scenario_workflow
from src.utils.helpers import compute_sha256


# Sample minimal test data
TEST_INPUT = {
    "topicWizardData": {
        "lessonInformation": {
            "level": "Practice",
            "lesson": "HarvestBowls must respond to Nature's Crust's $1 menu."
        },
        "scenarioOptions": [
            "HarvestBowls faces competition from Nature's Crust $1 menu",
            "TechCorp faces pricing pressure from competitor",
            "RetailCo deals with discount promotion"
        ],
        "selectedScenarioOption": "HarvestBowls faces competition from Nature's Crust $1 menu",
        "assessmentCriterion": [{"test": "data"}],
        "selectedAssessmentCriterion": [{"test": "data"}],
        "industryAlignedActivities": [{"test": "data"}],
        "selectedIndustryAlignedActivities": [{"test": "data"}],
        "simulationName": "Responding to $1 Menu Challenge",
        "workplaceScenario": {
            "scenario": "HarvestBowls needs a strategy",
            "background": {
                "organizationName": "HarvestBowls",
                "aboutOrganization": "A fast-casual restaurant"
            }
        }
    }
}


def test_ingestor_validates_input():
    """Test that ingestor validates input structure."""
    state = {
        "input_json": TEST_INPUT,
        "selected_scenario": 1,
        "node_logs": [],
        "validation_errors": []
    }
    
    from src.graph.nodes import ingestor_node
    result = ingestor_node(state)
    
    assert "locked_field_hashes" in result
    assert "scenario_options" in result
    assert len(result["scenario_options"]) == 3


def test_locked_fields_have_hashes():
    """Test that locked fields are hashed."""
    state = {
        "input_json": TEST_INPUT,
        "selected_scenario": 1,
        "node_logs": [],
        "validation_errors": []
    }
    
    from src.graph.nodes import ingestor_node
    result = ingestor_node(state)
    
    hashes = result["locked_field_hashes"]
    assert "scenarioOptions" in hashes
    assert "assessmentCriterion" in hashes
    assert len(hashes["scenarioOptions"]) == 64  # SHA-256 hex length


def test_analyzer_extracts_entities():
    """Test that analyzer extracts scenario entities."""
    from src.graph.nodes import extract_entities_from_scenario
    
    scenario = "HarvestBowls faces competition from Nature's Crust $1 menu"
    entities = extract_entities_from_scenario(scenario)
    
    assert "brand" in entities
    assert entities["brand"] == "HarvestBowls"


def test_entity_mapping_built():
    """Test that entity mapping is constructed."""
    from src.graph.nodes import build_entity_mapping
    
    current = {"brand": "HarvestBowls", "competitor": "Nature's Crust"}
    target = {"brand": "TechCorp", "competitor": "CompetitorX"}
    
    mapping = build_entity_mapping(current, target)
    
    assert mapping["HarvestBowls"] == "TechCorp"
    assert mapping["Nature's Crust"] == "CompetitorX"


def test_same_scenario_short_circuits():
    """Test that selecting the same scenario short-circuits."""
    state = {
        "input_json": TEST_INPUT,
        "selected_scenario": 0,
        "node_logs": [],
        "validation_errors": [],
        "retry_count": 0,
        "final_status": "PENDING"
    }
    
    from src.graph.nodes import ingestor_node, analyzer_node
    
    state = ingestor_node(state)
    state = analyzer_node(state)
    
    # Should short-circuit when same scenario
    assert state.get("final_status") == "OK"
    assert state.get("transformed_json") == TEST_INPUT


def test_hash_computation_deterministic():
    """Test that hash computation is deterministic."""
    from src.utils.helpers import compute_sha256
    
    data = {"test": "data", "nested": {"key": "value"}}
    
    hash1 = compute_sha256(data)
    hash2 = compute_sha256(data)
    
    assert hash1 == hash2
    assert len(hash1) == 64


def test_json_diff_detects_changes():
    """Test that JSON diff correctly identifies changes."""
    from src.utils.helpers import generate_json_diff
    
    original = {"a": "original", "b": {"c": "value"}}
    modified = {"a": "changed", "b": {"c": "value"}}
    
    changes = generate_json_diff(original, modified)
    
    assert len(changes) > 0


def test_keyword_search_finds_terms():
    """Test that keyword search finds terms in nested JSON."""
    from src.utils.helpers import search_keywords
    
    data = {
        "field1": "HarvestBowls is great",
        "nested": {
            "field2": "Nature's Crust launched"
        }
    }
    
    findings = search_keywords(data, ["HarvestBowls", "Nature's Crust"])
    
    assert len(findings) == 2


def test_workflow_state_initialization():
    """Test that workflow can be initialized."""
    initial_state = {
        "input_json": TEST_INPUT,
        "selected_scenario": 1,
        "node_logs": [],
        "validation_errors": [],
        "retry_count": 0,
        "final_status": "PENDING"
    }
    
    # This tests that the workflow graph is properly constructed
    assert scenario_workflow is not None


if __name__ == "__main__":
    print("Running tests...")
    pytest.main([__file__, "-v"])
