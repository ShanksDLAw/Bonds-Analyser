import pytest
import json
from pathlib import Path
from datetime import datetime
from src.feedback.engine import FeedbackEngine
from src.feedback.scenarios import ScenarioEngine, SCENARIO_PRESETS

@pytest.fixture
def feedback_engine(tmp_path):
    """Create a FeedbackEngine instance with temporary storage"""
    return FeedbackEngine(storage_path=str(tmp_path))

@pytest.fixture
def scenario_engine(feedback_engine):
    """Create a ScenarioEngine instance"""
    return ScenarioEngine(feedback_engine)

def test_feedback_initialization(feedback_engine):
    """Test feedback engine initialization and default values"""
    assert feedback_engine.storage_path.exists()
    assert feedback_engine.active_adjustments == {
        "model_weights": {"fundamental": 0.7, "market": 0.3},
        "scenarios": {},
        "user_overrides": []
    }

def test_save_feedback(feedback_engine):
    """Test saving feedback to JSON file"""
    test_input = {
        "model_weights": {"fundamental": 0.8, "market": 0.2},
        "scenarios": {"test_scenario": {"yield_multiplier": 1.5}},
        "overrides": [{"metric": "spread", "value": 0.05}]
    }
    
    feedback_engine.save_feedback(test_input)
    
    # Verify file was created
    files = list(feedback_engine.storage_path.glob("*.json"))
    assert len(files) == 1
    
    # Verify content
    saved_data = json.loads(files[0].read_text())
    assert saved_data["model_weights"] == test_input["model_weights"]
    assert "test_scenario" in saved_data["scenarios"]
    assert len(saved_data["user_overrides"]) == 1

def test_apply_adjustments(feedback_engine):
    """Test applying feedback adjustments to metrics"""
    raw_metrics = {
        "fundamental_risk": 0.3,
        "market_risk": 0.4,
        "yield": 0.05,
        "spread": 0.02
    }
    
    # Set custom weights and scenario
    feedback_engine.save_feedback({
        "model_weights": {"fundamental": 0.6, "market": 0.4},
        "scenarios": {"test": {"yield_multiplier": 1.5, "spread_multiplier": 2.0}}
    })
    
    adjusted = feedback_engine.apply_adjustments(raw_metrics)
    
    # Verify combined risk calculation
    assert adjusted["combined_risk"] == pytest.approx(0.3 * 0.6 + 0.4 * 0.4)
    
    # Verify scenario multipliers
    assert adjusted["yield"] == pytest.approx(0.05 * 1.5)
    assert adjusted["spread"] == pytest.approx(0.02 * 2.0)

def test_scenario_engine(scenario_engine):
    """Test scenario engine functionality"""
    # Test applying preset scenario
    scenario_engine.apply_scenario("recession_2023")
    saved = scenario_engine.feedback.active_adjustments["scenarios"]
    assert "recession_2023" in saved
    assert saved["recession_2023"] == SCENARIO_PRESETS["recession_2023"]
    
    # Test custom scenario
    custom_params = {
        "yield_multiplier": 1.3,
        "spread_multiplier": 1.4,
        "description": "Test scenario"
    }
    scenario_engine.create_custom_scenario(custom_params)
    
    saved = scenario_engine.feedback.active_adjustments["scenarios"]
    custom_scenarios = {k: v for k, v in saved.items() if k.startswith("custom_")}
    assert len(custom_scenarios) == 1
    assert list(custom_scenarios.values())[0] == custom_params

def test_feedback_persistence(feedback_engine):
    """Test feedback persistence across instances"""
    # Save initial feedback
    feedback_engine.save_feedback({
        "model_weights": {"fundamental": 0.9, "market": 0.1}
    })
    
    # Create new instance with same storage path
    new_engine = FeedbackEngine(storage_path=str(feedback_engine.storage_path))
    
    # Verify loaded weights
    assert new_engine.active_adjustments["model_weights"] == {
        "fundamental": 0.9,
        "market": 0.1
    }

def test_invalid_feedback_handling(feedback_engine):
    """Test handling of invalid feedback data"""
    # Test with missing fields
    feedback_engine.save_feedback({"invalid_key": "value"})
    assert feedback_engine.active_adjustments["model_weights"] == {
        "fundamental": 0.7,
        "market": 0.3
    }
    
    # Test with invalid file
    invalid_file = feedback_engine.storage_path / "invalid_feedback.json"
    invalid_file.write_text("invalid json content")
    
    # Should fall back to defaults
    new_engine = FeedbackEngine(storage_path=str(feedback_engine.storage_path))
    assert new_engine.active_adjustments["model_weights"] == {
        "fundamental": 0.7,
        "market": 0.3
    }