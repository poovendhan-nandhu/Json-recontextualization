"""
Test script for leaf-based adaptation.

Run with: python test_leaf_adapter.py
"""

import asyncio
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

async def test_indexer():
    """Test the indexer module."""
    from src.core.indexer import index_leaves, filter_string_leaves, get_leaf_stats

    # Simple test JSON
    test_json = {
        "topicWizardData": {
            "id": "test123",
            "lessonInformation": {
                "level": "Practice",
                "lesson": "Students learn about Accor Hotel management",
            },
            "assessmentCriterion": [
                {
                    "id": "klo1",
                    "keyLearningOutcome": "Learners will understand hotel operations",
                }
            ]
        }
    }

    # Index leaves
    leaves = index_leaves(test_json)
    print(f"\n=== INDEXER TEST ===")
    print(f"Total leaves: {len(leaves)}")

    for path, value in leaves[:5]:
        print(f"  {path} = {str(value)[:50]}")

    # Get stats
    stats = get_leaf_stats(leaves)
    print(f"\nStats: {stats}")

    # Filter strings
    strings = filter_string_leaves(leaves)
    print(f"String leaves: {len(strings)}")

    return True


async def test_classifier():
    """Test the classifier module."""
    from src.core.indexer import index_leaves
    from src.core.classifier import LeafClassifier, LeafStrategy

    # Test JSON with entities
    test_json = {
        "id": "test123",
        "company": "Accor Hotel Group",
        "manager": "John Smith",
        "description": "This simulation covers hotel management at Accor Hotel.",
        "duration": "30 minutes",
    }

    # Factsheet with mappings
    factsheet = {
        "company": {
            "name": "Global Beverages Inc",
            "old_name": "Accor Hotel",
            "industry": "beverage",
        },
        "manager": {
            "name": "Sarah Chen",
            "old_name": "John Smith",
        },
        "poison_list": ["hotel", "Accor", "hospitality"],
    }

    # Index and classify
    leaves = index_leaves(test_json)
    classifier = LeafClassifier(factsheet)
    classified = classifier.classify(leaves)

    print(f"\n=== CLASSIFIER TEST ===")
    for clf in classified:
        print(f"  {clf.path}: {clf.strategy.value} - {clf.reason}")
        if clf.replacement:
            print(f"    -> Replacement: {clf.replacement[:50]}")

    return True


async def test_full_adaptation():
    """Test the full leaf-based adaptation with sample JSON."""
    import os

    # Check if sample JSON exists
    sample_path = "sample_main.json"
    if not os.path.exists(sample_path):
        print(f"\n=== FULL ADAPTATION TEST ===")
        print(f"Sample JSON not found at {sample_path}")
        print("Skipping full adaptation test")
        return True

    # Load sample JSON
    with open(sample_path, "r") as f:
        input_json = json.load(f)

    print(f"\n=== FULL ADAPTATION TEST ===")
    print(f"Loaded sample JSON with {len(str(input_json))} chars")

    # Run leaf-based adaptation
    from src.stages.adaptation_engine import adapt_simulation_with_leaves

    result = await adapt_simulation_with_leaves(
        input_json=input_json,
        scenario_prompt="A beverage company called Global Beverages Inc is training their sales team on market analysis and customer engagement strategies.",
    )

    print(f"\nAdaptation Result:")
    print(f"  Total leaves: {result.stats.get('total_leaves', 'N/A')}")
    print(f"  Pre-filtered (no LLM): {result.stats.get('pre_filtered', 'N/A')}")
    print(f"  LLM evaluated: {result.stats.get('llm_evaluated', 'N/A')}")
    print(f"  Changes made: {result.stats.get('changes_made', 'N/A')}")
    print(f"  Kept unchanged: {result.stats.get('kept_unchanged', 'N/A')}")
    print(f"  Total time: {result.total_time_ms}ms")

    # Save adapted JSON
    output_path = "adapted_leaf_output.json"
    with open(output_path, "w") as f:
        json.dump(result.adapted_json, f, indent=2)
    print(f"\nAdapted JSON saved to: {output_path}")

    return True


async def main():
    """Run all tests."""
    print("=" * 60)
    print("LEAF-BASED ADAPTATION TEST SUITE")
    print("=" * 60)

    try:
        await test_indexer()
        await test_classifier()
        await test_full_adaptation()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)

    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(asyncio.run(main()))
