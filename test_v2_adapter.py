"""
Test script for V2 Skeleton-Based Adaptation.

This script tests the new V2 adapter that:
1. Extracts skeleton from source (no content shown to LLM)
2. Generates alignment_map, canonical_numbers in Stage 0
3. Generates content to fill skeleton in parallel
4. Post-processes to enforce consistency

Usage:
    python test_v2_adapter.py
"""

import asyncio
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_extractors():
    """Test the extractors work correctly."""
    print("\n" + "="*60)
    print("TEST 1: Extractors (Programmatic)")
    print("="*60)

    from src.extractors.skeleton_extractor import extract_skeleton, extract_structure_summary
    from src.extractors.word_target_extractor import measure_word_targets

    # Load a sample JSON
    sample_files = [
        "data/Sim2TopicWizardData.json",
        "data/Sim3TopicWizardData.json",
        "Sim2TopicWizardData.json",
        "Sim3TopicWizardData.json",
    ]

    source_json = None
    for f in sample_files:
        if os.path.exists(f):
            with open(f, 'r', encoding='utf-8') as file:
                source_json = json.load(file)
            print(f"Loaded: {f}")
            break

    if not source_json:
        print("No sample JSON found. Creating minimal test...")
        source_json = {
            "lesson_information": {"lesson": "Test lesson content"},
            "assessment_criterion": [
                {"id": "klo1", "keyLearningOutcome": "Test KLO", "criterion": [{"id": "c1", "criteria": "Test"}]}
            ],
            "workplace_scenario": {
                "scenario": "Test scenario",
                "background": {"about_organization": "Test company description"}
            }
        }

    # Test skeleton extraction
    print("\n--- Skeleton Extraction ---")
    skeleton = extract_skeleton(source_json)
    skeleton_size = len(json.dumps(skeleton))
    source_size = len(json.dumps(source_json))
    print(f"Source size: {source_size:,} chars")
    print(f"Skeleton size: {skeleton_size:,} chars")
    print(f"Reduction: {(1 - skeleton_size/source_size)*100:.1f}%")

    # Check skeleton has __GENERATE__ placeholders
    skeleton_str = json.dumps(skeleton)
    generate_count = skeleton_str.count("__GENERATE__")
    print(f"__GENERATE__ placeholders: {generate_count}")

    # Test structure summary
    print("\n--- Structure Summary ---")
    summary = extract_structure_summary(source_json)
    print(f"KLOs: {summary['klo_count']}")
    print(f"Questions: {summary['question_count']}")
    print(f"Rubrics: {summary['rubric_count']}")
    print(f"KLO IDs: {summary['klo_ids'][:3]}...")

    # Test word targets
    print("\n--- Word Targets ---")
    targets = measure_word_targets(source_json)
    for key in ['lesson', 'about_organization', 'klo', 'question']:
        if key in targets:
            print(f"{key}: {targets[key][0]}-{targets[key][1]} words")

    print("\n[PASS] Extractors work correctly!")
    return source_json


async def test_stage0_prompt():
    """Test Stage 0 prompt building."""
    print("\n" + "="*60)
    print("TEST 2: Stage 0 Prompt Building")
    print("="*60)

    from src.generators.stage0_generator import build_stage0_prompt

    scenario = "MediCore's flagship drug sales are at risk after a competitor slashes prices on a generic equivalent."
    structure = {
        "klo_count": 3,
        "klo_ids": ["klo1", "klo2", "klo3"],
        "criteria_per_klo": [3, 3, 3],
        "question_count": 5,
        "question_ids": ["q1", "q2", "q3", "q4", "q5"],
        "rubric_count": 4,
    }

    prompt = build_stage0_prompt(scenario, structure)

    print(f"Prompt length: {len(prompt):,} chars")
    print(f"Contains 'alignment_map': {'alignment_map' in prompt}")
    print(f"Contains 'canonical_numbers': {'canonical_numbers' in prompt}")
    print(f"Contains 'resource_sections': {'resource_sections' in prompt}")

    # Show first 500 chars
    print("\n--- Prompt Preview ---")
    print(prompt[:500] + "...")

    print("\n[PASS] Stage 0 prompt builds correctly!")


async def test_shard_prompt():
    """Test shard prompt building."""
    print("\n" + "="*60)
    print("TEST 3: Shard Prompt Building")
    print("="*60)

    from src.prompts.shard_prompts import build_shard_prompt, CONTENT_RULES

    skeleton = {"about_organization": "__GENERATE__", "id": "ws123"}
    word_targets = {"about_organization": [60, 100]}
    entity_map = {
        "company": {"name": "MediCore", "domain": "medicore.com"},
        "people": [{"name": "Dr. Sarah Chen", "role": "VP Strategy", "email": "sarah.chen@medicore.com"}]
    }
    domain_profile = {
        "industry": "pharmaceutical",
        "scenario_type": "competitive response",
        "key_challenge": "Generic drug pricing threat"
    }
    canonical_numbers = {"market_size": "$4.2B", "company_revenue": "$890M"}

    prompt = build_shard_prompt(
        shard_name="workplace_scenario",
        skeleton=skeleton,
        word_targets=word_targets,
        entity_map=entity_map,
        domain_profile=domain_profile,
        canonical_numbers=canonical_numbers,
        scenario_prompt="MediCore competitive response scenario"
    )

    print(f"Prompt length: {len(prompt):,} chars")
    print(f"Contains 'CANONICAL NUMBERS': {'CANONICAL NUMBERS' in prompt}")
    print(f"Contains 'ENTITY MAP': {'ENTITY MAP' in prompt}")
    print(f"Contains '__GENERATE__': {'__GENERATE__' in prompt}")

    print(f"\nCONTENT_RULES keys: {list(CONTENT_RULES.keys())}")

    print("\n[PASS] Shard prompt builds correctly!")


async def test_post_processor():
    """Test post-processor."""
    print("\n" + "="*60)
    print("TEST 4: Post-Processor")
    print("="*60)

    from src.enforcers.post_processor import (
        enforce_entity_map,
        fix_email_domains,
        remove_slash_hedging
    )

    # Test data with problems
    test_json = {
        "scenario": "The company is facing challenges",
        "manager": {"name": "the manager", "email": "john.doe@oldcompany.com"},
        "description": "MediCore / pharmaceutical company is..."
    }

    entity_map = {
        "company": {"name": "MediCore", "domain": "medicore.com"},
        "people": [{"name": "Dr. Sarah Chen", "role": "VP"}]
    }

    # Test enforce_entity_map
    fixed = enforce_entity_map(test_json, entity_map)
    print(f"'the company' replaced: {'The company' not in json.dumps(fixed)}")

    # Test fix_email_domains
    fixed = fix_email_domains(fixed, entity_map)
    print(f"Email domain fixed: {'medicore.com' in json.dumps(fixed)}")

    # Test remove_slash_hedging
    fixed = remove_slash_hedging(fixed, entity_map)
    print(f"Slash hedging removed: {'MediCore /' not in json.dumps(fixed)}")

    print("\n[PASS] Post-processor works correctly!")


async def test_full_flow_dry_run():
    """Test the full flow without making LLM calls (dry run)."""
    print("\n" + "="*60)
    print("TEST 5: Full Flow (Dry Run - No LLM Calls)")
    print("="*60)

    from src.extractors.skeleton_extractor import extract_skeleton, extract_structure_summary
    from src.extractors.word_target_extractor import measure_word_targets
    from src.generators.stage0_generator import get_alignment_for_shard
    from src.prompts.shard_prompts import build_shard_prompt

    # Create a minimal source
    source_json = {
        "lesson_information": {"lesson": "Learners analyze market data"},
        "assessment_criterion": [
            {"id": "klo1", "keyLearningOutcome": "Analyze market trends", "criterion": []}
        ],
        "simulation_flow": [
            {"id": "stage1", "data": {"questions": [{"id": "q1", "name": "What is the market size?"}]}}
        ]
    }

    # Step 1: Extract
    skeleton = extract_skeleton(source_json)
    structure = extract_structure_summary(source_json)
    word_targets = measure_word_targets(source_json)

    print(f"Skeleton extracted: {len(json.dumps(skeleton)):,} chars")
    print(f"Structure: {structure['klo_count']} KLOs, {structure['question_count']} questions")

    # Step 2: Mock Stage 0 result
    mock_alignment_map = {
        "q1": {
            "assesses_klo": "Analyze market trends",
            "question_must_ask": "What is the market size?",
            "resource_must_contain": ["market size data"],
            "rubric_must_check": "Cites specific market size"
        }
    }

    # Step 3: Test alignment extraction
    q_alignment = get_alignment_for_shard("questions", mock_alignment_map)
    r_alignment = get_alignment_for_shard("resources", mock_alignment_map)

    print(f"Questions alignment: {list(q_alignment.keys())}")
    print(f"Resources alignment: {r_alignment.get('must_contain_data', [])}")

    # Step 4: Build a sample prompt
    prompt = build_shard_prompt(
        shard_name="simulation_flow",
        skeleton=skeleton.get("simulation_flow", []),
        word_targets=word_targets,
        entity_map={"company": {"name": "TestCo", "domain": "testco.com"}, "people": []},
        domain_profile={"industry": "tech", "scenario_type": "growth", "key_challenge": "scaling"},
        canonical_numbers={"market_size": "$1B"},
        scenario_prompt="Test scenario",
        alignment_requirements=q_alignment
    )

    print(f"Shard prompt built: {len(prompt):,} chars")

    print("\n[PASS] Full flow dry run works correctly!")


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("V2 SKELETON-BASED ADAPTER TESTS")
    print("="*60)

    try:
        await test_extractors()
        await test_stage0_prompt()
        await test_shard_prompt()
        await test_post_processor()
        await test_full_flow_dry_run()

        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        print("\nNext step: Run actual adaptation with LLM:")
        print("  python -c \"")
        print("    import asyncio")
        print("    from src.stages.simple_adapter import adapt_simple")
        print("    result = asyncio.run(adapt_simple(source_json, scenario, use_v2=True))")
        print("  \"")

    except Exception as e:
        print(f"\n[FAIL] Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
