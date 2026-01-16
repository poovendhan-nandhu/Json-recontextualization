"""
Example Usage: Cartedo Validation Agent

Demonstrates how to use the validation agent to generate reports.
"""

import asyncio
from datetime import datetime

from .validation_agent import ValidationAgent, validate_single_adaptation
from .report_formatter import format_markdown_report


def create_sample_run_data(
    run_id: str,
    run_number: int,
    has_entity_leakage: bool = False,
    alignment_score: float = 0.95,
) -> dict:
    """Create sample run data for demonstration."""

    # Sample adapted JSON
    adapted_json = {
        "topicWizardData": {
            "simulationName": "Airline Customer Service Excellence",
            "overview": "Learn to handle passenger inquiries and service recovery",
            "simulationFlow": [
                {
                    "name": "Welcome",
                    "type": "stage",
                    "data": {
                        "name": "Welcome to SkyWings Airlines",
                        "description": "Introduction to airline operations"
                    }
                },
                {
                    "name": "Task1",
                    "type": "stage",
                    "data": {
                        "name": "Handle Booking Inquiry",
                        "description": "Help a passenger with their booking"
                    }
                }
            ],
            "assessmentCriterion": [
                {
                    "id": "klo1",
                    "keyLearningOutcome": "Demonstrate effective passenger communication"
                },
                {
                    "id": "klo2",
                    "keyLearningOutcome": "Apply service recovery techniques"
                }
            ],
            "workplaceScenario": {
                "background": {
                    "organizationName": "SkyWings Airlines" if not has_entity_leakage else "BurgerKing Fast Food",
                    "aboutOrganization": "Leading regional airline"
                }
            }
        }
    }

    # Sample factsheet
    factsheet = {
        "company": {
            "name": "SkyWings Airlines",
            "industry": "Aviation"
        },
        "poison_list": ["BurgerKing", "Fast Food", "restaurant", "menu", "order"],
        "replacement_hints": {
            "order": "booking",
            "customer": "passenger"
        }
    }

    # Sample validation report
    validation_report = {
        "shard_results": {
            "rubrics": [
                {
                    "rule_id": "entity_removal",
                    "score": 0.0 if has_entity_leakage else 1.0,
                    "issues": [{"message": "Found 'BurgerKing' reference"}] if has_entity_leakage else []
                },
                {
                    "rule_id": "structure_integrity",
                    "score": 1.0,
                    "issues": []
                }
            ]
        }
    }

    # Sample alignment report
    alignment_report = {
        "results": [
            {
                "rule_id": "klo_to_questions",
                "score": alignment_score,
                "issues": [] if alignment_score >= 0.95 else [
                    {"description": "KLO1 not fully mapped to questions"}
                ]
            },
            {
                "rule_id": "scenario_coherence",
                "score": 0.92,
                "issues": []
            }
        ]
    }

    return {
        "run_id": run_id,
        "run_number": run_number,
        "adapted_json": adapted_json,
        "validation_report": validation_report,
        "alignment_report": alignment_report,
        "fix_results": {},
        "factsheet": factsheet,
    }


async def demo_multi_run_validation():
    """Demonstrate validation across multiple runs."""

    print("=" * 60)
    print("CARTEDO VALIDATION AGENT - DEMO")
    print("=" * 60)

    # Create sample runs - 19 passing, 1 failing
    runs_data = []

    # 19 successful runs
    for i in range(1, 20):
        runs_data.append(create_sample_run_data(
            run_id=f"run-{i:03d}",
            run_number=i,
            has_entity_leakage=False,
            alignment_score=0.96,
        ))

    # 1 failing run (entity leakage)
    runs_data.append(create_sample_run_data(
        run_id="run-020",
        run_number=20,
        has_entity_leakage=True,
        alignment_score=0.92,
    ))

    # Create validation agent
    agent = ValidationAgent(
        original_scenario="Fast Food Operations",
        target_scenario="Airline Operations",
        simulation_purpose="Customer Service Training",
        acceptance_threshold=0.95,
    )

    # Generate report
    report_data = agent.report_generator.validate_and_report(runs_data)
    markdown_report = format_markdown_report(report_data)

    print("\n" + markdown_report)

    # Also show quick summary
    print("\n" + "=" * 60)
    print("QUICK SUMMARY (for APIs/dashboards)")
    print("=" * 60)

    summary = agent.generate_quick_summary(runs_data)
    for key, value in summary.items():
        print(f"  {key}: {value}")

    return markdown_report


async def demo_single_run_validation():
    """Demonstrate validation of a single run."""

    print("\n" + "=" * 60)
    print("SINGLE RUN VALIDATION")
    print("=" * 60)

    run_data = create_sample_run_data(
        run_id="single-test",
        run_number=1,
        has_entity_leakage=False,
        alignment_score=0.98,
    )

    report = validate_single_adaptation(
        adapted_json=run_data["adapted_json"],
        factsheet=run_data["factsheet"],
        validation_report=run_data["validation_report"],
        alignment_report=run_data["alignment_report"],
        original_scenario="Fast Food",
        target_scenario="Airline",
    )

    print("\n" + report)


if __name__ == "__main__":
    asyncio.run(demo_multi_run_validation())
    asyncio.run(demo_single_run_validation())
