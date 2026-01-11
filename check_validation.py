"""
Check validation report details
"""
import json

with open('test_output.json', 'r') as f:
    data = json.load(f)

# The test_output.json only has the transformed data, not the full response
# Let's make a proper request and check the full response

import requests

minimal_input = {
    "topicWizardData": {
        "lessonInformation": {
            "level": "Practice",
            "lesson": "HarvestBowls must respond to Nature's Crust's $1 menu challenge."
        },
        "scenarioOptions": [
            "HarvestBowls faces competition from Nature's Crust $1 menu in fast-casual dining",
            "TrendWave retailer faces BOGO promotion from ChicStyles in fashion retail",
            "TechNova faces smartphone discounts from BrightEdge during holiday season"
        ],
        "selectedScenarioOption": "HarvestBowls faces competition from Nature's Crust $1 menu in fast-casual dining",
        "assessmentCriterion": [{"test": "locked data"}],
        "selectedAssessmentCriterion": [{"test": "locked data"}],
        "industryAlignedActivities": [{"test": "locked data"}],
        "selectedIndustryAlignedActivities": [{"test": "locked data"}],
        "simulationName": "Strategic Response to $1 Menu Challenge",
        "workplaceScenario": {
            "scenario": "HarvestBowls needs a strategy to counter competitor pricing",
            "background": {
                "organizationName": "HarvestBowls",
                "aboutOrganization": "A fast-casual healthy food restaurant chain"
            },
            "challenge": {
                "currentIssue": "Nature's Crust launched a $1 value menu affecting traffic"
            }
        }
    }
}

print("Making transform request...")
response = requests.post(
    "http://localhost:8000/api/v1/transform",
    json={
        "input_json": minimal_input,
        "selected_scenario": 1
    },
    timeout=30
)

if response.status_code == 200:
    result = response.json()
    
    # Save full response
    with open('full_response.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print("\n" + "="*60)
    print("VALIDATION REPORT")
    print("="*60)
    
    report = result.get('validation_report', {})
    
    print(f"Final Status: {report.get('final_status')}")
    print(f"Schema Pass: {report.get('schema_pass')}")
    print(f"Locked Fields Compliant: {report.get('locked_fields_compliance')}")
    print(f"Consistency Score: {report.get('scenario_consistency_score', 0):.2f}")
    print(f"Runtime: {report.get('runtime_ms')}ms")
    print(f"Retries: {report.get('retries')}")
    
    print(f"\n{'='*60}")
    print("LOCKED FIELD VIOLATIONS")
    print("="*60)
    violations = report.get('locked_field_violations', [])
    if violations:
        for v in violations:
            print(f"  - {v}")
    else:
        print("  None")
    
    print(f"\n{'='*60}")
    print("SCHEMA ERRORS")
    print("="*60)
    schema_errors = report.get('schema_errors', [])
    if schema_errors:
        for e in schema_errors:
            print(f"  - {e}")
    else:
        print("  None")
    
    print(f"\n{'='*60}")
    print("CHANGED PATHS ({})".format(len(report.get('changed_paths', []))))
    print("="*60)
    for path in report.get('changed_paths', [])[:10]:
        print(f"  - {path}")
    
    print(f"\n{'='*60}")
    print("WARNINGS ({})".format(len(report.get('warnings', []))))
    print("="*60)
    for w in report.get('warnings', [])[:5]:
        print(f"  - {w}")
    
    print("\n💾 Full response saved to: full_response.json")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
