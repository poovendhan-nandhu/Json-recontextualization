"""
Test the streaming transformation endpoint
"""
import requests
import json
import time

BASE_URL = "http://localhost:8000"

# Minimal test data
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

print("="*70)
print("🌊 STREAMING TRANSFORMATION TEST")
print("="*70)
print()

payload = {
    "input_json": minimal_input,
    "selected_scenario": 1
}

print("📡 Connecting to streaming endpoint...")
print("🔄 Transforming HarvestBowls → TrendWave (Fashion Retail)")
print()

start_time = time.time()

try:
    response = requests.post(
        f"{BASE_URL}/api/v1/transform/stream",
        json=payload,
        stream=True,
        timeout=120
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    else:
        print("✅ Connected! Receiving events:")
        print("-" * 70)
        
        final_result = None
        
        # Read stream line by line
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                # SSE format: "data: {json}"
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]  # Remove "data: " prefix
                    try:
                        event_data = json.loads(data_str)
                        event_type = event_data.get('event')
                        
                        if event_type == 'start':
                            print(f"🚀 {event_data.get('message')}")
                        
                        elif event_type == 'node_start':
                            print(f"⚙️  [{event_data.get('node')}] {event_data.get('message')}")
                        
                        elif event_type == 'node_complete':
                            node = event_data.get('node')
                            status = event_data.get('status')
                            duration = event_data.get('duration_ms', 0)
                            
                            status_icon = "✅" if status == "success" else "❌"
                            print(f"{status_icon} [{node}] Completed in {duration}ms")
                        
                        elif event_type == 'complete':
                            final_result = event_data.get('result')
                            print(f"\n🎉 Transformation Complete!")
                        
                        elif event_type == 'error':
                            print(f"❌ Error: {event_data.get('message')}")
                    
                    except json.JSONDecodeError:
                        print(f"⚠️  Could not parse: {data_str[:100]}")
        
        elapsed = time.time() - start_time
        
        print("-" * 70)
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
        
        if final_result:
            report = final_result.get('validation_report', {})
            
            print("\n" + "="*70)
            print("📊 VALIDATION REPORT")
            print("="*70)
            print(f"Final Status: {report.get('final_status')}")
            print(f"Schema Pass: {report.get('schema_pass')}")
            print(f"Locked Fields Compliant: {report.get('locked_fields_compliance')}")
            print(f"Consistency Score: {report.get('scenario_consistency_score', 0):.2f}")
            print(f"Runtime: {report.get('runtime_ms')}ms")
            print(f"Changed Paths: {len(report.get('changed_paths', []))}")
            
            # Save output
            with open('stream_output.json', 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            print("\n💾 Full output saved to: stream_output.json")
        
        print("\n" + "="*70)

except requests.exceptions.Timeout:
    print(f"⏱️ Request timed out after {time.time() - start_time:.1f} seconds")

except KeyboardInterrupt:
    print("\n\n⛔ Interrupted by user")

except Exception as e:
    print(f"❌ Error: {str(e)}")

print("="*70)
