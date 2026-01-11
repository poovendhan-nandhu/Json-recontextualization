"""
Simple API test script
"""
import requests
import json

BASE_URL = "http://localhost:8000"

# Test 1: Health Check
print("=" * 60)
print("TEST 1: Health Check")
print("=" * 60)
response = requests.get(f"{BASE_URL}/api/v1/health")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

# Test 2: Root endpoint
print("=" * 60)
print("TEST 2: Root Endpoint")
print("=" * 60)
response = requests.get(f"{BASE_URL}/")
print(f"Status Code: {response.status_code}")
print(f"Response: {json.dumps(response.json(), indent=2)}")
print()

# Test 3: Transform endpoint with minimal data
print("=" * 60)
print("TEST 3: Transform Endpoint with Streaming")
print("=" * 60)

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

try:
    print("🌊 Connecting to streaming endpoint...")
    print("🔄 Transforming: HarvestBowls → TrendWave")
    print()
    
    import time
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/api/v1/transform/stream-openai",
        json={
            "input_json": minimal_input,
            "selected_scenario": 1  # Transform to TrendWave/Fashion scenario
        },
        stream=True,
        timeout=120
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    else:
        print("✅ Streaming progress:")
        print("-" * 60)
        
        final_result = None
        openai_chars = 0
        show_openai_preview = True
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                if decoded.startswith('data: '):
                    try:
                        event = json.loads(decoded[6:])
                        event_type = event.get('event')
                        
                        if event_type == 'node_start':
                            node = event.get('node', '')
                            print(f"⚙️  {node}")
                        elif event_type == 'node_complete':
                            node = event.get('node', '')
                            print(f"✅ {node}")
                        elif event_type == 'openai_stream_start':
                            print(f"🤖 OpenAI generating (streaming)...")
                        elif event_type == 'openai_chunk':
                            chunk = event.get('chunk', '')
                            openai_chars += len(chunk)
                            # Show first 200 chars as preview
                            if show_openai_preview and openai_chars <= 200:
                                print(chunk, end='', flush=True)
                            elif show_openai_preview and openai_chars > 200:
                                print("... [streaming continues]")
                                show_openai_preview = False
                        elif event_type == 'complete':
                            final_result = event.get('result')
                            print(f"\n🎉 Complete!")
                        elif event_type == 'error':
                            print(f"❌ {event.get('message')}")
                    except:
                        pass
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"\n⏱️  Total time: {elapsed:.2f}s")
        
        if final_result:
            print("\n✅ Transformation Successful!")
            report = final_result['validation_report']
            print(f"Final Status: {report['final_status']}")
            print(f"Schema Pass: {report['schema_pass']}")
            print(f"Locked Fields Compliant: {report['locked_fields_compliance']}")
            print(f"Consistency Score: {report['scenario_consistency_score']:.2f}")
            print(f"Changed Paths: {len(report['changed_paths'])}")
            print(f"Runtime: {report['runtime_ms']}ms")
            print(f"Retries: {report['retries']}")
            
            # Show a sample of transformed content
            transformed = final_result['output_json']['topicWizardData']
            print(f"\n📝 Transformed Organization: {transformed['workplaceScenario']['background']['organizationName']}")
            print(f"📝 Transformed Lesson: {transformed['lessonInformation']['lesson'][:100]}...")
            
            # Save output
            with open('test_output.json', 'w', encoding='utf-8') as f:
                json.dump(final_result['output_json'], f, indent=2, ensure_ascii=False)
            print("\n💾 Full output saved to: test_output.json")
        
except requests.exceptions.Timeout:
    print("⏱️ Request timed out (OpenAI may be slow or API key issue)")
except Exception as e:
    print(f"❌ Error: {str(e)}")

print("\n" + "=" * 60)
print("Tests Complete!")
print("=" * 60)
