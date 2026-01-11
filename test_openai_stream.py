"""
Test OpenAI streaming - see the JSON being generated in real-time!
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
print("🤖 OPENAI STREAMING TEST - Watch JSON Being Generated!")
print("="*70)
print()

payload = {
    "input_json": minimal_input,
    "selected_scenario": 1
}

print("📡 Connecting to OpenAI streaming endpoint...")
print("🔄 Transforming HarvestBowls → TrendWave")
print()

start_time = time.time()

try:
    response = requests.post(
        f"{BASE_URL}/api/v1/transform/stream-openai",
        json=payload,
        stream=True,
        timeout=180
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    else:
        print("✅ Connected! Streaming events:")
        print("-" * 70)
        
        final_result = None
        openai_output = []
        in_openai_stream = False
        
        # Read stream line by line
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        event_data = json.loads(data_str)
                        event_type = event_data.get('event')
                        elapsed = time.time() - start_time
                        
                        if event_type == 'start':
                            print(f"[{elapsed:5.1f}s] 🚀 {event_data.get('message')}")
                        
                        elif event_type == 'node_start':
                            node = event_data.get('node')
                            msg = event_data.get('message', '')
                            print(f"[{elapsed:5.1f}s] ⚙️  {node} {msg}")
                        
                        elif event_type == 'node_complete':
                            node = event_data.get('node')
                            print(f"[{elapsed:5.1f}s] ✅ {node} completed")
                        
                        elif event_type == 'openai_progress':
                            print(f"[{elapsed:5.1f}s] 🤖 {event_data.get('message')}")
                        
                        elif event_type == 'openai_stream_start':
                            print(f"\n[{elapsed:5.1f}s] 🌊 {event_data.get('message')}")
                            print("-" * 70)
                            print("📝 OpenAI Output (streaming):")
                            print()
                            in_openai_stream = True
                        
                        elif event_type == 'openai_chunk':
                            chunk = event_data.get('chunk', '')
                            total = event_data.get('total_chars', 0)
                            openai_output.append(chunk)
                            
                            # Print chunk in real-time
                            print(chunk, end='', flush=True)
                        
                        elif event_type == 'complete':
                            if in_openai_stream:
                                print()  # New line after streaming
                                print("-" * 70)
                                in_openai_stream = False
                            
                            final_result = event_data.get('result')
                            print(f"\n[{elapsed:5.1f}s] 🎉 Complete!")
                        
                        elif event_type == 'error':
                            print(f"[{elapsed:5.1f}s] ❌ Error: {event_data.get('message')}")
                            if event_data.get('traceback'):
                                print(event_data.get('traceback'))
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Parse error: {str(e)}")
        
        elapsed = time.time() - start_time
        
        print()
        print("=" * 70)
        print(f"⏱️  Total time: {elapsed:.2f} seconds")
        
        if final_result:
            report = final_result.get('validation_report', {})
            
            print()
            print("📊 VALIDATION REPORT")
            print("-" * 70)
            print(f"Status: {report.get('final_status')}")
            print(f"Schema Pass: {report.get('schema_pass')}")
            print(f"Locked Fields: {report.get('locked_fields_compliance')}")
            print(f"Consistency: {report.get('scenario_consistency_score', 0):.2f}")
            print(f"Changed Paths: {len(report.get('changed_paths', []))}")
            
            # Save outputs
            with open('openai_stream_output.json', 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            # Save just the streamed content
            with open('openai_streamed_text.txt', 'w', encoding='utf-8') as f:
                f.write(''.join(openai_output))
            
            print()
            print("💾 Saved:")
            print("   - Full result: openai_stream_output.json")
            print("   - Streamed text: openai_streamed_text.txt")
        
        print("=" * 70)

except requests.exceptions.Timeout:
    print(f"\n⏱️ Timeout after {time.time() - start_time:.1f}s")

except KeyboardInterrupt:
    print("\n\n⛔ Interrupted")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()
