"""
Test with sample_input.json file (STREAMING VERSION)
"""
import requests
import json
import time

# Load sample input (with UTF-8 encoding to handle special characters)
with open('sample_input.json', 'r', encoding='utf-8') as f:
    sample_data = json.load(f)

print("="*60)
print("SAMPLE FILE TRANSFORMATION TEST (STREAMING)")
print("="*60)
print(f"Loaded sample input")
print(f"File size: {len(json.dumps(sample_data))} characters (~{len(json.dumps(sample_data))//4} tokens)")
print(f"Current scenario: {sample_data['topicWizardData']['selectedScenarioOption'][:100]}...")
print()

# Transform to scenario index 1 (TrendWave/Fashion) using streaming
print("🌊 Starting streaming transformation...")
print("🔄 This is a large file - you'll see real-time progress")
print("⏱️  Expected time: 30-60 seconds")
print()

start_time = time.time()

try:
    response = requests.post(
        "http://localhost:8000/api/v1/transform/stream-openai",
        json={
            "input_json": sample_data,
            "selected_scenario": 1
        },
        stream=True,
        timeout=180  # 3 minutes for large files
    )
    
    if response.status_code != 200:
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    else:
        print("✅ Connected! Streaming progress:")
        print("-" * 60)
        
        final_result = None
        openai_chars = 0
        last_update = time.time()
        node_count = 0
        
        for line in response.iter_lines():
            if line:
                decoded = line.decode('utf-8')
                
                if decoded.startswith('data: '):
                    try:
                        event = json.loads(decoded[6:])
                        event_type = event.get('event')
                        elapsed = time.time() - start_time
                        
                        if event_type == 'start':
                            print(f"[{elapsed:5.1f}s] 🚀 {event.get('message')}")
                        
                        elif event_type == 'node_start':
                            node = event.get('node', '')
                            node_count += 1
                            print(f"[{elapsed:5.1f}s] ⚙️  [{node_count}/6] {node}")
                        
                        elif event_type == 'node_complete':
                            node = event.get('node', '')
                            print(f"[{elapsed:5.1f}s] ✅ {node} completed")
                        
                        elif event_type == 'openai_progress':
                            print(f"[{elapsed:5.1f}s] 🤖 {event.get('message')}")
                        
                        elif event_type == 'openai_stream_start':
                            print(f"[{elapsed:5.1f}s] 🌊 OpenAI streaming started...")
                        
                        elif event_type == 'openai_chunk':
                            openai_chars += len(event.get('chunk', ''))
                            # Update every second or every 1000 chars
                            if time.time() - last_update > 1.0 or openai_chars % 1000 < 50:
                                print(f"[{elapsed:5.1f}s] 📝 Generated {openai_chars:,} characters...", end='\r')
                                last_update = time.time()
                        
                        elif event_type == 'complete':
                            final_result = event.get('result')
                            print(f"\n[{elapsed:5.1f}s] 🎉 Transformation Complete!                    ")
                        
                        elif event_type == 'error':
                            print(f"[{elapsed:5.1f}s] ❌ Error: {event.get('message')}")
                    
                    except json.JSONDecodeError:
                        pass
        
        elapsed = time.time() - start_time
        print("-" * 60)
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
        
        if final_result:
            report = final_result.get('validation_report', {})
            output = final_result.get('output_json', {})
            
            print("\n" + "="*60)
            print("📊 RESULTS")
            print("="*60)
            
            print(f"\n✅ Status: {report.get('final_status')}")
            print(f"   Schema Pass: {report.get('schema_pass')}")
            print(f"   Locked Fields: {report.get('locked_fields_compliance')}")
            print(f"   Consistency: {report.get('scenario_consistency_score', 0):.2f}")
            print(f"   Processing Time: {report.get('runtime_ms')}ms")
            print(f"   Changed Paths: {len(report.get('changed_paths', []))}")
            
            # Show transformation
            if 'topicWizardData' in output:
                original_org = sample_data['topicWizardData']['workplaceScenario']['background']['organizationName']
                new_org = output['topicWizardData'].get('workplaceScenario', {}).get('background', {}).get('organizationName')
                print(f"\n📝 Transformed: {original_org} → {new_org}")
            
            # Show some violations/warnings if any
            if report.get('locked_field_violations'):
                print(f"\n⚠️  Locked Field Violations:")
                for v in report.get('locked_field_violations', [])[:3]:
                    print(f"   - {v}")
            
            if report.get('warnings'):
                print(f"\n⚠️  Warnings ({len(report.get('warnings', []))}):")
                for w in report.get('warnings', [])[:3]:
                    print(f"   - {w}")
            
            # Save outputs
            with open('transformed_output.json', 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            with open('transformed_data_only.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            
            print("\n💾 Saved outputs:")
            print("   - Full response: transformed_output.json")
            print("   - Data only: transformed_data_only.json")

except requests.exceptions.ReadTimeout:
    elapsed = time.time() - start_time
    print(f"\n⏱️ Request timed out after {elapsed:.1f} seconds")
    print("\n💡 Possible reasons:")
    print("   1. Sample file is very large (causing OpenAI to take longer)")
    print("   2. OpenAI API is slow right now")
    print("   3. Server might still be processing - check server logs")
    print("\n🔧 Try:")
    print("   - Check the uvicorn terminal for progress")
    print("   - Run with a smaller file first (test_api.py)")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n❌ Error after {elapsed:.1f} seconds: {str(e)}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
