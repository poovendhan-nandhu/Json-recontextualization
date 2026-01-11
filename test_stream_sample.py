"""
Test streaming with the large sample_input.json file
"""
import requests
import json
import time

# Load sample input
with open('sample_input.json', 'r', encoding='utf-8') as f:
    sample_data = json.load(f)

print("="*70)
print("🌊 STREAMING TRANSFORMATION - LARGE FILE TEST")
print("="*70)
print(f"File size: {len(json.dumps(sample_data))} characters")
print(f"Approx tokens: ~{len(json.dumps(sample_data))//4}")
print()

payload = {
    "input_json": sample_data,
    "selected_scenario": 1  # Transform to TrendWave
}

print("📡 Connecting to streaming endpoint...")
print("🔄 This is a large file - expect 30-60 seconds")
print()

start_time = time.time()

try:
    response = requests.post(
        "http://localhost:8000/api/v1/transform/stream-openai",
        json=payload,
        stream=True,
        timeout=180  # 3 minutes for large files
    )
    
    if response.status_code != 200: 
        print(f"❌ Error: {response.status_code}")
        print(response.text)
    else:
        print("✅ Connected! Receiving progress updates:")
        print("-" * 70)
        
        final_result = None
        last_update = time.time()
        
        # Read stream line by line
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                
                if decoded_line.startswith('data: '):
                    data_str = decoded_line[6:]
                    try:
                        event_data = json.loads(data_str)
                        event_type = event_data.get('event')
                        current_time = time.time()
                        elapsed = current_time - start_time
                        
                        if event_type == 'start':
                            print(f"[{elapsed:5.1f}s] 🚀 {event_data.get('message')}")
                        
                        elif event_type == 'node_start':
                            print(f"[{elapsed:5.1f}s] ⚙️  {event_data.get('node')} - {event_data.get('message')}")
                        
                        elif event_type == 'node_complete':
                            node = event_data.get('node')
                            status = event_data.get('status')
                            duration = event_data.get('duration_ms', 0)
                            
                            status_icon = "✅" if status == "success" else "❌"
                            print(f"[{elapsed:5.1f}s] {status_icon} {node} completed ({duration}ms)")
                        
                        elif event_type == 'complete':
                            final_result = event_data.get('result')
                            print(f"\n[{elapsed:5.1f}s] 🎉 Transformation Complete!")
                        
                        elif event_type == 'error':
                            print(f"[{elapsed:5.1f}s] ❌ Error: {event_data.get('message')}")
                        
                        last_update = current_time
                    
                    except json.JSONDecodeError as e:
                        print(f"⚠️  Parse error: {str(e)}")
            else:
                # Empty line - check if connection is alive
                if time.time() - last_update > 30:
                    print(f"[{time.time() - start_time:5.1f}s] ⏳ Still processing (OpenAI may be slow)...")
                    last_update = time.time()
        
        elapsed = time.time() - start_time
        
        print("-" * 70)
        print(f"\n⏱️  Total time: {elapsed:.2f} seconds")
        
        if final_result:
            report = final_result.get('validation_report', {})
            output = final_result.get('output_json', {})
            
            print("\n" + "="*70)
            print("📊 RESULTS")
            print("="*70)
            
            print(f"\n✅ Status: {report.get('final_status')}")
            print(f"   Schema Pass: {report.get('schema_pass')}")
            print(f"   Locked Fields: {report.get('locked_fields_compliance')}")
            print(f"   Consistency: {report.get('scenario_consistency_score', 0):.2f}")
            print(f"   Processing Time: {report.get('runtime_ms')}ms")
            print(f"   Changed Paths: {len(report.get('changed_paths', []))}")
            
            # Show transformation
            if 'topicWizardData' in output:
                org = output['topicWizardData'].get('workplaceScenario', {}).get('background', {}).get('organizationName')
                print(f"\n📝 Transformed Organization: {org}")
            
            # Save outputs
            with open('stream_sample_output.json', 'w', encoding='utf-8') as f:
                json.dump(final_result, f, indent=2, ensure_ascii=False)
            
            print("\n💾 Saved to: stream_sample_output.json")
        
        print("\n" + "="*70)

except requests.exceptions.Timeout:
    print(f"\n⏱️ Request timed out after {time.time() - start_time:.1f} seconds")
    print("💡 The file might be too large. Try a smaller file or increase timeout.")

except KeyboardInterrupt:
    print("\n\n⛔ Interrupted by user")

except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("="*70)
