"""
Test with sample_input.json file (STREAMING VERSION)
"""
import requests
import json
import time

# Load sample input (with UTF-8 encoding to handle special characters)
with open('sample_main.json', 'r', encoding='utf-8') as f:
    sample_data = json.load(f)

print("="*60)
print("SAMPLE FILE TRANSFORMATION TEST (STREAMING)")
print("="*60)
print(f"Loaded sample input")
print(f"File size: {len(json.dumps(sample_data))} characters (~{len(json.dumps(sample_data))//4} tokens)")
scenario_opt = sample_data['topicWizardData'].get('selectedScenarioOption', 'N/A')
print(f"Current scenario: {str(scenario_opt)[:100]}...")
print()

# Transform to scenario index 1 (TrendWave/Fashion) using streaming
print("[*] Starting streaming transformation...")
print("[*] This is a large file - you'll see real-time progress")
print("[*] Expected time: 30-60 seconds")
print()

start_time = time.time()

try:
    response = requests.post(
        "http://localhost:8000/api/v1/pipeline",
        json={
            "input_json": sample_data,
            "target_scenario_index": 1
        },
        timeout=600  # 10 minutes for full pipeline
    )
    
    elapsed = time.time() - start_time

    if response.status_code != 200:
        print(f"[ERROR] Error: {response.status_code}")
        print(response.text[:1000])
    else:
        print("[OK] Pipeline completed!")
        print("-" * 60)
        print(f"\n[TIME] Total time: {elapsed:.2f} seconds")

        result = response.json()
        report = result.get('pipeline_summary', {})
        output = result.get('adapted_json', {})

        print("\n" + "="*60)
        print("RESULTS")
        print("="*60)

        print(f"\n[OK] Status: {result.get('final_status')}")
        print(f"   Compliance Score: {report.get('compliance_score')}")
        print(f"   Compliance Passed: {report.get('compliance_passed')}")
        print(f"   Blocker Count: {report.get('blocker_count')}")
        print(f"   Warning Count: {report.get('warning_count')}")
        print(f"   Total Runtime: {report.get('total_runtime_ms')}ms")

        # Show transformation
        if output and 'topicWizardData' in output:
            orig_ws = sample_data.get('topicWizardData', {}).get('workplaceScenario', {})
            new_ws = output.get('topicWizardData', {}).get('workplaceScenario', {})
            original_org = orig_ws.get('background', {}).get('organizationName', 'N/A')
            new_org = new_ws.get('background', {}).get('organizationName', 'N/A')
            print(f"\n[*] Transformed: {original_org} -> {new_org}")

        # Save outputs
        with open('transformed_output.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        if output:
            with open('transformed_data_only.json', 'w', encoding='utf-8') as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        print("\n[*] Saved outputs:")
        print("   - Full response: transformed_output.json")
        print("   - Data only: transformed_data_only.json")

except requests.exceptions.ReadTimeout:
    elapsed = time.time() - start_time
    print(f"\n[TIME] Request timed out after {elapsed:.1f} seconds")
    print("\n[*] Possible reasons:")
    print("   1. Sample file is very large")
    print("   2. LLM API is slow")
    print("   3. Server might still be processing - check server logs")

except Exception as e:
    elapsed = time.time() - start_time
    print(f"\n[ERROR] Error after {elapsed:.1f} seconds: {str(e)}")

print("\n" + "="*60)
print("Test Complete")
print("="*60)
