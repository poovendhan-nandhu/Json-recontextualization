"""Test the sharder with sample_main.json"""
import json
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from stages.sharder import Sharder, shard_json, get_shard_summary, merge_shards
from models.shard import LockState

def main():
    # Load sample JSON
    print("Loading sample_main.json...")
    with open("sample_main.json", "r") as f:
        data = json.load(f)

    print(f"Loaded JSON with keys: {list(data.keys())}")

    # Create sharder and shard the data
    print("\n" + "="*60)
    print("SHARDING JSON")
    print("="*60)

    sharder = Sharder()
    collection = sharder.shard(data, scenario_prompt="Test scenario prompt")

    # Print summary
    summary = get_shard_summary(collection)
    print(f"\nTotal shards: {summary['total_shards']}")
    print(f"Locked shards: {summary['locked_shards']}")
    print(f"Unlocked shards: {summary['unlocked_shards']}")
    print(f"Blocker shards: {summary['blocker_shards']}")
    print(f"Total IDs extracted: {summary['total_ids_extracted']}")

    print("\n" + "-"*60)
    print("SHARD DETAILS")
    print("-"*60)

    for shard_info in summary['shards']:
        status = "🔒 LOCKED" if shard_info['locked'] else "🔓 unlocked"
        blocker = "⚠️ BLOCKER" if shard_info['is_blocker'] else ""
        print(f"\n  {shard_info['id']}: {shard_info['name']}")
        print(f"    Status: {status} {blocker}")
        print(f"    Paths: {shard_info['paths_count']}")
        print(f"    IDs extracted: {shard_info['ids_count']}")
        if shard_info['aligns_with']:
            print(f"    Aligns with: {', '.join(shard_info['aligns_with'])}")

    # Show some sample extracted content
    print("\n" + "="*60)
    print("SAMPLE CONTENT FROM SHARDS")
    print("="*60)

    for shard in collection.shards[:3]:
        print(f"\n--- {shard.name} ({shard.id}) ---")
        for path, content in list(shard.content.items())[:2]:
            if isinstance(content, dict):
                print(f"  {path}: {list(content.keys())[:5]}...")
            elif isinstance(content, list):
                print(f"  {path}: [{len(content)} items]")
            elif isinstance(content, str) and len(content) > 100:
                print(f"  {path}: {content[:100]}...")
            else:
                print(f"  {path}: {content}")

    # Test merge
    print("\n" + "="*60)
    print("TESTING MERGE")
    print("="*60)

    merged = merge_shards(collection, data)
    merged_hash = json.dumps(merged, sort_keys=True)
    original_hash = json.dumps(data, sort_keys=True)

    if merged_hash == original_hash:
        print("✅ Merge successful - output matches original (no changes made)")
    else:
        print("⚠️ Merge produced different output (expected if shards were modified)")

    # Test ID preservation
    print("\n" + "="*60)
    print("ID PRESERVATION CHECK")
    print("="*60)

    all_ids = collection.all_ids()
    print(f"Total IDs in shards: {len(all_ids)}")
    print(f"Sample IDs: {all_ids[:10]}")

    # Verify critical IDs exist
    critical_ids = [
        data.get("topicWizardData", {}).get("id"),
    ]
    for cid in critical_ids:
        if cid:
            status = "✅" if cid in all_ids else "❌"
            print(f"  {status} Main ID '{cid}' preserved: {cid in all_ids}")

    print("\n" + "="*60)
    print("SHARDER TEST COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
