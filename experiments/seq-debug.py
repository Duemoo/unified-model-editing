# debug_sequential.py - Check if sequential editing is working

import json
from pathlib import Path

def debug_sequential_editing(results_dir, alg_name, run_id):
    """Debug whether sequential editing is actually happening"""
    
    run_dir = Path(results_dir) / alg_name / run_id
    
    # Load params to check sequential flag
    params_file = run_dir / "params.json"
    if params_file.exists():
        with open(params_file, 'r') as f:
            params = json.load(f)
        print(f"🔍 Sequential setting: {params.get('sequential', 'NOT FOUND')}")
        print(f"🔍 Num edits: {params.get('n_edits', 'NOT FOUND')}")
        print(f"🔍 Algorithm: {params.get('algo_name', 'NOT FOUND')}")
    
    # Check if weight distances are increasing over time
    glue_files = list((run_dir / "glue_eval").glob("*_glue.json"))
    glue_files = sorted(glue_files, key=lambda x: int(x.name.split('_')[0]) if x.name.split('_')[0].isdigit() else -1)
    
    print(f"\n📊 Distance progression over {len(glue_files)} evaluations:")
    print("Edit# | Distance | SST Acc | NLI Acc | MMLU Present?")
    print("-" * 55)
    
    for glue_file in glue_files[:]:  # First 10 files
        with open(glue_file, 'r') as f:
            data = json.load(f)
        
        edit_num = data.get('edit_num', '?')
        
        # Check distance
        distances = data.get('distance_from_original', {})
        distance = distances.get('17', 0) if distances else 0
        
        # Check task performance
        sst_acc = data.get('sst', {}).get('correct', 0) / 100.0 if 'sst' in data else 0
        nli_acc = data.get('nli', {}).get('correct', 0) / 100.0 if 'nli' in data else 0
        mmlu_present = 'mmlu' in data
        
        print(f"{edit_num:5} | {distance:8.6f} | {sst_acc:7.3f} | {nli_acc:7.3f} | {mmlu_present}")
    
    # Check a few case files to see if edits are successful
    case_files = list(run_dir.glob("*_edits-case_*.json"))
    
    print(f"\n📝 Edit success rates in {len(case_files)} case files:")
    successful_edits = 0
    total_edits = 0
    
    for case_file in case_files[:5]:  # Check first 5
        with open(case_file, 'r') as f:
            data = json.load(f)
        
        rewrite_success = data['post']['rewrite_prompts_correct'][0] if data['post']['rewrite_prompts_correct'] else False
        if rewrite_success:
            successful_edits += 1
        total_edits += 1
        
        print(f"Case {data['case_id']}: {'✅' if rewrite_success else '❌'}")
    
    print(f"Overall success rate: {successful_edits}/{total_edits} = {successful_edits/total_edits*100:.1f}%")

if __name__ == "__main__":
    print("=== ROME ===")
    debug_sequential_editing('/mnt/sda/hoyeon/unified-model-editing/results/', 'ROME', 'run_005')
    
    print("\n=== ROME_NOISE ===")
    debug_sequential_editing('/mnt/sda/hoyeon/unified-model-editing/results/', 'ROME_NOISE', 'run_012')