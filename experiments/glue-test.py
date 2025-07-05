# Debug script to test GLUE evaluation
import sys
import os
sys.path.append('/mnt/sda/hoyeon/unified-model-editing')

from transformers import AutoModelForCausalLM, AutoTokenizer
from glue_eval.glue_eval import GLUEEval

# Test GLUE evaluation directly
def test_glue_eval():
    print("🔍 Testing GLUE Evaluation...")
    
    # Load model and tokenizer
    model_name = "gpt2-xl"
    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.pad_token = tok.eos_token
    
    # Set up few-shot parameters
    number_of_few_shots_dict = {
        'nli_number_of_few_shots': 2,
        'sst_number_of_few_shots': 3,
        'mmlu_number_of_few_shots': 3
    }
    
    # Initialize GLUE evaluator
    glue_eval = GLUEEval(model, tok, 20, **number_of_few_shots_dict)
    
    # Test evaluation
    glue_results = {'edit_num': -1}
    flags = [True, True, True, False, False, False, False, False]  # sst, mmlu, nli only
    
    try:
        glue_results = glue_eval.evaluate(glue_results, "test_output.json", False, *flags)
        print("✅ GLUE evaluation successful!")
        print("Results structure:")
        for key, value in glue_results.items():
            print(f"  {key}: {value}")
            
        # Check if task-specific results are present
        expected_tasks = ['sst', 'nli', 'mmlu']
        missing_tasks = [task for task in expected_tasks if task not in glue_results]
        if missing_tasks:
            print(f"❌ Missing task results: {missing_tasks}")
        else:
            print("✅ All expected task results present!")
            
    except Exception as e:
        print(f"❌ GLUE evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_glue_eval()