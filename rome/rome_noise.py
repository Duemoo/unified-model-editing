import torch
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from .rome_main import execute_rome, upd_matrix_match_shape
from .rome_hparams import ROMEHyperParams
from util import nethook

def extract_layer_number(weight_name):
    """Extract layer number from weight name like 'transformer.h.17.mlp.c_proj.weight'"""
    import re
    match = re.search(r'\.h\.(\d+)\.', weight_name)
    if match:
        return int(match.group(1))
    return 0  # fallback

def apply_rome_noise_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: ROMEHyperParams,
    copy=False,
    return_orig_weights=False,
    noise_matching_strategy="random_rank1",
    **kwargs,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    """
    Applies noise injection that matches ROME's delta properties.
    """
    
    # Get the noise strategy from a global variable or use default
    # You can access the args through the global scope
    import sys
    strategy = noise_matching_strategy
    
    # Try to get the strategy from command line args if available
    if hasattr(sys.modules['__main__'], 'args'):
        main_args = getattr(sys.modules['__main__'], 'args')
        if hasattr(main_args, 'noise_matching_strategy'):
            strategy = main_args.noise_matching_strategy
    
    print(f"🔧 NOISE INJECTION using strategy: {strategy}")
    
    if copy:
        model = deepcopy(model)

    weights_copy = {}
    objective_distances = {}  # Add this

    for i, request in enumerate(requests):
        print(f"Processing request {i}: {request['prompt'].format(request['subject'])} -> {request['target_new']['str']}")
        
        # First, compute what ROME would do (but don't apply it)
        print("Computing ROME deltas...")
        rome_deltas = execute_rome(model, tok, request, hparams)
        print(f"ROME computed {len(rome_deltas)} deltas")
        
        # Generate matched noise for each delta
        print(f"Generating matched noise with strategy: {strategy}")
        noise_deltas = generate_matched_noise(rome_deltas, strategy)
        print(f"Generated {len(noise_deltas)} noise deltas")

        # Apply the noise instead of ROME deltas
        with torch.no_grad():
            for w_name, noise_components in noise_deltas.items():
                # Extract layer number for objective distances
                layer_num = extract_layer_number(w_name)  # You'll need this helper
                if layer_num not in objective_distances:
                    objective_distances[layer_num] = {}
                
                if isinstance(noise_components, tuple) and len(noise_components) == 2 and noise_components[0] == "full_rank_matrix":
                    # Handle full rank case differently
                    upd_matrix = noise_components[1]
                    print(f"  Applying FULL RANK noise to {w_name}")
                else:
                    # Standard rank-1 case
                    noise_u, noise_v = noise_components
                    upd_matrix = noise_u.unsqueeze(1) @ noise_v.unsqueeze(0)
                    print(f"  Applying RANK-1 noise to {w_name}")
                
                w = nethook.get_parameter(model, w_name)
                upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

                if return_orig_weights and w_name not in weights_copy:
                    assert i == 0
                    weights_copy[w_name] = w.detach().clone()

                # DIAGNOSTIC: Print change magnitude
                change_norm = torch.norm(upd_matrix).item()
                print(f"    ||change|| = {change_norm:.6f}")
                
                # Store objective distances
                objective_distances[layer_num]['delta_norm'] = change_norm
                objective_distances[layer_num]['new_weights_norm'] = torch.norm(w + upd_matrix).item()
                objective_distances[layer_num]['original_weights_norm'] = torch.norm(w).item()
                
                w[...] += upd_matrix

        print(f"✅ Matched noise successfully inserted into {list(noise_deltas.keys())}")

    return model, weights_copy, objective_distances


def generate_matched_noise(
    rome_deltas: Dict[str, Tuple[torch.Tensor, torch.Tensor]], 
    strategy: str = "random_rank1"
) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Generate noise that matches the properties of ROME deltas.
    
    Returns:
        Dictionary with same structure as rome_deltas but with matched noise
    """
    noise_deltas = {}
    
    print(f"🎲 Generating noise for {len(rome_deltas)} layers using strategy '{strategy}'")
    
    # FIXED: Single loop that does everything
    for w_name, (rome_u, rome_v) in rome_deltas.items():
        print(f"\n  Processing layer: {w_name}")
        print(f"    ROME u shape: {rome_u.shape}, norm: {torch.norm(rome_u):.6f}")
        print(f"    ROME v shape: {rome_v.shape}, norm: {torch.norm(rome_v):.6f}")
        
        # Create the original ROME rank-1 matrix for comparison
        rome_matrix = rome_u.unsqueeze(1) @ rome_v.unsqueeze(0)
        rome_frobenius = torch.norm(rome_matrix, 'fro').item()
        print(f"    ROME matrix Frobenius norm: {rome_frobenius:.6f}")
        
        if strategy == "random_rank1":
            # Test: Does direction matter in rank-1 structure?
            noise_u = torch.randn_like(rome_u)
            noise_v = torch.randn_like(rome_v)
            
            # Normalize and scale to match ROME component norms
            noise_u = noise_u / torch.norm(noise_u) * torch.norm(rome_u)
            noise_v = noise_v / torch.norm(noise_v) * torch.norm(rome_v)
            
            noise_deltas[w_name] = (noise_u.detach(), noise_v.detach())
            
        elif strategy == "full_rank":
            # Test: Does rank-1 structure matter?
            noise_matrix = torch.randn(rome_u.shape[0], rome_v.shape[0], 
                                     device=rome_u.device, dtype=rome_u.dtype)
            # Scale to match Frobenius norm
            noise_matrix = noise_matrix * (rome_frobenius / torch.norm(noise_matrix, 'fro'))
            
            # FIXED: Use a clear tuple structure for full rank
            noise_deltas[w_name] = ("full_rank_matrix", noise_matrix.detach())
            
        elif strategy == "scaled_random":
            # Test: Does any property matter beyond magnitude?
            noise_u = torch.randn_like(rome_u)
            noise_v = torch.randn_like(rome_v)
            
            # Scale to match total magnitude (product of norms)
            current_magnitude = torch.norm(noise_u) * torch.norm(noise_v)
            target_magnitude = torch.norm(rome_u) * torch.norm(rome_v)
            scale = torch.sqrt(target_magnitude / current_magnitude)
            
            noise_deltas[w_name] = (noise_u.detach() * scale, noise_v.detach() * scale)
            
        else:
            raise ValueError(f"Unknown noise matching strategy: {strategy}")
        
        # Verification: Check that we're actually generating different matrices
        if strategy != "full_rank":
            noise_u, noise_v = noise_deltas[w_name]
            noise_matrix = noise_u.unsqueeze(1) @ noise_v.unsqueeze(0)
            noise_frobenius = torch.norm(noise_matrix, 'fro').item()
            
            # Cosine similarity between ROME and noise matrices
            cos_sim = torch.nn.functional.cosine_similarity(
                rome_matrix.flatten(), noise_matrix.flatten(), dim=0
            ).item()
            
            print(f"    ✓ Noise matrix Frobenius norm: {noise_frobenius:.6f}")
            print(f"    ✓ Cosine similarity with ROME: {cos_sim:.6f}")
            print(f"    ✓ Difference norm: {torch.norm(rome_matrix - noise_matrix, 'fro'):.6f}")
            
            # Sanity checks
            if noise_frobenius < 1e-8:
                print(f"    ❌ ERROR: Noise matrix is essentially zero!")
            if cos_sim > 0.99:
                print(f"    ⚠️  WARNING: Noise very similar to ROME (cos_sim={cos_sim:.6f})")
        else:
            # For full_rank, just report the matrix norm
            noise_matrix = noise_deltas[w_name][1]
            print(f"    ✓ Full rank noise matrix Frobenius norm: {torch.norm(noise_matrix, 'fro'):.6f}")
    
    return noise_deltas