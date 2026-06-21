#!/usr/bin/env python3
"""
Efficient model merging using SLERP and layer-by-layer processing.
This approach is more memory-efficient for large models.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import HfApi
import os
import gc
from safetensors.torch import save_file
import json

def slerp(val, low, high):
    """Spherical Linear Interpolation between two tensors."""
    low_norm = low / torch.norm(low, dim=1, keepdim=True)
    high_norm = high / torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm * high_norm).sum(1))
    so = torch.sin(omega)
    if torch.any(so < 1e-6):
        # Fall back to linear interpolation if vectors are nearly parallel
        return (1 - val) * low + val * high
    return torch.sin((1 - val) * omega) / so * low + torch.sin(val * omega) / so * high

def layer_wise_slerp_merge(model1_path, model2_path, output_path, merge_ratio=0.5, hf_token=None):
    """
    Merge models layer by layer using SLERP for better memory efficiency.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model  
        output_path: Path to save merged model
        merge_ratio: Ratio for merging (0.5 = equal blend)
        hf_token: Hugging Face token for authentication
    """
    if hf_token:
        os.environ['HF_TOKEN'] = hf_token
        print("HF_TOKEN set from parameter")
    
    print("=== Efficient SLERP Merge ===")
    print(f"Model 1: {model1_path}")
    print(f"Model 2: {model2_path}")
    print(f"Merge ratio: {merge_ratio}")
    print(f"Output: {output_path}")
    print()
    
    # Load configs first to check compatibility
    print("Loading configurations...")
    try:
        config1 = AutoConfig.from_pretrained(model1_path, trust_remote_code=True)
        config2 = AutoConfig.from_pretrained(model2_path, trust_remote_code=True)
        
        print(f"Model 1 architecture: {config1.architectures}")
        print(f"Model 2 architecture: {config2.architectures}")
        print(f"Model 1 hidden size: {getattr(config1, 'hidden_size', 'N/A')}")
        print(f"Model 2 hidden size: {getattr(config2, 'hidden_size', 'N/A')}")
    except Exception as e:
        print(f"Error loading configs: {e}")
        return False
    
    # Load tokenizers
    print("Loading tokenizers...")
    try:
        tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
        tokenizer2 = AutoTokenizer.from_pretrained(model2_path, trust_remote_code=True)
        print("Tokenizers loaded successfully")
    except Exception as e:
        print(f"Error loading tokenizers: {e}")
        return False
    
    # Use model1's tokenizer as base
    output_tokenizer = tokenizer1
    
    # Load models with low memory usage
    print("Loading models with memory optimization...")
    print("This may take several minutes...")
    
    try:
        # Load model 1 with CPU offloading
        model1 = AutoModelForCausalLM.from_pretrained(
            model1_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"Model 1 loaded: {sum(p.numel() for p in model1.parameters()) / 1e9:.1f}B parameters")
        
        # Load model 2 with CPU offloading  
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path,
            torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"Model 2 loaded: {sum(p.numel() for p in model2.parameters()) / 1e9:.1f}B parameters")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Trying with device_map='auto' instead...")
        try:
            model1 = AutoModelForCausalLM.from_pretrained(
                model1_path,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            model2 = AutoModelForCausalLM.from_pretrained(
                model2_path,
                torch_dtype=torch.float16,
                device_map="auto", 
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        except Exception as e2:
            print(f"Error with device_map='auto': {e2}")
            return False
    
    print("Models loaded successfully!")
    print()
    
    # Get state dicts
    print("Extracting state dictionaries...")
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    print(f"Model 1 has {len(state_dict1)} parameters")
    print(f"Model 2 has {len(state_dict2)} parameters")
    print()
    
    # Create merged state dict
    merged_state_dict = {}
    mismatched_keys = []
    skipped_keys = []
    
    print("Starting SLERP merge...")
    print("Processing parameters in batches for memory efficiency...")
    
    total_keys = len(state_dict1)
    batch_size = 50  # Process in batches
    processed_batches = 0
    
    for i, key in enumerate(state_dict1.keys()):
        if i % batch_size == 0:
            processed_batches += 1
            print(f"Batch {processed_batches}: Processing keys {i+1}-{min(i+batch_size, total_keys)}/{total_keys} ({(i+1)/total_keys*100:.1f}%)")
        
        if key in state_dict2:
            param1 = state_dict1[key]
            param2 = state_dict2[key]
            
            # Check if shapes match
            if param1.shape == param2.shape:
                # Use SLERP for weight matrices, linear for biases
                if len(param1.shape) >= 2 and param1.numel() > 1000:
                    # Reshape for SLERP (flatten to 2D)
                    original_shape = param1.shape
                    param1_flat = param1.view(param1.shape[0], -1)
                    param2_flat = param2.view(param2.shape[0], -1)
                    
                    # Apply SLERP
                    merged_flat = slerp(merge_ratio, param1_flat, param2_flat)
                    merged_param = merged_flat.view(original_shape)
                else:
                    # Use linear interpolation for small parameters
                    merged_param = (1 - merge_ratio) * param1 + merge_ratio * param2
                
                merged_state_dict[key] = merged_param
            else:
                print(f"Shape mismatch for {key}: {param1.shape} vs {param2.shape}")
                mismatched_keys.append(key)
                # Use model1's parameter
                merged_state_dict[key] = param1
        else:
            # Parameter only in model1
            skipped_keys.append(key)
            merged_state_dict[key] = state_dict1[key]
        
        # Clear memory periodically
        if i % 100 == 0:
            gc.collect()
    
    print()
    print(f"Merge completed!")
    print(f"Total parameters processed: {len(merged_state_dict)}")
    print(f"Mismatched keys: {len(mismatched_keys)}")
    print(f"Keys only in model1: {len(skipped_keys)}")
    print()
    
    # Check for keys only in model2
    keys_only_in_model2 = [k for k in state_dict2.keys() if k not in state_dict1]
    if keys_only_in_model2:
        print(f"Keys only in model2 ({len(keys_only_in_model2)}): {keys_only_in_model2[:5]}...")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model using safetensors for efficiency
    print("Saving merged model...")
    try:
        # Convert to CPU and save
        for key in merged_state_dict:
            if merged_state_dict[key].device != torch.device('cpu'):
                merged_state_dict[key] = merged_state_dict[key].cpu()
        
        # Save using safetensors
        save_file(merged_state_dict, os.path.join(output_path, "model.safetensors"))
        
        # Save config and tokenizer
        config1.save_pretrained(output_path)
        output_tokenizer.save_pretrained(output_path)
        
        # Save generation config
        generation_config = model1.generation_config
        generation_config.save_pretrained(output_path)
        
        print(f"Merged model saved to {output_path}")
        print("Merge completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return False
    
    finally:
        # Clean up
        del model1, model2, state_dict1, state_dict2, merged_state_dict
        gc.collect()

def upload_to_huggingface(model_path, repo_id, hf_token):
    """Upload the merged model to Hugging Face."""
    print(f"Uploading to {repo_id}...")
    try:
        api = HfApi(token=hf_token)
        
        # Create repository if it doesn't exist
        try:
            api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
            print(f"Repository {repo_id} ready")
        except Exception as e:
            print(f"Repository check: {e}")
        
        # Upload files
        api.upload_folder(
            folder_path=model_path,
            repo_id=repo_id,
            repo_type="model",
            token=hf_token
        )
        print(f"Successfully uploaded to {repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")
        return False

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Efficient SLERP merge for large models")
    parser.add_argument("--model1", type=str, default="./model1",
                        help="First model path (local or HF repo)")
    parser.add_argument("--model2", type=str, default="OBLITERATUS/Gemma-4-12B-OBLITERATED",
                        help="Second model path (local or HF repo)")
    parser.add_argument("--output", type=str, default="./merged_model",
                        help="Output path for merged model")
    parser.add_argument("--ratio", type=float, default=0.5,
                        help="Merge ratio (0.5 = equal blend)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face token for authentication")
    parser.add_argument("--upload", type=str, default=None,
                        help="Upload to this repo after merge (format: username/repo-name)")
    
    args = parser.parse_args()
    
    # Perform merge
    success = layer_wise_slerp_merge(args.model1, args.model2, args.output, args.ratio, args.token)
    
    if success and args.upload:
        if not args.token:
            print("Error: Token required for upload")
            return
        upload_to_huggingface(args.output, args.upload, args.token)

if __name__ == "__main__":
    main()