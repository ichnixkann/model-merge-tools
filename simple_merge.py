#!/usr/bin/env python3
"""
Simple script to merge two Hugging Face models using linear interpolation.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
import os
import shutil

def simple_merge(model1_path, model2_path, output_path, merge_ratio=0.5, hf_token=None):
    """
    Merge two models using simple linear interpolation.
    
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
    
    print(f"Loading model 1 from {model1_path}...")
    print("This may take several minutes for large models...")
    try:
        model1 = AutoModelForCausalLM.from_pretrained(
            model1_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload1"
        )
        tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
        print(f"Model 1 loaded successfully: {model1.config.architectures}")
        print(f"Model 1 parameters: {sum(p.numel() for p in model1.parameters()) / 1e9:.1f}B")
    except Exception as e:
        print(f"Error loading model 1: {e}")
        print("Suggestions:")
        print("1. Check if the model name is correct")
        print("2. Verify your Hugging Face token has access rights")
        print("3. Ensure you have sufficient disk space and memory")
        return False
    
    print(f"Loading model 2 from {model2_path}...")
    print("This may take several minutes for large models...")
    try:
        model2 = AutoModelForCausalLM.from_pretrained(
            model2_path, 
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder="./offload2"
        )
        print(f"Model 2 loaded successfully: {model2.config.architectures}")
        print(f"Model 2 parameters: {sum(p.numel() for p in model2.parameters()) / 1e9:.1f}B")
    except Exception as e:
        print(f"Error loading model 2: {e}")
        print("Suggestions:")
        print("1. Check if the model name is correct")
        print("2. Verify your Hugging Face token has access rights")
        print("3. Ensure you have sufficient disk space and memory")
        return False
    
    # Check if models have compatible architectures
    if str(model1.config.architectures) != str(model2.config.architectures):
        print("WARNING: Models have different architectures!")
        print(f"Model 1: {model1.config.architectures}")
        print(f"Model 2: {model2.config.architectures}")
        print("This merge might not work properly.")
    
    print("Performing linear interpolation merge...")
    
    # Get state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Create merged state dict
    merged_state_dict = {}
    mismatched_keys = []
    
    # Linear interpolation: merged = (1-ratio) * model1 + ratio * model2
    total_keys = len(state_dict1.keys())
    processed_keys = 0
    
    for key in state_dict1.keys():
        processed_keys += 1
        if processed_keys % 100 == 0:
            print(f"Processing: {processed_keys}/{total_keys} keys ({processed_keys/total_keys*100:.1f}%)")
        
        if key in state_dict2:
            # Check if shapes match
            if state_dict1[key].shape == state_dict2[key].shape:
                # Linear interpolation merge
                merged_param = (1 - merge_ratio) * state_dict1[key] + merge_ratio * state_dict2[key]
                merged_state_dict[key] = merged_param
            else:
                print(f"WARNING: Shape mismatch for {key}: {state_dict1[key].shape} vs {state_dict2[key].shape}")
                mismatched_keys.append(key)
                # Use model1's parameter
                merged_state_dict[key] = state_dict1[key]
        else:
            # Parameter only in model1
            merged_state_dict[key] = state_dict1[key]
    
    print(f"Processing complete: {processed_keys}/{total_keys} keys")
    
    # Check for keys only in model2
    for key in state_dict2.keys():
        if key not in state_dict1:
            print(f"WARNING: Key {key} only in model2, skipping")
    
    if mismatched_keys:
        print(f"Total mismatched keys: {len(mismatched_keys)}")
    
    # Load merged state dict into model1
    print("Loading merged weights...")
    try:
        model1.load_state_dict(merged_state_dict)
    except Exception as e:
        print(f"Error loading merged weights: {e}")
        return False
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    try:
        model1.save_pretrained(output_path, safe_serialization=True)
        tokenizer1.save_pretrained(output_path)
        print("Merge completed successfully!")
        print(f"Merged model saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return False

def upload_to_huggingface(model_path, repo_id, hf_token):
    """
    Upload the merged model to Hugging Face.
    
    Args:
        model_path: Path to the merged model
        repo_id: Target repository ID (username/repo-name)
        hf_token: Hugging Face token for authentication
    """
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
    
    parser = argparse.ArgumentParser(description="Merge two Hugging Face models")
    parser.add_argument("--model1", type=str, default="DeepHat/DeepHat-V1-7B", 
                        help="First model path")
    parser.add_argument("--model2", type=str, default="OBLITERATUS/Gemma-4-12B-OBLITERATED",
                        help="Second model path")
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
    success = simple_merge(args.model1, args.model2, args.output, args.ratio, args.token)
    
    if success and args.upload:
        if not args.token:
            print("Error: Token required for upload")
            return
        upload_to_huggingface(args.output, args.upload, args.token)

if __name__ == "__main__":
    main()