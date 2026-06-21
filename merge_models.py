#!/usr/bin/env python3
"""
Script to merge two Hugging Face models using task arithmetic.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import argparse
import os

def task_arithmetic_merge(model1_path, model2_path, output_path, merge_ratio=0.5):
    """
    Merge two models using task arithmetic.
    
    Args:
        model1_path: Path to first model
        model2_path: Path to second model  
        output_path: Path to save merged model
        merge_ratio: Ratio for merging (0.5 = equal blend)
    """
    print(f"Loading model 1 from {model1_path}...")
    model1 = AutoModelForCausalLM.from_pretrained(
        model1_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer1 = AutoTokenizer.from_pretrained(model1_path, trust_remote_code=True)
    
    print(f"Loading model 2 from {model2_path}...")
    model2 = AutoModelForCausalLM.from_pretrained(
        model2_path, 
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer2 = AutoTokenizer.from_pretrained(model2_path, trust_remote_code=True)
    
    print("Models loaded successfully!")
    print(f"Model 1 architecture: {model1.config.architectures}")
    print(f"Model 2 architecture: {model2.config.architectures}")
    
    # Check if models have compatible architectures
    if model1.config.architectures != model2.config.architectures:
        print("WARNING: Models have different architectures!")
        print("This merge might not work properly.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Aborting merge.")
            return
    
    print("Performing task arithmetic merge...")
    
    # Get state dicts
    state_dict1 = model1.state_dict()
    state_dict2 = model2.state_dict()
    
    # Create merged state dict
    merged_state_dict = {}
    
    # Task arithmetic: merged = base + ratio * (model2 - base)
    # Here we use model1 as base and model2 as the fine-tuned model
    for key in state_dict1.keys():
        if key in state_dict2:
            # Check if shapes match
            if state_dict1[key].shape == state_dict2[key].shape:
                # Task arithmetic merge
                delta = state_dict2[key] - state_dict1[key]
                merged_param = state_dict1[key] + merge_ratio * delta
                merged_state_dict[key] = merged_param
            else:
                print(f"WARNING: Shape mismatch for {key}: {state_dict1[key].shape} vs {state_dict2[key].shape}")
                # Use model1's parameter
                merged_state_dict[key] = state_dict1[key]
        else:
            # Parameter only in model1
            merged_state_dict[key] = state_dict1[key]
    
    # Load merged state dict into model1
    print("Loading merged weights...")
    model1.load_state_dict(merged_state_dict)
    
    # Save merged model
    print(f"Saving merged model to {output_path}...")
    model1.save_pretrained(output_path)
    tokenizer1.save_pretrained(output_path)
    
    print("Merge completed successfully!")
    print(f"Merged model saved to {output_path}")

def main():
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
    
    args = parser.parse_args()
    
    # Set token as environment variable if provided
    if args.token:
        os.environ['HF_TOKEN'] = args.token
        print("HF_TOKEN set from command line argument")
    
    # Perform merge
    task_arithmetic_merge(args.model1, args.model2, args.output, args.ratio)

if __name__ == "__main__":
    main()