# 🤖 Model Merge Tools

Tools and scripts for merging Hugging Face language models using various strategies including SLERP, task arithmetic, and linear interpolation.

## 🎯 Purpose

This repository contains tools for merging large language models, specifically designed for merging:
- **Model 1**: DeepHat/DeepHat-V1-7B
- **Model 2**: OBLITERATUS/Gemma-4-12B-OBLITERATED
- **Result**: ichnixkann/schweinshaxxe (7.6B parameters, SLERP merged)

## Prerequisites

1. **Python 3.10 or higher** (Python 3.14 may have compatibility issues)
2. **Hugging Face account** with a valid access token
3. **Sufficient disk space** (~30GB+ for the models)
4. **GPU** recommended (for faster processing)

## IDE Configuration

### VS Code
The project includes `.vscode/settings.json` with the correct Python interpreter path:
```bash
# VS Code should automatically detect the virtual environment
# If not, select: View > Command Palette > Python: Select Interpreter
# Choose: ./venv/bin/python
```

### PyCharm / JetBrains IDEs
1. File > Settings > Project > Python Interpreter
2. Click "Add Interpreter" > "Add Local Interpreter"
3. Select "Existing Environment"
4. Browse to `./venv/bin/python`
5. Click OK

### Alternative IDEs
Source the environment setup script:
```bash
source setup_env.sh
```

### Terminal Usage
Always activate the virtual environment:
```bash
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### Quick Environment Check
Verify your environment is configured correctly:
```bash
./run_with_env.sh verify_env.py
```

This will check that:
- Virtual environment is active
- All required dependencies are installed
- Python interpreter is correct

## Setup

### 1. Create a new virtual environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify environment

```bash
./run_with_env.sh verify_env.py
```

### 4. Configure IDE (Optional)

The project includes IDE configuration files:
- **VS Code**: `.vscode/settings.json` (auto-detected)
- **PyCharm**: `.idea/merge.iml` (auto-detected)
- **Other IDEs**: See `IDE_SETUP.md` for detailed instructions

If your IDE shows import errors, it's likely using the system Python instead of the virtual environment. See `IDE_SETUP.md` for solutions.

### 3. Authenticate with Hugging Face

```bash
huggingface-cli login
```

Enter your token when prompted, or set it as an environment variable:

```bash
export HF_TOKEN="your_token_here"
```

## Method 1: Using the Simple Merge Script

The provided `simple_merge.py` script performs a linear interpolation merge:

```bash
python simple_merge.py \
  --model1 "DeepHat/DeepHat-V1-7B" \
  --model2 "OBLITERATUS/Gemma-4-12B-OBLITERATED" \
  --output "./merged_model" \
  --ratio 0.5 \
  --token "your_token_here" \
  --upload "ichnixkann/schweinshaxxe"
```

This will:
1. Download both models
2. Perform the merge using linear interpolation
3. Save the merged model locally
4. Upload to your Hugging Face repository

## Method 2: Using MergeKit (Recommended for Task Arithmetic)

For proper task arithmetic merging, use the mergekit library:

### Install mergekit

```bash
pip install mergekit
```

### Use the provided configuration

The `merge_config.yaml` file is configured for task arithmetic:

```bash
mergekit-yaml merge_config.yaml ./merged_model --allow-crimes
```

### Upload to Hugging Face

```bash
huggingface-cli upload ichnixkann/schweinshaxxe ./merged_model . --repo-type model
```

## Alternative: Manual Step-by-Step Process

If automated scripts fail, follow these steps:

### 1. Download models manually

```bash
# Download model 1
hf download DeepHat/DeepHat-V1-7B --local-dir ./model1

# Download model 2  
hf download OBLITERATUS/Gemma-4-12B-OBLITERATED --local-dir ./model2
```

### 2. Run the merge script

```bash
python simple_merge.py \
  --model1 "./model1" \
  --model2 "./model2" \
  --output "./merged_model" \
  --ratio 0.5
```

### 3. Upload to Hugging Face

```bash
huggingface-cli upload ichnixkann/schweinshaxxe ./merged_model . --repo-type model
```

## Troubleshooting

### Authentication Issues

If you get authentication errors:
1. Verify your token at https://huggingface.co/settings/tokens
2. Ensure the token has "write" permissions for uploading
3. Try setting the token as an environment variable: `export HF_TOKEN="your_token"`

### Memory Issues

If you run out of memory:
1. Use `low_cpu_mem_usage=True` in the script
2. Download models to disk first, then load with `device_map="auto"`
3. Use a machine with more RAM or a GPU

### Architecture Compatibility

The two models have different architectures (7B vs 12B parameters). This may cause:
- Shape mismatches during merge
- Unexpected behavior in the merged model

If this occurs, consider:
- Using SLERP merging instead of task arithmetic
- Merging only specific layers
- Using a model size closer to the target

### Python Version Issues

Python 3.14 is very new and may have compatibility issues. Recommended:
- Use Python 3.10 or 3.11
- Or use conda: `conda create -n merge python=3.11`

## Expected Results

- **Merged model size**: Approximately the size of the larger model (12B)
- **Merge time**: 30 minutes to several hours depending on hardware
- **Upload time**: Additional time depending on network speed

## Verification

After merging, verify the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./merged_model")
tokenizer = AutoTokenizer.from_pretrained("./merged_model")

# Test generation
inputs = tokenizer("Hello, world!", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

## Support

For issues with:
- **MergeKit**: https://github.com/arcee-ai/mergekit
- **Transformers**: https://github.com/huggingface/transformers
- **Hugging Face Hub**: https://huggingface.co/docs/huggingface_hub

## Files Provided

### Core Merge Scripts
1. `efficient_merge.py` - SLERP-based merge script (recommended for large models)
2. `simple_merge.py` - Simple linear interpolation merge script
3. `merge_models.py` - Original task arithmetic script
4. `merge_config.yaml` - MergeKit configuration for task arithmetic

### Utility Scripts
5. `run_merge.sh` - Quick merge script with environment setup
6. `requirements.txt` - Python dependencies

### Configuration
7. `.vscode/settings.json` - VS Code configuration
8. `.idea/merge.iml` - PyCharm configuration
9. `.python-version` - Python version specification

### Documentation
10. `README.md` - This instruction file