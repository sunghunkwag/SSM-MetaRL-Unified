#!/usr/bin/env python3
"""Deploy fixes to Hugging Face Space"""

from huggingface_hub import HfApi
import os

# Initialize API
api = HfApi()

# Get token from environment
token = os.environ.get('HF_TOKEN')
if not token:
    raise ValueError("Please set HF_TOKEN environment variable")

# Files to upload
files_to_upload = [
    'app.py',
    'recursive_self_improvement.py',
    'models/cartpole_hybrid_real_model.pth',
    'FIXES_SUMMARY.md',
    'benchmarks/cartpole_benchmark.py',
    'benchmarks/results/benchmark_results.json',
    'benchmarks/results/plots/adaptation_comparison.png',
    'benchmarks/results/plots/learning_curves.png',
    'benchmarks/results/tables/benchmark_results.md',
    'tests/test_rsi.py',
    'tests/test_rsi_integration.py',
    'train_improved_model.py'
]

print("Uploading files to Hugging Face Space...")

for file_path in files_to_upload:
    if os.path.exists(file_path):
        try:
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=file_path,
                repo_id="stargatek1/SSM-MetaRL-Unified",
                repo_type="space",
                token=token
            )
            print(f"✓ Uploaded: {file_path}")
        except Exception as e:
            print(f"✗ Failed to upload {file_path}: {e}")
    else:
        print(f"⚠ File not found: {file_path}")

print("\n✅ Deployment complete!")

