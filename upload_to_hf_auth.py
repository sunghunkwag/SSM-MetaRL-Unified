#!/usr/bin/env python3
"""
Upload trained model and files to Hugging Face Space with authentication
"""
from huggingface_hub import HfApi, login
import os
import sys

# Hugging Face token (from command line argument)
if len(sys.argv) > 1:
    token = sys.argv[1]
else:
    print("Error: No token provided")
    sys.exit(1)

# Files to upload
files_to_upload = [
    "cartpole_hybrid_real_model.pth",
    "app.py",
    "MODEL_GENERATION_REPORT.md",
    "train_and_save_model.py",
    "verify_model.py",
    "training_log.txt"
]

# Hugging Face Space ID
space_id = "stargatek1/SSM-MetaRL-Unified"

print("=" * 60)
print("Uploading files to Hugging Face Space")
print("=" * 60)
print(f"Space: {space_id}")
print(f"Files to upload: {len(files_to_upload)}")
print()

# Check if files exist
for file in files_to_upload:
    if os.path.exists(file):
        size = os.path.getsize(file)
        print(f"✓ {file} ({size:,} bytes)")
    else:
        print(f"✗ {file} NOT FOUND")

print()
print("Authenticating and uploading...")
print()

try:
    # Login with token
    login(token=token)
    
    # Initialize API
    api = HfApi()
    
    # Upload each file
    for file in files_to_upload:
        if os.path.exists(file):
            print(f"Uploading {file}...")
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=space_id,
                repo_type="space",
                token=token,
                commit_message=f"Add {file} - pre-trained model and updated app"
            )
            print(f"✓ Uploaded {file}")
        else:
            print(f"✗ Skipped {file} (not found)")
    
    print()
    print("=" * 60)
    print("✅ Upload completed successfully!")
    print("=" * 60)
    print(f"View at: https://huggingface.co/spaces/{space_id}")
    print()
    print("The Space will automatically rebuild with the new files.")
    
except Exception as e:
    print()
    print("=" * 60)
    print("❌ Upload failed")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

