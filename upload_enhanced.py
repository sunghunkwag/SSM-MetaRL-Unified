#!/usr/bin/env python3
"""
Upload enhanced README to Space and create Model Hub repository
"""
from huggingface_hub import HfApi, create_repo, login
import sys
import os

if len(sys.argv) > 1:
    token = sys.argv[1]
else:
    print("Error: No token provided")
    sys.exit(1)

print("=" * 60)
print("Hugging Face Enhanced Upload")
print("=" * 60)

try:
    # Login
    login(token=token)
    api = HfApi()
    
    # 1. Update Space README
    print("\n1. Updating Space README...")
    space_id = "stargatek1/SSM-MetaRL-Unified"
    
    api.upload_file(
        path_or_fileobj="README_SPACE.md",
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
        token=token,
        commit_message="Update README with enhanced documentation and tags"
    )
    print(f"✓ Space README updated: {space_id}")
    
    # 2. Create Model Hub repository
    print("\n2. Creating Model Hub repository...")
    model_id = "stargatek1/ssm-metarl-cartpole"
    
    try:
        create_repo(
            repo_id=model_id,
            repo_type="model",
            token=token,
            exist_ok=True,
            private=False
        )
        print(f"✓ Model repository created: {model_id}")
    except Exception as e:
        print(f"ℹ Model repository may already exist: {e}")
    
    # 3. Upload model file to Model Hub
    print("\n3. Uploading model file to Model Hub...")
    api.upload_file(
        path_or_fileobj="cartpole_hybrid_real_model.pth",
        path_in_repo="cartpole_hybrid_real_model.pth",
        repo_id=model_id,
        repo_type="model",
        token=token,
        commit_message="Add pre-trained SSM-MetaRL model weights"
    )
    print(f"✓ Model file uploaded: {model_id}/cartpole_hybrid_real_model.pth")
    
    # 4. Upload Model Card
    print("\n4. Uploading Model Card...")
    api.upload_file(
        path_or_fileobj="MODEL_CARD.md",
        path_in_repo="README.md",
        repo_id=model_id,
        repo_type="model",
        token=token,
        commit_message="Add comprehensive Model Card documentation"
    )
    print(f"✓ Model Card uploaded: {model_id}/README.md")
    
    # 5. Upload supporting files to Model Hub
    print("\n5. Uploading supporting files...")
    
    supporting_files = [
        "MODEL_GENERATION_REPORT.md",
        "train_and_save_model.py",
        "verify_model.py",
        "training_log.txt"
    ]
    
    for file in supporting_files:
        if os.path.exists(file):
            api.upload_file(
                path_or_fileobj=file,
                path_in_repo=file,
                repo_id=model_id,
                repo_type="model",
                token=token,
                commit_message=f"Add {file}"
            )
            print(f"  ✓ {file}")
    
    print("\n" + "=" * 60)
    print("✅ All uploads completed successfully!")
    print("=" * 60)
    print(f"\n📍 Space: https://huggingface.co/spaces/{space_id}")
    print(f"📍 Model: https://huggingface.co/{ model_id}")
    print("\nThe Space and Model Hub are now cross-linked and optimized!")
    
except Exception as e:
    print("\n" + "=" * 60)
    print("❌ Upload failed")
    print("=" * 60)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

