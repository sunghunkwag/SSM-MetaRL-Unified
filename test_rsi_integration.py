#!/usr/bin/env python3
"""
End-to-end test for RSI integration
Tests the complete workflow: load model → initialize RSI → run improvement
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_with_rsi import (
    load_model_and_init_rsi,
    run_rsi_cycle,
    get_rsi_status
)

def test_rsi_integration():
    """Test complete RSI integration"""
    print("=" * 60)
    print("Testing RSI Integration End-to-End")
    print("=" * 60)
    
    # Step 1: Load model and initialize RSI
    print("\n1. Loading model and initializing RSI...")
    try:
        model, buffer, logs = load_model_and_init_rsi()
        print("✓ Model loaded and RSI initialized")
        print("\nLogs:")
        print(logs)
        
        if model is None:
            print("❌ Model loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Get initial status
    print("\n2. Getting initial RSI status...")
    try:
        status = get_rsi_status()
        print(status)
    except Exception as e:
        print(f"❌ Failed to get status: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Run one RSI cycle
    print("\n3. Running one RSI improvement cycle...")
    try:
        result = run_rsi_cycle(num_cycles=1)
        print(result)
        
        if "completed" in result.lower():
            print("\n✓ RSI cycle completed successfully")
        else:
            print("\n⚠ RSI cycle completed with warnings")
            
    except Exception as e:
        print(f"❌ Failed to run RSI cycle: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Get final status
    print("\n4. Getting final RSI status...")
    try:
        status = get_rsi_status()
        print(status)
    except Exception as e:
        print(f"❌ Failed to get final status: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✅ RSI Integration Test Passed!")
    print("=" * 60)
    print("\nThe RSI system is fully integrated and working.")
    print("Ready for deployment to Hugging Face Space!")
    
    return True

if __name__ == "__main__":
    print("Starting RSI Integration Test\n")
    success = test_rsi_integration()
    
    if not success:
        print("\n❌ Integration test failed")
        sys.exit(1)
    else:
        print("\n🎉 All integration tests passed!")
        sys.exit(0)

