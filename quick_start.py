#!/usr/bin/env python3
"""
Quick Start Test for Vector Database Environment
This script tests if your environment is ready for Quest 1
"""

import sys
import subprocess

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, install if not"""
    if import_name is None:
        import_name = package_name
    
    try:
        __import__(import_name)
        print(f"✅ {package_name} is installed")
        return True
    except ImportError:
        print(f"⚠️  {package_name} not found. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully")
            return True
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package_name}")
            return False

def test_basic_vector_operations():
    """Test basic vector operations for Quest 1"""
    print("\n🎮 Testing Quest 1 readiness...")
    
    try:
        import numpy as np
        
        # Test basic vector operations
        vec_a = np.array([1, 0, 1, 0, 1])
        vec_b = np.array([1, 0, 1, 0, 1])
        
        # Calculate cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        magnitude_a = np.linalg.norm(vec_a)
        magnitude_b = np.linalg.norm(vec_b)
        similarity = dot_product / (magnitude_a * magnitude_b)
        
        print(f"   Vector A: {vec_a}")
        print(f"   Vector B: {vec_b}")
        print(f"   Cosine Similarity: {similarity:.3f}")
        print("🎉 Ready for Quest 1: Vector Playground!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("🚀 Vector Database Environment Quick Start")
    print("=" * 50)
    
    # Check essential packages for Quest 1
    packages = [
        ("numpy", None),
        ("matplotlib", None),
        ("pandas", None)
    ]
    
    all_good = True
    for package, import_name in packages:
        if not check_and_install_package(package, import_name):
            all_good = False
    
    if all_good:
        test_basic_vector_operations()
        print("\n🎯 Next Steps:")
        print("1. Open roadMap.md and start Quest 1: Vector Playground")
        print("2. For advanced quests, install: pip install chromadb sentence-transformers")
        print("3. Activate your environment with: source venv/bin/activate")
    else:
        print("\n❌ Some packages failed to install. Please check your environment.")

if __name__ == "__main__":
    main()