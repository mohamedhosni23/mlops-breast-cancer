"""
Run Multiple Pipeline Experiments
=================================
Runs several pipeline variations for comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training_pipeline import training_pipeline

if __name__ == "__main__":
    
    # Experiment 1: Baseline (already ran, but let's log it)
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 1: Baseline (100 trees, depth=10)")
    print("="*60)
    training_pipeline(n_estimators=100, max_depth=10)
    
    # Experiment 2: More trees
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 2: More trees (200 trees, depth=10)")
    print("="*60)
    training_pipeline(n_estimators=200, max_depth=10)
    
    # Experiment 3: Deeper trees
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 3: Deeper trees (100 trees, depth=20)")
    print("="*60)
    training_pipeline(n_estimators=100, max_depth=20)
    
    # Experiment 4: Smaller model
    print("\n" + "="*60)
    print("🔬 EXPERIMENT 4: Smaller model (50 trees, depth=5)")
    print("="*60)
    training_pipeline(n_estimators=50, max_depth=5)
    
    print("\n" + "="*60)
    print("✅ ALL EXPERIMENTS COMPLETE!")
    print("="*60)
