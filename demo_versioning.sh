#!/bin/bash
echo "=============================================="
echo "🚀 MLOps Deployment Demo: v1 → v2 → Rollback"
echo "=============================================="

echo ""
echo "📦 Step 1: Create v1 (baseline model)"
echo "----------------------------------------------"
python api/version_manager.py create --version v1 --description "Baseline Random Forest model"

echo ""
echo "📦 Step 2: Train improved model and create v2"
echo "----------------------------------------------"
# Use the optimized model as v2
cp models/model_optimized.joblib models/model.joblib 2>/dev/null || echo "Using current model"
python api/version_manager.py create --version v2 --description "Optuna optimized model"

echo ""
echo "📋 Step 3: List all versions"
echo "----------------------------------------------"
python api/version_manager.py list

echo ""
echo "🚀 Step 4: Deploy v2"
echo "----------------------------------------------"
python api/version_manager.py deploy --version v2

echo ""
echo "🔄 Step 5: Simulate issue - Rollback to v1"
echo "----------------------------------------------"
python api/version_manager.py rollback --version v1

echo ""
echo "📋 Step 6: Verify current version"
echo "----------------------------------------------"
python api/version_manager.py current

echo ""
echo "=============================================="
echo "✅ Deployment Demo Complete!"
echo "=============================================="
