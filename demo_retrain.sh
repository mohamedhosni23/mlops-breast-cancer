#!/bin/bash
echo "=============================================="
echo "🔄 MLOps Retrain Demo"
echo "=============================================="

echo ""
echo "📋 Step 1: Check if retrain is needed"
echo "----------------------------------------------"
python src/retrain.py check

echo ""
echo "🔄 Step 2: Run manual retrain"
echo "----------------------------------------------"
python src/retrain.py retrain

echo ""
echo "📋 Step 3: View retrain history"
echo "----------------------------------------------"
python src/retrain.py history

echo ""
echo "=============================================="
echo "✅ Retrain Demo Complete!"
echo "=============================================="
echo ""
echo "To start scheduled retraining, run:"
echo "  python src/retrain.py schedule --interval 60"
echo ""
