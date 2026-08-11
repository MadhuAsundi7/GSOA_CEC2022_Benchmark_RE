#!/usr/bin/env bash
# reproduce.sh — compile and run all algorithms to reproduce results from scratch
# Warning: this will overwrite result files. Expected runtime: ~30-60 min.
set -e
echo "=== GSOA Full Reproducibility Run ==="
echo "This will rerun all 30-independent-run experiments."
echo "Press Ctrl+C within 5 seconds to cancel."
sleep 5

bash build_all.sh all

echo ""
echo "=== Running GSOA CEC2022 D=10 (30 runs x 12 functions) ==="
cd src/gsoa/cec2022 && cp -r ../../cec2022_evaluator/input_data_d10 input_data
./gsoa_d10 && cd ../../..

echo "=== Running GSOA CEC2022 D=20 ==="
cd src/gsoa/cec2022 && cp -r ../../cec2022_evaluator/input_data_d20 input_data
./gsoa_d20 && cd ../../..

echo "=== Running GSOA Mechanical ==="
for prob in b1 b2 b3; do
    prob_dir=$(ls -d src/gsoa/mechanical/${prob^^}* | head -1)
    cd $prob_dir && ./gsoa_$prob && cd ../../..
done

echo "=== Running Baselines CEC2022 ==="
cd src/baselines/cec2022 && cp -r ../../cec2022_evaluator/input_data_d10 input_data
./baselines && cd ../../..

echo "=== Running EA4eig (Python) ==="
cd src/competitors/EA4eig
g++ -O3 -std=c++11 -shared -fPIC -o libcec22.so \
    ../../cec2022_evaluator/cec22_test_func.cpp wrapper.cpp
python3 run_batch.py 10 1 12 && cd ../../..

echo "=== Generating figures ==="
python3 analysis/plot_results.py

echo ""
echo "=== Done! Results in results/, figures in analysis/outputs/ ==="
