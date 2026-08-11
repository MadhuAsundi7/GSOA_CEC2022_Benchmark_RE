#!/usr/bin/env bash
# build_all.sh — compile all C++ algorithms
# Usage: bash build_all.sh [d10|d20|mech|all]  (default: all)
set -e
TARGET=${1:-all}
CEC_EVAL="src/cec2022_evaluator/cec22_test_func.cpp"
CXXFLAGS="-O2 -std=c++11"

echo "=== GSOA Build Script ==="

# ── GSOA CEC2022 ────────────────────────────────────────────────────────
if [[ "$TARGET" == "d10" || "$TARGET" == "all" ]]; then
    echo "[1/8] GSOA CEC2022 D=10"
    cd src/gsoa/cec2022
    cp -r ../../cec2022_evaluator/input_data_d10 input_data 2>/dev/null || true
    mkdir -p results
    g++ $CXXFLAGS -o gsoa_d10 ../../../$CEC_EVAL gsoa_cec2022_d10.cpp
    cd ../../..
fi

if [[ "$TARGET" == "d20" || "$TARGET" == "all" ]]; then
    echo "[2/8] GSOA CEC2022 D=20"
    cd src/gsoa/cec2022
    cp -r ../../cec2022_evaluator/input_data_d20 input_data 2>/dev/null || true
    g++ $CXXFLAGS -o gsoa_d20 ../../../$CEC_EVAL gsoa_cec2022_d20.cpp
    cd ../../..
fi

# ── GSOA Mechanical ─────────────────────────────────────────────────────
if [[ "$TARGET" == "mech" || "$TARGET" == "all" ]]; then
    for prob_dir in B1_pressure_vessel B2_spring B3_welded_beam; do
        prob=$(echo $prob_dir | cut -d_ -f1 | tr '[:upper:]' '[:lower:]')
        name=$(echo $prob_dir | cut -d_ -f2-)
        echo "[3-5/8] GSOA Mechanical: $prob_dir"
        cd src/gsoa/mechanical/$prob_dir
        mkdir -p results
        g++ $CXXFLAGS -o gsoa_$prob \
            ../../../../src/problem_evaluators/problem_eval_${prob}_${name}.cpp \
            gsoa_$prob.cpp
        cd ../../../..
    done
fi

# ── Baselines ─────────────────────────────────────────────────────────
if [[ "$TARGET" == "d10" || "$TARGET" == "all" ]]; then
    echo "[6/8] Baselines CEC2022"
    cd src/baselines/cec2022
    cp -r ../../cec2022_evaluator/input_data_d10 input_data 2>/dev/null || true
    mkdir -p results
    g++ $CXXFLAGS -o baselines ../../../$CEC_EVAL gsa_pso_de_ga_cec2022.cpp
    cd ../../..
fi

if [[ "$TARGET" == "mech" || "$TARGET" == "all" ]]; then
    echo "[7/8] Baselines Mechanical"
    for prob_dir in B1_pressure_vessel B2_spring B3_welded_beam; do
        prob=$(echo $prob_dir | cut -d_ -f1 | tr '[:upper:]' '[:lower:]')
        name=$(echo $prob_dir | cut -d_ -f2-)
        cd src/baselines/mechanical
        mkdir -p results
        g++ $CXXFLAGS -o baselines_$prob \
            ../../../src/problem_evaluators/problem_eval_${prob}_${name}.cpp \
            baselines_${prob}_${name}.cpp
        cd ../../..
    done
fi

# ── S-LSHADE-DP ──────────────────────────────────────────────────────
if [[ "$TARGET" == "d10" || "$TARGET" == "all" ]]; then
    echo "[8/8] S-LSHADE-DP"
    cd src/competitors/S-LSHADE-DP
    g++ $CXXFLAGS -o slshadedp main.cpp s_lshade_dp.cpp search_algorithm.cpp
    cd ../../..
fi

echo ""
echo "=== Build complete ==="
echo "Run ./src/gsoa/cec2022/gsoa_d10 to reproduce GSOA CEC2022 D=10 results."
echo "Run python3 analysis/plot_results.py to reproduce all figures."
