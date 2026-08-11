# GSOA: Gravitational Slingshot Optimization Algorithm

**Supplementary code and data for:**  
*"GSOA: Gravitational Slingshot Optimization Algorithm with p-best Guidance and Adaptive Restart"*

---

## Repository Structure

```
GSOA-GitHub/
│
├── src/
│   ├── gsoa/                          # Proposed algorithm (GSOA)
│   │   ├── cec2022/
│   │   │   ├── gsoa_cec2022_d10.cpp  # CEC2022, D=10 (Phase-1 α search + 30 runs)
│   │   │   └── gsoa_cec2022_d20.cpp  # CEC2022, D=20
│   │   ├── mechanical/
│   │   │   ├── conv_recorder.h        # Shared convergence recorder header
│   │   │   ├── B1_pressure_vessel/gsoa_b1.cpp
│   │   │   ├── B2_spring/gsoa_b2.cpp
│   │   │   └── B3_welded_beam/gsoa_b3.cpp
│   │
│   ├── baselines/                     # Classical metaheuristics (GSA, PSO, DE, GA)
│   │   ├── cec2022/gsa_pso_de_ga_cec2022.cpp
│   │   └── mechanical/
│   │       ├── baselines_b1_pressure_vessel.cpp
│   │       ├── baselines_b2_spring.cpp
│   │       └── baselines_b3_welded_beam.cpp
│   │
│   ├── competitors/                   # CEC2022 top-ranked entries (official source)
│   │   ├── EA4eig/                    # Python translation of official MATLAB winner
│   │   │   ├── ea4eig.py
│   │   │   ├── run_batch.py
│   │   │   └── wrapper.cpp            # C++ bridge to CEC2022 evaluator
│   │   ├── S-LSHADE-DP/               # Official C++ submission
│   │   ├── NL-SHADE-LBC/              # Official C++ submission
│   │   └── NL-SHADE-RSP-MID/          # Official C++ submission (requires Armadillo)
│   │
│   ├── cec2022_evaluator/             # Official CEC2022 test function evaluator
│   │   ├── cec22_test_func.cpp        # Unmodified official source
│   │   ├── input_data_d10/            # Shift/rotation data, D=10 (54 files)
│   │   └── input_data_d20/            # Shift/rotation data, D=20 (54 files)
│   │
│   └── problem_evaluators/            # Penalty-function evaluators (mechanical)
│       ├── problem_eval_b1_pressure_vessel.cpp
│       ├── problem_eval_b2_spring.cpp
│       └── problem_eval_b3_welded_beam.cpp
│
├── results/                           # Raw 30-run result data
│   ├── cec2022/
│   │   ├── d10/   # 9 algorithms × 12 functions = 108 files (16 rows × 30 cols each)
│   │   └── d20/   # 108 files
│   └── mechanical/
│       ├── B1/    # 9 algorithms × 1 file each (16 rows × 30 cols)
│       ├── B2/
│       └── B3/
│
└── analysis/
    └── plot_results.py                # Reproduce all figures and tables (see below)
```

---

## Quick Start — Reproduce All Figures

```bash
pip install numpy scipy matplotlib
python3 analysis/plot_results.py
```

Outputs written to `analysis/outputs/`. Reproduces all convergence plots, statistical tables (Friedman, Wilcoxon, Kruskal-Wallis), and Friedman rank charts from the raw 30-run data.

---

## Building and Running the Code

### Requirements
- **C++**: g++ with C++11 or later (`g++ -O2 -std=c++11`)
- **NL-SHADE-RSP-MID only**: Armadillo linear algebra library (`apt install libarmadillo-dev`)
- **EA4eig**: Python 3.8+, NumPy, SciPy

---

### GSOA — CEC2022

```bash
cd src/gsoa/cec2022

# D=10
g++ -O2 -std=c++11 -o gsoa_d10 \
    ../../cec2022_evaluator/cec22_test_func.cpp \
    gsoa_cec2022_d10.cpp

cp -r ../../cec2022_evaluator/input_data_d10 input_data
mkdir -p results
./gsoa_d10          # runs Phase-1 (alpha grid search) + Phase-2 (30 runs)

# D=20
g++ -O2 -std=c++11 -o gsoa_d20 \
    ../../cec2022_evaluator/cec22_test_func.cpp \
    gsoa_cec2022_d20.cpp
cp -r ../../cec2022_evaluator/input_data_d20 input_data
./gsoa_d20
```

Output: `results/GSOA_{funcID}_{D}.txt` — 17 rows × 30 columns (rows 0–15: errors at 16 checkpoints; row 16: FES used).

---

### GSOA — Mechanical Design

```bash
# Example: Pressure Vessel Design (B1)
cd src/gsoa/mechanical/B1_pressure_vessel

g++ -O2 -std=c++11 -o gsoa_b1 \
    ../../../problem_evaluators/problem_eval_b1_pressure_vessel.cpp \
    gsoa_b1.cpp
mkdir -p results
./gsoa_b1
```

Output: `results/GSOA_B1.txt` — 16 rows × 30 columns.

Same pattern for B2 (spring) and B3 (welded beam).

---

### Classical Baselines — CEC2022

```bash
cd src/baselines/cec2022
g++ -O2 -std=c++11 -o baselines \
    ../../cec2022_evaluator/cec22_test_func.cpp \
    gsa_pso_de_ga_cec2022.cpp
cp -r ../../cec2022_evaluator/input_data_d10 input_data
mkdir -p results
./baselines
```

---

### EA4eig (Python)

```bash
cd src/competitors/EA4eig

# Compile the C++ CEC2022 evaluator bridge
g++ -O3 -std=c++11 -shared -fPIC -o libcec22.so \
    ../../cec2022_evaluator/cec22_test_func.cpp \
    wrapper.cpp

# Run all 12 functions, D=10, 30 runs each
python3 run_batch.py 10 1 12
```

---

### S-LSHADE-DP

```bash
cd src/competitors/S-LSHADE-DP
g++ -O2 -std=c++11 -o slshadedp \
    main.cpp s_lshade_dp.cpp search_algorithm.cpp
./slshadedp
```

---

### NL-SHADE-LBC

```bash
cd src/competitors/NL-SHADE-LBC
g++ -O2 -std=c++11 -o nlshadelbc nl_shade_lbc.cpp
./nlshadelbc
```

---

### NL-SHADE-RSP-MID (requires Armadillo)

```bash
apt-get install libarmadillo-dev   # Ubuntu/Debian
cd src/competitors/NL-SHADE-RSP-MID
g++ -std=c++17 -O2 nl_shade_rsp_mid.cpp -fopenmp -larmadillo -o nlshadersp
export OMP_NUM_THREADS=1
./nlshadersp 0 30   # runs 0 to 29
```

---

## Data Format

Every result file is a whitespace-delimited matrix:

| Dimension | Meaning |
|---|---|
| Rows 0–15 | Best-so-far error at 16 log-spaced FES checkpoints: `FES_k = D^(k/5 − 3) × MaxFES` |
| Columns 0–29 | Independent runs (column j used seed = 42 + j) |

---

## Algorithm Parameters

| Parameter | Value |
|---|---|
| Population size N | 30 |
| G₀ | 100 |
| G decay λ | 20 × D_REF / D (D_REF = 10) |
| p-best rate | 0.20 |
| Stagnation threshold T_stag | max(10, T_max / 20) |
| α (slingshot coefficient) | Grid-searched over {−2.0, −1.5, ..., 1.5, 2.0} |
| MaxFES (CEC2022 D=10) | 200,000 |
| MaxFES (CEC2022 D=20) | 1,000,000 |
| MaxFES (mechanical) | 200,000 |
| Runs | 30 |

---

## Portability Notes

The three C++ competition entries (S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID) were modified **only** for build portability:
- Path separators updated
- `main()` wrapped to allow multi-run batching
- No algorithmic or parameter changes

EA4eig is the official CEC2022 MATLAB winner code translated to Python and validated against Octave output on a subset of functions.

---

## Citation

If you use this code or data, please cite:

```
[To be filled upon acceptance]
```

---

## License

Source code for GSOA (the proposed algorithm) is released under the MIT License.  
The competitor algorithm source files retain their original licenses from the respective CEC2022 competition submissions.
