// Shared convergence recorder: same 16 log-spaced FES-checkpoint format
// (FES_k = D^(k/5-3) x MaxFES) used throughout this study for CEC2022,
// so results here are directly comparable in format. No "known optimum"
// exists for these penalized objectives, so the raw best-so-far fitness
// is recorded directly (no subtraction) - lower is better throughout.
#pragma once
#include <vector>
#include <cmath>

struct ConvRecorderFES {
    static const unsigned int MAXFES = 200000;
    static const int D = 4;
    std::vector<unsigned int> fesPoints;
    std::vector<double> values;
    int cp_idx = 0;

    ConvRecorderFES() {
        fesPoints.resize(16);
        for (int i = 0; i < 16; i++)
            fesPoints[i] = (unsigned int)round(pow((double)D, (i / 5.0) - 3.0) * MAXFES);
    }
    void update(unsigned int FES, double bsf) {
        while (cp_idx < 16 && FES >= fesPoints[cp_idx]) {
            values.push_back(bsf);
            cp_idx++;
        }
    }
    void finish(unsigned int FES, double bsf) {
        while (cp_idx < 16) {
            values.push_back(bsf);
            cp_idx++;
        }
    }
};

