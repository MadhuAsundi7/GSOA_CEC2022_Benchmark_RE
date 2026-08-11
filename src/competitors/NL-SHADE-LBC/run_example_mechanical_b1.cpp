// Driver for NL-SHADE-LBC on Appendix B.1 - Pressure Vessel Design.
// This file replaces only main() from the official source; the Optimizer
// class (Initialize/MainCycle/Clean and all its internal logic) is
// completely untouched - included directly from the original file with
// its own main() excluded via NLSHADE_LBC_NO_MAIN.
#define NLSHADE_LBC_NO_MAIN
#include "nl_shade_lbc.cpp"

int main() {
    float f;
    FILE* rs;
    rs = fopen("input_data/Rand_Seeds.txt", "r+");
    int Seeds[1000];
    for (int i = 0; i != 1000; i++) {
        fscanf(rs, "%f", &f);
        Seeds[i] = int(f);
    }

    GNVars = 4;
    MaxFEval = 200000;
    fopt = 0;

    const int N_RUNS = 30;
    for (int runNum = 0; runNum < N_RUNS; runNum++) {
        ResultsArray[0][runNum][16] = MaxFEval;
        seed1 = Seeds[runNum];
        std::mt19937 generator_uni_i(seed1);
        std::mt19937 generator_uni_r(seed1 + 100);
        std::mt19937 generator_norm(seed1 + 200);
        std::mt19937 generator_cachy(seed1 + 300);
        std::mt19937 generator_uni_i_2(seed1 + 400);

        globalbestinit = false;
        initfinished = false;
        LastFEcount = 0;
        NFEval = 0;
        Optimizer OptZ;
        OptZ.Initialize(GNVars * 23, GNVars, 1, runNum, 20 * GNVars, 1);
        OptZ.MainCycle();
        OptZ.Clean();
    }
    double mean30 = 0;
    for (int r = 0; r < N_RUNS; r++) mean30 += ResultsArray[0][r][15];
    mean30 /= N_RUNS;
    cout << "NL-SHADE-LBC 30-run mean = " << mean30 << endl;

    ofstream fout("results/NL-SHADE-LBC_B1.txt");
    for (int step = 0; step != 16; step++) {
        for (int r = 0; r < N_RUNS; r++) fout << ResultsArray[0][r][step] << "\t";
        fout << "\n";
    }
    fout.close();
    return 0;
}
