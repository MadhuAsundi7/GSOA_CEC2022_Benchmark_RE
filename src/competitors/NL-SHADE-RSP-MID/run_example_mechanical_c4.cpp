// Driver for NL-SHADE-RSP-MID on Appendix B.1 - Pressure Vessel Design.
// Only main() is replaced; the Optimizer class and all restart/clustering
// logic are completely untouched (K_MEANS_AS_NEAREST restart mechanism
// active exactly as in the CEC2022 version - see #define at top of file).
#include <string>
#define NLSHADE_RSP_NO_MAIN
#include "nl_shade_rsp_mid.cpp"

int main(int argc, char** argv) {
    int start_run = 0, end_run = 30;
    if (argc > 2) { start_run = atoi(argv[1]); end_run = atoi(argv[2]); }

    float f;
    FILE* rs;
    rs = fopen("input_data/Rand_Seeds.txt", "r+");
    int Seeds[1000];
    for (int i = 0; i != 1000; i++) {
        fscanf(rs, "%f", &f);
        Seeds[i] = int(f);
    }

    maxFES = 200000;
    fopt = 0;

    for (int runNum = start_run; runNum < end_run; runNum++) {
        int POP_SIZE = 4 * 5;
        ResultsArray[0][runNum][16] = maxFES;
        seed1 = Seeds[runNum];
        generator.seed(seed1);

        globalBestFitinit = false;
        initfinished = false;
        LastFEcount = 0;
        NFEval = 0;
        Optimizer OptZ;
        OptZ.Initialize(POP_SIZE, 4, 1, runNum, 20 * 4, 2.1);
        isRestart = false;
        do {
            OptZ.MainCycle(fopt);
            POP_SIZE = 400;
            OptZ.restart(POP_SIZE, 4, 1, runNum, 20 * 4, 2.1);
            isRestart = true;
        } while (globalBestFit - fopt > MIN_ERROR && NFEval < maxFES);
        OptZ.Clean();
        cout << "run " << runNum << " final=" << globalBestFit << endl;
    }

    ofstream fout("results/NL-SHADE-RSP-MID_C4_r" + to_string(start_run) + "-" + to_string(end_run) + ".txt");
    unsigned int fesPoints[16];
    for (int i = 0; i < 16; i++)
        fesPoints[i] = (unsigned int)round(pow((double)4, (i / 5.0) - 3.0) * maxFES);
    for (int step = 0; step != 16; step++) {
        for (int r = start_run; r < end_run; r++)
            fout << ResultsArray[0][r][step] << "\t";
        fout << "\n";
    }
    fout.close();
    return 0;
}
