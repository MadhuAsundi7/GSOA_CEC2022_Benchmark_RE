#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace std;

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);

static const int D = 10;
static const int N = 30;
static const double LB = -100.0, UB = 100.0;
static const double EPS = 1e-12;
static const double D_REF = 10.0;
static const double G_DECAY = 20.0 * D_REF / D;
static const unsigned int MAXFES = 200000;
static const int T_MAX = MAXFES / (2 * N);
static const double FISTAR[12] = {300,400,600,800,900,1800,2000,2200,2300,2400,2600,2700};
static const double P_BEST_RATE = 0.2;
static const int STAG_LIMIT = max(10, T_MAX / 20);
static const double RELATIVE_DIVERSITY_THRESHOLD = 0.05;
static const double alphaGrid[9] = {-2.0,-1.5,-1.0,-0.5,0.0,0.5,1.0,1.5,2.0};

mt19937 rng(42);
uniform_real_distribution<double> U01(0.0, 1.0);
double randu(double a, double b) { return a + (b - a) * U01(rng); }

void evalPop(const vector<double> &X, vector<double> &f, int func_num) {
    vector<double> xin(N * D);
    for (int i = 0; i < N * D; i++) xin[i] = X[i];
    f.assign(N, 0.0);
    cec22_test_func(xin.data(), f.data(), D, N, func_num);
}

double meanPairwiseDist(const vector<double> &X) {
    double sum = 0.0; int cnt = 0;
    for (int i = 0; i < N; i++)
        for (int j = i + 1; j < N; j++) {
            double d2 = 0;
            for (int d = 0; d < D; d++) { double diff = X[i*D+d]-X[j*D+d]; d2 += diff*diff; }
            sum += sqrt(d2); cnt++;
        }
    return sum / cnt;
}

void run_gsoa(int func_num, double alpha, double *convergence, unsigned seed, int &restarts_used) {
    rng.seed(seed);
    vector<double> X(N * D), V(N * D, 0.0), Xc(N * D), f(N), fc(N);
    for (int i = 0; i < N * D; i++) X[i] = randu(LB, UB);

    unsigned int FES = 0;
    double bsf = 1e300;
    int stagCount = 0;
    restarts_used = 0;
    vector<unsigned int> checkpoints(16);
    for (int i = 0; i < 16; i++)
        checkpoints[i] = (unsigned int)round(pow((double)D, (i / 5.0) - 3.0) * MAXFES);
    int cp_idx = 0;
    for (int k = 0; k < 17; k++) convergence[k] = 0.0;
    auto recordIfDue = [&]() {
        while (cp_idx < 16 && FES >= checkpoints[cp_idx]) {
            convergence[cp_idx] = bsf - FISTAR[func_num - 1];
            if (convergence[cp_idx] < 0) convergence[cp_idx] = 0;
            cp_idx++;
        }
    };

    evalPop(X, f, func_num);
    FES += N;
    for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
    recordIfDue();
    double initDiversity = meanPairwiseDist(X);

    int pbest_n = max(2, (int)round(P_BEST_RATE * N));
    uniform_int_distribution<int> pbestDist(0, pbest_n - 1);

    for (int t = 0; t < T_MAX; t++) {
        if (FES >= MAXFES) break;

        if (stagCount >= STAG_LIMIT) {
            int bi = min_element(f.begin(), f.end()) - f.begin();
            vector<double> bestX(D);
            for (int d = 0; d < D; d++) bestX[d] = X[bi*D+d];
            for (int i = 0; i < N; i++) {
                if (i == bi) continue;
                for (int d = 0; d < D; d++) { X[i*D+d] = randu(LB, UB); V[i*D+d] = 0.0; }
            }
            for (int d = 0; d < D; d++) X[bi*D+d] = bestX[d];
            V.assign(N*D, 0.0);
            evalPop(X, f, func_num);
            FES += N;
            for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
            recordIfDue();
            stagCount = 0;
            restarts_used++;
            if (FES >= MAXFES) break;
        }

        double fb = f[0], fw = f[0];
        for (int i = 1; i < N; i++) { if (f[i] < fb) fb = f[i]; if (f[i] > fw) fw = f[i]; }
        vector<double> m(N), Mm(N);
        double msum = 0;
        for (int i = 0; i < N; i++) { m[i] = (fw - f[i]) / (fw - fb + EPS); msum += m[i]; }
        for (int i = 0; i < N; i++) Mm[i] = m[i] / (msum + EPS);
        double G = 100.0 * exp(-G_DECAY * t / (double)T_MAX);
        vector<double> Fforce(N * D, 0.0);
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double dist2 = 0;
                for (int d = 0; d < D; d++) { double diff = X[i*D+d]-X[j*D+d]; dist2 += diff*diff; }
                double dist = sqrt(dist2) + EPS;
                double w = G * Mm[j] / dist;
                for (int d = 0; d < D; d++) Fforce[i*D+d] += w * Mm[i] * (X[i*D+d]-X[j*D+d]);
            }
        for (int i = 0; i < N; i++)
            for (int d = 0; d < D; d++) {
                double A = Fforce[i*D+d] / (Mm[i] + EPS);
                V[i*D+d] = U01(rng) * V[i*D+d] + A;
            }
        for (int i = 0; i < N*D; i++) {
            double v = X[i] + V[i];
            if (v < LB) v = LB; if (v > UB) v = UB;
            Xc[i] = v;
        }
        evalPop(Xc, fc, func_num);
        FES += N;

        if (alpha != 0.0) {
            vector<int> order(N);
            for (int i = 0; i < N; i++) order[i] = i;
            sort(order.begin(), order.end(), [&](int a, int b){ return f[a] < f[b]; });
            double sgn = (alpha > 0) ? 1.0 : -1.0;
            double mag = fabs(alpha);
            for (int i = 0; i < N; i++) {
                if (fc[i] >= f[i]) {
                    int pbest_idx = order[pbestDist(rng)];
                    for (int d = 0; d < D; d++) {
                        double ri = U01(rng);
                        double target = X[pbest_idx*D+d];
                        double v = X[i*D+d] + ri*sgn*mag*(target-X[i*D+d]);
                        if (v < LB) v = LB; if (v > UB) v = UB;
                        X[i*D+d] = v;
                        V[i*D+d] = 0.0;
                    }
                } else {
                    for (int d = 0; d < D; d++) X[i*D+d] = Xc[i*D+d];
                }
            }
            evalPop(X, f, func_num);
            FES += N;
        } else {
            for (int i = 0; i < N*D; i++) X[i] = Xc[i];
            f = fc;
        }

        double bsf_before = bsf;
        for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
        recordIfDue();
        if (bsf < bsf_before - 1e-12) stagCount = 0;
        else stagCount++;
    }
    while (cp_idx < 16) {
        convergence[cp_idx] = bsf - FISTAR[func_num - 1];
        if (convergence[cp_idx] < 0) convergence[cp_idx] = 0;
        cp_idx++;
    }
    convergence[16] = (double)FES;
}

int main(int argc, char** argv) {
    int start_f = 1, end_f = 12;
    if (argc > 2) { start_f = atoi(argv[1]); end_f = atoi(argv[2]); }
    const int N_RUNS = 30;

    ofstream timeLog("results/GSOA_time.txt", ios::app);
    for (int func_num = start_f; func_num <= end_f; func_num++) {
        auto t0 = chrono::high_resolution_clock::now();

        // Phase 1: alpha grid search (single run each)
        double bestAlpha = 0.0, bestFinal = 1e300;
        for (int ai = 0; ai < 9; ai++) {
            vector<double> conv(17);
            int restarts = 0;
            run_gsoa(func_num, alphaGrid[ai], conv.data(), 42, restarts);
            if (conv[15] < bestFinal) { bestFinal = conv[15]; bestAlpha = alphaGrid[ai]; }
        }

        // Phase 2: 30 independent runs at fixed best alpha
        vector<vector<double>> allConv(N_RUNS, vector<double>(17));
        int totalRestarts = 0;
        for (int r = 0; r < N_RUNS; r++) {
            int restarts = 0;
            run_gsoa(func_num, bestAlpha, allConv[r].data(), 42 + r, restarts);
            totalRestarts += restarts;
        }

        auto t1 = chrono::high_resolution_clock::now();
        double elapsed = chrono::duration<double>(t1 - t0).count();
        timeLog << func_num << "\t" << elapsed << "\t" << bestAlpha << "\t" << totalRestarts << "\n";
        timeLog.flush();

        stringstream fname;
        fname << "results/GSOA_" << func_num << "_" << D << ".txt";
        ofstream fout(fname.str());
        for (int row = 0; row < 17; row++) {
            for (int r = 0; r < N_RUNS; r++) fout << allConv[r][row] << "\t";
            fout << "\n";
        }
        cout << "F" << func_num << " done, alpha=" << bestAlpha << ", mean_final="
             << [&](){ double s=0; for(int r=0;r<N_RUNS;r++) s+=allConv[r][15]; return s/N_RUNS; }()
             << ", time=" << elapsed << "s" << endl;
    }
    timeLog.close();
    return 0;
}
