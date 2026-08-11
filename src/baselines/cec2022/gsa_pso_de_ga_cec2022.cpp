// GSA / PSO / DE / GA on official CEC2022 suite, D=10, single run.
// Mechanics translated directly from the user's Python GSOA-benchmark script
// (run_gsa, run_pso, run_de, run_ga) - same formulas, ported to C++ against
// the official cec22_test_func evaluator, with FES-based budget (200000)
// matching the winner codes, and the same 17-point convergence checkpoint
// format used throughout this comparison.

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
#include <numeric>
#include <chrono>

using namespace std;

double *OShift, *M, *y, *z, *x_bound;
int ini_flag = 0, n_flag, func_flag, *SS;
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);

static const int D = 10;
static const int N = 30;
static const double LB = -100.0, UB = 100.0;
static const double EPS = 1e-12;
static const unsigned int MAXFES = 200000;
static const int T_MAX = MAXFES / N;   // 1 pop-eval per iteration for these baselines
static const int T_MAX_REF = MAXFES / (2 * N);   // GSA does 2 evals/iter (v6) - decay pacing
static const double FISTAR[12] = {300,400,600,800,900,1800,2000,2200,2300,2400,2600,2700};

mt19937 rng(42);
uniform_real_distribution<double> U01(0.0, 1.0);
double randu(double a, double b) { return a + (b - a) * U01(rng); }

void evalPop(const vector<double> &X, vector<double> &f, int func_num, int npts) {
    vector<double> xin(npts * D);
    for (int i = 0; i < npts * D; i++) xin[i] = X[i];
    f.assign(npts, 0.0);
    cec22_test_func(xin.data(), f.data(), D, npts, func_num);
}

struct ConvRecorder {
    vector<unsigned int> checkpoints;
    int cp_idx = 0;
    double conv[17];
    int func_num;
    ConvRecorder(int fn) : func_num(fn) {
        checkpoints.resize(16);
        for (int i = 0; i < 16; i++)
            checkpoints[i] = (unsigned int)round(pow((double)D, (i / 5.0) - 3.0) * MAXFES);
        for (int k = 0; k < 17; k++) conv[k] = 0.0;
    }
    void update(unsigned int FES, double bsf) {
        while (cp_idx < 16 && FES >= checkpoints[cp_idx]) {
            conv[cp_idx] = bsf - FISTAR[func_num - 1];
            if (conv[cp_idx] < 0) conv[cp_idx] = 0;
            cp_idx++;
        }
    }
    void finish(unsigned int FES, double bsf) {
        while (cp_idx < 16) {
            conv[cp_idx] = bsf - FISTAR[func_num - 1];
            if (conv[cp_idx] < 0) conv[cp_idx] = 0;
            cp_idx++;
        }
        conv[16] = (double)FES;
    }
};

// ---------------- GSA (v6: shares Fix-A G_DECAY with GSOA, 2 evals/iter) ----------------
void run_gsa(int func_num, double *outConv, unsigned seed) {
    rng.seed(seed);
    vector<double> X(N * D), V(N * D, 0.0), f(N);
    for (int i = 0; i < N * D; i++) X[i] = randu(LB, UB);

    ConvRecorder rec(func_num);
    unsigned int FES = 0;
    double bsf = 1e300;
    static const double D_REF = 10.0;
    const double G_DECAY = 20.0 * D_REF / D;   // Fix A, shared with GSOA (v6 gsa_step)
    int t = 0;

    while (FES < MAXFES) {
        // --- gsa_step: eval f = fn(X), track gbest ---
        evalPop(X, f, func_num, N);
        FES += N;
        for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
        rec.update(FES, bsf);
        if (FES >= MAXFES) break;

        double fb = f[0], fw = f[0];
        for (int i = 1; i < N; i++) { if (f[i] < fb) fb = f[i]; if (f[i] > fw) fw = f[i]; }
        vector<double> m(N), Mm(N); double msum = 0;
        for (int i = 0; i < N; i++) { m[i] = (fw - f[i]) / (fw - fb + EPS); msum += m[i]; }
        for (int i = 0; i < N; i++) Mm[i] = m[i] / (msum + EPS);
        double G = 100.0 * exp(-G_DECAY * t / (double)T_MAX_REF);

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
            X[i] = v;
        }
        // --- post-move eval f2 = fn(X), track gbest (v6: 2nd eval per iter) ---
        evalPop(X, f, func_num, N);
        FES += N;
        for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
        rec.update(FES, bsf);
        t++;
    }
    rec.finish(FES, bsf);
    memcpy(outConv, rec.conv, sizeof(rec.conv));
}

// ---------------- PSO (w=0.729, c1=c2=1.494) ----------------
void run_pso(int func_num, double *outConv, unsigned seed) {
    rng.seed(seed);
    const double w = 0.729, c1 = 1.494, c2 = 1.494;
    vector<double> X(N*D), V(N*D), f(N), pb(N*D), fpb(N);
    for (int i = 0; i < N*D; i++) { X[i] = randu(LB, UB); V[i] = randu(-(UB-LB), UB-LB); }
    evalPop(X, f, func_num, N);
    unsigned int FES = N;
    pb = X; fpb = f;
    int gbi = min_element(f.begin(), f.end()) - f.begin();
    vector<double> gb(D); for (int d = 0; d < D; d++) gb[d] = X[gbi*D+d];
    double fgb = f[gbi];

    ConvRecorder rec(func_num);
    double bsf = fgb;
    rec.update(FES, bsf);

    for (int t = 0; t < T_MAX; t++) {
        if (FES >= MAXFES) break;
        for (int i = 0; i < N; i++)
            for (int d = 0; d < D; d++) {
                double r1 = U01(rng), r2 = U01(rng);
                V[i*D+d] = w*V[i*D+d] + c1*r1*(pb[i*D+d]-X[i*D+d]) + c2*r2*(gb[d]-X[i*D+d]);
                double v = X[i*D+d] + V[i*D+d];
                if (v < LB) v = LB; if (v > UB) v = UB;
                X[i*D+d] = v;
            }
        evalPop(X, f, func_num, N);
        FES += N;
        for (int i = 0; i < N; i++) {
            if (f[i] < fpb[i]) { fpb[i] = f[i]; for (int d=0; d<D; d++) pb[i*D+d]=X[i*D+d]; }
        }
        int bi = min_element(fpb.begin(), fpb.end()) - fpb.begin();
        if (fpb[bi] < fgb) { fgb = fpb[bi]; for (int d=0; d<D; d++) gb[d]=pb[bi*D+d]; }
        if (fgb < bsf) bsf = fgb;
        rec.update(FES, bsf);
    }
    rec.finish(FES, bsf);
    memcpy(outConv, rec.conv, sizeof(rec.conv));
}

// ---------------- DE (rand/1/bin, F=0.8, CR=0.9) ----------------
void run_de(int func_num, double *outConv, unsigned seed) {
    rng.seed(seed);
    vector<double> X(N*D), fX(N);
    for (int i = 0; i < N*D; i++) X[i] = randu(LB, UB);
    evalPop(X, fX, func_num, N);
    unsigned int FES = N;

    ConvRecorder rec(func_num);
    double bsf = *min_element(fX.begin(), fX.end());
    rec.update(FES, bsf);

    uniform_int_distribution<int> idxDist(0, N-1);
    uniform_int_distribution<int> dimDist(0, D-1);

    for (int t = 0; t < T_MAX; t++) {
        if (FES >= MAXFES) break;
        for (int i = 0; i < N; i++) {
            int a, b, c;
            do { a = idxDist(rng); } while (a == i);
            do { b = idxDist(rng); } while (b == i || b == a);
            do { c = idxDist(rng); } while (c == i || c == a || c == b);
            vector<double> trial(D);
            vector<bool> cross(D);
            bool any = false;
            for (int d = 0; d < D; d++) { cross[d] = U01(rng) < 0.9; if (cross[d]) any = true; }
            if (!any) cross[dimDist(rng)] = true;
            for (int d = 0; d < D; d++) {
                double mut = X[a*D+d] + 0.8*(X[b*D+d]-X[c*D+d]);
                if (mut < LB) mut = LB; if (mut > UB) mut = UB;
                trial[d] = cross[d] ? mut : X[i*D+d];
            }
            double ft;
            vector<double> ftv;
            evalPop(trial, ftv, func_num, 1);
            ft = ftv[0];
            FES++;
            if (ft <= fX[i]) { for (int d=0; d<D; d++) X[i*D+d]=trial[d]; fX[i]=ft; }
            if (fX[i] < bsf) bsf = fX[i];
            rec.update(FES, bsf);
            if (FES >= MAXFES) break;
        }
    }
    rec.finish(FES, bsf);
    memcpy(outConv, rec.conv, sizeof(rec.conv));
}

// ---------------- GA (SBX crossover + polynomial mutation, elitism) ----------------
void run_ga(int func_num, double *outConv, unsigned seed) {
    rng.seed(seed);
    const double pc = 0.9, eta_c = 20.0, eta_m = 20.0, pm = 1.0/D;
    vector<double> X(N*D), fX(N);
    for (int i = 0; i < N*D; i++) X[i] = randu(LB, UB);
    evalPop(X, fX, func_num, N);
    unsigned int FES = N;

    ConvRecorder rec(func_num);
    double bsf = *min_element(fX.begin(), fX.end());
    rec.update(FES, bsf);

    uniform_int_distribution<int> idxDist(0, N-1);

    for (int t = 0; t < T_MAX; t++) {
        if (FES >= MAXFES) break;
        vector<double> nX(N*D);
        vector<int> sel(N);
        for (int i = 0; i < N; i++) {
            int i1 = idxDist(rng), i2 = idxDist(rng);
            sel[i] = (fX[i1] < fX[i2]) ? i1 : i2;
        }
        for (int i = 0; i < N; i++) for (int d = 0; d < D; d++) nX[i*D+d] = X[sel[i]*D+d];

        for (int i = 0; i + 1 < N; i += 2) {
            if (U01(rng) < pc) {
                for (int d = 0; d < D; d++) {
                    double u = U01(rng);
                    double bc = (u <= 0.5) ? pow(2*u, 1.0/(eta_c+1)) : pow(1.0/(2*(1-u+EPS)), 1.0/(eta_c+1));
                    double p1 = nX[i*D+d], p2 = nX[(i+1)*D+d];
                    double c1v = 0.5*((1+bc)*p1 + (1-bc)*p2);
                    double c2v = 0.5*((1-bc)*p1 + (1+bc)*p2);
                    if (c1v < LB) c1v = LB; if (c1v > UB) c1v = UB;
                    if (c2v < LB) c2v = LB; if (c2v > UB) c2v = UB;
                    nX[i*D+d] = c1v; nX[(i+1)*D+d] = c2v;
                }
            }
        }
        for (int i = 0; i < N; i++)
            for (int d = 0; d < D; d++) {
                if (U01(rng) < pm) {
                    double xv = nX[i*D+d];
                    double u = U01(rng);
                    double delta = min(xv - LB, UB - xv) / (UB - LB + EPS);
                    double dq;
                    if (u < 0.5)
                        dq = pow(2*u + (1-2*u)*pow(1-delta, eta_m+1), 1.0/(eta_m+1)) - 1;
                    else
                        dq = 1 - pow(2*(1-u) + 2*(u-0.5)*pow(1-delta, eta_m+1), 1.0/(eta_m+1));
                    double v = xv + dq * (UB - LB);
                    if (v < LB) v = LB; if (v > UB) v = UB;
                    nX[i*D+d] = v;
                }
            }

        vector<double> fnX;
        evalPop(nX, fnX, func_num, N);
        FES += N;

        int bi = min_element(fX.begin(), fX.end()) - fX.begin();
        int wn = max_element(fnX.begin(), fnX.end()) - fnX.begin();
        if (fX[bi] < fnX[wn]) {
            for (int d = 0; d < D; d++) nX[wn*D+d] = X[bi*D+d];
            fnX[wn] = fX[bi];
        }
        X = nX; fX = fnX;
        double mn = *min_element(fX.begin(), fX.end());
        if (mn < bsf) bsf = mn;
        rec.update(FES, bsf);
    }
    rec.finish(FES, bsf);
    memcpy(outConv, rec.conv, sizeof(rec.conv));
}

int main() {
    const int N_RUNS = 30;
    ofstream timeLog("results/baselines_time.txt");

    struct AlgEntry { const char* name; void(*fn)(int,double*,unsigned); };
    AlgEntry algs[4] = {
        {"GSA", run_gsa}, {"PSO", run_pso}, {"DE", run_de}, {"GA", run_ga}
    };

    for (auto &alg : algs) {
        double totalTime = 0;
        vector<vector<vector<double>>> allData(12, vector<vector<double>>(N_RUNS, vector<double>(17)));
        for (int func_num = 1; func_num <= 12; func_num++) {
            auto t0 = chrono::high_resolution_clock::now();
            for (int r = 0; r < N_RUNS; r++) {
                alg.fn(func_num, allData[func_num-1][r].data(), 42 + r);
            }
            auto t1 = chrono::high_resolution_clock::now();
            double elapsed = chrono::duration<double>(t1 - t0).count();
            totalTime += elapsed;

            stringstream fn; fn << "results/" << alg.name << "_" << func_num << "_" << D << ".txt";
            ofstream fo(fn.str());
            for (int row = 0; row < 17; row++) {
                for (int r = 0; r < N_RUNS; r++) fo << allData[func_num-1][r][row] << "\t";
                fo << "\n";
            }
            cout << alg.name << " F" << func_num << " done [" << elapsed << "s]" << endl;
        }
        timeLog << alg.name << "\t" << totalTime << "\t" << (totalTime/(12*N_RUNS)) << "\n";
    }
    timeLog.close();
    return 0;
}
