// GSOA on Appendix B.1 - Pressure Vessel Design.
// Combines EGSOA's elite-target slingshot (pull toward random top-20%
// individual instead of population mean) WITH GSOA-R's diversity-collapse
// restart mechanism (stagnation-triggered reinitialization, elitist).

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
#include "conv_recorder.h"

using namespace std;

void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);

static const int D = 4;
static const int N = 30;
static const double LB = -100.0, UB = 100.0;
static const double EPS = 1e-12;
static const double D_REF = 10.0;
static const double G_DECAY = 20.0 * D_REF / D;
static const unsigned int MAXFES = 200000;
static const int T_MAX = MAXFES / (2 * N);
static const double P_BEST_RATE = 0.2;
static const int STAG_LIMIT = max(10, T_MAX / 20);
static const double RELATIVE_DIVERSITY_THRESHOLD = 0.05;

mt19937 rng(42);
uniform_real_distribution<double> U01(0.0, 1.0);
double randu(double a, double b) { return a + (b - a) * U01(rng); }

void evalPop(const vector<double> &X, vector<double> &f) {
    vector<double> xin(N * D);
    for (int i = 0; i < N; i++)
        for (int j = 0; j < D; j++)
            xin[i * D + j] = X[i * D + j];
    f.assign(N, 0.0);
    cec22_test_func(xin.data(), f.data(), D, N, 1);
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

void run_gsoa(double alpha, ConvRecorderFES &rec, unsigned seed, int &restarts_used) {
    rng.seed(seed);
    vector<double> X(N * D), V(N * D, 0.0), Xc(N * D), f(N), fc(N);
    for (int i = 0; i < N * D; i++) X[i] = randu(LB, UB);

    unsigned int FES = 0;
    double bsf = 1e300;
    int stagCount = 0;
    restarts_used = 0;

    evalPop(X, f);
    FES += N;
    for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
    rec.update(FES, bsf);
    double initDiversity = meanPairwiseDist(X);

    int pbest_n = max(2, (int)round(P_BEST_RATE * N));
    uniform_int_distribution<int> pbestDist(0, pbest_n - 1);

    for (int t = 0; t < T_MAX; t++) {
        if (FES >= MAXFES) break;

        if (stagCount >= STAG_LIMIT) {
            int bi = min_element(f.begin(), f.end()) - f.begin();
            vector<double> bestX(D);
            for (int d = 0; d < D; d++) bestX[d] = X[bi * D + d];
            for (int i = 0; i < N; i++) {
                if (i == bi) continue;
                for (int d = 0; d < D; d++) { X[i*D+d] = randu(LB, UB); V[i*D+d] = 0.0; }
            }
            for (int d = 0; d < D; d++) X[bi*D+d] = bestX[d];
            V.assign(N*D, 0.0);
            evalPop(X, f);
            FES += N;
            for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
            rec.update(FES, bsf);
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
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                if (i == j) continue;
                double dist2 = 0;
                for (int d = 0; d < D; d++) { double diff = X[i*D+d]-X[j*D+d]; dist2 += diff*diff; }
                double dist = sqrt(dist2) + EPS;
                double w = G * Mm[j] / dist;
                for (int d = 0; d < D; d++) Fforce[i*D+d] += w * Mm[i] * (X[i*D+d]-X[j*D+d]);
            }
        }
        for (int i = 0; i < N; i++)
            for (int d = 0; d < D; d++) {
                double A = Fforce[i*D+d] / (Mm[i] + EPS);
                V[i*D+d] = U01(rng) * V[i*D+d] + A;
            }

        for (int i = 0; i < N * D; i++) {
            double v = X[i] + V[i];
            if (v < LB) v = LB; if (v > UB) v = UB;
            Xc[i] = v;
        }
        evalPop(Xc, fc);
        FES += N;

        double bsf_before = bsf;

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
                        double target = X[pbest_idx * D + d];
                        double v = X[i*D+d] + ri * sgn * mag * (target - X[i*D+d]);
                        if (v < LB) v = LB; if (v > UB) v = UB;
                        X[i*D+d] = v;
                        V[i*D+d] = 0.0;
                    }
                } else {
                    for (int d = 0; d < D; d++) X[i*D+d] = Xc[i*D+d];
                }
            }
            evalPop(X, f);
            FES += N;
        } else {
            for (int i = 0; i < N * D; i++) X[i] = Xc[i];
            f = fc;
        }

        for (int i = 0; i < N; i++) if (f[i] < bsf) bsf = f[i];
        rec.update(FES, bsf);

        if (bsf < bsf_before - 1e-12) stagCount = 0;
        else stagCount++;
    }
    rec.finish(FES, bsf);
}

int main() {
    double alphaGrid[9] = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0};
    double bestFinal = 1e300, bestAlpha = 0.0;
    ConvRecorderFES bestRec;
    int bestRestarts = 0;

    for (int ai = 0; ai < 9; ai++) {
        ConvRecorderFES rec;
        int restarts = 0;
        run_gsoa(alphaGrid[ai], rec, 42, restarts);
        double finalVal = rec.values.back();
        cout << "alpha=" << alphaGrid[ai] << " final=" << finalVal << " restarts=" << restarts << endl;
        if (finalVal < bestFinal) {
            bestFinal = finalVal;
            bestAlpha = alphaGrid[ai];
            bestRec = rec;
            bestRestarts = restarts;
        }
    }
    cout << "BEST alpha=" << bestAlpha << " final=" << bestFinal << " restarts=" << bestRestarts << endl;

    ofstream fout("results/GSOA_B3.txt");
    for (size_t i = 0; i < bestRec.fesPoints.size(); i++)
        fout << bestRec.fesPoints[i] << "\t" << bestRec.values[i] << "\n";
    fout.close();
    ofstream meta("results/GSOA_B3_meta.txt");
    meta << "best_alpha\t" << bestAlpha << "\nfinal\t" << bestFinal << "\nrestarts\t" << bestRestarts << "\n";
    return 0;
}
