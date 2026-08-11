// Appendix B.2 - Design of Tension/Compression Spring
// x1=d (wire diameter), x2=D (mean coil diameter), x3=N (active coils)
// Minimize: f(x) = (N+2) D d^2
// Subject to:
//   g1 = 1 - D^3 N / (71785 d^4) <= 0
//   g2 = (4D^2 - dD)/(12566(Dd^3 - d^4)) + 1/(5108 d^2) - 1 <= 0
//   g3 = 1 - 140.45 d / (D^2 N) <= 0
//   g4 = (D+d)/1.5 - 1 <= 0
// Bounds (true): 0.05<=d<=2, 0.25<=D<=1.3, 2<=N<=15

#include <cmath>

static const double PENALTY = 1.0e6;
static const int D_DIM = 3;
static const double TRUE_LB[3] = {0.05, 0.25, 2.0};
static const double TRUE_UB[3] = {2.0, 1.3, 15.0};

static inline double rescale(double xn, int dim) {
    double t = (xn + 100.0) / 200.0;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return TRUE_LB[dim] + t * (TRUE_UB[dim] - TRUE_LB[dim]);
}

static double eval_one(const double *xn) {
    double x[3];
    for (int i = 0; i < 3; i++) x[i] = rescale(xn[i], i);
    double d = x[0], Dc = x[1], N = x[2];

    double f = (N + 2.0) * Dc * d * d;

    double g1 = 1.0 - (Dc*Dc*Dc*N) / (71785.0 * d*d*d*d);
    double g2 = (4.0*Dc*Dc - d*Dc) / (12566.0 * (Dc*d*d*d - d*d*d*d)) + 1.0/(5108.0*d*d) - 1.0;
    double g3 = 1.0 - (140.45*d) / (Dc*Dc*N);
    double g4 = (Dc + d) / 1.5 - 1.0;

    double viol = 0.0;
    if (g1 > 0) viol += g1;
    if (g2 > 0) viol += g2;
    if (g3 > 0) viol += g3;
    if (g4 > 0) viol += g4;

    return f + PENALTY * viol;
}

void cec22_test_func(double *x, double *f, int nx, int mx, int func_num) {
    for (int i = 0; i < mx; i++) {
        f[i] = eval_one(&x[i * nx]);
    }
}
