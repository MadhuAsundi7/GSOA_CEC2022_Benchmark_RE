// Appendix B.1 - Design of Pressure Vessel
// Drop-in replacement for cec22_test_func(x, f, nx, mx, func_num), used
// unmodified by every algorithm's core logic (GSOA, GSA, PSO, DE, GA,
// S-LSHADE-DP, NL-SHADE-LBC, NL-SHADE-RSP-MID all call this exact signature).
//
// Every algorithm in this study searches natively in [-100,100]^D. Rather
// than touch any algorithm's internal bound-handling code (all hardcode a
// uniform scalar [-100,100] region), this evaluator affinely rescales each
// dimension from [-100,100] to the problem's true per-variable bounds, then
// computes the statically-penalized objective in true coordinates. This
// keeps every algorithm's own search/mutation/selection code byte-identical
// to its CEC2022 version - only the fitness landscape it sees changes.
//
// Minimize:
//   f(x) = 0.6224 x1 x3 x4 + 1.7781 x2 x3^2 + 3.1661 x1^2 x4 + 19.84 x1^2 x3
// Subject to:
//   g1 = -x1 + 0.0193 x3 <= 0
//   g2 = -x2 + 0.00954 x3 <= 0
//   g3 = -pi x3^2 x4 - (4/3) pi x3^3 + 1296000 <= 0
//   g4 = x4 - 240 <= 0
// Bounds (true): 0<=x1<=99, 0<=x2<=99, 10<=x3<=200, 10<=x4<=200

#include <cmath>

static const double PENALTY = 1.0e6;
static const int D = 4;
static const double TRUE_LB[4] = {0.0, 0.0, 10.0, 10.0};
static const double TRUE_UB[4] = {99.0, 99.0, 200.0, 200.0};

static inline double rescale(double xn, int dim) {
    // xn in [-100,100] -> true bound range for this dimension
    double t = (xn + 100.0) / 200.0;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return TRUE_LB[dim] + t * (TRUE_UB[dim] - TRUE_LB[dim]);
}

static double eval_one(const double *xn) {
    double x[4];
    for (int i = 0; i < 4; i++) x[i] = rescale(xn[i], i);

    double f = 0.6224*x[0]*x[2]*x[3] + 1.7781*x[1]*x[2]*x[2]
             + 3.1661*x[0]*x[0]*x[3] + 19.84*x[0]*x[0]*x[2];

    double g1 = -x[0] + 0.0193*x[2];
    double g2 = -x[1] + 0.00954*x[2];
    double g3 = (-M_PI*x[2]*x[2]*x[3] - (4.0/3.0)*M_PI*x[2]*x[2]*x[2] + 1296000.0) / 1.0e6;  // normalized: raw g3 is O(1e6), others are O(1-100)
    double g4 = x[3] - 240.0;

    double viol = 0.0;
    if (g1 > 0) viol += g1;
    if (g2 > 0) viol += g2;
    if (g3 > 0) viol += g3;
    if (g4 > 0) viol += g4;

    return f + PENALTY * viol;
}

// Drop-in replacement: same signature as the official CEC2022 evaluator.
// x is nx*mx, row-major (individual i's genes at x[i*nx .. i*nx+nx-1]),
// matching the exact layout every algorithm in this study already uses.
void cec22_test_func(double *x, double *f, int nx, int mx, int func_num) {
    for (int i = 0; i < mx; i++) {
        f[i] = eval_one(&x[i * nx]);
    }
}
