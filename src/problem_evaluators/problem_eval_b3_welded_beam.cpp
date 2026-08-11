// Appendix B.3 - Design of Welded Beam
// x1,x2,x3,x4 -> weld dims. Minimize:
//   f(x) = 1.10471 x1^2 x2 + 0.04811 x3 x4 (14.0 + x2)
// Subject to 7 constraints (g1=tau-taumax, g2=sigma-sigmamax, g3=x1-x4,
// g4=cost-related, g5=0.125-x1, g6=delta-deltamax, g7=P-Pc(buckling)).
// g1 and g7 have natural magnitude ~O(1e3-1e4) (stress/buckling-load units)
// vs the other constraints' O(0.1-10), so both are normalized /1000 before
// penalizing, same rationale as B1's g3 normalization.
// Bounds (true): 0.1<=x1,x4<=2.0, 0.1<=x2,x3<=10.0

#include <cmath>

static const double PENALTY = 1.0e6;
static const int D_DIM = 4;
static const double TRUE_LB[4] = {0.1, 0.1, 0.1, 0.1};
static const double TRUE_UB[4] = {2.0, 10.0, 10.0, 2.0};

static const double P_load = 6000.0, L = 14.0, E = 30.0e6, G = 12.0e6;
static const double TAU_MAX = 13600.0, SIGMA_MAX = 30000.0, DELTA_MAX = 0.25;

static inline double rescale(double xn, int dim) {
    double t = (xn + 100.0) / 200.0;
    if (t < 0.0) t = 0.0;
    if (t > 1.0) t = 1.0;
    return TRUE_LB[dim] + t * (TRUE_UB[dim] - TRUE_LB[dim]);
}

static double eval_one(const double *xn) {
    double x[4];
    for (int i = 0; i < 4; i++) x[i] = rescale(xn[i], i);
    double x1 = x[0], x2 = x[1], x3 = x[2], x4 = x[3];

    double f = 1.10471*x1*x1*x2 + 0.04811*x3*x4*(14.0 + x2);

    double M = P_load * (L + x2/2.0);
    double R = sqrt(x2*x2/4.0 + ((x1+x3)/2.0)*((x1+x3)/2.0));
    double J = 2.0 * (sqrt(2.0)*x1*x2*(x2*x2/12.0 + ((x1+x3)/2.0)*((x1+x3)/2.0)));
    double tau_p = P_load / (sqrt(2.0)*x1*x2);
    double tau_pp = M*R/J;
    double tau = sqrt(tau_p*tau_p + 2.0*tau_p*tau_pp*(x2/(2.0*R)) + tau_pp*tau_pp);

    double sigma = 6.0*P_load*L / (x4*x3*x3);
    double delta = 4.0*P_load*L*L*L / (E*x3*x3*x3*x4);
    double Pc = (4.013*E*sqrt(x3*x3*pow(x4,6.0)/36.0)/(L*L)) * (1.0 - x3/(2.0*L)*sqrt(E/(4.0*G)));

    double g1 = (tau - TAU_MAX) / 1000.0;      // normalized: stress-scale
    double g2 = sigma - SIGMA_MAX;
    double g3 = x1 - x4;
    double g4 = 0.10471*x1*x1 + 0.04811*x3*x4*(14.0+x2) - 5.0;
    double g5 = 0.125 - x1;
    double g6 = delta - DELTA_MAX;
    double g7 = (P_load - Pc) / 1000.0;         // normalized: force-scale

    double viol = 0.0;
    if (g1 > 0) viol += g1;
    if (g2 > 0) viol += g2;
    if (g3 > 0) viol += g3;
    if (g4 > 0) viol += g4;
    if (g5 > 0) viol += g5;
    if (g6 > 0) viol += g6;
    if (g7 > 0) viol += g7;

    return f + PENALTY * viol;
}

void cec22_test_func(double *x, double *f, int nx, int mx, int func_num) {
    for (int i = 0; i < mx; i++) {
        f[i] = eval_one(&x[i * nx]);
    }
}
