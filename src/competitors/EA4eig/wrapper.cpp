double *OShift = nullptr, *M = nullptr, *y = nullptr, *z = nullptr, *x_bound = nullptr;
int ini_flag = 0, n_flag, func_flag, *SS = nullptr;

void cec22_test_func(double *x, double *f, int nx, int mx, int func_num);

extern "C" void cec22_wrapper(double *x, double *f, int nx, int mx, int func_num) {
    cec22_test_func(x, f, nx, mx, func_num);
}

extern "C" void reset_ini_flag() {
    ini_flag = 0;
}
