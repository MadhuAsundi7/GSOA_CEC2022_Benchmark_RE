"""
EA4eig - faithful Python translation of the official CEC2022 competition
MATLAB source (Run_EA4eig.m), evaluated against the exact same compiled
CEC2022 evaluator (libcec22.so) used by every other algorithm in this study.

Structure mirrors Run_EA4eig.m one-to-one: 4 adaptive sub-strategies
(CoBiDE, IDE, CMA-ES, jSO) selected via roulette-wheel bandit selection,
linear population-size reduction, and the same convergence-checkpoint
logging format (16 FES-fraction checkpoints + FES-to-target).
"""
import numpy as np
import ctypes
import os

LIBDIR = os.path.dirname(os.path.abspath(__file__))
_lib_cache = {}

def get_lib(D):
    key = D
    if key not in _lib_cache:
        lib = ctypes.CDLL(os.path.join(LIBDIR, 'libcec22.so'))
        lib.cec22_wrapper.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int]
        _lib_cache[key] = lib
    return _lib_cache[key]

def cec_eval(X, func_num, D):
    """X: (n_pts, D) array (row-major, matches nx*mx C layout used throughout this study)."""
    lib = get_lib(D)
    n = X.shape[0]
    Xc = np.ascontiguousarray(X, dtype=np.float64)
    f = np.zeros(n, dtype=np.float64)
    lib.cec22_wrapper(Xc.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       D, n, func_num)
    return f

# ---------------- helpers (direct translation of the .m helper files) ----------------
def cauchy_rnd(rng, x0, gamma):
    return x0 + gamma * np.tan(np.pi * (rng.random() - 0.5))

def nahvyb(rng, N, k):
    """random sample of k indices (0-based) from range(N) without repetition"""
    opora = list(range(N))
    out = np.zeros(k, dtype=np.int64)
    for i in range(k):
        idx = int(rng.random() * len(opora))
        out[i] = opora[idx]
        del opora[idx]
    return out

def nahvyb_expt(rng, N, k, expt):
    """random sample of k indices (0-based) from range(N), excluding indices in expt (list or scalar)"""
    opora = list(range(N))
    if np.isscalar(expt):
        expt = [expt]
    for e in expt:
        if e in opora:
            opora.remove(e)
    out = np.zeros(k, dtype=np.int64)
    for i in range(k):
        idx = int(rng.random() * len(opora))
        out[i] = opora[idx]
        del opora[idx]
    return out if k > 1 else out[0]

def roulete(rng, cutpoints):
    h = len(cutpoints)
    ss = sum(cutpoints)
    p_min = min(cutpoints) / ss
    cp = np.cumsum(cutpoints) / ss
    res = int(np.sum(cp < rng.random()))
    return res, p_min  # 0-based result

def zrcad(y, a, b):
    """mirror y into [a,b], element-wise, vectorized"""
    y = y.copy()
    for _ in range(50):  # safety cap; converges almost always in 1-2 iters
        below = y < a
        above = y > b
        if not (below.any() or above.any()):
            break
        y = np.where(above, 2 * b - y, y)
        y = np.where(below, 2 * a - y, y)
    return y


def run_ea4eig(func_num, D, maxFES, fistar, seed):
    rng = np.random.default_rng(seed)
    a = -100.0 * np.ones(D)
    b = 100.0 * np.ones(D)

    fmin_stage = np.array([round(D ** (i / 5.0 - 3.0) * maxFES) for i in range(16)])
    stage = len(fmin_stage)
    val_2_reach = 1e-8

    h = 4
    N_init = 100
    N = N_init
    Nmin = 10
    n0 = 2
    ni = np.zeros(h) + n0
    success = np.zeros(h)

    Run_RecordFEsFactor = list(fmin_stage)
    run_funcvals = []

    P = a + rng.random((N, D)) * (b - a)
    Pf = cec_eval(P, func_num, D)
    FES = N
    bsf_fit_var = np.min(Pf)
    suc_total = 0

    delta = 1.0 / (5 * h)

    gmax = round(maxFES / N)
    T = gmax / 10.0
    GT = int(gmax // 2)
    gt = GT
    g = 0
    Tcurr = 0

    CBps = 0.5
    peig = 0.4
    ceig = 0

    CBF = np.zeros(N)
    CBCR = np.zeros(N)
    for i in range(N):
        CBF[i] = cauchy_rnd(rng, 0.65, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 1, 0.1)
        while CBF[i] < 0:
            CBF[i] = cauchy_rnd(rng, 0.65, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 1, 0.1)
        if CBF[i] > 1:
            CBF[i] = 1
        CBCR[i] = cauchy_rnd(rng, 0.1, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 0.95, 0.1)
        CBCR[i] = min(max(CBCR[i], 0), 1)

    # CMA-ES params
    sigma = (b[0] - a[0]) / 2.0
    oldPop = P.T.copy()
    myeps = 1e-6

    mu = N / 2.0
    weights = np.log(mu + 0.5) - np.log(np.arange(1, int(mu) + 1))
    mu = int(np.floor(mu))
    weights = weights / np.sum(weights)
    mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

    cc = (4 + mueff / D) / (D + 4 + 2 * mueff / D)
    cs = (mueff + 2) / (D + mueff + 5)
    c1 = 2 / ((D + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((D + 2) ** 2 + mueff))
    damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (D + 1)) - 1) + cs

    pc_ = np.zeros(D)
    ps_ = np.zeros(D)
    B = np.eye(D)
    Dg = np.ones(D)
    CC = B @ np.diag(Dg ** 2) @ B.T
    invsqrtC = B @ np.diag(Dg ** -1) @ B.T
    eigeneval = 0
    chiN = D ** 0.5 * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

    # jSO params
    Asize_max = round(N * 2.6)
    H = 5
    MF = 0.3 * np.ones(H)
    MCR = 0.8 * np.ones(H)
    MF[H - 1] = 0.9
    MCR[H - 1] = 0.9
    k_mem = 0
    Asize = 0
    A = np.zeros((0, D))
    pmax = 0.25
    pmin = pmax / 2

    div_stage = []
    if FES >= Run_RecordFEsFactor[0]:
        run_funcvals.append(bsf_fit_var)
        Run_RecordFEsFactor.pop(0)
    FESterm = 0

    while (FES < maxFES) and ((bsf_fit_var - fistar) >= val_2_reach):
        hh, p_min = roulete(rng, ni)  # 0-based strategy index
        if p_min < delta:
            ni = np.zeros(h) + n0

        # ============================================================
        if hh == 0:  # CoBiDE
            Q = np.zeros((N, D))
            Qf = np.zeros(N)
            if rng.random() < peig:
                ceig = 1
                order = np.argsort(Pf)
                keep = order[:round(N * CBps)]
                Popeig = P[keep]
                covM = np.cov(Popeig.T)
                eigvals, EigVect = np.linalg.eigh(covM)
                for i in range(N):
                    vyb = nahvyb_expt(rng, N, 3, i)
                    r1, r2, r3 = P[vyb[0]], P[vyb[1]], P[vyb[2]]
                    v = r1 + CBF[i] * (r2 - r3)
                    v = zrcad(v, a, b)
                    y_ = P[i]
                    yeig = EigVect.T @ y_
                    veig = EigVect.T @ v
                    change = np.where(rng.random(D) < CBCR[i])[0]
                    if len(change) == 0:
                        change = [int(D * rng.random())]
                    yeig[change] = veig[change]
                    y_ = EigVect @ yeig
                    y_ = zrcad(y_, a, b)
                    Q[i] = y_
                    FES += 1
                Qf = cec_eval(Q, func_num, D)
            else:
                for i in range(N):
                    vyb = nahvyb_expt(rng, N, 3, i)
                    r1, r2, r3 = P[vyb[0]], P[vyb[1]], P[vyb[2]]
                    v = r1 + CBF[i] * (r2 - r3)
                    v = zrcad(v, a, b)
                    y_ = P[i].copy()
                    change = np.where(rng.random(D) < CBCR[i])[0]
                    if len(change) == 0:
                        change = [int(D * rng.random())]
                    y_[change] = v[change]
                    y_ = zrcad(y_, a, b)
                    Q[i] = y_
                    FES += 1
                Qf = cec_eval(Q, func_num, D)

            for i in range(N):
                if Qf[i] <= Pf[i]:
                    P[i] = Q[i]
                    Pf[i] = Qf[i]
                    success[hh] += 1
                    ni[hh] += 1
                else:
                    CBF[i] = cauchy_rnd(rng, 0.65, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 1, 0.1)
                    while CBF[i] < 0:
                        CBF[i] = cauchy_rnd(rng, 0.65, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 1, 0.1)
                    if CBF[i] > 1:
                        CBF[i] = 1
                    CBCR[i] = cauchy_rnd(rng, 0.1, 0.1) if rng.random() < 0.5 else cauchy_rnd(rng, 0.95, 0.1)
                    CBCR[i] = min(max(CBCR[i], 0), 1)
            ceig = 0
            fmin = np.min(Pf)
            if fmin < bsf_fit_var:
                bsf_fit_var = fmin

        # ============================================================
        elif hh == 1:  # IDE
            order = np.argsort(Pf)
            P = P[order]; Pf = Pf[order]; CBF = CBF[order]; CBCR = CBCR[order]
            Q = P.copy()
            IDEps = 0.1 + 0.9 * 10 ** (5 * (g / gmax - 1))
            pd = 0.1 * IDEps
            SRT = 0.0 if g < gt else 0.1

            for i in range(N):
                vyb = nahvyb_expt(rng, N, 4, i)
                o, r1, r2, r3 = vyb[0], vyb[1], vyb[2], vyb[3]
                if g <= gt:
                    o = i
                xo, xr1, xr2, xr3 = P[o].copy(), P[r1], P[r2], P[r3].copy()
                indperturb = np.where(rng.random(D) < pd)[0]
                pom = a + rng.random(D) * (b - a)
                xr3[indperturb] = pom[indperturb]

                Fo = o / N + 0.1 * rng.standard_normal()
                while Fo <= 0 or Fo > 1:
                    Fo = o / N + 0.1 * rng.standard_normal()

                high_ind_S = int(IDEps * N)
                if o > high_ind_S:
                    if r1 > high_ind_S:
                        candidates = [c for c in range(high_ind_S) if c not in list(vyb) + [i]]
                        if len(candidates) > 0:
                            r1 = candidates[int(rng.random() * len(candidates))]
                            xr1 = P[r1]

                if (g > gt) and (rng.random() < 0.5):
                    Q[i] = P[i] + Fo * (xr1 - xo) + Fo * (xr2 - xr3)
                else:
                    Q[i] = xo + Fo * (xr1 - xo) + Fo * (xr2 - xr3)

            if rng.random() < peig:
                ceig = 1
                order2 = np.argsort(Pf)
                keep = order2[:round(N * CBps)]
                Popeig = P[keep]
                covM = np.cov(Popeig.T)
                eigvals, EigVect = np.linalg.eigh(covM)
                for i in range(N):
                    y_ = P[i]; v = Q[i]
                    yeig = EigVect.T @ y_
                    veig = EigVect.T @ v
                    change = np.where(rng.random(D) < CBCR[i])[0]
                    if len(change) == 0:
                        change = [int(D * rng.random())]
                    yeig[change] = veig[change]
                    y_ = EigVect @ yeig
                    Q[i] = zrcad(y_, a, b)
            else:
                ceig = 0
                for i in range(N):
                    CR = i / N + 0.1 * rng.standard_normal()
                    while CR < 0 or CR > 1:
                        CR = i / N + 0.1 * rng.standard_normal()
                    jrand = int(D * rng.random())
                    for j in range(D):
                        if not (rng.random() <= CR or j == jrand):
                            Q[i, j] = P[i, j]
                        if Q[i, j] < a[j] or Q[i, j] > b[j]:
                            Q[i, j] = a[j] + rng.random() * (b[j] - a[j])

            Qf = cec_eval(Q, func_num, D)
            FES += N
            indsucc = np.where(Qf <= Pf)[0]
            success[hh] += len(indsucc)
            ni[hh] += len(indsucc)
            SR = len(indsucc) / N
            if g < gt:
                if SR <= SRT:
                    Tcurr += 1
                else:
                    Tcurr = 0
                if Tcurr >= T:
                    gt = g

            P[indsucc] = Q[indsucc]; Pf[indsucc] = Qf[indsucc]
            order3 = np.argsort(Pf)
            P = P[order3]; Pf = Pf[order3]; CBF = CBF[order3]; CBCR = CBCR[order3]

            fmin = Pf[0]
            if fmin < bsf_fit_var:
                bsf_fit_var = fmin
            g += 1

        # ============================================================
        elif hh == 2:  # CMA-ES
            order = np.argsort(Pf)
            P = P[order]; Pf = Pf[order]; CBF = CBF[order]; CBCR = CBCR[order]

            xmean = P[:mu].T @ weights
            Pop = np.zeros((D, N))
            PopFit = np.zeros(N)
            for kk in range(N):
                Pop[:, kk] = xmean + sigma * (B @ (Dg * rng.standard_normal(D)))
                zrca = Pop[:, kk] < a
                Pop[zrca, kk] = (oldPop[zrca, kk] + a[zrca]) / 2
                zrcb = Pop[:, kk] > b
                Pop[zrcb, kk] = (oldPop[zrcb, kk] + b[zrcb]) / 2

                PopFit[kk] = cec_eval(Pop[:, kk:kk+1].T, func_num, D)[0]
                FES += 1
                maxind = np.argmax(Pf)
                if PopFit[kk] < Pf[maxind]:
                    P[maxind] = Pop[:, kk]
                    Pf[maxind] = PopFit[kk]
                    success[hh] += 1
                    ni[hh] += 1

            FitInd = np.argsort(PopFit)
            xold = xmean.copy()
            xmean = Pop[:, FitInd[:mu]] @ weights
            fmin = np.min(Pf)
            if fmin < bsf_fit_var:
                bsf_fit_var = fmin

            oldPop = Pop.copy()

            ps_[:] = (1 - cs) * ps_ + np.sqrt(cs * (2 - cs) * mueff) * (invsqrtC @ (xmean - xold)) / sigma
            hsig = (np.sum(ps_ ** 2) / (1 - (1 - cs) ** (2 * FES / N)) / D) < (2 + 4 / (D + 1))
            pc_[:] = (1 - cc) * pc_ + hsig * np.sqrt(cc * (2 - cc) * mueff) * (xmean - xold) / sigma

            artmp = (1.0 / sigma) * (Pop[:, FitInd[:mu]] - xold[:, None])
            CC[:] = ((1 - c1 - cmu) * CC + c1 * (np.outer(pc_, pc_) + (1 - hsig) * cc * (2 - cc) * CC)
                      + cmu * artmp @ np.diag(weights) @ artmp.T)

            sigma = sigma * np.exp((cs / damps) * (np.linalg.norm(ps_) / chiN - 1))
            if sigma > 1e300 or sigma < 1e-300 or np.isnan(sigma):
                sigma = (b[0] - a[0]) / 2

            if FES - eigeneval > N / (c1 + cmu) / D / 10:
                eigeneval = FES
                CC[:] = np.triu(CC) + np.triu(CC, 1).T
                eigvals, B_ = np.linalg.eigh(CC)
                Dg[:] = np.sqrt(np.maximum(eigvals, 1e-300))
                B[:] = B_
                invsqrtC[:] = B @ np.diag(Dg ** -1) @ B.T

            if PopFit[FitInd[0]] <= myeps or np.max(Dg) > 1e7 * np.min(Dg):
                break

        # ============================================================
        else:  # hh == 3, jSO
            Fpole = -1 * np.ones(N)
            CRpole = -1 * np.ones(N)
            SCR, SF = [], []
            suc = 0
            Q = np.zeros((N, D))
            deltafce = -1 * np.ones(N)
            pp = pmax - (pmax - pmin) * (FES / maxFES)

            use_eig = rng.random() < peig
            if use_eig:
                ceig = 1
                order2 = np.argsort(Pf)
                keep = order2[:round(N * CBps)]
                Popeig = P[keep]
                covM = np.cov(Popeig.T)
                eigvals, EigVect = np.linalg.eigh(covM)
            else:
                ceig = 0

            order_p = np.argsort(Pf)
            pom_sorted = P[order_p]

            for i in range(N):
                rr = int(nahvyb(rng, H, 1)[0])
                CR = MCR[rr] + np.sqrt(0.1) * rng.standard_normal()
                CR = min(max(CR, 0), 1)
                if FES < 0.25 * maxFES:
                    CR = max(CR, 0.7)
                elif FES < 0.5 * maxFES:
                    CR = max(CR, 0.6)
                F = -1
                while F <= 0:
                    Fr = rng.random() * np.pi - np.pi / 2
                    F = 0.1 * np.tan(Fr) + MF[rr]
                if F > 1:
                    F = 1
                if FES < 0.6 * maxFES and F > 0.7:
                    F = 0.7
                Fpole[i] = F
                CRpole[i] = CR

                p = max(2, int(np.ceil(pp * N)))
                xpbest = pom_sorted[int(p * rng.random())]

                xi = P[i]
                vyb = nahvyb_expt(rng, N, 1, i)
                r1 = P[int(vyb)]

                sjed = np.vstack([P, A]) if Asize > 0 else P
                vyb2 = nahvyb_expt(rng, N + Asize, 1, [i, int(vyb)])
                r2 = sjed[int(vyb2)]

                if FES < 0.2 * maxFES:
                    Fw = 0.7 * F
                elif FES < 0.4 * maxFES:
                    Fw = 0.8 * F
                else:
                    Fw = 1.2 * F

                v = xi + Fw * (xpbest - xi) + F * (r1 - r2)

                if use_eig:
                    y_ = xi.copy()
                    yeig = EigVect.T @ y_
                    veig = EigVect.T @ v
                    change = np.where(rng.random(D) < CBCR[i])[0]
                    if len(change) == 0:
                        change = [int(D * rng.random())]
                    yeig[change] = veig[change]
                    y_ = EigVect @ yeig
                    Q[i] = y_
                else:
                    y_ = xi.copy()
                    change = np.where(rng.random(D) < CR)[0]
                    if len(change) == 0:
                        change = [int(D * rng.random())]
                    y_[change] = v[change]
                    Q[i] = zrcad(y_, a, b)

            Qf = cec_eval(Q, func_num, D)
            for i in range(N):
                if Qf[i] < Pf[i]:
                    deltafce[i] = Pf[i] - Qf[i]
                    suc += 1
                    if Asize < Asize_max:
                        A = np.vstack([A, P[i]]) if Asize > 0 else P[i:i+1].copy()
                        Asize += 1
                    else:
                        ktere = int(nahvyb(rng, Asize, 1)[0])
                        A[ktere] = P[i]
                    SCR.append(CRpole[i])
                    SF.append(Fpole[i])
                if Qf[i] <= Pf[i]:
                    P[i] = Q[i]; Pf[i] = Qf[i]

            if suc > 0:
                MCR_old = MCR[k_mem]; MF_old = MF[k_mem]
                platne = np.where(deltafce != -1)[0]
                delty = deltafce[platne]
                sum_delta = np.sum(delty)
                vahyw = delty / sum_delta
                SCR = np.array(SCR); SF = np.array(SF)
                mSCR = np.max(SCR)
                if MCR[k_mem] == -1 or mSCR == 0:
                    MCR[k_mem] = -1
                else:
                    meanSCRpomjm = vahyw * SCR
                    meanSCRpomci = meanSCRpomjm * SCR
                    MCR[k_mem] = np.sum(meanSCRpomci) / np.sum(meanSCRpomjm)
                meanSFpomjm = vahyw * SF
                meanSFpomci = meanSFpomjm * SF
                MF[k_mem] = np.sum(meanSFpomci) / np.sum(meanSFpomjm)
                MCR[k_mem] = (MCR[k_mem] + MCR_old) / 2
                MF[k_mem] = (MF[k_mem] + MF_old) / 2
                k_mem = (k_mem + 1) % H

            FES += N
            fmin = np.min(Pf)
            if fmin < bsf_fit_var:
                bsf_fit_var = fmin
            success[hh] += suc
            ni[hh] += suc

        # ---- shared bookkeeping every iteration ----
        curmin = np.min(Pf)
        if curmin < bsf_fit_var:
            bsf_fit_var = curmin
            if (bsf_fit_var - fistar) < val_2_reach:
                FESterm = FES

        if len(Run_RecordFEsFactor) > 0 and FES >= Run_RecordFEsFactor[0]:
            run_funcvals.append(bsf_fit_var)
            Run_RecordFEsFactor.pop(0)

        optN = round(((Nmin - N_init) / maxFES) * FES + N_init)
        if N > optN:
            diffPop = N - optN
            if N - diffPop < Nmin:
                diffPop = N - Nmin
            N = N - diffPop
            order4 = np.argsort(Pf)
            P = P[order4][:N]; Pf = Pf[order4][:N]
            CBF = CBF[order4][:N]; CBCR = CBCR[order4][:N]
            Asize_max = round(N * 2.6)
            while Asize > Asize_max:
                idx_del = int(nahvyb(rng, Asize, 1)[0])
                A = np.delete(A, idx_del, axis=0)
                Asize -= 1
            mu = int(np.floor(N / 2))
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
            # oldPop must also track reduced population for CMA-ES consistency
            if oldPop.shape[1] > N:
                oldPop = oldPop[:, order4][:, :N] if oldPop.shape[1] == len(order4) else oldPop[:, :N]

    # -------- finalize --------
    run_funcvals = np.array(run_funcvals) - fistar
    if len(run_funcvals) < stage:
        run_funcvals = np.concatenate([run_funcvals, np.full(stage - len(run_funcvals), bsf_fit_var - fistar)])
    if FESterm == 0:
        FESterm = maxFES

    convergence = np.concatenate([run_funcvals, [FESterm]])
    return convergence
