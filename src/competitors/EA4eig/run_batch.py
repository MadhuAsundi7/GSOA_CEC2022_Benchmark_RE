import sys
import time
import numpy as np
import os
from ea4eig import run_ea4eig

D = int(sys.argv[1])
start_f = int(sys.argv[2])
end_f = int(sys.argv[3])
start_run = int(sys.argv[4]) if len(sys.argv) > 4 else 0
end_run = int(sys.argv[5]) if len(sys.argv) > 5 else 30

MAXFES = 200000 if D == 10 else 1000000
FISTAR = [300,400,600,800,900,1800,2000,2200,2300,2400,2600,2700]

os.makedirs('results', exist_ok=True)

for func_num in range(start_f, end_f + 1):
    t0 = time.time()
    n_runs_batch = end_run - start_run
    all_conv = np.zeros((n_runs_batch, 17))
    for i, r in enumerate(range(start_run, end_run)):
        conv = run_ea4eig(func_num=func_num, D=D, maxFES=MAXFES, fistar=FISTAR[func_num-1], seed=1000 + func_num*100 + r)
        all_conv[i] = conv
    t1 = time.time()
    fname = f'results/EA4eig_{func_num}_{D}_r{start_run}-{end_run}.txt'
    np.savetxt(fname, all_conv.T, fmt='%.8e', delimiter='\t')
    print(f'F{func_num} D={D} runs[{start_run}:{end_run}] done in {t1-t0:.1f}s ({(t1-t0)/n_runs_batch:.2f}s/run)', flush=True)

