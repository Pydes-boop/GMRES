import time
import numpy as np

# This one typically doesnt work too well for elsa as it requires a square matrix A so it is simply implemented via numpy

def apply(A, x):
    return np.asarray(np.dot(A, x)).reshape(-1)

def log(text, logging):
    if logging: 
        print(text)

def GMRES(A, b, x0, nmax_iter, epsilon = None, logging=False):

    log("Starting preperations... ", logging=logging)

    ti = time.perf_counter()

    r0 = np.asarray(b).reshape(-1) - apply(A, x0).reshape(-1)

    h = np.zeros((nmax_iter + 1, nmax_iter))
    w = [np.zeros(len(r0))] * nmax_iter
    e = np.zeros(nmax_iter + 1)
    y = [0] * nmax_iter 

    w[0] = r0 / np.linalg.norm(r0)

    e[0] = np.linalg.norm(r0)

    elapsed_time = time.perf_counter() - ti

    log("Preperations done, took: " + str(elapsed_time) + "s", logging=logging)

    # --- 2. Iterate ---
    for k in range(nmax_iter):

        log("Iteration | residual | elapsed time | total time", logging=logging)

        t = time.perf_counter()

        q = apply(A, w[k].T)

        for i in range(k + 1):
            h[i, k] = apply(w[i], q)
            q = q - h[i, k] * w[i]

        h[k+1, k] = np.linalg.norm(q)

        if (h[k+1, k] != 0 and k != nmax_iter - 1):
            w[k + 1] = q / h[k+1,k]

        y = np.linalg.lstsq(h, e, rcond=None)[0]

        w_copy = np.reshape(np.asarray(w), (nmax_iter, len(w[0])))
        w_copy = w_copy.T

        x = x0 + apply(w_copy, y)
        r = b - apply(A, x)

        elapsed_time = time.perf_counter() - t
        ti += elapsed_time
        log(str(k) + " | " + str(np.linalg.norm(r)) + " | " + str(elapsed_time)[:6] + " | " + str(time)[:6], logging=logging)

        if epsilon is not None:
            if np.linalg.norm(np.asarray(r)) <= epsilon:
                print("Reached Convergence at: " + str(k) + "/" + str(nmax_iter))
                break


    return x, r

def GMRES_res(A, B, b, x0, nmax_iter, restarts, epsilon = None):
    x = x0
    for r in range(restarts):
        x, r = GMRES(A, B, b, x, nmax_iter, epsilon=epsilon)