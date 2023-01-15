import time
import numpy as np

def apply(A, x):
    return np.asarray(np.dot(A, x)).reshape(-1)

def log(text, logging):
    if logging: 
        print(text)

def BAGMRES(A, B, b, x0, nmax_iter, epsilon = None, logging=False):

    log("Starting preperations... ", logging=logging)

    ti = time.perf_counter()

    # Saving this shape as in tomographic reconstruction cases with elsa this is not a vector so we have to translate the shape
    x0_shape = np.shape(np.asarray(x0))

    # r0 = Bb - BAx
    r0 = apply(B, b).reshape(-1) - apply(B, apply(A, x0)).reshape(-1)

    h = np.zeros((nmax_iter + 1, nmax_iter))
    w = [np.zeros(len(r0))] * nmax_iter
    e = np.zeros(nmax_iter + 1)
    y = [0] * nmax_iter

    e[0] = np.linalg.norm(r0)

    w[0] = r0 / np.linalg.norm(r0)

    elapsed_ti = time.perf_counter() - ti
    log("Preperations done, took: " + str(elapsed_ti) + "s", logging=logging)

    # --- 2. Iterate ---
    for k in range(nmax_iter):

        log("Iteration | residual | elapsed ti | total ti", logging=logging)

        t = time.perf_counter()

        # q = BAw_k
        q = apply(B, apply(A, np.reshape(w[k], x0_shape))).reshape(-1)

        for i in range(k+1):
            h[i, k] = apply(q.T, w[i])
            q = q - h[i, k] * w[i]
        
        h[k+1, k] = np.linalg.norm(q)

        if (h[k + 1, k] != 0 and k != nmax_iter - 1):
            w[k+1] = q/h[k+1, k]

        # Solving minimization problem using numpy leastsquares
        y = np.linalg.lstsq(h, e, rcond=None)[0]

        # transforming list of vectors to a matrix
        w_copy = np.reshape(np.asarray(w), (nmax_iter, len(w[0]))).T

        # applying estimated guess from our generated krylov subspace to our initial guess x0
        x = np.asarray(x0) + np.reshape(apply(w_copy, y), x0_shape)

        # calculating a residual
        r = np.asarray(b).reshape(-1) - (apply(A, np.reshape(x, x0_shape))).reshape(-1)

        elapsed_ti = time.perf_counter() - t
        ti += elapsed_ti
        log(str(k) + " | " + str(np.linalg.norm(r)) + " | " + str(elapsed_ti)[:6] + " | " + str(ti)[:6], logging=logging)

        if epsilon is not None:
            if np.linalg.norm(np.asarray(r)) <= epsilon:
                print("Reached Convergence at: " + str(k) + "/" + str(nmax_iter))
                break

    return x, r

def BAGMRES_res(A, B, b, x0, nmax_iter, restarts, epsilon = None, logging=False):
    x = x0
    for r in range(restarts):
        x, r = BAGMRES(A, B, b, x, nmax_iter, epsilon=epsilon, logging=logging)