def RMS(x, a, b, delta, t, gradfunc, v = 0):
    for idx in range(1, t):
        g = gradfunc(x)
        v = b * v  + (1 - b) * (g * g)
        
        e = delta / np.sqrt(idx)
        a_new = a / np.sqrt(idx)        
        
        A = np.diag(np.sqrt(v)) + e * np.eye(len(v))
        x = x - a_new * np.linalg.inv(A) @ g
    return x