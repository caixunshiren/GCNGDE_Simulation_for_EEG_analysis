
RK4 = ((  0,),
       (1/2, 1/2,),
       (1/2,   0,  1/2,),
       (  1,   0,    0,   1,),
       (1/6, 1/3, 1/3, 1/6,))

EF = ((0,),
      (1,))

def explicit_RK(b_tableau, f, x0, t0, t1, N):
    h = (t1 - t0) / float(N)  # calculate step size
    x = x0  # initialize saved dynamics
    mesh = (t0 + h * i for i in range(N))  # generator of time values
    for time in mesh:

        k = [f(x, time + h * b_tableau[0][0])]  # Covers the first row of the Butcher tableau.
        for i, row in enumerate(b_tableau[1:-1]):  # Covers the middle rows of the Butcher tableau.
            k.append(f(x + sum(w * k[idx] * h for idx, w in enumerate(row[1:])), time + row[0] * h))  # calculate k's.
        x = x + sum(w * k_i * h for k_i, w in zip(k, b_tableau[-1]))  # calculate timestep
    return x

