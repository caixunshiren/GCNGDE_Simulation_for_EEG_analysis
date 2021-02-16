# This no worky!!
# import multiprocessing
#
# def calc_stuff(a):
#     return a*a
#
# pool = multiprocessing.Pool(4)
# offset=100
# out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))
# out1, out2, out3 = zip(*pool.map(calc_stuff, range(0, 10 * offset, offset)))

from math import sqrt
from joblib import Parallel, delayed
import time

t0 = time.time()
k=100000
bro=Parallel(n_jobs=2)(delayed(sqrt)(i ** 2) for i in range(k))
print('Time elapsed {}'.format(time.time()-t0))

t0 = time.time()
bro=[]
for i in range(k):
    bro.append(sqrt(i**2))
print('Time elapsed {}'.format(time.time()-t0))