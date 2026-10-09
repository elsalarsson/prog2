""" MA3.py

Student:
Mail:
Reviewed by:
Date reviewed:

"""
import random
import matplotlib.pyplot as plt
import math as m
import concurrent.futures as future
from statistics import mean 
from time import perf_counter as pc
import functools
import numpy as np
from numba import njit

# Exc1
def approximate_pi(n):
    n = list(zip([random.uniform(-1,1) for i in range(n)],[random.uniform(-1,1) for i in range(n)]))
    #plot
    n_c = [e for e in n if m.sqrt(e[0]**2 + e[1]**2) <= 1]
    n_s = [e for e in n if m.sqrt(e[0]**2 + e[1]**2) > 1]
    x_s = [e[0] for e in n_s]
    y_s = [e[1] for e in n_s]
    x_c = [e[0] for e in n_c]
    y_c = [e[1] for e in n_c]
    plt.scatter(x_s, y_s, color='blue')
    plt.scatter(x_c,y_c, color='red')
    plt.savefig('approximate_pi.png')
    plt.show()
    #calc
    return 4*(len(n_c)/len(n))

# Exc2, approximation

def sphere_volume(n, d): 
    n =  [[random.uniform(-1,1) for i in range(d)] for j in range(n)]
    f = lambda x: x**2
    sphere_n = [p for p in n if functools.reduce(lambda x, y: x+y, map(f, p)) <= 1]
    ratio = len(sphere_n) / len(n)
    cube_V = (2*1)**d
    return ratio * cube_V

#Exc2, real value
def hypersphere_exact(n, d): 
    return ((m.pi)**(d/2)) / (m.gamma((d/2)+1))

#Exc3: numba version
n=1000000
d=11
a = np.random.uniform(-1, 1, size=(n, d))
@njit
def sphere_volume_numba(n:int, d:int)->float:
    sphere_n = 0
    for p in a:
        len_p = 0 
        for i in p:
            len_p += i**2
        len_p = m.sqrt(len_p)
        if len_p <= 1:
            sphere_n += 1
    ratio = sphere_n / len(a)
    cube_V = (2)**d
    return ratio * cube_V

def sphere_filter(p):
        sub_a = np.random.uniform(-1, 1, size=(p, d))
        sphere_n = 0
        for point in sub_a:
            len_p = 0
            for i in point:
                len_p += i**2
            len_p = m.sqrt(len_p)
            if len_p <= 1:
                sphere_n += 1
        return sphere_n

#Exc4: parallel code - parallelize actual computations by splitting data
def sphere_volume_parallel(n, d, np=10):
    with future.ProcessPoolExecutor() as ex:
        sub = n//np
        p = [sub for i in range(np)]
        results = ex.map(sphere_filter, p)
    sphere_ns = sum(results)
    ratio = sphere_ns / sum(p)
    cube_V = (2)**d
    return ratio * cube_V

def main():
    # Exc1
    dots = [1000, 10000, 100000]
    for n in dots:
        approximate_pi(n)

    # Exc2
    n = 100000
    d = 2
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    n = 100000
    d = 11
    sphere_volume(n, d)
    print(f"Actual volume of {d} dimentional sphere = {hypersphere_exact(n,d)}")

    # Exc3
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc3: Sequential time of {d} and {n}: {stop-start}")
    print("What is numba time?")
    
    start = pc()
    sphere_volume_numba(n, d)
    stop = pc()
    print(f"Numba time of {d} and {n}: {stop-start}")

    # Exc4
    n = 1000000
    d = 11
    start = pc()
    sphere_volume(n, d)
    stop = pc()
    print(f"Exc4: Sequential time of {d} and {n}: {stop-start}")
    print("What is parallel time?")

    start = pc()
    sphere_volume_parallel(n, d, 10)
    stop = pc()
    print(f"Parallel time of {d} and {n}: {stop-start}")

    
    

if __name__ == '__main__':
	main()
