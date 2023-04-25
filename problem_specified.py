#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:32:55 2021

@author: jhonhu
"""

# Import libraries
import numpy as np
import math

# Example 1
# def fun(x):
#     """ Objective function, f(x) """
#     fval = math.exp(np.prod(x)) - 0.5*(x[0]**3 + x[1]**3 + 1)**2
#     return fval


# def dfun(x):
#     """ Gradient of the objective function """
#     dim_x = np.size(x)
#     df = np.array([0.0 for i in range(dim_x)])
#     s = np.prod(x)
#     t = x[0]**3 + x[1]**3 + 1
#     df[0] = s/x[0]*math.exp(s) - 3*t*x[0]**2
#     df[1] = s/x[1]*math.exp(s) - 3*t*x[1]**2
#     df[2] = s/x[2]*math.exp(s)
#     df[3] = s/x[3]*math.exp(s)
#     df[4] = s/x[4]*math.exp(s)
#     return df


# def cons(x):
#     """ Constraint functions, h_i(x) == 0, g_i(x) >= 0 """
#     g = np.array([])
#     h = np.array([np.sum(x**2)-10, x[1]*x[2]-5*x[3]*x[4], x[0]**3+x[1]**3+1])
#     return h, g


# def dcons(x):
#     """ Jacobi matrix of constraint functions """
#     Ai = np.array([])
#     Ae = np.array([2*x, [0, x[2], x[1], -5*x[4], -5*x[3]], [3*x[0]**2, 3*x[1]**2, 0, 0, 0]])
#     return Ae, Ai


# Example 2
# def fun(x):
#     """ Objective function, f(x) """
#     fval = np.sum(x**2) - 16*x[0] -10*x[1]
#     return fval


# def dfun(x):
#     """ Gradient of the objective function """
#     dim_x = np.size(x)
#     df = np.array([0.0 for i in range(dim_x)])
#     df[0] = 2*x[0] - 16
#     df[1] = 2*x[1] - 10
#     return df


# def cons(x):
#     """ Constraint functions, h_i(x) == 0, g_i(x) >= 0 """
#     h = np.array([])
#     g = np.array([-x[0]**2+6*x[0]-4*x[1]+11, x[0]*x[1]-3*x[1]-math.exp(x[0]-3)+1 ,x[0], x[1]])
#     return h, g


# def dcons(x):
#     """ Jacobi matrix of constraint functions """
#     Ae = np.array([])
#     Ai = np.array([[-2*x[0]+6, -4], [x[1]-math.exp(x[0]-3), x[0]-3], [1, 0], [0, 1]])
#     return Ae, Ai


# Example 3
def fun(x):
    """ Objective function, f(x) """
    fval = -math.pi*pow(x[0],2)*x[1]
    return fval


def dfun(x):
    """ Gradient of the objective function """
    dim_x = np.size(x)
    df = np.array([0.0 for i in range(dim_x)])
    df[0] = -2*math.pi*x[0]*x[1]
    df[1] = -math.pi*pow(x[0],2)
    return df


def cons(x):
    """ Constraint functions, h_i(x) == 0, g_i(x) >= 0 """
    h = np.array([math.pi*np.prod(x)+math.pi*pow(x[0],2)-150])
    g = np.array([x[0], x[1]])
    return h, g


def dcons(x):
    """ Jacobi matrix of constraint functions """
    Ae = np.array([[math.pi*x[1]+2*math.pi*x[0], math.pi*x[0]]])
    Ai = np.array([[1, 0], [0, 1]])
    return Ae, Ai