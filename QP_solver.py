#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 12:45:17 2021

@author: jhonhu
"""

# Import libraries
import numpy as np
import copy


def phi(ep, a, b):
    """ Fischer-Burmeister smoothing function """
    val = a + b - pow(a**2 + b**2 + 2*ep**2, 0.5)
    return val


def dah(ep, d, mu, lam, B, df, Ai, g, Ae, h):
    """ System function H(z) """
    dim_x = np.size(df)
    dim_mu = np.size(h)
    dim_lam = np.size(g)
    
    dh = np.array([0.0 for i in range(dim_x + dim_mu + dim_lam + 1)])
    dh[0] = ep
    if dim_mu > 0 and dim_lam > 0:            
        Ae_T_mu = np.matmul(Ae.T, mu)
        Ai_T_lam = np.matmul(Ai.T, lam)
        dh[1:dim_x+1] = np.matmul(B,d) - Ae_T_mu - Ai_T_lam + df
        dh[dim_x+1:dim_x+dim_mu+1] = h + np.matmul(Ae, d)
        Ai_2d = np.reshape(Ai,(dim_lam,dim_x))
        for i in range(dim_lam):
            dh[dim_x+dim_mu+1+i] = phi(ep, lam[i], g[i]+np.sum(Ai_2d[i]*d))
    elif dim_lam == 0:
        Ae_T_mu = np.matmul(Ae.T, mu)
        dh[1:dim_x+1] = np.matmul(B,d) - Ae_T_mu + df
        dh[dim_x+1:dim_x+dim_mu+1] = h + np.matmul(Ae, d)
    elif dim_mu == 0:
        Ai_T_lam = np.matmul(Ai.T, lam)
        dh[1:dim_x+1] = np.matmul(B,d) - Ai_T_lam + df
        Ai_2d = np.reshape(Ai,(dim_lam,dim_x))
        for i in range(dim_lam):
            dh[dim_x+1+i] = phi(ep, lam[i], g[i]+np.sum(Ai_2d[i]*d))
    
    return dh


def ddv(ep, d, lam, Ai, g):
    """ Derivative of Phi=[..., phi(ep, lam[i]], g[1]+Ai[i]*d), ...] """
    dim_x = np.size(d)
    dim_lam = np.size(g)
    dd1 = np.zeros((dim_lam, dim_lam))
    dd2 = np.zeros((dim_lam, dim_lam))
    v1 = np.array([0.0 for i in range(dim_lam)])
    Ai = np.reshape(Ai,(dim_lam,dim_x))
    for i in range(dim_lam):
        fm = pow(lam[i]**2 + (g[i]+np.sum(Ai[i]*d))**2 + 2*ep**2, 0.5)  # originating from the F-B smoothing function
        dd1[i,i] = 1 - lam[i]/fm
        dd2[i,i] = 1 - (g[i]+np.sum(Ai[i]*d))/fm
        v1[i] = -2*ep/fm
        
    return dd1, dd2, v1
 

def JacobiH(ep, d, mu, lam, B, df, Ai, g, Ae, h):
    """ Jacobi martix of H(z) """
    dim_x = np.size(d)
    dim_mu = np.size(mu)
    dim_lam = np.size(lam)
    
    dd1, dd2, v1 = ddv(ep, d, lam, Ai, g)
    if dim_mu > 0 and dim_lam > 0:
        A0 = np.array([0.0 for i in range(dim_x+dim_mu+dim_lam+1)])
        A0[0] = 1
        A1 = np.hstack((np.zeros((dim_x,1)), B, -Ae.T, -Ai.T))
        A2 = np.hstack((np.zeros((dim_mu,1)), Ae.reshape((dim_mu, dim_x)), np.zeros((dim_mu, dim_mu)), np.zeros((dim_mu,dim_lam))))
        A3 = np.hstack((np.reshape(v1,(dim_lam,1)), np.matmul(dd2, Ai.reshape((dim_lam, dim_x))), np.zeros((dim_lam, dim_mu)), dd1))
        A = np.vstack((A0, A1, A2, A3))
    elif dim_lam == 0:
        A0 = np.array([0.0 for i in range(dim_x+dim_mu+1)])
        A0[0] = 1
        A1 = np.hstack((np.zeros((dim_x,1)), B, -Ae.T))
        A2 = np.hstack((np.zeros((dim_mu,1)), Ae.reshape((dim_mu, dim_x)), np.zeros((dim_mu, dim_mu))))
        A = np.vstack((A0, A1, A2))
    elif dim_mu == 0:
        A0 = np.array([0.0 for i in range(dim_x+dim_lam+1)])
        A0[0] = 1
        A1 = np.hstack((np.zeros((dim_x,1)), B, -Ai.T))
        A2 = np.hstack((np.reshape(v1,(dim_lam,1)), np.matmul(dd2, Ai.reshape((dim_lam, dim_x))), dd1))
        A = np.vstack((A0, A1, A2))
   
    return A 
   

def quadprog_smoothNewton(B, df, Ai, g, Ae, h, maxk=100):
    """ quadprog_smoothNewton solves the quadratic programming problem using the smoothing Newton method"""
    
    dim_x = np.size(df)
    dim_mu = np.size(h)
    dim_lam = np.size(g)
    
    # Initialization
    gamma = 0.05
    epsilon = 0.000001
    ep0 = 0.05
    u = np.array([0.0 for i in range(dim_x + dim_mu + dim_lam + 1)])
    u[0] = ep0
    
    k = 0
    d_k = np.array([1.0 for i in range(dim_x)])
    ep_k = 0.05
    mu_k = ep_k*np.array([1.0 for i in range(dim_mu)])
    lam_k = ep_k*np.array([1.0 for i in range(dim_lam)])
    # z_k = np.hstack((np.array([ep_k]), d_k, mu_k, lam_k))
    
    while k < maxk:
        
        dh = dah(ep_k, d_k, mu_k, lam_k, B, df, Ai, g, Ae, h)
        mp = np.linalg.norm(dh)
        if mp < epsilon:
            break
        
        # Calculating the Newton step for H(z) = 0
        A = JacobiH(ep_k, d_k, mu_k, lam_k, B, df, Ai, g, Ae, h)
        beta = gamma * (np.linalg.norm(dh)) * min(1, np.linalg.norm(dh))
        b = beta*u - dh
        dz = np.linalg.solve(A, b)
        if dim_mu > 0  and dim_lam >0:
            de = dz[0]
            dd = dz[1:dim_x+1]
            dmu = dz[dim_x+1:dim_x+dim_mu+1]
            dlam = dz[dim_x+dim_mu+1:]
        elif dim_lam == 0:
            de = dz[0]
            dd = dz[1:dim_x+1]
            dmu = dz[dim_x+1:]
            dlam = np.array([])
        elif dim_mu == 0:
            de = dz[0]
            dd = dz[1:dim_x+1]
            dlam = dz[dim_x+1:]
            dmu = np.array([])
            
        # Armijo linear serach
        rho = 0.5
        sigma = 0.2
        im = 0
        while im < 20:
            alpha = rho**im
            dh1 = dah(ep_k+alpha*de, d_k+alpha*dd, mu_k+alpha*dmu, lam_k+alpha*dlam, B, df, Ai, g, Ae, h)
            if np.linalg.norm(dh1) <= (1 - sigma*(1-gamma*ep0)*alpha)*np.linalg.norm(dh):
                mk = im
                break
            im += 1
            if im == 20:
                mk = 10
            
        # Updating the variables, including ep, d, mu, and lam
        alpha = rho**mk
        if dim_mu > 0 and dim_lam > 0:
            new_ep = ep_k + alpha*de
            new_d = d_k + alpha*dd
            new_mu = mu_k + alpha*dmu
            new_lam = lam_k + alpha*dlam
        elif dim_lam == 0:
            new_ep = ep_k + alpha*de
            new_d = d_k + alpha*dd
            new_mu = mu_k + alpha*dmu
            new_lam = np.array([])
        elif dim_mu == 0:
            new_ep = ep_k + alpha*de
            new_d = d_k + alpha*dd
            new_lam = lam_k + alpha*dlam
            new_mu = np.array([])
        
        ep_k = copy.deepcopy(new_ep)
        d_k = copy.deepcopy(new_d)
        mu_k = copy.deepcopy(new_mu)
        lam_k = copy.deepcopy(new_lam)
        
        k += 1
        
    val = 0.5*np.sum(d_k*(np.matmul(B, d_k))) + np.sum(d_k*df)
        
    return d_k, mu_k, lam_k, val
