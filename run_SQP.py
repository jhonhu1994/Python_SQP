#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 16:03:17 2021

@author: jhonhu
"""

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import copy
from QP_solver import quadprog_smoothNewton
from problem_specified import fun, dfun, cons, dcons


def merit_l1(x, sigma):
    """ l1-merit function"""
    f = fun(x)
    h, g = cons(x)
    dim_h = np.size(h)
    dim_g = np.size(g)
    if dim_h > 0 and dim_g > 0:
        gn  = np.maximum(-g, 0)
        norm_h = np.linalg.norm(h, ord=1)
        norm_g = np.linalg.norm(gn, ord=1)
        merit_l1_value = f + 1.0/sigma*(norm_h + norm_g)
    elif dim_h == 0:
        gn  = np.maximum(-g, 0)
        norm_g = np.linalg.norm(gn, ord=1)
        merit_l1_value = f + 1.0/sigma*norm_g
    elif dim_g == 0:
        norm_h = np.linalg.norm(h, ord=1)
        merit_l1_value = f + 1.0/sigma*norm_h
    
    return merit_l1_value


def d_merit_l1(x, sigma, dx):
    """ Predicted reduction  of l1-merit function """
    df = dfun(x)
    h, g = cons(x)
    dim_h = np.size(h)
    dim_g = np.size(g)
    if dim_h > 0 and dim_g > 0:
        gn  = np.maximum(-g, 0)
        norm_g = np.linalg.norm(gn, ord=1)
        norm_h = np.linalg.norm(h, ord=1)
        d_merit_l1_value = np.sum(df*dx) - 1.0/sigma*(norm_h + norm_g)
    elif dim_h == 0:
        gn  = np.maximum(-g, 0)
        norm_g = np.linalg.norm(gn, ord=1)
        d_merit_l1_value = np.sum(df*dx) - 1.0/sigma*norm_g
    elif dim_g == 0:
        norm_h = np.linalg.norm(h, ord=1)
        d_merit_l1_value = np.sum(df*dx) - 1.0/sigma*norm_h
        
    return d_merit_l1_value


def dla(x, mu, lam):
    """ Derivative of Lagrangian function """
    df = dfun(x)
    Ae, Ai = dcons(x)
    dim_h = np.size(mu)
    dim_g = np.size(lam)
    if dim_h > 0 and dim_g > 0:
        Ae_T_mu = np.matmul(Ae.T, mu)
        Ai_T_lam = np.matmul(Ai.T, lam)
        gradient_of_la = df - Ae_T_mu - Ai_T_lam
    elif dim_h == 0:
        Ai_T_lam = np.matmul(Ai.T, lam)
        gradient_of_la = df - Ai_T_lam
    elif dim_g == 0:
        Ae_T_mu = np.matmul(Ae.T, mu)
        gradient_of_la = df - Ae_T_mu
        
    return gradient_of_la



def nlp_solver_SQP(x_k, mu_k, lam_k, log=False, maxIter=150):
    """ nlp_solver_SQP solves the constrained optimization problem using the sequential quadratic programming (SQP) method """

    # Initialization
    epsilon1 = 0.000001  # Optimality tolerance
    epsilon2 = 0.000001  # Step tolerance
    epsilon3 = 0.000001  # Constraint tolerance
    sigma = 0.8  # Initial penalty parameter in the l1-merit function
    ksi = 0.8  # Powell modification parameter

    k = 0
    dim_x = np.size(x_k)  # x -- primal optimization valiable
    dim_mu = np.size(mu_k)  # mu -- dual valiables associated with equality constraints, h_i(x) = 0
    dim_lam = np.size(lam_k)  # lam -- dual valiables associated with inequality constraints, g_i(x) >= 0
    B_k = np.eye(dim_x)
    df_k = dfun(x_k)
    h_k, g_k = cons(x_k)
    Ae_k, Ai_k = dcons(x_k)
    
    merit_l1_value = []
    
    while k < maxIter:

        # Solving the QP subproblem
        y_qp, mu_qp, lam_qp, _ = quadprog_smoothNewton(B_k, df_k, Ai_k, g_k, Ae_k, h_k)
        
        # Checking the stop criterion
        gradient_of_Lagrangian = dla(x_k, mu_k, lam_k)
        mp1 = np.linalg.norm(gradient_of_Lagrangian, ord=1)
        mp2 = np.linalg.norm(y_qp, ord=1)
        mp3 = np.linalg.norm(h_k, ord=1) + np.linalg.norm(np.maximum(-g_k, 0), ord=1)
        if mp3 < epsilon3:
            if mp1 < epsilon1 or mp2 < epsilon2:
                break
             
        # Updating the penalty parameter in the l1-merit function
        deta = 0.05
        if dim_mu > 0 and dim_lam > 0:
            tau = max(np.linalg.norm(mu_qp, ord=np.inf), np.linalg.norm(lam_qp, ord=np.inf))
        elif dim_mu == 0:
            tau = np.linalg.norm(lam_qp, ord=np.inf)
        elif dim_lam == 0:
            tau = np.linalg.norm(mu_qp, ord=np.inf)
        if sigma * (tau + deta) >= 1:
            sigma = 1.0/(tau + 2*deta)
          
        # Armijo linear serach
        eta = 0.1
        rho = 0.5
        merit_l1_value.append(merit_l1(x_k, sigma))
        if k < 8:
            im = 0
            while im < 20:
                x_im = x_k + (rho**im)*y_qp
                if (merit_l1(x_im, sigma) - merit_l1_value[k]) < eta*(rho**im)*d_merit_l1(x_k,sigma,y_qp):
                    mk = im
                    break
                im += 1
                if im == 20:
                    mk = 10
        else:  # Watchdog technique to relax the Maratos effect
            r = 3
            im = 0
            while im < 20:
                x_im = x_k + (rho**im)*y_qp
                if (merit_l1(x_im, sigma) - max(merit_l1_value[k+1-r:])) < eta*(rho**im)*d_merit_l1(x_k,sigma,y_qp):
                    mk = im
                    break
                im += 1
                if im == 20:
                    mk = 10
         
        alpha = rho**mk
        new_x = x_k + alpha*y_qp
             
        
        # Updating the relevant variables, including x_k, mu_k, lam_k, B_k, df_k, h_k, g_k, Ae_k, and Ai_k
        df_k = dfun(new_x)
        h_k, g_k = cons(new_x)
        Ae_k, Ai_k = dcons(new_x)
        A_k = np.vstack((Ae_k.reshape((dim_mu, dim_x)), Ai_k.reshape((dim_lam, dim_x))))
        
        # dualVariable = np.linalg.solve(A_k.T, df_k)
        dualVariable = np.matmul(np.linalg.pinv(A_k.T), df_k)
        if dim_mu > 0 and dim_lam > 0:
            mu_k = dualVariable[0:dim_mu]
            lam_k = dualVariable[dim_mu:]
        elif dim_mu == 0:
            lam_k = dualVariable
        elif dim_lam == 0:
            mu_k = dualVariable
        
        dx = alpha*y_qp
        y_k = dla(new_x, mu_k, lam_k) - dla(x_k, mu_k, lam_k)
        thre_curvature = alpha*(1-ksi)*np.sum(y_qp * np.matmul(B_k, y_qp))
        if np.sum(dx*y_k) >= thre_curvature:  # Powell modification
            z_k = y_k * 1.0
        else:
            dz = y_k - np.matmul(B_k, dx)
            theta = (thre_curvature - np.sum(dx*y_k)) / np.sum(dx*dz)
            z_k = y_k + theta*dz
        # thre_curvature = np.sum(dx * np.matmul(B_k, dx))
        # if np.sum(dx*y_k) >= (1-ksi)*thre_curvature:  # Powell modification
        #     z_k = y_k * 1.0
        # else:
        #     theta = ksi*thre_curvature/(thre_curvature-np.sum(dx*y_k))
        #     z_k = theta*y_k + (1-theta)*np.matmul(B_k, dx)
        zz = np.matmul(z_k[:, None], z_k[None, :])       
        Bd = np.matmul(B_k, dx)
        BB = np.matmul(Bd[:, None], Bd[None, :])
        new_B = B_k + zz/(np.sum(dx*z_k)) - BB/(np.sum(dx*Bd))  # BFGS formula
               
        
        x_k = copy.deepcopy(new_x)
        B_k = copy.deepcopy(new_B)
        
        k += 1
        
    val = fun(x_k)
    
    if log:
        plt.title("Change of the merit function at each iteration of SQP")
        plt.plot(merit_l1_value, 'bo')
        plt.show()
        
    return x_k, mu_k, lam_k, val
