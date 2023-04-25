#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 15:51:22 2021

@author: jhonhu
"""

import numpy as np
from run_SQP import nlp_solver_SQP


x_0 = np.array([3, 2])  # primal optimization valiable x
mu_0 = np.array([0])  # dual valiables associated with equality constraints, h_i(x) = 0
lam_0 = np.array([0,0])  # dual valiables associated with inequality constraints, g_i(x) >= 0

x_op, mu_op, lam_op, fval = nlp_solver_SQP(x_0, mu_0, lam_0, True)
