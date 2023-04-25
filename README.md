# A Python Implementation of SQP

This GitHub repository hosts a Python version of the Sequential Quadratic Programming (SQP) method.


## Introduction of SQP
SQP methods and interior-point methods are two main representative approaches for solving constrained (smooth) nonlinear optimization problems. In general, SQP methods can handle the minimization of functions subject to both linear and nonlinear constraints, which may be written in the form

$$
\min_ {\mathbf{x}\in\mathbb{R}^n}\\;f(\mathbf{x})\qquad\qquad\qquad\qquad\qquad\\\\
\mbox{s.t.}\\;\\;h_ i(\mathbf{x})=0,\\;i\in\mathcal{E}=\{1,\cdots,l\},\\\\
\qquad\quad\quad h_j(\mathbf{x})\geq 0,\\;i\in\mathcal{I}=\{l+1,\cdots,m\},\tag{1}
$$

where $f(\cdot)$ and the functions $c_ i(\cdot)$ are all smooth, real-valued functions on a subset of $\mathbb{R}^ n$,  at least one of whom is nonlinear. By convention, we call $f$ the objective function, where $c_i,\\,i\in\mathcal{E}$ are the equality constraints and $c_ i,\\,i\in\mathcal{I}$  are the inequality constraints. SQP methods approach a stationary solution to (1) by solving a sequence of quadratic programming (QP) subproblems in which a quadratic model of the objective function is minimized subject to a linearization of the constraints. Specifically, at the $k$-th iteration, SQP methods involve solving a QP subproblem of the following form:

$$
\mathbf{x}_k\leftarrow\arg\,\min_ {\mathbf{x}\in\mathbb{R}^n}\;f(\mathbf{x}_ {k-1})+\nabla f(\mathbf{x}_ {k+1})^\mathrm{T}(\mathbf{x}-\mathbf{x}_ {k-1})+\frac{1}{2}(\mathbf{x}-\mathbf{x}_ {k-1})^\mathrm{T}\mathbf{H}_ {k}(\mathbf{x}-\mathbf{x}_ {k-1})
$$

$$
\mbox{s.t.}\;\;\nabla h_ i(\mathbf{x}_ {k-1})^\mathrm{T}(\mathbf{x}-\mathbf{x}_ {k-1})+h_ i(\mathbf{x}_ {k-1})=0,\;i\in\mathcal{E},\\
\nabla h_ i(\mathbf{x}_ {k-1})^\mathrm{T}(\mathbf{x}-\mathbf{x}_ {k-1})+h_ i(\mathbf{x}_ {k-1})\geq0,\;i\in\mathcal{I},\quad\tag{2}
$$

where $\mathbf{H}_ k$ is the Hessian matrix of the Lagrangian function for (1), i.e., $\mathcal{L}(\mathbf{x},\{\lambda\}_ {i=1}^m)=f(\mathbf{x})-\sum_ {i}\lambda_ ih_i(\mathbf{x})$. In practice, $\mathbf{H}_ k$ is usually approximated by a positive matrix $\mathbf{B}_ k$, e.g., quasi-Newton approximations.

Since Wilson proposed the ﬁrst SQP method in his 1963 PhD thesis, SQP methods have evolved considerably. This repository adopts the classical Han-Powell SQP method[[1]](#han)[[2]](#powell).  

## Usage

In this repository, the user can find:

- '_run_\__SQP.py_', which implements the outer iterative progress of the Han-Powell SQP method for solving (1).
-  '_QP\_solver.py_', a subroutine that will be needed in the '_run_\__SQP.py_', which solves the QP subproblem (2) using the smooth Newton method (the user can adopt an alternative QP solver, such as _qpsolvers.solve\_qp_ and _Cvxopt.solvers.qp_).
- '_problem\_specified.py_', in which the user should specify the optimization problem at hand, including the objective function $f(\mathbf{x})$ and its gradient, the vector of  $l$ equality constraint functions $h_ i(\mathbf{x}),\,i\in\mathcal{E}$ and the corresponding Jacobian matrix, and the vector of  $m-l$ inequality constraint functions $h_ i(\mathbf{x}),\,i\in\mathcal{I}$ and the corresponding Jacobian matrix (Three simple examples has been presented to illustrate the usage).
- '_demo.py_', which runs one of the three examples.

## Environment
In order to run the code in this repository the following software packages are needed:
* `Python 3` (for reference we use Python 3.6.8), with the following packages:`numpy`, `matplotlib`, `copy`, `math`.


## Reference

<a id='han'></a> [1] Han S P. A globally convergent method for nonlinear programming, J. Optim. Theory Appl., 22 (1977), pp. 297–309.  

<a id='powell'></a> [2] A fast algorithm for nonlinearly constrained optimization calculations, in Numerical Analysis, Dundee 1977, G.A. Watson, ed., no. 630 in Lecture Notes in Mathematics, Heidelberg, Berlin, New York, 1978, Springer Verlag, pp. 144–157.  
