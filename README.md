# lazy_nlp_qd.jl

Quick and dirty Ipopt interface with methods for both sparse and dense problems. 

This package exports the following two functions:
- `lazy_nlp_qd.fmincon_dense`
- `lazy_nlp_qd.fmincon_sparse`


## Install 
```julia 
import Pkg
Pkg.add(url="https://github.com/kevin-tracy/lazy_nlp_qd.jl.git")
```


## NLP Formats 

### Sparse

For sparse problems, we use the following NLP format:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad f(x) \\
\text{subject to} & \quad  x_L \leq x \leq x_U, \\
                  & \quad  c_L \leq c(x) \leq c_U,
\end{align*}
$$

where $f(x)$ is our cost and $c(x)$ is our constraint function. We can easily make equality constraints by setting the appropriate indices of $c_L[{\text{eqidx}}] =c_U[{\text{eqidx}}] $.

### Dense

For dense problems, we use the following NLP format:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad f(x) \\
\text{subject to} & \quad  x_L \leq x \leq x_U, \\
                  & \quad  c_{eq}(x) = 0, \\ 
                  & \quad  c_L \leq c_{ineq}(x) \leq c_U,
\end{align*}
$$


where $f(x)$ is our cost and $c_{eq}(x)$ is our equality constraint function, and $c_{ineq}(x)$ is our inequality constraint function.

## Examples 

- `test/fmincon_dense_test.jl` for a generic dense example
- `test/fmincon_sparse_test.jl` for a generic sparse example (this is what the quickstart is)
- `test/trajopt_test.jl` for a sparse trajopt example


## Quickstart (sparse)

Here we are going to solve the following QP:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{subject to} & \quad Ax = b, \\
                  & \quad Gx \leq h,
\end{align*}
$$

by first converting it into our accepted format:

$$
\begin{align*}
\underset{x}{\text{minimize}} & \quad \frac{1}{2}x^TQx + q^Tx \\
\text{subject to} & \quad 0 \leq Ax - b \leq 0 , \\
                  & \quad -\infty \leq Gx - h \leq 0,
\end{align*}
$$

```julia 
using Test
using LinearAlgebra
using SparseArrays 
import lazy_nlp_qd

let 
    nx = 30 
    ny = 10 
    nz = 20
    # check test/fmincon_sparse_test.jl for this function
    Q, q, A, b, G, h, x_solution = gen_sparse_qp(nx, ny, nz, 0.3)

    function my_cost(params, x)
        return .5*x'*params.Q*x + params.q'x 
    end

    function my_cost_gradient!(params, grad, x)
        grad .= params.Q*x + params.q 
        return nothing
    end

    function my_constraint!(params, cval, x)
        cval .= [params.A; params.G] * x - [params.b; params.h]
        return nothing 
    end

    function my_constraint_jacobian!(params, conjac, x)
        conjac .= sparse([params.A; params.G])
        return nothing 
    end
    
    # primal variable bounds 
    x_l = -Inf * ones(nx)
    x_u =  Inf * ones(nx)

    # constraint bounds 
    c_l = [zeros(ny); -Inf * ones(nz)]
    c_u = [zeros(ny); zeros(nz)]

    # sparse jacobian matrix with correct sparsity pattern
    temp_con_jac = sparse([A;G])
    
    # things I need for my functions 
    params = (
        Q = Q, 
        q = q, 
        A = A, 
        b = b, 
        G = G, 
        h = h
    )

    # initial guess
    x0 = .1*randn(nx)

    """ we are solving the following problem:
    
        minimize_x        cost(x)
        subject to        c_L <= constraint(x) <= c_U
                          x_L <= x <= x_U
    """
 
    x = lazy_nlp_qd.sparse_fmincon(my_cost::Function,
                                   my_cost_gradient!::Function,
                                   my_constraint!::Function,
                                   my_constraint_jacobian!::Function,
                                   temp_con_jac,
                                   x_l::Vector,
                                   x_u::Vector,
                                   c_l::Vector,
                                   c_u::Vector,
                                   x0::Vector,
                                   params::NamedTuple;
                                   tol = 1e-4,
                                   c_tol = 1e-4,
                                   max_iters = 1_000,
                                   print_level = 5)

    @test norm(x - x_solution, Inf) < 1e-3
end
```

## Quickstart (dense)

```julia
using Test
using LinearAlgebra
using SparseArrays 
import lazy_nlp_qd

let 
    nx = 30 
    ny = 10 
    nz = 20 
    Q, q, A, b, G, h, x_solution = gen_sparse_qp(nx, ny, nz, 1.0)
    Q = Matrix(Q)
    A = Matrix(A)
    G = Matrix(G)

    function my_cost(params, x)
        return .5*x'*params.Q*x + params.q'x 
    end

    function my_equality_constraint(params, x)
        return params.A*x - params.b 
    end

    function my_inequality_constraint(params, x)
        return params.G*x - params.h
    end

    # primal variable bounds 
    x_l = -Inf * ones(nx)
    x_u =  Inf * ones(nx)

    # constraint bounds (for the inequality constraint)
    c_l = -Inf * ones(nz)
    c_u = zeros(nz)

    # sparse jacobian matrix with correct sparsity pattern
    temp_con_jac = sparse([A;G])
    
    # things I need for my functions 
    params = (
        Q = Q, 
        q = q, 
        A = A, 
        b = b, 
        G = G, 
        h = h
    )

    # initial guess
    x0 = .1*randn(nx)

    diff_type = :auto # or :finite for finite diff 

    x = lazy_nlp_qd.fmincon_dense(my_cost::Function,
                                  my_equality_constraint::Function,
                                  my_inequality_constraint::Function,
                                  x_l::Vector,
                                  x_u::Vector,
                                  c_l::Vector,
                                  c_u::Vector,
                                  x0::Vector,
                                  params::NamedTuple,
                                  diff_type::Symbol;
                                  tol = 1e-4,
                                  c_tol = 1e-4,
                                  max_iters = 1_000,
                                  verbose = true)

    @test norm(x - x_solution, Inf) < 1e-3
end
```
