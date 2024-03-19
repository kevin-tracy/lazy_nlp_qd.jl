
function dynamics(params::NamedTuple, x::Vector, u)
    # cartpole ODE, parametrized by params. 

    # cartpole physical parameters 
    mc, mp, l = params.mc, params.mp, params.l

    g = 9.81
    
    q = x[1:2]
    qd = x[3:4]

    s = sin(q[2])
    c = cos(q[2])

    H = [mc+mp mp*l*c; mp*l*c mp*l^2]
    C = [0 -mp*qd[2]*l*s; 0 0]
    G = [0, mp*g*l*s]
    B = [1, 0]

    qdd = -H\(C*qd + G - B*u[1])
    xdot = [qd;qdd]
    return xdot 

end

# vanilla rk4 explicit integrator 
function rk4(params,x,u)
    dt = params.dt 
    k1 = dt * dynamics(params, x,        u)
    k2 = dt * dynamics(params, x + k1/2, u)
    k3 = dt * dynamics(params, x + k2/2, u)
    k4 = dt * dynamics(params, x + k3,   u)
    x + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
end

function create_idx(nx,nu,N)
    # This function creates some useful indexing tools for Z 
    # x_i = Z[idx.x[i]]
    # u_i = Z[idx.u[i]]
    
    # our Z vector is [x0, u0, x1, u1, …, xN]
    nz = (N-1) * nu + N * nx # length of Z 
    x = [(i - 1) * (nx + nu) .+ (1 : nx) for i = 1:N]
    u = [(i - 1) * (nx + nu) .+ ((nx + 1):(nx + nu)) for i = 1:(N - 1)]
    
    # constraint indexing for the (N-1) dynamics constraints when stacked up
    c = [(i - 1) * (nx) .+ (1 : nx) for i = 1:(N + 1)]
    nc = (N + 1) * nx # (N-1)*nx 
    
    return (nx=nx,nu=nu,N=N,nz=nz,nc=nc,x= x,u = u,c = c)
end

# user-defined functions for trajectory optimization 


function cost(params, Z)
    # return trajectory cost 
    Xref, Uref = params.Xref, params.Uref 
    idx = params.idx  
    Q, R, Qf = params.Q, params.R, params.Qf 

    J = 0.0 
    for i = 1:idx.N-1 
        dx = Z[idx.x[i]] - Xref[i]
        du = Z[idx.u[i]] - Uref[i] 
        J += 0.5*dx'*Q*dx + 0.5*du'*R*du 
    end
    
    dx = Z[idx.x[idx.N]] - Xref[idx.N]
    J += 0.5*dx'*Qf*dx 
    return J 
end

function cost_gradient!(params, grad, Z)
    # calculate cost gradient and put it in grad 
    grad .= FD.gradient(_Z -> cost(params, _Z), Z)
    return nothing 
end

function constraint!(params, cval, Z)
    # calculate constraint value and put it in cval 
    idx = params.idx 
    x0, xf = params.x0, params.xf 
    for i = 1:idx.N-1
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
        xi₊ = Z[idx.x[i+1]] 
        # you could put whatever integrator you wanted to here (implicit)
        cval[idx.c[i]] = xi₊ - rk4(params, xi,ui) 
    end
    cval[idx.c[idx.N]] = Z[idx.x[1]] - x0
    cval[idx.c[idx.N+1]] = Z[idx.x[idx.N]] - xf
    return nothing 
end

function constraint_jacobian!(params, conjac, Z)
    # calculate constraint jacobian and put it in conjac (a sparse matrix)
    idx = params.idx 

    for i = 1:idx.N-1
        xi = Z[idx.x[i]]
        ui = Z[idx.u[i]]
        xi₊ = Z[idx.x[i+1]] 
        
        # diff the following constraint wrt xi, ui, xi₊
        conjac[idx.c[i],idx.x[i]] .= -FD.jacobian(_x->rk4(params, _x,ui),xi)
        conjac[idx.c[i],idx.u[i]] .= -FD.jacobian(_u->rk4(params, xi,_u),ui)
        conjac[idx.c[i],idx.x[i+1]] .= I(idx.nx)
    end
    
    # diff the initial and terminal condition constraints wrt x1 and xN respectively
    conjac[idx.c[idx.N],idx.x[1]] .= I(idx.nx)
    conjac[idx.c[idx.N+1],idx.x[idx.N]] .= I(idx.nx)

    return nothing
end

let 

    # problem size 
    nx = 4 
    nu = 1 
    dt = 0.05
    tf = 2.0 
    t_vec = 0:dt:tf 
    N = length(t_vec)
    
    # LQR cost 
    Q = diagm(ones(nx))
    R = 0.1*diagm(ones(nu))
    Qf = 10*diagm(ones(nx))
    
    # indexing 
    idx = create_idx(nx,nu,N)
    
    # initial and goal states 
    x0 = [0, 0, 0, 0]
    xf = [0, pi, 0, 0]
    Xref = [1*xf for _ = 1:N]
    Uref = [zeros(1) for _ = 1:N-1]
    
    # load all useful things into params 
    params = (
        Q = Q,
        R = R,
        Qf = Qf, 
        x0 = x0, 
        xf = xf, 
        Xref = Xref, 
        Uref = Uref,
        dt = dt,
        N = N, 
        idx = idx,
        mc = 1.0, 
        mp = 0.2, 
        l = 0.5
    )
    
    # primal bounds 
    x_l = -Inf * ones(idx.nz)
    x_u =  Inf * ones(idx.nz)

    # constraint bounds 
    n_cons = (N + 1) * nx
    c_l = zeros(n_cons)
    c_u = zeros(n_cons)

    # initial guess 
    Z0 = .01*randn(idx.nz)

    # another way to initialize is 
    # for i = 1:N-1
    #     Z0[idx.x[i]] = # something 
    #     Z0[idx.u[i]] = # something 
    # end
    # Z[idx.x[N]] = # something 

    # evaluate the constraint jacobian once to get the sparsity structure 
    temp_con_jac = spzeros(n_cons, idx.nz)
    constraint_jacobian!(params, temp_con_jac, Z0)

    # call ipopt
    Z = lazy_nlp_qd.sparse_fmincon(cost::Function,
                                   cost_gradient!::Function,
                                   constraint!::Function,
                                   constraint_jacobian!::Function,
                                   temp_con_jac,
                                   x_l::Vector,
                                   x_u::Vector,
                                   c_l::Vector,
                                   c_u::Vector,
                                   Z0::Vector,
                                   params::NamedTuple;
                                   tol = 1e-4,
                                   c_tol = 1e-4,
                                   max_iters = 1_000,
                                   verbose = true)
end




        