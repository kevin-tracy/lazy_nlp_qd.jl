
function gen_sparse_qp(nx, ny, nz, fill_ratio)
    Q = sprandn(nx, nx, fill_ratio)
    Q = Q'*Q + I 

    A = sprandn(ny, nx, fill_ratio)
    A[1:ny,1:ny] += I 

    G = sprandn(nz, nx, fill_ratio)

    x = randn(nx)
    y = randn(ny)

    s = zeros(nz)
    z = zeros(nz)

    for i = 1:nz
        if randn() < 0 
            s[i] = abs(randn())
        else
            z[i] = abs(randn())
        end
    end

    b = A * x 
    h = G * x + s 

    q = -(Q*x + A'*y + G'*z)

    @assert norm(Q*x + q + A'*y + G'*z) < 1e-10 
    @assert abs(s'*z) < 1e-10 
    @assert minimum(s) >= 0 
    @assert minimum(z) >= 0 
    @assert norm(A*x - b) < 1e-10 
    @assert norm(G*x + s - h) < 1e-10 
    println("created sparse QP successfully")

    Q, q, A, b, G, h, x
end

let 


    nx = 30 
    ny = 10 
    nz = 20 
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
