using LinearAlgebra
using SparseArrays

import MathOptInterface as MOI
import Ipopt 

struct SparseProblemMOI <: MOI.AbstractNLPEvaluator
    n_nlp::Int
    m_nlp::Int
    obj_grad::Bool
    con_jac::Bool
    sparsity_jac
    hessian_lagrangian::Bool
    params::NamedTuple
    cost # ::Function
    cost_gradient!
    constraint! # ::Function
    constraint_jacobian!
    temp_con_jac
end


function SparseProblemMOI(n_nlp,
                          m_nlp,
                          params,
                          cost,
                          cost_gradient!,
                          constraint!,
                          constraint_jacobian!,
                          temp_con_jac)

    sparsity_jac = get_sparsity_pattern(temp_con_jac)
    obj_grad=true
    con_jac=true
    hessian_lagrangian=false

    SparseProblemMOI(
        n_nlp::Int,
        m_nlp::Int,
        obj_grad::Bool,
        con_jac::Bool,
        sparsity_jac,
        hessian_lagrangian::Bool,
        params::NamedTuple,
        cost, # ::Function
        cost_gradient!,
        constraint!, # ::Function
        constraint_jacobian!,
        temp_con_jac
    )
end


function MOI.eval_objective(prob::MOI.AbstractNLPEvaluator, x)
    prob.cost(prob.params, x)
end

function MOI.eval_objective_gradient(prob::MOI.AbstractNLPEvaluator, grad_f, x)
    prob.cost_gradient!(prob.params, grad_f, x)
    return nothing 
end

function MOI.eval_constraint(prob::MOI.AbstractNLPEvaluator,c,x)
    prob.constraint!(prob.params, c, x)
    # c .= prob.con(prob.params, x)
    return nothing
end

function MOI.eval_constraint_jacobian(prob::MOI.AbstractNLPEvaluator, jac, x)
    # evaluate the con jac and put it in prob.params.temp_jac 

    prob.constraint_jacobian!(prob.params, prob.temp_con_jac, x)
    # pull this out and put it in jac the vector 
    jac .= prob.temp_con_jac.nzval
    return nothing 
end

function MOI.features_available(prob::MOI.AbstractNLPEvaluator)
    return [:Grad, :Jac]
end

MOI.initialize(prob::MOI.AbstractNLPEvaluator, features) = nothing
MOI.jacobian_structure(prob::MOI.AbstractNLPEvaluator) = prob.sparsity_jac

function get_sparsity_pattern(jac)
    jac_rows, jac_cols, _ = findnz(jac)
    return [(r,c) for (r,c) in zip(jac_rows, jac_cols)]
end

function sparse_fmincon(cost::Function,
                        cost_gradient!::Function,
                        constraint!::Function,
                        constraint_jacobian!::Function,
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
                        verbose = true)::Vector
    
    n_primals = length(x0)
    n_con = length(c_l)
    
    # verbose && println("---------checking dimensions of everything----------")
    @assert length(x0) == length(x_l) == length(x_u)
    @assert length(c_l) == length(c_u) == n_con
    @assert all(x_u .>= x_l)
    @assert all(c_u .>= c_l)
    

    prob = SparseProblemMOI(n_primals,
                            n_con,
                            params,
                            cost,
                            cost_gradient!,
                            constraint!,
                            constraint_jacobian!,
                            temp_con_jac)

    # prob = SparseProblemMOI(n_primals, n_con, params, cost, constraint, constraint_jacobian!, temp_conjac)

    nlp_bounds = MOI.NLPBoundsPair.(c_l, c_u)
    block_data = MOI.NLPBlockData(nlp_bounds, prob, true)

    solver = Ipopt.Optimizer()
    solver.options["max_iter"] = max_iters
    solver.options["tol"] = tol
    solver.options["constr_viol_tol"] = c_tol
    
    if verbose 
        solver.options["print_level"] = 5
    else
        solver.options["print_level"] = 0 
    end

    x = MOI.add_variables(solver,prob.n_nlp)

    for i = 1:prob.n_nlp
        MOI.add_constraint(solver, x[i], MOI.LessThan(x_u[i]))
        MOI.add_constraint(solver, x[i], MOI.GreaterThan(x_l[i]))
        MOI.set(solver, MOI.VariablePrimalStart(), x[i], x0[i])
    end

    # Solve the problem
    verbose && println("---------IPOPT beginning solve----------------------")

    MOI.set(solver, MOI.NLPBlock(), block_data)
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.optimize!(solver)

    # Get the solution
    res = MOI.get(solver, MOI.VariablePrimal(), x)
    
    return res 
    
end
    

    
