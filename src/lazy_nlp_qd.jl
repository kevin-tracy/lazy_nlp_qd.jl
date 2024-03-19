__precompile__(true)
module lazy_nlp_qd


using LinearAlgebra
using SparseArrays
import MathOptInterface as MOI
import Ipopt 
# import ForwardDiff as FD 
# using StaticArrays

include("fmincon_sparse.jl")

export sparse_fmincon

end # module lazy_nlp_qd
