using Test
using LinearAlgebra
using SparseArrays 

import lazy_nlp_qd

import Random
Random.seed!(1234)

@testset "fmincon_sparse_tests" begin 
    include("fmincon_sparse_test.jl")
end

# right now I can't export this without overwriting 
# a bunch of custom MOI methods that are specific 
# to sparse/dense. I need to figure out a way around.
# @testset "fmincon_dense_tests" begin 
#     include("fmincon_dense_test.jl")
# end

