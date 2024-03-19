using Test
using LinearAlgebra
using SparseArrays 

import lazy_nlp_qd

import Random
Random.seed!(1234)

@testset "fmincon_sparse_tests" begin 
    include("fmincon_sparse_test.jl")
end

