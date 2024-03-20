using Test
using LinearAlgebra
using SparseArrays 

import lazy_nlp_qd
import ForwardDiff as FD 

import Random
Random.seed!(1234)

@testset "fmincon_sparse_tests" begin 
    include("fmincon_sparse_test.jl")
end

@testset "sparse trajopt test" begin 
    include("trajopt_test.jl")
end

@testset "fmincon_dense_tests" begin 
    include("fmincon_dense_test.jl")
end

