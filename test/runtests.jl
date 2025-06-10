using Test
using TextSpace
using Random
using Unicode
using Downloads 

@testset "TextSpace.jl Test Suite" begin
    @testset "Plumbing" begin
        include("preprocessing/__init__.jl")
    end

    @testset "Pipelines" begin
        include("pipeline/preprocessing_pipeline_tests.jl")
    end

    @testset "Basic Tests" begin
        @test true
    end

    # future testsets for embeddings and utils can follow the same structure
end
