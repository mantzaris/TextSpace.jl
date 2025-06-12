using Test
using TextSpace
using Random
using Unicode
using Downloads 

@testset "TextSpace.jl Test Suite" begin
    @testset "Plumbing" begin
        include("preprocessing/__init__.jl")  # Loads all preprocessing tests
    end

    @testset "Pipelines" begin
        include("pipeline/__init__.jl")  # Now loads all pipeline tests uniformly
    end

    @testset "Basic Tests" begin
        @test true  # Your basic smoke tests
    end
end