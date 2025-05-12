using TextSpace
using Test

using Random
using Downloads 


include("CharacterEmbeddings/character_embeddings_test_gateway.jl")

# include("preprocessing/preprocessing_test_gateway.jl")


@testset "basic tests" begin
    # Test 1: Default behavior (no punctuation or emoji removal)
    text1 = "Hello, World!"
    #only lowercasing and whitespace normalization occur.
    @test text1 == "Hello, World!"

end
