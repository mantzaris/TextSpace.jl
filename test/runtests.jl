using TextSpace
using Test

using Random
using Downloads 

# include(joinpath(@__DIR__, "..", "src", "preprocessing", "Vocabulary.jl"))
# include(joinpath(@__DIR__, "..", "src", "preprocessing", "SubwordProcessing.jl"))

# include("SubwordEmbeddings/subword_embeddings_test_gateway.jl")
# include("WordEmbeddings/word_embeddings_test_gateway.jl")
# include("CharacterEmbeddings/character_embeddings_test_gateway.jl")
# include("preprocessing/preprocessing_test_gateway.jl")

# include("util-tests/__init__.jl")
include("pipeline/preprocessing_pipeline_tests.jl")



@testset "basic root test" begin
    # Test 1: Default behavior (no punctuation or emoji removal)
    text1 = "Hello, World!"
    #only lowercasing and whitespace normalization occur.
    @test text1 == "Hello, World!"
end
