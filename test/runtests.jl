using TextSpace
using Test

using Random
using Downloads 


# include("SubwordEmbeddings/subword_embeddings_test_gateway.jl")
# include("WordEmbeddings/word_embeddings_test_gateway.jl")
# include("CharacterEmbeddings/character_embeddings_test_gateway.jl")

# include("preprocessing/__init__.jl")
include("util_tests/__init__.jl")
# include("pipeline/__init__.jl")


@testset "root test Hello Word!" begin
    text1 = "Hello, World!"
    @test text1 == "Hello, World!"
end
