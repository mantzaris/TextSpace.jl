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


include("util-tests/__init__.jl")
include("pipeline/__init__.jl")


@testset "root test Hello Word!" begin
    text1 = "Hello, World!"
    @test text1 == "Hello, World!"
end
