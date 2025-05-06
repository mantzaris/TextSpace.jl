using TextSpace
using Test

using Random


include(joinpath(@__DIR__, "..", "src", "preprocessing", "TextVectorization.jl"))
include(joinpath(@__DIR__, "..", "src", "preprocessing", "Vocabulary.jl"))
include(joinpath(@__DIR__, "..", "src", "preprocessing", "Tokenization.jl"))

#test stemming
include("preprocessing_stemming_tests.jl")

#test the textnormalization
include("preprocessing_textnormalization_tests.jl")

#test the vocabulary
include("preprocessing_vocaculary_tests.jl")

#test the text vectorization
include("preprocessing_textvectorization_tests.jl")

#test the text tokenization
include("preprocessing_tokenization_tests.jl")

#test preprocessing pipelines
include("preprocessing_pipeline_tests.jl")


@testset "basic tests" begin
    # Test 1: Default behavior (no punctuation or emoji removal)
    text1 = "Hello, World!"
    #only lowercasing and whitespace normalization occur.
    @test text1 == "Hello, World!"

end
