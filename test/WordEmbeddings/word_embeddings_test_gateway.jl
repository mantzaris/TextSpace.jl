
using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

# const WE = TextSpace.WordEmbeddings
# const V  = TextSpace.Preprocessing.Vocabulary
# const PP = TextSpace.Preprocessing


@testset "word embedding tests" begin
    # Test 1: Default behavior (no punctuation or emoji removal)
    text1 = "Hello, Word!"
    #only lowercasing and whitespace normalization occur.
    @test text1 == "Hello, Word!"

end