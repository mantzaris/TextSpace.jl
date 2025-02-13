using TextSpace
using Test




@testset "clean_text tests" begin
    # Test 1: Default behavior (no punctuation or emoji removal)
    text1 = "Hello, World!   "
    #only lowercasing and whitespace normalization occur.
    @test clean_text(text1) == "hello, world!"

    #remove punctuation
    text2 = "Hello, World!   "
    # punctuation removed, the comma and exclamation point are dropped.
    @test clean_text(text2, remove_punctuation=true) == "hello world"

    # Test 3: Remove emojis
    text3 = "Hello 😊 World!   "
    @test clean_text(text3, remove_emojis=true) == "hello world!"

    # Test 4: Remove both punctuation and emojis
    text4 = "Hello, 😊 World!   How are you?"
    #removing both punctuation and emojis should result in just the words, separated by a single space.
    @test clean_text(text4, remove_punctuation=true, remove_emojis=true) == "hello world how are you"
end
