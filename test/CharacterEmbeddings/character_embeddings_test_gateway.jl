
using Random
using TextSpace.CharacterEmbeddings.CharacterEmbeddingUtilities: windowify,
      skipgram_pairs, build_skipgram_pairs

@testset "Character-Embedding utilities" begin
    ids = 1:10 |> collect

    # windowify
    w = windowify(ids, 4, 2)
    @test length(w) == 4 && w[1] == [1,2,3,4]

    # skipgram_pairs radius 1
    p = skipgram_pairs([1,2,3,4], 1)
    @test (2,1) in p && !( (1,3) in p )

    # build_skipgram_pairs
    c, o = build_skipgram_pairs(ids; win=4, stride=4, radius=1, rng = MersenneTwister(0))
    @test length(c) == length(o) == 12
    @test all(1 ≤ x ≤ 10 for x in c)
end
