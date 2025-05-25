

const SWU = TextSpace.SubwordEmbeddings.SubwordEmbeddingUtilities

@testset "windowify simple" begin
    toks = 1:5

    expected_ctx = [
        (1, [2]),
        (2, [1,3]),
        (3, [2,4]),
        (4, [3,5]),
        (5, [4])
    ]
    @test collect(SWU.windowify(toks; window_size=1)) == expected_ctx

    sg_pairs = collect(SWU.windowify(toks; window_size=1, as_pairs=true))
    @test length(sg_pairs) == 8
    @test first(sg_pairs)  == (1,2)
    @test last(sg_pairs)   == (5,4)
end



@testset "Sub-word Skip-Gram / CBOW wrappers" begin
    ids = 1:5 # toy token stream
    radius = 1

    #  Skip-Gram
    sg = SWU.skipgram_pairs(ids, radius)
    @test length(sg) == 8
    @test first(sg)  == (1,2)
    @test last(sg)   == (5,4)

    #  CBOW
    cb = SWU.cbow_pairs(ids, radius)
    expected = [
        (1, [2]),
        (2, [1,3]),
        (3, [2,4]),
        (4, [3,5]),
        (5, [4])
    ]
    @test cb == expected
end