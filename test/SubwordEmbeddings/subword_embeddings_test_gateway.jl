
using Serialization

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


@testset "BPE round-trip" begin
    enc = SWU.load_encoder("cl100k_base")
    ids = SWU.encode("hello world", enc)
    @test SWU.decode(ids, enc) == "hello world"
end


@testset "BPE vocab & ID range" begin
    enc  = SWU.load_encoder("cl100k_base")
    ids  = SWU.encode("JuliaLang is fast 🏎️", enc)

    # TikToken cl100k_base hard-codes 100 277 entries
    @test length(enc.vocab) == 100_277
    @test maximum(ids) < length(enc.vocab)
    @test minimum(ids) >= 0

    @test SWU.used_vocab_size(enc) ≤ length(enc.vocab)     # still true
    @test maximum(ids) < SWU.used_vocab_size(enc)
end


@testset "encode -> skip-gram -> window math" begin
    txt   = "hello world, hello Julia"
    ids   = SWU.encode(txt, SWU.load_encoder()) # sub-word IDs
    pairs = SWU.skipgram_pairs(ids, 1)     # radius = 1

    expected = 2length(ids) - 2              
    @test length(pairs) == expected

    @test first(pairs) == (ids[1],  ids[2])      # sanity
    @test last(pairs)  == (ids[end], ids[end-1])

    for (c, ctx) in pairs
        @test abs(findfirst(==(c), ids) - findfirst(==(ctx), ids)) ≤ 1
    end
end


@testset "BPE encoder persistence" begin
    enc = SWU.load_encoder()
    tmp = mktempdir()
    path = joinpath(tmp, "enc.bin")
    serialize(path, enc)

    enc2 = deserialize(path)
    str  = "sub-word embedding test"
    @test SWU.decode(SWU.encode(str, enc2), enc2) == str
    @test SWU.used_vocab_size(enc2) == SWU.used_vocab_size(enc)
end


@testset "TikToken vs GPT-2 encoders differ" begin
    tikt  = SWU.load_encoder("cl100k_base")
    gpt2  = SWU.load_encoder("gpt2")

    txt   = "hello world"
    ids_a = SWU.encode(txt, tikt)
    ids_b = SWU.encode(txt, gpt2)

    @test ids_a != ids_b
    @test SWU.decode(ids_a, tikt) == txt
    @test SWU.decode(ids_b, gpt2) == txt
end


@testset "multilingual round-trip" begin
    enc = SWU.load_encoder("cl100k_base")   # reuse official TikToken

    str = "Καλημέρα κόσμε, こんにちは世界 🌍"        # Greek + Japanese + emoji
    ids = SWU.encode(str, enc)

    @test isa(ids, Vector{Int})
    @test !isempty(ids)
    @test maximum(ids) < SWU.used_vocab_size(enc)  # stays within vocab

    @test SWU.decode(ids, enc) == str     #   round-trip
end


@testset "encoder save/load helper" begin
    enc   = SWU.load_encoder("cl100k_base")   # pre-trained

    tmp   = mktempdir()
    path  = joinpath(tmp, "tok.bin")
    SWU.save_encoder(path, enc)

    enc2  = SWU.load_encoder_from_file(path)      
    str   = "Καλημέρα κόσμε 🌍"
    @test SWU.decode(SWU.encode(str, enc2), enc2) == str
    @test SWU.used_vocab_size(enc2) == SWU.used_vocab_size(enc)
end
