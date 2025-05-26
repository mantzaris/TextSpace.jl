
using Serialization, Random, Statistics, AliasTables
using Zygote, Flux
using StatsBase: countmap
using AliasTables: AliasTable, rand

const SWU = TextSpace.SubwordEmbeddings.SubwordEmbeddingUtilities
const SWE = TextSpace.SubwordEmbeddings

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


@testset "subword sgns smoke" begin
    corpus = ["hello world", "hello Julia"]
    m, enc = SubwordEmbeddings.train!(corpus; epochs=1, batch=4, emb_dim=32)
    v = SubwordEmbeddings.vector(m, enc, "hello")
    @test length(v) == 32
end


#CBOW 1-epoch smoke-test
@testset "subword CBOW smoke" begin
    corpus = ["flux is amazing", "hello flux"]
    m, enc = SubwordEmbeddings.train!(corpus;
                                      objective   = :cbow,
                                      epochs      = 1,
                                      batch       = 4,
                                      emb_dim     = 32,
                                      k_neg       = 2)
    # vectors have expected length
    v = SubwordEmbeddings.vector(m, enc, "flux")
    @test length(v) == 32
end


#Embedding matrix vs vocab size
@testset "embedding rows = used vocab size" begin
    enc = SWU.load_encoder("gpt2")
    vN  = SWU.used_vocab_size(enc)
    m   = SubwordEmbeddings.SkipGramModel(vN, 16) #tiny dim

    @test size(SubwordEmbeddings.embeddings(m), 2) == vN
end


#save_embeddings / load_embeddings round-trip
@testset "embedding save/load round-trip" begin
    corpus = ["tiny corpus"]
    m, enc  = SubwordEmbeddings.train!(corpus; epochs=1, batch=2, emb_dim=8)

    tmp  = mktempdir()
    file = joinpath(tmp, "sub_emb.bin")
    SubwordEmbeddings.save_embeddings(file, m, enc)

    m2, enc2 = SubwordEmbeddings.load_embeddings(file)
    tok = "tiny"
    @test SubwordEmbeddings.vector(m2, enc2, tok) ≈
          SubwordEmbeddings.vector(m,  enc,  tok)
end


@testset "vector() unknown-token still encodes" begin
    enc = SWU.load_encoder()
    m   = SubwordEmbeddings.SkipGramModel(SWU.used_vocab_size(enc), 12)

    vec_known = SubwordEmbeddings.vector(m, enc, "hello")
    vec_unk   = SubwordEmbeddings.vector(m, enc, "NON_EXISTENT_TOKEN_XYZ")

    @test length(vec_known) == 12 == length(vec_unk)   # correct size
    @test vec_known != vec_unk      # returns *some* unk vector
end


@testset "skip-gram radius-2 window maths" begin
    txt  = "one two three four five six"
    ids  = SWU.encode(txt, SWU.load_encoder("gpt2"))
    pairs = SWU.skipgram_pairs(ids, 2)       # radius = 2

    n = length(ids); r = 2
    expected = 2n*r - r*(r+1)              
    @test length(pairs) == expected

    # every pair distance ≤ r
    for (c, ctx) in pairs
        @test abs(findfirst(==(c), ids) -
                  findfirst(==(ctx), ids)) <= r
    end
end


@testset "cbow radius-3 context length" begin
    ids = SWU.encode("a b c d e f g h", SWU.load_encoder())
    tuples = SWU.cbow_pairs(ids, 3)
    for (ctr, ctx) in tuples
        @test length(ctx) <= 3*2    # left+right
        @test !(ctr in ctx)
    end
end


@testset "1-batch SGNS loss drop" begin
    corpus = ["good bad good bad", "bad good"]
    enc    = SWU.load_encoder()
    ids    = vcat(SWU.encode.(corpus, Ref(enc))...)
    pairs  = SWU.skipgram_pairs(ids, 2)

    pc, po = first.(pairs), last.(pairs)
    vocabN = SWU.used_vocab_size(enc)
    m      = SWE.SkipGramModel(vocabN, 16)

    # tiny negative set just for test
    freqs  = countmap(ids); toks = collect(keys(freqs))
    tbl    = AliasTables.AliasTable(Float64.(values(freqs)).^0.75)
    nc     = repeat(pc, 2)
    no     = toks[rand(tbl, length(pc)*2)]

    loss_before = SWE.sg_loss(m, pc, po, nc, no)
    gs = Zygote.gradient(() -> SWE.sg_loss(m, pc, po, nc, no),
                         Flux.params(m))
    Flux.Optimise.update!(Flux.Adam(1e-2), Flux.params(m), gs)
    loss_after  = SWE.sg_loss(m, pc, po, nc, no)

    @test loss_after < loss_before    # one gradient step helped
end


#model -> encoder save -> load round-trip
@testset "subword model save/load" begin
    corpus = ["abc def ghi", "def ghi jkl"]
    m, enc = SWE.train!(corpus; epochs=1, batch=4, emb_dim=8)

    tmp  = mktempdir()
    file = joinpath(tmp, "subword.bin")
    SWE.save_embeddings(file, m, enc)

    m2, enc2 = SWE.load_embeddings(file)
    str = "def"
    @test SWE.vector(m2, enc2, str) ≈ SWE.vector(m, enc, str)
end

