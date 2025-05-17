
using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

const WE = TextSpace.WordEmbeddings
const V  = TextSpace.Preprocessing.Vocabulary  

make_vocab(N) = let
    tok2id = Dict("<unk>" => 1)
    for i in 1:N
        tok2id[string(i)] = i + 1
    end
    id2tok = Vector{String}(undef, length(tok2id))
    for (tok, id) in tok2id
        id2tok[id] = tok
    end
    V(tok2id, id2tok, Dict{Int,Int}(), 1)
end



@testset "Word-Embeddings - training smoke-run" begin
    vocab  = make_vocab(6)                       # ids 2…7 are "words"
    ids    = rand(2:7, 600)                      # fake corpus (no OOV)

    model = WE.train!(ids, vocab;
                      objective = :skipgram,     # or :cbow
                      emb_dim   = 16,
                      batch     = 128,
                      epochs    = 1,
                      k_neg     = 2,
                      lr        = 1e-1,
                      rng       = MersenneTwister(42))

    E = WE.embeddings(model)                     # 16 × |V|
    @test size(E) == (16, length(vocab.id2token))
    @test std(vec(E)) > 0                        # not all identical / zero
    @test all(isfinite, E)

    # round-trip through disk
    tmp = tempname() * ".tsv"
    WE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)

    model2, vocab2 = WE.load_embeddings(tmp)
    @test E ≈ WE.embeddings(model2)              # identical weights
    @test vocab.token2id == vocab2.token2id      # identical vocab mapping
    rm(tmp; force = true)
end


"helper: build a toy word vocabulary with <unk>=1"
make_vocab(N) = let
    tok2id = Dict("<unk>" => 1)
    for i in 1:N
        tok2id[string(i)] = i + 1
    end
    id2tok = Vector{String}(undef, length(tok2id))
    for (tok, id) in tok2id
        id2tok[id] = tok
    end
    V(tok2id, id2tok, Dict{Int,Int}(), 1)       # 1 is the unk-id
end

@testset "deterministic Skip-Gram" begin
    rng_pairs = MersenneTwister(77)
    ids       = rand(rng_pairs, 2:7, 1_000)    # corpus
    vocab     = make_vocab(6)

    Random.seed!(123)
    m1 = WE.train!(ids, vocab; epochs=1, rng = copy(rng_pairs))

    Random.seed!(123)
    m2 = WE.train!(ids, vocab; epochs=1, rng = copy(rng_pairs))

    @test isequal(WE.embeddings(m1), WE.embeddings(m2))
end

@testset "SG loss drops" begin
    ids   = repeat(2:5, 250)                 # small synthetic corpus
    vocab = make_vocab(4)

    # fixed probe pairs
    C,O   = WE.WordEmbeddingUtilities.build_word_pairs(
                ids; mode=:skipgram, win=32, stride=32, radius=2,
                rng=MersenneTwister(9))
    C, O  = C[1:128], O[1:128]

    init  = WE.SkipGramModel(length(vocab.id2token), 16)
    l0    = WE.sg_loss(init, C, O, repeat(C,5), rand(1:length(vocab.id2token),length(O)*5))

    trained = WE.train!(ids, vocab; epochs=5, emb_dim=16, batch=256)
    l1 = WE.sg_loss(trained, C, O, repeat(C,5), rand(1:length(vocab.id2token),length(O)*5))

    @test l1 < l0
end

@testset "OOV handled as <unk>" begin
    vocab = make_vocab(6)                    #  <unk> has id = 1
    m     = WE.SkipGramModel(length(vocab.id2token), 8)

    # both unknown tokens map to the <unk> column
    unk_vec = WE.vector(m, vocab, "<unk>")
    @test WE.vector(m, vocab, "not-in-vocab") === unk_vec
    @test WE.vector(m, vocab, "💥")           === unk_vec
end


@testset "nearest-neighbour sanity" begin
    using LinearAlgebra: dot, norm            # local import

    vocab = make_vocab(5)
    ids   = repeat(2:6, 200)
    m     = WE.train!(ids, vocab; epochs = 1, emb_dim = 8)

    E   = WE.embeddings(m)
    cos(a, b) = dot(a, b) / (norm(a) * norm(b) + eps())

    sims = [cos(E[:, 2], E[:, j]) for j in 1:size(E, 2)]   # token “1”
    @test argmax(sims) == 2                                # self-similarity max
end

