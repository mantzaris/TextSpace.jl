using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

const CE = TextSpace.CharacterEmbeddings
const V  = TextSpace.Preprocessing.Vocabulary

@testset "Character-Embeddings - training smoke-run" begin
    
    ids = rand(1:6, 600) # fake corpus with 6 chars
    tokmap = Dict("<unk>" => 1, "1"=>2, "2"=>3, "3"=>4, "4"=>5, "5"=>6, "6"=>7)
    id2tok = Vector{String}(undef, length(tokmap))
    for (tok, id) in tokmap
        id2tok[id] = tok
    end
    vocab = Vocabulary(tokmap, id2tok, Dict{Int,Int}(), 1)


    model = CE.train!(ids, vocab;
                      emb_dim = 16,
                      batch   = 128,
                      epochs  = 1,
                      k_neg   = 2,
                      lr      = 1e-1,
                      rng     = MersenneTwister(42))


    E = CE.embeddings(model)   # (16 × 7) matrix
    @test size(E) == (16, length(vocab.id2token))

    # embeddings should not all be identical after one update
    @test Statistics.std(vec(E)) > 0.0

    # saving works and file exists
    tmp = tempname()*".tsv"
    CE.save_embeddings(tmp, model, vocab)
    @test isfile(tmp)
    rm(tmp; force=true)
end


@testset "deterministic Skip-Gram" begin
    rng_pairs = MersenneTwister(42)
    ids       = rand(rng_pairs, 1:8, 2_000)

    #basic vocabulary 
    tok2id = Dict(string(i)=>i for i in 1:8);  tok2id["<unk>"] = 9
    id2tok = [string.(1:8)..., "<unk>"]
    vocab  = V(tok2id, id2tok, Dict{Int,Int}(), 9)

    # data 1 
    Random.seed!(123)
    m1 = CE.train!(ids, vocab;
                   epochs = 1,
                   lr     = 0.01,          # smaller LR, learning rate eta
                   batch  = 256,
                   k_neg  = 2,
                   rng    = copy(rng_pairs))

    # data 2 
    Random.seed!(123)
    m2 = CE.train!(ids, vocab;
                   epochs = 1,
                   lr     = 0.01,
                   batch  = 256,
                   k_neg  = 2,
                   rng    = copy(rng_pairs))

    @test isequal(CE.embeddings(m1), CE.embeddings(m2))    # deterministic
    @test !any(isnan, CE.embeddings(m1))                   # finite values
end



@testset "loss drops" begin
    #timy corpus  (1_000 tokens, ids 1-5)
    ids   = repeat(1:5, 200)   # Vector{Int} length 1000

    tok2id = Dict(string(i)=>i for i in 1:5)
    tok2id["<unk>"] = 6
    id2tok = [string.(1:5)..., "<unk>"]
    vocab  = V(tok2id, id2tok, Dict{Int,Int}(), 6)
    
    #initial random model, measure baseline loss
    init_model = CE.SkipGramModel(length(vocab.id2token), 16)
    l1 = CE.sg_loss(init_model,
                    [1], [2], repeat([1],5), rand(1:length(vocab.id2token), 5))

    #train for two epochs on the tiny corpus
    trained = CE.train!(ids, vocab;
                        epochs = 2,
                        emb_dim = 16,
                        lr     = 0.05,
                        batch  = 64,
                        rng    = MersenneTwister(1))

    #get the loss again on the trained model
    l2 = CE.sg_loss(trained,
                    [1], [2], repeat([1],5), rand(1:length(vocab.id2token), 5))

    @test l2 < l1  #loss should be going down
end


@testset "save / reload" begin
    #toy corpus & vocabulary
    ids   = repeat(1:3, 100)                     # 300 tokens, ids ∈ 1:3
    tok2id = Dict("1"=>1, "2"=>2, "3"=>3, "<unk>"=>4)
    id2tok = ["1","2","3","<unk>"]
    vocab  = V(tok2id, id2tok, Dict{Int,Int}(), 4)

    #train a  single-epoch Skip-Gram model
    model = CE.train!(ids, vocab; epochs = 1, emb_dim = 8, batch = 64)

    #save and reload the embedding matrix
    tmp = tempname()*".tsv"
    CE.save_embeddings(tmp, model, vocab)

    #read file: drop first column (token) and collect the vectors
    cols = (parse.(Float64, split(line, '\t')[2:end]) for line in eachline(tmp))
    E = hcat(cols...) |> Matrix # emb_dim × vocab_size

    @test E ≈ CE.embeddings(model)  # round-trip exact
    rm(tmp; force = true)
end



