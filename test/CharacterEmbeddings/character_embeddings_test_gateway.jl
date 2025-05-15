using Test, Random
using TextSpace                  # pulls in CharacterEmbeddings
using Statistics

const CE = TextSpace.CharacterEmbeddings

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
