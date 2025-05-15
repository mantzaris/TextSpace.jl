module CharacterEmbeddings

using Flux, Random
using Statistics: mean 
using Random: randn


include(joinpath(@__DIR__, "CharacterEmbeddingUtilities", "__init__.jl"))
using .CharacterEmbeddingUtilities: build_char_pairs


export train!, embeddings, save_embeddings  


struct SkipGramModel
    emb :: Flux.Embedding          # (emb_dim, vocab_size)
end
Flux.@layer SkipGramModel


function SkipGramModel(vocab_size::Int, emb_dim::Int)
    w = randn(Float32, emb_dim, vocab_size) .* 0.01f0 #sigma≈0.01
    SkipGramModel(Flux.Embedding(w))
end

sigm(x) = 1 ./ (1 .+ exp.(-x))      

function sg_loss(m, pos_c, pos_o, neg_c, neg_o)
    l_pos = log.(sigm.(dot_scores(m, pos_c, pos_o)))
    l_neg = log.(sigm.(-dot_scores(m, neg_c, neg_o)))
    return -mean(vcat(l_pos, l_neg))
end


function dot_scores(m::SkipGramModel, centers, contexts)
    z_c = m.emb(centers)
    z_o = m.emb(contexts)
    return vec(sum(z_c .* z_o; dims = 1))
end

function train!(ids::Vector{Int}, vocab;
    objective::Symbol = :skipgram,
    emb_dim::Int = 128,
    radius::Int = 5,
    epochs::Int = 5,
    batch::Int = 1024,
    k_neg::Int = 5,
    lr = 1e-2,
    rng = Random.GLOBAL_RNG)

    @assert objective == :skipgram  "only :skipgram supported for now"

    centers, contexts = build_char_pairs(ids; mode=:skipgram, radius=radius, rng=rng)
    N          = length(centers)
    vocab_size = length(vocab.id2token)

    model = SkipGramModel(vocab_size, emb_dim)
    opt   = Flux.Adam(lr)

    # pre-allocate negative buffers
    neg_c = similar(centers, batch*k_neg)
    neg_o = similar(contexts, batch*k_neg)

    for epoch in 1:epochs
        for i in 1:batch:N
            j      = min(i+batch-1, N)
            pos_c  = centers[i:j]
            pos_o  = contexts[i:j]
    
            # negative sampling
            neg_o  = rand(rng, 1:vocab_size, length(pos_o)*k_neg)
            neg_c  = repeat(pos_c, k_neg)          # length(pos_o)*k_neg
    
            gs = Flux.gradient(()->sg_loss(model, pos_c, pos_o, neg_c, neg_o),
                               Flux.params(model))
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        @info "epoch $epoch finished"
    end
    return model
end


embeddings(m::SkipGramModel) = cpu(m.emb.weight)          # emb_dim times vocab

function save_embeddings(path, m::SkipGramModel, vocab)
    open(path, "w") do io
        for (tok, vec) in zip(vocab.id2token, eachcol(embeddings(m)))
            println(io, join((tok, vec...), '\t'))
        end
    end
end


end # module
