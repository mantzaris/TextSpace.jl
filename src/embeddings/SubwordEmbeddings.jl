module SubwordEmbeddings

using Flux, Random, Statistics, LinearAlgebra, Zygote
using Zygote: Buffer
using AliasTables: AliasTable, sample, rand      
using StatsBase: countmap
import BytePairEncoding as BPE


include(joinpath(@__DIR__, "SubwordEmbeddingUtilities", "__init__.jl"))
const SWU = SubwordEmbeddingUtilities 

export train!, SkipGramModel, embeddings,
       save_embeddings, load_embeddings, vector


struct SkipGramModel
    emb::Flux.Embedding     # (d by |V|)
end
Flux.@layer SkipGramModel
embeddings(m::SkipGramModel) = cpu(m.emb.weight)
SkipGramModel(vN::Int, d::Int) =
    SkipGramModel(Flux.Embedding(randn(Float32, d, vN) .* 0.01f0))


sigm(x) = 1f0 ./ (1f0 .+ exp.(-x))
dot_scores(m, ctr, ctx) = vec(sum(m.emb(ctr) .* m.emb(ctx); dims = 1))

sg_loss(m, pc, po, nc, no) =
    mean(vcat(Flux.softplus.(-dot_scores(m, pc, po)),
              Flux.softplus.( dot_scores(m, nc, no))))

function cbow_scores(m::SkipGramModel,
                     ctxs::Vector{<:AbstractVector{Int}},
                     ctrs::Vector{Int})
    @assert length(ctxs) == length(ctrs)
    n   = length(ctrs)
    buf = Buffer(Vector{Float32}(undef, n))
    W   = embeddings(m)
    @inbounds for k in 1:n
        mu = mean(@view W[:, ctxs[k]]; dims = 2)[:, 1]
        buf[k] = dot(mu, @view W[:, ctrs[k]])
    end
    return copy(buf)
end

cbow_loss(m, pcx, pctr, ncx, nctr) =
    mean(vcat(Flux.softplus.(-cbow_scores(m, pcx, pctr)),
              Flux.softplus.( cbow_scores(m, ncx, nctr))))

# util
@inline function repeat_vec(src::AbstractVector{T}, k::Int) where T
    dest = Vector{T}(undef, length(src)*k)
    @inbounds for t in 0:k-1, i in eachindex(src)
        dest[t*length(src)+i] = src[i]
    end
    return dest
end

used_vocab_size(enc::BPE.BPEEncoder) = length(enc.vocab.list)


#   trainer           
function train!(corpus::Vector{String};
                objective::Symbol = :skipgram,       # or :cbow
                encoder_name::String = "cl100k_base",
                emb_dim::Int = 256, radius::Int = 5,
                epochs::Int = 5, batch::Int = 2048,
                k_neg::Int = 5, lr = 1e-3,
                rng = Random.GLOBAL_RNG)

    @assert objective in (:skipgram, :cbow)

    enc   = SWU.load_encoder(encoder_name)
    ids   = vcat(SWU.encode.(corpus, Ref(enc))...)
    pairs = objective === :skipgram ?
            SWU.skipgram_pairs(ids, radius) :
            SWU.cbow_pairs(ids, radius)

    vocabN = used_vocab_size(enc)
    model  = SkipGramModel(vocabN, emb_dim)
    opt    = Flux.Adam(lr)

    freqs    = countmap(ids)       # Dict(id => freq)
    tokens   = collect(keys(freqs))                 
    weights  = Float64.(values(freqs)).^0.75
    neg_tab  = AliasTable(weights)

    if objective === :skipgram
        posC, posO = first.(pairs), last.(pairs)
    else
        posO, posC = first.(pairs), last.(pairs)   # CBOW order
    end
    N = length(posC)

    for epoch in 1:epochs
        for i in 1:batch:N
            j  = min(i+batch-1, N)
            if objective === :skipgram
                pc, po = posC[i:j], posO[i:j]
                nc = repeat_vec(pc, k_neg)
                # draw indices with replacement
                idx = rand(rng, neg_tab, length(po) * k_neg)   # Vector{Int}

                # map indices back to token IDs
                no  = tokens[idx]

                gs = gradient(() -> sg_loss(model, pc, po, nc, no),
                              Flux.params(model))
            else
                ctx_vecs = posC[i:j]          # Vector{Vector{Int}}
                centres  = posO[i:j]          # Vector{Int}

                neg_ctx  = vcat(Iterators.repeated(ctx_vecs, k_neg)...)  # k_neg copies
                neg_ctr  = tokens[rand(rng, neg_tab, length(centres)*k_neg)]

                gs = gradient(() -> cbow_loss(model, ctx_vecs, centres, neg_ctx, neg_ctr),
                            Flux.params(model))
            end
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        @info "epoch $epoch finished"
    end
    return model, enc
end


# persistence
using Serialization

save_embeddings(path, m::SkipGramModel, enc) =
    Serialization.serialize(path, (emb = embeddings(m), enc = enc))

function load_embeddings(path)
    data = Serialization.deserialize(path)
    m = SkipGramModel(size(data.emb, 2), size(data.emb, 1))
    m.emb.weight .= data.emb
    return m, data.enc
end

vector(m::SkipGramModel, enc, tok::AbstractString) =
    @view embeddings(m)[:, enc.encode(tok)[1]]

    vector(m, enc, c::Char) = vector(m, enc, string(c))



end