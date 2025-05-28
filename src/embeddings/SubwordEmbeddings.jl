module SubwordEmbeddings

using Flux, Random, Statistics, LinearAlgebra, Zygote
using Zygote: Buffer
using AliasTables: AliasTable, sample, rand      
using StatsBase: countmap
import BytePairEncoding as BPE


include(joinpath(@__DIR__, "SubwordEmbeddingUtilities", "__init__.jl"))
const SWU = SubwordEmbeddingUtilities 

export train!, SkipGramModel, embeddings,
       save_embeddings, load_embeddings, vector,
       wordvec, each_token, nearest_words, nearest_tokens, tokenvec


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
                enc            = nothing,
                objective::Symbol = :skipgram,        # or :cbow
                encoder_name::String = "cl100k_base",
                emb_dim::Int  = 256,
                radius::Int   = 5,
                epochs::Int   = 5,
                batch::Int    = 2_048,
                k_neg::Int    = 5,
                lr            = 1e-3,
                rng           = Random.GLOBAL_RNG)

    @assert objective ∈ (:skipgram, :cbow)

    enc === nothing && (enc = SWU.load_encoder(encoder_name))

    # ------------------------------------------------------------------ 1
    ids   = vcat(SWU.encode.(corpus, Ref(enc))...)             # flatten
    pairs = objective === :skipgram ?
            SWU.skipgram_pairs(ids, radius) :
            SWU.cbow_pairs(ids,  radius)                       # = (centre, ctx)
    # ------------------------------------------------------------------

    vocabN = SWU.used_vocab_size(enc)          #  <-- fully–qualified
    model  = SkipGramModel(vocabN, emb_dim)
    opt    = Flux.Adam(lr)

    # negative–sampling table
    freqs   = countmap(ids)
    tok_ids = collect(keys(freqs))                      # Vector{Int}
    wts     = Float64.(values(freqs)).^0.75
    negtab  = AliasTable(wts)

    # positive tensors
    if objective === :skipgram
        posC, posO = first.(pairs), last.(pairs)        # ctr , ctx
    else
        posO, posC = first.(pairs), last.(pairs)        # centre , ctxs
    end
    N = length(posC)

    for epoch in 1:epochs
        for i in 1:batch:N
            j  = min(i+batch-1, N)

            if objective === :skipgram
                pc, po = posC[i:j], posO[i:j]
                nc     = repeat_vec(pc, k_neg)
                no     = tok_ids[rand(rng, negtab, length(po)*k_neg)]

                gs = gradient(() -> sg_loss(model, pc, po, nc, no),
                              Flux.params(model))

            else
                ctx_pos  = posC[i:j]                     # Vector{Vector{Int}}
                ctr_pos  = posO[i:j]
                ctx_neg  = vcat(Iterators.repeated(ctx_pos, k_neg)...)
                ctr_neg  = tok_ids[rand(rng, negtab, length(ctr_pos)*k_neg)]

                gs = gradient(() -> cbow_loss(model, ctx_pos, ctr_pos,
                                              ctx_neg,  ctr_neg),
                              Flux.params(model))
            end
            Flux.Optimise.update!(opt, Flux.params(model), gs)
        end
        @info "epoch $epoch finished"
    end
    return model, enc
end


function train_custom!(corpus::Vector{String};
                       enc::BPE.BPEEncoder,
                       objective::Symbol = :skipgram,
                       emb_dim::Int     = 256,
                       radius::Int      = 5,
                       epochs::Int      = 5,
                       batch::Int       = 2_048,
                       k_neg::Int       = 5,
                       lr               = 1e-3,
                       rng              = Random.GLOBAL_RNG)

    @assert objective ∈ (:skipgram, :cbow)

    raw_ids = vcat(SWU.encode.(corpus, Ref(enc))...) .+ 1    
    pairs   = objective === :skipgram ?
              SWU.skipgram_pairs(raw_ids, radius) :
              SWU.cbow_pairs(raw_ids,  radius)

    vocabN  = maximum(raw_ids)   # matrix needs up-to-max columns
    model   = SkipGramModel(vocabN, emb_dim)
    opt     = Flux.Adam(lr)

    freqs   = countmap(raw_ids)
    tok_ids = collect(keys(freqs))   # already base-1
    wts     = Float64.(values(freqs)).^0.75
    negtab  = AliasTable(wts)

    posC, posO = objective === :skipgram ? (first.(pairs), last.(pairs)) : (last.(pairs), first.(pairs))
    N = length(posC)

    for epoch in 1:epochs
        for i in 1:batch:N
            j  = min(i+batch-1, N)

            if objective === :skipgram
                pc, po = posC[i:j], posO[i:j]
                nc     = repeat_vec(pc, k_neg)
                no     = tok_ids[rand(rng, negtab, length(po)*k_neg)]
                gs = gradient(() -> sg_loss(model, pc, po, nc, no),
                              Flux.params(model))
            else
                ctx_pos = posC[i:j]; ctr_pos = posO[i:j]
                ctx_neg = vcat(Iterators.repeated(ctx_pos, k_neg)...)
                ctr_neg = tok_ids[rand(rng, negtab, length(ctr_pos)*k_neg)]
                gs = gradient(() -> cbow_loss(model, ctx_pos, ctr_pos,
                                              ctx_neg,  ctr_neg),
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



"""
    each_token(enc) -> iterator

Yield `(token::String, id::Int)` pairs for every defined entry in the
encoder's vocabulary, skipping internal #undef gaps.
"""
each_token(enc) =
    ( (enc.vocab.list[idx], idx)
      for idx in eachindex(enc.vocab.list) if isassigned(enc.vocab.list, idx) )


"""
    tokenvec(model, token_id) -> AbstractVector

Return the embedding column for the given BPE *token id*.
(This is mostly useful inside higher-level helpers.)
"""
tokenvec(m::SkipGramModel, id::Integer) = @view embeddings(m)[:, id]

"""
    wordvec(model, encoder, word::AbstractString) -> AbstractVector

Mean of the sub-token embeddings that spell `word`.
If the encoder splits the word into *n* pieces, the result is the 𝐥₂-mean
over those *n* columns.
"""
function wordvec(m::SkipGramModel, enc, word::AbstractString)
    ids = enc.encode(word)
    isempty(ids) && error("`$word` produced no sub-tokens with this encoder")
    mean(tokenvec(m, id) for id in ids)
end



        
"""
    nearest_tokens(model, encoder, query; k=5, α=0) -> Vector{Tuple{String,Float64}}

Return the `k` most-similar BPE tokens to the `query` token *string*.
If `α > 0`, tokens whose corpus frequency exceeds `α` median frequency
are skipped (handy to ignore 'the', 'and', etc.).
"""
function nearest_tokens(m, enc, query; k=5, α=0)
    vq   = wordvec(m, enc, query)                  # single-token or multi-token
    freqs = nothing
    if α > 0
        freqs = counts(enc.encode.(enc.decode.(each_token(enc)...)))
        medf  = median(values(freqs))
    end

    sims = Tuple{String,Float64}[]
    for (tok,id) in each_token(enc)
        α>0 && get(freqs, id, 0) > α*medf && continue
        push!(sims, (tok, dot(vq, tokenvec(m,id)) /
                          (norm(vq)*norm(tokenvec(m,id))+eps())))
    end
    sort!(sims; by=last, rev=true)[1:min(k,length(sims))]
end

"""
    nearest_words(model, encoder, word; k=5, α=0) -> Vector{Tuple{String,Float64}}

Human-friendly neighbour list: restrict to alphabetic tokens and optionally
prune super-frequent items (see `α` above).
"""
is_word(t) = occursin(r"^[A-Za-z]+$", t)

function nearest_words(m, enc, word; k=5, α=0)
    vq   = wordvec(m, enc, word)
    freqs = nothing
    if α>0
        freqs = counts(enc.encode.(enc.decode.(each_token(enc)...)))
        medf  = median(values(freqs))
    end

    sims = Tuple{String,Float64}[]
    for (tok,id) in each_token(enc)
        is_word(tok) || continue
        α>0 && get(freqs, id, 0) > α*medf && continue
        push!(sims, (tok, dot(vq, tokenvec(m,id)) /
                          (norm(vq)*norm(tokenvec(m,id))+eps())))
    end
    sort!(sims; by=last, rev=true)[1:min(k,length(sims))]
end






end