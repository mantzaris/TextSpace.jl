
using LinearAlgebra
using Statistics
using Random


"""
    pad_sequences(seqs; maxlen = nothing, pad_value = 0, trunc = :post)

Return a column-major matrix where each column is a padded/truncated sequence.

* `maxlen`   — target length.  If `nothing` use the longest sequence.
* `pad_value`— value to insert in empty positions.
* `trunc`    — `:pre` or `:post` (keep head or tail when truncating).
"""
function pad_sequences(seqs::AbstractVector{<:AbstractVector{<:Integer}};
                       maxlen::Union{Nothing,Int}=nothing,
                       pad_value::Integer = 0,
                       trunc::Symbol = :post)

    isempty(seqs) && return Matrix{typeof(pad_value)}(undef, 0, 0)

    maxlen === nothing && (maxlen = maximum(length.(seqs)))
    m = fill(pad_value, maxlen, length(seqs))

    for (i, s) in pairs(seqs)
        if length(s) > maxlen                       # truncation
            trunc === :post ? (s = s[1:maxlen]) :
                               (s = s[end-maxlen+1:end])
        end
        m[1:length(s), i] .= s                      # copy
    end
    return m
end


"""
    one_hot(seq, vocab_size) -> Matrix{Bool}

Column-major binary matrix `(vocab_size, length(seq))`.
"""
function one_hot(seq::Vector{<:Integer}, vocab_size::Integer)
    vocab_size <= 0 && return BitMatrix(undef, 0, length(seq))  # 0xN
    
    mat = falses(vocab_size, length(seq))

    for (j, id) in enumerate(seq)
        1 <= id <= vocab_size && (mat[id, j] = true)
    end
    return mat
end


"""
    bow_counts(seq, vocab_size) -> Vector{Int}

Simple term-frequency vector (Bag-of-Words).
"""
function bow_counts(seq::AbstractVector{<:Integer}, vocab_size::Integer)
    vocab_size <= 0 && return Vector{Int}(undef, 0)   # graceful edge

    v = zeros(Int, vocab_size)
    for id in seq
        1 <= id <= vocab_size && (v[id] += 1)
    end
    return v
end


"""
    bow_matrix(docs, vocab_size) -> Matrix{Int}

Stack `bow_counts` column-wise for many documents.
"""
function bow_matrix(docs::AbstractVector{<:AbstractVector{<:Integer}},
                    vocab_size::Integer)
    # graceful exits
    isempty(docs)      && return Matrix{Int}(undef, 0, 0)
    vocab_size <= 0    && return Matrix{Int}(undef, 0, length(docs))

    mat = zeros(Int, vocab_size, length(docs))
    for (i, d) in pairs(docs)
        mat[:, i] .= bow_counts(d, vocab_size)
    end

    return mat
end


"""
    tfidf_matrix(docs, vocab_size; smooth_idf = 1.0) → Matrix{Float64}

Return a column-major TF-IDF matrix.  
`smooth_idf` ≥ 0 controls additive smoothing (set to 0 for raw IDF).
"""
function tfidf_matrix(docs::AbstractVector{<:AbstractVector{<:Integer}},
                      vocab_size::Integer;
                      smooth_idf::Real = 1.0)

    vocab_size ≤ 0      && return Matrix{Float64}(undef, 0, length(docs))
    isempty(docs)       && return Matrix{Float64}(undef, 0, 0)
    smooth_idf < 0      && throw(ArgumentError("smooth_idf must be ≥ 0"))

    tf  = Float64.(bow_matrix(docs, vocab_size))        # V × N
    df  = vec(sum(tf .> 0; dims = 2))                   # length V
    idf = log.((length(docs) .+ smooth_idf) ./ (df .+ smooth_idf))
    return tf .* idf                                    # broadcast
end


"""
    batch_iter(seqs, batch_size;
               shuffle     = true,
               pad_value   = 0,
               rng         = Random.GLOBAL_RNG)

Create an iterator that yields `(matrix, subset_indices)` pairs.

* Each `matrix` is a column-major padded batch (via `pad_sequences`).
* `subset_indices` is a `Vector{Int}` giving the original positions of the
  sequences (handy for back-mapping predictions).
* `batch_size` must be ≥ 1.  
  If `seqs` is empty the iterator terminates immediately.
"""
function batch_iter(seqs::AbstractVector{<:AbstractVector{<:Integer}},
                    batch_size::Integer;
                    shuffle::Bool   = true,
                    pad_value::Int  = 0,
                    rng::AbstractRNG = Random.GLOBAL_RNG)

    batch_size ≥ 1 ||
        throw(ArgumentError("batch_size must be ≥ 1, got $batch_size"))

    isempty(seqs) && return (_ -> nothing), 1            # 0×0 iterator

    idx = collect(eachindex(seqs))
    shuffle && Random.shuffle!(rng, idx)

    function _iterate(state)
        start = state
        start > length(idx) && return nothing            # exhausted

        stop  = min(start + batch_size - 1, length(idx))
        subset  = idx[start:stop]                        # vector of indices
        matrix  = pad_sequences(seqs[subset]; pad_value = pad_value)

        return (matrix, subset), stop + 1
    end

    return _iterate, 1
end


