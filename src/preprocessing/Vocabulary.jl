

using JSON


struct Vocabulary
    token2id::Dict{String,Int}
    id2token::Vector{String}
    counts::Dict{Int,Int}
    unk_id::Int
end

Vocabulary() = Vocabulary(Dict{String,Int}(), String[], Dict{Int,Int}(), 0)



function build_vocabulary(
    tokens::Vector{String};
    min_freq::Int=0,
    max_vocab_size::Int=typemax(Int),
    special_tokens::Vector{String} = String[]
)
    #count frequencies
    freq = Dict{String, Int}()
    for token in tokens
        freq[token] = get(freq, token, 0) + 1
    end

    #filter by min_freq
    filtered_tokens = Dict{String, Int}()
    for (token, count) in freq
        if count >= min_freq
            filtered_tokens[token] = count
        end
    end

    #sort tokens by frequency (descending)
    sorted_tokens = sort(collect(keys(filtered_tokens)), 
                         by = t -> filtered_tokens[t], 
                         rev = true)

    #truncate if necessary
    if length(sorted_tokens) > max_vocab_size
        sorted_tokens = sorted_tokens[1:max_vocab_size]
    end

    #create token-to-index mapping, with special tokens first
    token_to_index = Dict{String, Int}()
    index_to_token = String[]

    
    # insert special tokens first, but only once each
    for st in special_tokens
        if !haskey(token_to_index, st)          #  guard
            push!(index_to_token, st)
            token_to_index[st] = length(index_to_token)
        end
    end

    #insert remaining tokens, skipping duplicates
    for tk in sorted_tokens
        if !haskey(token_to_index, tk)
            push!(index_to_token, tk)
            token_to_index[tk] = length(index_to_token)
        end
    end

    return Dict(
        "token_to_index" => token_to_index,
        "index_to_token" => index_to_token,
        "freq"           => freq
    )
end


"""
    convert_tokens_to_ids(tokens, vocab;
                          add_new=false,
                          update_counts=true) → Vector{Int}

Map string tokens to integer ids using `vocab::Vocabulary`.

* If `add_new=true` unknown tokens are appended to the vocabulary
  (needed while **training** the vocab).
* If `update_counts=true` increments `vocab.counts[id]` for every hit.
"""
function convert_tokens_to_ids(tokens::AbstractVector{<:AbstractString},
                               vocab::Vocabulary;
                               add_new::Bool      = false,
                               update_counts::Bool = true)

    @assert vocab.unk_id >= 1 "Vocabulary.unk_id must be a positive Int"

    out = Vector{Int}(undef, length(tokens))
    for (i, tok) in enumerate(tokens)

        id = get(vocab.token2id, tok, 0)        # 0 means OOV for now

        if id == 0
            if add_new
                id = length(vocab.id2token) + 1
                vocab.token2id[tok] = id
                push!(vocab.id2token, tok)
                update_counts && (vocab.counts[id] = 1)
            else
                id = vocab.unk_id
                update_counts && (vocab.counts[id] = get(vocab.counts, id, 0) + 1)
            end
        else
            update_counts && (vocab.counts[id] = get(vocab.counts, id, 0) + 1)
        end

        out[i] = id
    end
    return out
end


"""
    convert_ids_to_tokens(ids, vocab) -> Vector{String}

Inverse mapping. Unknown ids return `"<unk>"` by convention.
"""
function convert_ids_to_tokens(ids::Vector{<:Integer}, vocab::Vocabulary)
    unk = "<unk>"
    toks = Vector{String}(undef, length(ids))
    for (i, id) in enumerate(ids)
        toks[i] = 1 <= id <= length(vocab.id2token) ? vocab.id2token[id] : unk
    end
    return toks
end


"""
    convert_batch_tokens_to_ids(docs, vocab;
                                pad_value = vocab.unk_id,
                                kwargs...) -> Matrix{Int}

High-level helper: calls `convert_tokens_to_ids` on each document and
pads to a column-major matrix via `pad_sequences` (from TextVectorization.jl).
`docs` is a `Vector{Vector{String}}`.
"""
function convert_batch_tokens_to_ids(docs::Vector{Vector{String}},
                                     vocab::Vocabulary;
                                     pad_value::Int = vocab.unk_id,
                                     kwargs...)

    seqs = [convert_tokens_to_ids(d, vocab; kwargs...) for d in docs]
    pad_sequences(seqs; pad_value)
end


function save_vocabulary(vocab::Dict, filename::String)
    open(filename, "w") do io
        JSON.print(io, vocab)
    end
end

function load_vocabulary(filename::String)
    return JSON.parsefile(filename)
end