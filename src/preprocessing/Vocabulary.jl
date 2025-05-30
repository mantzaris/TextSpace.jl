

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
    build_vocabulary_bpe(corpus; vocab_size=30_000, special_tokens=[])

Learn a Byte-Pair Encoding (BPE) vocabulary from `corpus`, a vector of
sentences (strings).  
Returns a `Dict` with keys:

* `token_to_index` - `Dict{String,Int}`
* `index_to_token` - `Vector{String}`
* `freq`           - frequency table of final sub-words
* `merges`         - list of merged pairs (for debugging)

Special tokens are inserted first (duplicates removed) and never merged.
`vocab_size` is the **maximum** final size **including** special tokens.
The algorithm stops once either (a) no more pairs occur >= 2 times or
(b) the target size is met.
"""
function build_vocabulary_bpe(
    corpus::AbstractVector{<:AbstractString};
    vocab_size::Int             = 30_000,
    special_tokens::Vector{String} = String[]
)
    #split into words, then into char tokens
    tokenized   = [split(sent) for sent in corpus]
    sentences   = Vector{Vector{Vector{String}}}(undef, length(tokenized))
    for (i, words) in pairs(tokenized)
        sentences[i] = [char_tokens(w) for w in words]   # ← helper we refactored earlier
    end

    #initialise bookkeeping
    merges = Vector{Tuple{String,String}}()
    specials = unique(special_tokens)  #remove dups, keep first order

    #main merge loop - stop when we hit vocab_size or run out of pairs
    while true
        pair_freq = count_pair_frequencies(sentences)
        isempty(pair_freq) && break

        # findmax gives (max_value, corresponding_key)
        best_count, best_pair = findmax(pair_freq)   # Int, Tuple

        best_count < 2 && break

        merge_pair_in_sentences!(sentences, best_pair)
        push!(merges, best_pair)

        n_syms = length(extract_all_symbols(sentences)) + length(specials)
        n_syms >= vocab_size && break
    end

    #collect final sub-word frequencies
    sub_freq = Dict{String,Int}()
    for sent in sentences, word in sent, tok in word
        sub_freq[tok] = get(sub_freq, tok, 0) + 1
    end

    #build index ↔ token arrays
    sorted = sort(collect(keys(sub_freq));
                  by = t -> sub_freq[t], rev = true)

    #respect vocab_size cap (after specials)
    remain = max(vocab_size - length(specials), 0)
    if length(sorted) > remain
        sorted = sorted[1:remain]
    end

    tok2id = Dict{String,Int}()
    id2tok = String[]

    for st in specials
        push!(id2tok, st)
        tok2id[st] = length(id2tok)
    end
    for sb in sorted
        push!(id2tok, sb)
        tok2id[sb] = length(id2tok)
    end

    return Dict(
        "token_to_index" => tok2id,
        "index_to_token" => id2tok,
        "freq"           => sub_freq,
        "merges"         => merges
    )
end



"""
    char_tokens(word::String; eos::Union{Nothing,String}=nothing)

Split `word` into a vector of **single-character *strings***.
If `eos` is given (e.g. `"</w>"`) it is appended as a boundary marker.
Multicode-point graphemes are kept whole.
"""
function char_tokens(word::AbstractString; eos=nothing)
    toks = [String(g) for g in Unicode.graphemes(word)]
    if eos !== nothing
        push!(toks, eos)
    end
    return toks
end


"""
    extract_all_symbols(sentences) -> Set{String}

Return the set of distinct sub-word tokens found in a nested structure
`sentences :: AbstractVector{<:AbstractVector{<:AbstractVector{String}}}`.
"""
function extract_all_symbols(sentences)::Set{String}
    syms = Set{String}()
    for sentence in sentences, word in sentence, tok in word
        push!(syms, tok)
    end
    return syms
end


"""
    count_pair_frequencies(sentences) -> Dict{Tuple{String,String},Int}

Return a frequency table of **adjacent token pairs _within each word_** for a
nested structure

sentences :: AbstractVector{<:AbstractVector{<:AbstractVector}}


The levels correspond to `[sentence][word][token]`.  If a word has fewer than
two tokens it contributes nothing.
"""
function count_pair_frequencies(sentences)::Dict{Tuple{String,String},Int}
    pair_freq = Dict{Tuple{String,String},Int}()

    for sentence in sentences, word in sentence
        n = length(word)
        if n < 2
            continue   # nothing to count
        end
        for i in 1:n-1
            pair = (word[i], word[i+1])
            pair_freq[pair] = get(pair_freq, pair, 0) + 1
        end
    end
    return pair_freq
end


"""
    merge_pair_in_sentences!(sentences, pair) -> nothing

In-place BPE merge: for every `word` inside every `sentence` replace each
*non-overlapping* occurrence of the adjacent token pair `pair =
(a, b)` with the new token `a*b` (string concatenation).

`sentences` must be a three-level nested container
`Vector{Vector{Vector{String}}}` (or any `AbstractVector` hierarchy with
the same element types).  The function mutates `sentences` and returns
`nothing`.
"""
function merge_pair_in_sentences!(
    sentences::AbstractVector,
    pair::Tuple{String,String},
)::Nothing
    a, b       = pair
    new_symbol = a * b

    for sentence in sentences, wi in eachindex(sentence)
        word   = sentence[wi]
        new_w  = String[]
        i, n   = 1, length(word)

        while i <= n
            if i < n && word[i] == a && word[i+1] == b
                push!(new_w, new_symbol)
                i += 2                    # skip the merged pair
            else
                push!(new_w, word[i])
                i += 1
            end
        end
        sentence[wi] = new_w
    end
    return nothing
end



"""
    build_vocabulary_wordpiece(corpus; vocab_size=30_000, special_tokens=[])

Greedy WordPiece vocabulary builder.

* `corpus`           - vector of sentences (strings)
* `vocab_size`       - maximum size **including** special tokens
* `special_tokens`   - e.g. `["<pad>", "<cls>", "<sep>"]`

Returns a `Dict` with keys `"token_to_index"`, `"index_to_token"`,
`"freq"`.  `[UNK]` is always inserted (deduplicated if already present).
"""
function build_vocabulary_wordpiece(
    corpus::AbstractVector{<:AbstractString};
    vocab_size::Int = 30_000,
    special_tokens::Vector{String} = String[]
)
    base_tokens = unique(vcat(special_tokens, "[UNK]"))   # dedup, keep order

    tok2id = Dict{String,Int}()
    id2tok = String[]
    for t in base_tokens
        push!(id2tok, t);  tok2id[t] = length(id2tok)
    end

    freq = Dict{String,Int}()

    # greedy discovery 
    for _pass in 1:5
        for sent in corpus, w in split(sent)
            i = firstindex(w)
            while i <= lastindex(w)
                # largest sub-token already in vocab
                found = false
                best  = ""
                j = lastindex(w)
                while j >= i
                    cand = w[i:j]
                    if haskey(tok2id, cand)
                        best   = cand;  found = true
                        freq[cand] = get(freq,cand,0) + 1
                        i = nextind(w, j)            # jump after match
                        break
                    end
                    j = prevind(w,j)
                end
                if !found
                    char_sub = string(w[i])
                    if length(tok2id) < vocab_size && !haskey(tok2id, char_sub)
                        push!(id2tok, char_sub);  tok2id[char_sub] = length(id2tok)
                    end
                    freq[char_sub] = get(freq,char_sub,0) + 1
                    i = nextind(w,i)
                end
            end
        end
        length(tok2id) >= vocab_size && break
    end

    #hold the most frequent newly-discovered tokens
    cap = vocab_size - length(base_tokens)
    new_sorted = sort(collect(keys(freq)); by = t->freq[t], rev = true)
    new_sorted = filter(t -> !(t in base_tokens), new_sorted)[1:min(cap,end)]

    tok2id  = Dict{String,Int}()
    id2tok  = String[]
    for t in base_tokens; push!(id2tok,t); tok2id[t]=length(id2tok); end
    for t in new_sorted;  push!(id2tok,t); tok2id[t]=length(id2tok); end

    return Dict("token_to_index"=>tok2id,
                "index_to_token"=>id2tok,
                "freq"=>freq)
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
function convert_tokens_to_ids(tokens::Vector{String},
                               vocab::Vocabulary;
                               add_new::Bool      = false,
                               update_counts::Bool = true)

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
    convert_ids_to_tokens(ids, vocab) → Vector{String}

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
                                kwargs...) → Matrix{Int}

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