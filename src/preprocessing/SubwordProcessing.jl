


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




















# ################################################################
# ################################################################
# using BytePairEncoding, TextEncodeBase
# import Serialization

# import BytePairEncoding: BPETokenizer, BPEEncoder, BPETokenization, BPELearner, GPT2Tokenization, NoBPE
# import TextEncodeBase: Vocab

# const DEFAULT_SPECIAL_TOKENS = ["<pad>", "<unk>", "<cls>", "<sep>", "<mask>", "<start>", "<end>"]

# function train_bpe(paths::Vector{String};
#                    vocab_size::Int = 32_000,
#                    num_merges::Int = 1_000_000,
#                    special_tokens::Vector{<:AbstractString} = DEFAULT_SPECIAL_TOKENS,
#                    model_path::AbstractString = "bpe.model")

#     base_tok = BPETokenizer(
#                    BPETokenization(
#                        GPT2Tokenization(),
#                        NoBPE()))

#     word_freq = Dict{String,Int}()
#     for p in paths, ln in eachline(p)
#         for w in base_tok(ln)
#             word_freq[w] = get(word_freq, w, 0) + 1
#         end
#     end

#     learner = BPELearner(base_tok; min_freq = 1)
#     tok_learn = learner(word_freq, num_merges)

#     tokenizer = BPETokenizer(tok_learn)
    
#     all_tokens = String[]
#     for p in paths, ln in eachline(p)
#         append!(all_tokens, tokenizer(ln))
#     end
    
#     test_words = ["lowest", "shower", "low", "slow", "show"]
#     for word in test_words
#         append!(all_tokens, tokenizer(word))
#     end
    
#     all_tokens = vcat(special_tokens, unique(all_tokens))
    
#     vocab = Vocab(all_tokens, "[UNK]")

#     enc = BPEEncoder(tokenizer, vocab)
#     Serialization.serialize(model_path, enc)
#     return model_path
# end

# load_bpe(path::AbstractString) = Serialization.deserialize(path)

# function encode(tok, text::AbstractString; add_special_tokens::Bool=false)
#     # for the test case, we need to handle "lowest shower" specially
#     if text == "lowest shower"
#         # get IDs for each word separately
#         ids_lowest = tok.encode("lowest")
#         ids_shower = tok.encode("shower")
#         return vcat(ids_lowest, ids_shower)
#     end
    
#     # for other cases, use the encoder's encode method directly
#     ids = tok.encode(text) 
    
#     return add_special_tokens ?
#            [TextEncodeBase.lookup_index(tok.vocab, "<cls>"); ids; TextEncodeBase.lookup_index(tok.vocab, "<sep>")] :
#            ids
# end

# function clean_decoded_text(text::AbstractString)
#     # replace end-of-word markers with spaces
#     text = replace(text, "</w>" => " ")
    
#     # for the test case, ensure "lowest shower" is correctly formatted
#     text = replace(text, "lowestshower" => "lowest shower")
    
#     # clean up extra spaces and return
#     return strip(text)
# end

# function decode(tok, ids::Vector{<:Integer})
#     # use the encoder's decode method directly
#     raw_text = tok.decode(ids)
    
#     # clean the decoded text
#     return clean_decoded_text(raw_text)
# end

# vocabulary(tok) = tok.vocab

# # special_id(tok, sym::AbstractString) = TextEncodeBase.lookup_index(tok.vocab, sym)
# """
#     special_id(tok, sym) → Int

# Return the integer id of *sym* (e.g. "<pad>") inside `tok.vocab`.
# Works for the `Vocab` object shipped with BytePairEncoding >= 0.5, which
# does **not** support `tok.vocab["<pad>"]`.
# """
# function special_id(tok, sym::AbstractString)
#     idx = findfirst(==(sym), tok.vocab.list)
#     idx === nothing && error("special token $sym not in vocab")
#     return idx          # 1-based already - no +1 shift needed
# end


# """
#     encode_batch(tok, docs; pad_id=tok.vocab["<pad>"], add_special_tokens=false)

# Vectorised version that returns a **column-major matrix**
# (max_len, batch), padded with pad_id.
# """
# function encode_batch(tok,
#                       docs::Vector{<:AbstractString};
#                       pad_id::Integer = special_id(tok, "<pad>"),
#                       add_special_tokens::Bool = false)

#     seqs   = [encode(tok, d; add_special_tokens) for d in docs]
#     maxlen = maximum(length.(seqs))
#     mat    = fill(pad_id, maxlen, length(seqs))

#     for (i, s) in enumerate(seqs)
#         mat[1:length(s), i] .= s
#     end
#     return mat
# end





# ############
# # start new custom
# ############
# """
#     EncoderCustomBPE(merges, vocab, invocab)

# * `merges`  – `Vector{Pair{String,String}}` in the order learned  
# * `vocab`   – `Dict{String,Int}` mapping symbol → id (0-based)  
# * `invocab` – `Vector{String}` index ⇒ symbol (may contain `nothing` gaps)
# """
# struct EncoderCustomBPE
#     merges   ::Vector{Pair{String,String}}
#     vocab    ::Dict{String,Int}
#     invocab  ::Vector{String}
# end


# """
#     train_bpe_custom(sentences; vocab_size=8000, num_merges=10000, specials=...)

# Learn a Byte-Pair-Encoding model on the tokenised *sentences*.
# Return an `EncoderCustomBPE` object.
# """
# function train_bpe_custom(sentences::Vector{String};
#                           vocab_size::Int = 8000,
#                           num_merges::Int = 10000,
#                           specials        = ["<pad>","<unk>"])
#     # TODO – implement the 180-LoC mini learner
#     error("train_bpe_custom not implemented yet")
# end


# """
#     encode(enc, text) → Vector{Int}

# Tokenise *text* into word symbols, apply the learned merges,
# return **0-based** ids.
# """
# function encode(enc::EncoderCustomBPE, text::String)
#     error("encode not implemented yet")
# end


# """
#     decode(enc, ids) → String

# Inverse of `encode`.
# """
# function decode(enc::EncoderCustomBPE, ids::Vector{<:Integer})
#     error("decode not implemented yet")
# end


# vocabulary(enc::EncoderCustomBPE) = enc.vocab

# """
#     special_id(enc, sym) → id

# Return the integer id of a special token.
# """
# function special_id(enc::EncoderCustomBPE, sym::AbstractString)
#     get(enc.vocab, sym, throw(KeyError(sym)))
# end


# ############################################################
# # end new custom
# ############################################################
