module CharTokenizer


import ..VocabularyCore: Vocabulary 

import ...Plumbing: tokenize_char
import ..TextVectorization: pad_sequences


export chars_to_ids, encode_char_batch

"""
    chars_to_ids(chars, vocab;
                 add_new       = false,
                 update_counts = true) -> Vector{Int}

Convert a vector of grapheme-cluster strings to their integer ids using the
`Vocabulary` object.

* `add_new = true` - unseen tokens are appended to the vocabulary.
* `update_counts` - when `true`, `vocab.counts[id] += 1`
  (or `1` if the entry was missing).

Backward-compatible: keyword names unchanged.

`Vocabulary` interface expected
---

vocab.token2id :: Dict{String,Int}
vocab.id2token :: Vector{String}
vocab.counts :: Dict{Int,Int}
vocab.unk_id :: Int

"""
function chars_to_ids(chars::AbstractVector{<:AbstractString},
                      vocab::Vocabulary;
                      add_new::Bool       = false,
                      update_counts::Bool = true)::Vector{Int}

    out = Vector{Int}(undef, length(chars))

    for (i, ch) in pairs(chars)
        id = get(vocab.token2id, ch, vocab.unk_id)

        if id == vocab.unk_id && add_new
            id = length(vocab.id2token) + 1
            vocab.token2id[ch] = id
            push!(vocab.id2token, String(ch))
            if update_counts
                vocab.counts[id] = 1  # first time seen
            end
        elseif update_counts
            vocab.counts[id] = get(vocab.counts, id, 0) + 1
        end

        out[i] = id
    end
    return out
end


"""
    encode_char_batch(tok_batch, vocab; eos="</w>", pad_value=vocab.unk_id)

`tok_batch` is `Vector{Vector{String}}`, eg the word tokens **per
sentence** that Pipeline already has.  We flatten each sentence to a
single character stream, append the EOS marker between words, map to ids
and finally pad.
"""
function encode_char_batch(tok_batch::Vector{Vector{String}},
                           vocab::Vocabulary;
                           eos::Union{String,Nothing}="</w>",
                           pad_value::Int = vocab.unk_id)

    #type declaration
    id_seqs = Vector{Vector{Int}}()
    
    for sentence_tokens in tok_batch
        # Join words in sentence
        sentence_str = join(sentence_tokens, "")
        
        # Split into individual characters
        chars = [string(ch) for ch in sentence_str]
        
        # Add EOS as single token
        if eos !== nothing
            push!(chars, eos)
        end
        
        # Convert to IDs
        ids = chars_to_ids(chars, vocab; add_new=false)
        push!(id_seqs, ids)
    end

    return pad_sequences(id_seqs; pad_value=pad_value)
end



end # module
