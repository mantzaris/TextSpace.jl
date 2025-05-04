
const DEFAULTS = Dict(
    :unwrap_lines        => true,
    :paragraph_min_chars => 25,
    :sentence_split      => true,
    :lower               => true,
    :strip_punctuation   => true,
    :remove_stopwords    => false,
    :lemmatize           => false,
    :stem                => false,
    :ngram               => 1,
    :add_special_tokens  => true,
    :pad_value           => 0
)


"""
    process_document(text;
                     tok,                   # required SubwordTokenizer
                     vocab = nothing,       # optional Vocabulary (word-level)
                     return = :batch,       # :batch | :ids | :tokens | :sentences
                     kwargs...)

End-to-end preprocessing.

*Paragraph → Sentence splitting* happens automatically if `sentence_split=true`.  
The BPE tokenizer (`tok`) is mandatory when you request `:ids` or `:batch`.

`return` options  
| value        | output type                              |
|--------------|------------------------------------------|
| `:batch`     | padded matrix `(max_len, num_units)`     |
| `:ids`       | `Vector{Vector{Int}}`                    |
| `:tokens`    | `Vector{Vector{String}}`                 |
| `:sentences` | `Vector{String}` (after splitting)       |
"""
function process_document(text::AbstractString;
                          tok::SubwordTokenizer,
                          vocab::Union{Nothing,Vocabulary}=nothing;
                          return::Symbol = :batch,
                          kwargs...)

    opt = merge(DEFAULTS, Dict(kwargs))

    #Paragraph split  (and unwrap hard-wraps)
    paras = split_paragraphs(text;
                             unwrap  = opt[:unwrap_lines]) |>
            p -> filter_paragraphs(p; min_chars = opt[:paragraph_min_chars])

    #Sentence split (optional)  → sentences :: Vector{String}
    sentences = opt[:sentence_split] ?
                vcat([split_sentences(p) for p in paras]...) :
                paras

    return === :sentences && return sentences

    # Tokenise each sentence  -> Vector{Vector{String}}
    token_seqs = tokenize_batch(sentences;
                                strip_punctuation = opt[:strip_punctuation],
                                lower             = opt[:lower],
                                remove_stopwords  = opt[:remove_stopwords],
                                lemmatize         = opt[:lemmatize],
                                stem              = opt[:stem],
                                ngram             = opt[:ngram])

    return === :tokens && return token_seqs

    #Tokens -> ids  (BPE or word-level depending on args)
    if vocab === nothing          # BPE route
        id_seqs = [encode(tok, join(ts, ' ');
                           add_special_tokens = opt[:add_special_tokens]) for ts in token_seqs]
    else                          # word-level route
        id_seqs = [tokens_to_ids(ts, vocab) for ts in token_seqs]
    end

    return === :ids && return id_seqs

    # Pad to matrix  (sequence_length × batch)  ready for GPUs
    pad_val = vocab === nothing ? tok.pad_id : opt[:pad_value]
    batch   = pad_sequences(id_seqs; pad_value = pad_val)
    return batch
end


"""
    document_batch_iter(text;
                        tok,
                        vocab = nothing,
                        max_tokens = 256,
                        kwargs...)

Stream paragraph windows so each batch has ≤ `max_tokens` tokens
(according to the chosen tokenizer).  Useful for long docs.
Returns an iterator of padded matrices.
"""
function document_batch_iter(text; tok, vocab=nothing,
                             max_tokens::Int=256, kwargs...)

    # same paragraph splitting as above
    paras = split_paragraphs(text; unwrap=true) |>
            p -> filter_paragraphs(p; min_chars = 25)

    wind = paragraph_windows(paras, max_tokens;
                             stride = max_tokens,
                             tokenizer = p -> encode(tok, p))

    function _iter(state)
        chunk, nxt = iterate(wind, state)     # (Vector{String}, next_state)
        chunk === nothing && return nothing

        batch = process_document(join(chunk, "\n\n");
                                 tok, vocab;
                                 return = :batch,
                                 kwargs...)

        return batch, nxt
    end
    return _iter, 1
end

