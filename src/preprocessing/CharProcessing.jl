
using Unicode


"""
    tokenize_char(text::String;
                  normalize  = true,
                  form       = :NFC,
                  keep_space = false,
                  lower      = false)  → Vector{String}

Return a `Vector{String}` where each element is one **grapheme cluster**.
Defaults:
* Unicode-normalise to NFC (`normalize=true`, `form=:NFC`)
* Removes spaces unless `keep_space=true`
* Optional lower-casing
"""
function tokenize_char(text::String;
                       normalize::Bool = true,
                       form::Symbol    = :NFC,
                       keep_space::Bool = false,
                       lower::Bool      = false)

    normalize && (text = Unicode.normalize(text, form))
    lower     && (text = lowercase(text))

    iter = Unicode.graphemes(text)   # respects combined emoji, accents, etc.
    chars = collect(iter)

    keep_space ? chars :
    filter(c -> !isspace(c[1]), chars)   # c is a String of 1-N codepoints
end


"""
    chars_to_ids(chars::Vector{String}, vocab; add_new=false) → Vector{Int}
"""
function chars_to_ids(chars::Vector{String},
                      vocab::Vocabulary;
                      add_new::Bool = false)

    out = Vector{Int}(undef, length(chars))
    for (i, ch) in enumerate(chars)
        id = get(vocab.token2id, ch, vocab.unk_id)
        if id == vocab.unk_id && add_new
            id = length(vocab.id2token) + 1
            vocab.token2id[ch] = id
            push!(vocab.id2token, ch)
        end
        out[i] = id
    end
    return out
end


"""
    encode_char_batch(texts::Vector{String}, vocab;
                      pad_value = vocab.unk_id,
                      kwargs...) → Matrix{Int}

One-liner that runs `tokenize_char` + `chars_to_ids` + `pad_sequences`.
"""
function encode_char_batch(texts::Vector{String},
                           vocab::Vocabulary;
                           pad_value::Int = vocab.unk_id,
                           kwargs...)

    seqs = [tokenize_char(t; kwargs...) for t in texts]
    idseqs = [chars_to_ids(s, vocab) for s in seqs]
    pad_sequences(idseqs; pad_value)
end

