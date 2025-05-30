
using Unicode


"""
    tokenize_char(text;
                  normalize   = true,
                  form        = :NFC,
                  keep_space  = false,
                  lower       = false) -> Vector{String}

Split *text* into a vector of **Unicode grapheme clusters**.

Keyword flags — identical to the original API
---------------------------------------------
* `normalize`  - when `true` canonical-normalise to the given `form`
                 (`:NFC`, `:NFD`, `:NFKC`, `:NFKD`).
* `keep_space` - include space-like graphemes (`isspace`) in the output.
* `lower`      - convert *text* to lowercase **after** normalisation.

The function accepts any `AbstractString` and always returns
`Vector{String}`.  It raises `ArgumentError` on an unsupported
normal-form symbol and works on Julia 1.6 -> 1.11.

Idempotence: `join(tokenize_char(txt; keep_space=true)) == txt`.
"""
function tokenize_char(text::AbstractString;
                       normalize::Bool      = true,
                       form::Symbol         = :NFC,
                       keep_space::Bool     = false,
                       lower::Bool          = false)::Vector{String}

    normalize && (form in (:NFC, :NFD, :NFKC, :NFKD) ||
                  throw(ArgumentError("Unsupported Unicode form $form"));
                  text = Unicode.normalize(text, form))

    lower      && (text = lowercase(text))

    # produce one String per grapheme
    tokens = [String(g) for g in Unicode.graphemes(text)]

    keep_space ? tokens :
    [t for t in tokens if !isspace(first(t))]
end


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
    encode_char_batch(texts, vocab;
                      pad_value = vocab.unk_id,
                      kwargs...) -> Matrix{Int}

Pipeline helper = `tokenize_char` → `chars_to_ids` -> `pad_sequences`.

* `texts` can be any `AbstractVector{<:AbstractString}`.
* Extra `kwargs...` are forwarded **unchanged** to `tokenize_char`
  (e.g. `lower=true`, `keep_space=true`, `do_remove_accents=true`, ...).
* Padding happens along rows, so the resulting matrix shape is
  `(max_length, length(texts))`.
"""
function encode_char_batch(texts::AbstractVector{<:AbstractString},
                           vocab::Vocabulary;
                           pad_value::Int = vocab.unk_id,
                           kwargs...)

    seqs   = [tokenize_char(t; kwargs...)            for t in texts]
    idseqs = [chars_to_ids(s, vocab; add_new=false)  for s in seqs]
    return pad_sequences(idseqs; pad_value=pad_value)
end


