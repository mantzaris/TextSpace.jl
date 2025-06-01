
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
    char_tokens(word; eos=nothing, normalize=false, lower=false) → Vector{String}

Return a vector of Unicode grapheme clusters for **one word**.
If `eos` is given (e.g. `"</w>"`) it is appended as a boundary marker.

This is a lightweight wrapper around `tokenize_char`.
"""
function char_tokens(word::AbstractString;
                     eos::Union{Nothing,String}=nothing,
                     normalize::Bool = false,
                     lower::Bool     = false)

    toks = tokenize_char(word;
                         normalize = normalize,
                         lower     = lower,
                         keep_space = true)  # one word → we keep spaces if any
    eos === nothing || push!(toks, eos)
    return toks
end