

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