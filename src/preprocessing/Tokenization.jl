
using Unicode  
import Base: isempty


if !@isdefined(_PUNCT_CHARS)
    const STOPWORDS_EN = Set([
        "a","an","and","are","as","at","be","by","for","from","has","he","in",
        "is","it","its","of","on","that","the","to","was","were","will","with"
    ])

    const _PUNCT_CHARS = Set{Char}(".,!?:;\"'’“”()[]{}<>")

    #words  = 1st branch  [\p{L}\p{N}_']+
    #punct. = 2nd branch  [^\p{L}\p{N}\s]   (any non-alnum, non-ws)
    const TOKEN_REGEX               = r"[\p{L}\p{N}_']+|[^\p{L}\p{N}\s]"
    const TOKEN_REGEX_WITH_WS       = r"\s+|[\p{L}\p{N}_']+|[^\p{L}\p{N}\s]"
    const WHITESPACE_REGEX          = r"^\s+$"
end


"""load_stopwords(lang="en") → Set{String}"""
load_stopwords(lang::AbstractString = "en") =
    lang == "en" ? STOPWORDS_EN :
    error("No stop-word list bundled for language '$lang'.")


"""
    basic_tokenize(text; keep_whitespace=false) -> Vector{String}

Splits `text` into words and punctuation.  Whitespace is discarded unless
`keep_whitespace=true`.
"""
function basic_tokenize(text::AbstractString; keep_whitespace::Bool=false)
    pat = keep_whitespace ? TOKEN_REGEX_WITH_WS : TOKEN_REGEX
    toks = [m.match for m in eachmatch(pat, text)]
    keep_whitespace ? toks :
        filter(t -> !occursin(WHITESPACE_REGEX, t) && t != "\n", toks)
end


"""strip_punctuation(tok)  - remove leading/trailing punctuation."""
strip_punctuation(tok::AbstractString) = strip(String(tok), _PUNCT_CHARS)


function ngrams(tokens::Vector{<:AbstractString}, n::Int)
    n < 1 && throw(ArgumentError("n must be >= 1, got $n"))
    n == 1 && return tokens
    len = length(tokens)
    len < n && return String[]
    [join(tokens[i:i+n-1], '_') for i in 1:len-n+1]
end


"""
    tokenize(text;
             lang                 = "en",
             strip_punctuation    = true,
             lower                = true,
             remove_stopwords     = false,
             stopwords            = STOPWORDS_EN,
             ngram                = 1,
             keep_whitespace      = false)

Returns `Vector{String}` of processed tokens.
"""
function tokenize(text::AbstractString;
                  lang::AbstractString = "en",
                  strip_punctuation::Bool = true,   #original name
                  lower::Bool            = true,
                  remove_stopwords::Bool = false,
                  stopwords::Set{String} = STOPWORDS_EN,
                  ngram::Int             = 1,
                  keep_whitespace::Bool  = false)

    toks = basic_tokenize(text; keep_whitespace)

    if strip_punctuation                     # Bool arg
        # fully-qualify the *function* to avoid the name clash
        toks = Plumbing.strip_punctuation.(toks)
    end

    lower && (toks = lowercase.(toks))
    remove_stopwords && (toks = [t for t in toks if !(t in stopwords)])

    toks = filter!(!isempty, toks)
    ngram > 1 && (toks = ngrams(toks, ngram))
    return toks
end


"""
    tokenize_batch(docs; threaded=false, kwargs...)

Tokenise every document in `docs` (any `AbstractVector` of strings) with
the same keyword options accepted by `tokenize`.

* If `threaded=true` **and** Julia was started with more than one thread
  (`JULIA_NUM_THREADS > 1`), the batch is processed in parallel.

Returns a `Vector{Vector{String}}` aligned with `docs`.
"""
function tokenize_batch(docs::AbstractVector{<:AbstractString};
                        threaded::Bool = false,
                        kwargs...)

    if threaded && Threads.nthreads() > 1
        out = Vector{Vector{String}}(undef, length(docs))
        Threads.@threads for i in eachindex(docs)
            out[i] = tokenize(docs[i]; kwargs...)
        end
        return out
    else
        return [tokenize(d; kwargs...) for d in docs]
    end
end

