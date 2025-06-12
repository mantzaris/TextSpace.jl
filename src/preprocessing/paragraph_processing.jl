
if !@isdefined(PAR_SEP)
    const NL      = r"\R"            # Unicode newline class
    const PAR_SEP = r"\R{2,}"        # ≥ two consecutive newlines
    const _BLANK_RE = r"[\p{Zs}\u200B\u200C\u200D]+"
end


"""
    unwrap_lines(text) -> String

Collapse *hard-wrapped* lines **inside the same sentence** while preserving
paragraph boundaries.

* A newline that is **not** immediately preceded by `.`, `?`, or `!`
  **and** is **not** followed by another newline is replaced with a single
  space.

* Sequences of two or more consecutive newlines (`\\R{2,}`) are kept intact
  so paragraph breaks survive.

This is UTF-8 safe and language-agnostic; it merely looks at the punctuation
character right before the newline.
"""
unwrap_lines(txt::AbstractString) =
    replace(txt, r"(?<![.\?!\r\n])\R(?!\R)" => " ")


function split_paragraphs(text::AbstractString;
                          unwrap::Bool = false,     # match docstring
                          normalize::Bool = true)
    isempty(text) && return String[]
    sep = r"\R\s*\R"                               # blank lines include spaces
    paras = split(text, sep; keepempty=false)
    unwrap   && (paras = map(unwrap_lines, paras))
    normalize && (paras = map(p -> normalize_whitespace(strip(p)), paras))
    return paras
end


function paragraph_windows(paras::Vector{String},
                           max_tokens::Int;
                           stride::Int = max_tokens,
                           tokenizer)

    max_tokens >= 1 || throw(ArgumentError("max_tokens >= 1"))
    stride     >= 1 || throw(ArgumentError("stride >= 1"))

    function _iter(state)
        i = state
        i > length(paras) && return nothing

        j, toks = i, 0
        while j <= length(paras)
            t = length(tokenizer(paras[j]))
            if t == 0                             # avoid infinite loop
                t = 1                             # treat as 1 token
            end
            if toks + t > max_tokens
                break
            end
            toks += t
            j += 1
        end

        if j == i                                 # first paragraph too long
            chunk = [paras[i]]
            j     = i + 1
        else
            chunk = paras[i:j-1]
        end
        next_state = min(i + stride, length(paras) + 1)
        return chunk, next_state
    end
    return _iter, 1
end


function merge_short_paragraphs(paragraphs::Vector{String};
                                min_chars::Int = 40,
                                min_sents::Union{Nothing,Int} = nothing,
                                sentence_splitter = split_sentences)

    paras = copy(paragraphs)   
    out = String[]
    i   = 1
    while i ≤ length(paras)
        p = paras[i]
        short = length(p) < min_chars ||            
                (min_sents !== nothing &&
                 length(sentence_splitter(p)) < min_sents)

        if short && !isempty(out)       # merge into previous
            out[end] = strip(out[end] * " " * p)
        elseif short && i < length(paras)    # merge with next
            paras[i+1] = strip(p * " " * paras[i+1])
        else
            push!(out, p)
        end
        i += 1
    end
    return out
end


_is_blank_paragraph(p::AbstractString) = isempty(strip(replace(p, _BLANK_RE => " ")))


drop_empty_paragraph(paras::Vector{<:AbstractString}) =
    [String(p) for p in paras if !_is_blank_paragraph(p)]


"""
    filter_paragraphs(paras; min_chars = 25)

Return only those paragraphs whose character length ≥ `min_chars`.

* Accepts any `AbstractVector{<:AbstractString}`.
* Always returns a `Vector{String}` (converting `SubString` if necessary).
"""
function filter_paragraphs(paras::AbstractVector{<:AbstractString};
                           min_chars::Int = 25)
    [String(p) for p in paras if _par_width(p) ≥ min_chars]
end


_par_width(p::AbstractString) = textwidth(p)