if !@isdefined(ABBREV_EN)
    const ABBREV_EN = [
        "Mr","Mrs","Ms","Dr","Prof","Sr","Jr","St","No",
        "Inc","Ltd","Co","vs"
    ]
    const ABBREV_RE_EN =
        Regex("^\\s*(?:" * join(ABBREV_EN, "|") * ")\\.\\s*\$", "i")

    const SENTEND_RE =
        Regex("(?:\\.{3}|[.!?])+[”\"']?(?:\\s+|" * "\$" * ")")
end

_is_abbrev_end(chunk::AbstractString, abbr::Regex) = begin
    m = match(r"(?i)(\p{L}+)\.\s*[”\"']?$", chunk)
    m !== nothing && occursin(abbr, m.match)
end

_is_abbrev_end(chunk::AbstractString, abbr::Set{String}) = begin
    m = match(r"(?i)(\p{L}+)\.\s*[”\"']?$", chunk)
    m !== nothing && (lowercase(m.captures[1]) in abbr)
end

function split_sentences(text::AbstractString;
                         lang::AbstractString = "en",
                         abbreviations = ABBREV_RE_EN)

    lang ≠ "en" && @warn "split_sentences currently tuned for English."

    text = normalize_whitespace(text)
    starts, sentences = 1, String[]

    for m in eachmatch(SENTEND_RE, text)
        stop   = m.offset + length(m.match) - 1
        chunk  = strip(@view text[starts:stop])

        _is_abbrev_end(chunk, abbreviations) && continue

        push!(sentences, chunk)
        starts = stop + 1
    end

    starts ≤ lastindex(text) &&
        push!(sentences, strip(@view text[starts:end]))

    return sentences
end


function strip_outer_quotes(s::AbstractString)
    firstindex(s) == lastindex(s) && return s

    first   = s[firstindex(s)]
    closing = first == '"'  ? '"'  :
              first == '“' ? '”'  : nothing

    closing === nothing && return s                # no valid opener
    s[lastindex(s)] == closing || return s         # mismatched closer

    lo = nextind(s, firstindex(s))
    hi = prevind(s, lastindex(s))
    return s[lo:hi]
end




struct SlidingSentenceWindow{T}
    sents::Vector{T}
    max_tokens::Int
    stride::Int
    function SlidingSentenceWindow(s::Vector{T}, max_tokens::Int;
                                   stride::Int = max_tokens) where {T}
        max_tokens >= 1 || throw(ArgumentError("max_tokens must be >= 1"))
        stride      >= 1 || throw(ArgumentError("stride must be >= 1"))
        new{T}(s, max_tokens, stride)
    end
end

Base.IteratorSize(::Type{<:SlidingSentenceWindow}) = Base.SizeUnknown()
Base.eltype(::Type{SlidingSentenceWindow{T}}) where {T} = Vector{T}

function Base.iterate(win::SlidingSentenceWindow, state::Int = 1)
    state > length(win.sents) && return nothing

    hi    = min(state + win.stride - 1, length(win.sents))
    chunk = win.sents[state:hi]

    sum(length.(chunk)) > win.max_tokens && (chunk = [win.sents[state]])
    return chunk, state + length(chunk)
end




