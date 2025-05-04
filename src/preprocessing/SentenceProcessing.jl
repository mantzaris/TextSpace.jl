using Languages                 
import Base.Iterators: take

# pre-compiled regexes to help the micro-segmentation
const ABBREV  = r"(?:Mr|Ms|Dr|Prof|Sr|Jr|Inc|Ltd|vs)\.$"
const SENTEND = r"[.!?]+[”\"\']?(\s+|$)"  # punctuation boundary

"""
    split_sentences(text; lang = "en", abbreviations = ABBREV)

Simple rule-based segmenter that handles common abbreviations.
"""
function split_sentences(text::AbstractString;
                         lang::AbstractString = "en",
                         abbreviations::Regex = ABBREV)

    text = normalize_whitespace(text)
    starts, sentences = 1, String[]
    for m in eachmatch(SENTEND, text)
        stop  = m.offset + length(m.match) - 1
        chunk = @view text[starts:stop]
        # skip false split if we ended on an abbreviation
        if !occursin(abbreviations, chunk)
            push!(sentences, strip(chunk))
            starts = stop + 1
        end
    end
    # tail
    if starts <= lastindex(text)
        push!(sentences, strip(@view text[starts:end]))
    end
    return sentences
end


strip_outer_quotes(s::AbstractString) =
    (startswith(s, ('"', '“')) && endswith(s, ('"', '”'))) ? s[2:end-1] : s

struct SlidingSentenceWindow{T}
    sents::Vector{T}
    max_tokens::Int
    stride::Int
end

Base.iterate(win::SlidingSentenceWindow, state = 1) =
    state > length(win.sents) ? nothing :
    begin
        chunk = win.sents[state:min(state+win.stride-1,length(win.sents))]
        tok_n = sum(length.(chunk))
        if tok_n > win.max_tokens
            # fall back to sentence-by-sentence
            chunk = [win.sents[state]]
        end
        next_state = state + length(chunk)
        chunk, next_state
    end

# convenience constructor
SlidingSentenceWindow(sents, max_tokens; stride=max_tokens) =
    SlidingSentenceWindow(sents, max_tokens, stride)