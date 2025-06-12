module Plumbing

using Unicode
import Base: isempty

include("clean_text.jl")
include("text_normalization.jl")
include("tokenization.jl")
include("char_processing.jl")
include("sentence_processing.jl")
include("paragraph_processing.jl")


export clean_text, strip_zero_width, normalize_whitespace,
       tokenize, tokenize_batch,
       tokenize_char,
       split_sentences,
       split_paragraphs,
       filter_paragraphs

end
