module Plumbing


include("CleanText.jl")
include("TextNormalization.jl")
include("Tokenization.jl")
include("CharProcessing.jl")
include("SentenceProcessing.jl")
include("ParagraphProcessing.jl")


export clean_text,
       normalize_whitespace,
       tokenize, 
       tokenize_batch,
       tokenize_char,
       split_sentences,
       split_paragraphs,
       filter_paragraphs


end
