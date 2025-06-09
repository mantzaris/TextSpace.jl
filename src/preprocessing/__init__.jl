module Plumbing

include("CleanText.jl")
include("TextNormalization.jl")
include("Tokenization.jl")
include("CharProcessing.jl")
include("SentenceProcessing.jl")
include("ParagraphProcessing.jl")

export clean_text, strip_zero_width, normalize_whitespace,
        remove_punctuation, remove_emojis, remove_accents,
        tokenize, tokenize_batch, unwrap_lines,
        tokenize_char, char_tokens,
        split_sentences,
        split_paragraphs,
        filter_paragraphs, normalize_unicode, paragraph_windows, 
        merge_short_paragraphs, _is_blank_paragraph, drop_empty_paragraph, 
        strip_outer_quotes, SlidingSentenceWindow,
        basic_tokenize, strip_punctuation, ngrams, WHITESPACE_REGEX

end