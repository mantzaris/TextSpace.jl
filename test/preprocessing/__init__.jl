using Random
using Unicode

using TextSpace
using TextSpace.Plumbing
using TextSpace.Plumbing: tokenize_char, char_tokens, remove_punctuation, normalize_whitespace, remove_emojis, remove_accents, clean_text, strip_zero_width,
                            normalize_unicode, normalize_whitespace, basic_tokenize, ngrams, strip_punctuation, tokenize, tokenize_batch,
                            split_sentences, strip_outer_quotes, SlidingSentenceWindow,
                            unwrap_lines, split_paragraphs, paragraph_windows, merge_short_paragraphs, _is_blank_paragraph, drop_empty_paragraph, filter_paragraphs

import TextSpace.Plumbing: basic_tokenize, WHITESPACE_REGEX 


include("preprocessing_char_tests.jl")
include("preprocessing_cleantext_tests.jl")
include("preprocessing_textnormalization_tests.jl")
include("preprocessing_tokenization_tests.jl")

include("preprocessing_sentence_tests.jl")
include("preprocessing_paragraph_tests.jl")
