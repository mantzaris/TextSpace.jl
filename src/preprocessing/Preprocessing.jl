#A place to gather “higher-level” or “combined” functionality. For instance, orchestrating cleaning + tokenization in one pass, or bridging multiple low-level routines from the other files.

#=
# Functions Needed for Full Text-Based Embedding Support

## 2. **Subword Tokenization (For BPE/SentencePiece)**
 `subword_tokenize(text::String; model::Any)` - Uses BPE, WordPiece, or SentencePiece.
  [PAD], [CLS], [SEP], [UNK], [MASK]. This is crucial for transformer-based embeddings. Ensure your build_vocabulary_bpe(...) or build_vocabulary_wordpiece(...) functions can optionally inject these special tokens at fixed indices for consistent usage.

## 3. **Word-Level Processing**
 `tokenize(text, mode=:word)` - Already implemented.

## 4. **Phrase Detection (For Multi-Word Expressions)**
 `detect_phrases(tokens::Vector{String})` - Identifies frequent word combinations (e.g., "New York").
 detect not only bigrams but also flexible collocations or domain-specific entities. Some libraries do multi-word expressions using Pointwise Mutual Information or frequency thresholds.

## 5. **Sentence Segmentation (For Sentence Embeddings)**
 `split_sentences(text::String)` - Splits text into sentences using punctuation and language rules.

## 6. **Paragraph Processing**
 `split_paragraphs(text::String)` - Splits text into paragraphs based on newlines or indentation.

## 7. **Document-Level Processing**
 `process_document(text::String)` - Handles full document cleaning, tokenization, and structuring.

## 8. **Text Vectorization for Embeddings**
 `convert_tokens_to_ids(tokens::Vector{String}, vocab::Dict{String, Int})` - Maps tokens to indices.
 `pad_sequences(sequences::Vector{Vector{Int}], max_len::Int)` - Ensures uniform input length for embedding models.

 # TODO: Add stopword removal & optional stemming/lemmatization in clean_text()
# TODO: Implement tokenize_subword(text::String; tokenizer::Any) for BPE, WordPiece
# TODO: Implement split_sentences(text::String) for sentence segmentation
# TODO: Implement split_paragraphs(text::String) for paragraph segmentation
# TODO: Implement process_document(text::String) to handle full document preprocessing
# TODO: Implement detect_phrases(tokens::Vector{String}) for phrase-based tokenization
# TODO: Implement convert_tokens_to_ids(tokens::Vector{String}, vocab::Dict{String, Int})
# TODO: Implement pad_sequences(sequences::Vector{Vector{Int}], max_len::Int)


#TODO: tokenize_subword(text::String; tokenizer::Any)
#TODO: tokenize_char(text::String)
#TODO: tokenize_sentence(text::String)
#tokenize_subword(text::String, vocab::Dict{String, Int})

## Additional Recommendations for a Comprehensive Embeddings Pipeline

1. **Unicode & Diacritics Normalization**
   - Consider adding functions for robust normalization (e.g., NFKC, accent stripping) to handle multilingual text.
   - Could be included in `clean_text()` or as a separate utility in `TextNormalization.jl`.

2. **Stemming / Lemmatization**
   - Offer optional stemming or lemmatization, especially useful for classic NLP tasks (though many modern subword methods do not require it).
   - Could be toggled via a keyword argument in `clean_text()` or implemented as separate functions (e.g., `stem_tokens(tokens)`, `lemmatize_tokens(tokens)`).

3. **Stopword Removal**
   - Provide an optional step to remove frequent, less meaningful words (common in older n-gram pipelines or certain domain-specific tasks).
   - Maintain a default stopword list (for English or multiple languages), or allow users to supply their own.

4. **Robust Sentence Splitting**
   - Improve beyond naive period-based splitting for languages with complex punctuation (e.g. Spanish, French) or no whitespace (Chinese, Japanese).
   - Possibly integrate or port advanced heuristics from existing libraries or implement standard rules (e.g., newline plus punctuation, ignoring abbreviations).

5. **Language-Specific Tokenizers**
   - Some languages (Chinese, Japanese, Thai) need specialized tokenization. Consider allowing external plug-ins or supporting them natively if community contributions arise.

6. **Advanced Phrase Detection**
   - Expand `detect_phrases` to handle multi-word expressions beyond bigrams, using statistical methods (e.g., PMI-based approaches).
   - Provide tuning parameters for frequency thresholds or chunk size.

7. **Pipeline Orchestration (`process_document`)**
   - Let `process_document(text::String)` combine multiple steps (cleaning, optional normalization, tokenization).
   - Allow for keyword arguments or a configuration object to tailor each sub-step (e.g., `do_cleanup = true`, `tokenization_mode = :subword`, etc.).

8. **Batch Processing & Large Corpus Handling**
   - Consider batch-based or streaming methods for handling large datasets more efficiently.
   - Multi-threading or distributed processing (e.g., using `Threads.@threads` or `Distributed`) can significantly speed up large corpus embeddings.

10. **Analysis & Diagnostics (Optional)**
   - Include helper utilities to analyze vocabulary coverage, out-of-vocabulary rates, or subword merges (e.g., `analyze_vocab_usage(tokens, vocab)`).
   - Could reside in `utils/` or a future `analysis/` folder.

11. **Multilingual Extension Plans**
   - Mention in your documentation whether you will rely on community PRs for languages with complex tokenization rules or if you plan to incorporate them natively.

12. **External Integration**
   - Some users may want to use external libraries for subword tokenization (e.g., Python libraries or SentencePiece C++). Consider optionally providing wrappers in Julia for advanced performance or direct usage.



You have SerializationUtilities.jl in utils/. Make sure your vocabulary-building methods have a standard approach for saving/loading the token dictionary (e.g., JSON, binary format).
=#

module Preprocessing

# always include sub‑files relative to THIS file
include(joinpath(@__DIR__, "TextNormalization.jl"))
include(joinpath(@__DIR__, "Stemming.jl"))

export stem_text, clean_text

end #END MODULE


# """
#     tokenize_word(text::String; mode::Symbol=:word) -> Vector{String}

# Tokenizes the input text into a vector of tokens.  
# - If `mode` is `:word` (default), the text is split into word-level tokens.
# - If `mode` is `:char`, the text is split into individual characters.

# # Examples
# ```julia
# julia> tokenize_word("Hello, Julia! 😊", mode=:word)
# ["hello", ",", "julia", "!", "😊"]

# julia> tokenize_word("Hello, Julia! 😊", mode=:char)
# ['h', 'e', 'l', 'l', 'o', ',', ' ', 'j', 'u', 'l', 'i', 'a', '!', ' ', '😊']
# ```
# """ 
# function tokenize_word(text::String; mode::Symbol = :word) 
#     #TODO: optional strip punctuation
#     if mode == :word 
#         #Unicode-aware regex: # - [\p{L}\p{N}]+ matches sequences of letters, numbers, or underscores. 
#         #[^\s\p{L}\p{N}]+ matches one or more characters that are not whitespace or part of a word (e.g., punctuation, symbols, emoji). 
#         pattern = r"[\p{L}\p{N}]+|[^\s\p{L}\p{N}]+"
#         tokens = [m.match for m in eachmatch(pattern, text)]
#         return tokens 
#     elseif mode == :char 
#         return collect(text) 
#     else 
#         error("Unsupported tokenization mode: $mode. Use :word or :char.") 
#     end 
# end