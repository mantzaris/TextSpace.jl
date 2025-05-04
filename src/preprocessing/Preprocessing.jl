
#=
# Functions Needed for Full Text-Based Embedding Support

## Additional Recommendations for a Comprehensive Embeddings Pipeline

7. **Pipeline Orchestration (`process_document`)**
   - Let `process_document(text::String)` combine multiple steps (cleaning, optional normalization, tokenization).
   - Allow for keyword arguments or a configuration object to tailor each sub-step (e.g., `do_cleanup = true`, `tokenization_mode = :subword`, etc.).

8. **Batch Processing & Large Corpus Handling**
   - Consider batch-based or streaming methods for handling large datasets more efficiently.
   - Multi-threading or distributed processing (e.g., using `Threads.@threads` or `Distributed`) can significantly speed up large corpus embeddings.

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