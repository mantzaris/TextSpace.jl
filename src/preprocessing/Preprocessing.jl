
#=

=#

module Preprocessing

include(joinpath(@__DIR__, "Vocabulary.jl"))          #  -> Vocabulary, convert_tokens_to_ids, …
include(joinpath(@__DIR__, "IdMapping.jl"))           #  -> convert_tokens_to_ids, ids↔tokens
include(joinpath(@__DIR__, "TextNormalization.jl"))   #  -> clean_text / normalize_whitespace
include(joinpath(@__DIR__, "Stemming.jl"))
include(joinpath(@__DIR__, "Lemmatization.jl"))
include(joinpath(@__DIR__, "Tokenization.jl"))        #  -> tokenize
include(joinpath(@__DIR__, "CharProcessing.jl"))      #  -> tokenize_char
include(joinpath(@__DIR__, "SentenceProcessing.jl"))  #  -> split_sentences
include(joinpath(@__DIR__, "ParagraphProcessing.jl")) #  -> split_paragraphs, paragraph_windows
include(joinpath(@__DIR__, "SubwordTokenization.jl")) #  -> train_bpe, load_bpe, encode, …
include(joinpath(@__DIR__, "TextVectorization.jl"))   #  -> pad_sequences, tfidf_matrix, …
include(joinpath(@__DIR__, "PhraseDetection.jl"))     #  -> learn_phrases, merge_phrases
include(joinpath(@__DIR__, "DocumentProcessing.jl"))  #  -> process_document, document_batch_iter


"""
    fit_subword_tokenizer(corpus_paths;
                          vocab_size   = 32_000,
                          model_type   = "bpe",
                          model_prefix = "spm",
                          kwargs...)          -> SubwordTokenizer

Learn a **pure-Julia Byte-Pair Encoding (BPE)** model from the text files
in `corpus_paths` and return a ready-to-use `SubwordTokenizer`.  
Call this **once** at project start, then reuse the tokenizer everywhere.

### When to use
* Before training or inferring any **sub-word** embedding model.
* When you need a reproducible vocabulary:  
  `Serialization.serialize("tok.bin", tok)`.

### Examples
```julia
julia> using TextSpace

# Train on two corpora, get a tokenizer with 24 k merges
tok = fit_subword_tokenizer(["data/wiki.txt", "data/news.txt"];
                            vocab_size = 24_000)

# Persist for future sessions
Serialization.serialize("tok.bin", tok)
tok  = Serialization.deserialize("tok.bin")
```
"""
function fit_subword_tokenizer(corpus_paths;
                               vocab_size   = 32_000,
                               model_type   = "bpe",
                               model_prefix = "spm",
                               kwargs...)
    model_path = train_bpe(corpus_paths;
                           vocab_size   = vocab_size,
                           model_type   = model_type,
                           model_prefix = model_prefix,
                           kwargs...)
    return load_bpe(model_path)      # defined in SubwordTokenization.jl
end


"""
    preprocess_document(text;
                        tok,                       # SubwordTokenizer (REQUIRED)
                        max_tokens    = 512,
                        return        = :batch,    # :batch | :ids | :sentences
                        kwargs...)                 -> Matrix / Vector

**Whole document -> tensor** in one call.

1. Cleans & normalises  
2. Splits into paragraphs & sentences  
3. Encodes each sentence with the sub-word `tok`  
4. Pads to `(≤max_tokens, n_sentences)` matrix if `return == :batch`

### When to use
* Sentence-embedding or classification models that accept batched
  sentences (SBERT-style).
* Quick inspection of sentence segmentation (`return = :sentences`).

### Examples
```julia
julia> using TextSpace
julia> x = preprocess_document(read("article.txt", String); tok)
512x42 Matrix{Int32}

# Just sentence strings
julia> preprocess_document(text; tok, return=:sentences)[1:3]
3-element Vector{String}:
 "Dr. Smith went home."
 "He slept."
 "Meanwhile, the cat played."
```
"""
function preprocess_document(text::AbstractString;
                             tok,
                             max_tokens::Int = 512,
                             return::Symbol  = :batch;
                             kwargs...)

    # delegate to DocumentProcessing.jl
    return process_document(text;
                            tok          = tok,
                            return       = return,
                            max_tokens   = max_tokens;
                            kwargs...)
end


"""
    preprocess_paragraphs(text;
                          tok,
                          max_tokens = 256) -> Vector{Matrix{Int}}

Split `text` into paragraphs, then into **windows** of <= `max_tokens`
sub-word tokens.  Each window becomes a padded matrix.

### When to use
* Long-form **retrieval-augmented generation (RAG)** or QA where you chunk
  documents into fixed token budgets.
* Streaming books or reports through a Transformer with a modest length
  limit.

### Examples 
```julia
julia> using TextSpace
julia> batches = preprocess_paragraphs(read("novel.txt", String);
                                       tok, max_tokens=256)

julia> size.(batches) |> first
(256, 11)   # 256 tokens x 11 sentences in first chunk
```
"""
function preprocess_paragraphs(text::AbstractString;
                               tok;
                               max_tokens::Int = 256)

    ps        = split_paragraphs(text; unwrap=true) |> filter_paragraphs
    iterator  = paragraph_windows(ps, max_tokens;
                                  tokenizer = p -> encode(tok, p))

    batches = Matrix{Int}[]
    for chunk in iterator
        push!(batches,
              preprocess_document(join(chunk, "\n\n");
                                  tok, max_tokens=max_tokens))
    end
    return batches
end


"""
    preprocess_sentences(text;
                         tok      = nothing,     # sub-word if provided
                         vocab    = nothing,     # else word-level IDs
                         return   = :batch,      # :batch | :ids | :tokens
                         kwargs...)              -> Matrix / Vector

**Sentence-centric** pipeline.

* If `tok` is given sub-word (BPE) ids.  
* Otherwise pass a word-level `vocab`.

### When to use
* STS, NLI, or any model that consumes one sentence at a time.
* Rapid word-level baselines without fitting a BPE tokenizer.

### Examples
```julia
julia> using TextSpace

## sub-word mode
ids   = preprocess_sentences(text; tok, return=:ids)   # Vector{Vector{Int}}
batch = preprocess_sentences(text; tok)                # padded matrix

## word mode
vocab = Vocabulary(Dict{String,Int}(), String[], Dict{Int,Int}(), 0)
batch_word = preprocess_sentences(text;
                                  vocab=vocab, return=:batch)  # grows vocab
```
"""
function preprocess_sentences(text::AbstractString;
                              tok      = nothing,
                              vocab    = nothing,
                              return::Symbol = :batch;
                              kwargs...)

    sentences = split_sentences(text)
    if tok !== nothing  # sub-word
        idseqs = [encode(tok, s; add_special_tokens=true) for s in sentences]
        return return == :ids    ? idseqs :
               return == :sentences ? sentences :
               pad_sequences(idseqs; pad_value = tok.pad_id)

    else  # word
        @assert vocab !== nothing "Need `vocab` when `tok` isn't provided."
        tokens = tokenize_batch(sentences; kwargs...)
        idseqs = [convert_tokens_to_ids(t, vocab) for t in tokens]
        return return == :tokens ? tokens :
               return == :ids    ? idseqs :
               pad_sequences(idseqs; pad_value = vocab.unk_id)
    end
end


"""
    preprocess_chars(text;
                     vocab,                   # char Vocabulary
                     return = :batch,         # :batch | :ids | :tokens
                     kwargs...)               -> Matrix / Vector

Character-level (Unicode **grapheme cluster**) pipeline.

### When to use
* Char-CNNs, byte- or char-level language models, noisy-text robustness.

### Examples
```julia
julia> using TextSpace

# Build a character vocabulary (training phase)
char_vocab = Vocabulary(Dict{String,Int}(), String[], Dict{Int,Int}(), 0)

# Process a new string
x = preprocess_chars("Café 👨‍🚀!"; vocab=char_vocab)  # padded matrix

# Inspect char tokens
preprocess_chars("naïve"; vocab=char_vocab, return=:tokens)
# -> ["n", "a", "ï", "v", "e"]
```
"""
function preprocess_chars(text::AbstractString;
                          vocab,
                          return::Symbol = :batch;
                          kwargs...)

    chars  = tokenize_char(text; kwargs...)
    ids    = convert_tokens_to_ids(chars, vocab)
    return return == :ids ? [ids] :
           return == :tokens ? [chars] :
           pad_sequences([ids]; pad_value = vocab.unk_id)
end



export  fit_subword_tokenizer,
        preprocess_document,
        preprocess_paragraphs,
        preprocess_sentences,
        preprocess_chars



end #END MODULE

