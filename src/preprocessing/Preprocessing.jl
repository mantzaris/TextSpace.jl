module Preprocessing

include(joinpath(@__DIR__, "Vocabulary.jl"))          #  -> Vocabulary, convert_tokens_to_ids, …
# include(joinpath(@__DIR__, "IdMapping.jl"))           #  -> convert_tokens_to_ids, ids↔tokens
include(joinpath(@__DIR__, "TextNormalization.jl"))   #  -> clean_text / normalize_whitespace
include(joinpath(@__DIR__, "Stemming.jl"))
include(joinpath(@__DIR__, "Lemmatization.jl"))
include(joinpath(@__DIR__, "Tokenization.jl"))        #  -> tokenize
include(joinpath(@__DIR__, "CharProcessing.jl"))      #  -> tokenize_char
include(joinpath(@__DIR__, "SentenceProcessing.jl"))  #  -> split_sentences
include(joinpath(@__DIR__, "ParagraphProcessing.jl")) #  -> split_paragraphs, paragraph_windows
include(joinpath(@__DIR__, "SubwordTokenization.jl")) #  -> train_bpe, load_bpe, encode, …
include(joinpath(@__DIR__, "TextVectorization.jl"))   #  -> pad_sequences, tfidf_matrix, …
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
function process_document(text::AbstractString,
   tok;                     # SubwordTokenizer (positional)
   stage::Symbol  = :batch,
   max_tokens::Int = 512,
   clean::Bool    = true,
   lower::Bool    = true,
   stem::Bool     = false,
   lemmatize::Bool = false,
   kwargs...)

   # 1 · optional cleaning
   if clean
      text = clean_text(text)
      text = normalize_whitespace(text)
   end

   # 2 · sentence split
   sentences = split_sentences(text)

   if stage == :sentences
      return sentences
   end

   # 3 · tokenise
   tokens = tokenize_batch(sentences;
      lower     = lower,
      stem      = stem,
      lemmatize = lemmatize)

   if stage == :tokens
      return tokens
   end

   # 4 · sub-word encode  (tok is required for this helper)
   idseqs = [encode(tok, join(t, ' '); add_special_tokens = true)
   for t in tokens]

   if stage == :ids
      return idseqs
   end

   # 5 · pad to matrix
   return pad_sequences(idseqs; pad_value = tok.pad_id)
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
function preprocess_sentences(text::AbstractString;
   tok   = nothing,
   vocab = nothing,
   out::Symbol = :batch,          # :batch | :ids | :tokens | :sentences
   kwargs...)

sentences = split_sentences(text)

   if tok !== nothing
   idseqs = [encode(tok, s; add_special_tokens = true) for s in sentences]

   if out == :ids
      return idseqs
   elseif out == :sentences
      return sentences
   else                           # default :batch
      return pad_sequences(idseqs; pad_value = tok.pad_id)
      end
   end

   @assert vocab !== nothing "Need `vocab` when `tok` is not provided."

   tokens = tokenize_batch(sentences; kwargs...)
   idseqs = [convert_tokens_to_ids(t, vocab) for t in tokens]

   if out == :tokens
      return tokens
   elseif out == :ids
      return idseqs
   else                               # :batch
      return pad_sequences(idseqs; pad_value = vocab.unk_id)
   end
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
   mode::Symbol = :batch,      # ← safe keyword
   kwargs...)

   sentences = split_sentences(text)

   if tok !== nothing
      idseqs = [encode(tok, s; add_special_tokens = true) for s in sentences]

      if mode == :ids
         return idseqs
      elseif mode == :sentences
         return sentences
      elseif mode == :batch           # padded matrix
         return pad_sequences(idseqs; pad_value = tok.pad_id)
      else
         error("Unknown mode $(mode). Use :batch | :ids | :sentences")
      end
   end

   @assert vocab !== nothing "Provide `vocab` when `tok` is not given."

   tokens = tokenize_batch(sentences; kwargs...)
   idseqs = [convert_tokens_to_ids(t, vocab) for t in tokens]

   if mode == :tokens
      return tokens
   elseif mode == :ids
      return idseqs
   elseif mode == :batch
      return pad_sequences(idseqs; pad_value = vocab.unk_id)
   else
      error("Unknown mode $(mode). Use :batch | :ids | :tokens")
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
   out::Symbol = :batch,              # :batch | :ids | :tokens
   kwargs...)

   chars = tokenize_char(text; kwargs...)
   ids   = convert_tokens_to_ids(chars, vocab)

   if out == :ids
      return [ids]
   elseif out == :tokens
      return [chars]
   else                                           # :batch
      return pad_sequences([ids]; pad_value = vocab.unk_id)
   end
end



export  fit_subword_tokenizer,
        preprocess_document,
        preprocess_paragraphs,
        preprocess_sentences,
        preprocess_chars



end #END MODULE

