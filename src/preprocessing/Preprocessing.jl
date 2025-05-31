module Preprocessing

include(joinpath(@__DIR__, "CleanText.jl"))
include(joinpath(@__DIR__, "Vocabulary.jl"))          #  -> Vocabulary, convert_tokens_to_ids, 
include(joinpath(@__DIR__, "TextNormalization.jl"))   #  -> clean_text / normalize_whitespace
include(joinpath(@__DIR__, "Stemming.jl"))
include(joinpath(@__DIR__, "Lemmatization.jl"))
include(joinpath(@__DIR__, "Tokenization.jl"))        #  -> tokenize
include(joinpath(@__DIR__, "CharProcessing.jl"))      #  -> tokenize_char
include(joinpath(@__DIR__, "SentenceProcessing.jl"))  #  -> split_sentences
include(joinpath(@__DIR__, "ParagraphProcessing.jl")) #  -> split_paragraphs, paragraph_windows
include(joinpath(@__DIR__, "SubwordProcessing.jl")) #  -> train_bpe, load_bpe, encode, 
include(joinpath(@__DIR__, "TextVectorization.jl"))   #  -> pad_sequences, tfidf_matrix, 
include(joinpath(@__DIR__, "DocumentProcessing.jl"))  #  -> process_document, document_batch_iter


# Export only the high-level pipeline functions
export preprocess_for_char_embeddings,
       preprocess_for_sentence_embeddings,
       preprocess_for_paragraph_embeddings,
       preprocess_for_subword_embeddings,
       preprocess_for_document_embeddings,
       preprocess_for_word_embeddings


# --- Pipeline Function Implementations ---


"""
    preprocess_for_char_embeddings(corpus_input::AbstractString;
                                   from_file::Bool                 = true,
                                   vocab::Union{Vocabulary,Nothing}= nothing,
                                   vocab_options::Dict             = Dict(),
                                   clean_options::Dict             = Dict(),
                                   char_options::Dict              = Dict(),
                                   id_options::Dict                = Dict())
        → NamedTuple{char_ids::Vector{Int},
                     vocabulary::Vocabulary,
                     chars::Vector{String},
                     cleaned_text::String}

High-level helper that **cleans, tokenises, builds/uses a character
vocabulary and converts the text to integer IDs** - everything you need
before feeding a character-level embedding model.


DEFAULT PIPELINE
1. **Read input**

   *If `from_file=true`* and `corpus_input` is a valid path, the file is
   loaded; otherwise the argument is treated as a raw string.

2. **Clean** - `clean_text` with defaults

| flag (kw in `clean_options`) | default | effect |
|------------------------------|---------|--------|
| `:unicode_normalize`         | `true`  | NFC canonical form. |
| `:do_remove_accents`         | `false` | Strip combining marks. |
| `:do_remove_punctuation`     | `false` | Remove Unicode punctuation. |
| `:do_remove_symbols`         | `false` | Remove currency, math, emoji. |
| `:do_remove_emojis`          | `false` | Remove all emoji blocks. |
| `:case_transform`            | `:none` | `:lower | :upper | :none`. |
| `:collapse_whitespace`      | `false` | Collapse runs of blanks to one<br>space **after** cleaning. |

-> `collapse_whitespace` is *intercepted* by this helper and is **not**
forwarded to `clean_text`.

3. **Tokenise to graphemes** - `tokenize_char`

   `char_options` are forwarded verbatim.  Default
   `Dict(:keep_space => true)` keeps ordinary spaces; pass
   `Dict(:keep_space=>false)` if you want them dropped.

4. **Vocabulary**

   *When `vocab === nothing`* a fresh vocabulary is built from the
   observed characters using `build_vocabulary(chars; vocab_options...)`.

   Minimal defaults  
   `vocab_options = Dict(:min_freq=>1,  :special_tokens => ["<unk>"])`

   *When a `Vocabulary` is supplied* it is left intact **except** that
   `ensure_unk!` is called - if `unk_id ≤ 0` a new `<unk>` entry is
   appended and a repaired copy is returned.

5. **IDs** - `chars_to_ids`

   `id_options` go straight through (`add_new`, `update_counts`, ...).
   Default `Dict(:add_new=>false, :update_counts=>true)`.

   *`update_counts=true` mutates `vocab.counts`*


ARGUMENTS

- `corpus_input` :: `AbstractString` — raw text *or* path
- `from_file`    - set `false` if the argument is guaranteed to be text
- `vocab`        - pre-built `Vocabulary` or `nothing`.
- `vocab_options`,`clean_options`,`char_options`,`id_options`
  - see tables above

RETURNS Named tuple  

```julia
(
    char_ids      = Vector{Int},        # integer sequence
    vocabulary    = Vocabulary,         # built or repaired vocab
    chars         = Vector{String},     # grapheme tokens
    cleaned_text  = String              # final cleaned text
)

julia> txt = "Hello  World!  😊";
julia> res = preprocess_for_char_embeddings(txt; from_file=false);

julia> res.chars
["H","e","l","l","o"," ","W","o","r","l","d","!"," ","😊"]

# lock vocabulary, lowercase, drop punctuation & spaces
julia> opts = (
         clean_options = Dict(
             :case_transform        => :lower,
             :do_remove_punctuation => true),
         char_options  = Dict(:keep_space=>false)
       )
julia> res2 = preprocess_for_char_embeddings(txt; vocab=res.vocabulary; opts...)
````
"""
function preprocess_for_char_embeddings(corpus_input::Union{AbstractString, String};
                                        from_file::Bool = true,
                                        vocab::Union{Vocabulary, Nothing}=nothing,
                                        vocab_options::Dict=Dict(),
                                        clean_options::Dict=Dict(),
                                        char_options::Dict=Dict(),
                                        id_options::Dict=Dict())

    #read Corpus (if file path)
    text = from_file && isfile(corpus_input) ?
           read(corpus_input, String) :
           String(corpus_input) 

    # Minimal cleaning suitable for character models
    default_clean_options = Dict(
        :unicode_normalize => true,
        :do_remove_accents => false,
        :do_remove_punctuation => false,
        :do_remove_symbols => false,
        :do_remove_emojis => false,
        :collapse_whitespace  => false,
        :case_transform => :none # Usually keep case for char models
    )

    final_clean_options = merge(default_clean_options, clean_options)
    collapse_ws = pop!(final_clean_options, :collapse_whitespace, false)

    cleaned_text = clean_text(text; final_clean_options...)

    if collapse_ws
        cleaned_text = normalize_whitespace(cleaned_text; preserve_newlines=false)
    end

    #character Tokenization
    default_char_options = Dict(:keep_space => true)
    final_char_options   = merge(default_char_options, char_options)
    chars = tokenize_char(cleaned_text; final_char_options...)

    #determine/Build Vocabulary
    if vocab === nothing
        opts = merge(Dict(:min_freq=>1, :special_tokens=>["<unk>"]), vocab_options)
        haskey(opts, :special_tokens) || (opts[:special_tokens] = ["<unk>"])
    
        vdict   = build_vocabulary(chars; opts...)
        unk_id  = get(vdict["token_to_index"], "<unk>", 0)
        if unk_id == 0
            push!(vdict["index_to_token"], "<unk>")
            vdict["token_to_index"]["<unk>"] = length(vdict["index_to_token"])
            unk_id = vdict["token_to_index"]["<unk>"]
        end
        counts = Dict{Int,Int}()

        if haskey(vdict, "freq")
            for (tok, cnt) in vdict["freq"]
                id = get(vdict["token_to_index"], tok, 0)   # 0 if token was filtered
                id == 0 && continue                         # skip missing tokens
                counts[id] = cnt
            end
        end

    
        final_vocab = Vocabulary(vdict["token_to_index"],
                                 vdict["index_to_token"],
                                 counts,
                                 unk_id)
    else
        final_vocab = ensure_unk!(vocab)   # auto-repair ##final_vocab = vocab
    end

    #convert characters to IDs
    default_id_options = Dict(:add_new => false, :update_counts => true)
    final_options = merge(default_id_options, id_options)
    final_id_options   = merge(default_id_options, id_options)

    ids = chars_to_ids(chars, final_vocab; final_id_options...)

    return (char_ids=ids,
            vocabulary=final_vocab,
            chars=chars,
            cleaned_text=cleaned_text)
end



"""
    preprocess_for_subword_embeddings(text::String, bpe_tokenizer; clean_options=Dict(), sentence_options=Dict(), encode_options=Dict()) -> Vector{Vector{Int}}

Prepares text for subword-based embeddings (e.g., BERT).

Pipeline:
1. Cleans text using `CleanText.clean_text`.
2. Splits cleaned text into sentences using `SentenceProcessing.split_sentences`.
3. Encodes each sentence into subword IDs using `SubwordTokenization.encode`.

# Arguments
- `text::String`: The input text.
- `bpe_tokenizer`: A pre-trained BPE tokenizer (e.g., from `BytePairEncoding.jl` or loaded via `SubwordTokenization.load_bpe`).

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text`.
- `sentence_options::Dict`: Options passed directly to `SentenceProcessing.split_sentences`.
- `encode_options::Dict`: Options passed directly to `SubwordTokenization.encode` for each sentence (e.g., `Dict(:add_special_tokens=>true)`).

# Retu    - `Vector{Vector{Int}}`: A vector of vectors of subword IDs (one inner vector per sentence).

# Examples

```julia
# Assume `bpe` is a loaded BytePairEncoding tokenizer
# using BytePairEncoding, TextEncodeBase
# bpe = load_bpe("path/to/bpe/model") # Or however it's loaded

# Example 1: Basic usage with a hypothetical BPE tokenizer
text = "Subword example. Another sentence."
# Assuming `bpe` exists and is configured
# result = preprocess_for_subword_embeddings(text, bpe)

# Expected output structure (IDs depend heavily on the BPE model):
# [[id1, id2, id3], [id4, id5]]

# Example 2: Keep punctuation during cleaning for BPE
text = "Keep punctuation? Yes!"
# result = preprocess_for_subword_embeddings(text, bpe;
#     clean_options=Dict(:remove_punctuation=>false, :case_transform=>:none)
# )

# Expected output structure (IDs depend heavily on the BPE model):
# [[id_k, id_punc, id_q], [id_y, id_exc]]
# Note: Actual usage requires a valid `bpe` object.
```
"""
function preprocess_for_subword_embeddings(text::String, bpe_tokenizer;
                                           clean_options::Dict=Dict(),
                                           sentence_options::Dict=Dict(),
                                           encode_options::Dict=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    # 2. Sentence Splitting
    sentences = split_sentences(cleaned_text; sentence_options...)

    # 3. Encode each sentence
    sentence_ids = [SubwordTokenization.encode(bpe_tokenizer, sentence; encode_options...) for sentence in sentences]

    return sentence_ids
end

"""
    preprocess_for_subword_embeddings(text;
                                      bpe_vocab_size   = 16_000,
                                      bpe_merges       = 200_000,
                                      special_tokens   = ("<pad>","<unk>","<cls>","<sep>","<mask>"),
                                      clean_options    = Dict(),
                                      sentence_options = Dict(),
                                      encode_options   = Dict())
High-level helper that

1. **learns (or reuses) a BPE encoder** on the supplied *text*,
2. cleans & sentence-splits the text just like the other preprocessors,
3. returns `Vector{Vector{Int}}` ready for `SubwordEmbeddings.train!`.

The learned encoder is returned too so you can reuse it on other text.

```julia
sentences, enc = Preprocessing.preprocess_for_subword_embeddings(long_txt;
                                                           bpe_vocab_size = 8_000)
model, _  = SubwordEmbeddings.train!(ids; enc = enc)

"""
function preprocess_for_subword_embeddings(text::AbstractString;
        # BPE hyper-params
        bpe_vocab_size::Int = 16_000,
        bpe_merges::Int     = 200_000,
        special_tokens      = DEFAULT_SPECIAL_TOKENS,
        clean_options    ::Dict = Dict(),
        sentence_options ::Dict = Dict())

    cleaned    = clean_text(text; clean_options...)
    sentences  = split_sentences(cleaned; sentence_options...)

    enc = train_bpe_custom(sentences;
                           vocab_size = bpe_vocab_size,
                           num_merges = bpe_merges,
                           specials   = special_tokens)

    return sentences, enc
end


 
function preprocess_for_word_embeddings(
        corpus_input::Union{AbstractString,String};
        from_file::Bool     = true,
        min_count::Integer  = 1,
        vocab_options::Dict = Dict(),
        clean_options::Dict = Dict())   # ← NEW

    #  read corpus 
    text = (from_file && isfile(corpus_input)) ?
           read(corpus_input,String) : String(corpus_input)

    #  cleaning 
    default_clean_opts = Dict(
        :unicode_normalize     => true,
        :do_remove_accents     => false,
        :do_remove_punctuation => true,
        :do_remove_symbols     => false,
        :do_remove_emojis      => true,
        :case_transform        => :lower,
    )
    final_clean = merge(default_clean_opts, clean_options)      # MERGE
    cleaned_text = clean_text(text; final_clean...)

    #  tokenisation 
    sentences = split_sentences(cleaned_text)
    default_tok_opts = Dict(
        :strip_punctuation => true,
        :lower             => true,
        :remove_stopwords  => false,
        :lemmatize         => true,
        :stem              => false,
    )
    tokenised = [tokenize(s; default_tok_opts...) for s in sentences]

    #  vocabulary 
    merge!(vocab_options, Dict(
        :min_freq       => min_count,
        :special_tokens => get(vocab_options, :special_tokens, ["<unk>"]),
    ))
    delete!(vocab_options, :min_count)

    if !("<unk>" in vocab_options[:special_tokens])
        push!(vocab_options[:special_tokens], "<unk>")
    end

    vdict  = build_vocabulary(vcat(tokenised...); vocab_options...)
    unk_id = get(vdict["token_to_index"], "<unk>", 0)

    counts = Dict{Int,Int}()
    if haskey(vdict, "freq")
        for (tok,cnt) in vdict["freq"]
            id = get(vdict["token_to_index"], tok, 0)
            id == 0 && continue
            counts[id] = cnt
        end
    end

    vocab = Vocabulary(vdict["token_to_index"], vdict["index_to_token"], counts, unk_id)

    #  ids 
    sent_ids = [tokens_to_ids(toks, vocab; add_new=false) for toks in tokenised]  #TODO: replace with convert_tokens_to_ids

    return (word_ids            = sent_ids,
            vocabulary          = vocab,
            cleaned_text        = cleaned_text,
            tokenized_sentences = tokenised)
end




"""
    preprocess_for_sentence_embeddings(corpus_input::Union{AbstractString, String};
                                             build_vocab::Bool=false,
                                             vocab_options::Dict=Dict(),
                                             sentence_options::Dict=Dict())
                                             -> @NamedTuple{sentences::Vector{String},
                                                            tokenized_sentences::Vector{Vector{String}},
                                                            sentence_token_ids::Union{Vector{Vector{Int}}, Nothing},
                                                            vocabulary::Union{Vocabulary, Nothing},
                                                            cleaned_text::String}

Prepares a corpus (from string or file path) for sentence-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Splits into sentences, tokenizes each, and optionally builds a shared vocabulary.

Pipeline (Defaults Applied):
1. Reads corpus if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, ...)`
3. Splits into sentences: `split_sentences(...)` using `sentence_options`.
4. Tokenizes each sentence: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, ...)`
5. (Optional) Builds vocabulary: `build_vocabulary(...)` from all sentence tokens if `build_vocab=true`.
6. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` for each sentence if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input corpus text (as a single string) or the path to a text file.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from all sentence tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed to `Vocabulary.build_vocabulary` if `build_vocab=true`.
- `sentence_options::Dict`: Options passed to `SentenceProcessing.split_sentences`.

# Returns
- `NamedTuple`: A named tuple containing:
    - `sentences::Vector{String}`: List of cleaned sentences before tokenization.
    - `tokenized_sentences::Vector{Vector{String}}`: Each sentence as a list of its tokens.
    - `sentence_token_ids::Union{Vector{Vector{Int}}, Nothing}`: Each sentence as a list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The vocabulary built across all sentences (or `nothing`).
    - `cleaned_text::String`: The initial cleaned text before sentence splitting.


# Examples

```julia
# Example 1: Basic usage, get tokenized sentences and build vocab
text = "First sentence. Second sentence, with lemma."
result = preprocess_for_sentence_embeddings(text; build_vocab=true, vocab_options=Dict(:min_freq=>1))

# Expected output structure (IDs depend on generated vocab):
# (sentences = ["first sentence", "second sentence with lemma"], 
#  tokenized_sentences = [["first", "sentence"], ["second", "sentence", "with", "lemma"]], 
#  sentence_token_ids = [[id1, id2], [id3, id2, id4, id5]], 
#  vocabulary = Vocabulary(...), 
#  cleaned_text = "first sentence second sentence with lemma")
```

```julia
# Example 2: Get only cleaned sentences (no tokenization/vocab)
text = "Keep punctuation? Maybe. Yes!"
result = preprocess_for_sentence_embeddings(text; 
   build_vocab=false, 
   clean_options=Dict(:remove_punctuation=>false, :case_transform=>:none) # Keep punctuation and case
)

# Expected output:
# (sentences = ["Keep punctuation?", "Maybe.", "Yes!"], 
#  tokenized_sentences = [["keep", "punctuation"], ["maybe"], ["yes"]], # Tokenization still applies defaults
#  sentence_token_ids = nothing, 
#  vocabulary = nothing, 
#  cleaned_text = "Keep punctuation? Maybe. Yes!") 
# Note: tokenized_sentences still uses default tokenization (lowercase, remove punct, lemmatize)
# If you need tokenized sentences matching the cleaned sentences, you'd need more options or separate steps.
```
"""
function preprocess_for_sentence_embeddings(corpus_input::Union{AbstractString, String};
                                            build_vocab::Bool=false,
                                            vocab_options::Dict=Dict(),
                                            sentence_options::Dict=Dict())

    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
                                      
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    #  Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    #  Sentence Splitting
    sentences_split = split_sentences(cleaned_text; sentence_options...)

    # Tokenize each sentence (using defaults)
    tokenized_sentences = [Tokenization.tokenize(sent; default_tokenize_options...) for sent in sentences_split]

    # Optional Vocabulary Building and ID Conversion
    sent_vocab = nothing
    sent_token_ids = nothing

    if build_vocab
        # Build vocab from *all* tokens across sentences
        all_tokens = vcat(tokenized_sentences...)
        
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(all_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\\'\'<unk>\\' token not found in generated vocabulary, using ID 0. Ensure \\'\'<unk>\\' is in special_tokens."
        end
        sent_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                             vocab_dict["index_to_token"],
                                             Dict{Int,Int}(), # Initialize counts
                                             unk_id)

        # Convert tokens to IDs for each sentence using the built vocab
        sent_token_ids = [Tokenization.tokens_to_ids(tokens, sent_vocab; default_id_options...) for tokens in tokenized_sentences]
    end #TODO: replace with convert_tokens_to_ids

    return (sentences=sentences_split, # Return the split sentences
            tokenized_sentences=tokenized_sentences,
            sentence_token_ids=sent_token_ids,
            vocabulary=sent_vocab,
            cleaned_text=cleaned_text)
end





"""
    preprocess_for_paragraph_embeddings(corpus_input::Union{AbstractString, String};
                                              build_vocab::Bool=false,
                                              vocab_options::Dict=Dict(),
                                              paragraph_options::Dict=Dict(),
                                              filter_options::Union{Dict, Nothing}=nothing)
                                              -> @NamedTuple{paragraphs::Vector{String},
                                                             tokenized_paragraphs::Vector{Vector{String}},
                                                             paragraph_token_ids::Union{Vector{Vector{Int}}, Nothing},
                                                             vocabulary::Union{Vocabulary, Nothing},
                                                             cleaned_text::String}

Prepares a corpus (from string or file path) for paragraph-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Splits into paragraphs, optionally filters them, tokenizes each, and optionally builds a shared vocabulary.

Pipeline (Defaults Applied):
1. Reads corpus if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, ...)`
3. Splits into paragraphs: `split_paragraphs(...)` using `paragraph_options`.
4. (Optional) Filters paragraphs: `filter_paragraphs(...)` using `filter_options`.
5. Tokenizes each paragraph: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, ...)`
6. (Optional) Builds vocabulary: `build_vocabulary(...)` from all paragraph tokens if `build_vocab=true`.
7. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` for each paragraph if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input corpus text (as a single string) or the path to a text file.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from all paragraph tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed to `Vocabulary.build_vocabulary` if `build_vocab=true`.
- `paragraph_options::Dict`: Options passed to `ParagraphProcessing.split_paragraphs`.
- `filter_options::Union{Dict, Nothing}`: Options passed to `ParagraphProcessing.filter_paragraphs`. Set to `nothing` (default) to skip filtering.

# Returns
- `NamedTuple`: A named tuple containing:
    - `paragraphs::Vector{String}`: List of cleaned (and potentially filtered) paragraphs before tokenization.
    - `tokenized_paragraphs::Vector{Vector{String}}`: Each paragraph as a list of its tokens.
    - `paragraph_token_ids::Union{Vector{Vector{Int}}, Nothing}`: Each paragraph as a list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The vocabulary built across all paragraphs (or `nothing`).
    - `cleaned_text::String`: The initial cleaned text before paragraph splitting.


# Examples

```julia
# Example 1: Basic usage, build vocabulary
text = "First paragraph.\n\nSecond paragraph, with more words."
result = preprocess_for_paragraph_embeddings(text; build_vocab=true, vocab_options=Dict(:min_freq=>1))

# Expected output structure (IDs depend on generated vocab):
# (paragraphs = ["first paragraph", "second paragraph with more word"], 
#  tokenized_paragraphs = [["first", "paragraph"], ["second", "paragraph", "with", "more", "word"]], 
#  paragraph_token_ids = [[id1, id2], [id3, id2, id4, id5, id6]], 
#  vocabulary = Vocabulary(...), 
#  cleaned_text = "first paragraph second paragraph with more word")
```

```julia
# Example 2: Filter short paragraphs, don't build vocab
text = "Short para.\n\nThis is a much longer paragraph that should pass the filter."
result = preprocess_for_paragraph_embeddings(text; 
   build_vocab=false, 
   filter_options=Dict(:min_chars=>20)
)

# Expected output:
# (paragraphs = ["this be a much long paragraph that should pass the filter"], 
#  tokenized_paragraphs = [["this", "be", "a", "much", "long", "paragraph", "that", "should", "pass", "the", "filter"]], 
#  paragraph_token_ids = nothing, 
#  vocabulary = nothing, 
#  cleaned_text = "short para this be a much long paragraph that should pass the filter")
```
"""
function preprocess_for_paragraph_embeddings(corpus_input::Union{AbstractString, String};
                                             build_vocab::Bool=false,
                                             vocab_options::Dict=Dict(),
                                             paragraph_options::Dict=Dict(),
                                             filter_options::Union{Dict, Nothing}=nothing)

    # --- Define Default Options --- 
    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    # Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    # Paragraph Splitting
    paragraphs_raw = split_paragraphs(cleaned_text; paragraph_options...)

    # Optional Filtering
    paragraphs_filtered = if filter_options !== nothing
        filter_paragraphs(paragraphs_raw; filter_options...)
    else
        paragraphs_raw
    end

    # Tokenize each paragraph (using defaults)
    tokenized_paragraphs = [Tokenization.tokenize(para; default_tokenize_options...) for para in paragraphs_filtered]

    # Optional Vocabulary Building and ID Conversion
    para_vocab = nothing
    para_token_ids = nothing

    if build_vocab
        # Build vocab from *all* tokens across paragraphs
        all_tokens = vcat(tokenized_paragraphs...)
        
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(all_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\'<unk>\' token not found in generated vocabulary, using ID 0. Ensure \'<unk>\' is in special_tokens."
        end
        para_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                             vocab_dict["index_to_token"],
                                             Dict{Int,Int}(), # Initialize counts
                                             unk_id)

        # Convert tokens to IDs for each paragraph using the built vocab
        para_token_ids = [Tokenization.tokens_to_ids(tokens, para_vocab; default_id_options...) for tokens in tokenized_paragraphs]
    end

    return (paragraphs=paragraphs_filtered, # Return the filtered paragraphs
            tokenized_paragraphs=tokenized_paragraphs,
            paragraph_token_ids=para_token_ids,
            vocabulary=para_vocab,
            cleaned_text=cleaned_text)
end




"""
    preprocess_for_document_embeddings(corpus_input::Union{AbstractString, String};
                                       build_vocab::Bool=false,
                                       vocab_options::Dict=Dict())
                                       -> @NamedTuple{tokens::Vector{String},
                                                      cleaned_text::String,
                                                      token_ids::Union{Vector{Int}, Nothing},
                                                      vocabulary::Union{Vocabulary, Nothing}}

Prepares a document (from string or file path) for document-level embeddings.
Applies common cleaning and tokenization defaults (lowercasing, punctuation removal, lemmatization).
Optionally builds a vocabulary and returns token IDs.

Pipeline (Defaults Applied):
1. Reads document if `corpus_input` is a file path.
2. Cleans text: `clean_text(..., unicode_normalize=true, case_transform=:lower, remove_punctuation=true, remove_symbols=false, remove_emojis=true)`
3. Tokenizes the entire cleaned text: `tokenize(..., strip_punctuation=true, lower=true, lemmatize=true, remove_stopwords=false, stem=false)`
4. (Optional) Builds vocabulary: `build_vocabulary(...)` using `vocab_options` if `build_vocab=true`.
5. (Optional) Converts tokens to IDs: `tokens_to_ids(...)` if `build_vocab=true`.

# Arguments
- `corpus_input::Union{AbstractString, String}`: The input document text (as a single string) or the path to a text file containing the document.

# Keyword Arguments
- `build_vocab::Bool`: If `true`, builds a vocabulary from the document tokens and returns token IDs. Defaults to `false`.
- `vocab_options::Dict`: Options passed directly to `Vocabulary.build_vocabulary` if `build_vocab=true` (e.g., `Dict(:min_freq=>1, :special_tokens=>["<unk>"])`). Ensure `<unk>` is included.

# Returns
- `NamedTuple`: A named tuple containing:
    - `tokens::Vector{String}`: The flat list of tokens after cleaning and tokenization.
    - `cleaned_text::String`: The document text after cleaning.
    - `token_ids::Union{Vector{Int}, Nothing}`: The flat list of token IDs (or `nothing`).
    - `vocabulary::Union{Vocabulary, Nothing}`: The built `Vocabulary` object (or `nothing`).

# Examples

```julia
# Example 1: Basic usage, get tokens, don't build vocab
text = "This is the entire document. It has two sentences."
result = preprocess_for_document_embeddings(text)

# Expected output:
# (tokens = ["this", "be", "the", "entire", "document", "it", "have", "two", "sentence"], 
#  cleaned_text = "this be the entire document it have two sentence", 
#  token_ids = nothing, 
#  vocabulary = nothing)
```

```julia
# Example 2: Build vocabulary and get token IDs
text = "Another document. It is short."
result = preprocess_for_document_embeddings(text; 
   build_vocab=true, 
   vocab_options=Dict(:min_freq=>1)
)

# Expected output structure (IDs depend on generated vocab):
# (tokens = ["another", "document", "it", "be", "short"], 
#  cleaned_text = "another document it be short", 
#  token_ids = [id1, id2, id3, id4, id5], 
#  vocabulary = Vocabulary(...))
```
"""
function preprocess_for_document_embeddings(corpus_input::Union{AbstractString, String};
                                            build_vocab::Bool=false,
                                            vocab_options::Dict=Dict())

    default_clean_options = Dict(
        :unicode_normalize => true,
        :remove_accents => false,
        :remove_punctuation => true,
        :remove_symbols => false,
        :remove_emojis => true,
        :case_transform => :lower
    )
    # Tokenize the whole document as one sequence
    default_tokenize_options = Dict(
        :strip_punctuation => true,
        :lower => true,
        :remove_stopwords => false,
        :lemmatize => true,
        :stem => false
    )
    default_id_options = Dict(:add_new=>false, :update_counts=>true)

    # Read Corpus (if file path)
    text = if isfile(corpus_input)
        read(corpus_input, String)
    else
        String(corpus_input) # Ensure it's a String
    end

    # Cleaning (using defaults)
    cleaned_text = clean_text(text; default_clean_options...)

    # Tokenize the entire cleaned text (using defaults)
    doc_tokens = Tokenization.tokenize(cleaned_text; default_tokenize_options...)

    # Optional Vocabulary Building and ID Conversion
    doc_vocab = nothing
    doc_token_ids = nothing

    if build_vocab
        # Ensure <unk> is present for vocabulary building
        vocab_build_opts = merge(Dict(:special_tokens => ["<unk>"]), vocab_options)
        if !("<unk>" in get(vocab_build_opts, :special_tokens, []))
            push!(vocab_build_opts[:special_tokens], "<unk>")
        end

        vocab_dict = build_vocabulary(doc_tokens; vocab_build_opts...)

        # Create Vocabulary struct instance
        unk_token = "<unk>"
        unk_id = get(vocab_dict["token_to_index"], unk_token, 0)
        if unk_id == 0
            @warn "\'<unk>\' token not found in generated vocabulary, using ID 0. Ensure \'<unk>\' is in special_tokens."
        end
        doc_vocab = VocabularyModule.Vocabulary(vocab_dict["token_to_index"],
                                            vocab_dict["index_to_token"],
                                            Dict{Int,Int}(), # Initialize counts
                                            unk_id)

        # Convert tokens to IDs using the built vocab
        doc_token_ids = Tokenization.tokens_to_ids(doc_tokens, doc_vocab; default_id_options...)  #TODO: replace with convert_tokens_to_ids
    end

    return (tokens=doc_tokens,
            cleaned_text=cleaned_text,
            token_ids=doc_token_ids,
            vocabulary=doc_vocab)
end



end #END MODULE

