module Preprocessing

include(joinpath(@__DIR__, "Vocabulary.jl"))          #  -> Vocabulary, convert_tokens_to_ids, …
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


# Export only the high-level pipeline functions
export preprocess_for_char_embeddings,
       preprocess_for_sentence_embeddings,
       preprocess_for_paragraph_embeddings,
       preprocess_for_subword_embeddings,
       preprocess_for_document_embeddings,
       preprocess_for_word_embeddings

# --- Pipeline Function Implementations ---

"""
    preprocess_for_char_embeddings(text::String, vocab::Vocabulary; clean_options=Dict(), char_options=Dict(), id_options=Dict()) -> Vector{Int}

Prepares text for character-level embeddings.

Pipeline:
1. (Optional) Cleans text using `CleanText.clean_text`.
2. Tokenizes into characters (graphemes) using `CharProcessing.tokenize_char`.
3. Converts characters to IDs using `CharProcessing.chars_to_ids`.

# Arguments
- `text::String`: The input text.
- `vocab::Vocabulary`: A pre-built character vocabulary.

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text` (e.g., `Dict(:case_transform=>:lower)`). Set to `nothing` to skip cleaning.
- `char_options::Dict`: Options passed directly to `CharProcessing.tokenize_char` (e.g., `Dict(:keep_space=>true)`).
- `id_options::Dict`: Options passed directly to `CharProcessing.chars_to_ids` (e.g., `Dict(:add_new=>false)`).

# Returns
- `Vector{Int}`: A vector of character IDs.
"""
function preprocess_for_char_embeddings(text::String, vocab::Vocabulary;
                                        clean_options::Union{Dict, Nothing}=Dict(),
                                        char_options::Dict=Dict(),
                                        id_options::Dict=Dict())
    # 1. Optional Cleaning
    processed_text = if clean_options !== nothing
        clean_text(text; clean_options...)
    else
        text
    end

    # 2. Character Tokenization
    chars = tokenize_char(processed_text; char_options...)

    # 3. Convert to IDs
    ids = chars_to_ids(chars, vocab; id_options...)

    return ids
end

"""
    preprocess_for_sentence_embeddings(text::String; clean_options=Dict(), sentence_options=Dict()) -> Vector{String}

Prepares text for sentence-level embeddings (e.g., Sentence-BERT).

Pipeline:
1. Cleans text using `CleanText.clean_text`.
2. Splits cleaned text into sentences using `SentenceProcessing.split_sentences`.

# Arguments
- `text::String`: The input text.

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text` (e.g., `Dict(:remove_punctuation=>false)`).
- `sentence_options::Dict`: Options passed directly to `SentenceProcessing.split_sentences`.

# Returns
- `Vector{String}`: A vector of cleaned sentences.
"""
function preprocess_for_sentence_embeddings(text::String;
                                            clean_options::Dict=Dict(),
                                            sentence_options::Dict=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    # 2. Sentence Splitting
    sentences = split_sentences(cleaned_text; sentence_options...)

    return sentences
end

"""
    preprocess_for_paragraph_embeddings(text::String; clean_options=Dict(), paragraph_options=Dict(), filter_options=Dict()) -> Vector{String}

Prepares text for paragraph-level embeddings.

Pipeline:
1. Cleans text using `CleanText.clean_text`.
2. Splits cleaned text into paragraphs using `ParagraphProcessing.split_paragraphs`.
3. (Optional) Filters paragraphs using `ParagraphProcessing.filter_paragraphs`.

# Arguments
- `text::String`: The input text.

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text`.
- `paragraph_options::Dict`: Options passed directly to `ParagraphProcessing.split_paragraphs` (e.g., `Dict(:unwrap=>true)`).
- `filter_options::Union{Dict, Nothing}`: Options passed directly to `ParagraphProcessing.filter_paragraphs` (e.g., `Dict(:min_chars=>25)`). Set to `nothing` to skip filtering.

# Returns
- `Vector{String}`: A vector of cleaned (and potentially filtered) paragraphs.
"""
function preprocess_for_paragraph_embeddings(text::String;
                                             clean_options::Dict=Dict(),
                                             paragraph_options::Dict=Dict(),
                                             filter_options::Union{Dict, Nothing}=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    # 2. Paragraph Splitting
    paragraphs = split_paragraphs(cleaned_text; paragraph_options...)

    # 3. Optional Filtering
    if filter_options !== nothing
        paragraphs = filter_paragraphs(paragraphs; filter_options...)
    end

    return paragraphs
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

# Returns
- `Vector{Vector{Int}}`: A vector of vectors of subword IDs (one inner vector per sentence).
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
    preprocess_for_document_embeddings(text::String; clean_options=Dict()) -> String

Prepares text for document-level embeddings (e.g., Doc2Vec) by cleaning.

Pipeline:
1. Cleans text using `CleanText.clean_text`.

Note: Some document embedding methods might require tokenized input. Consider using `preprocess_for_word_embeddings` and flattening the result, or modifying this function if needed.

# Arguments
- `text::String`: The input text.

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text` (e.g., `Dict(:remove_punctuation=>true, :case_transform=>:lower)`).

# Returns
- `String`: The cleaned document text.
"""
function preprocess_for_document_embeddings(text::String; clean_options::Dict=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    return cleaned_text
end

"""
    preprocess_for_word_embeddings(text::String, vocab::Vocabulary; clean_options=Dict(), sentence_options=Dict(), tokenize_options=Dict(), id_options=Dict()) -> Vector{Vector{Int}}

Prepares text for word-level embeddings (e.g., Word2Vec, GloVe).

Pipeline:
1. Cleans text using `CleanText.clean_text`.
2. Splits cleaned text into sentences using `SentenceProcessing.split_sentences`.
3. Tokenizes each sentence into words using `Tokenization.tokenize`.
4. Converts word tokens to IDs using `Tokenization.tokens_to_ids`.

# Arguments
- `text::String`: The input text.
- `vocab::Vocabulary`: A pre-built word vocabulary.

# Keyword Arguments
- `clean_options::Dict`: Options passed directly to `CleanText.clean_text`.
- `sentence_options::Dict`: Options passed directly to `SentenceProcessing.split_sentences`.
- `tokenize_options::Dict`: Options passed directly to `Tokenization.tokenize` for each sentence (e.g., `Dict(:remove_stopwords=>true, :lemmatize=>true)`).
- `id_options::Dict`: Options passed directly to `Tokenization.tokens_to_ids` for each sentence (e.g., `Dict(:add_new=>false)`).

# Returns
- `Vector{Vector{Int}}`: A vector of vectors of word token IDs (one inner vector per sentence).
"""
function preprocess_for_word_embeddings(text::String, vocab::Vocabulary;
                                        clean_options::Dict=Dict(),
                                        sentence_options::Dict=Dict(),
                                        tokenize_options::Dict=Dict(),
                                        id_options::Dict=Dict())
    # 1. Cleaning
    cleaned_text = clean_text(text; clean_options...)

    # 2. Sentence Splitting
    sentences = split_sentences(cleaned_text; sentence_options...)

    # 3. Tokenize each sentence
    tokenized_sentences = [Tokenization.tokenize(sentence; tokenize_options...) for sentence in sentences]

    # 4. Convert tokens to IDs for each sentence
    sentence_ids = [Tokenization.tokens_to_ids(tokens, vocab; id_options...) for tokens in tokenized_sentences]

    return sentence_ids
end






end #END MODULE

