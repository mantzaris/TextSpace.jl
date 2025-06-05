"""
LearnBPE.jl - Standalone Byte Pair Encoding Learning Module

This module provides functionality to learn BPE tokenizers from scratch,
save them to files, and load them back. It's designed to be used independently
from other BPE implementations.

Usage:
    include("LearnBPE.jl")
    using .LearnBPE: learn_bpe, save_bpe, load_bpe

"""

module LearnBPE

using ...Plumbing
using JSON3

export BPETokeniser, learn_bpe, save_bpe, load_bpe


struct BPETokeniser
    merges :: Vector{Tuple{String,String}}
    vocab  :: Union{Dict{String,Int},Nothing}
end

"""
    learn_bpe(corpus; vocab_size=10000, min_frequency=2, special_tokens=["<unk>", "<s>", "</s>", "<pad>"])

Learn BPE merges and vocabulary from a text corpus using the Byte Pair Encoding algorithm.

# Arguments
- `corpus`: Input text corpus (Vector{String}, String, or file path)
- `vocab_size::Int=10000`: Target vocabulary size
- `min_frequency::Int=2`: Minimum frequency for a pair to be considered for merging
- `special_tokens::Vector{String}`: Special tokens to include in vocabulary

# Returns
- `BPETokeniser`: A tokenizer object with learned merges and vocabulary

# Examples
```julia
# From vector of strings
corpus = ["hello world", "hello there", "world peace"]
tokenizer = learn_bpe(corpus, vocab_size=1000)

# From single string
text = "machine learning natural language processing"
tokenizer = learn_bpe(text, vocab_size=500)

# From file
tokenizer = learn_bpe("path/to/corpus.txt", vocab_size=5000)

# With custom parameters
tokenizer = learn_bpe(
    corpus,
    vocab_size=8000,
    min_frequency=3,
    special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask>"]
)
```
"""
function learn_bpe(corpus; 
                   vocab_size::Int=10000, 
                   min_frequency::Int=2,
                   special_tokens::Vector{String}=["<unk>", "<s>", "</s>", "<pad>"])
    
    # Handle different input types
    if isa(corpus, String)
        if isfile(corpus)
            # Read from file
            println("Reading corpus from file: $corpus")
            corpus_text = read(corpus, String)
            corpus_lines = [corpus_text]
        else
            # Single string
            corpus_lines = [corpus]
        end
    elseif isa(corpus, AbstractVector{<:AbstractString})
        corpus_lines = corpus
    else
        error("Corpus must be a String, AbstractVector{<:AbstractString}, or file path")
    end
    
    println("Starting BPE learning on corpus of $(length(corpus_lines)) documents...")
    
    # Step 1: Preprocess corpus and initialize word frequencies
    word_freqs = _preprocess_corpus(corpus_lines)
    println("Preprocessed corpus: $(length(word_freqs)) unique words")
    
    # Step 2: Initialize vocabulary with characters and special tokens
    vocab = _initialize_vocabulary(word_freqs, special_tokens)
    println("Initial vocabulary size: $(length(vocab))")
    
    # Step 3: Learn BPE merges iteratively
    merges = _learn_merges(word_freqs, vocab, vocab_size, min_frequency)
    println("Learned $(length(merges)) merge operations")
    
    # Step 4: Create final vocabulary
    final_vocab = _create_final_vocabulary(vocab, merges, special_tokens)
    println("Final vocabulary size: $(length(final_vocab))")
    
    return BPETokeniser(merges, final_vocab)
end

"""
    save_bpe(tokenizer::BPETokeniser, filepath::String; format::String="json")

Save a BPE tokenizer to a file.

# Arguments
- `tokenizer::BPETokeniser`: The tokenizer to save
- `filepath::String`: Path where to save the tokenizer
- `format::String="json"`: Output format ("json" or "txt")

# Examples
```julia
tokenizer = learn_bpe(corpus, vocab_size=5000)

# Save as JSON (recommended)
save_bpe(tokenizer, "my_tokenizer.json")

# Save as text files (separate merges and vocab)
save_bpe(tokenizer, "my_tokenizer", format="txt")
```
"""
function save_bpe(tokenizer::BPETokeniser, filepath::String; format::String="json")
    if format == "json"
        _save_bpe_json(tokenizer, filepath)
    elseif format == "txt"
        _save_bpe_txt(tokenizer, filepath)
    else
        error("Unsupported format: $format. Use 'json' or 'txt'")
    end
    
    println("Tokenizer saved to: $filepath")
end

"""
    load_bpe(filepath::String) -> BPETokeniser

Load a BPE tokenizer from a file.

# Arguments
- `filepath::String`: Path to the saved tokenizer file

# Returns
- `BPETokeniser`: The loaded tokenizer

# Examples
```julia
# Load from JSON
tokenizer = load_bpe("my_tokenizer.json")

# Load from text files (will look for .merges.txt and .vocab.json)
tokenizer = load_bpe("my_tokenizer")
```
"""
function load_bpe(filepath::String)::BPETokeniser
    if endswith(filepath, ".json")
        return _load_bpe_json(filepath)

    # recognise *either* naming convention 
    elseif  isfile(filepath * ".merges.txt") || isfile(filepath * "_merges.txt") ||
            isfile(filepath * ".vocab.json") || isfile(filepath * "_vocab.json")

        return _load_bpe_txt(filepath)
    else
        error("Could not find tokenizer files at: $filepath")
    end
end
# function load_bpe(filepath::String)::BPETokeniser
#     if endswith(filepath, ".json")
#         return _load_bpe_json(filepath)
#     elseif isfile(filepath * ".merges.txt") || isfile(filepath * ".vocab.json")
#         return _load_bpe_txt(filepath)
#     else
#         error("Could not find tokenizer files at: $filepath")
#     end
# end

# ============================================================================
# Internal Helper Functions
# ============================================================================

"""
    _preprocess_corpus(corpus::Vector{String}) -> Dict{Vector{String}, Int}

Preprocess the corpus by tokenizing into words and counting frequencies.
Each word is split into characters with end-of-word marker.
"""
function _preprocess_corpus(corpus::AbstractVector{<:AbstractString})
    word_freqs = Dict{Vector{String}, Int}()
    
    for text in corpus
        # Simple tokenization: split on whitespace and punctuation
        words = Plumbing.tokenize(text; lower=true, strip_punctuation=true) 
        
        for word in words
            if !isempty(word)
                # Split word into characters and add end-of-word marker
                chars = [string(c) for c in word]
                push!(chars, "</w>")  # End-of-word marker
                
                # Count frequency
                word_freqs[chars] = get(word_freqs, chars, 0) + 1
            end
        end
    end
    
    return word_freqs
end


"""
    _initialize_vocabulary(word_freqs::Dict{Vector{String}, Int}, 
                          special_tokens::Vector{String}) -> Set{String}

Initialize vocabulary with all characters found in the corpus plus special tokens.
"""
function _initialize_vocabulary(word_freqs::Dict{Vector{String}, Int}, special_tokens::Vector{String})
    vocab = Set{String}()
    
    # Add special tokens
    for token in special_tokens
        push!(vocab, token)
    end
    
    # Add all characters found in the corpus
    for word_chars in keys(word_freqs)
        for char in word_chars
            push!(vocab, char)
        end
    end
    
    return vocab
end

"""
    _learn_merges(word_freqs::Dict{Vector{String}, Int}, vocab::Set{String}, 
                  vocab_size::Int, min_frequency::Int) -> Vector{Tuple{String,String}}

Learn BPE merges by iteratively finding and merging the most frequent pairs.
"""
function _learn_merges(word_freqs::Dict{Vector{String}, Int}, vocab::Set{String}, 
                       vocab_size::Int, min_frequency::Int)
    merges = Vector{Tuple{String,String}}()
    current_word_freqs = deepcopy(word_freqs)
    
    while length(vocab) < vocab_size
        # Find the most frequent pair
        pair_freqs = _count_pairs(current_word_freqs)
        
        if isempty(pair_freqs)
            break
        end
        
        # Get the most frequent pair
        best_pair = _get_most_frequent_pair(pair_freqs, min_frequency)
        
        if best_pair === nothing
            break
        end
        
        # Merge the pair in all words
        current_word_freqs = _merge_pair(current_word_freqs, best_pair)
        
        # Add merge to list and new token to vocabulary
        push!(merges, best_pair)
        new_token = best_pair[1] * best_pair[2]
        push!(vocab, new_token)
        
        if length(merges) % 100 == 0
            println("  Learned $(length(merges)) merges, vocab size: $(length(vocab))")
        end
    end
    
    return merges
end

"""
    _count_pairs(word_freqs::Dict{Vector{String}, Int}) -> Dict{Tuple{String,String}, Int}

Count the frequency of all adjacent pairs in the current word representations.
"""
function _count_pairs(word_freqs::Dict{Vector{String}, Int})
    pair_freqs = Dict{Tuple{String,String}, Int}()
    
    for (word_chars, freq) in word_freqs
        for i in 1:(length(word_chars)-1)
            pair = (word_chars[i], word_chars[i+1])
            pair_freqs[pair] = get(pair_freqs, pair, 0) + freq
        end
    end
    
    return pair_freqs
end

"""
    _get_most_frequent_pair(pair_freqs::Dict{Tuple{String,String}, Int}, 
                           min_frequency::Int) -> Union{Tuple{String,String}, Nothing}

Get the most frequent pair that meets the minimum frequency threshold.
"""
function _get_most_frequent_pair(pair_freqs::Dict{Tuple{String,String}, Int}, min_frequency::Int)
    if isempty(pair_freqs)
        return nothing
    end
    
    # Filter pairs by minimum frequency
    valid_pairs = filter(p -> p.second >= min_frequency, pair_freqs)
    
    if isempty(valid_pairs)
        return nothing
    end
    
    # Return the pair with maximum frequency
    return argmax(valid_pairs)
end

"""
    _merge_pair(word_freqs::Dict{Vector{String}, Int}, 
                pair::Tuple{String,String}) -> Dict{Vector{String}, Int}

Merge all occurrences of the given pair in all words.
"""
function _merge_pair(word_freqs::Dict{Vector{String}, Int}, pair::Tuple{String,String})
    new_word_freqs = Dict{Vector{String}, Int}()
    
    for (word_chars, freq) in word_freqs
        new_chars = _merge_pair_in_word(word_chars, pair)
        new_word_freqs[new_chars] = get(new_word_freqs, new_chars, 0) + freq
    end
    
    return new_word_freqs
end

"""
    _merge_pair_in_word(word_chars::Vector{String}, 
                        pair::Tuple{String,String}) -> Vector{String}

Merge all occurrences of the pair in a single word.
"""
function _merge_pair_in_word(word_chars::Vector{String}, pair::Tuple{String,String})
    if length(word_chars) < 2
        return word_chars
    end
    
    new_chars = String[]
    i = 1
    
    while i <= length(word_chars)
        if i < length(word_chars) && word_chars[i] == pair[1] && word_chars[i+1] == pair[2]
            # Merge the pair
            push!(new_chars, pair[1] * pair[2])
            i += 2
        else
            # Keep the character as is
            push!(new_chars, word_chars[i])
            i += 1
        end
    end
    
    return new_chars
end

"""
    _create_final_vocabulary(vocab::Set{String}, merges::Vector{Tuple{String,String}}, 
                            special_tokens::Vector{String}) -> Dict{String,Int}

Create the final vocabulary dictionary with token-to-ID mappings.
"""
function _create_final_vocabulary(vocab::Set{String}, merges::Vector{Tuple{String,String}}, 
                                 special_tokens::Vector{String})
    final_vocab = Dict{String,Int}()
    
    # Start with special tokens (they get the first IDs)
    id = 0
    for token in special_tokens
        if token in vocab
            final_vocab[token] = id
            id += 1
        end
    end
    
    # Add remaining vocabulary items
    remaining_tokens = setdiff(vocab, Set(special_tokens))
    for token in sort(collect(remaining_tokens))  # Sort for deterministic ordering
        final_vocab[token] = id
        id += 1
    end
    
    return final_vocab
end

# ============================================================================
# Save/Load Helper Functions
# ============================================================================
function save_bpe(tokenizer::BPETokeniser,
                  filepath::String;
                  format::Union{String,Symbol} = "json")

    # normalise the keyword to a lower-case String
    fmt = lowercase(String(format))

    if fmt == "json" || fmt == "both"
        _save_bpe_json(tokenizer, endswith(filepath, ".json") ?
                                   filepath : filepath * ".json")
    end

    if fmt == "txt"  || fmt == "both"
        stem = endswith(filepath, ".json") ? filepath[1:end-5] : filepath
        _save_bpe_txt(tokenizer, stem)           # produces   stem.merges.txt + stem.vocab.json
    end

    fmt ∈ ("json", "txt", "both") ||                       # final guard
        error("Unsupported format ‘$(format)’. Choose :json, :txt or :both")
end


"""
    _save_bpe_json(tokenizer::BPETokeniser, filepath::String)

Save tokenizer to JSON format (HuggingFace compatible).
"""
function _save_bpe_json(tokenizer::BPETokeniser, filepath::String)
    # Ensure .json extension
    if !endswith(filepath, ".json")
        filepath = filepath * ".json"
    end
    
    # Create HuggingFace-compatible format
    data = Dict(
        "version" => "1.0",
        "model" => Dict(
            "type" => "BPE",
            "merges" => [collect(merge) for merge in tokenizer.merges],
            "vocab" => tokenizer.vocab
        ),
        "added_tokens" => [],
        "normalizer" => nothing,
        "pre_tokenizer" => nothing,
        "post_processor" => nothing,
        "decoder" => nothing
    )
    
    # Write to file
    open(filepath, "w") do f
        JSON3.pretty(f, data)
    end
end

"""
    _save_bpe_txt(tokenizer::BPETokeniser, filepath::String)

Save tokenizer to text format (separate merges and vocab files).
"""
function _save_bpe_txt(tokenizer::BPETokeniser, stem::AbstractString)
    # GPT-2 convention: “_merges.txt”, optional “_vocab.json”
    merges_path = stem * "_merges.txt"
    vocab_path  = stem * "_vocab.json"

    # ── 1. merges.txt ───────────────────────────────────────────────
    open(merges_path, "w") do io
        println(io, "#version: 1.0")
        for (l,r) in tokenizer.merges
            println(io, l, ' ', r)
        end
    end

    # ── 2. vocab.json (only if present) ─────────────────────────────
    if tokenizer.vocab !== nothing
        open(vocab_path, "w") do io
            JSON3.pretty(io, tokenizer.vocab)
        end
    end
end

"""
    _load_bpe_json(filepath::String) -> BPETokeniser

Load tokenizer from JSON format.
"""
function _load_bpe_json(filepath::String)
    data = JSON3.read(read(filepath, String))
    
    # Extract merges
    merges = Vector{Tuple{String,String}}()
    if haskey(data, :model) && haskey(data[:model], :merges)
        for merge_pair in data[:model][:merges]
            push!(merges, (String(merge_pair[1]), String(merge_pair[2])))
        end
    end
    
    # Extract vocabulary
    vocab = nothing
    if haskey(data, :model) && haskey(data[:model], :vocab) && data[:model][:vocab] !== nothing
        vocab = Dict{String,Int}(String(k) => Int(v) for (k,v) in pairs(data[:model][:vocab]))
    end
    
    return BPETokeniser(merges, vocab)
end

"""
    _load_bpe_txt(filepath::String) -> BPETokeniser

Load tokenizer from text format (separate merges and vocab files).
"""
function _load_bpe_txt(stem::AbstractString)
    # figure-out file names 
    merges_file = if isfile(stem * "_merges.txt")
        stem * "_merges.txt"
    elseif isfile(stem * ".merges.txt")
        stem * ".merges.txt"
    else
        error("No merges file found for tokenizer stem “$(stem)”")
    end

    vocab_file  =  isfile(stem * "_vocab.json")  ? stem * "_vocab.json" :
                  isfile(stem * ".vocab.json")   ? stem * ".vocab.json" :
                  nothing            # vocab is optional (GPT-2 style)

    # read merges
    merges = Tuple{String,String}[]
    for line in eachline(merges_file)
        line = strip(line)
        (isempty(line) || startswith(line, '#')) && continue
        parts = split(line)
        length(parts) >= 2 && push!(merges, (parts[1], parts[2]))
    end

    # read vocab (if present)
    vocab = vocab_file === nothing ? nothing :
            Dict{String,Int}(String(k) => Int(v) for
                             (k,v) in pairs(JSON3.read(read(vocab_file,String))))

    return BPETokeniser(merges, vocab)
end


end # module LearnBPE

