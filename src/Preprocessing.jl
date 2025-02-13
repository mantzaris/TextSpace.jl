


"""
    clean_text(text::String; normalize::Bool=true, remove_punctuation::Bool=false, remove_emojis::Bool=false) -> String

Cleans the given text by:
- Optionally normalizing Unicode characters.
- Converting the text to lowercase.
- Optionally removing punctuation.
- Optionally removing emojis.
- Normalizing whitespace.

# Examples
```julia
julia> clean_text("Hello, Julia! 😊", remove_punctuation=true, remove_emojis=true)
"hello julia"

"""
function clean_text(text::String; 
                    normalize::Bool=true, 
                    remove_punctuation::Bool=false, remove_emojis::Bool=false) 
    #optionally normalize Unicode (NFC) 
    if normalize 
        text = Unicode.normalize(text, :NFC) 
    end

    #convert to lowercase
    text_clean = lowercase(text)

    #optionally remove punctuation
    if remove_punctuation
        #replace any Unicode punctuation character with an empty string.
        text_clean = replace(text_clean, r"\p{P}" => "")
    end

    #optionally remove emojis
    if remove_emojis
        # Define a helper function to check if a character is an emoji
        is_emoji(c::Char) = begin
            cp = UInt32(c)
            return (cp >= 0x1F600 && cp <= 0x1F64F) ||  # Emoticons
                   (cp >= 0x1F300 && cp <= 0x1F5FF) ||  # Misc Symbols and Pictographs
                   (cp >= 0x1F680 && cp <= 0x1F6FF) ||  # Transport and Map Symbols
                   (cp >= 0x2600 && cp <= 0x26FF)   ||  # Misc Symbols
                   (cp >= 0x2700 && cp <= 0x27BF)         # Dingbats
        end
        text_clean = join(filter(c -> !is_emoji(c), text_clean))
    end

    #normalize whitespace (replace multiple spaces with a single space, and trim)
    text_clean = strip(replace(text_clean, r"\s+" => " "))

    return text_clean
end


"""
    tokenize(text::String; mode::Symbol=:word) -> Vector{String}

Tokenizes the input text into a vector of tokens.  
- If `mode` is `:word` (default), the text is split into word-level tokens.
- If `mode` is `:char`, the text is split into individual characters.

# Examples
```julia
julia> tokenize("Hello, Julia! 😊", mode=:word)
["hello", ",", "julia", "!", "😊"]

julia> tokenize("Hello, Julia! 😊", mode=:char)
['h', 'e', 'l', 'l', 'o', ',', ' ', 'j', 'u', 'l', 'i', 'a', '!', ' ', '😊']

""" 
function tokenize(text::String; mode::Symbol = :word) 
    if mode == :word 
        #Unicode-aware regex: # - [\p{L}\p{N}]+ matches sequences of letters, numbers, or underscores. 
        #[^\s\p{L}\p{N}]+ matches one or more characters that are not whitespace or part of a word (e.g., punctuation, symbols, emoji). 
        pattern = r"[\p{L}\p{N}]+|[^\s\p{L}\p{N}]+"
        tokens = [m.match for m in eachmatch(pattern, text)]
        return tokens 
    elseif mode == :char 
        return collect(text) 
    else 
        error("Unsupported tokenization mode: $mode. Use :word or :char.") 
    end 
end


function build_vocabulary(tokens::Vector{String};
    min_freq::Int = 0,
    max_vocab_size::Int = typemax(Int),
    special_tokens::Vector{String} = String[])

    #count the frequency of each token
    freq = Dict{String, Int}()
    for token in tokens
        freq[token] = get(freq, token, 0) + 1
    end

    #filter tokens based on the minimum frequency
    filtered_tokens = Dict{String, Int}()
    for (token, count) in freq
        if count >= min_freq
            filtered_tokens[token] = count
        end
    end

    #sort tokens by frequency in descending order
    sorted_tokens = sort(collect(keys(filtered_tokens)), by = x -> filtered_tokens[x], rev = true)

    #limit the vocabulary size (if necessary)
    if length(sorted_tokens) > max_vocab_size
        sorted_tokens = sorted_tokens[1:max_vocab_size]
    end

    #create token-to-index mapping, reserving indices for special tokens first (if any)
    token_to_index = Dict{String, Int}()
    index_to_token = String[]

    #add special tokens (if provided)
    for token in special_tokens
        push!(index_to_token, token)
        token_to_index[token] = length(index_to_token)
    end

    # Add remaining tokens, ensuring no duplicates with special tokens
    for token in sorted_tokens
        if !haskey(token_to_index, token)
            push!(index_to_token, token)
            token_to_index[token] = length(index_to_token)
        end
    end

    return Dict("token_to_index" => token_to_index,
                    "index_to_token" => index_to_token,
                    "freq"           => freq)
end

