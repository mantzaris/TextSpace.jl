module LoadBPE
export BPETokeniser, load_bpe, testo

import ...resource      # helper for bundled paths
using JSON3
using Base: basename


struct BPETokeniser
    merges :: Vector{Tuple{String,String}}
    vocab  :: Union{Dict{String,Int},Nothing}
end

const _ARTEFACTS = Dict(
    "gpt2_merges.txt"               => (; merges="gpt2_merges.txt",          vocab=nothing),
    "mGPT_61Lang1pnt9M_merges.txt"  => (; merges="mGPT_61Lang1pnt9M_merges.txt",
                                         vocab ="mGPT_61Lang1pnt9M_vocab.json"),
    "RoBERTa-base_merges.txt"       => (; merges="RoBERTa-base_merges.txt",
                                         vocab ="RoBERTa-base_vocab.json"),
    "Mistral-24B_32_768ctrl.json"   => (; merges="gpt2_merges.txt",          # fallback
                                         vocab ="Mistral-24B_32_768ctrl.json"),
    "XML-RoBERTa_100Lang.json"      => (; merges="gpt2_merges.txt",          # fallback
                                         vocab ="XML-RoBERTa_100Lang.json"),
)

#  helper to read GPT-2 style merges.txt 
function _read_merges(txtfile)
    lines = readlines(txtfile)
    merges = Tuple{String,String}[]
    
    for line in lines
        line = strip(line)
        # Skip empty lines and header lines (starting with #)
        if isempty(line) || startswith(line, '#')
            continue
        end
        
        parts = split(line)
        if length(parts) >= 2
            # Take only the first two parts for the merge
            push!(merges, (parts[1], parts[2]))
        end
        # Skip lines that don't have at least 2 parts
    end
    
    return merges
end

# Fixed version of _to_vocab that handles different JSON structures
function _to_vocab(obj)
    if isa(obj, AbstractDict)
        # Handle regular dict format (like RoBERTa-base_vocab.json)
        return Dict{String,Int}(String(k) => Int(v) for (k,v) in pairs(obj))
    else
        error("Unsupported vocab format: expected Dict, got $(typeof(obj))")
    end
end

_to_merges(arr) = Tuple{String,String}.( (s[1], s[2]) for s in arr )


function _to_str_int(obj)
    if isa(obj, AbstractDict)
        # Regular dict format: {"token": id, ...}
        return Dict{String,Int}(string(k) => Int(v) for (k,v) in pairs(obj))
    elseif isa(obj, AbstractVector)
        # List format: [["token", score], ...] or [["token", id], ...]
        vocab_dict = Dict{String,Int}()
        for (i, item) in enumerate(obj)
            if isa(item, AbstractVector) && length(item) >= 1
                # Format: [["token", score], ...] - use index as ID
                token = string(item[1])
                # Use index as ID (0-based to match typical tokenizer behavior)
                vocab_dict[token] = i - 1
            elseif isa(item, Tuple) && length(item) >= 2
                # Format: [("token", id), ...]
                token = string(item[1])
                id = isa(item[2], Integer) ? Int(item[2]) : i - 1
                vocab_dict[token] = id
            else
                error("Unsupported vocab item format: $(typeof(item))")
            end
        end
        return vocab_dict
    else
        error("Unsupported vocab format in JSON: expected Dict or Vector, got $(typeof(obj))")
    end
end

# helper - keep only the first two items of every merge rule
_normalise_merges(arr) =
    Tuple{String,String}.( (String(r[1]), String(r[2])) for r in arr )


function load_bpe(spec::AbstractString)::BPETokeniser
    name = basename(spec)

    if name == "gpt2_merges.txt"
        merges = _read_merges(resource("gpt2_merges.txt"))
        return BPETokeniser(merges, nothing)

    elseif name == "RoBERTa-base_merges.txt"
        merges = _read_merges(resource("RoBERTa-base_merges.txt"))
        raw   = JSON3.read(read(resource("RoBERTa-base_vocab.json"), String))
        vocab  = _to_vocab(raw)
        return BPETokeniser(merges, vocab)

    elseif name == "mGPT_61Lang1pnt9M_merges.txt"
        merges = _read_merges(resource("mGPT_61Lang1pnt9M_merges.txt"))
        raw   = JSON3.read(read(resource("mGPT_61Lang1pnt9M_vocab.json"), String))
        vocab  = _to_vocab(raw)
        return BPETokeniser(merges, vocab)

    # user-supplied file path
    elseif isfile(spec)
        return _load_external_bpe(spec)   # as in your current file
    else
        error("Unknown BPE spec '$(spec)'.  Use one of the bundled names or an existing file.")
    end
end

function _load_external_bpe(path::AbstractString)
    if endswith(path, ".txt")
        return BPETokeniser(_read_merges(path), nothing)
    elseif endswith(path, ".json")
        data = JSON3.read(read(path, String))
        
        # Handle different JSON structures
        if haskey(data, :model)
            # HuggingFace tokenizer format
            model = data[:model]
            
            # Extract vocab
            vocab = nothing
            if haskey(model, :vocab)
                vocab_data = model[:vocab]
                vocab = _to_str_int(vocab_data)
            end
            
            # Extract merges
            merges = Tuple{String,String}[]
            if haskey(model, :merges)
                merges_data = model[:merges]
                if isa(merges_data, AbstractVector)
                    merges = _normalise_merges(merges_data)
                end
            end
            
            return BPETokeniser(merges, vocab)
            
        elseif haskey(data, "merges") || haskey(data, :merges)
            # Simple format with merges at top level
            merges_key = haskey(data, "merges") ? "merges" : :merges
            merges = Tuple{String,String}.(data[merges_key])
            return BPETokeniser(merges, nothing)
        else
            # Assume it's a simple vocab dict
            vocab = _to_str_int(data)
            return BPETokeniser(Tuple{String,String}[], vocab)
        end
    end
    error("Could not interpret external BPE file '$(path)'")
end

end # module



#  short segments from each

# gpt2_merges.txt ,
# h e
# i n
# r e
# o n...

# gpt2_vocab.json,
# {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6, "(": 7, ")": 8, "*": 9, "+": 10, ",": 11, "-": 12,...

# mGPT_61Lang1pnt9M_merges.txt, 

# Ġ t
# Ġ a
# Ð ¾
# Ġ s
# ã ģ...

# mGPT_61Lang1pnt9M_vocab.json,
# {"<s>":0,"<pad>":1,"</s>":2,"<unk>":3,"<mask>":4,"<|endoftext|>":5,"<case>":6,"!":7,"\"":8,"#":9,...

# RoBERTa-base_merges.txt,

# Ġ a
# h e
# i n
# r e
# o n...

# RoBERTa-base_vocab.json,
# {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3, ".": 4, "Ġthe": 5, ",": 6, "Ġto": 7, "Ġand": 8, ...

# Mistral-24B_32_768ctrl.json,

#   "version": "1.0",
#   "truncation": null,
#   "padding": null,
#   "added_tokens": [
#     {
#       "id": 0,
#       "content": "<unk>",
#       "single_word": false,
#       "lstrip": false,
#       "rstrip": false,
#       "normalized": false,
#       "special": true
#     },
#     {
#       "id": 1,
#       "content": "<s>",
#       "single_word": false,
#       "lstrip": false,
#       "rstrip": false,
#       "normalized": false,
#       "special": true
#     },
#     {
#       "id": 2,
#       "content": "</s>",
#       "single_word": false,
#       "lstrip": false,
#       "rstrip": false,
#       "normalized": false,
#       "special": true
#     },...

# read("Mistral-24B_32_768ctrl.json")
# JSON3.Object{Vector{UInt8}, Vector{UInt64}} with 9 entries:
 
#   :version        => "1.0"
#   :truncation     => nothing
#   :padding        => nothing
#   :added_tokens   => Object[{…
#   :normalizer     => nothing
#   :pre_tokenizer  => {…
#   :post_processor => {…
#   :decoder        => {…
#   :model          => {…
# julia> tmp[:decoder]
# JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}} with 4 entries:
#   :type             => "ByteLevel"
#   :add_prefix_space => true
#   :trim_offsets     => true
#   :use_regex        => true

# julia> tmp[:model]
# JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}} with 10 entries:
#   :type                      => "BPE"
#   :dropout                   => nothing
#   :unk_token                 => nothing
#   :continuing_subword_prefix => nothing
#   :end_of_word_suffix        => nothing
#   :fuse_unk                  => false
#   :byte_fallback             => false
#   :ignore_merges             => true
#   :vocab                     => {…
#   :merges                    => Array[["Ġ", "Ġ"], ["Ġ", "t"], ["e", "r"], ["i", "n"], ["Ġ", "…

# XML-RoBERTa_100Lang.json,
# julia> tmp =JSON3.read("XML-RoBERTa_100Lang.json")
# JSON3.Object{Vector{UInt8}, Vector{UInt64}} with 9 entries:
#   :version        => "1.0"
#   :truncation     => nothing
#   :padding        => nothing
#   :added_tokens   => Object[{…
#   :normalizer     => {…
#   :pre_tokenizer  => {…
#   :post_processor => {…
#   :decoder        => {…
#   :model          => {…

# julia> tmp[:model]
# JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}} with 2 entries:
#   :unk_id => 3
#   :vocab  => Array[Any["<s>", 0], Any["<pad>", 0], Any["</s>", 0], Any["<unk>", 0], Any[",", …

# julia> tmp[:decoder]
# JSON3.Object{Vector{UInt8}, SubArray{UInt64, 1, Vector{UInt64}, Tuple{UnitRange{Int64}}, true}} with 4 entries:
#   :type             => "Metaspace"
#   :replacement      => "▁"
#   :str_rep          => "▁"
#   :add_prefix_space => true

