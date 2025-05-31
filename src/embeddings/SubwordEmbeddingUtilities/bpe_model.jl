import BytePairEncoding as BPE
import Serialization


load_encoder(name::String = "cl100k_base") = BPE.load_tiktoken_encoder(name) # or "gpt2"


encode(text, enc::BPE.BPEEncoder) = enc.encode(text)


decode(ids ::Vector{<:Integer}, enc::BPE.BPEEncoder) = enc.decode(ids)


"""
Highest valid token-id + 1 for this encoder.  
Works for both TikToken (cl100k_base) and GPT-2
"""
used_vocab_size(enc::BPE.BPEEncoder) = length(enc.vocab.list)


"""
    full_vocab_size(enc::BPE.BPEEncoder) → Int

Total length of the tiktoken vocabulary **including** the internal `#undef`
gaps.  
(Use the old `used_vocab_size(::BPEEncoder)` if you need the “defined-slot
count”.)
"""
full_vocab_size(enc::BPE.BPEEncoder) = length(enc.vocab.list)

"""
    save_encoder(path, enc)

Serialize the `BPEEncoder` to a binary file.
"""
save_encoder(path::AbstractString, enc::BPE.BPEEncoder) =
    Serialization.serialize(path, enc)

"""
    load_encoder_from_file(path) -> enc

Deserialize an encoder previously saved with `save_encoder`.
"""
load_encoder_from_file(path::AbstractString) =
    Serialization.deserialize(path)

export load_encoder, encode, decode, used_vocab_size,
       save_encoder, load_encoder_from_file



# """
#     segment_bpe(word, merges; eos = nothing) → Vector{String}
# Greedy pair-merge replay used at inference time.
# """
# function segment_bpe(word::AbstractString,
#                      merges::Dict{Tuple{String,String},String};
#                      eos=nothing)
#     toks = char_tokens(word; eos=eos)
#     while true
#         #  find candidate pair that exists in merges
#         best = nothing
#         for i in 1:length(toks)-1
#             p = (toks[i], toks[i+1])
#             haskey(merges, p) && (best = (i, p); break)
#         end
#         best === nothing && break
#         i, p = best
#         splice!(toks, i:i+1, merges[p])   # in-place merge
#     end
#     return toks
# end

# """
#     segment_wordpiece(word, vocab; unk_token = "[UNK]") → Vector{String}

# Return the list of WordPiece sub-tokens for `word` using a **greedy
# longest-match** strategy against the trained `vocab::Vocabulary`.

# * If no sub-string of the remaining span is in the vocabulary, the
#   routine emits `unk_token` (default `"[UNK]"`) and advances **one**
#   Unicode scalar (`nextind`) so the algorithm always makes progress.
# * The function is **stateless**: it never mutates `vocab`.
# * Works for any Unicode word (handles combining marks, ZWJ emoji, CJK).

# Example
# ```julia
# tokzr   = build_vocabulary_wordpiece(["lower", "lowest"]; vocab_size=50)
# vocab   = Vocabulary(tokzr["token_to_index"],
#                      tokzr["index_to_token"], Dict{Int,Int}(), 3)  # 3 = [UNK]
# seg     = segment_wordpiece("lowest", vocab)
# # → ["low", "est"]         (typical WordPiece split)

# """
# function segment_wordpiece(
# word::AbstractString,
# vocab::Vocabulary;
# unk_token::String = "[UNK]"
# )::Vector{String}

# toks = String[]
# i    = firstindex(word)

# while i ≤ lastindex(word)
#     # greedy longest match in vocab 
#     j      = lastindex(word)
#     match  = nothing
#     while j ≥ i
#         cand = word[i:j]
#         if haskey(vocab.token2id, cand)
#             match = cand
#             break
#         end
#         j = prevind(word, j)
#     end

#     if match === nothing
#         # fallback: single-char → known? else unk_token
#         char_sub = string(word[i])
#         push!(toks,
#               haskey(vocab.token2id, char_sub) ? char_sub : unk_token)
#         i = nextind(word, i)                       # advance one char
#     else
#         push!(toks, match)
#         i = nextind(word, j)                       # jump after match
#     end
# end
# return toks

# end

