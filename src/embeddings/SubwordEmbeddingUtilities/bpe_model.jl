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