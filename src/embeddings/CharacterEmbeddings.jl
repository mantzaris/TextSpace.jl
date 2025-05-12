module CharacterEmbeddings

using Flux

include(joinpath(@__DIR__, "CharacterEmbeddingUtilities", "__init__.jl"))
using .CharacterEmbeddingUtilities: windowify, skipgram_pairs, build_skipgram_pairs


export train!, embeddings, save_embeddings   # <- public API only

# ... full Skip-Gram example from earlier ...



end # module
