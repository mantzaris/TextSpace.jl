module SubwordEmbeddings

using Flux, Random

include(joinpath(@__DIR__, "SubwordEmbeddingUtilities", "__init__.jl"))
using .SubwordEmbeddingUtilities

export windowify, make_skipgram_pairs, make_cbow_pairs, learn_bpe, encode_bpe

end

