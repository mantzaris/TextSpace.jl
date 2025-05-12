using Random
include("windowify.jl")
include("skipgram_pairs.jl")

"""
    build_skipgram_pairs(idstream; win=128, stride=64, radius=5, rng=GLOBAL_RNG)

Slice a corpus into windows, create skip-gram pairs, shuffle and return two
aligned vectors `(centers, contexts)`.
"""
function build_skipgram_pairs(ids::Vector{Int};
                              win::Int = 128,
                              stride::Int = 64,
                              radius::Int = 5,
                              rng = Random.GLOBAL_RNG)

    wins  = windowify(ids, win, stride)
    pairs = reduce(vcat, (skipgram_pairs(w, radius) for w in wins))
    Random.shuffle!(rng, pairs)
    return first.(pairs), last.(pairs)
end