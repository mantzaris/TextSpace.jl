"""
    skipgram_pairs(tokens, radius)

Return `Vector{Tuple{Int,Int}}` of (centre, context) indices
within ±`radius` of each position — **Skip-Gram style**.
"""
function skipgram_pairs(tokens::AbstractVector{Int}, radius::Int)
    # Re-use the generic windowifier
    return windowify(tokens; window_size=radius, as_pairs=true)
end