"""
    cbow_pairs(tokens, radius)

Return `Vector{Tuple{Int, Vector{Int}}}` where the second field
is the list of context IDs — **CBOW style**.
"""
function cbow_pairs(tokens::AbstractVector{Int}, radius::Int)
    return windowify(tokens; window_size=radius, as_pairs=false)
end