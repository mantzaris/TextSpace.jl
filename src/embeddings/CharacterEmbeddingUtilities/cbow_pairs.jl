
"""
    cbow_pair(win_ids, radius)

Return a pair `(context_ids::Vector{Int}, target_id)` for each position.
The context is the surrounding tokens within ±`radius`, *excluding* center.
"""
function cbow_pairs(win::AbstractVector{Int}, radius::Int)
    pairs = Vector{Tuple{Vector{Int},Int}}()
    for i in eachindex(win)
        context = win[max(1,i-radius):min(end,i+radius)]
        deleteat!(context, i - firstindex(win) + 1)  # remove center
        push!(pairs, (collect(context), win[i]))
    end
    return pairs
end
