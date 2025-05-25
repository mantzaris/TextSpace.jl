
function windowify(tokens::AbstractVector{<:Integer};
                   window_size::Integer = 5,
                   as_pairs::Bool = false)

    @assert window_size >= 1 "window_size must be ≥ 1"
    n = length(tokens)
    T = eltype(tokens)

    if as_pairs
        # Skip-Gram: flat (centre, context) tuples
        pairs = Vector{Tuple{T, T}}()

        for i in 1:n
            left  = max(i - window_size, 1)
            right = min(i + window_size, n)

            for j in left:right
                j == i && continue # skip the centre itself
                push!(pairs, (tokens[i], tokens[j]))
            end
        end

        return pairs
    else
        # CBOW: (centre, Vector{context}) tuples
        windows = Vector{Tuple{T, Vector{T}}}()

        for i in 1:n
            left  = max(i - window_size, 1)
            right = min(i + window_size, n)

            ctx = Vector{T}()
            for j in left:right
                j == i && continue
                push!(ctx, tokens[j])
            end
            push!(windows, (tokens[i], ctx))
        end

        return windows
    end
end