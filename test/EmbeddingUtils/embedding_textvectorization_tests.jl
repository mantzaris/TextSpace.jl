include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "TextVectorization.jl"))


@testset "pad_sequences" begin
    seqs = [[1,2,3,4], [5,6]]

    m = pad_sequences(seqs; pad_value = 0)          # default maxlen = 4
    @test size(m) == (4,2)
    @test m[:,1] == [1,2,3,4]
    @test all(m[3:4,2] .== 0)                      # padded

    # explicit maxlen > longest  ⇒ extra pad
    m2 = pad_sequences(seqs; maxlen = 5, pad_value = -1)
    @test m2[5,1] == -1
    @test m2[3:5,2] == [-1,-1,-1]

    # truncation, keep tail when trunc = :pre
    m3 = pad_sequences(seqs; maxlen = 3, trunc = :pre)
    @test m3[:,1] == [2,3,4]                       # tail kept
end


@testset "pad_sequences - UTF-8 / edge hammer" begin
    #empty corpus returns 0x0 matrix
    E = pad_sequences(Vector{Vector{Int}}())
    @test size(E) == (0,0)

    #mix of normal and *zero-length* sequences
    seqs = [[1,2], Int[], [3]]
    M = pad_sequences(seqs; maxlen = 3, pad_value = -9)
    @test size(M) == (3,3)
    @test M[:,2] == fill(-9, 3)          # empty sequence -> full pad
    @test M[1:2,1] == [1,2] && M[3,1] == -9

    #truncation mode :post (keep head)
    long = [[1,2,3,4,5]]
    P = pad_sequences(long; maxlen = 3, trunc = :post)
    @test P[:,1] == [1,2,3]

    #truncation mode :pre (keep tail)
    Q = pad_sequences(long; maxlen = 3, trunc = :pre)
    @test Q[:,1] == [3,4,5]

    #unicode-sized batch, default maxlen = longest
    u = [[0x1F680], [10,20,30,40], [99]]     # 🚀 token is just an Int
    U = pad_sequences(u; pad_value = 0)
    @test size(U) == (4,3)                   # longest = 4
    @test U[:,2] == [10,20,30,40]            # no change
    @test all(U[2:4,1] .== 0)                # padded
end


@testset "one_hot" begin
    oh = one_hot([2,4,1], 5)
    @test oh[2,1]            # id 2 hot in column 1
    @test oh[4,2]            # id 4 hot in column 2
    @test oh[1,3]            # id 1 hot in column 3
    @test count(>(0), oh) == 3  # exactly three ones
end


@testset "one_hot - wide-spectrum hammer" begin
    #basic case, all IDs valid
    seq  = [1, 3, 2, 3]
    V    = 3
    M    = one_hot(seq, V)

    @test size(M) == (V, length(seq))
    @test M isa BitMatrix        # uses bit-packing
    @test M[:,1] == Bool[1,0,0]  # id 1
    @test M[:,2] == Bool[0,0,1]  # id 3
    @test M[:,3] == Bool[0,1,0]  # id 2
    @test M[:,4] == Bool[0,0,1]  # id 3 again

    #out-of-vocabulary and negative IDs -> ignored
    seq2 = [0, 4, -1, 2]
    M2   = one_hot(seq2, V)
    @test sum(M2) == 1          # only id 2 valid
    @test M2[:,4] == Bool[0,1,0]

    #duplicate IDs across positions
    dup = one_hot([2,2,2], 2)
    @test dup == BitMatrix([0 0 0;
                            1 1 1])             # row 2 all ones

    #empty sequence
    empty = one_hot(Int[], 5)
    @test size(empty) == (5, 0)

    #different integer element type
    seq16 = Int16[1,2]
    M16   = one_hot(seq16, 2)
    @test M16[:,1] == Bool[1,0] && M16[:,2] == Bool[0,1]

    #large vocab, small seq (shape only)     
    big = one_hot([1000], 2000)
    @test size(big) == (2000,1) && big[1000,1]

    #(Optional) assert on vocab_size <= 0
    Z = one_hot([1,2,3], 0)
    @test size(Z) == (0, 3) && sum(Z) == 0
end


@testset "BoW bow_counts" begin
    seq  = [1,2,2,5]
    vsize = 5
    @test bow_counts(seq, vsize) == [1, 2, 0, 0, 1]   # ← fixed

    docs = [[1,2], [2,2,4]]
    bm   = bow_matrix(docs, vsize)
    @test bm[:,1] == [1,1,0,0,0]
    @test bm[:,2] == [0,2,0,1,0]
end


@testset "bow_counts - UTF-8 / edge hammer" begin
    #basic duplicates and ordering
    seq  = [3, 1, 3, 2, 3]
    V    = 3
    v    = bow_counts(seq, V)
    @test v == [1, 1, 3]        # counts for ids 1,2,3

    #out-of-vocab, zero, negative IDs -> ignored
    seq2 = [0, -1, 5, 2]
    v2   = bow_counts(seq2, V)
    @test v2 == [0, 1, 0]

    #empty sequence
    @test bow_counts(Int[], V) == zeros(Int, V)

    #vocab_size larger than max id
    big = bow_counts(seq, 10)
    @test big[1:3] == v && all(big[4:end] .== 0)

    #different integer element type
    seq16 = Int16[1, 1, 2]
    v16   = bow_counts(seq16, 2)
    @test v16 == [2, 1]

    #vocab_size <= 0 -> empty vector (after refactor)
    z = bow_counts([1,2,3], 0)
    @test length(z) == 0
end


@testset "bow_matrix - UTF-8 / edge hammer" begin
    # helper corpus (ids are arbitrary Ints)
    docs = [[1,1,3],            # Doc 1
            [2,-5,0,2],         # Doc 2  (OOV & zero ignored)
            [3,3,3,3]]          # Doc 3
    V = 3

    M = bow_matrix(docs, V)

    #shape and exact counts
    @test size(M) == (V, length(docs))
    @test M[:,1] == [2,0,1]     # two 1s, one 3
    @test M[:,2] == [0,2,0]     # two 2s
    @test M[:,3] == [0,0,4]

    #empty document list -> 0x0 matrix
    E = bow_matrix(Vector{Vector{Int}}(), V)
    @test size(E) == (0,0)

    #vocab_size <= 0 -> 0xN matrix
    Z = bow_matrix(docs, 0)
    @test size(Z) == (0, length(docs))

    #docs containing empty sequences handled
    M2 = bow_matrix([Int[], [1]], 2)
    @test M2[:,1] == zeros(Int,2) && M2[:,2] == [1,0]

    #different integer element type
    docs16 = [Int16[1, 2],  Int16[2]]   # second doc has a single 2
    M16    = bow_matrix(docs16, 2)

    # expected counts:
    #   col1 -> [1 (id1), 1 (id2)]
    #   col2 -> [0,  1]
    @test M16 == [1 0;
                  1 1]

    @test eltype(M16) == Int
end


@testset "tfidf_matrix (tiny example)" begin
    docs = [[1,1], [1,2]]
    tfidf = tfidf_matrix(docs, 2; smooth_idf = 0)   # raw IDF

    @test isapprox(tfidf[1,1], 0.0; atol=1e-12)     # idf=0 because word 1 in both docs
    @test tfidf[2,2] > 0.0                          # idf(log 2) > 0 for word 2
end


@testset "tfidf_matrix - hammer" begin
    #tiny toy corpus: verify exact IDF + TFIDF
    #  doc0: ids 1,2      doc1: id 1       doc2: id 3
    docs = [[1,2], [1], [3]]
    V    = 3              # vocab_size
    M    = tfidf_matrix(docs, V; smooth_idf = 1.0)

    #IDF with smoothing 1 = log((N+1)/(df+1))  where N = 3
    idf_expected = [
        log((3+1)/(2+1)),   # id 1 appears in 2 docs
        log((3+1)/(1+1)),   # id 2 appears in 1 doc
        log((3+1)/(1+1))    # id 3 appears in 1 doc
    ]

    # column 1: tf = [1,1,0]  → tfidf
    @test isapprox(M[:,1], [1,1,0] .* idf_expected; atol=1e-12)
    # column 2: tf = [1,0,0]
    @test isapprox(M[:,2], [1,0,0] .* idf_expected)
    # column 3: tf = [0,0,1]
    @test isapprox(M[:,3], [0,0,1] .* idf_expected)

    #smooth_idf = 0  (raw IDF)
    M0 = tfidf_matrix(docs, V; smooth_idf = 0.0)
    idf_raw = [log(3/2), log(3/1), log(3/1)]
    @test isapprox(M0[:,1], [1,1,0] .* idf_raw)

    #OOV and zero IDs ignored
    docs2 = [[0, -1, 4, 1]]
    M2    = tfidf_matrix(docs2, V)
    @test M2[:,1] == [log((1+1)/(1+1)), 0, 0]   # only id 1 counted

    #empty corpus ->  Vx0 matrix
    E = tfidf_matrix(Vector{Vector{Int}}(), 5)
    @test size(E) == (0, 0)

    #vocab_size <= 0  -> 0xN matrix
    Z = tfidf_matrix(docs, 0)
    @test size(Z) == (0, length(docs))

    #negative smoothing throws
    @test_throws ArgumentError tfidf_matrix(docs, V; smooth_idf = -0.5)
end


@testset "batch_iter" begin
    seqs = [[1,2,3,4], [5,6], [7]]
    itr, state = batch_iter(seqs, 2; shuffle = false, pad_value = 0)

    (mat1, sub1), state = itr(state)
    @test sub1 == [1,2]
    @test size(mat1) == (4, 2)              # longest seq in batch = 4

    (mat2, sub2), state = itr(state)
    @test sub2 == [3]                       # last doc
    @test size(mat2) == (1, 1)              # longest seq in this batch = 1
    @test mat2[1,1] == 7                    # value preserved, no padding needed

    @test itr(state) === nothing            # iterator exhausted
end


@testset "batch_iter - robustness hammer" begin
    #helper data: five variable-length docs   
    docs = [[1,2,3], [4], [5,6], [7,8,9,10], [11]]

    #deterministic shuffling with RNG
    rng1 = MersenneTwister(42)
    it1, st1 = batch_iter(docs, 2; shuffle=true, rng=rng1)

    rng2 = MersenneTwister(42)
    it2, st2 = batch_iter(docs, 2; shuffle=true, rng=rng2)

    subs1 = []
    subs2 = []
    s = st1;  while (r = it1(s)) !== nothing; (M, sub), s = r; push!(subs1, sub); end
    s = st2;  while (r = it2(s)) !== nothing; (M, sub), s = r; push!(subs2, sub); end
    @test subs1 == subs2      # same shuffle order

    #no shuffle keeps natural order
    it_ns, st_ns = batch_iter(docs, 3; shuffle=false)
    (Mns, subns), _ = it_ns(st_ns)
    @test subns == [1,2,3]      # first three docs

    #custom pad_value and size checks
    it_pad, st_pad = batch_iter(docs, 4; pad_value=-1, shuffle=false)
    (Mpad, subpad), _ = it_pad(st_pad)
    @test size(Mpad) == (4, 4)       # maxlen in batch = 4
    @test Mpad[4,2] == -1            # padded doc #2

    #batch_size larger than corpus -> single batch
    it_big, st_big = batch_iter(docs, 10; shuffle=false)
    c = 0; s = st_big; while (r = it_big(s)) !== nothing; c += 1; (_, _), s = r; end
    @test c == 1

    #empty corpus -> iterator ends immediately
    ie, se = batch_iter(Vector{Vector{Int}}(), 2)
    @test ie(se) === nothing

    #invalid batch_size throws   
    @test_throws ArgumentError batch_iter(docs, 0)
end
