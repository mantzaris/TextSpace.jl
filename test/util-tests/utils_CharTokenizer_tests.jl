using TextSpace
using TextSpace.utils
using TextSpace.utils.VocabularyCore
using TextSpace.utils.CharTokenizer
using TextSpace.utils.CharTokenizer: chars_to_ids, encode_char_batch
using TextSpace.utils.VocabularyCore: Vocabulary


function make_vocab()
    #id 1 is <UNK> by convention
    vocab = Vocabulary("<UNK>")

    #pre-seeding two known characters so we can test both in-vocab and OOV paths
    @assert chars_to_ids(["a"], vocab; add_new = true, update_counts = false) == [2]
    @assert chars_to_ids(["b"], vocab; add_new = true, update_counts = false) == [3]

    empty!(vocab.counts)
    return vocab
end

@testset "chars_to_ids" begin
    vocab = make_vocab()

    @testset "known chars" begin
        ids = chars_to_ids(["a","b","a"], vocab)
        @test ids == [2, 3, 2]
    end

    @testset "unknown char, add_new = false" begin
        ids = chars_to_ids(["z"], vocab; add_new = false)
        @test ids == [vocab.unk_id]            # falls back to <UNK>
        @test !haskey(vocab.token2id, "z")     # not inserted
    end

    @testset "unknown char, add_new = true" begin
        new_ids = chars_to_ids(["z"], vocab; add_new = true, update_counts = false)
        @test new_ids[1] == vocab.token2id["z"]
        @test vocab.id2token[end] == "z"
    end

    @testset "update_counts toggle" begin
        cnt_before = deepcopy(vocab.counts)
        _ = chars_to_ids(["a","a"], vocab; update_counts = true)
        @test vocab.counts[2] == get(cnt_before, 2, 0) + 2

        cnt_before2 = deepcopy(vocab.counts)
        _ = chars_to_ids(["a"], vocab; update_counts = false)
        @test vocab.counts == cnt_before2        # unchanged
    end

    @testset "empty input" begin
        @test chars_to_ids(String[], vocab) == Int[]
    end
end


@testset "paragraph corpus, dynamic growth" begin
    para = """
        The quick brown 🦊 jumps over 13 lazy dogs!  Καλημέρα κόσμε.
        – こんにちは世界 -  Привет, Γεια!  🚀
    """

    # One code-point per element, as Strings
    chars = [string(c) for c in para]

    vocab = Vocabulary("<UNK>")
    ids   = chars_to_ids(chars, vocab; add_new = true, update_counts = true)

    @test length(ids) == length(chars)
    @test length(vocab.id2token) == 1 + length(unique(chars))
    @test sum(values(vocab.counts)) == length(chars)

    # round-trip
    rev = convert_ids_to_tokens(ids, vocab)
    @test rev == chars
end




function prime_vocab!(vocab::Vocabulary, tok_batch; eos="</w>")
    needed_chars = String[]
    for sent in tok_batch
        append!(needed_chars, [string(c) for c in join(sent, "")])
    end
    eos === nothing || push!(needed_chars, eos)

    # add_new = true populates token2id / id2token; counts not needed
    chars_to_ids(needed_chars, vocab; add_new = true, update_counts = false)
    empty!(vocab.counts)         # start tests with zero counts
    return vocab
end


@testset "encode_char_batch - basic EOS & padding" begin
    tok_batch = [["hi"], ["bye"]]             # two "sentences"
    vocab     = prime_vocab!(Vocabulary("<UNK>"), tok_batch)

    mat = encode_char_batch(tok_batch, vocab)  # default eos="</w>"

    #matrix has one column per sentence
    @test size(mat, 2) == length(tok_batch)

    for (col, sent) in enumerate(tok_batch)
        sent_str = join(sent, "")
        chars    = [string(c) for c in sent_str]
        push!(chars, "</w>")
        ids      = chars_to_ids(chars, vocab; add_new = false, update_counts = false)

        #non-padded prefix equals ids
        @test mat[1:length(ids), col] == ids

        #remaining rows are padded with unk_id
        @test all(mat[length(ids)+1:end, col] .== vocab.unk_id)
    end
end


@testset "encode_char_batch - no EOS, custom pad value" begin
    tok_batch = [["a"], ["bbbbb"]]
    custom_pad = 0
    vocab = prime_vocab!(Vocabulary("<UNK>"), tok_batch; eos = nothing)

    mat = encode_char_batch(tok_batch, vocab; eos = nothing,
                            pad_value = custom_pad)

    @test size(mat, 2) == 2 #same orientation check

    #sequence 1 (length 1)
    @test mat[1,1]      == chars_to_ids(["a"], vocab; add_new=false)[1]
    @test all(mat[2:end,1] .== custom_pad)

    #sequence 2 (length 5)
    ids2 = chars_to_ids([string(c) for c in "bbbbb"], vocab; add_new=false)
    @test mat[1:length(ids2),2] == ids2
end


@testset "encode_char_batch - paragraph integration" begin
    para = """
        Julia is fun.  It is fast and expressive!
        Compiler magic turns your high-level code into machine-tight loops.
    """

    # crude sentence & word tokenisation
    sentences = split(para, r"[.!]\s*"; keepempty = false)
    tok_batch = [
        String.(split(strip(s), r"\s+"; keepempty=false))  # ← converts to String
        for s in sentences
    ]

    vocab = prime_vocab!(Vocabulary("<UNK>"), tok_batch)   # re-use helper

    mat = encode_char_batch(tok_batch, vocab)    # default eos="</w>"

    # basic shape checks
    nonpad_counts = vec(sum(mat .!= vocab.unk_id, dims = 1))
    @test size(mat, 1) == maximum(nonpad_counts)   # rows = longest sentence + EOS
    @test size(mat, 2) == length(tok_batch)        # one column per sentence

    # round-trip every sentence
    for (col, words) in enumerate(tok_batch)
        chars = [string(c) for c in join(words, "")]
        push!(chars, "</w>")
        ids = chars_to_ids(chars, vocab; add_new = false, update_counts = false)

        @test mat[1:length(ids), col] == ids
        @test all(mat[length(ids)+1:end, col] .== vocab.unk_id)
    end
end

