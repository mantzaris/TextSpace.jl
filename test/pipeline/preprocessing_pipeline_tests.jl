using Test
using TextSpace                     # loads the package
using TextSpace.Plumbing: tokenize  # import ONE helper to prove it works


@testset "Plumbing smoke-test" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
end


@testset "pipeline - default word path" begin
    corpus = "Hello, World!\n\nGood-bye moon."

    # build tiny vocab from the corpus itself
    vocab = TextSpace.utils.Vocabulary()
    for t in TextSpace.Plumbing.tokenize_batch(
             TextSpace.Plumbing.split_sentences(corpus))
        TextSpace.utils.convert_tokens_to_ids(t, vocab; add_new=true)
    end

    mat = TextSpace.preprocess(corpus; encoder=vocab)   # ← no :target kw
    @test size(mat, 2) == 2                             # 2 sentences → 2 columns
end



@testset "word-level pipeline" begin
    doc = "Hello world.\nGood-bye moon."

    # build tiny vocab
    vocab = TextSpace.utils.Vocabulary()
    for t in TextSpace.Plumbing.tokenize_batch(
                TextSpace.Plumbing.split_sentences(doc))
        TextSpace.utils.convert_tokens_to_ids(t, vocab; add_new=true)
    end

    sent  = preprocess(doc; encoder=vocab, mode=:sentences)
    toks  = preprocess(doc; encoder=vocab, mode=:tokens)
    ids   = preprocess(doc; encoder=vocab, mode=:ids)
    batch = preprocess(doc; encoder=vocab)          # default :batch

    @test sent == ["Hello world.", "Good-bye moon."]
    @test first(toks) == ["hello","world"]
    @test first(ids)  == [2,3]                      # ids after <unk>
    @test size(batch, 2) == 2                       # two sentences
end