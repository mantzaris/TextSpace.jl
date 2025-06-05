using Test
using TextSpace
using TextSpace.Plumbing: tokenize, split_sentences, tokenize_batch
using TextSpace.utils: Vocabulary, convert_tokens_to_ids
import TextSpace.utils as utils    # for load_bpe

@testset "Plumbing basics" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
    @test split_sentences("Hi. Bye!") == ["Hi.", "Bye!"]
end

@testset "_front_pass helper" begin
    raw = "Hi\u200D THERE!  😊\nBye."
    s, t = TextSpace.Pipeline._front_pass(raw;
                                          do_remove_zero_width = true,
                                          do_remove_emojis = true,
                                          case_transform   = :lower)
    @test s == ["hi there!", "bye."]
    @test first(t) == ["hi","there","!"]
end


@testset "new preprocess API" begin
    txt = "Hello🙂  WORLD!\u200d\nSecond line."

    mat = preprocess(txt)   # defaults word / batch
    @test size(mat, 2) == 1 

    # tokens + ids together
    toks, ids = preprocess(txt; output = :both, clean = false)
    @test length(toks) == length(ids)

    # sub-word ids  (TODO: enable when encode_batch is ready)
    @test_broken preprocess(txt; granularity      = :char,
                                split_sentences = false,
                                clean           = false,
                                output          = :ids)

    #char ids, single sequence (no split, no cleaning)
    cid = preprocess(txt; granularity     = :char,
                            split_sentences = false,
                            clean           = false,
                            output          = :ids)
    @test !isempty(cid)
end


