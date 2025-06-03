using Test
using TextSpace
using TextSpace.Plumbing: tokenize, split_sentences, tokenize_batch
using TextSpace.utils: Vocabulary, convert_tokens_to_ids
import TextSpace.utils as utils    # for load_bpe

@testset "Plumbing basics" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
    @test split_sentences("Hi. Bye!") == ["Hi.", "Bye!"]
end

@testset "Word-level preprocess" begin
    doc = "Hello, World!\n\nGood-bye moon."

    # tiny throw-away vocab built from the doc
    vocab = Vocabulary()
    for t in tokenize_batch(split_sentences(doc))
        convert_tokens_to_ids(t, vocab; add_new=true)
    end

    batch = TextSpace.preprocess(doc; encoder=vocab)           # mode=:batch
    ids   = TextSpace.preprocess(doc; encoder=vocab, mode=:ids)
    toks  = TextSpace.preprocess(doc; encoder=vocab, mode=:tokens)
    sents = TextSpace.preprocess(doc; encoder=vocab, mode=:sentences)

    @test size(batch, 2) == 2          # two sentences -> two columns
    @test length(ids)    == 2
    @test length(toks)   == 2
    @test sents == ["hello, world!", "good-bye moon."]
end

@testset "resource & BPE loader" begin
    # 1) manual loader
    gpt_path = TextSpace.resource("gpt2_merges.txt")
    @test isfile(gpt_path)
    tok1 = utils.load_bpe(gpt_path)
    @test length(tok1.merges) > 0
    @test tok1.vocab === nothing      # merges-only file

    # 2) symbolic loader (resolve_subtok)
    tok2 = TextSpace.Pipeline.resolve_subtok(:gpt2)
    @test length(tok2.merges) == length(tok1.merges)
end
