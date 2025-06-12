

const LBPE = TextSpace.Utils.LearnBPE   # module path
const BPET = LBPE.BPETokeniser          # handy alias

@testset "LearnBPE - basic training" begin
    corpus = [
        "hug my dog",
        "hug my cat",
        "dogs and cats hug back"
    ]
    special = ["<unk>", "<s>", "</s>", "<pad>"]

    tok = LBPE.learn_bpe(corpus; vocab_size=50, min_frequency=1,
                                   special_tokens=special)

    @test tok isa BPET                   

    @test !isempty(tok.merges)
    @test all(merge isa Tuple{String,String} for merge in tok.merges)

    @test tok.vocab !== nothing
    @test all(haskey(tok.vocab, t) for t in special)
    @test all(haskey(tok.vocab, c) for c in ["h","u","g","</w>"])

    @test length(tok.vocab) ≤ 50
end


@testset "LearnBPE - paragraph corpus" begin
    paragraph = """
        Natural-language processing (NLP) enables computers to understand
        human language.  Modern NLP pipelines rely on tokenisation,
        sub-word encoding (such as Byte-Pair Encoding, BPE) and statistical
        language models.  As datasets grow, learned BPE vocabularies can
        capture domain-specific terms—tokenising ‘language-model’,
        ‘tokenisation’ and "pipeline" more effectively than naive
        whitespace splitting.
        """

    # treat each sentence as a 'document' - split on full-stops
    corpus = split(paragraph, '.') |> x -> filter(!isempty ∘ strip, x)

    # special tokens we want guaranteed in the final vocab
    specials = ["<unk>", "<s>", "</s>", "<pad>"]

    tok = LBPE.learn_bpe(corpus;
                         vocab_size     = 120,   # small for the example
                         min_frequency  = 2,
                         special_tokens = specials)

    @test tok isa BPET
    @test !isempty(tok.merges)              # learned >= 1 merge
    @test tok.vocab !== nothing
    @test length(tok.vocab) <= 120           # respected budget
    @test all(haskey(tok.vocab, s) for s in specials)

    #couple of tokens we expect to appear in tiny corpus
    @test haskey(tok.vocab, "language</w>")
    @test haskey(tok.vocab, "tokenisation</w>")
end


@testset "LearnBPE - save/load round-trip" begin
    #train a tiny model 
    tok_orig = LBPE.learn_bpe(
        ["alpha beta beta gamma delta"],   # toy corpus
        vocab_size    = 40,
        min_frequency = 1,
        special_tokens = ["<unk>", "<pad>"]
    )

    # create an isolated temp folder 
    tmpdir   = mktempdir()
    base     = joinpath(tmpdir, "my_tok_" * string(uuid4()))

    #JSON only 
    LBPE.save_bpe(tok_orig, base * ".json"; format = "json")   #string
    @test isfile(base * ".json")

    tok_json = LBPE.load_bpe(base * ".json")
    @test tok_json.merges == tok_orig.merges
    @test tok_json.vocab  == tok_orig.vocab

    #  JSON + GPT-2-style text 
    LBPE.save_bpe(tok_orig, base; format = "both")             #  string
    @test isfile(base * ".json")
    @test isfile(base * "_merges.txt")

    tok_txt = LBPE.load_bpe(base)   # base path -> text files variant
    @test tok_txt.merges == tok_orig.merges
    @test tok_txt.vocab  == tok_orig.vocab
end
