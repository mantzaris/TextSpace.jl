
@testset "LoadBPE - merges-only artefact" begin
    gpt_path = TextSpace.resource("gpt2_merges.txt")

    tok = load_bpe(gpt_path)

    @test isa(tok, BPETokeniser)                   # correct type
    @test length(tok.merges) > 10                  # something was read
    @test tok.vocab === nothing                    # merges-only -> no vocab

    # sanity: first entry is a pair of Strings
    @test isa(tok.merges[1], Tuple{String,String})
end

@testset "LoadBPE - explicit artefacts" begin
    names = [
        "gpt2_merges.txt",
        "RoBERTa-base_merges.txt",
        "mGPT_61Lang1pnt9M_merges.txt",
        "Mistral-24B_32_768ctrl.json",
        "XML-RoBERTa_100Lang.json",
    ]

    for n in names
        tok = load_bpe(TextSpace.resource(n))
        @test isa(tok, BPETokeniser)
        # merges may be empty for XLM-R, but the field must exist
        @test isa(tok.merges, Vector{Tuple{String,String}})
        # vocab may be nothing (GPT-2) or Dict
        @test tok.vocab === nothing || isa(tok.vocab, Dict)
    end
end

@testset "LoadBPE - deep-sanity of bundled artefacts" begin
    # GPT-2 (merges-only)
    gpt = load_bpe(TextSpace.resource("gpt2_merges.txt"))
    @test gpt.vocab === nothing
    @test ("h","e") in gpt.merges         # first merge in original file

    # RoBERTa-base
    rob = load_bpe(TextSpace.resource("RoBERTa-base_merges.txt"))
    @test length(rob.merges) >= 49_000     # full list around 49 992 lines
    @test rob.vocab["<unk>"] == 3             # canonical id
    @test ("Ġ","t") in rob.merges || ("Ġ","the") in rob.merges

    # mGPT 61-lang
    mgpt = load_bpe(TextSpace.resource("mGPT_61Lang1pnt9M_merges.txt"))
    @test haskey(mgpt.vocab, "<s>") && mgpt.vocab["<s>"] == 0
    @test length(mgpt.merges) > 40_000

    # Mistral controller JSON 
    mis = load_bpe(TextSpace.resource("Mistral-24B_32_768ctrl.json"))
    @test mis.vocab["<unk>"] == 0
    @test length(mis.merges) > 0  # Mistral ships a merges list

    #  XLM-R (100-lang)
    xlmr = load_bpe(TextSpace.resource("XML-RoBERTa_100Lang.json"))
    @test xlmr.vocab["<unk>"] == 3
    @test isempty(xlmr.merges)  # SP export - merges intentionally empty
end
