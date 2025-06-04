using Test
using TextSpace
using TextSpace.Plumbing: tokenize, split_sentences, tokenize_batch
using TextSpace.utils: Vocabulary, convert_tokens_to_ids
import TextSpace.utils as utils    # for load_bpe

@testset "Plumbing basics" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
    @test split_sentences("Hi. Bye!") == ["Hi.", "Bye!"]
end

@testset "clean-kw hook" begin
    txt = "Hello \u200B🙂 WORLD!"  # ZERO-WIDTH SPACE + emoji + caps

    out = TextSpace.preprocess(txt;
                               mode                 = :sentences,
                               do_remove_emojis     = true,
                               case_transform       = :lower,
                               do_remove_zero_width = true)

    @test out == ["hello  world!"]   # emoji & ZWSP removed, lowercase ok
end

@testset "front-half cleaning" begin
    raw = "Hello\u200D  🙂  WORLD!\nNext-Line"

    # no extra cleaning, but sentence split is ON -> ZWJ kept, spaces collapsed
    out1 = TextSpace.preprocess(raw;
                                clean=false,
                                mode=:sentences)          # default split_sentences=true
    @test out1 == ["Hello\u200D 🙂 WORLD!", "Next-Line"]

    # strip zero-width *before* splitter
    out2 = TextSpace.preprocess(raw;
                                clean=false,
                                do_remove_zero_width=true,
                                mode=:sentences)
    @test out2 == ["Hello 🙂 WORLD!", "Next-Line"]

    #  full clean: lower-case, emoji gone, spaces collapsed
    out3 = TextSpace.preprocess(raw;
                                do_remove_emojis=true,
                                case_transform=:lower,
                                mode=:sentences)
    @test out3 == ["hello world!", "next-line"]

    # preserve the double-space by **skipping** the splitter
    out4 = TextSpace.preprocess(raw;
                                split_sentences=false,  
                                do_remove_zero_width=true,
                                do_remove_emojis=true,
                                case_transform=:lower,
                                mode=:sentences)
    @test out4 == ["hello  world!\nnext-line"]            # two blanks maintained
end

@testset "paragraph - front-half + word-route smoke-test" begin
    raw_paragraph = """
        Hello\u200D, WORLD!!  🙂  This is line one.
        And here's line two—with dashes.

        New   paragraph: numbers 1, 2,  3…
    """

    # raw tokens, no cleaning 
    toks_raw = TextSpace.preprocess(raw_paragraph;
                                    mode  = :tokens,
                                    clean = false)      # keep mess
    @test !isempty(toks_raw)
    @test all(isa.(toks_raw, Vector{String}))

    # cleaned tokens (lower-case, emoji and ZWJ stripped) 
    toks_clean = TextSpace.preprocess(raw_paragraph;
                                      mode                 = :tokens,
                                      do_remove_zero_width = true,
                                      do_remove_emojis     = true,
                                      case_transform       = :lower)
    @test !isempty(toks_clean)
    @test toks_clean != toks_raw                      # cleaning changed output

    # word-route -> padded ID matrix 
    voc = Vocabulary()
    for t in tokenize_batch(split_sentences(raw_paragraph))
        convert_tokens_to_ids(t, voc; add_new = true)  # quick vocab build
    end

    mat = TextSpace.preprocess(raw_paragraph;
                               route   = :word,        # mapping branch
                               encoder = voc,
                               mode    = :batch)       # padded Matrix{Int}
    @test size(mat, 2) == length(split_sentences(raw_paragraph))
    @test size(mat, 1) ≥ 1
end





