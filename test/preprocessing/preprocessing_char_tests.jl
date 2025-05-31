include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "CharProcessing.jl"))


@testset "tokenize_char" begin
    txt = "Café 😊"

    # default: NFC, no spaces, original case
    toks = tokenize_char(txt)
    @test toks == ["C","a","f","é","😊"]

    # lower-case, keep space
    toks2 = tokenize_char(txt; lower = true, keep_space = true)
    @test toks2 == ["c","a","f","é"," ","😊"]

    # combined accents survive when skipping normalisation (:NFD)
    nfd = Unicode.normalize("é", :NFD)          # 'e' + combining acute
    toks3 = tokenize_char(nfd; normalize = false)
    @test length(toks3) == 1                    # one grapheme cluster
end


@testset "tokenize_char - complex graphemes" begin
    # ZWJ family      👩🏽‍🚀  (woman + skin-tone + zwj + rocket)
    @test tokenize_char("👩🏽‍🚀") == ["👩🏽‍🚀"]

    # Flag sequence   🇨🇦  (regional indicators)
    @test tokenize_char("🇨🇦") == ["🇨🇦"]

    # Hindi combining matra  क + ि  to कि  (single grapheme)
    dev = "कि"
    @test tokenize_char(dev) == ["कि"]
end


@testset "tokenize_char - whitespace handling" begin
    s = "a \t\n b"
    @test tokenize_char(s; keep_space=false) == ["a","b"]
    toks = tokenize_char(s; keep_space=true)
    @test count(t -> isspace(first(t)), toks) == 4
end


@testset "tokenize_char - lower & NFC idempotence" begin
    src  = "Núñez"
    toks = tokenize_char(src; lower=true)
    @test join(toks) == lowercase(Unicode.normalize(src, :NFC))
end


@testset "tokenize_char - error branch" begin
    @test_throws ArgumentError tokenize_char("x"; form=:XYZ)
end


@testset "chars_to_ids" begin
    voc = Vocabulary(Dict("<unk>" => 1), ["<unk>"], Dict{Int,Int}(), 1)
    ids = chars_to_ids(["a","b","a"], voc; add_new = true)

    @test ids == [2,3,2]                   # 'a' got id 2, 'b' id 3
    @test voc.id2token[2:3] == ["a","b"]   # vocabulary grew as expected
    @test chars_to_ids(["z"], voc) == [1]  # unk when add_new = false
end


@testset "chars_to_ids - core behaviour" begin
    voc = Vocabulary(Dict("<unk>" => 1), ["<unk>"], Dict{Int,Int}(), 1)

    ids = chars_to_ids(["a","b","a"], voc; add_new = true)
    @test ids == [2,3,2]
    @test voc.id2token[2:3] == ["a","b"]
    @test voc.token2id["a"] == 2
    @test voc.counts == Dict(2=>2, 3=>1)

    #unknown token when add_new=false
    @test chars_to_ids(["z"], voc) == [1]
    @test get(voc.counts, 1, 0) == 1      # '<unk>' count incremented
end


@testset "chars_to_ids - update_counts flag" begin
    voc = Vocabulary(Dict("<unk>" => 1, "x"=>2), ["<unk>","x"], Dict(1=>10,2=>5), 1)
    ids = chars_to_ids(["x","x"], voc; add_new=false, update_counts=false)
    @test ids == [2,2]
    @test voc.counts == Dict(1=>10, 2=>5)   # unchanged
end


@testset "chars_to_ids - add_new=false keeps vocab size" begin
    voc = Vocabulary(Dict("<unk>"=>1, "y"=>2), ["<unk>","y"], Dict(), 1)
    _   = chars_to_ids(["q","y"], voc; add_new=false)
    @test length(voc.id2token) == 2    # no new token appended
    @test voc.token2id["y"] == 2
end


@testset "chars_to_ids - generic SubString input" begin
    base = "abc"
    subs = [base[1:1], base[2:2], base[3:3]]
    voc  = Vocabulary(Dict("<unk>"=>1), ["<unk>"], Dict(), 1)
    ids  = chars_to_ids(subs, voc; add_new=true)
    @test ids == [2,3,4]
    #ensure stored tokens are independent String objects
    @test all(x->isa(x,String), voc.id2token[2:4])
end


@testset "encode_char_batch" begin
    voc = Vocabulary(Dict("<unk>" => 1, "h"=>2, "i"=>3, "😊"=>4),
                     ["<unk>","h","i","😊"],
                     Dict{Int,Int}(), 1)

    mat = encode_char_batch(["hi","😊h"], voc)
    @test size(mat) == (2, 2)              # longest seq = 2
    @test mat[:,1] == [2,3]                # 'h i'
    @test mat[1,2] == 4 && mat[2,2] == 2   # '😊 h' (padded already OK)
end


@testset "encode_char_batch - lower + custom pad" begin
    voc = Vocabulary(Dict("<unk>" => 0, "h"=>1, "i"=>2, " "=>3),
                     ["<unk>","h","i"," "],
                     Dict{Int,Int}(), 0)

    batch  = ["Hi ", "i"]
    mat    = encode_char_batch(batch, voc; lower=true, keep_space=true,
                               pad_value=0)     # explicit pad id = 0

    @test size(mat) == (3, 2) # longest seq = 3 graphemes ("h","i"," ")
    @test mat[:,1] == [1,2,3] # "h i space"
    @test mat[:,2] == [2,0,0] # padded with zeros
end


@testset "encode_char_batch - unknowns remain unk_id" begin
    voc = Vocabulary(Dict("<unk>" => 99), ["<unk>"], Dict(), 99)
    mat = encode_char_batch(["xyz"], voc)
    @test all(mat .== 99)
end


@testset "char_tokens" begin
    #ASCII, no eos
    @test char_tokens("hello") == ["h","e","l","l","o"]

    #unicode grapheme
    @test char_tokens("café") == ["c","a","f","é"]
    @test char_tokens("👩🏽‍🚀") == ["👩🏽‍🚀"]     # single grapheme

    #end-of-word marker
    @test char_tokens("hi"; eos="</w>") == ["h","i","</w>"]

    #empty string edge-case
    @test char_tokens("") == String[]

    substr = SubString("hello-world", 1, 5)   # "hello"
    @test char_tokens(substr) == ["h","e","l","l","o"]
end


@testset "char_tokens - robust Unicode hammer" begin
    using Unicode

    zwsp   = "\u200B"                       # ZERO-WIDTH SPACE
    flagUS = "🇺🇸"                          # two regional indicators, one grapheme
    family = "👨‍👩‍👧‍👦"                   # family emoji (ZWJ sequence)
    astro  = "👩🏽‍🚀"                       # skin-tone + ZWJ
    kiss   = "👩‍❤️‍💋‍👨"                  # complex ZWJ chain
    combé  = "e\u0301"                      # e + COMBINING ACUTE
    multiC = "a\u0300\u0316"                # a + grave + combining sub-dot
    loneCM = "\u0301"                       # COMBINING ACUTE by itself
    empty  = ""

    words  = [ "hello",
               flagUS, family, astro, kiss,
               combé, multiC, loneCM,
               "abc$zwsp",                   # embedded zero-width space
               empty ]

    # helper to build the reference using Unicode.graphemes (ground truth)
    expected(w; eos=nothing) = begin
        toks = [String(g) for g in Unicode.graphemes(String(w))]
        eos === nothing || push!(toks, eos)
        toks
    end

    #for every quirky word, result matches Unicode.graphemes
    for w in words
        @test char_tokens(w) == expected(w)
        @test char_tokens(w; eos="</w>") == expected(w; eos="</w>")
    end

    #every output token is a non-empty, valid UTF-8 String
    for w in words, tok in char_tokens(w)
        @test isa(tok, String) && !isempty(tok) && isvalid(tok)
    end

    #`eos` marker appended exactly once           
    sample = char_tokens("hi"; eos="</e>")
    @test sample[end] == "</e>" && count(==("</e>"), sample) == 1

    #SubString input behaves identically       
    sub = SubString("😊-xyz", 1, 1)            # the smiley face only
    @test char_tokens(sub) == [ "😊" ]
end


@testset "char_tokens - paragraph smoke test" begin
    para = "The café—naïve coöperation 🤯👩🏽‍🚀.\nLine2 with  汉字  and Ωmega."

    tokens = char_tokens(para)                   # yes, this keeps spaces
    @test length(tokens) == length(Unicode.graphemes(para))
    @test all(isvalid, tokens) && !any(==(""), tokens)
end
