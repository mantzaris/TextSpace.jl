include(joinpath(@__DIR__, "..", "src", "preprocessing", "CharProcessing.jl"))


@testset "tokenize_char" begin
    txt = "Café 😊"

    # default: NFC, no spaces, original case
    toks = tokenize_char(txt)
    @test toks == ["C","a","f","é","😊"]

    # lower-case, keep space
    toks2 = tokenize_char(txt; lower = true, keep_space = true)
    @test toks2 == ["c","a","f","é"," ","😊"]

    # ensure combined accents survive when we skip normalisation (:NFD)
    nfd = Unicode.normalize("é", :NFD)  # 'e' + combining acute
    toks3 = tokenize_char(nfd; normalize = false)
    @test length(toks3) == 2            # two graphemes without NFC merge
end

@testset "chars_to_ids" begin
    voc = Vocabulary(Dict("<unk>" => 1), ["<unk>"], Dict{Int,Int}(), 1)
    ids = chars_to_ids(["a","b","a"], voc; add_new = true)

    @test ids == [2,3,2]                   # “a” got id 2, “b” id 3
    @test voc.id2token[2:3] == ["a","b"]   # vocabulary grew as expected
    @test chars_to_ids(["z"], voc) == [1]  # unk when add_new = false
end


@testset "encode_char_batch" begin
    voc = Vocabulary(Dict("<unk>" => 1, "h"=>2, "i"=>3, "😊"=>4),
                     ["<unk>","h","i","😊"],
                     Dict{Int,Int}(), 1)

    mat = encode_char_batch(["hi","😊h"], voc)
    @test size(mat) == (2, 2)              # longest seq = 2
    @test mat[:,1] == [2,3]                # “h i”
    @test mat[1,2] == 4 && mat[2,2] == 2   # “😊 h” (padded already OK)
end