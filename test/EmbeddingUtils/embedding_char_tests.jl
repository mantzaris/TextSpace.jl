



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































