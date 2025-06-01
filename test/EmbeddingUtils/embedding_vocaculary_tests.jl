include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "Vocabulary.jl"))


@testset "build_vocabulary — happy path" begin
    toks = ["hello", "world", "hello", "foo"]
    vdict = build_vocabulary(toks;                    # <- function under test
                             min_freq      = 1,
                             special_tokens = ["<pad>", "<unk>"])

    ti  = vdict["token_to_index"]
    it  = vdict["index_to_token"]
    freq = vdict["freq"]

    @test it[1] == "<pad>"           # special tokens are first
    @test it[2] == "<unk>"
    @test ti["hello"] == 3           # after 2 specials, 'hello' is third
    @test freq["hello"] == 2         # counted twice
    @test length(it) == 5            # <pad> <unk> hello world foo
end

@testset "convert_tokens_to_ids <-> ids_to_tokens round-trip" begin
    voc = Vocabulary(Dict("<unk>" => 1),        # token2id
                     ["<unk>"],                 # id2token
                     Dict{Int,Int}(),           # counts
                     1)                         # unk_id

    ids = convert_tokens_to_ids(["foo","foo","bar"], voc;
                                add_new = true, update_counts = false)

    @test ids == [2,2,3]
    @test voc.id2token == ["<unk>","foo","bar"]

    toks_back = convert_ids_to_tokens(ids, voc)
    @test toks_back == ["foo","foo","bar"]
end



@testset "min_freq filters singletons" begin
    toks  = ["a","b","a","c","d","d","d"]         # b and c appear once
    vdict = build_vocabulary(toks; min_freq = 2)

    @test !haskey(vdict["token_to_index"], "b")   # filtered out
    @test !haskey(vdict["token_to_index"], "c")
    @test vdict["freq"]["d"] == 3                 # freq table is still full
end


@testset "max_vocab_size truncates after sorting by freq" begin
    toks  = ["w","x","y","z","w","x","y"]         # w=2, x=2, y=2, z=1
    vdict = build_vocabulary(toks;
                             max_vocab_size = 2)  # keep the 2 most frequent

    kept = keys(vdict["token_to_index"])
    @test length(kept) == 2
    @test all(k in ("w","x","y") for k in kept)    # z cannot be present
end


@testset "special tokens are unique and kept in order" begin
    toks  = ["foo","bar"]
    vdict = build_vocabulary(toks;
                             special_tokens = ["<pad>","<unk>","<pad>"]) # duplicate

    it = vdict["index_to_token"]
    @test it[1:2] == ["<pad>","<unk>"]   # duplicate removed
    @test count(==( "<pad>" ), it) == 1  # only once in the final list
end


@testset "convert_tokens_to_ids with add_new = false (OOV logic)" begin
    # Build a tiny fixed vocab first
    voc = Vocabulary(Dict("foo"=>1,"bar"=>2,"<unk>"=>3),
                     ["foo","bar","<unk>"],
                     Dict{Int,Int}(),
                     3)            # unk_id = 3

    ids = convert_tokens_to_ids(["foo","baz"], voc;
                                add_new = false, update_counts = false)

    @test ids == [1,3]                       # baz mapped to unk_id
    @test !haskey(voc.token2id, "baz")       # vocabulary unchanged
end


@testset "counts are updated correctly" begin
    voc = Vocabulary(Dict("<unk>" => 1),
                     ["<unk>"],
                     Dict{Int,Int}(),
                     1)

    _ = convert_tokens_to_ids(["foo","foo","bar"], voc;
                              add_new = true, update_counts = true)

    @test voc.counts[2] == 2   # foo seen twice
    @test voc.counts[3] == 1   # bar seen once
    @test !haskey(voc.counts, 1) # <unk> never used -> no entry
end


@testset "convert_batch_tokens_to_ids pads to max length" begin
    voc = Vocabulary(Dict("<unk>" => 1),
                     ["<unk>"],
                     Dict{Int,Int}(),
                     1)

    batch = [["a","b","c","d"], ["e","f"]]
    M = convert_batch_tokens_to_ids(batch, voc;
                                    add_new = true,
                                    update_counts = false)

    @test size(M) == (4, 2)                 # longest doc length × n_docs
    @test all(M[3:end, 2] .== 1)            # rows 3-4 of column 2 padded with unk_id
end


@testset "save_vocabulary -> load_vocabulary round-trip" begin
    vdict = build_vocabulary(["α","β","β","γ"])
    tmp   = tempname()

    save_vocabulary(vdict, tmp)
    vdict2 = load_vocabulary(tmp)

    @test vdict2 == vdict              # JSON round-trip exact
end


@testset "build_vocabulary - free-form messy paragraph" begin
    # glyph helpers 
    nbsp   = "\u00A0"; zwsp = "\u200B"; rle = "\u202B"; pdf = "\u202C"
    famemo = "👨‍👩‍👧‍👦"; astro = "👩🏽‍🚀"
    combé  = "e\u0301"; objrep = "\uFFFC"

    para1 = """
    Once upon a time, there was a naïve coöperation between café-owners—
    but suddenly things went 🤯.$(zwsp)
    Meanwhile, 数学 is fun;$(nbsp) however, $(rle)مرحبا بالعالم$(pdf) was written
    backwards, and $(astro) went to the 🌖 in one small-step…
    """

    para2 = """
    ﬁreﬂlies flew past $(famemo), but the objet$(nbsp)d’art was actually $(objrep).
    Zero-width again:$(zwsp)$(zwsp) done.  Accént: $(combé).
    """

    #simple tokenizer: letters, numbers, emoji
    tokenize(txt) = strip.(split(replace(txt, r"[^\p{L}\p{N}\p{Emoji_Presentation}]" => " "),
                                     ' '; keepempty = false))
    tokens   = vcat(tokenize(para1), tokenize(para2))
    tokens_s = String.(tokens)

    specials = ["<pad>", "<unk>"]
    cap      = 40                                         # regular-token cap
    vocab    = build_vocabulary(tokens_s;
                                max_vocab_size = cap,
                                special_tokens = specials)

    id2tok = vocab["index_to_token"];  tok2id = vocab["token_to_index"]

    #specials unique & leading
    @test id2tok[1:2] == specials

    #final length <= cap + nSpecials
    @test length(id2tok) <= cap + length(specials)

    #all tokens valid, non-empty UTF-8
    @test all(isa(t,String) && !isempty(t) && isvalid(t) for t in id2tok)

    #no token is only control / whitespace
    ctrl_or_space = r"^[\p{Cc}\s]+$"
    @test !any(occursin(ctrl_or_space, t) for t in id2tok)

    #at least one of the exotic glyphs survived the cap *or* the
    # vocabulary is completely full (which means truncation decided)
    exotic_pool = ["🤯","数学",combé,famemo,astro]
    survived = any(haskey(tok2id, ex) for ex in exotic_pool)
    @test survived || length(id2tok) == cap + length(specials)

    #frequency dictionary counts only corpus tokens;
    #    <unk> never appears, so get(...,0) should be 0
    @test get(vocab["freq"], "<unk>", 0) == 0

end


@testset "ensure_unk! - behaviour matrix" begin
    # bring the type and the helper in without clashing with a
    # similarly-named struct that may exist in Main
    # import TextSpace.Preprocessing: Vocabulary, ensure_unk!

    #already-valid vocabulary - same object returned
    v_ok = Vocabulary(Dict("<unk>"=>1, "a"=>2),
                      ["<unk>","a"],
                      Dict(1=>3, 2=>7),
                      1)

    ret_ok = ensure_unk!(v_ok)
    @test ret_ok === v_ok
    @test v_ok.unk_id == 1 && v_ok.id2token[1] == "<unk>"

    #unk_id = 0 and "<unk>" missing -> fresh vocab with new unk
    v_missing = Vocabulary(Dict("b"=>1), ["b"], Dict{Int,Int}(), 0)

    ret_missing = ensure_unk!(v_missing)

    @test ret_missing !== v_missing
    @test ret_missing.unk_id == 2
    @test ret_missing.token2id["<unk>"] == 2
    @test ret_missing.id2token == ["b", "<unk>"]
    @test !haskey(v_missing.token2id, "<unk>")          # original untouched

    #unk_id = 0 but "<unk>" already present -> duplicate by design
    v_dup = Vocabulary(Dict("<unk>"=>1, "c"=>2),
                       ["<unk>","c"],
                       Dict{Int,Int}(),
                       0)

    ret_dup = ensure_unk!(v_dup)

    @test ret_dup !== v_dup
    @test ret_dup.unk_id == 3
    @test ret_dup.token2id["<unk>"] == 3
    @test count(==("<unk>"), ret_dup.id2token) == 2      # two entries

    #counts dict is deep-copied, not aliased
    cnts = Dict(1=>5)
    v_cnt = Vocabulary(Dict("x"=>1), ["x"], cnts, 0)

    ret_cnt = ensure_unk!(v_cnt)

    @test ret_cnt.counts == cnts
    @test ret_cnt.counts !== cnts
end




