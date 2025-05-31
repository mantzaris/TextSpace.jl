include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SubwordProcessing.jl"))




@testset "build_vocabulary_bpe - heavy UTF-8 hammer" begin
    # weird glyphs 
    zwsp  = "\u200B"                     # ZERO-WIDTH SPACE
    nbsp  = "\u00A0"                     # NO-BREAK SPACE
    rle   = "\u202B";  pdf = "\u202C"    # bidi controls
    combÉ = "e\u0301"                    # e + COMBINING ACUTE
    ligfi = "ﬁ";    ligfl = "ﬂ"          # ligature chars
    fam   = "👨‍👩‍👧‍👦";  astro = "👩🏽‍🚀";  boom = "🤯"
    cjk   = "漢字";  greek = "Ωmega"
    objrep = "\uFFFC"                    # OBJECT REPLACEMENT CHAR
    longA  = repeat("a", 40)             # forces deep merge chain

    paraA = """
    $longA $longA $longA   # repeated to drive merges
    Tabs,\tnew-lines, and $nbsp non-breaking spaces. $zwsp$zwsp
    $boom happened at the café when $astro waved hello to $fam on the 🌖.


    $cjk studies & $greek lessons mixed with $ligfi$ligfl and $combÉ.
    """

    paraB = """
    Zero-width joiners:$zwsp$zwsp$zwsp done…
    Here comes RTL: $(rle)مرحبا بالعالم$(pdf) then back.  Object$(objrep).
    Long CJK run: $(repeat(cjk*" ", 10))
    """

    corpus = split(paraA * '\n' * paraB, '\n'; keepempty = false)

    specials = ["<pad>", "<unk>", "<cls>"]
    cap      = 60
    vbpe = build_vocabulary_bpe(corpus;
                                vocab_size     = cap,
                                special_tokens = specials)

    id2tok = vbpe["index_to_token"];  tok2id = vbpe["token_to_index"]

    #merge loop ran and produced multi-char token
    @test !isempty(vbpe["merges"])
    @test any(length(t) > 1 for t in id2tok)

    #size <= cap + specials
    @test length(id2tok) <= cap + length(specials)

    #specials deduped & leading
    @test id2tok[1:3] == specials

    #tokens valid, non-empty UTF-8
    @test all(isa(t,String) && !isempty(t) && isvalid(t) for t in id2tok)

    #no pure control/space tokens
    ctrl_blank = r"^[\p{Cc}\s]+$"
    @test !any(occursin(ctrl_blank, t) for t in id2tok)

    #at least one exotic glyph survived truncation
    @test any(haskey(tok2id, ex) for ex in (boom, astro, cjk, combÉ, ligfi))
end


@testset "build_vocabulary_bpe - extended coverage" begin
    #small corpus: at least one merge + multi-char token 
    corpus1 = ["banana banana"]
    v1      = build_vocabulary_bpe(corpus1;
                                   vocab_size     = 20,
                                   special_tokens = ["<pad>"])

    @test !isempty(v1["merges"])                      # merge loop ran
    @test any(length(tok) > 1 for tok in v1["index_to_token"])

    #vocab-size cap & duplicate special-token handling 
    bigcorpus = [repeat("x y z ", 10),
                 repeat("x y ",  5),
                 repeat("x ",    20)]
    vmax   = 6
    v2     = build_vocabulary_bpe(bigcorpus;
                                  vocab_size     = vmax,
                                  special_tokens = ["<s>", "</s>", "<s>", "<unk>"])

    @test length(v2["index_to_token"]) == vmax
    @test v2["index_to_token"][1:3] == ["<s>", "</s>", "<unk>"]

    #unicode / emoji robustness    
    uni_corpus = ["汉 汉 字", "👾 👾", "汉 👾"]
    v3         = build_vocabulary_bpe(uni_corpus; vocab_size = 25)

    for tok in ("汉", "字", "👾")
        @test haskey(v3["token_to_index"], tok)
    end
    @test all(isa(t,String) && !isempty(t) for t in v3["index_to_token"])
end


@testset "build_vocabulary_bpe" begin
    #ASCII corpus that forces at least one merge      
    corpus = ["aaaa",            # "a a a a" inside one word -> ('a','a') pair
              "a b a b",
              "c"]

    vdict = build_vocabulary_bpe(corpus;
                                 vocab_size     = 10,
                                 special_tokens = ["<pad>", "<unk>"])

    ti = vdict["token_to_index"];      it = vdict["index_to_token"]

    #specials first, unique
    @test it[1:2] == ["<pad>", "<unk>"]
    @test count(==("<pad>"), it) == 1

    #size respects the cap
    @test length(it) <= 10

    #we know at least one merge must have happened
    @test !isempty(vdict["merges"])

    #frequency table should account for every character token
    total_chars = sum(sum(length.(split(sent))) for sent in corpus)
    freq_sum    = sum(values(vdict["freq"]))

    #after merging, the total number of tokens cannot exceed the original
    @test 0 < freq_sum <= total_chars

    #emoji / Unicode corpus 
    emoji_corpus = ["👩‍🚀 🚀 👩‍🚀",
                    "🚀 🚀"]

    vdict2 = build_vocabulary_bpe(emoji_corpus; vocab_size = 15)

    @test haskey(vdict2["token_to_index"], "👩‍🚀")
    @test haskey(vdict2["token_to_index"], "🚀")

    #either some merge occurred (length>1 tokens) or merge list is empty
    merged_any = any(length(tok) > 1 for tok in vdict2["index_to_token"])
    @test merged_any || isempty(vdict2["merges"])

    #empty corpus edge-case
    empty = build_vocabulary_bpe(String[]; vocab_size = 5,
                                 special_tokens = ["<s>"])
    @test empty["index_to_token"] == ["<s>"]
end


@testset "count_pair_frequencies" begin
    # Helper: two sentences
    ps = [
        [ ["h","e","l","l","o"], ["o"] ],        # sentence 1
        [ ["h","e"] ]                            # sentence 2
    ]

    freq = count_pair_frequencies(ps)

    expected = Dict(
        ("h","e") => 2,   # appears in both sentences
        ("e","l") => 1,
        ("l","l") => 1,
        ("l","o") => 1
    )

    @test freq == expected

    # Word of length 1 should not create any pair
    single = [ [ ["x"] ] ]
    @test isempty(count_pair_frequencies(single))

    # Empty input edge case
    @test isempty(count_pair_frequencies(Vector{Vector{Vector{String}}}()))
end


@testset "count_pair_frequencies - extended" begin
    #regular multi-sentence, multi-word mix (baseline coverage)
    corpus = [
        [ ["h","e","l","l","o"], ["👋"] ],      # sentence 1
        [ ["h","e"], ["l","l","o"] ],           # sentence 2
        [ ["👩‍🚀","🚀"] ]                       # sentence 3 (emoji tokens)
    ]

    freq = count_pair_frequencies(corpus)

    @test freq[("h","e")] == 2        # (hello + second sentence)
    @test freq[("e","l")] == 1
    @test freq[("l","l")] == 2        # two distinct words provide it
    @test freq[("l","o")] == 2
    @test freq[("👩‍🚀","🚀")] == 1
    @test !haskey(freq, ("👋","h"))   # no cross-word counting
    @test !haskey(freq, ("o","👋"))

    #repeated identical token in one  word
    repeated = [ [ ["a","a","a","a"] ] ]   # 4 chars to 3 adjacent pairs
    repf = count_pair_frequencies(repeated)
    @test repf[("a","a")] == 3

    #length-1 and empty words / sentences
    shorty = [
        [ ["x"] ],               # single-token word
        [],                      # completely empty sentence
        [ ["y","z"] ]
    ]
    sf = count_pair_frequencies(shorty)
    @test sf == Dict(( "y","z" ) => 1)     # only one valid pair

    #empty corpus
    @test isempty(count_pair_frequencies(Vector{Vector{Vector{String}}}()))

    #input immutability (function is pure)
    deep = deepcopy(corpus)
    _ = count_pair_frequencies(corpus)
    @test corpus == deep              # no mutation occurred
end


@testset "count_pair_frequencies - UTF-8 hammer" begin
    zwsp = "\u200B"                             # ZERO-WIDTH SPACE

    sentence1 = ["👩🏽‍🚀🤯", "漢字", "漢字"]
    sentence2 = ["👩🏽‍🚀🤯", "banana", "x$(zwsp)z"]

    docs = [
        [char_tokens(w) for w in sentence1],
        [char_tokens(w) for w in sentence2]
    ]

    freq = count_pair_frequencies(docs)

    #pair counts we can predict exactly         
    @test freq[("👩🏽‍🚀","🤯")] == 2          # word appears in both sentences
    @test freq[("漢","字")]     == 2          # “漢字” twice in sentence 1
    @test freq[("b","a")]       == 1          # banana pairs
    @test freq[("a","n")]       == 2
    @test freq[("n","a")]       == 2
    @test freq[("x", zwsp)]     == 1          # zero-width space inside word
    @test freq[(zwsp,"z")]      == 1

    #global consistency: total pairs == sum(|word|-1)   
    total_pairs = sum(max(length(w) - 1, 0) for s in docs for w in s)
    @test sum(values(freq)) == total_pairs

    #edge-case sanity: empty corpus returns empty Dict  
    @test isempty(count_pair_frequencies(Vector{Vector{Vector{String}}}()))
end


@testset "count_pair_frequencies - paragraph UTF-8 hammer" begin
    using Unicode

    # Weird helpers
    zwsp   = "\u200B"                      # ZERO-WIDTH SPACE
    nbsp   = "\u00A0"                      # NO-BREAK SPACE
    rle    = "\u202B";  
    pdf = "\u202C"     # bidi controls
    family = "👨‍👩‍👧‍👦";    
    astro = "👩🏽‍🚀";  
    flag = "🇪🇺"   # emoji
    boom   = "🤯"
    cjk    = "漢字";        arab  = "مرحبا"
    combé  = "e\u0301";     ligfi = "ﬁ"
    longX  = repeat("x", 50)               # force big pair count xx

    para1 = """
    The $flag agreed – but $longX didn't.  $(boom)s  Tabs,\tnew lines,
    and $nbsp non-breaking spaces    mix with $zwsp zero-widths.  $family
    met $astro on the 🌖.  Greek Ωmega meets $cjk and $arab.
    """

    para2 = """
    Ligatures like $ligfi and accents like $combé matter.$zwsp
    Here comes RTL: $(rle)$arab بالعالم$(pdf) then back.
    A sequence of emoji 👩‍❤️‍💋‍👨 forms tricky graphemes.
    """

    #convert to BPE input shape: [sentence][word][token] 
    corpus   = split(para1 * '\n' * para2, '\n'; keepempty = false)
    sentences = [ [char_tokens(w) for w in split(s)] for s in corpus ]

    pf = count_pair_frequencies(sentences)

    #specific pairs we can predict exactly     
    @test pf[("x","x")] == 49                        # "x"*50 -> 49 overlaps
    @test pf[("a","g")] == 1
    @test pf[("$(boom)","s")] == 1
    @test pf[("漢","字")]    == 1                    # CJK pair once

    #RTL segment broken into graphemes: we can't assert exact pair, but count
    @test sum(v for (k,v) in pf if k[1] == "م") >= 1  #arabic letter Meem start

    #global sanity: total pair count matches sum(|word|-1)     
    total_pairs = sum(max(length(w)-1, 0) for s in sentences for w in s)
    @test sum(values(pf)) == total_pairs

    #all keys are valid 2-tuple String tokens
    @test all(isa(k[1],String) && isa(k[2],String) for k in keys(pf))

    #no pair frequency is zero               
    @test all(v > 0 for v in values(pf))

    #empty corpus still returns empty Dict   
    @test isempty(count_pair_frequencies(Vector{Vector{Vector{String}}}()))
end


@testset "build_vocabulary_wordpiece" begin
    corpus = ["low lower lowest", "new newer newest"]
    vocab  = build_vocabulary_wordpiece(corpus;
                                        vocab_size     = 12,
                                        special_tokens = ["<pad>", "<cls>"])

    it = vocab["index_to_token"]; ti = vocab["token_to_index"]

    @test it[1:3] == ["<pad>", "<cls>", "[UNK]"]
    @test length(it) <= 12
    special_count = 3
    @test length(it) > special_count               # <- updated assertion

    @test count(==("<pad>"), it) == 1
    @test haskey(ti, "[UNK]") && ti["[UNK]"] == 3

    uni = build_vocabulary_wordpiece(["汉字 汉", "字"], vocab_size=15)
    @test haskey(uni["token_to_index"], "汉")
    @test haskey(uni["token_to_index"], "字")
end


@testset "build_vocabulary_wordpiece - weird UTF-8 corpus" begin
    zwsp   = "\u200B"                 # ZERO-WIDTH SPACE
    nbsp   = "\u00A0"                 # NO-BREAK SPACE
    combé  = "e\u0301"                # e + COMBINING ACUTE
    famemo = "👨‍👩‍👧‍👦"              # family emoji (ZWJ sequence)
    astro  = "👩🏽‍🚀"                # skin-tone + ZWJ

    corpus = [
        "$(zwsp)$(zwsp)",             # two invisibles only
        "abc$(zwsp) def",             # ZWSP inside text
        "foo$(nbsp)bar",              # NBSP between tokens
        combé,                        # combining grapheme
        "$(famemo) $(astro)"          # ZWJ emoji + space + ZWJ emoji
    ]

    vocab_cap = 20
    vdict = build_vocabulary_wordpiece(corpus;
                                       vocab_size     = vocab_cap,
                                       special_tokens = ["<pad>", "<unk>"])

    id2tok = vdict["index_to_token"]

    # size respects cap
    @test length(id2tok) <= vocab_cap

    # every token is non-empty, valid UTF-8
    @test all(isa(t,String) && !isempty(t) && isvalid(t) for t in id2tok)

    # no empty-string tokens slipped in
    @test !any(t -> t == "", id2tok)

    # specials deduplicated and first
    @test id2tok[1:2] == ["<pad>", "<unk>"]
end


@testset "build_vocabulary_wordpiece - free-form messy paragraph" begin
    zwsp   = "\u200B"             # ZERO-WIDTH SPACE
    nbsp   = "\u00A0"             # NO-BREAK SPACE
    rle    = "\u202B"             # RIGHT-TO-LEFT EMBEDDING
    pdf    = "\u202C"             # POP DIRECTIONAL FORMAT
    combé  = "e\u0301"            # e + COMBINING ACUTE
    famemo = "👨‍👩‍👧‍👦"        # family emoji (ZWJ sequence)
    astro  = "👩🏽‍🚀"            # skin-tone + ZWJ

    para = """
    Once upon a time, there was a naïve coöperation between café-owners—
    but suddenly things went 🤯. They said: “ﬁreﬂlies?  No,  ﬂoof!”$(zwsp)
    Meanwhile, 数学 is fun;$(nbsp) however, $(rle)مرحبا بالعالم$(pdf) was written
    backwards, and $(astro) went to the 🌖 in one small-step…\tTabs, new-lines,
    zero-width-joiners, and \uFFFC (U+FFFC OBJECT) all mix *together*.
    $(famemo)
    """

    sentences = split(para, '\n'; keepempty = false)      # portable splitlines

    vocab_cap = 60
    vdict = build_vocabulary_wordpiece(sentences;
                                       vocab_size     = vocab_cap,
                                       special_tokens = ["<pad>", "<cls>", "<unk>"])

    id2tok = vdict["index_to_token"]

    #cap respected
    @test length(id2tok) <= vocab_cap

    #each token is non-empty valid UTF-8
    @test all(isa(t,String) && !isempty(t) && isvalid(t) for t in id2tok)

    #no token is only control / whitespace
    ctrl_or_space = r"^[\p{Cc}\s]+$"
    @test !any(occursin(ctrl_or_space, t) for t in id2tok)

    #specials deduped & leading
    @test id2tok[1:3] == ["<pad>", "<cls>", "<unk>"]

    #at least one non-ASCII glyph survived (emoji or CJK)
    @test any(occursin(r"[🤯👩🏽‍🚀🌖数]", t) for t in id2tok)
end


@testset "merge_pair_in_sentences!" begin
    # helper builder
    function make(sentences)
        deepcopy(sentences)   # keep original intact for comparison
    end

    #single merge inside word
    ps  = make([ [ ["h","e","l","l","o"] ] ])
    merge_pair_in_sentences!(ps, ("l","l"))
    @test ps == [ [ ["h","e","ll","o"] ] ]

    #overlapping pattern ("a","a") in "aaaa" should give 2
    ps  = make([ [ ["a","a","a","a"] ] ])
    merge_pair_in_sentences!(ps, ("a","a"))
    @test ps == [ [ ["aa","aa"] ] ]

    #no merge required (pair absent)
    orig = [ [ ["x","y","z"] ] ]
    ps   = make(orig)
    merge_pair_in_sentences!(ps, ("a","b"))
    @test ps == orig    # corpus unchanged

    #multiple sentences + words  
    ps = make([
        [ ["a","b"], ["b","c"] ],         # sentence 1
        [ ["a","b","b","c"] ]         # sentence 2
    ])
    merge_pair_in_sentences!(ps, ("b","c"))
    # expected: sentence1 : ["a","b"] , ["bc"], sentence2 : ["a","b","bc"]
    @test ps == [
        [ ["a","b"], ["bc"] ],
        [ ["a","b","bc"] ]
    ]

    #word length <= 1 handled gracefully
    ps = make([ [ ["x"], ["y"] ] ])
    merge_pair_in_sentences!(ps, ("x","y"))   # nothing to merge
    @test ps == [ [ ["x"], ["y"] ] ]

    #unicode / emoji tokens
    ps = make([ [ ["👩‍🚀","🚀"] ] ])
    merge_pair_in_sentences!(ps, ("👩‍🚀","🚀"))
    @test ps == [ [ ["👩‍🚀🚀"] ] ]
end


@testset "merge_pair_in_sentences! - UTF-8 hammer" begin
    zwsp   = "\u200B"                       # ZERO-WIDTH SPACE
    #has many overlapping "a","a" pairs, plus cross-word "a a"
    s1 = [
        ["a","a","a","a"],                  # "aaaa"
        ["x","a","a"],                      # "xaa"
        ["👩","🏽"]                         # emoji pair merge later
    ]
    #mixes CJK, repeats, zero-width space inside word
    s2 = [
        ["漢","字"],                        # no merge here
        ["a","a","b"], ["a"], ["a","a"],    # cross-word "a a"
        ["x",zwsp,"x"]                      # zero-width in middle
    ]

    docs = [deepcopy(s1), deepcopy(s2)]     # assemble corpus

    #merge ("a","a") first                   
    merge_pair_in_sentences!(docs, ("a","a"))

    #expected results
    @test docs[1][1] == ["aa","aa"]         # "aaaa" -> 2 non-overlapping "aa"
    @test docs[1][2] == ["x","aa"]          # only inner pair merged
    @test docs[1][3] == ["👩","🏽"]          # untouched

    #no cross-word merges: ["a"] + ["a","a"] stays separate
    @test docs[2][3] == ["a"] && docs[2][4] == ["aa"]

    #zero-width split word still length 3 (no false merge)
    @test docs[2][5] == ["x", zwsp, "x"]

    # ("a","a") pair no longer appears adjacent inside any word
    @test all(
        all(i == length(w) || w[i:i+1] != ["a","a"] for i in 1:length(w)-1)
        for sent in docs for w in sent
    )

    #now merge an emoji pair ("👩","🏽")             
    merge_pair_in_sentences!(docs, ("👩","🏽"))
    @test docs[1][3] == ["👩🏽"]            # merged into one grapheme
    @test !haskey(count_pair_frequencies([["👩","🏽"]]), ("👩","🏽"))  # sanity

    #function is in-place and returns nothing     
    ret = merge_pair_in_sentences!(docs, ("x","x"))
    @test ret === nothing
end


@testset "extract_all_symbols" begin
    # helper to build the nested structure
    ps = [
        [ ["h","e"], ["l","l","o"] ],
        [ ["h","e"], ["👩‍🚀"] ]
    ]

    syms = extract_all_symbols(ps)

    @test syms == Set(["h","e","l","o","👩‍🚀"])

    #no duplicates
    @test length(syms) == 5

    #empty input edge-case
    empty_ps = Vector{Vector{Vector{String}}}()
    @test isempty(extract_all_symbols(empty_ps))
end


@testset "extract_all_symbols - paragraph UTF-8 hammer" begin
    using Unicode

    # helpers
    zwsp  = "\u200B"                     # ZERO-WIDTH SPACE
    nbsp  = "\u00A0"                     # NO-BREAK SPACE
    rle   = "\u202B"; pdf = "\u202C"     # bidi controls
    fam   = "👨‍👩‍👧‍👦"; astro = "👩🏽‍🚀"; boom = "🤯"
    cjk   = "漢字";  arab = "مرحبا"
    combé = "e\u0301"; ligfi = "ﬁ"

    paragraphs = [
        "Hello café \n $boom $(astro)$(zwsp)!!!",
        "Studies in $cjk and Ωmega; $nbsp $rle$arab$pdf world... Yay",
        "Ligatures: $ligfi; accents: $combé."
    ]

    #build nested structure exactly like build_vocabulary_bpe
    docs = [ [char_tokens(w) for w in split(p)] for p in paragraphs ]

    #expected set computed the long way
    expected = Set{String}()
    for s in docs, w in s, t in w
        push!(expected, t)
    end

    #function under test
    syms = extract_all_symbols(docs)

    @test syms == expected                           # exact match
    @test all(isa(t,String) && !isempty(t) && isvalid(t) for t in syms)

    #contains a few representative exotic tokens
    for ex in ("👩🏽‍🚀", "🤯", "漢", "字", combé)
        @test ex in syms
    end

    #no duplicates -> set length equals unique count
    uniq_cnt = length(unique(Iterators.flatten(Iterators.flatten(docs))))
    @test length(syms) == uniq_cnt

    #empty corpus still returns empty set
    @test isempty(extract_all_symbols(Vector{Vector{Vector{String}}}()))
end