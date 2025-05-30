include(joinpath(@__DIR__, "..", "..", "src", "preprocessing", "SubwordTokenization.jl"))


using Serialization, Random, Statistics, AliasTables, StatsBase, Downloads
using Statistics: mean
using LinearAlgebra
using Zygote, Flux
using StatsBase: countmap
using AliasTables: AliasTable, rand

import BytePairEncoding as BPE

const SWU = TextSpace.SubwordEmbeddings.SubwordEmbeddingUtilities
const SWE = TextSpace.SubwordEmbeddings

const PP = TextSpace.Preprocessing











####################################
# PRE TRAINED BPE (down)
###################################

# @testset "windowify simple" begin
#     toks = 1:5

#     expected_ctx = [
#         (1, [2]),
#         (2, [1,3]),
#         (3, [2,4]),
#         (4, [3,5]),
#         (5, [4])
#     ]
#     @test collect(SWU.windowify(toks; window_size=1)) == expected_ctx

#     sg_pairs = collect(SWU.windowify(toks; window_size=1, as_pairs=true))
#     @test length(sg_pairs) == 8
#     @test first(sg_pairs)  == (1,2)
#     @test last(sg_pairs)   == (5,4)
# end


# @testset "Sub-word Skip-Gram / CBOW wrappers" begin
#     ids = 1:5 # toy token stream
#     radius = 1

#     #  Skip-Gram
#     sg = SWU.skipgram_pairs(ids, radius)
#     @test length(sg) == 8
#     @test first(sg)  == (1,2)
#     @test last(sg)   == (5,4)

#     #  CBOW
#     cb = SWU.cbow_pairs(ids, radius)
#     expected = [
#         (1, [2]),
#         (2, [1,3]),
#         (3, [2,4]),
#         (4, [3,5]),
#         (5, [4])
#     ]
#     @test cb == expected
# end


# @testset "BPE round-trip" begin
#     enc = SWU.load_encoder("cl100k_base")
#     ids = SWU.encode("hello world", enc)
#     @test SWU.decode(ids, enc) == "hello world"
# end


# @testset "BPE vocab & ID range" begin
#     enc  = SWU.load_encoder("cl100k_base")
#     ids  = SWU.encode("JuliaLang is fast 🏎️", enc)

#     # TikToken cl100k_base hard-codes 100 277 entries
#     @test length(enc.vocab) == 100_277
#     @test maximum(ids) < length(enc.vocab)
#     @test minimum(ids) >= 0

#     @test SWU.used_vocab_size(enc) ≤ length(enc.vocab)     # still true
#     @test maximum(ids) < SWU.used_vocab_size(enc)
# end


# @testset "encode -> skip-gram -> window math" begin
#     txt   = "hello world, hello Julia"
#     ids   = SWU.encode(txt, SWU.load_encoder()) # sub-word IDs
#     pairs = SWU.skipgram_pairs(ids, 1)     # radius = 1

#     expected = 2length(ids) - 2              
#     @test length(pairs) == expected

#     @test first(pairs) == (ids[1],  ids[2])      # sanity
#     @test last(pairs)  == (ids[end], ids[end-1])

#     for (c, ctx) in pairs
#         @test abs(findfirst(==(c), ids) - findfirst(==(ctx), ids)) ≤ 1
#     end
# end


# @testset "BPE encoder persistence" begin
#     enc = SWU.load_encoder()
#     tmp = mktempdir()
#     path = joinpath(tmp, "enc.bin")
#     serialize(path, enc)

#     enc2 = deserialize(path)
#     str  = "sub-word embedding test"
#     @test SWU.decode(SWU.encode(str, enc2), enc2) == str
#     @test SWU.used_vocab_size(enc2) == SWU.used_vocab_size(enc)
# end


# @testset "TikToken vs GPT-2 encoders differ" begin
#     tikt  = SWU.load_encoder("cl100k_base")
#     gpt2  = SWU.load_encoder("gpt2")

#     txt   = "hello world"
#     ids_a = SWU.encode(txt, tikt)
#     ids_b = SWU.encode(txt, gpt2)

#     @test ids_a != ids_b
#     @test SWU.decode(ids_a, tikt) == txt
#     @test SWU.decode(ids_b, gpt2) == txt
# end


# @testset "multilingual round-trip" begin
#     enc = SWU.load_encoder("cl100k_base")   # reuse official TikToken

#     str = "Καλημέρα κόσμε, こんにちは世界 🌍"        # Greek + Japanese + emoji
#     ids = SWU.encode(str, enc)

#     @test isa(ids, Vector{Int})
#     @test !isempty(ids)
#     @test maximum(ids) < SWU.used_vocab_size(enc)  # stays within vocab

#     @test SWU.decode(ids, enc) == str     #   round-trip
# end


# @testset "encoder edge cases" begin
#     enc = SWU.load_encoder()

#     #empty string -> zero tokens
#     @test isempty(SWU.encode("", enc))

#     #single character encodes to >= 1 ID
#     @test !isempty(SWU.encode("a", enc))

#     #very long string still encodes
#     long_str = repeat("a", 10_000)
#     @test !isempty(SWU.encode(long_str, enc))

#     #emoji round-trip
#     emoji = "😀🤖👾🔥"
#     @test SWU.decode(SWU.encode(emoji, enc), enc) == emoji

#     #mixed scripts round-trip
#     mixed = "English العربية 中文 Русский"
#     @test SWU.decode(SWU.encode(mixed, enc), enc) == mixed
# end



# @testset "encoder save/load helper" begin
#     enc   = SWU.load_encoder("cl100k_base")   # pre-trained

#     tmp   = mktempdir()
#     path  = joinpath(tmp, "tok.bin")
#     SWU.save_encoder(path, enc)

#     enc2  = SWU.load_encoder_from_file(path)      
#     str   = "Καλημέρα κόσμε 🌍"
#     @test SWU.decode(SWU.encode(str, enc2), enc2) == str
#     @test SWU.used_vocab_size(enc2) == SWU.used_vocab_size(enc)
# end


# @testset "subword sgns smoke" begin
#     corpus = ["hello world", "hello Julia"]
#     m, enc = SubwordEmbeddings.train!(corpus; epochs=1, batch=4, emb_dim=32)
#     v = SubwordEmbeddings.vector(m, enc, "hello")
#     @test length(v) == 32
# end


# #CBOW 1-epoch smoke-test
# @testset "subword CBOW smoke" begin
#     corpus = ["flux is amazing", "hello flux"]
#     m, enc = SubwordEmbeddings.train!(corpus;
#                                       objective   = :cbow,
#                                       epochs      = 1,
#                                       batch       = 4,
#                                       emb_dim     = 32,
#                                       k_neg       = 2)
#     # vectors have expected length
#     v = SubwordEmbeddings.vector(m, enc, "flux")
#     @test length(v) == 32
# end


# #Embedding matrix vs vocab size
# @testset "embedding rows = used vocab size" begin
#     enc = SWU.load_encoder("gpt2")
#     vN  = SWU.used_vocab_size(enc)
#     m   = SubwordEmbeddings.SkipGramModel(vN, 16) #tiny dim

#     @test size(SubwordEmbeddings.embeddings(m), 2) == vN
# end


# #save_embeddings / load_embeddings round-trip
# @testset "embedding save/load round-trip" begin
#     corpus = ["tiny corpus"]
#     m, enc  = SubwordEmbeddings.train!(corpus; epochs=1, batch=2, emb_dim=8)

#     tmp  = mktempdir()
#     file = joinpath(tmp, "sub_emb.bin")
#     SubwordEmbeddings.save_embeddings(file, m, enc)

#     m2, enc2 = SubwordEmbeddings.load_embeddings(file)
#     tok = "tiny"
#     @test SubwordEmbeddings.vector(m2, enc2, tok) ≈
#           SubwordEmbeddings.vector(m,  enc,  tok)
# end


# @testset "vector() unknown-token still encodes" begin
#     enc = SWU.load_encoder()
#     m   = SubwordEmbeddings.SkipGramModel(SWU.used_vocab_size(enc), 12)

#     vec_known = SubwordEmbeddings.vector(m, enc, "hello")
#     vec_unk   = SubwordEmbeddings.vector(m, enc, "NON_EXISTENT_TOKEN_XYZ")

#     @test length(vec_known) == 12 == length(vec_unk)   # correct size
#     @test vec_known != vec_unk      # returns *some* unk vector
# end


# @testset "skip-gram radius-2 window maths" begin
#     txt  = "one two three four five six"
#     ids  = SWU.encode(txt, SWU.load_encoder("gpt2"))
#     pairs = SWU.skipgram_pairs(ids, 2)       # radius = 2

#     n = length(ids); r = 2
#     expected = 2n*r - r*(r+1)              
#     @test length(pairs) == expected

#     # every pair distance ≤ r
#     for (c, ctx) in pairs
#         @test abs(findfirst(==(c), ids) -
#                   findfirst(==(ctx), ids)) <= r
#     end
# end


# @testset "cbow radius-3 context length" begin
#     ids = SWU.encode("a b c d e f g h", SWU.load_encoder())
#     tuples = SWU.cbow_pairs(ids, 3)
#     for (ctr, ctx) in tuples
#         @test length(ctx) <= 3*2    # left+right
#         @test !(ctr in ctx)
#     end
# end


# @testset "1-batch SGNS loss drop" begin
#     corpus = ["good bad good bad", "bad good"]
#     enc    = SWU.load_encoder()
#     ids    = vcat(SWU.encode.(corpus, Ref(enc))...)
#     pairs  = SWU.skipgram_pairs(ids, 2)

#     pc, po = first.(pairs), last.(pairs)
#     vocabN = SWU.used_vocab_size(enc)
#     m      = SWE.SkipGramModel(vocabN, 16)

#     # tiny negative set just for test
#     freqs  = countmap(ids); toks = collect(keys(freqs))
#     tbl    = AliasTables.AliasTable(Float64.(values(freqs)).^0.75)
#     nc     = repeat(pc, 2)
#     no     = toks[rand(tbl, length(pc)*2)]

#     loss_before = SWE.sg_loss(m, pc, po, nc, no)
#     gs = Zygote.gradient(() -> SWE.sg_loss(m, pc, po, nc, no),
#                          Flux.params(m))
#     Flux.Optimise.update!(Flux.Adam(1e-2), Flux.params(m), gs)
#     loss_after  = SWE.sg_loss(m, pc, po, nc, no)

#     @test loss_after < loss_before    # one gradient step helped
# end


# #model -> encoder save -> load round-trip
# @testset "subword model save/load" begin
#     corpus = ["abc def ghi", "def ghi jkl"]
#     m, enc = SWE.train!(corpus; epochs=1, batch=4, emb_dim=8)

#     tmp  = mktempdir()
#     file = joinpath(tmp, "subword.bin")
#     SWE.save_embeddings(file, m, enc)

#     m2, enc2 = SWE.load_embeddings(file)
#     str = "def"
#     @test SWE.vector(m2, enc2, str) ≈ SWE.vector(m, enc, str)
# end


# @testset "tiny corpus learns" begin
#     corpus = vcat(fill("cat dog", 100), fill("fish", 100))
#     rng    = MersenneTwister(42)

#     m, enc = SWE.train!(corpus; epochs = 30, batch = 32,
#                         emb_dim = 8, lr = 1f-2, rng = rng)

#     e       = SWE.embeddings(m)
#     cat_ids = enc.encode("cat")           # ["cat", " cat"] both map to cat-ish IDs
#     dog_ids = enc.encode("dog")
#     fish_id = enc.encode("fish")[1]

#     # average over token variants
#     cat = mean(e[:, id] for id in cat_ids)
#     dog = mean(e[:, id] for id in dog_ids)
#     fish = e[:, fish_id]

#     cos(u,v) = dot(u,v) / √(dot(u,u)*dot(v,v) + eps())

#     @test cos(cat, dog) > cos(cat, fish) + 0.05   # clear margin
# end


# @testset "tiny corpus learns (deterministic)" begin
#     # 200 lines of the positive pair, 200 of an unrelated word
#     corpus = vcat(fill("cat dog", 200), fill("fish", 200))
#     rng    = MersenneTwister(42)

#     m, enc = SWE.train!(corpus; epochs = 25, batch = 32,
#                         emb_dim = 8, lr = 1e-2, rng = rng)

#     W = SWE.embeddings(m)

#     # TikToken has two IDs for each of "cat" and "dog"
#     cat_vec = mean(W[:, id] for id in enc.encode("cat"))
#     dog_vec = mean(W[:, id] for id in enc.encode("dog"))
#     fish_vec = W[:, enc.encode("fish")[1]]

#     cos(u,v) = dot(u,v) / sqrt(dot(u,u)*dot(v,v) + eps())
#     @test cos(cat_vec, dog_vec) > cos(cat_vec, fish_vec) + 0.05
# end


# @testset "integration: test vs testing similarity" begin
#     corpus = vcat(fill("test testing", 200), fill("unrelated word", 200))
#     rng    = MersenneTwister(42)

#     m, enc = SWE.train!(corpus; epochs = 10, batch = 32,
#                         emb_dim = 16, lr = 1e-2, rng = rng)

#     W = SWE.embeddings(m)

#     test_vec     = mean(W[:, id] for id in enc.encode("test"))
#     testing_vec  = mean(W[:, id] for id in enc.encode("testing"))
#     unrelated_vec = W[:, enc.encode("unrelated")[1]]

#     cos(u,v) = dot(u,v) / sqrt(dot(u,u) * dot(v,v) + eps())

#     @test cos(test_vec, testing_vec) > cos(test_vec, unrelated_vec) + 0.00001
# end


# cos(u,v) = dot(u,v) / sqrt(dot(u,u) * dot(v,v) + eps())

# @testset "semantic clusters - animals vs fruits vs tech" begin
#     # corpus construction
#     animals  = ["cat", "dog", "mouse", "kitten", "puppy"]
#     fruits   = ["apple", "orange", "banana", "pear", "kiwi"]
#     tech     = ["computer", "laptop", "keyboard", "screen", "monitor"]

#     make_sentences(words, n) =
#         [join(rand(words, 3), " ") for _ in 1:n]

#     corpus = vcat( make_sentences(animals, 200),
#                    make_sentences(fruits,  200),
#                    make_sentences(tech,    200) )

#     #  train
#     rng = MersenneTwister(2024)
#     m, enc = SWE.train!(corpus;
#                         epochs   = 10,      # still quick
#                         batch    = 64,
#                         emb_dim  = 24,
#                         lr       = 1e-2,
#                         rng      = rng)

#     W = SWE.embeddings(m)

#     vec(word) = mean(W[:, id] for id in enc.encode(word))

#     animal_vec  = vec("cat")
#     fruit_vec   = vec("apple")
#     tech_vec    = vec("computer")

#     # checks 
#     @test cos(animal_vec, vec("dog"))       > cos(animal_vec, fruit_vec)
#     @test cos(fruit_vec,  vec("banana"))    > cos(fruit_vec,  tech_vec)
#     @test cos(tech_vec,   vec("laptop"))    > cos(tech_vec,   animal_vec)

#     # stronger: inside-cluster mean vs cross-cluster mean
#     mean_animal = mean(vec(w) for w in animals)
#     mean_fruit  = mean(vec(w) for w in fruits)
#     mean_tech   = mean(vec(w) for w in tech)

#     @test cos(mean_animal, mean_animal)  > cos(mean_animal, mean_fruit)
#     @test cos(mean_animal, mean_animal)  > cos(mean_animal, mean_tech)
#     @test cos(mean_fruit,  mean_fruit)   > cos(mean_fruit,  mean_tech)
# end


# @testset "encoder robustness (round-trip)" begin
#     for enc_name in ("cl100k_base", "gpt2")
#         enc = SWU.load_encoder(enc_name)

#         for input in (
#             "Regular English text",
#             "Text with numbers 123456",
#             raw"Text with symbols !@#$%^&*()",
#             raw"Mixed      whitespace   and\ttabs",
#             raw"URL: https://example.com/path?query=value",
#             "Code: function test( ) { return 42; }"
#         )
#             @test SWU.decode(SWU.encode(input, enc), enc) == input
#         end
#     end
# end


# paragraph1 = raw"""
#     It is a truth universally acknowledged, that a single man in
#     possession of a good fortune, must be in want of a wife.  However
#     little known the feelings or views of such a man may be on his first
#     entering a neighbourhood, this truth is so well fixed in the minds
#     of the surrounding families, that he is considered the rightful
#     property of some one or other of their daughters.
# """
# @testset "paragraph round-trip & training smoke" begin
#     # encode - decode
#     enc  = SWU.load_encoder()
#     ids  = SWU.encode(paragraph1, enc)
#     @test !isempty(ids)
#     @test SWU.decode(ids, enc) == paragraph1

#     # windowing works on multi-sentence stream
#     pairs = SWU.skipgram_pairs(ids, 3)    
#     @test length(pairs) == 2length(ids)*3 - 3*4   

#     #  one mini-batch SGNS lowers loss
#     pc, po = first.(pairs), last.(pairs)
#     m      = SWE.SkipGramModel(SWU.used_vocab_size(enc), 32)

#     freqs  = StatsBase.countmap(ids)
#     toks   = collect(keys(freqs))
#     tbl    = AliasTables.AliasTable(Float64.(values(freqs)).^0.75)

#     nc = repeat(pc, 2);  no = toks[rand(tbl, length(pc)*2)]

#     loss₁ = SWE.sg_loss(m, pc, po, nc, no)

#     gs = Zygote.gradient(() -> SWE.sg_loss(m, pc, po, nc, no),
#                          Flux.params(m))
#     Flux.Optimise.update!(Flux.Adam(1e-2), Flux.params(m), gs)

#     loss2 = SWE.sg_loss(m, pc, po, nc, no)
#     @test loss2 < loss₁
# end



# @testset "Alice in Wonderland - sub-word pipeline" begin
#     #Download book    
#     url  = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
#     path = tempname() * ".txt"

#     try
#         Downloads.download(url, path)
#     catch e
#         @info "Network unavailable - skipping Alice test" exception = e
#         return            # graceful skip
#     end

#     raw_txt = read(path, String)
#     @test occursin("ADVENTURES IN WONDERLAND", raw_txt)

#     # Encoder round-trip on 10 kB         
#     enc     = SWU.load_encoder()                # cl100k_base
#     snippet = first(raw_txt, 10_000)
#     ids     = SWU.encode(snippet, enc)
#     @test SWU.decode(ids, enc) == snippet
#     @test !isempty(ids)

#     # One-epoch Skip-Gram loop on 50 kB  
#     corpus = String.(split(first(raw_txt, 50_000), '\n'))
#     rng    = MersenneTwister(4242)

#     m, enc = SWE.train!(corpus;
#                         epochs   = 1,
#                         batch    = 2048,
#                         emb_dim  = 32,
#                         lr       = 5f-3,
#                         rng      = rng)

#     #  manual loss-drop check on the first 8 k pairs
#     full_ids = SWU.encode(first(raw_txt, 8_000), enc)
#     pairs    = SWU.skipgram_pairs(full_ids, 3)

#     pc, po = first.(pairs), last.(pairs)

#     freqs  = countmap(full_ids)
#     toks   = collect(keys(freqs))
#     tbl    = AliasTable(Float64.(values(freqs)).^0.75)

#     nc = repeat(pc, 2)
#     no = toks[rand(tbl, length(pc)*2)]

#     loss_before = SWE.sg_loss(m, pc, po, nc, no)

#     gs = Zygote.gradient(() -> SWE.sg_loss(m, pc, po, nc, no),
#                          Flux.params(m))
#     Flux.Optimise.update!(Flux.Adam(1e-2), Flux.params(m), gs)

#     loss_after  = SWE.sg_loss(m, pc, po, nc, no)
#     @test loss_after < loss_before

#     #(Optional) quick semantic sanity disabled by default   
#     #    Set ALICE_SEMANTIC=true to enable, may run ~5 s extra 
#     if get(ENV, "ALICE_SEMANTIC", "false") == "true"
#         W   = SWE.embeddings(m)
#         vec = w -> mean(W[:, id] for id in enc.encode(w))
#         alice  = vec("Alice")
#         rabbit = vec("Rabbit")
#         queen  = vec("Queen")
#         cos(u,v) = dot(u,v) / sqrt(dot(u,u)*dot(v,v) + eps())
#         @test cos(alice, rabbit) > cos(alice, queen)
#     end

#     # Clean-up     
#     rm(path; force = true)
#     @test !isfile(path)
# end


# @testset "helper sanity 1" begin
    
#     corpus = ["cat dog", "white rabbit", "rabbit hole", "cat ran"]    # 4 lines
#     rng    = MersenneTwister(42)

#     model, enc = SWE.train!(corpus;              # raw strings
#                             epochs   = 1,
#                             batch    = 16,
#                             emb_dim  = 16,
#                             rng      = rng)

#     # Helpers: wordvec / nearest_tokens / nearest_words
#     tok, _ = first(SWE.each_token(enc))   # first defined BPE token

#     @test SWE.is_word("Alice") == true

#     v1 = SWE.wordvec(model, enc, "cat")
#     v2 = SWE.wordvec(model, enc, "rabbit")
#     @test length(v1) == length(v2) == size(SWE.embeddings(model), 1)

#     nbrs = SWE.nearest_tokens(model, enc, tok; k = 3)
#     @test length(nbrs) == 3

#     nbrw = SWE.nearest_words(model, enc, "cat"; k = 5)
#     @test nbrw[1][1] == "cat"          # top neighbour is the word itself
# end


# @testset "helper sanity 2" begin
#     #tiny deterministic setup 
#     corpus = ["cat dog", "white rabbit", "rabbit hole", "cat ran"]
#     rng    = MersenneTwister(42)

#     model, enc = SWE.train!(corpus;    # raw Vector{String}
#                             epochs   = 1,
#                             batch    = 32,
#                             emb_dim  = 16,
#                             rng      = rng)

#     # each_token - no #undef gaps  
#     tokens = collect(SWE.each_token(enc))
#     @test !isempty(tokens)
#     @test all(isassigned(enc.vocab.list, id) for (_,id) in tokens)

#     # tokenvec - column length matches embedding dim 
#     first_tok, first_id = first(tokens)
#     col = SWE.tokenvec(model, first_id)
#     @test length(col) == size(SWE.embeddings(model), 1)

#     # wordvec - mean of its pieces 
#     v_cat    = SWE.wordvec(model, enc, "cat")
#     v_rabbit = SWE.wordvec(model, enc, "rabbit")
#     @test length(v_cat) == length(v_rabbit) == size(SWE.embeddings(model), 1)

#     #nearest_tokens - returns k items, first is query itself
#     nt = SWE.nearest_tokens(model, enc, first_tok; k = 4)
#     @test length(nt) == 4
#     @test nt[1][1] == first_tok       # self-similarity

#     # nearest_words - alphabetic only and k items
#     nw = SWE.nearest_words(model, enc, "cat"; k = 3)
#     @test length(nw) == 3
#     @test all(occursin(r"^[A-Za-z]+$", t[1]) for t in nw)
#     @test nw[1][1] == "cat"
# end