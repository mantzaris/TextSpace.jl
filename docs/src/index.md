# TextSpace.jl

```@contents

```


# TextSpace

Documentation for [TextSpace](https://github.com/mantzaris/TextSpace.jl).

```@index
```

# Character Embeddings Ex

```julia

julia> using Downloads, TextSpace, Random, LinearAlgebra

julia> CE = TextSpace.CharacterEmbeddings
TextSpace.CharacterEmbeddings

julia> alice_url = "https://www.gutenberg.org/files/11/11-0.txt";

julia> tmpfile  = joinpath(mktempdir(), "alice.txt");

julia> Downloads.download(alice_url, tmpfile);

julia> prep = preprocess_for_char_embeddings(tmpfile);

julia> ids, vocab = prep.char_ids, prep.vocabulary;

julia> length(ids)
116679

julia> length(vocab.id2token) |> println  # distinct characters
75

julia> model = CE.train!(ids, vocab;
                                objective = :skipgram,   # could be :cbow
                                emb_dim   = 32,
                                batch     = 512,
                                epochs    = 10,
                                k_neg     = 5,
                                rng       = MersenneTwister(123));
[ Info: epoch 1 finished
[ Info: epoch 2 finished
[ Info: epoch 3 finished
[ Info: epoch 4 finished
[ Info: epoch 5 finished
[ Info: epoch 6 finished
[ Info: epoch 7 finished
[ Info: epoch 8 finished
[ Info: epoch 9 finished
[ Info: epoch 10 finished

julia> E = CE.embeddings(model); #32 by |V| matrix

julia> cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps()); #helper fn for distances

julia> function nearest(ch; k = 5)
                  v = CE.vector(model, vocab, ch)
                  sims = [(t, cosine(v, col)) for (t,col) in zip(vocab.id2token, eachcol(E))]
                  sort!(sims; by = last, rev = true)[1:k]
              end;

julia> nearest("a")
5-element Vector{Tuple{String, Float64}}:
 ("a", 0.9999999999999998)
 ("e", 0.9278853952977884)
 ("t", 0.9144872079046347)
 ("s", 0.8980403459592878)
 ("i", 0.8953089503008627)

 julia> nearest("b")
5-element Vector{Tuple{String, Float64}}:
 ("b", 0.9999999999999997)
 ("m", 0.6669438471337544)
 ("p", 0.6291869695215444)
 ("k", 0.6161584847252951)
 ("v", 0.5560094024508363)

julia> nearest("z")
5-element Vector{Tuple{String, Float64}}:
 ("z", 0.999999871368756)
 ("E", 0.9223647657701302)
 ("B", 0.9201309209419235)
 ("(", 0.9165986726089074)
 ("P", 0.9129365161033225)

julia> any(isnan, E)
false

julia> sentence = "Knowledge is knowing a tomato is a fruit. Wisdom is not putting a tomato in a fruit salad."
"Knowledge is knowing a tomato is a fruit. Wisdom is not putting a tomato in a fruit salad."

julia> PP = TextSpace.Preprocessing

julia> sid  = PP.chars_to_ids(PP.tokenize_char(sentence), vocab);

julia> using Statistics: mean

julia> svec = mean(E[:, sid]; dims = 2) |> vec; #32-dim sentence vector

julia> println(svec)
Float32[0.0033476388, 0.0050592553, 0.0016037462, 0.0037245215, 0.009814645, -0.0034747466, -0.0066566155, 0.16300926, -0.0153919635, -0.27329186, 0.0077496497, -0.00096031267, -0.008754102, 0.004796173, -0.028046768, -0.0047062347, 0.02946253, -0.58761173, 0.008234083, 0.13165528, 0.0032494946, -0.056638658, -0.049376927, 0.012857347, 0.001997499, 0.0039835414, 0.07918961, -0.15194848, 0.00029468472, 0.03266388, -0.13388161, -0.053712953]

julia> norm(svec)
0.7233994f0

julia> using TSne, Plots #visually inspect the embedding space of the characters relative to each other

julia> Y = tsne(permutedims(E),2,50);   # |V| × 2

julia> scatter(Y[:,1], Y[:,2], text = vocab.id2token,
               markersize = 0, legend = false,
               title = "t-SNE of Alice-in-Wonderland character embeddings")

julia> sentences = split(read(tmpfile,String), '\n'; keepempty=false);

julia> length(sentences)
2496

julia> svectors = [mean(E[:, PP.chars_to_ids(PP.tokenize_char(s), vocab)];
                       dims=2) |> vec for s in sentences];

julia> length(svectors)
2496

julia> function nearest_sentence(char; top=3)
           v   = CE.vector(model, vocab, char)
           idx = sortperm([cos(v,s) for s in svectors]; rev=true)[1:top]
           sentences[idx]
       end

#which sentences in the book are most like "e"
julia> nearest_sentence("e")
5-element Vector{Tuple{SubString{String}, Float64}}:
 ("to have the experiment tried.", 0.9768103076061682)
 ("the teapot.", 0.97597783399397)
 ("getting somewhere near the centre of the earth. Let me see: that would", 0.9756655564052366)
 ("she had someone to listen to her. The Cat seemed to think that there", 0.9736785506348397)
 ("The Knave of Hearts, he stole those tarts,", 0.973097050960944)

#which sentences in the book are most like "m"
julia> nearest_sentence("m")
5-element Vector{Tuple{SubString{String}, Float64}}:
 ("picked up.”", 0.7860741345586871)
 ("of yours.’”", 0.7645134975129732)
 ("from?”", 0.756078507735434)
 ("elbow.”", 0.7140707633117682)
 ("moved.", 0.7113644307713963)

#which sentences in the book are most like '?'
julia> nearest_sentence('?')
5-element Vector{Tuple{SubString{String}, Float64}}:
 ("CHAPTER III.", 0.7113396838385628)
 ("CHAPTER II.", 0.6993569841941243)
 ("CHAPTER I.", 0.6816405089759888)
 (" CHAPTER VII.   A Mad Tea-Party", 0.6041056519919575)
 ("CHAPTER VIII.", 0.5875109533598397)

```


# Word Embeddings Ex

```julia
using Downloads, TextSpace, Random, LinearAlgebra, Statistics

const WE = TextSpace.WordEmbeddings

const PP = TextSpace.Preprocessing  # helpers for cleaning / tokenising

alice_url = "https://www.gutenberg.org/files/11/11-0.txt"

tmpfile = joinpath(mktempdir(), "alice_words.txt")

Downloads.download(alice_url, tmpfile)

# `preprocess_for_word_embeddings` returns a vector‐of‐sentences, each a vector of IDs
prep = PP.preprocess_for_word_embeddings(tmpfile; from_file = true,
                                                min_count = 5) # drop rare words

sent_ids = prep.word_ids #Vector{Vector{Int}}

vocab = prep.vocabulary

flat_ids = vcat(sent_ids...) # 1-D corpus for Skip-Gram/CBOW training

@show length(flat_ids) # approximately 25 000 tokens after trimming

@show length(vocab.id2token) # unique words remaining

model = WE.train!(flat_ids, vocab;
                  objective = :skipgram,
                  emb_dim   = 64, #embedding dimension
                  batch     = 1024,
                  epochs    = 5,
                  k_neg     = 5,
                  rng       = MersenneTwister(123))

E = WE.embeddings(model) #embedding is a 64 by |V| matrix (CPU Array)

cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps())

function nearest(tok; k = 5)
    v  = WE.vector(model, vocab, tok)
    sims = [(t, cosine(v, col))
            for (t,col) in zip(vocab.id2token, eachcol(E))]
    sort!(sims; by = last, rev = true)[1:k]
end

nearest("alice")      # closest words to 'alice'
5-element Vector{Tuple{String, Float64}}:
 ("alice", 1.0)
 ("said", 0.7302538750833167)
 ("to", 0.6439354006261235)
 ("it", 0.6217598049069093)
 ("she", 0.6119838841651302)

nearest("rabbit")
5-element Vector{Tuple{String, Float64}}:
 ("rabbit", 1.0000000896666614)
 ("white", 0.818244167158882)
 ("saw", 0.37895408892625565)
 ("hastily", 0.3704275813369187)
 ("three", 0.36109669318023857)

julia> nearest("cat")
5-element Vector{Tuple{String, Float64}}:
 ("cat", 1.0000000764600523)
 ("right", 0.42743823270147446)
 ("mad", 0.42369448323556974)
 ("hatter", 0.42302991414277147)
 ("caterpillar", 0.3938569272541852)

julia> function encode_sentence(s::AbstractString)
           ids = PP.tokens_to_ids(split(lowercase(s)), vocab; add_new = false)
           isempty(ids) && return zeros(Float32, size(E,1))   # all OOV -> zero vec
           return mean(E[:, ids]; dims = 2) |> vec
       end

julia> sentences = split(read(tmpfile, String), '\n'; keepempty = false)
2496-element Vector{SubString{String}}:
 "*** START OF THE PROJECT GUTENBERG EBOOK 11 ***"
 "[Illustration]"
...

julia> svectors  = encode_sentence.(sentences)

# Which sentence is 'closest to 'wonderland'?
julia> target = WE.vector(model, vocab, "cat")

julia> idx = argmax(cosine.(Ref(target), svectors))
2393

julia> @show sentences[idx]
sentences[idx] = "    For this must ever be"
"    For this must ever be"

julia> using TSne, Plots

julia> proj = tsne(permutedims(E), 2, 50) #|V| by 2

julia> scatter(proj[:,1], proj[:,2];
               text        = vocab.id2token,
               markersize  = 0,
               legend      = false,
               title       = "t-SNE of Alice-in-Wonderland **word** embeddings")

julia> scatter(proj[1:100,1], proj[1:100,2];
               text        = vocab.id2token,
               markersize  = 0,
               legend      = false,
               title       = "t-SNE of Alice-in-Wonderland **word** embeddings: 100")

```




# Sub-word (BPE) Embeddings Example


```julia
julia> using Downloads, TextSpace, Random, LinearAlgebra, Statistics

julia> const SWE = TextSpace.SubwordEmbeddings   
TextSpace.SubwordEmbeddings

#small corpus for using Pretrained BPE
julia> alice_url = "https://www.gutenberg.org/files/11/11-0.txt"
"https://www.gutenberg.org/files/11/11-0.txt"

julia> tmpfile   = joinpath(mktempdir(), "alice_sub.txt")
"/tmp/jl_Kf8vXd/alice_sub.txt"

julia> Downloads.download(alice_url, tmpfile)
"/tmp/jl_Kf8vXd/alice_sub.txt"

julia> lines = String.(split(read(tmpfile, String), '\n'; keepempty = false))
2496-element Vector{String}:
 "*** START OF THE PROJECT GUTENBERG EBOOK 11 ***"
...

julia> model, enc = SWE.train!(lines; # now Vector{String}
                               encoder_name = "cl100k_base",
                               epochs       = 10, #really short training
                               batch        = 2048,
                               emb_dim      = 64,
                               radius       = 5,
                               rng          = MersenneTwister(123))

julia> E = SWE.embeddings(model)      # 64 by |V| Float32 matrix

julia> cosine(a,b) = dot(a,b) / (norm(a)*norm(b) + eps()) #helper function

julia> function wordvec(word) #helper function
           ids = enc.encode(word)                 # BytePairEncoding API
           mean(E[:, ids]; dims = 2) |> vec
       end

julia> function nearest(word; k = 5) #helper function 
    v = wordvec(word)
    sims = [(tok, cosine(v, E[:, id])) for (tok,id) in SWE.each_token(enc)]
    sort!(sims; by = last, rev = true)[1:k]
end

julia> nearest("alice")
5-element Vector{Tuple{String, Float64}}:
 ("alice", 0.9999999999999575)
 (" hade", 0.5131048298879446)
 ("expiration", 0.48866187822890006)
 (" problematic", 0.48084224146381127)
 ("\trc", 0.4801181579918874)

julia> nearest("rabbit")
5-element Vector{Tuple{String, Float64}}:
 ("rabbit", 0.9999999170183584)
 ("('-',", 0.5016094944372554)
 (".properties", 0.47971542221121205)
 (" ACT", 0.469640182630054)
 (".userData", 0.4604995969573127)

julia> nearest("cat")
5-element Vector{Tuple{String, Float64}}:
 ("cat", 0.999999999999999)
 (" evening", 0.9248988543803454)
 ("William", 0.9212584156353408)
 ("ars", 0.9024920623025695)
 (" Who", 0.8968998649983356)

#when training on a small corpus many of the words are unseen
julia> seen_ids = unique(vcat(enc.encode.(lines)...))   # IDs that appear
3913-element Vector{Int64}:

julia> function nearest_word_seen(word; k=5)
           v = wordvec(word)
           sims = [(enc.vocab.list[id], cosine(v, E[:, id]))
                   for id in seen_ids if is_word(enc.vocab.list[id])]
           sort!(sims; by = last, rev = true)[1:k]
       end

julia> nearest_word_seen("alice")
5-element Vector{Tuple{String, Float64}}:
 ("thing", 0.32003743130824647)
 ("the", 0.3015098276822004)
 ("an", 0.28568369350261397)
 ("common", 0.25842654965108713)
 ("als", 0.25803399997313126)

julia> nearest_word_seen("rabbit")
5-element Vector{Tuple{String, Float64}}:
 ("What", 0.23954452836137805)
 ("Why", 0.22050971152471946)
 ("or", 0.19594570781380372)
 ("Let", 0.18518634659958685)
 ("think", 0.18162724520582288)

julia> nearest_word_seen("cat")
5-element Vector{Tuple{String, Float64}}:
 ("cat", 0.9999999172449788)
 ("yster", 0.807033835912888)
 ("est", 0.7975417191124465)
 ("ro", 0.7965722728462085)
 ("adder", 0.7905969983375803)

julia> sentence = "The quick brown fox jumps over the lazy dog."
"The quick brown fox jumps over the lazy dog."

julia> svec = mean(E[:, enc.encode(sentence)]; dims = 2) |> vec
64-element Vector{Float32}:








```












## Functions

```@autodocs
Modules = [TextSpace]
Private = false
Order = [:function]
```