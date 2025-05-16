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

## Functions

```@autodocs
Modules = [TextSpace]
Private = false
Order = [:function]
```