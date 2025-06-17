module Pipeline

import ..TextSpace: resource
import ..Plumbing
using ..Plumbing: clean_text
import ..Utils as Utils

const  LBPE = Utils.LearnBPE
const  BPE  = Utils.LoadBPE

export preprocess,
       preprocess_word,
       preprocess_char,
       preprocess_subword,
       preprocess_subword_learn


# helper: front half  (clean -> split -> tokenise)
"""
    _front_pass(text;
                split_sentences       = true,
                clean                 = true,
                do_remove_zero_width  = false,
                clean_kw...) → (sentences, tokens)

1. optionally strip zero-width code-points
2. sentence segmentation  
3. per-sentence cleaning (broadcasts every `clean_kw...`)  
4. word tokenisation **keeping punctuation** (`strip_punctuation = false`)
"""
function _front_pass(text::AbstractString;
                     split_sentences::Bool      = true,
                     clean::Bool                = true,
                     do_remove_zero_width::Bool = false,
                     clean_kw...)

    # 0  ultra-fast sweep for ZWJ / ZWSP 
    do_remove_zero_width && (text = Plumbing.strip_zero_width(text))

    # 1  sentence boundaries
    sents = split_sentences ? Plumbing.split_sentences(text) : [text]

    # 2 cleaning (broadcast)
    if clean
        sents = Plumbing.clean_text.(sents;
                                     clean_kw...,
                                     collapse_spaces = !do_remove_zero_width)
    end

    # 3 word/punct tokenisation  (punctuation kept)
    toks = Plumbing.tokenize_batch(sents; strip_punctuation = false)

    return sents, toks
end

_prepare_vocab(v, toks) = v === nothing ? _build_vocab(toks) : v

_build_vocab(tok_batch) = begin
    voc = Utils.Vocabulary()
    for t in tok_batch
        Utils.convert_tokens_to_ids(t, voc; add_new = true, update_counts = false)
    end
    voc
end

"""
    _get_bpe(spec) -> Utils.LoadBPE.BPETokeniser

* `spec isa Utils.LoadBPE.BPETokeniser` → returned unchanged  
* `spec isa Symbol  | String`           → passed to `Utils.load_bpe`  
  (bundled keyword or file path)  
* `spec === nothing`                    → default to bundled GPT-2 merges
"""
function _get_bpe(spec)
    if spec isa BPE.BPETokeniser || spec isa LBPE.BPETokeniser
        return spec
    elseif spec === nothing
        return Utils.load_bpe(resource("gpt2_merges.txt"))
    else
        return Utils.load_bpe(String(spec))
    end
end


function preprocess(text::AbstractString; 
                    granularity::Symbol = :word,
                    word_vocab = nothing,
                    subword_tokenizer = nothing,
                    char_eos = "</w>",
                    split_sentences::Bool = true,
                    char_vocab = nothing,             # NEW: explicit char vocab
                    output::Symbol = :batch, #???
                    clean::Bool     = true,
                    do_remove_zero_width::Bool = false,
                    clean_kw...)

    _, tokens = _front_pass(text;
                        split_sentences,
                        clean,
                        do_remove_zero_width,
                        clean_kw...)

    encoder_ref = nothing
    pad_value   = 0
    ids         = Vector{Vector{Int}}()

    if granularity === :word
        vocab        = _prepare_vocab(word_vocab, tokens)
        ids          = [Utils.convert_tokens_to_ids(t, vocab) for t in tokens]
        encoder_ref  = vocab
        pad_value    = vocab.unk_id

    elseif granularity === :subword
        bpe          = _get_bpe(subword_tokenizer)
        ids          = Utils.bpe_encode_batch(bpe, tokens)
        encoder_ref  = bpe
        pad_value    = 0

    elseif granularity === :char
        vocab = char_vocab === nothing ? _build_char_vocab(tokens) : char_vocab
        padded_matrix = Utils.encode_char_batch(tokens, vocab; eos = char_eos)
        
        ids = [padded_matrix[:, i] for i in 1:size(padded_matrix, 2)] #just works...
        
        encoder_ref = vocab
        pad_value = vocab.unk_id


    else
        error("granularity must be :word, :subword or :char (got $granularity)")
    end

    return output === :ids    ? (ids,  encoder_ref) :
           output === :both   ? (tokens, ids, encoder_ref) :
                                (Utils.pad_sequences(ids; pad_value = pad_value),
                                 encoder_ref)
end


preprocess_word(text; kw...)   =
    preprocess(text; granularity = :word,   output = :both, kw...)

preprocess_char(text; kw...)   =
    preprocess(text; granularity = :char,   output = :both, kw...)

preprocess_subword(text; subtok = nothing, kw...) =
    preprocess(text; granularity = :subword,
               subword_tokenizer = subtok,  output = :both, kw...)

"""
preprocess_subword_learn(corpus; kwargs…) → (tokens, ids, bpe)
Convenience wrapper: first *learn* a BPE on `corpus` (vector of docs or string),
then immediately encode the same corpus with that freshly-trained model.
Returns `(tokens, ids, my_bpe)`.
"""
function preprocess_subword_learn(corpus::Vector{String}; 
                                  vocab_size = 10_000,
                                  min_frequency = 2,
                                  special_tokens = ["<unk>"],
                                  output::Symbol = :both,
                                  kw...)
    bpe = LBPE.learn_bpe(corpus;
                         vocab_size = vocab_size,
                         min_frequency = min_frequency,
                         special_tokens = special_tokens)

    combined_text = join(corpus, " ")

    return preprocess(combined_text; granularity=:subword, 
                      subword_tokenizer=bpe, output=output, kw...)
end





_build_char_vocab(tokens) = begin
    voc = Utils.Vocabulary()
    
    # Process characters from words
    for sent in tokens, word in sent, ch in String(word)
        Utils.convert_tokens_to_ids([string(ch)], voc; add_new=true, update_counts=false)
    end
    
    # include EOS token in vocabulary
    Utils.convert_tokens_to_ids(["</w>"], voc; add_new=true, update_counts=false)
    
    voc
end




end  # module Pipeline



# txt = "Hug my dog. Hug my cat."

# # word route - build vocab automatically:
# wp = word_pre(txt)
# wp.ids        # → Matrix{Int}
# wp.encoder    # → Vocabulary to serialise

# # sub-word with built-in GPT-2 merges:
# sp = subword_pre(txt; tokenizer = :gpt2, output = :both)
# sp.tokens     # original sub-tokens
# sp.ids        # ids ready for the embedding layer

# # learn a brand-new BPE on a corpus:
# lp = subword_learn(["alpha beta beta", "beta gamma"])
# LBPE.save_bpe(lp.encoder, "my_corpus_bpe.json")
