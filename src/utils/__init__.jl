module utils


include("TextVectorization.jl")
using .TextVectorization: pad_sequences


include("Vocabulary.jl")
using .VocabularyCore
using .VocabularyCore: Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens


include("CharTokenizer.jl")
using .CharTokenizer: chars_to_ids, encode_char_batch


# subwords and BPE
include("LoadBPE.jl")
using .LoadBPE: BPETokeniser, load_bpe, bpe_encode_batch


include("LearnBPE.jl")
using .LearnBPE: learn_bpe, save_bpe


export BPETokeniser, load_bpe, bpe_encode_batch

export learn_bpe, save_bpe

export pad_sequences

export chars_to_ids, encode_char_batch

export Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens, 
            pad_sequences, testo


end #module