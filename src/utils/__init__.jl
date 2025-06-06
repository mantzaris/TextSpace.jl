module utils #lowercase to avoid clashes with Julia's Utils packages


include("Vocabulary.jl")
using .VocabularyCore
using .VocabularyCore: Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens


include("TextVectorization.jl")
using .TextVectorization: pad_sequences


include("CharTokeniser.jl")
using .CharTokeniser: chars_to_ids, encode_char_batch
export chars_to_ids, encode_char_batch


export Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens, 
            pad_sequences, testo


# subwords and BPE
include("LoadBPE.jl")
using .LoadBPE: BPETokeniser, load_bpe, bpe_encode_batch
export BPETokeniser, load_bpe, bpe_encode_batch

include("LearnBPE.jl")
using .LearnBPE: learn_bpe, save_bpe
export learn_bpe, save_bpe

end #module