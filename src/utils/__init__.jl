module utils #lowercase to avoid clashes with Julia's Utils packages


include("Vocabulary.jl")
using .VocabularyCore
using .VocabularyCore: Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens


include("TextVectorization.jl")
using .TextVectorization: pad_sequences


export Vocabulary, convert_tokens_to_ids, convert_ids_to_tokens, 
            pad_sequences, testo


# subwords and BPE
include("LoadBPE.jl")
using .LoadBPE: BPETokeniser, load_bpe
export BPETokeniser, load_bpe

end #module