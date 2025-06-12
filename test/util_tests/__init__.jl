using Random
using UUIDs

using TextSpace
using TextSpace.utils
using TextSpace.utils.VocabularyCore
using TextSpace.utils.CharTokenizer
using TextSpace.utils.CharTokenizer: chars_to_ids, encode_char_batch
using TextSpace.utils: learn_bpe, save_bpe, load_bpe, BPETokeniser
using TextSpace.utils.TextVectorization
using TextSpace.utils.TextVectorization: pad_sequences, one_hot, bow_counts, bow_matrix, tfidf_matrix, batch_iter
using TextSpace.utils.VocabularyCore: Vocabulary
using TextSpace.utils.VocabularyCore: build_vocabulary, 
        convert_tokens_to_ids, convert_ids_to_tokens,
        convert_batch_tokens_to_ids, save_vocabulary, load_vocabulary, ensure_unk!


include("char_tokenizer_tests.jl")
include("vocabulary_tests.jl")
include("text_vectorization_tests.jl")
include("learn_bpe_tests.jl")
include("load_bpe_tests.jl")