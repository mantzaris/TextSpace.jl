using Random
using UUIDs

using TextSpace
using TextSpace.Utils
using TextSpace.Utils.VocabularyCore
using TextSpace.Utils.CharTokenizer
using TextSpace.Utils.CharTokenizer: chars_to_ids, encode_char_batch
using TextSpace.Utils: learn_bpe, save_bpe, load_bpe, BPETokeniser
using TextSpace.Utils.TextVectorization
using TextSpace.Utils.TextVectorization: pad_sequences, one_hot, bow_counts, bow_matrix, tfidf_matrix, batch_iter
using TextSpace.Utils.VocabularyCore: Vocabulary
using TextSpace.Utils.VocabularyCore: build_vocabulary, 
        convert_tokens_to_ids, convert_ids_to_tokens,
        convert_batch_tokens_to_ids, save_vocabulary, load_vocabulary, ensure_unk!


include("char_tokenizer_tests.jl")
include("vocabulary_tests.jl")
include("text_vectorization_tests.jl")
include("learn_bpe_tests.jl")
include("load_bpe_tests.jl")