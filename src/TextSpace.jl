#=
# Common and Useful Embedding Types in NLP and LLMs

## 1. Text-Based Embeddings
- **Character Embeddings** – Capture morphological patterns (e.g., CharLSTM, FastText).
- **Subword Embeddings** – Handle rare words using BPE, SentencePiece (e.g., WordPiece, Unigram).
- **Word Embeddings** – Fixed-length word vectors (e.g., Word2Vec, GloVe, FastText).
- **Phrase Embeddings** – Represent multi-word expressions (e.g., "New York", "deep learning").
- **Sentence Embeddings** – Capture sentence-level meaning (e.g., SBERT, Universal Sentence Encoder).
- **Paragraph Embeddings** – Represent longer text spans (e.g., Doc2Vec).
- **Document Embeddings** – Whole document representation (aggregated or learned).

## 2. Model-Specific Embeddings in LLMs
- **Token Embeddings** – Core subword embeddings used by transformers.
- **Positional Embeddings** – Encode token order (Absolute, Relative, or RoPE).
- **Contextualized Embeddings** – Dynamic word representations (e.g., BERT, GPT).
- **Segment Embeddings** – Distinguish sentence pairs in models like BERT.
- **Task-Specific Embeddings** – Used in instruction-tuned models (e.g., FLAN-T5, ChatGPT).

## 3. Multimodal and Specialized Embeddings
- **Knowledge Graph Embeddings** – Encode entities & relations (e.g., TransE, ComplEx, RotatE).
- **Vision-Language Embeddings** – Align images and text (e.g., CLIP, Flamingo, GPT-4V).
- **Audio Embeddings** – Speech representation (e.g., Whisper, Wav2Vec).
- **Hybrid Embeddings** – Combine multiple embeddings (e.g., Flair: character + word + contextual).
=#

module TextSpace

using Unicode

directories = ["preprocessing","embeddings","utils"]

for d in directories
    for f in sort(readdir(joinpath(@__DIR__, d)))
        if endswith(f, ".jl")
            include(joinpath(@__DIR__, d, f))
        end
    end
end

export clean_text, tokenize, build_vocabulary



end #END MODULE