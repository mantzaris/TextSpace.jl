using Test
using TextSpace
using TextSpace.Plumbing: tokenize, split_sentences, tokenize_batch
using TextSpace.Utils: Vocabulary, convert_tokens_to_ids
import TextSpace.Utils as Utils    # for load_bpe
import TextSpace.Pipeline
import TextSpace.Plumbing as Plumbing



#TODO: plumbing MORE
@testset "Plumbing basics" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
    @test split_sentences("Hi. Bye!") == ["Hi.", "Bye!"]
end


#TODO: Pipeline MORE
@testset "Pipeline - word level" begin
    sent = "Hug my dog!"
    tok, ids, voc = Pipeline.preprocess_word(sent)

    @test length(tok) == 1                     
    @test tok[1] == ["hug", "my", "dog", "!"] 
    @test length(ids) == length(tok)           
    @test voc isa TextSpace.Utils.Vocabulary
    @test all(isa.(ids[1], Int))          
end


@testset "Character Debug" begin
    sent = "AB"
    tok, ids, voc = Pipeline.preprocess_char(sent; char_eos = "</w>")
    
    println("Input: $sent")
    println("Tokens: $tok") 
    println("IDs[1]: $(ids[1])")
    println("IDs[1] length: $(length(ids[1]))")
    println("Vocab keys: $(sort(collect(keys(voc.token2id))))")
    
    # Check for expected characters
    for char in ["a", "b", "</w>"]
        println("'$char' in vocab: $(haskey(voc.token2id, char))")
    end
end

@testset "Character Processing Deep Debug" begin
    sent = "AB"
    
    println("\n=== DETAILED CHARACTER PROCESSING DEBUG ===")
    
    #Check _front_pass
    sentences, tokens = Pipeline._front_pass(sent; split_sentences=true, clean=true)
    println("After _front_pass:")
    println("  tokens: $tokens")
    
    #Manual character processing
    println("\nManual character processing:")
    for (i, sentence_tokens) in enumerate(tokens)
        sentence_str = join(sentence_tokens, "")
        println("  Joined: '$sentence_str'")
        
        chars = [string(ch) for ch in sentence_str]
        println("  Characters: $chars")
        
        push!(chars, "</w>")
        println("  With EOS: $chars")
        
        # This should give us 3 elements: ["a", "b", "</w>"]
    end
    
    #Test actual function
    char_vocab = Pipeline._build_char_vocab(tokens)
    result = Utils.encode_char_batch(tokens, char_vocab; eos="</w>")
    println("  encode_char_batch result: $result")
    println("  Result size: $(size(result))")
    
    # Full pipeline
    tok, ids, voc = Pipeline.preprocess_char(sent; char_eos = "</w>")
    println("  Final ids[1]: $(ids[1])")
    println("  Final length: $(length(ids[1]))")
end



@testset "Pipeline - char level" begin
    sent = "AB"
    tok, ids, voc = Pipeline.preprocess_char(sent; char_eos = "</w>")

    @test tok[1] == ["ab"]          
    @test length(ids[1]) == 3            
    @test voc.token2id["a"] isa Int   
    @test voc.token2id["b"] isa Int        
    @test voc.token2id["</w>"] isa Int     
    
    # Round-trip test (should work as-is)
    new_ids, same_voc = Pipeline.preprocess(sent;
                                            granularity=:char,
                                            char_vocab=voc,
                                            output=:ids)
    @test same_voc === voc
    @test new_ids == ids
end


# ---------------------------------------------------------------------
@testset "Pipeline - sub-word (bundled GPT-2)" begin
    text = "Hello world"
    tok, ids, bpe = Pipeline.preprocess_subword(text)  # default GPT-2 merges

    @test bpe isa TextSpace.Utils.LoadBPE.BPETokeniser
    @test !isempty(bpe.merges)
    @test length(ids) == 1 && !isempty(ids[1])
end


# ---------------------------------------------------------------------
@testset "Pipeline - learn + apply BPE on the fly" begin
    corpus = ["hug my dog", "hug my cat"]
    tok, ids, my_bpe = Pipeline.preprocess_subword_learn(corpus;
                                                         vocab_size=50,
                                                         min_frequency=1)

    @test my_bpe isa TextSpace.Utils.LearnBPE.BPETokeniser
    @test !isempty(my_bpe.merges)
    # Encoder works on unseen text with the same model
    new_ids, enc = Pipeline.preprocess("hug dog";
                                       granularity=:subword,
                                       subword_tokenizer=my_bpe,
                                       output=:ids)
    @test enc === my_bpe
    @test !isempty(new_ids[1])
end



# #---------------------------------
# @testset "Pipeline.jl Basic Tests" begin
    
#     @testset "Basic Text Processing" begin
#         text = "Hello world! How are you?"
        
#         # Test word-level preprocessing
#         tokens, ids, vocab = preprocess_word(text)
        
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
#         @test isa(vocab, Utils.Vocabulary)
#         @test length(tokens) == length(ids)
#         @test length(tokens) > 0
#         @test all(length(sent) > 0 for sent in tokens)
        
#         println("word preprocessing: $(length(tokens)) sentences, vocab size: $(length(vocab.id2token))")
#     end
    
#     @testset "Character-Level Processing" begin
#         text = "Hello world!"
        
#         # Test character-level preprocessing
#         tokens, ids, vocab = preprocess_char(text)
        
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
#         @test isa(vocab, Utils.Vocabulary)
#         @test length(tokens) == length(ids)
        
#         # Check that character vocab contains expected characters
#         @test "h" in vocab.id2token || "H" in vocab.id2token
#         @test "e" in vocab.id2token
#         @test "l" in vocab.id2token
#         @test "o" in vocab.id2token
        
#         println("✅ Character preprocessing: $(length(tokens)) sentences, char vocab size: $(length(vocab.id2token))")
#     end
    
#     @testset "Subword Processing" begin
#         text = "Hello world! How are you today?"
        
#         # Test subword preprocessing (using default GPT-2)
#         tokens, ids, bpe = preprocess_subword(text)
        
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
#         @test isa(bpe, Union{Utils.LoadBPE.BPETokeniser, Utils.LearnBPE.BPETokeniser})
#         @test length(tokens) == length(ids)
#         @test length(tokens) > 0
        
#         println("✅ Subword preprocessing: $(length(tokens)) sentences, $(length(bpe.merges)) merges")
#     end
    
#     @testset "BPE Learning" begin
#         corpus = [
#             "Hello world how are you",
#             "Hello there how is everything", 
#             "World peace is important",
#             "How are you doing today"
#         ]
        
#         # Test BPE learning and preprocessing
#         tokens, ids, learned_bpe = preprocess_subword_learn(corpus; vocab_size=100, min_frequency=2)
        
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
#         @test isa(learned_bpe, Utils.LearnBPE.BPETokeniser)
#         @test length(tokens) == length(corpus)
#         @test length(learned_bpe.merges) >= 0
#         @test learned_bpe.vocab !== nothing
        
#         println("✅ BPE learning: $(length(learned_bpe.merges)) merges learned, vocab size: $(length(learned_bpe.vocab))")
#     end
    
#     @testset "Output Formats" begin
#         text = "Simple test."
        
#         # Test different output formats
#         batch_result = preprocess(text; granularity=:word, output=:batch)
#         ids_result = preprocess(text; granularity=:word, output=:ids)
#         both_result = preprocess(text; granularity=:word, output=:both)
        
#         # batch format: (padded_matrix, encoder)
#         @test length(batch_result) == 2
#         @test isa(batch_result[1], Vector{Vector{Int}})
#         @test isa(batch_result[2], Utils.Vocabulary)
        
#         # ids format: (ids, encoder)
#         @test length(ids_result) == 2
#         @test isa(ids_result[1], Vector{Vector{Int}})
#         @test isa(ids_result[2], Utils.Vocabulary)
        
#         # both format: (tokens, ids, encoder)
#         @test length(both_result) == 3
#         @test isa(both_result[1], Vector{Vector{String}})
#         @test isa(both_result[2], Vector{Vector{Int}})
#         @test isa(both_result[3], Utils.Vocabulary)
        
#         println("output formats: batch, ids, both all working")
#     end
    
#     @testset "Text Processing Options" begin
#         text = "Hello    world!\n\nHow are you?"
        
#         # Test with different processing options
#         tokens1, _, _ = preprocess_word(text; split_sentences=true, clean=true)
#         tokens2, _, _ = preprocess_word(text; split_sentences=false, clean=true)
#         tokens3, _, _ = preprocess_word(text; split_sentences=true, clean=false)
        
#         # With sentence splitting should have more sentences
#         @test length(tokens1) >= length(tokens2)
        
#         # All should return valid token structures
#         @test isa(tokens1, Vector{Vector{String}})
#         @test isa(tokens2, Vector{Vector{String}})
#         @test isa(tokens3, Vector{Vector{String}})
        
#         println("✅ Text options: split_sentences and clean working")
#     end
    
#     @testset "Vocabulary Reuse" begin
#         text1 = "Hello world"
#         text2 = "Hello there"
        
#         # Build vocabulary from first text
#         _, _, vocab1 = preprocess_word(text1)
        
#         # Reuse vocabulary for second text
#         _, ids2, vocab2 = preprocess_word(text2; word_vocab=vocab1)
        
#         # Should be the same vocabulary object
#         @test vocab1 === vocab2
#         @test length(vocab1.id2token) == length(vocab2.id2token)
        
#         # Should handle known and unknown tokens
#         @test isa(ids2, Vector{Vector{Int}})
#         @test length(ids2) > 0
        
#         println("✅ Vocabulary reuse: same vocab object maintained")
#     end
    
#     @testset "Edge Cases" begin
#         # Test empty string
#         try
#             tokens, ids, vocab = preprocess_word("")
#             @test isa(tokens, Vector{Vector{String}})
#             @test isa(ids, Vector{Vector{Int}})
#             println("✅ Empty string handled gracefully")
#         catch e
#             println("⚠️  Empty string handling: $e")
#         end
        
#         # Test single word
#         tokens, ids, vocab = preprocess_word("hello")
#         @test length(tokens) >= 1
#         @test all(length(sent) > 0 for sent in tokens if length(sent) > 0)
        
#         # Test punctuation only
#         tokens, ids, vocab = preprocess_word("!@#$")
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
        
#         # Test Unicode
#         tokens, ids, vocab = preprocess_word("Hello 世界 café")
#         @test isa(tokens, Vector{Vector{String}})
#         @test isa(ids, Vector{Vector{Int}})
        
#         println("✅ Edge cases: single word, punctuation, Unicode all handled")
#     end
    
#     @testset "Granularity Consistency" begin
#         text = "Hello world! How are you?"
        
#         # Test all granularities return consistent structure
#         word_result = preprocess(text; granularity=:word, output=:both)
#         char_result = preprocess(text; granularity=:char, output=:both)
#         subword_result = preprocess(text; granularity=:subword, output=:both)
        
#         # All should return (tokens, ids, encoder) tuples
#         @test length(word_result) == 3
#         @test length(char_result) == 3
#         @test length(subword_result) == 3
        
#         # All should have same number of sentences
#         @test length(word_result[1]) == length(char_result[1])
#         @test length(word_result[1]) == length(subword_result[1])
        
#         # All should have corresponding IDs
#         @test length(word_result[1]) == length(word_result[2])
#         @test length(char_result[1]) == length(char_result[2])
#         @test length(subword_result[1]) == length(subword_result[2])
        
#         println("✅ Granularity consistency: word, char, subword all return consistent structures")
#     end
    
#     @testset "Error Handling" begin
#         text = "Hello world"
        
#         # Test invalid granularity
#         @test_throws ErrorException preprocess(text; granularity=:invalid)
        
#         # Test invalid output format
#         try
#             result = preprocess(text; granularity=:word, output=:invalid)
#             # If no error, check that it defaults to batch format
#             @test length(result) == 2
#         catch e
#             @test isa(e, ErrorException) || isa(e, MethodError)
#         end
        
#         println("✅ Error handling: invalid parameters properly caught")
#     end
# end



# @testset "Pipeline Core Functions" begin
    
#     @testset "Word Processing" begin
#         tokens, ids, vocab = preprocess_word("Hello world!")
        
#         @test length(tokens) > 0
#         @test length(ids) == length(tokens)
#         @test vocab.unk_id >= 0
#         @test "hello" in vocab.id2token || "Hello" in vocab.id2token
#     end
    
#     @testset "Character Processing" begin
#         tokens, ids, vocab = preprocess_char("Hi there")
        
#         @test length(tokens) > 0
#         @test length(ids) == length(tokens)
#         @test "h" in vocab.id2token || "H" in vocab.id2token
#         @test "i" in vocab.id2token
#     end
    
#     @testset "Subword Processing" begin
#         tokens, ids, bpe = preprocess_subword("Hello world")
        
#         @test length(tokens) > 0
#         @test length(ids) == length(tokens)
#         @test length(bpe.merges) >= 0
#     end
    
#     @testset "BPE Learning" begin
#         corpus = ["hello world", "hello there", "world peace"]
#         tokens, ids, bpe = preprocess_subword_learn(corpus; vocab_size=50)
        
#         @test length(tokens) == 3
#         @test length(ids) == 3
#         @test bpe.vocab !== nothing
#     end
    
#     @testset "Output Formats" begin
#         text = "Test sentence."
        
#         batch = preprocess(text; output=:batch)
#         ids_only = preprocess(text; output=:ids)
#         both = preprocess(text; output=:both)
        
#         @test length(batch) == 2
#         @test length(ids_only) == 2
#         @test length(both) == 3
#     end
# end


# @testset "Pipeline Smoke Tests" begin
    
#     @testset "Basic Functionality" begin
#         # Simple word test
#         result = preprocess_word("hello")
#         @test length(result) == 3  # (tokens, ids, vocab)
        
#         # Simple char test  
#         result = preprocess_char("hi")
#         @test length(result) == 3  # (tokens, ids, vocab)
        
#         # Simple subword test
#         result = preprocess_subword("hello")
#         @test length(result) == 3  # (tokens, ids, bpe)
        
#         println("✅ All preprocessing functions return correct tuple length")
#     end
    
#     @testset "Data Types" begin
#         tokens, ids, vocab = preprocess_word("test")
        
#         @test isa(tokens, Vector)
#         @test isa(ids, Vector)
#         @test hasfield(typeof(vocab), :id2token)
        
#         println("✅ All return types are correct")
#     end
    
#     @testset "Non-Empty Results" begin
#         tokens, ids, vocab = preprocess_word("hello world")
        
#         @test length(tokens) > 0
#         @test length(ids) > 0
#         @test length(vocab.id2token) > 1  # At least <unk> + some tokens
        
#         println("✅ All results are non-empty")
#     end
# end

# println("🚀 Smoke tests passed - Pipeline.jl is working!")

