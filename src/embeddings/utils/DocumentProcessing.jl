# """
#     process_document(text_or_sents, tok;
#                      vocab          = nothing,
#                      token_mode     = :auto,     # :auto | :word | :subword
#                      sentence_split = true,
#                      clean_text     = true,
#                      mode           = :batch,    # :batch | :ids | :tokens | :sentences
#                      token_kwargs...)            # forwarded to `tokenize(_batch)`

# High-level preprocessing for either *word-level* or *sub-word* pipelines.

# `token_mode` rules  
# ------------------
# | `:auto`     | word-level if `vocab` is supplied, otherwise sub-word. *(Previous behaviour.)* |
# | `:word`     | **forces** word-level branch. `vocab` **must** be a `Vocabulary`. |
# | `:subword`  | **forces** sub-word branch. `tok` must implement `encode` & `pad_id`; `vocab` must be `nothing`. |

# Return values
# -------------
# Same as before (`:sentences`, `:tokens`, `:ids`, `:batch`).

# **Only word-level is fully implemented below.**  The `TODO` shows exactly where
# to insert your sub-word logic later.
# """
# function process_document(text_or_sents,
#                           tok;
#                           vocab::Union{Nothing,Vocabulary} = nothing,
#                           token_mode::Symbol = :auto,
#                           sentence_split::Bool = true,
#                           clean_text::Bool     = true,
#                           mode::Symbol         = :batch,
#                           token_kwargs...)

#     mode ∈ (:batch, :ids, :tokens, :sentences) ||
#         throw(ArgumentError("mode must be :batch, :ids, :tokens, or :sentences; got $mode"))

#     # ───────────────────── Decide word vs. sub-word ──────────────────────
#     branch =
#         token_mode === :word     ? :word     :
#         token_mode === :subword  ? :subword  :
#         vocab === nothing        ? :subword  : :word         # :auto fallback

#     branch === :word && vocab === nothing &&
#         throw(ArgumentError("token_mode=:word requires a Vocabulary."))

#     branch === :subword && vocab !== nothing &&
#         throw(ArgumentError("token_mode=:subword is incompatible with a Vocabulary."))

#     # ───────────────────── 1. Sentences (shared) ─────────────────────────
#     sentences =
#         if isa(text_or_sents, AbstractString)
#             paras = split_paragraphs(text_or_sents; unwrap = true) |>
#                     p -> filter_paragraphs(p; min_chars = 25)

#             raw = sentence_split ? vcat(split_sentences.(paras)...) : String.(paras)
#             clean_text ? clean_text.(raw) : raw
#         else
#             s = String.(text_or_sents)
#             clean_text ? clean_text.(s) : s
#         end

#     mode === :sentences && return sentences

#     # ───────────────────── 2. Tokens (shared) ────────────────────────────
#     token_seqs = tokenize_batch(sentences; token_kwargs...)
#     mode === :tokens && return token_seqs

#     # ───────────────────── 3. IDs  (branch-specific) ─────────────────────
#     if branch === :word
#         id_seqs = [convert_tokens_to_ids(ts, vocab) for ts in token_seqs]
#         pad_val = vocab.unk_id

#     else  # branch == :subword  (implement later)
#         # -- TODO ---------------------------------------------------------
#         # id_seqs = [encode(tok, join(ts, ' ');
#         #                    add_special_tokens = true) for ts in token_seqs]
#         # pad_val = tok.pad_id
#         throw(ErrorException("sub-word branch not implemented yet."))
#     end
#     mode === :ids && return id_seqs

#     # ───────────────────── 4. Padded batch (shared) ──────────────────────
#     return pad_sequences(id_seqs; pad_value = pad_val)
# end
