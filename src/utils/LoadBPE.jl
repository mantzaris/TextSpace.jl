module LoadBPE
export BPETokeniser, load_bpe

import ...resource      # helper for bundled paths
using JSON3
using Base: basename

struct BPETokeniser
    merges :: Vector{Tuple{String,String}}
    vocab  :: Union{Dict{String,Int},Nothing}
end

const _ARTEFACTS = Dict(
    "gpt2_merges.txt"               => (; merges="gpt2_merges.txt",          vocab=nothing),
    "mGPT_61Lang1pnt9M_merges.txt"  => (; merges="mGPT_61Lang1pnt9M_merges.txt",
                                         vocab ="mGPT_61Lang1pnt9M_vocab.json"),
    "RoBERTa-base_merges.txt"       => (; merges="RoBERTa-base_merges.txt",
                                         vocab ="RoBERTa-base_vocab.json"),
    "Mistral-24B_32_768ctrl.json"   => (; merges="gpt2_merges.txt",          # fallback
                                         vocab ="Mistral-24B_32_768ctrl.json"),
    "XML-RoBERTa_100Lang.json"      => (; merges="gpt2_merges.txt",          # fallback
                                         vocab ="XML-RoBERTa_100Lang.json"),
)

#  helper to read GPT-2 style merges.txt 
_read_merges(txtfile) =
    Tuple.(split.(readlines(txtfile)))   # Vector{Tuple{String,String}}

#  public loader 
"""
    load_bpe(path_or_name) → BPETokeniser

* *Bundled name* (`"gpt2_merges.txt"`, `"mGPT_…"` …)  ⇒ loads from
  `src/resources/`.
* *Absolute/relative path*  ⇒ loads that file (and its sister if present).

No guessing: only the exact names in the table are recognised.
"""
function load_bpe(spec::AbstractString)
    # 1) resolve bundled name → full path(s)
    if haskey(_ARTEFACTS, basename(spec))
        meta   = _ARTEFACTS[basename(spec)]
        m_path = resource(meta.merges)
        v_path = meta.vocab === nothing ? nothing : resource(meta.vocab)
    else
        # user supplied a custom file path
        m_path = spec
        v_path = endswith(spec, "_merges.txt") ?
                 replace(spec, "_merges.txt" => "_vocab.json") :
                 nothing
    end

    # 2) read merges
    merges = endswith(m_path, ".txt") ?
             _read_merges(m_path) :
             Tuple{String,String}.(JSON3.read(read(m_path,String))["merges"])

    # 3) read vocab (if any)
    vocab  = nothing
    if v_path !== nothing && isfile(v_path)
        if endswith(v_path, ".json")
            raw = JSON3.read(read(v_path,String))
            vocab = isa(raw, Dict) ? Dict{String,Int}(raw) :
                    Dict{String,Int}((tok,i) for (i,(tok,_)) in
                                     enumerate(raw))   # ctrl.json / xlmr
        end
    end

    return BPETokeniser(merges, vocab)
end

end # module
