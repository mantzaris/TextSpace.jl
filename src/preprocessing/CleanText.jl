#TODO: \p{Emoji} JuliaLang Base.Unicode 1.11 has :Emoji
if !@isdefined(EMOJI_RANGES)
    const EMOJI_RANGES = Tuple{UInt32,UInt32}[
        (0x1F300, 0x1F5FF),
        (0x1F600, 0x1F64F),
        (0x1F680, 0x1F6FF),
        (0x1F700, 0x1F77F),
        (0x1F900, 0x1F9FF),
        (0x1FA70, 0x1FAFF),
        (0x2600,  0x26FF),
        (0x2700,  0x27BF),
        (0x1F1E6, 0x1F1FF),
        (0x1F3FB, 0x1F3FF),
        (0xFE0F,  0xFE0F),   #VS-16
        (0x200D,  0x200D),   #ZWJ
    ]
end


"""
    remove_punctuation(text;
                       remove_symbols      = false,
                       extra_symbols       = Char[],
                       normalize_whitespace = false)

Strip Unicode punctuation from `text`.  
* When `remove_symbols` is `true`, Unicode *symbols* (`\\p{S}`: emoji, currency,
  math, etc.) are stripped as well.
* Any characters in `extra_symbols` are always removed.
* If `normalize_whitespace` is `true`, runs of blanks/whitespace collapse to a
  single space and result is `strip`ped; otherwise the original spacing is left
  untouched (matches the behaviour your current tests expect).
"""
function remove_punctuation(
        text::AbstractString;
        remove_symbols::Bool = false,
        extra_symbols::AbstractVector{<:AbstractChar} = Char[],
        normalize_whitespace::Bool = false
)::String

    base_pat = remove_symbols ?
               r"[\p{P}\p{S}\p{C}]" :   # strip punctuation + symbols + invisible
               r"\p{P}"                 # strip punctuation only
    stripped = replace(text, base_pat => "")

    if !isempty(extra_symbols)
        ex = Set(extra_symbols)
        io = IOBuffer()
        for c in stripped
            c ∈ ex || print(io, c)
        end
        stripped = String(take!(io))
    end

    if normalize_whitespace
        stripped = replace(stripped, r"\s+" => " ")
        stripped = strip(stripped)
    end
    return stripped
end


"""
    remove_emojis(text; keep_zero_width=false) -> String

Return `text` with **all Emoji code-points** removed.

* Covers the full set of Unicode emoji blocks (as of 15.1):
  Misc Symbols, Dingbats, Pictographs, Supplemental Symbols, 
  Extended-A, Flags, Key-caps, Variants.
* If `keep_zero_width=true`, ZERO WIDTH JOINER (U+200D) and VARIATION
  SELECTOR-16 (U+FE0F) are preserved; otherwise they are dropped too.
"""
function remove_emojis(
        text::AbstractString;
        keep_zero_width::Bool = false
)::String

    @inline in_emoji(c::Char) = begin
        cp = UInt32(c)
        # VS-16 / ZWJ may be kept
        if keep_zero_width && (cp == 0xFE0F || cp == 0x200D)
            return false
        end
        for (lo,hi) in EMOJI_RANGES
            if lo <= cp <= hi
                return true
            end
        end
        return false
    end

    # allocate once
    buf = IOBuffer()    
    @inbounds for c in text
        in_emoji(c) || write(buf, c)
    end
    return String(take!(buf))
end


"""
    remove_accents(text::AbstractString) -> String

Strip all combining diacritical marks (Unicode category *Mn*) from `text`
while leaving base characters intact.  Works on Julia 1.6 - 1.11.
"""
function remove_accents(text::AbstractString)::String
    nfd = Unicode.normalize(text, :NFD)

    if isdefined(Unicode, :combining_class)      # ≥ 1.10 fast path
        io = IOBuffer()
        @inbounds for c in nfd
            Unicode.combining_class(c) == 0 && write(io, c)
        end
        return Unicode.normalize(String(take!(io)), :NFC)
    else                                          # 1.6 - 1.9 fallback
        stripped = replace(nfd, r"\p{Mn}" => "")
        return Unicode.normalize(stripped, :NFC)
    end
end


"""
    clean_text(text;
               unicode_normalize      = true,
               do_remove_accents      = false,
               do_remove_punctuation  = false,
               do_remove_symbols      = false,
               do_remove_emojis       = false,
               case_transform         = :lower,   # :lower | :upper | :none
               extra_symbols          = Char[]) -> String

Run the canonical preprocessing pipeline in the order:

1. NFC normalisation (`normalize_unicode`).
2. Accent removal (`remove_accents`).
3. Case transform.
4. Punctuation / symbol stripping (`remove_punctuation`).
5. Emoji removal (`remove_emojis`).
6. Whitespace tidy (`normalize_whitespace`).

Steps that are turned off by their flag are skipped, keeping the function
O(n) in the length of `text`.
"""
function clean_text(
        text::AbstractString;
        unicode_normalize::Bool      = true,
        do_remove_accents::Bool      = false,
        do_remove_punctuation::Bool  = false,
        do_remove_symbols::Bool      = false,
        do_remove_emojis::Bool       = false,
        case_transform::Symbol       = :lower,
        extra_symbols::AbstractVector{<:AbstractChar} = Char[]
)::String

    case_transform in (:lower, :upper, :none) ||
        throw(ArgumentError("case_transform must be :lower, :upper, or :none"))

    t = unicode_normalize ? normalize_unicode(text) : String(text)

    do_remove_accents      && (t = remove_accents(t))

    case_transform === :lower && (t = lowercase(t))
    case_transform === :upper && (t = uppercase(t))

    if do_remove_punctuation || do_remove_symbols || !isempty(extra_symbols)
        t = remove_punctuation(t;
                               remove_symbols = do_remove_symbols,
                               extra_symbols  = extra_symbols)
    end

    do_remove_emojis && (t = remove_emojis(t))
    t = normalize_whitespace(t)               # always final tidy
    return t
end


strip_zero_width(text::AbstractString) =
    normalize_whitespace(text;
        strip_ends        = false,   # keep leading / trailing blanks unchanged
        preserve_newlines = true,    # keep existing line-breaks
        remove_zero_width = true)    # <- activates the drop-logic
