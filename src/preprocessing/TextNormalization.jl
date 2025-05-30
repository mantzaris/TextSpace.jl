#(future) text normalization (add: accent stripping, stemming, lemmatization)

"""
    normalize_unicode(text; form = :NFC) -> String

Return *text* in the requested Unicode normal form (`:NFC`, `:NFD`,
`:NFKC`, `:NFKD`).  Throws `ArgumentError` if *form* is not one of those
four symbols.
"""
function normalize_unicode(text::AbstractString; form::Symbol = :NFC)
    form in (:NFC, :NFD, :NFKC, :NFKD) ||
        throw(ArgumentError("Unsupported normalization form: $form"))

    return Unicode.normalize(text, form)
end


"""
    normalize_whitespace(text;
                         strip_ends        = true,
                         preserve_newlines = false,
                         remove_zero_width = false) -> String

 • Collapse runs of whitespace to a single space.
   - If `preserve_newlines=true`, 'hard' new-line characters are kept
     while spaces/tabs/CR/FF collapse.

 • Optionally strip leading/trailing blanks (`strip_ends = true`).

 • Optionally remove common zero-width code-points
   (ZWSP, ZWNJ, ZWJ, NBSP-like BOM).

The helper is UTF-8-safe and leaves non-whitespace graphemes unchanged.
"""
function normalize_whitespace(text::AbstractString;
                              strip_ends::Bool        = true,
                              preserve_newlines::Bool = false,
                              remove_zero_width::Bool = false)

    isempty(text) && return ""                      # quick exit

    t = text

    if remove_zero_width
        # ZWSP U+200B, ZWNJ U+200C, ZWJ U+200D, BOM U+FEFF
        t = replace(t, r"[\u200B\u200C\u200D\uFEFF]+" => "")
    end

    if preserve_newlines
        # collapse blanks except LF; keep a single space
        t = replace(t, r"[ \t\f\r]+" => " ")
        # remove blanks that precede newline(s)
        t = replace(t, r" +\n" => "\n")
    else
        # any Unicode whitespace → one space
        t = replace(t, r"\s+" => " ")
    end

    strip_ends && (t = strip(t))                    # leading/trailing

    return t
end
