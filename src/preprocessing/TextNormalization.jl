# (Optional) Additional text normalization (e.g., accent stripping, stemming, lemmatization)
function normalize_unicode(text::String; form::Symbol=:NFC) 
    return Unicode.normalize(text, form)
end


function normalize_whitespace( text::String; strip_ends::Bool=true, preserve_newlines::Bool=false, remove_zero_width::Bool=false ) local t = text

    # remove zero-width or invisible characters if requested
    if remove_zero_width
        #common zero-width characters: \u200B (ZWSP), \u200C (ZWNJ) 
        #\u200D (ZWJ), \uFEFF (BOM/ZWNBSP)
        t = replace(t, r"[\u200B\u200C\u200D\uFEFF]+" => "")
    end
    
    # decide how to collapse whitespace
    if preserve_newlines
        #replace runs of spaces/tabs, but leave newlines intact
        # pattern: `[ \t]+` => " "
        t = replace(t, r"[ \t\f\r]+" => " ")
    else
        # Replace any run of whitespace characters (including newlines) with a single space
        # \s matches [ \t\r\n\f] plus Unicode vertical tab
        t = replace(t, r"\s+" => " ")
    end
    
    #strip leading and trailing 
    if strip_ends
        t = strip(t)
    end
    
    return t    
end