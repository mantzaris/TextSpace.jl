# (Optional) Additional text normalization (e.g., accent stripping, stemming, lemmatization)
function normalize_unicode(text::String; form::Symbol=:NFC) 
    return Unicode.normalize(text, form)
end


function normalize_whitespace(text::String)
    return strip(replace(text, r"\s+" => " "))
end