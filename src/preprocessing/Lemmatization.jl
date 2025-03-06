
function lemmatize_text(text::String; language::String="english") 
    words = split(text) 
    lemmas = [lemmatize_word(w; language=language) for w in words] 
    return join(lemmas, ' ') 
end

function lemmatize_word(word::String; language::String="english")
    #convert to lowercase to match dictionary patterns 
    lw = lowercase(word)

    if language == "english" #TODO: expand on this
        irregular_nouns = Dict(
            "mice"     => "mouse",
            "geese"    => "goose",
            "men"      => "man",
            "women"    => "woman",
            "children" => "child",
            "feet"     => "foot",
            "teeth"    => "tooth"
        )
        if haskey(irregular_nouns, lw)
            return irregular_nouns[lw]
        end
    
        #check for known irregular verbs
        # TODO: expand
        irregular_verbs = Dict(
            "were" => "be",
            "was"  => "be",
            "ran"  => "run",
            "ate"  => "eat"
        )
        if haskey(irregular_verbs, lw)
            return irregular_verbs[lw]
        end
    
        #simple verb transformation: remove trailing "ing" or "ed" if present
        # TODO: expand
        if endswith(lw, "ing")
            return lw[1:end-3]
        elseif endswith(lw, "ed")
            return lw[1:end-2]
        end
    end
    
    return lw
end


function endswith(word::String, suffix::String) 
    return length(word) >= length(suffix) && word[end-length(suffix)+1:end] == suffix 
end