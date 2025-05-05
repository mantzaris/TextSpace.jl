include(joinpath(@__DIR__, "..", "src", "preprocessing", "TextNormalization.jl"))






##############################################################################
#  normalization_tests.jl
##############################################################################

# ── bring the helpers under test ────────────────────────────────────────────
# (adjust the path / filename if you placed these two functions elsewhere)

##############################################################################
#  normalize_unicode
##############################################################################
@testset "normalize_unicode" begin
    decomposed = "Cafe\u0301"                 # "Café" (e + COMBINING ACUTE)
    composed   = "Café"                       # NFC form

    @test normalize_unicode(decomposed) == composed          # default :NFC
    @test normalize_unicode(composed; form = :NFD) != composed   # round‑trip differs
    @test Unicode.normalize(composed, :NFD) ==
          normalize_unicode(composed; form = :NFD)            # same API
end

##############################################################################
#  normalize_whitespace  — default behaviour
##############################################################################
@testset "normalize_whitespace — defaults" begin
    txt = "  foo\t\tbar\nbaz   "
    @test normalize_whitespace(txt) == "foo bar baz"          # collapsed + trimmed
end

##############################################################################
#  preserve_newlines flag
##############################################################################
@testset "preserve_newlines = true" begin
    txt  = "foo \t  bar\nbaz\r\nqux"
    out  = normalize_whitespace(txt; preserve_newlines = true)

    @test out == "foo bar\nbaz\nqux"                          # spaces collapsed, NL kept
    @test occursin('\n', out)
    @test !occursin('\t', out)                                # tabs gone
end

##############################################################################
#  strip_ends = false
##############################################################################
@testset "strip_ends = false" begin
    txt = "   foo bar   "
    @test normalize_whitespace(txt; strip_ends = false) ==
          " foo bar "                                         # leading/trailing kept
end

##############################################################################
#  remove_zero_width flag
##############################################################################
@testset "remove_zero_width = true" begin
    zwsp = "\u200B"                               # zero-width space
    txt  = "foo" * zwsp * "bar  baz"
    @test normalize_whitespace(txt; remove_zero_width = true) ==
          "foobar baz"                            # zwsp removed, blanks collapsed
end


