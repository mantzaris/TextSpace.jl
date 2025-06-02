

using Test
using TextSpace                     # loads the package
using TextSpace.Plumbing: tokenize  # import ONE helper to prove it works

@testset "Plumbing smoke-test" begin
    @test tokenize("Hello, World!") == ["hello", "world"]
end