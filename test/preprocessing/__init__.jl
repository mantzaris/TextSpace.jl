module PreprocessingTests

# Import the main module
using TextSpace.Preprocessing

# Include all test files
include("CleanTextTests.jl")
include("TextNormalizationTests.jl")
include("TokenizationTests.jl")
include("CharProcessingTests.jl")
include("SentenceProcessingTests.jl")
include("ParagraphProcessingTests.jl")
include("IntegrationTests.jl")

end
