using TextSpace
using Documenter

DocMeta.setdocmeta!(TextSpace, :DocTestSetup, :(using TextSpace); recursive=true)

makedocs(;
    modules=[TextSpace],
    authors="Alexander V. Mantzaris",
    sitename="TextSpace.jl",
    format=Documenter.HTML(;
        canonical="https://mantzaris.github.io/TextSpace.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mantzaris/TextSpace.jl",
)
