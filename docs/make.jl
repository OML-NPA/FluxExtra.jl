
using FluxExtra, Documenter

makedocs(modules=[FluxExtra,FluxExtra.Normalizations],
    sitename = "FluxExtra.jl",
    pages = ["Home" => "index.md",
            "Layers" => "layers.md",
            "Functions" => "functions.md"],
    authors = "Open Machine Learning Association",
    format = Documenter.HTML(prettyurls = false)
)

deploydocs(
    repo = "github.com/OML-NPA/FluxExtra.jl.git",
    devbranch = "main"
)
