using Documenter, DynamicHMC

makedocs(modules = [DynamicHMC],
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         clean = true,
         sitename = "DynamicHMC.jl",
         authors = "Tamás K. Papp",
         checkdocs = :exports,
         # strict = true,
         pages = ["Introduction" => "index.md",
                  "A worked example" => "worked_example.md",
                  "Documentation" => "interface.md"])

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git")
