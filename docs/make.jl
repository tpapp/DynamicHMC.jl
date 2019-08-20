using Documenter, DynamicHMC

makedocs(modules = [DynamicHMC],
         format = Documenter.HTML(),
         clean = true,
         sitename = "DynamicHMC.jl",
         authors = "Tamás K. Papp",
         checkdocs = :exports,
         # strict = true,
         pages = [
             "Documentation" => "index.md",
         ])

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git")
