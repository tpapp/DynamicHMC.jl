using Documenter, DynamicHMC

makedocs(modules = [DynamicHMC],
         format = :html,
         clean = true,
         sitename = "DynamicHMC.jl",
         authors = "Tamás K. Papp",
         checkdocs = :exports,
         pages = [
             "Overview" => "index.md",
             "High-level API" => "api.md",
             "Low-level building blocks" => "lowlevel.md",
         ])

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git")
