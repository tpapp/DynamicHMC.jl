using Documenter, DynamicHMC

makedocs(modules = [DynamicHMC],
         format = :html,
         clean = true,
         sitename = "DynamicHMC.jl",
         authors = "Tamás K. Papp",
         html_prettyurls = haskey(ENV, "TRAVIS"), # clean URLs building on Travis
         pages = [
             "Overview" => "index.md",
             "High-level API" => "api.md",
             "Low-level building blocks" => "lowlevel.md",
         ])

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git",
           target = "build",
           deps = nothing,
           make = nothing,
           julia = "0.6")
