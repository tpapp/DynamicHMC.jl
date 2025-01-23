using Documenter, DynamicHMC

makedocs(modules = [DynamicHMC],
         format = Documenter.HTML(; prettyurls = get(ENV, "CI", nothing) == "true"),
         clean = true,
         sitename = "DynamicHMC.jl",
         authors = "Tamás K. Papp",
         checkdocs = :exports,
         linkcheck = true,
         linkcheck_ignore = [r"^.*xcelab\.net.*$", r"^.*stat\.columbia\.edu.*$"],
         pages = ["Introduction" => "index.md",
                  "A worked example" => "worked_example.md",
                  "Documentation" => "interface.md"])

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git",
           push_preview = true)
