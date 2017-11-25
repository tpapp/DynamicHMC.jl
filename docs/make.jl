using Documenter, DynamicHMC

makedocs(
    modules = [DynamicHMC],
    format = :html,
    clean = true,
    sitename = "DynamicHMC.jl",
    authors = "Tamás K. Papp",
    pages = [
        "Overview" => "index.md",
        "High-level API" => "api.md",
        "Low-level building blocks" => "lowlevel.md",
    ],
    # Use clean URLs when built on Travis
    html_prettyurls = haskey(ENV, "TRAVIS"),
)

deploydocs(repo = "github.com/tpapp/DynamicHMC.jl.git", julia = "0.6")
