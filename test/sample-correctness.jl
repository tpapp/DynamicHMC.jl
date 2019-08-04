#####
##### sample correctness tests
#####

####
#### LogDensityTestSuite is under active development, use the latest
#### FIXME remove code below when that package stabilizies and use Project.toml
####

try
    using LogDensityTestSuite
catch
    @info "installing LogDensityTestSuite"
    import Pkg
    Pkg.API.add(Pkg.PackageSpec(; url = "https://github.com/tpapp/LogDensityTestSuite.jl"))
    using LogDensityTestSuite
end
