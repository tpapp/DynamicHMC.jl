Pkg.clone("https://github.com/tpapp/MCMCDiagnostics.jl.git")
Pkg.add("DiffResults")         # remove this and the next line when released ...
Pkg.checkout("DiffResults")    # ... DiffResults precompiles
Pkg.clone(pwd()); Pkg.build("DynamicHMC"); Pkg.test("DynamicHMC"; coverage=true)
