import Pkg
Pkg.build("DynamicHMC")
Pkg.test("DynamicHMC"; coverage=true)
