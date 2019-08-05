include("common.jl")

macro include_testset(filename)
    @assert filename isa AbstractString
    quote
        @testset $(filename) begin
            include($(filename))
        end
    end
end

@include_testset("test-trees.jl")
@include_testset("test-hamiltonian.jl")
@include_testset("test-NUTS.jl")
@include_testset("test-stepsize.jl")
@include_testset("test-mcmc.jl")
# @include_testset("test-sample-normal.jl")
# @include_testset("test-normal-mcmc.jl")
@include_testset("test-diagnostics.jl")
# @include_testset("test-reporting.jl") FIXME being rewritten
@include_testset("sample-correctness.jl")
