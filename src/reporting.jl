#####
##### Reporting progress.
#####

"""
$(TYPEDEF)

A placeholder type for not reporting any information.
"""
struct NoProgressReport end

"""
$(SIGNATURES)

Report to the given `reporter`.

The second argument can be

1. a string, which is displayed as is (this is supported by all reporters).

2. or a step in an MCMC chain with a known number of steps for progress reporters (see
[`make_mcmc_reporter`](@ref)).

`meta` arguments are key-value pairs.

In this context, a *step* is a NUTS transition, not a leapfrog step.
"""
report(reporter::NoProgressReport, step::Union{AbstractString,Integer}; meta...) = nothing

"""
$(SIGNATURES)

Return a reporter which can be used for progress reports with a known number of
`total_steps`. May return the same reporter, or a related object. Will display `meta` as
key-value pairs.
"""
make_mcmc_reporter(reporter::NoProgressReport, total_steps; meta...) = reporter

"""
$(TYPEDEF)

Report progress into the `Logging` framework, using `@info`.

For the information reported, a *step* is a NUTS transition, not a leapfrog step.

# Fields

$(FIELDS)
"""
Base.@kwdef struct LogProgressReport{T}
    "ID of chain. Can be an arbitrary object, eg `nothing`."
    chain_id::T = nothing
    "Always report progress past `step_interval` of the last report."
    step_interval::Int = 100
    "Always report progress past this much time (in seconds) after the last report."
    time_interval_s::Float64 = 1000.0
end

"""
$(SIGNATURES)

Assemble log message metadata.

Currently, it adds `chain_id` *iff* it is not `nothing`.
"""
_log_meta(chain_id::Nothing, meta) = meta

_log_meta(chain_id, meta) = (chain_id = chain_id, meta...)

function report(reporter::LogProgressReport, message::AbstractString; meta...)
    @info message _log_meta(reporter.chain_id, meta)...
    nothing
end

"""
$(TYPEDEF)

A composite type for tracking the state for which the last log message was emitted, for MCMC
reporting with a given total number of steps (see [`make_mcmc_reporter`](@ref).

# Fields

$(FIELDS)
"""
mutable struct LogMCMCReport{T}
    "The progress report sink."
    log_progress_report::T
    "Total steps for this stage."
    total_steps::Int
    "Index of the last reported step."
    last_reported_step::Int
    "The last time a report was logged (determined using `time_ns`)."
    last_reported_time_ns::UInt64
end

function report(reporter::LogMCMCReport, message::AbstractString; meta...)
    @info message _log_meta(reporter.log_progress_report.chain_id, meta)...
    nothing
end

function make_mcmc_reporter(reporter::LogProgressReport, total_steps::Integer; meta...)
    @info "Starting MCMC" total_steps = total_steps meta...
    LogMCMCReport(reporter, total_steps, -1, time_ns())
end

function report(reporter::LogMCMCReport, step::Integer; meta...)
    @unpack (log_progress_report, total_steps, last_reported_step,
             last_reported_time_ns) = reporter
    @unpack chain_id, step_interval, time_interval_s = log_progress_report
    @argcheck 1 ≤ step ≤ total_steps
    Δ_steps = step - last_reported_step
    t_ns = time_ns()
    Δ_time_s = (t_ns - last_reported_time_ns) / 1_000_000_000
    if last_reported_step < 0 || Δ_steps ≥ step_interval || Δ_time_s ≥ time_interval_s
        seconds_per_step = Δ_time_s / Δ_steps
        meta_progress = (step = step,
                         seconds_per_step = round(seconds_per_step; sigdigits = 2),
                         estimated_seconds_left = round((total_steps - step) *
                                                        seconds_per_step; sigdigits = 2))
        @info "MCMC progress" merge(_log_meta(chain_id, meta_progress), meta)...
        reporter.last_reported_step = step
        reporter.last_reported_time_ns = t_ns
    end
    nothing
end

"""
$(SIGNATURES)

Return a default reporter, taking the environment into account. Keyword arguments are passed
to constructors when applicable.
"""
function default_reporter(; kwargs...)
    if isinteractive()
        LogProgressReport(; kwargs...)
    else
        NoProgressReport()
    end
end
