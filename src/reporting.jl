#####
##### Reporting progress.
#####

export ReportSilent, ReportIO

"""
$(TYPEDEF)

Subtypes implement

1. [`start_progress!`](@ref), which is used to start a particular iteration,

2. [`report!`](@ref), which triggers the display of progress,

3. [`end_progress!`](@ref) which "frees" the progress report, which can then be reused.
"""
abstract type AbstractReport end

"""
A placeholder type for not reporting any information.
"""
struct ReportSilent <: AbstractReport end

report!(::ReportSilent, objects...) = nothing

start_progress!(::ReportSilent, ::AbstractString; total_count = nothing) = nothing

end_progress!(::ReportSilent) = nothing

"""
$(TYPEDEF)

Display progress by printing lines to `io` if `countΔ` iterations *and* `time_nsΔ` nanoseconds
have passed since the last display.

$(FIELDS)
"""
mutable struct ReportIO{TIO <: IO} <: AbstractReport
    "IO stream for reporting."
    io::TIO
    "Color for report messages."
    print_color::Union{Symbol, Int}
    "Expected total count. When unknown, set to `nothing`."
    total_count::Union{Int, Nothing}
    "For comparing current count to the count at the last report. Not binding when negative."
    countΔ::Int
    "For comparing time to the time at the last report (in ns). Not binding when negative."
    time_nsΔ::Int
    "Time of starting the process. `nothing` unless start_progress! was called."
    start_time_ns::Union{UInt, Nothing}
    "Count when a report was last printed. `< 0` before `start_progress!`."
    last_printed_count::Int
    "Time (in ns) when a report was last printed."
    last_printed_time_ns::UInt
end

"""
    $SIGNATURES

Report to the given stream `io` (defaults to `stderr`).

See the documentation of the type for keyword arguments.
"""
function ReportIO(; io::IO = stderr, print_color = :blue,
                  total_count::Union{Nothing, Integer} = nothing,
                  countΔ::Integer = total_count isa Integer ? total_count ÷ 10 : 100,
                  time_nsΔ::Integer = 10^9)
    countΔ ≤ 0 && time_nsΔ ≤ 0 && @warn "progress report will be printed for every step"
    ReportIO(io, print_color, total_count, countΔ, time_nsΔ, nothing, 0, time_ns())
end

"""
    $SIGNATURES

Start a progress meter for an iteration.

`total_count` can be overwritten by a keyword argument.

After calling this function, [`report!`](@ref) should be used at every step with an integer.
"""
function start_progress!(report::ReportIO, msg; total_count = report.total_count)
    @unpack io, print_color = report
    @argcheck report.start_time_ns ≡ nothing "end_progress was not called"
    totalmsg = total_count ≡ nothing ? "unknown number of" : total_count
    msg *= " ($(totalmsg) steps)"
    printstyled(io, msg, '\n'; color = print_color, bold = true)
    report.total_count = total_count
    report.start_time_ns = report.last_printed_time_ns = time_ns()
    report.last_printed_count = 0
    nothing
end

function _report_avg_msg(report::ReportIO, count, _time_ns)
    s_per_iteration = (_time_ns - report.start_time_ns) / count / 1_000_000_000
    "$(round(s_per_iteration; sigdigits = 2)) s/step"
end

"""
    $SIGNATURES

Terminate a progress meter.
"""
function end_progress!(report::ReportIO, count::Integer = report.total_count::Integer)
    avgmsg = _report_avg_msg(report, count, time_ns())
    printstyled(report.io, "$(avgmsg) ...done\n"; bold = true, color = report.print_color)
    report.start_time_ns = nothing
end

"""
    $SIGNATURES

Display `report` via the appropriate mechanism. `count` is the index of the current step.
"""
function report!(report::ReportIO, count::Integer)
    @unpack io, countΔ, time_nsΔ, start_time_ns, last_printed_count, last_printed_time_ns,
        total_count = report
    @argcheck start_time_ns ≠ nothing "start_progress! was not called."
    _time_ns = time_ns()
    ispastcount = countΔ ≤ count - last_printed_count
    ispasttime = time_nsΔ ≤ _time_ns - last_printed_time_ns
    if ispastcount && ispasttime
        msg = "step $(count)"
        if total_count isa Int
            msg *= " (of $(total_count))"
        end
        msg *= ", " * _report_avg_msg(report, count, _time_ns)
        printstyled(io, msg, '\n'; color = report.print_color)
        report.last_printed_time_ns = _time_ns
        report.last_printed_count = count
    end
    nothing
end
