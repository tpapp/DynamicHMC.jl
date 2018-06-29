export ReportSilent, ReportIO

"""
Subtypes implement [`report!`](@ref), [`start_progress!`](@ref), and
[`end_progress!`](@ref).
"""
abstract type AbstractReport end

"""
A placeholder type for not reporting any information.
"""
struct ReportSilent <: AbstractReport end

report!(::ReportSilent, objects...) = nothing

start_progress!(::ReportSilent, ::Union{Int, Nothing}, ::Any) = nothing

end_progress!(::ReportSilent) = nothing

mutable struct ReportIO{TIO <: IO} <: AbstractReport
    io::TIO
    color::Union{Symbol, Int}
    step_count::Int
    total::Union{Int, Nothing}
    last_count::Union{Int, Nothing}
    last_time::UInt
end

"""
    $SIGNATURES

Report to the given stream `io` (defaults to `stderr`).

For progress bars, emit new information every after `step_count` steps.

`color` is used with `print_with_color`.
"""
ReportIO(; io = stderr, color = :blue, step_count = 100) =
    ReportIO(io, color, step_count, nothing, nothing, zero(UInt))

"""
    $SIGNATURES

Start a progress meter for an iteration. The second argument is either

- `nothing`, if the total number of steps is unknown,

- an integer, for the total number of steps.

After calling this function, [`report!`](@ref) should be used at every step with
an integer.
"""
function start_progress!(report::ReportIO, total, msg)
    if total isa Integer
        msg *= " ($(total) steps)"
    end
    printstyled(report.io, msg, '\n'; bold = true, color = report.color)
    report.total = total
    report.last_count = 0
    report.last_time = time_ns()
    nothing
end

"""
    $SIGNATURES

Terminate a progress meter.
"""
function end_progress!(report::ReportIO)
    printstyled(report.io, " ...done\n"; bold = true, color = report.color)
    report.last_count = nothing
end

"""
    $SIGNATURES

Display `objects` via the appropriate mechanism.

When a single `Int` is given, it is treated as the index of the current step.
"""
function report!(report::ReportIO, count::Int)
    @unpack io, step_count, color, total = report
    @argcheck report.last_count isa Int "start_progress! was not called."
    if count % step_count == 0
        msg = "step $(count)"
        if total isa Int
            msg *= "/$(total)"
        end
        t = time_ns()
        s_per_iteration = (t - report.last_time) / step_count / 1000
        msg *= ", $(round(s_per_iteration; sigdigits = 2)) s/step"
        printstyled(io, msg, '\n'; color = color)
        report.last_time = t
        report.last_count = count
    end
    nothing
end
