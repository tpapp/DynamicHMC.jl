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

start_progress!(::ReportSilent, ::Union{Int, Void}, ::Any) = nothing

end_progress!(::ReportSilent) = nothing

mutable struct ReportIO{TIO <: IO} <: AbstractReport
    io::TIO
    color::Union{Symbol, Int}
    step_count::Int
    total::Union{Int, Void}
    last_count::Union{Int, Void}
    last_time::UInt
end

"""
    $SIGNATURES

Report to the given stream `io` (defaults to `STDERR`).

For progress bars, emit new information every after `step_count` steps.

`color` is used with `print_with_color`.
"""
ReportIO(; io = STDERR, color = :blue, step_count = 100) =
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
    print_with_color(report.color, report.io, msg, '\n'; bold = true)
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
    print_with_color(report.color, report.io, " ...done\n"; bold = true)
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
        msg *= ", $(signif(s_per_iteration, 2)) s/step"
        print_with_color(color, io, msg, '\n')
        report.last_time = t
        report.last_count = count
    end
    nothing
end
