#####
##### utilities
#####

####
#### error messages
####

"""
$(TYPEDEF)

The error type used by this package. Debug information should be printed without truncation,
with full precision.

$(FIELDS)
"""
struct DynamicHMCError <: Exception
    message::String
    debug_information::NamedTuple
end

"""
$(SIGNATURES)

Throw a `DynamicHMCError` with given message, keyword arguments used for debug information.
"""
_error(message::AbstractString; kwargs...) = throw(DynamicHMCError(message, NamedTuple(kwargs)))

function Base.showerror(io::IO, error::DynamicHMCError)
    (; message, debug_information) = error
    printstyled(io, "DynamicHMC error: ", error; color = :red)
    for (key, value) in pairs(debug_information)
        print(io, "\n  ")
        printstyled(io, string(key); color = :blue, bold = true)
        printstyled(io, " = ", value)
    end
    nothing
end
