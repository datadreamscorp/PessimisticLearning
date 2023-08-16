using Distributions
using Agents
using Base.Threads

include("pessimistic_learning_Numeric.jl")

@agent Kelly NoSpaceAgent begin
    estimate::Float64
    social_estimate::Float64
    history::Vector{Int}
end


function initialize_kelly_learning(;
    N=100
)

    model = ABM(
        Kelly,
        nothing,
        properties = Dict(
            :N => N,
        ),
    )

    for i in 1:model.N
        add_agent!(model, 0.5, 0.5, [1])
    end

    return model
end
