module VarianceAverseSocialLearning

using Distributions
using Agents


function calculate_kelly(estimated_p)
    k = 2*estimated_p - 1
    k > 0 ? k : 0
end


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

function simulate_success_rates(; u=0.7, λ=1000, n=100)
    [
        u / rand( Pareto(λ) )
        for i in 1:n
    ]
end

end # module VarianceAverseSocialLearning
