module PessimisticLearning

export initialize_pessimistic_learning, trauma
export g, g_ruin, probruin, probruin2, probruin_numeric, s_star, s_star_numeric#, optimal_stake

include("../src/pessimistic_learning_ABM.jl")
include("../src/pessimistic_learning_Numeric.jl")

end # module PessimisticLearning
