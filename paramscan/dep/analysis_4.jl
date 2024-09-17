#############
#ANALYSIS 4
@everywhere using Pkg
@everywhere Pkg.activate("..")

@everywhere include("../src/pessimistic_learning_ABM.jl")

@everywhere using CSV, Distributed
@everywhere using Agents, Random, Distributions, Statistics, StatsBase

@everywhere total_ticks = 10

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 10000,
    :n => [5, 10, 25],
    :T => [1000],
    :t => [2, 25],
    :λ => [2.5],
    :ν => [1, 2, 3],
    :init_sens => [0.01, 0.05, 0.1],
    :strategies => ["UB", "CB", "PB", "HORIZONTAL"],
    :mu_soc_v => [0.01, 0.05],
    :envshift => [5],
    :lambda_shift => [6.0],
    :seed => 1:10|>collect
)

	mdata = [
        :opt_s, 
        :opt_payoff, 
        :s_median, 
        :s_lerror, 
        :s_herror, 
        :Vbar
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata=mdata,
            n = total_ticks,
			parallel=true,
			when_model = 0:total_ticks|>collect,
			showprogress = true
	)

CSV.write("../data/analysis_4.csv", mdf)
