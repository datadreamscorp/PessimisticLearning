#############
#ANALYSIS 3
@everywhere using Pkg
@everywhere Pkg.activate("..")

@everywhere include("../src/pessimistic_learning_ABM.jl")

@everywhere using CSV, Distributed
@everywhere using Agents, Random, Distributions, Statistics, StatsBase

@everywhere total_ticks = 20

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 10000,
    :n => [5, 25],
    :T => [1000],
    :t => [25],
    :λ => [6.0],
    :aleph => [0.65, 0.8, 0.95],
    :init_soc_v => 0.0:0.05:1.0|>collect,
    :init_sens => [0.1],
    :strategies => ["UB"],
    :envshift => [2, 5, 10],
    :lambda_shift => [2.0, 3.0, 4.0, 5.0],
    :periodic => true,
    :seed => 1:5|>collect
)

	mdata = [
        :opt_s, 
        :opt_payoff, 
        :s_median, 
        :s_lerror, 
        :s_herror, 
        :Vbar,
        :gVbar
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

CSV.write("../data/analysis_3.csv", mdf)
