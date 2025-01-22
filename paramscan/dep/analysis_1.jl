#############
#ANALYSIS 1 - EVOLUTION OF LEARNING PORTFOLIOS
using Pkg
Pkg.activate("..")

using Distributed
addprocs(90)

@everywhere begin 

    using PessimisticLearning, Agents, CSV, Random

    total_gens = 2500

    seeds = rand(Xoshiro(17395646456), 1:100000, 25)

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 1000,
    :n => [1, 5, 10, 15],
    :m => [1, 5, 10, 15],
    :T => [100],
    :t => 15,
    :u => [0.55, 0.6, 0.65, 0.75],
    :aleph => 0.05:0.05:0.95|>collect,
    :envshift => [1, 3000],
    :periodic => true,
    :peg_aleph => true,
    :u_shift => [0.65, 0.85],
    :mu_std => 0.01,
    :mu_soc_h => 0.001,
    :mu_soc_v => 0.001,
    :mu_sens => 0.001,
    :mu_L => 0.01,
    :strategies => ["UB&PB"],
    :selection => true,
    :mixed => false,
    :seed => seeds
)

	mdata = [
        :Vbar,
        :soc_h_median,
        :soc_h_lerror,
        :soc_h_herror,
        :soc_v_median,
        :soc_v_lerror,
        :soc_v_herror,
        :sens_median,
        :sens_lerror,
        :sens_herror,
        :freq_ub,
        :freq_pb,
        :s_mean,
        :s_median,
        :s_lerror, 
        :s_herror,
        :s_ltail,
        :s_htail,
        :s_end_mean,
        :s_end_median,
        :s_end_lerror, 
        :s_end_herror,
        :s_young_mean,
        :s_young_median,
        :s_young_lerror, 
        :s_young_herror,
        :s_child_mean,
        :s_child_median,
        :s_child_lerror, 
        :s_child_herror,
        :concentration,
        :concentration_lerror,
        :concentration_herror
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata = mdata,
            n = total_gens,
			parallel = true,
			when_model = [total_gens],
			showprogress = true
	)

CSV.write("../data/analysis_1.csv", mdf)
