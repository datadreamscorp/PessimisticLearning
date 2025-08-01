#############
#ANALYSIS 2 - STRATIFIED POPULATIONS
using Pkg
Pkg.activate("..")

using Distributed
addprocs(80)

@everywhere begin #INCLUDE MODEL CODE AND NECESSARY LIBRARIES

    using PessimisticLearning, Agents, CSV, Random

    total_gens = 2500

    seeds = rand(Xoshiro(5735333478), 1:100000, 25)

	parameters = Dict( #ALTER THIS DICTIONARY TO DEFINE PARAMETER DISTRIBUTIONS
    :N => 1000,
    :n => 20,
    :m => 20,
    :T => 500,
    :t => 10,
    :μ => 0.0:0.05:1.0|>collect,
    :λ => 0.0:0.05:1.0|>collect,
    :mu_std => 0.01,
    :mu_soc_h => 0.001,
    :mu_soc_v => 0.001,
    :mu_sens => 0.001,
    :mu_L => 0.01,
    :strategies => ["UB&PB"],
    :mu_parochial => 0.01,
    :selection => true,
    :mixed => true,
    :mixed_freq => 0.5,
    :mixed_aleph1 => 0.95,
    :mixed_aleph2 => [0.05, 0.45, 0.85],
    :seed => seeds
)

	mdata = [
        :Vbar_g0,
        :soc_h_median_g0,
        :soc_h_lerror_g0,
        :soc_h_herror_g0,
        :soc_v_median_g0,
        :soc_v_lerror_g0,
        :soc_v_herror_g0,
        :sens_median_g0,
        :sens_lerror_g0,
        :sens_herror_g0,
        :freq_ub_g0,
        :freq_pb_g0,
        :freq_parochial_g0,
        :s_mean_g0,
        :s_median_g0,
        :s_lerror_g0, 
        :s_herror_g0,
        :s_ltail_g0,
        :s_htail_g0,
        :s_end_mean_g0,
        :s_end_median_g0,
        :s_end_lerror_g0, 
        :s_end_herror_g0,
        :s_young_mean_g0,
        :s_young_median_g0,
        :s_young_lerror_g0, 
        :s_young_herror_g0,
        :s_child_mean_g0,
        :s_child_median_g0,
        :s_child_lerror_g0, 
        :s_child_herror_g0,
        :concentration_g0,
        :concentration_lerror_g0,
        :concentration_herror_g0,
        :Vbar_g1,
        :soc_h_median_g1,
        :soc_h_lerror_g1,
        :soc_h_herror_g1,
        :soc_v_median_g1,
        :soc_v_lerror_g1,
        :soc_v_herror_g1,
        :sens_median_g1,
        :sens_lerror_g1,
        :sens_herror_g1,
        :freq_ub_g1,
        :freq_pb_g1,
        :freq_parochial_g1,
        :s_mean_g1,
        :s_median_g1,
        :s_lerror_g1, 
        :s_herror_g1,
        :s_ltail_g1,
        :s_htail_g1,
        :s_end_mean_g1,
        :s_end_median_g1,
        :s_end_lerror_g1, 
        :s_end_herror_g1,
        :s_young_mean_g1,
        :s_young_median_g1,
        :s_young_lerror_g1, 
        :s_young_herror_g1,
        :s_child_mean_g1,
        :s_child_median_g1,
        :s_child_lerror_g1, 
        :s_child_herror_g1,
        :concentration_g1,
        :concentration_lerror_g1,
        :concentration_herror_g1,
        ]

end

#USE THIS LINE AFTER DEFINITIONS TO BEGIN PARAMETER SCANNING
_, mdf = paramscan(
            parameters, initialize_pessimistic_learning;
            mdata=mdata,
            n = total_gens,
			parallel=true,
			when_model = [total_gens],
			showprogress = true
	)

CSV.write("../data/analysis_2.csv", mdf)
