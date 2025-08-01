### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ a377064f-27c7-420c-aaf3-f33f7b180d37
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PessimisticLearning
	using StatsBase, Random, Distributions, Agents, Plots, CSV, DataFrames
	using PlutoUI, LaTeXStrings
end

# ╔═╡ d7675a4c-e997-11ef-16bd-25ec23ea18c7
begin
	modelio = initialize_pessimistic_learning(
		N = 500,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		seed = 7557697,
	)

	adata, mdata = run!(
		modelio, 
		2500,
		adata=[:s],
		mdata=[
			:Vbar, :s_median, :s_young_median, :s_child_median, :soc_v_median, 
			:soc_h_median, :sens_median, :sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
end

# ╔═╡ Cell order:
# ╠═a377064f-27c7-420c-aaf3-f33f7b180d37
# ╠═d7675a4c-e997-11ef-16bd-25ec23ea18c7
