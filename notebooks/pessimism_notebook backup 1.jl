### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ cfb0a045-a40d-4019-987f-e4dc1c92074e
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using PessimisticLearning
	using StatsBase, Random, Distributions, Agents, Plots, CSV, DataFrames
	using PlutoUI, LaTeXStrings

	md"""
	## The Development of Risk Attitudes and their Cultural Transmission
	
	#### Alejandro Pérez Velilla, Bret Beheim, Paul E. Smaldino.
	"""
end

# ╔═╡ 1f7f1128-8eea-4695-b8cd-044d7c91890b
md"""
## Main text
"""

# ╔═╡ e8b7171b-9810-402c-818c-61c12fb3fec7
md"""
#### Figure 1
"""

# ╔═╡ 896ff8a7-0f61-4be3-adf4-d8e55ad16874
begin
	env_plot = plot( 
		plot_powerdist.(
		[1.0, 1.5, 2.0, 3.0, 4.0, 5.0], n=1000)..., 
		layout=(2,3), 
		link=:all,
		#plot_title="power-distributed success rates (u)",
	)
	
	savefig(env_plot, "../images/fig1_env.pdf")
	
	env_plot
end

# ╔═╡ ffd166fa-9d48-477a-8d16-f5e866551dfc
md"""
#### Figure 2
"""

# ╔═╡ b739131f-a173-4994-8143-d6c52ac07cb3
begin
	λ = 1.0:1.0:5.0
	xaxis2 = 0.001:0.001:1.0
	payplot = plot(
		xaxis2,
		g.(xaxis2, l=λ[1]),
		label = "$(λ[1])",
		legendtitle = L"λ",
		legendtitlefontsize=10,
		legendfontsize=8,
		legend=:topright,
		lw = 2,
		xlab = "",
		ylab = L"G (s)",
		alpha = 0.5,
		ylim=(0.9, 1.35),
		dpi=300,
		palette=:Dark2_5
	)
	
	for l in λ[2:length(λ)]
		plot!(
		xaxis2,
		PessimisticLearning.g.(xaxis2, l=l),
		label = "$l",
		lw = 2,
		alpha = 0.5
	)
	end

	for l in λ
		scatter!(
			( xaxis2[ findmax(g.(xaxis2, l=l))[2] ], findmax(g.(xaxis2, l=l))[1] ),
			color="black",
			label=""
		)
	end

	hline!([1.0], lw=1, ls=:dash, color="black", label="")
	annotate!([0.125], [1.25], [text(L"\aleph \rightarrow 1", 13, color="black")])
	
	Random.seed!(1234567)
	splots_barrier = [
		simplots(
			1.0:1.0:5.0, 0:0.01:1, 
			true, 10000, 
			aleph, 
			aleph == 0.65 ? true : false, 
			aleph == 0.65 ? true : false,
			aleph == 0.65 ? true : false
		) 
		for aleph in [0.95, 0.8, 0.65]
	]

	splot = plot(
		splots_barrier...,
		layout=(3,1),
		#plot_title="mean payoffs under absorbing barriers",
		link=:all,
		ylim=(0.9, 1.3),
		dpi=300,
		#legend=false,
		title="",
		xlabelfontsize=12,
	)

	full_payplot = plot(
		payplot,
		splot,
		layout=(2,1),
		size=(600,700)
	)
	
	savefig(full_payplot, "../images/fig3_kelly.pdf")

	splots_barrier = nothing
	GC.gc()
	
	full_payplot
end

# ╔═╡ 5e5d671d-0b35-4404-b022-c2fe8bdde7c2
md"""
#### Figure 3
"""

# ╔═╡ aa8d557f-6436-407f-a0f7-afa823a35129
begin
	indv_plot = plot(
		plot(
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.65, t=2, seed=1), ylabel=L"\bar{V} \quad | \quad \tau = "*"2", ylabelfontsize=12, legend=false, title=L"\aleph = "*"0.65", titlefontsize=12 ), 
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.8, t=2, seed=1), legend=false, title=L"\aleph = "*"0.80", titlefontsize=12 ),
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.95, t=2, seed=1), legend=false, title=L"\aleph = "*"0.95", titlefontsize=12 ),
			layout=(1,3)
		),
		plot(
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.65, t=25, seed=1), ylabel=L"\bar{V} \quad | \quad \tau = "*"25", ylabelfontsize=12, xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8, legend=false ),
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.8, t=25, seed=1), legend=false, xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8 ),
			plot( plot_indv_learning([1.5, 3.0, 6.0]; n=25, aleph=0.95, t=25, seed=1), xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8 ),
			layout=(1,3), legendfontsize=9, legendtitlefontsize=11
		),
		layout=(2,1),
		link=:all,
		margins=2Plots.mm
	)
	
	savefig(indv_plot, "../images/fig3_indv.pdf")

	indv_plot
end

# ╔═╡ f3fec0a1-40fc-4168-93a5-bb0648086d64
md"""
#### Figure 4
"""

# ╔═╡ d008b259-7a84-4f7e-99ed-e4b283775591
begin
	sens_plot = plot([ 
		plot( plot_sensitivity.([5, 25], i, 1000, 25; seed=1)..., layout=(2,1) )
		for i in [0.65, 0.80, 0.95]
		]...,
		layout=(1,3), legendfontsize=10, legendtitlefontsize=11
	)
	savefig(sens_plot, "../images/fig4_sensitivity.pdf")
	sens_plot
end

# ╔═╡ adeef0c1-d6dd-4d44-9d59-866cd6b47c21
md"""
#### Figure 5
"""

# ╔═╡ 71768709-5e37-43fc-91d9-cd47e659e1d8
begin
	dev_plot = plot(
		plot( development_plot(2.0, xlabel="", t=true, legend=false), xaxis=false, xlim=(0,1)),
		plot( development_plot(3.0, xlabel="", legend=false), xaxis=false, xlim=(0,1)),
		development_plot(6.0, legend=:topleft),
		layout=(3,1), link=:all, legendfontsize=10, ylim=(0, 1200), size=(800, 800), margins=1Plots.mm
	)
	savefig(dev_plot, "../images/fig5_development.pdf")
	dev_plot
end

# ╔═╡ d8af2358-06f5-4190-b60c-eea14d12e5e3
md"""
#### Figure 6
"""

# ╔═╡ ee1b6bf1-3074-4b5d-8895-aaf034ba3f31
begin
	aleph_plotter = plot(
		aleph_plot(1.5, legend=true, ylab=L"\bar{V}"),
		aleph_plot(3.0),
		aleph_plot(6.0),
		link=:all,
		layout=(1,3),
		dpi=300,
		size=(600, 300),
		legendfontsize=7,
		legendtitlefontsize=10
	)
	
	savefig(aleph_plotter, "../images/fig6_aleph.pdf")

	aleph_plotter
end

# ╔═╡ 1d29ca8c-12f6-4803-a6cb-7d5f27576a02
md"""
#### Figure 7
"""

# ╔═╡ aebf5e0d-7520-4f6e-9dff-493c4e7a5bb0
begin
	attitude_plot = plot(
		plot( plot_conservative_attitudes(1.5), ylabel="median stake "*L"(s)", xlabel="" ),
		plot( plot_conservative_attitudes(3.0), legend=false, xlabelfontsize=12 ),
		plot( plot_conservative_attitudes(6.0), legend=false, xlabel="" ),
		layout=(1,3), link=:all, size=(700, 300), margins=4Plots.mm, legendfontsize=10, legendtitlefontsize=12,
	)
	#savefig(attitude_plot, "../images/fig7_attitudes.pdf")
	#attitude_plot

	conservative_payoff_plot = plot(
		plot( plot_conservative_payoffs(1.5), ylabel=L"\bar{V}", xlabel="" ),
		plot( plot_conservative_payoffs(3.0), legend=false, xlabelfontsize=12 ),
		plot( plot_conservative_payoffs(6.0), legend=false, xlabel="" ),
		layout=(1,3), link=:all, size=(700, 300), margins=4Plots.mm
	)
	#savefig(conservative_payoff_plot, "../images/fig8_payoffs.pdf")
	#conservative_payoff_plot

	full_conservative_plot = plot(
		plot(attitude_plot, xlab=""),
		plot(conservative_payoff_plot, title="", legend=false),
		layout=(2,1), size=(650, 600), dpi=300, xrotation=90
	)

	savefig(full_conservative_plot, "../images/fig7_payoffs_attitudes.pdf")

	full_conservative_plot
end

# ╔═╡ 738983ff-eaff-4628-97f3-25455458f047
md"""
#### Figure 8
"""

# ╔═╡ 7300889d-373f-424d-9cdc-f28856cb0f23
begin
	abmplot = plot(
		run_ABM_plot(N=10000, aleph=0.65, t=25, legend=:topleft),
		run_ABM_plot(N=10000, aleph=0.95, t=25, plot_lab="(B)", legend=false),
		layout=(1,2), size=(800,400), 
		margins=3Plots.mm, xtickfontsize=5, xrotation=90
	)
	
	savefig(abmplot, "../images/fig8_time.pdf")

	GC.gc()
	
	abmplot
end

# ╔═╡ f37c0c19-e434-4d80-8b3a-31732df114dd
md"""
#### Figure 9
"""

# ╔═╡ 6ea0d83d-b3df-443a-9dd0-baf314ded951
begin
	inequality_plot = plot(
		plot(
			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=10, freq=0.75, leg=false),  leg=:bottomleft, title=L"f = 0.25"),
			
			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=10, freq=0.5, leg=false), ylab="", title=L"f = 0.5"),

			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=10, freq=0.25), ylab="", title=L"f = 0.75", leg=false),
			layout=(1,3), xlab=""
		),
		plot(
			mixed_pop_plot(mshift=6.0, n=25, eshift=5, freq=0.75, leg=false),
			
			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=5, freq=0.5, leg=false), ylabel=""),

			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=5, freq=0.25, leg=false), ylabel=""),
			
			layout=(1,3),
			xlabel=""
		),
		plot(
			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=2, freq=0.75, leg=false), xlab=""),
			
			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=2, freq=0.5, leg=false), ylabel=""),

			plot(mixed_pop_plot(mshift=6.0, n=25, eshift=2, freq=0.25, leg=false), ylabel="", xlab=""),
			
			layout=(1,3)
		),
		layout=(3,1), link=:all, size=(800, 600), margins=5Plots.mm, legendtitlefontsize=10, legendfontsize=9
	)

	savefig(inequality_plot, "../images/fig9_inequality.pdf")

	inequality_plot
end

# ╔═╡ 25ae55f1-d1d8-4974-ac88-e5c4e6bfd699
md"""
#### Figure 10
"""

# ╔═╡ 26ea2476-a0da-4f33-8e4b-6d3619db30b6
begin
	_, mdat = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.95), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat2 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.75), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat3 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.5), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat4 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.0), 20, mdata=[:Vbar_g0, :Vbar_g1] )

	poverty_trap_plot = plot(
		0:20, mdat.Vbar_g0, 
		ls=:solid, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="", legendtitle=L"α",
		xlab="time (generations)",
		ylab=L"\bar{V}",
		xlabelfontsize=8,
	)
	plot!(
		0:20, mdat2.Vbar_g0,
		ls=:dash, lw=2, 
		color=palette(:Dark2_5)[1], 
		label=""
	)
	plot!(
		0:20, mdat3.Vbar_g0, 
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label=""
	)
	plot!(
		0:20, mdat4.Vbar_g0, 
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label=""
	)
	plot!(
		0:20, mdat3.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="", ls=:dashdotdot, alpha=0.5
	)
	plot!(
		0:20, mdat.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="",
		ls=:solid,
		alpha=0.5
	)
	plot!(
		0:20, mdat2.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="",
		ls=:dash,
		alpha=0.5
	)
	plot!(
		0:20, mdat4.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="",
		ls=:dot,
		alpha=0.5
	)

	plot!(
		[], 
		lw=2, color="black", 
		label="0.95", ls=:solid
	)
	plot!(
		[],
		ls=:dash, lw=2, 
		color="black", 
		label="0.75"
	)
	plot!(
		[], 
		ls=:dashdotdot, lw=2, 
		color="black", 
		label="0.5"
	)
	plot!(
		[], 
		ls=:dot, lw=2, 
		color="black", 
		label="0.0"
	)
	

	hline!(
		[1.0], ls=:dash, color="black", label="", alpha=0.75
	)
	vline!(
		[10], ls=:solid, color="black", label="", alpha=1
	)

	annotate!([5], [1.26], [text("advantaged class", 9, color=palette(:Dark2_5)[2])] )
	annotate!([5], [0.81], [text("disadvantaged class", 9, color=palette(:Dark2_5)[1])] )


	mdat = nothing
	mdat2 = nothing
	mdat3 = nothing
	mdat4 = nothing
	GC.gc()


	_, mdat5 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.95), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat6 = run!( initialize_pessimistic_learning(mixed=true, parochial=true, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.95), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat7 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=3, mixed_L2=3, envshift=10, init_soc_v=0.5), 20, mdata=[:Vbar_g0, :Vbar_g1] )

	poverty_trap_plot_ingroup = plot(
		0:20, mdat5.Vbar_g0, 
		ls=:solid, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="", legendtitle="learning strategies "*L"(\alpha = 0.95)",
		legendtitlefontsize=10,
		legendfontsize=8,
		xlab="time (generations)",
		ylab=L"\bar{V}",
		xlabelfontsize=8
	)
	plot!(
		0:20, mdat5.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="", ls=:solid, alpha=0.5
	)
	plot!(
		0:20, mdat7.Vbar_g0,
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label=""
	)
	plot!(
		0:20, mdat7.Vbar_g1, 
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[2], 
		label="", alpha=0.5
	)
	plot!(
		0:20, mdat6.Vbar_g0,
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label=""
	)
	plot!(
		0:20, mdat6.Vbar_g1, 
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[2], 
		label="", alpha=0.5
	)

	
	plot!(
		[],
		lw=2, color="black", 
		label="non-parochial", ls=:solid, alpha=0.75
	)
	plot!(
		[], 
		ls=:dot, lw=2, 
		color="black", 
		label="non-parochial payoff bias"
	)
	plot!(
		[],
		ls=:dashdotdot, lw=2, 
		color="black", 
		label="parochial learning"
	)
	
	
	hline!(
		[1.0], ls=:dash, color="black", label="", alpha=0.75
	)
	vline!(
		[10], ls=:solid, color="black", label="", alpha=1
	)
	annotate!([5], [0.9], [text("unequal group conditions", 9, color="black")] )
	annotate!([15], [1.1], [text("equal group conditions", 9, color="black")] )


	mdat5 = nothing
	mdat6 = nothing
	mdat7 = nothing
	GC.gc()
	
	total_poverty = plot(
		plot(poverty_trap_plot, title="(A) exploratory vs. conservative learning styles"),
		plot(poverty_trap_plot_ingroup, title="(B) parochial vs. non-parochial (conservative) learning styles", titlefontsize=11),
		layout=(2,1), size=(600,700)
	)

	savefig(total_poverty, "../images/fig10_learningstyles.pdf")

	total_poverty
end

# ╔═╡ ceacb32a-71ca-49c6-ab92-f5f33fc32b31
md"""
## Supplemental Figures
"""

# ╔═╡ a3dec80f-60df-44b1-adc5-855cf06b492a
md"""
#### Figure S1
"""

# ╔═╡ 3a2082c4-ef09-4b53-af1b-411792729560
#=╠═╡
begin
	probruin_plot = plot(
		1.5:0.05:5,
		[probruin(l, 0.65, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", label="0.65", legendtitle=L"\aleph",
		xlabel="environmental uncertainty "*L"(\lambda)",
		ylabel=L"p_{\textrm{ruin}}"
	)
	plot!(
		1.5:0.05:5,
		[probruin_numeric(l, 0.65, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", alpha=0.5, label=""
	)

	plot!(
		1.5:0.05:5,
		[probruin(l, 0.8, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", label="0.80", legendtitle=L"\aleph", ls=:dash
	)
	plot!(
		1.5:0.05:5,
		[probruin_numeric(l, 0.8, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", alpha=0.5, ls=:dash, label=""
	)

	plot!(
		1.5:0.05:5,
		[probruin(l, 0.95, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", label="0.95", legendtitle=L"\aleph", ls=:dashdotdot
	)
	plot!(
		1.5:0.05:5,
		[probruin_numeric(l, 0.95, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", alpha=0.5, ls=:dashdotdot, label=""
	)

	savefig(probruin_plot, "../images/sup1_probruin.pdf")

	probruin_plot
end
  ╠═╡ =#

# ╔═╡ 9f647ffb-ae0c-4650-9c9d-2af79f324fc9
md"""
#### Figure S2
"""

# ╔═╡ ea485e6d-2cfa-48b7-b02d-d86578ca0d37
#=╠═╡
begin
	expected_plot = plot(
		1:0.01:6,
		expected_stake.(1:0.01:6),
		legend=:bottomright,
		label=L"\mathbb{E}[s^{*}(\mathbf{U})]",
		legendfontsize=7,
		ylab="stake "*L"(s)",
		#xlab="environmental stability (λ)",
		lw=2,
		ls=:dash,
		color="black"
	)
	scatter!([1],[0], label=" ", ms=0, mc=:white, msc=:white)
	plot!(
		1:0.01:6,
		optimal_fraction.(1:0.01:6) .* expected_stake.(1:0.01:6),
		label=L"s^{*}(\mathbb{E}[\mathbf{U}])",
		lw=2,
		color="black"
	)
	#=
	jensen_gap_math = plot(
		1:0.01:5,
		expected_stake.(1:0.01:5) - optimal_fraction.(1:0.01:5) .* expected_stake.(1:0.01:5),
		label="",
		ylab="Jensen gap",
		#xlab="environmental stability (λ)",
		lw=2,
		#ls=:dash,
		color="black"
	)
	=#
	Random.seed!(12345)
	jensen_gap = Vector{Vector{Float64}}()
	tries = [2, 5, 10, 25]
	
	for t in tries
		mstake = meanstake.(20000, t, 1:0.01:6)
		#smean = stakemean.(20000, t, 1:0.01:6)
		
		push!( jensen_gap, mstake .- optimal_fraction.(1:0.01:6) .* expected_stake.(1:0.01:6) )
	end

	jensen_gap_sim = plot(
		1:0.01:6,
		jensen_gap[1],
		label = "$(tries[1])",
		legendtitle=L"τ",
		legendtitlefontsize=8,
		legendfontsize=6,
		xlab = "environmental uncertainty "*L"(λ)",
		ylab = "Jensen gap",
		lw=2
		)
	counter = 2
	for p in jensen_gap[2:length(jensen_gap)]
		plot!(
		1:0.01:6,
		p,
		label = "$(tries[counter])",
		#xlab = "environmental stability (λ)",
		#ylab = "Jensen gap (ŝ - s*)",
		lw=2
		)
		global counter += 1
	end

	plot!(
		1:0.01:6,
		expected_stake.(1:0.01:6) - optimal_fraction.(1:0.01:6) .* expected_stake.(1:0.01:6),
		label="",
		#ylab="Jensen gap",
		#xlab="environmental stability (λ)",
		lw=2,
		#ls=:dash,
		color="black"
	)
	
	jenplot = plot(expected_plot, jensen_gap_sim, layout=(2,1), dpi=300)

	savefig(jenplot, "../images/sup2_jensen.pdf")

	jenplot
end
  ╠═╡ =#

# ╔═╡ 0a012538-3c0e-41eb-a5f9-36d44242eff1
md"""
#### Figure S3
"""

# ╔═╡ b34de91d-299c-4bce-a4d0-6f9c65487583
#=╠═╡
begin
	indv_plot2 = plot(
			plot(
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.65, t=2, seed=1), ylabel=L"\bar{V} \quad | \quad \tau = "*"2", ylabelfontsize=12, legend=false, title=L"\aleph = "*"0.65", titlefontsize=12 ), 
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.8, t=2, seed=1), legend=false, title=L"\aleph = "*"0.80", titlefontsize=12 ),
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.95, t=2, seed=1), legend=false, title=L"\aleph = "*"0.95", titlefontsize=12 ),
				layout=(1,3)
			),
			plot(
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.65, t=25, seed=1), ylabel=L"\bar{V} \quad | \quad \tau = "*"25", ylabelfontsize=12, xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8, legend=false ),
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.8, t=25, seed=1), legend=false, xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8 ),
				plot( plot_indv_learning([1.5, 3.0, 6.0]; n=5, aleph=0.95, t=25, seed=1), xlabel="weight of peer influence "*L"(β)", xlabelfontsize=8 ),
				layout=(1,3)
			),
			layout=(2,1),
			link=:all,
			margins=2Plots.mm
		)
	
	savefig(indv_plot2, "../images/sup3_indv2.pdf")
	
	indv_plot2
end
  ╠═╡ =#

# ╔═╡ 0d7c4a92-d250-4d85-ac7f-5d6a5503a337
md"""
#### Figure S4
"""

# ╔═╡ 189cca21-b09b-4b56-ab6c-82f3120cd31f
#=╠═╡
begin
	abmplot_sup = plot(
		run_ABM_plot(aleph=0.65, t=2),
		run_ABM_plot(aleph=0.95, t=2, legend=false),
		layout=(1,2), size=(800,400), 
		margins=3Plots.mm, xtickfontsize=5, xrotation=90
	)
	
	savefig(abmplot_sup, "../images/sup4_time_lowt.pdf")

	GC.gc()
	
	abmplot_sup
end
  ╠═╡ =#

# ╔═╡ 4b57e85c-4c65-444e-85b4-1698c589a4a4
md"""
#### Figure S5
"""

# ╔═╡ 01a70f2d-7c1d-404f-8501-82b7d4a44d10
#=╠═╡
begin
	resilience_plot = plot(
		plot(
			plot( plot_resilience(2.0, 2), legend=false, xlabel="", titlefontsize=10, ylabelfontsize=10 ),
			plot( plot_resilience(4.0, 2), ylabel="", xlabel="", titlefontsize=10, xlabelfontsize=12 ),
			plot( plot_resilience(5.0, 2), legend=false, ylabel="", xlabel="", titlefontsize=10 ),
			layout=(1,3), link=:all, size=(700, 300), margins=2Plots.mm, legend=false
		),
		plot(
			plot( plot_resilience(2.0, 5), legend=false, xlabel="", title="", titlefontsize=10, ylabelfontsize=10 ),
			plot( plot_resilience(4.0, 5), ylabel="", xlabel="", title="", titlefontsize=10, xlabelfontsize=12 ),
			plot( plot_resilience(5.0, 5), legend=false, ylabel="", xlabel="", title="", titlefontsize=10 ),
			layout=(1,3), link=:all, size=(700, 300), margins=2Plots.mm, legend=false
		),
		plot(
			plot( plot_resilience(2.0, 10), legend=false, xlabel="", title="", titlefontsize=10, ylabelfontsize=10 ),
			plot( plot_resilience(4.0, 10), legend=false, ylabel="", title="", titlefontsize=10, xlabelfontsize=12 ),
			plot( plot_resilience(5.0, 10), legend=:bottomright, ylabel="", xlabel="", title="", titlefontsize=10 ),
			layout=(1,3), link=:all, size=(700, 300), margins=2Plots.mm, legendfontsize=6, legendtitlefontsize=7
		),
		layout=(3,1), size=(600, 500), xtickfontsize=4, ytickfontsize=6
	)
	savefig(resilience_plot, "../images/sup5_resilience.pdf")
	resilience_plot
end
  ╠═╡ =#

# ╔═╡ 1114fba2-37d4-475b-93a8-7976596f8406
md"""
#### Figure S6
"""

# ╔═╡ 373db5ba-e2d9-4a7e-abed-fdab8a048e27
#=╠═╡
begin
	aleph_geo_plotter = plot(
		plot(
			aleph_geo_plot(2.0, 2, ylab=L"\bar{V}_G \quad | \quad \kappa = 2", title=L"λ = "*"2.0"),
			aleph_geo_plot(3.0, 2, title=L"λ = "*"3.0"),
			aleph_geo_plot(4.0, 2, title=L"λ = "*"4.0"),
			#link=:all,
			layout=(1,3),
			dpi=300,
			size=(700, 400),
			ylim=(0.7, 1.3)
		),
		plot(
			aleph_geo_plot(2.0, 5, ylab=L"\bar{V}_G \quad | \quad \kappa = 5"),
			aleph_geo_plot(3.0, 5),
			aleph_geo_plot(4.0, 5),
			#link=:all,
			layout=(1,3),
			dpi=300,
			size=(700, 400),
			ylim=(0.7, 1.3)
		),
		plot(
			aleph_geo_plot(2.0, 10, ylab=L"\bar{V}_G \quad | \quad \kappa = 10", xlab="wealth buffer "*L"(ℵ)", legend=true),
			aleph_geo_plot(3.0, 10, xlab="wealth buffer "*L"(ℵ)"),
			aleph_geo_plot(4.0, 10, xlab="wealth buffer "*L"(ℵ)"),
			#link=:all,
			layout=(1,3),
			dpi=300,
			size=(700, 400),
			ylim=(0.7, 1.3)
		),
		layout=(3,1),
		size=(700, 500),
		link=:all
	)
	
	savefig(aleph_geo_plotter, "../images/sup6_geo_aleph.pdf")

	aleph_geo_plotter
end
  ╠═╡ =#

# ╔═╡ 8510dc09-6bb6-4430-b332-f59de69ebabb
md"""
#### Deprecated
"""

# ╔═╡ 59cd6ed6-53cc-45f1-9a16-b62a72a00243
#=╠═╡
begin
	varplot = plot(
		[l/(l+1) for l in 1:0.01:100],
		[( (l/(l+1))*(1 - (l/(l+1))) )/( l + 2 ) for l in 1:0.01:100],
		c="black",
		ylab="Var "*L"(U)",
		xlab=L"\mathbb{E}(U)",
		legend=false,
		lw=2,
		dpi=300
	)
	
	savefig(varplot, "../images/dep1_var.pdf")
	varplot
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─cfb0a045-a40d-4019-987f-e4dc1c92074e
# ╟─1f7f1128-8eea-4695-b8cd-044d7c91890b
# ╟─e8b7171b-9810-402c-818c-61c12fb3fec7
# ╠═896ff8a7-0f61-4be3-adf4-d8e55ad16874
# ╟─ffd166fa-9d48-477a-8d16-f5e866551dfc
# ╠═b739131f-a173-4994-8143-d6c52ac07cb3
# ╟─5e5d671d-0b35-4404-b022-c2fe8bdde7c2
# ╠═aa8d557f-6436-407f-a0f7-afa823a35129
# ╟─f3fec0a1-40fc-4168-93a5-bb0648086d64
# ╠═d008b259-7a84-4f7e-99ed-e4b283775591
# ╟─adeef0c1-d6dd-4d44-9d59-866cd6b47c21
# ╠═71768709-5e37-43fc-91d9-cd47e659e1d8
# ╟─d8af2358-06f5-4190-b60c-eea14d12e5e3
# ╠═ee1b6bf1-3074-4b5d-8895-aaf034ba3f31
# ╟─1d29ca8c-12f6-4803-a6cb-7d5f27576a02
# ╠═aebf5e0d-7520-4f6e-9dff-493c4e7a5bb0
# ╟─738983ff-eaff-4628-97f3-25455458f047
# ╠═7300889d-373f-424d-9cdc-f28856cb0f23
# ╟─f37c0c19-e434-4d80-8b3a-31732df114dd
# ╠═6ea0d83d-b3df-443a-9dd0-baf314ded951
# ╟─25ae55f1-d1d8-4974-ac88-e5c4e6bfd699
# ╠═26ea2476-a0da-4f33-8e4b-6d3619db30b6
# ╟─ceacb32a-71ca-49c6-ab92-f5f33fc32b31
# ╟─a3dec80f-60df-44b1-adc5-855cf06b492a
# ╠═3a2082c4-ef09-4b53-af1b-411792729560
# ╟─9f647ffb-ae0c-4650-9c9d-2af79f324fc9
# ╠═ea485e6d-2cfa-48b7-b02d-d86578ca0d37
# ╟─0a012538-3c0e-41eb-a5f9-36d44242eff1
# ╠═b34de91d-299c-4bce-a4d0-6f9c65487583
# ╟─0d7c4a92-d250-4d85-ac7f-5d6a5503a337
# ╠═189cca21-b09b-4b56-ab6c-82f3120cd31f
# ╟─4b57e85c-4c65-444e-85b4-1698c589a4a4
# ╠═01a70f2d-7c1d-404f-8501-82b7d4a44d10
# ╟─1114fba2-37d4-475b-93a8-7976596f8406
# ╠═373db5ba-e2d9-4a7e-abed-fdab8a048e27
# ╟─8510dc09-6bb6-4430-b332-f59de69ebabb
# ╠═59cd6ed6-53cc-45f1-9a16-b62a72a00243
