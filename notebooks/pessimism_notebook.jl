### A Pluto.jl notebook ###
# v0.20.4

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

# ╔═╡ ffd166fa-9d48-477a-8d16-f5e866551dfc
md"""
#### Figure 1 - The Effects of Absorbing Boundaries
"""

# ╔═╡ b739131f-a173-4994-8143-d6c52ac07cb3
begin
	λ = 0.6:0.1:0.9
	xaxis2 = 0.001:0.001:1.0
	payplot = plot(
		xaxis2,
		g.(xaxis2, l=λ[1]),
		label = L"0.1",
		legendtitle = L"\epsilon",
		legendtitlefontsize=15,
		legendfontsize=12,
		legend=:topleft,
		lw = 3,
		xlab = L"\mathrm{stake}\ (s)",
		ylab = L"\mathrm{average\ growth\ rate\ } (\bar{V})",
		ylabelfontsize=15,
		xlabelfontsize=15,
		ylim=(0.9, 1.6),
		dpi=300,
		color=palette(:tokyo10)[1],
		grid=false,
		xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
		yticks=([1.0], [L"1.0"])
	)
	
	for l in 2:length(λ)
		plot!(
			xaxis2,
			g.(xaxis2, l=λ[l]),
			label = L"%$(round(λ[l] - 0.5, digits=1))",
			color=palette(:tokyo10)[(2*l)-1],
			lw = 3,
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
	annotate!([0.125], [1.25], [text(L"\aleph \rightarrow 1", 18, color="black")])

	plotruin3 = plot(
		xaxis2,
		g_ruin.(xaxis2, u=λ[1], ℵ=0.95),
		label = L"%$(λ[1] - 0.5)",
		legendtitle = L"\mathrm{edge}",
		legendtitlefontsize=10,
		legendfontsize=8,
		legend=false,
		lw = 3,
		ylim=(0.6, 1.6),
		dpi=300,
		color=palette(:tokyo10)[1],
		grid=false,
		xticks=false,
		yticks=([1.0], [L"1.0"])
	)
	annotate!([0.125], [1.4], [text(L"ℵ = 0.95", 13, color="black")])
	hline!([1.0], lw=1, ls=:dash, color="black", label="")
	for l in 2:length(λ)
		plot!(
			xaxis2,
			g_ruin.(xaxis2, u=λ[l], ℵ=0.95),
			label = L"%$(round(λ[l] - 0.5, digits=1))",
			color=palette(:tokyo10)[(2*l)-1],
			lw = 3,
		)
	end
	for l in 1:length(λ)
		scatter!(
			(
			collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))],
			g_ruin(collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))], u=λ[l], ℵ=0.95)
			),
			label="",
			color="black",
			marker=:circle
		)
		scatter!(
			(
			s_star(λ[l], 0.95),
			g_ruin(s_star(λ[l], 0.95), u=λ[l], ℵ=0.95)
			),
			label="",
			color="black",
			marker=:xcross
		)
	end

	plotruin2 = plot(
		xaxis2,
		g_ruin.(xaxis2, u=λ[1], ℵ=0.5),
		label = L"%$(λ[1] - 0.5)",
		legendtitle = L"\epsilon",
		legendtitlefontsize=10,
		legendfontsize=8,
		legend=false,
		lw = 3,
		ylabelfontsize=15,
		ylim=(0.6, 1.6),
		dpi=300,
		color=palette(:tokyo10)[1],
		grid=false,
		xticks=false,
		yticks=([1.0], [L"1.0"])
	)
	annotate!([0.125], [1.4], [text(L"ℵ = 0.5", 13, color="black")])
	hline!([1.0], lw=1, ls=:dash, color="black", label="")
	for l in 2:length(λ)
		plot!(
			xaxis2,
			g_ruin.(xaxis2, u=λ[l], ℵ=0.5),
			label = L"%$(round(λ[l] - 0.5, digits=1))",
			color=palette(:tokyo10)[(2*l)-1],
			lw = 3,
		)
	end
	for l in 1:length(λ)
		scatter!(
			(
			collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))],
			g_ruin(collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))], u=λ[l], ℵ=0.5)
			),
			label="",
			color="black",
			marker=:circle
		)
		scatter!(
			(
			s_star(λ[l], 0.5),
			g_ruin(s_star(λ[l], 0.5), u=λ[l], ℵ=0.5)
			),
			label="",
			color="black",
			marker=:xcross
		)
	end
	
	plotruin1 = plot(
		xaxis2,
		g_ruin.(xaxis2, u=λ[1], ℵ=0.05),
		label = L"%$(λ[1] - 0.5)",
		legendtitle = L"\epsilon",
		legendtitlefontsize=10,
		legendfontsize=8,
		legend=false,
		lw = 3,
		xlab = L"\mathrm{stake}\ (s)",
		ylabelfontsize=15,
		xlabelfontsize=15,
		ylim=(0.6, 1.6),
		dpi=300,
		color=palette(:tokyo10)[1],
		grid=false,
		xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
		yticks=([1.0], [L"1.0"])	
	)
	annotate!([0.125], [1.4], [text(L"ℵ = 0.05", 13, color="black")])
	hline!([1.0], lw=1, ls=:dash, color="black", label="")
	for l in 2:length(λ)
		plot!(
			xaxis2,
			g_ruin.(xaxis2, u=λ[l], ℵ=0.05),
			label = L"%$(round(λ[l] - 0.5, digits=1))",	
			color=palette(:tokyo10)[(2*l)-1],
			lw = 3,
		)
	end
	for l in 1:length(λ)
		scatter!(
			(
			collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))],
			g_ruin(collect(xaxis2)[argmax(g.(xaxis2, l=λ[l]))], u=λ[l], ℵ=0.05)
			),
			label="",
			color="black",
			marker=:circle
		)
		scatter!(
			(
			s_star(λ[l], 0.05),
			g_ruin(s_star(λ[l], 0.05), u=λ[l], ℵ=0.05)
			),
			label="",
			color="black",
			marker=:xcross
		)
	end
	lens!(
		[0.0, 0.075], 
		[0.975, 1.05], 
		inset = (1, bbox(0.5, 0.0, 0.4, 0.4)),
		#xticks=false,
		xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
		yticks=([1.0], [L"1.0"])
	)

	splot = plot(
		plotruin3, plotruin2, plotruin1,
		layout=(3,1),
		link=:all,
		dpi=300,
		title="",
		xlabelfontsize=15,
		grid=false
	)

	full_payplot = plot(
		payplot,
		splot,
		layout=(1,2),
		size=(900,500),
		margin=3Plots.mm,
		dpi=300,
	)
	
	savefig(full_payplot, "../images/fig1_kelly.pdf")

	splots_barrier = nothing
	GC.gc()
	
	full_payplot
end

# ╔═╡ fe9e037d-d510-4026-83e0-005a61fa005a
md"""
#### Figure 2 - A Tale of Three Brothers
"""

# ╔═╡ 21c738fa-0e6c-4592-a332-b0a564a23fa2
begin
	modelium = initialize_pessimistic_learning(N=3, u = 0.65, soc_h=0.25, aleph = 0.15, n=2, T=100, mu_sens = 1.0, selection=true, seed=47)

	svecs_young = [a.s_vec_young for a in allagents(modelium)|>collect]
	svecs = [a.s_vec for a in allagents(modelium)|>collect]
	pvecs = [a.payoff_vec for a in allagents(modelium)|>collect]
	ruin1 = (filter(x -> x > 0.0, pvecs[2])|>length) + 1
	ruin2 = (filter(x -> x > 0.0, pvecs[3])|>length) + 1
	
	svecs_youngplot = plot(
		svecs_young[1],
		ylabel=L"\mathrm{mean\ juvenile\ stake}",
		xlabel=L"\mathrm{child\ timeline}",
		ylim=(0.0, 1.0),
		xticks=([1, 17], [L"0", L"\tau"]),
		xtickfontsize=12,
		yticks=([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], [L"%$a" for a in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]),
		label="",
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(svecs_young[2], color=palette(:lipariS)[6], lw=2, label="")
	plot!(svecs_young[3], color=palette(:lipariS)[8], lw=2, label="")
	hline!([2*0.65 - 1], ls=:dash, color=:black, label=L"\mathrm{kelly\ stake}")
	
	svecsplot = plot(
		svecs[1],
		ylabel=L"\mathrm{mean\ adult\ stake}",
		xticks=false, 
		yticks=([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], [L"%$a" for a in [0, 0.1, 0.2, 0.3, 0.4, 0.5]]),
		label=L"\mathrm{Juan}", 
		legendfontsize=7,
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(svecs[2][1:ruin1], label=L"\mathrm{Roberto}", color=palette(:lipariS)[6], lw=2)
	plot!(svecs[3][1:ruin2], label=L"\mathrm{Gabriel}", color=palette(:lipariS)[8], lw=2)
	hline!([s_star(0.65, 0.15)], ls=:dash, color=:black, label=L"\mathrm{optimal\ stake}")
	scatter!(
		[ruin1, ruin2],
		[svecs[2][ruin1], svecs[3][ruin2]],
		label=L"\mathrm{ruin\ event}",
		markershape=:xcross,
		color="black"
	)
	
	pvecsplot = plot(
		pvecs[1], 
		xticks=([0, 101], [L"0", L"T"]),
		yticks=([0, 1, 2, 3, 4, 5, 6], [L"%$a" for a in [0, 1, 2, 3, 4, 5, 6]]),
		xlabel=L"\mathrm{adult\ timeline}",
		ylabel=L"\mathrm{log\ wealth}",
		legend=false, 
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(pvecs[2], lw=2, color=palette(:lipariS)[6])
	plot!(pvecs[3], lw=2, color=palette(:lipariS)[8])
	scatter!(
		[ruin1, ruin2],
		[pvecs[2][ruin1], pvecs[3][ruin2]],
		markershape=:xcross,
		color="black"
	)

	broplot = plot(
		svecsplot,
		pvecsplot,
		layout=(2,1),
		dpi=300
	)

	full_broplot = plot(
		svecs_youngplot,
		broplot,
		layout=(1,2)
	)

	savefig(full_broplot, "../images/fig2_bros.pdf")

	full_broplot
end

# ╔═╡ f030487a-a4ed-461d-9405-fa69155f6e9e
begin
	sens = 0.1
	soc_h = 0.15
	
	#aleph = 0.05
	model = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.05, soc_h = soc_h, seed=87212106, steps=2)
	model2 = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.05, soc_h = soc_h, sens = sens, seed=4779634, steps=3)

	model3 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.05, soc_h = soc_h, seed=93549759, steps=2)
	model4 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.05, soc_h = soc_h, sens = sens, seed=64842295, steps=3)

	model5 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.05, soc_h = soc_h, seed=49156430, steps=2)
	model6 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.05, soc_h = soc_h, sens = sens, seed=1282556077, steps=3)

	#aleph = 0.5
	model7 = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.5, soc_h = soc_h, seed=78078065, steps=2)
	model8 = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.5, soc_h = soc_h, sens = sens, seed=57217575, steps=3)

	model9 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.5, soc_h = soc_h, seed=88901522, steps=2)
	model10 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.5, soc_h = soc_h, sens = sens, seed=62601494, steps=3)

	model11 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.5, soc_h = soc_h, seed=10384399, steps=2)
	model12 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.5, soc_h = soc_h, sens = sens, seed=5206168, steps=3)

	#aleph = 0.95
	model13 = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.95, soc_h = soc_h,  seed=47124811, steps=2)
	model14 = initialize_pessimistic_learning(N=5000, u = 0.55, aleph = 0.95, soc_h = soc_h, sens = sens, seed=92889375, steps=3)

	model15 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.95, soc_h = soc_h, seed=67138817, steps=2)
	model16 = initialize_pessimistic_learning(N=5000, u = 0.65, aleph = 0.95, soc_h = soc_h, sens = sens, seed=92714379, steps=3)

	model17 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.95, soc_h = soc_h, seed=41528844, steps=2)
	model18 = initialize_pessimistic_learning(N=5000, u = 0.75, aleph = 0.95, soc_h = soc_h, sens = sens, seed=41095108, steps=3)

md"""
#### Figure 3 - The Population Effects of Social Trauma
"""
end

# ╔═╡ 41f7d1c6-2a94-4fdf-a23e-6e93502cddd7
begin
	s(x) = x ≤ 0.5 ? 0 : 2*x - 1
	
	p1 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model)|>collect]), 
		bins=10, alpha=0.5, legend=false, xlim=(0,0.65), 
		yticks=(0:1000:4000, [L"%$a" for a in 0:1000:4000]),
		title=L"\epsilon = 0.05", ylab=L"ℵ = 0.05", 
		xticks=false, grid=false, ytickfontsize=8,
		color=palette(:tokyo10)[1]
	)
	histogram!( 
		[a.s_mean for a in allagents(model2)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.55, 0.05)], lw=1, color="black", ls=:dash, label="")
	
	p2 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model3)|>collect]), 
		bins=20, alpha=0.5, legend=false, xlim=(0,0.65), 
		color=palette(:tokyo10)[1], 
		title=L"\epsilon = 0.15", xticks=false, yticks=false, grid=false 
	)
	histogram!( 
		[a.s_mean for a in allagents(model4)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.65, 0.05)], lw=1, color="black", ls=:dash, label="")

	p3 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model5)|>collect]), bins=10, alpha=0.5, legend=false, xlim=(0,0.65), 
		title=L"\epsilon = 0.25", color=palette(:tokyo10)[1],
		xticks=false, yticks=false, grid=false 
	)
	histogram!( 
		[a.s_mean for a in allagents(model6)|>collect], 
		bins=25, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.75, 0.05)], lw=1, color="black", ls=:dash, label="")

	p4 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model7)|>collect]), 
		bins=10, alpha=0.5, legend=false, 
		xlim=(0,0.65), ylab=L"ℵ = 0.5", 
		yticks=(0:1000:4000, [L"%$a" for a in 0:1000:4000]),
		xticks=false, grid=false, ytickfontsize=8, xtickfontsize=8,
		color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model8)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] )
	vline!([s_star(0.55, 0.5)], lw=1, color="black", ls=:dash, label="")

	p5 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model9)|>collect]), 
		bins=20, alpha=0.5, legend=false, xlim=(0,0.65), 
		xticks=false, yticks=false, grid=false, color=palette(:tokyo10)[1]
	)
	histogram!( 
		[a.s_mean for a in allagents(model10)|>collect], 
		bins=15, alpha=0.5, color=palette(:tokyo10)[7]
	)
	vline!([s_star(0.65, 0.5)], lw=1, color="black", ls=:dash, label="")

	p6 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model11)|>collect]), 
		bins=10, alpha=0.5, legend=false, xlim=(0,0.65), 
		xticks=false, yticks=false, grid=false, color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model12)|>collect], 
		bins=25, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.75, 0.5)], lw=1, color="black", ls=:dash, label="")

	p7 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model13)|>collect]), 
		bins=10, alpha=0.5, legend=:topright, 
		legendfontsize=8, label=L"\mathrm{after\ juvenile\ stage}", 
		xlim=(0,0.65), ylab=L"ℵ = 0.95", grid=false, 
		yticks=(0:1000:4000, [L"%$a" for a in 0:1000:4000]),
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		xtickfontsize=8, ytickfontsize=8,
		color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model14)|>collect], 
		bins=10, alpha=0.5, label=L"\mathrm{lifetime\ average}",
		color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.55, 0.95)], lw=1, color="black", ls=:dash, label="")

	p8 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model15)|>collect]), 
		bins=10, alpha=0.5, xlim=(0,0.65), yticks=false, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		grid=false, xtickfontsize=8, xlab=L"\mathrm{stake\ } (s)", xlabelfontsize=20, legend=false,
		color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model16)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] )
	vline!([s_star(0.65, 0.95)], lw=1, color="black", ls=:dash, label="")

	p9 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model17)|>collect]), 
		bins=10, alpha=0.5, legend=false, legendfontsize=7, 
		xlim=(0,0.65), yticks=false, grid=false, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		xtickfontsize=8, color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model18)|>collect], 
		bins=15, alpha=0.5, label="lifetime average",
		color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.75, 0.95)], lw=1, color="black", ls=:dash, label="")

	dev_plot = plot(
		p1, p2, p3, p4, p5, p6, p7, p8, p9,
		layout=(3,3), link=:all, size=(650, 400)
	)

	savefig(dev_plot, "../images/fig3_development.pdf")
	
	dev_plot
end

# ╔═╡ 5e5d671d-0b35-4404-b022-c2fe8bdde7c2
begin
	dat = CSV.read("../data/analysis_0.csv", DataFrame)
	
	dat0 = dat[
		dat.time .== 2500 .&&
		#dat.N .== 1000 .&& 
		dat.T .== 100 .&& 
		dat.n .== 10 .&&
		dat.t .== 15
		,:]
	
	mdat0_1 = dat0[dat0.u .== 0.55, :]
	mdat0_2 = dat0[dat0.u .== 0.6, :]
	mdat0_3 = dat0[dat0.u .== 0.65, :]
	mdat0_4 = dat0[dat0.u .== 0.75, :]

	grouped_soch = combine(
		groupby(mdat0_1, :aleph), 
	    :soc_h_median => mean => :mean_soc_h_median,
		:soc_h_lerror => mean => :mean_soc_h_lerror,
		:soc_h_herror => mean => :mean_soc_h_herror,
		:sens_median => mean => :mean_sens_median,
		:sens_lerror => mean => :mean_sens_lerror,
		:sens_herror => mean => :mean_sens_herror,
		:s_median => mean => :mean_s_median,
		:s_lerror => mean => :mean_s_lerror,
		:s_herror => mean => :mean_s_herror,
		:s_young_median => mean => :mean_s_young_median,
		:s_young_lerror => mean => :mean_s_young_lerror,
		:s_young_herror => mean => :mean_s_young_herror,
		:Vbar => mean => :mean_Vbar,
	)
	grouped_soch2 = combine(
		groupby(mdat0_2, :aleph), 
	    :soc_h_median => mean => :mean_soc_h_median,
		:soc_h_lerror => mean => :mean_soc_h_lerror,
		:soc_h_herror => mean => :mean_soc_h_herror,
		:sens_median => mean => :mean_sens_median,
		:sens_lerror => mean => :mean_sens_lerror,
		:sens_herror => mean => :mean_sens_herror,
		:s_median => mean => :mean_s_median,
		:s_lerror => mean => :mean_s_lerror,
		:s_herror => mean => :mean_s_herror,
		:s_young_median => mean => :mean_s_young_median,
		:s_young_lerror => mean => :mean_s_young_lerror,
		:s_young_herror => mean => :mean_s_young_herror,
		:Vbar => mean => :mean_Vbar,
	)
	grouped_soch3 = combine(
		groupby(mdat0_3, :aleph), 
	    :soc_h_median => mean => :mean_soc_h_median,
		:soc_h_lerror => mean => :mean_soc_h_lerror,
		:soc_h_herror => mean => :mean_soc_h_herror,
		:sens_median => mean => :mean_sens_median,
		:sens_lerror => mean => :mean_sens_lerror,
		:sens_herror => mean => :mean_sens_herror,
		:s_median => mean => :mean_s_median,
		:s_lerror => mean => :mean_s_lerror,
		:s_herror => mean => :mean_s_herror,
		:s_young_median => mean => :mean_s_young_median,
		:s_young_lerror => mean => :mean_s_young_lerror,
		:s_young_herror => mean => :mean_s_young_herror,
		:Vbar => mean => :mean_Vbar,
	)
	grouped_soch4 = combine(
		groupby(mdat0_4, :aleph), 
	    :soc_h_median => mean => :mean_soc_h_median,
		:soc_h_lerror => mean => :mean_soc_h_lerror,
		:soc_h_herror => mean => :mean_soc_h_herror,
		:sens_median => mean => :mean_sens_median,
		:sens_lerror => mean => :mean_sens_lerror,
		:sens_herror => mean => :mean_sens_herror,
		:s_median => mean => :mean_s_median,
		:s_lerror => mean => :mean_s_lerror,
		:s_herror => mean => :mean_s_herror,
		:s_young_median => mean => :mean_s_young_median,
		:s_young_lerror => mean => :mean_s_young_lerror,
		:s_young_herror => mean => :mean_s_young_herror,
		:Vbar => mean => :mean_Vbar,
	)
	
	md"""
	#### Figure 4 - Baseline Scenario
	"""
end

# ╔═╡ 8a2c7c0c-e59d-4727-ab06-c0367298fb8b
begin
	peer_inf = plot(
	    grouped_soch4.aleph, grouped_soch4.mean_soc_h_median, 
		ribbon = (
		grouped_soch4.mean_soc_h_median .- grouped_soch4.mean_soc_h_lerror, 
		grouped_soch4.mean_soc_h_herror .- grouped_soch4.mean_soc_h_median
		),
		fillalpha=0.2,
		xlabelfontsize = 15,
		color=palette(:managua10)[8],
	    ylabel = L"\mathrm{peer\ reinforcement\ } (\beta)",
		ylabelfontsize = 11,
	    title = "",
		legend = false,
		legendtitle = L"\bar{u}",
	    label = L"0.6",
		lw=2,
		ylim=(0,1),
		grid=false,
		xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
	)
	plot!(
	    grouped_soch3.aleph, grouped_soch3.mean_soc_h_median, 
		ribbon = (
		grouped_soch3.mean_soc_h_median .- grouped_soch3.mean_soc_h_lerror, 
		grouped_soch3.mean_soc_h_herror .- grouped_soch3.mean_soc_h_median
		),
		color=palette(:managua10)[6],
	    fillalpha=0.2, lw=2,
		ylim=(0,1)
	)
	plot!(
	    grouped_soch2.aleph, grouped_soch2.mean_soc_h_median, 
		ribbon = (
		grouped_soch2.mean_soc_h_median .- grouped_soch2.mean_soc_h_lerror, 
		grouped_soch2.mean_soc_h_herror .- grouped_soch2.mean_soc_h_median
		),
		color=palette(:managua10)[3],
	    fillalpha=0.2, lw=2,
		ylim=(0,1)
	)
	plot!(
	    grouped_soch.aleph, grouped_soch.mean_soc_h_median, 
		ribbon = (
		grouped_soch.mean_soc_h_median .- grouped_soch.mean_soc_h_lerror, 
		grouped_soch.mean_soc_h_herror .- grouped_soch.mean_soc_h_median
		),
		color=palette(:managua10)[1],
	    fillalpha=0.2, lw=2,
		ylim=(0,1)
	)

	sensplot = plot(
	    grouped_soch4.aleph, grouped_soch4.mean_sens_median, 
		ribbon = (
		grouped_soch4.mean_sens_median .- grouped_soch4.mean_sens_lerror, 
		grouped_soch4.mean_sens_herror .- grouped_soch4.mean_sens_median
		),
		fillalpha=0.2,
		xlabelfontsize = 15,
		color=palette(:managua10)[8],
		ylabel = L"\mathrm{sensitivity\ } (\delta)",
		ylabelfontsize = 11,
		title = "",
		legend = false,
		legendtitle = L"\lambda",
	    label = L"2",
		lw=2,
		ylim=(0,1),
		grid=false,
		xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
		)
	plot!(
	    grouped_soch3.aleph, grouped_soch3.mean_sens_median, 
		ribbon = (
		grouped_soch3.mean_sens_median .- grouped_soch3.mean_sens_lerror, 
		grouped_soch3.mean_sens_herror .- grouped_soch3.mean_sens_median
		),
		color=palette(:managua10)[6],
	    fillalpha=0.2, label = L"3", lw=2,
		ylim=(0,1)
	)
	plot!(
	    grouped_soch2.aleph, grouped_soch2.mean_sens_median, 
		ribbon = (
		grouped_soch2.mean_sens_median .- grouped_soch2.mean_sens_lerror, 
		grouped_soch2.mean_sens_herror .- grouped_soch2.mean_sens_median
		),
		color=palette(:managua10)[3],
	    fillalpha=0.2, label = L"3", lw=2,
		ylim=(0,1)
	)
	plot!(
	    grouped_soch.aleph, grouped_soch.mean_sens_median, 
		ribbon = (
		grouped_soch.mean_sens_median .- grouped_soch.mean_sens_lerror, 
		grouped_soch.mean_sens_herror .- grouped_soch.mean_sens_median
		),
		color=palette(:managua10)[1],
	    fillalpha=0.2, label = L"6", lw=2,
		ylim=(0,1)
	)

	stakeplot = plot(
		1:0,
		xlabelfontsize = 15,
		ylabel = L"\mathrm{stake\ } (s)",
		ylabelfontsize = 15,
		title = "",
		legend = :top,
		legendtitle = L"\epsilon",
		label = "",
		ylim=(-0.05,0.5),
		grid=false,
		xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"]),
		size=(500, 700),
		dpi=300,
		margins=2Plots.mm
	)
	plot!(
	    grouped_soch4.aleph, grouped_soch4.mean_s_median, 
		ribbon = (
		grouped_soch4.mean_s_median .- grouped_soch4.mean_s_lerror, grouped_soch4.mean_s_herror .- grouped_soch4.mean_s_median
		),
		alpha=1.0,
	    fillalpha=0.15, label = L"0.25", lw=2,
		color=palette(:managua10)[8],
	)
	plot!(
		grouped_soch4.aleph, s_star.(0.75, grouped_soch4.aleph), 
		fillalpha=0.2, label = "", lw=2, ls=:dash,
		color=palette(:managua10)[8], alpha=0.5
		)
	plot!(
	    grouped_soch3.aleph, grouped_soch3.mean_s_median, 
		ribbon = (
		grouped_soch3.mean_s_median .- grouped_soch3.mean_s_lerror, grouped_soch3.mean_s_herror .- grouped_soch3.mean_s_median
		),
		alpha=1.0,
	    fillalpha=0.15, label = L"0.15", lw=2,
		color=palette(:managua10)[6],
	)
	plot!(
		grouped_soch3.aleph, s_star.(0.65, grouped_soch3.aleph), 
		fillalpha=0.2, label = "", lw=2, ls=:dash,
		color=palette(:managua10)[6], alpha=0.5
		)
	plot!(
	    grouped_soch2.aleph, grouped_soch2.mean_s_median, 
		ribbon = (
		grouped_soch2.mean_s_median .- grouped_soch2.mean_s_lerror, grouped_soch2.mean_s_herror .- grouped_soch2.mean_s_median
		),
	    fillalpha=0.2, label = L"0.10", lw=2,
		color=palette(:managua10)[3],
	)
	plot!(
	    grouped_soch2.aleph, s_star.(0.6, grouped_soch2.aleph), 
		alpha=0.5,
	    fillalpha=0.2, label = "", lw=2, ls=:dash,
		color=palette(:managua10)[3],
	)
	plot!(
		grouped_soch.aleph, grouped_soch.mean_s_median, 
		ribbon = (
		grouped_soch.mean_s_median .- grouped_soch.mean_s_lerror, grouped_soch.mean_s_herror .- grouped_soch.mean_s_median
		),
		fillalpha=0.35, color=palette(:managua10)[1], label = L"0.05",
		lw=2,
	)
	plot!(
	    grouped_soch.aleph, s_star.(0.55, grouped_soch.aleph), 
		alpha=0.75,
	    fillalpha=0.2, label = "", lw=2, ls=:dash,
		color=palette(:managua10)[1],
	)
	annotate!([-0.15], [-0.125], [text(L"\mathrm{wealth \ buffer\ } (\aleph)", "black", 17)])
	
	peersens = plot(
		peer_inf,
		sensplot,
		layout=(2,1),
		size=(500, 700),
		dpi=300,
		margins=2Plots.mm
	)

	indplot = plot(
		peersens,
		stakeplot,
		layout=(1,2),
		size=(600, 400),
		bottom_margin=10Plots.mm,
		top_margin=4Plots.mm
	)

	savefig(indplot, "../images/fig4_peers.pdf")

	indplot
	
end

# ╔═╡ f3fec0a1-40fc-4168-93a5-bb0648086d64
md"""
#### Figure 5 - Elder Influence Enabled
"""

# ╔═╡ aef1aaeb-5cd1-4121-81ab-490a24a89af0
begin
	function plot_elder_portfolio(dat; pal=:managua10, envshift=3000, ushift=0.85)
		
		dat3 = dat[
				dat.time .== 2500 .&&
				#dat.N .== 1000 .&& 
				dat.T .== 100 .&&
				dat.n .== 10 .&&
				dat.m .== 10 .&&
				dat.envshift .== envshift .&&
				dat.u_shift .== 0.85
				,:]
		
		mdat3_1 = dat3[dat3.u .== 0.55, :]
		mdat3_2 = dat3[dat3.u .== 0.6, :]
		mdat3_3 = dat3[dat3.u .== 0.65, :]
		mdat3_4 = dat3[dat3.u .== 0.75, :]
	
		grouped_socv = combine(
			groupby(mdat3_1, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_lerror => mean => :mean_s_lerror,
			:s_herror => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv2 = combine(
			groupby(mdat3_2, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_lerror => mean => :mean_s_lerror,
			:s_herror => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv3 = combine(
			groupby(mdat3_3, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_lerror => mean => :mean_s_lerror,
			:s_herror => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv4 = combine(
			groupby(mdat3_4, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_lerror => mean => :mean_s_lerror,
			:s_herror => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
	
		eld_inf = plot(
		    grouped_socv4.aleph, grouped_socv4.mean_soc_v_median, 
			ribbon = (
			grouped_socv4.mean_soc_v_median .- grouped_socv4.mean_soc_v_lerror, 
			grouped_socv4.mean_soc_v_herror .- grouped_socv4.mean_soc_v_median
			),
			fillalpha=0.2,
			xlabelfontsize = 15,
			color=palette(pal)[8],
		    ylabel = L"\mathrm{elder\ influence\ } (\alpha)",
			ylabelfontsize = 12,
		    title = "",
			legend = false,
			legendtitle = L"\bar{u}",
		    label = L"0.6",
			lw=2,
			ylim=(0,1),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_soc_v_median, 
			ribbon = (
			grouped_socv3.mean_soc_v_median .- grouped_socv3.mean_soc_v_lerror, 
			grouped_socv3.mean_soc_v_herror .- grouped_socv3.mean_soc_v_median
			),
			color=palette(pal)[6],
		    fillalpha=0.2, label = L"0.675", lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_soc_v_median, 
			ribbon = (
			grouped_socv2.mean_soc_v_median .- grouped_socv2.mean_soc_v_lerror, 
			grouped_socv2.mean_soc_v_herror .- grouped_socv2.mean_soc_v_median
			),
			color=palette(pal)[3],
		    fillalpha=0.2, label = L"0.75", lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv.aleph, grouped_socv.mean_soc_v_median, 
			ribbon = (
			grouped_socv.mean_soc_v_median .- grouped_socv.mean_soc_v_lerror, 
			grouped_socv.mean_soc_v_herror .- grouped_socv.mean_soc_v_median
			),
			color=palette(pal)[1],
		    fillalpha=0.2, label = L"0.75", lw=2,
			ylim=(0,1)
		)
	
		eld_sens = plot(
		    grouped_socv4.aleph, grouped_socv4.mean_sens_median, 
			ribbon = (
			grouped_socv4.mean_sens_median .- grouped_socv4.mean_sens_lerror, 
			grouped_socv4.mean_sens_herror .- grouped_socv4.mean_sens_median
			),
			fillalpha=0.2,
			xlabelfontsize = 15,
			color=palette(pal)[8],
			ylabel = L"\mathrm{sensitivity\ } (\delta)",
			ylabelfontsize = 12,
			title = "",
			legend = false,
			legendtitle = L"\lambda",
		    label = L"2",
			lw=2,
			ylim=(0,1),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
			)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_sens_median, 
			ribbon = (
			grouped_socv3.mean_sens_median .- grouped_socv3.mean_sens_lerror, 
			grouped_socv3.mean_sens_herror .- grouped_socv3.mean_sens_median
			),
			color=palette(pal)[6],
		    fillalpha=0.2, label = L"3", lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_sens_median, 
			ribbon = (
			grouped_socv2.mean_sens_median .- grouped_socv2.mean_sens_lerror, 
			grouped_socv2.mean_sens_herror .- grouped_socv2.mean_sens_median
			),
			color=palette(pal)[3],
		    fillalpha=0.2, label = L"3", lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv.aleph, grouped_socv.mean_sens_median, 
			ribbon = (
			grouped_socv.mean_sens_median .- grouped_socv.mean_sens_lerror, 
			grouped_socv.mean_sens_herror .- grouped_socv.mean_sens_median
			),
			color=palette(pal)[1],
		    fillalpha=0.2, label = L"6", lw=2,
			ylim=(0,1)
		)
	
		eld_peerinf = plot(
		    grouped_socv4.aleph, grouped_socv4.mean_soc_h_median, 
			ribbon = (
			grouped_socv4.mean_soc_h_median .- grouped_socv4.mean_soc_h_lerror, grouped_socv4.mean_soc_h_herror .- grouped_socv4.mean_soc_h_median
			),
			fillalpha=0.2,
			xlabelfontsize = 15,
			color=palette(pal)[8],
		    ylabel = L"\mathrm{peer\ reinforcement\ } (\beta)",
			ylabelfontsize = 12,
		    title = "",
			legend = false,
			legendtitle = L"\bar{u}",
		    label = L"0.6",
			lw=2,
			ylim=(0,1),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_soc_h_median, 
			ribbon = (
			grouped_socv3.mean_soc_h_median .- grouped_socv3.mean_soc_h_lerror, grouped_socv3.mean_soc_h_herror .- grouped_socv3.mean_soc_h_median
			),
			color=palette(pal)[6],
		    fillalpha=0.2, lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_soc_h_median, 
			ribbon = (
			grouped_socv2.mean_soc_h_median .- grouped_socv2.mean_soc_h_lerror, grouped_socv2.mean_soc_h_herror .- grouped_socv2.mean_soc_h_median
			),
			color=palette(pal)[3],
		    fillalpha=0.2, lw=2,
			ylim=(0,1)
		)
		plot!(
		    grouped_socv.aleph, grouped_socv.mean_soc_h_median, 
			ribbon = (
			grouped_socv.mean_soc_h_median .- grouped_socv.mean_soc_h_lerror, grouped_socv.mean_soc_h_herror .- grouped_socv.mean_soc_h_median
			),
			color=palette(pal)[1],
		    fillalpha=0.2, lw=2,
			ylim=(0,1)
		)
	
		pb_plot = plot(
			grouped_socv4.aleph, grouped_socv4.freq_pb,
			fillalpha=0.2,
			xlabelfontsize = 15,
			color=palette(pal)[8],
			ylabel = L"\mathrm{payoff\ bias}",
			ylabelfontsize = 12,
			title = "",
			legend = :top,
			legendtitle = L"\epsilon",
			legendtitlefontsize = 10,
			label = L"0.25",
			lw=2,
			ylim=(0,1),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0", L"0.25", L"0.5", L"0.75", L"1"])
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.freq_pb, 
		    fillalpha=0.2, label = L"0.15", lw=2,
			ylim=(0,1), color=palette(pal)[6]
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.freq_pb, 
		    fillalpha=0.2, label = L"0.10", lw=2,
			ylim=(0,1), color=palette(pal)[3]
		)
		plot!(
		    grouped_socv.aleph, grouped_socv.freq_pb, 
		    fillalpha=0.2, label = L"0.05", lw=2,
			ylim=(0,1), color=palette(pal)[1]
		)
		annotate!([-0.15], [-0.25], [text(L"\mathrm{wealth \ buffer\ } (\aleph)", "black", 17)])
		
		eld_stake = plot(
			1:0,
			xlabelfontsize = 15,
			ylabel = L"\mathrm{stake\ } (s)",
			ylabelfontsize = 12,
			title = "",
			legend = false,
			legendtitle = L"\epsilon",
			legendtitlefontsize=10,
			label = "",
			ylim=(-0.05,0.5),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25], [L"0", L"0.25"]),
			size=(500, 700),
			dpi=300,
		)
		plot!(
		    grouped_socv4.aleph, grouped_socv4.mean_s_median, 
			ribbon = (
			grouped_socv4.mean_s_median .- grouped_socv4.mean_s_lerror, 
			grouped_socv4.mean_s_herror .- grouped_socv4.mean_s_median
			),
		    fillalpha=0.2, label = L"0.25", lw=2,
			color=palette(pal)[8],
		)
		if envshift == 1
			plot!(
				grouped_socv4.aleph, s_star.((0.75+ushift)/2, grouped_socv4.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[8], alpha=0.5
			)
		else
			plot!(
				grouped_socv4.aleph, s_star.(0.75, grouped_socv4.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[8], alpha=0.5
			)
		end
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_s_median, 
			ribbon = (
			grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			),
		    fillalpha=0.2, label = L"0.15", lw=2,
			color=palette(pal)[6],
		)
		if envshift == 1
			plot!(
				grouped_socv3.aleph, s_star.((0.65+ushift)/2, grouped_socv3.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[6], alpha=0.5
			)
		else
			plot!(
				grouped_socv3.aleph, s_star.(0.65, grouped_socv3.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[6], alpha=0.5
			)
		end
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_s_median, 
			ribbon = (
			grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			),
		    fillalpha=0.2, label = L"0.10", lw=2,
			color=palette(pal)[3],
		)
		if envshift == 1
			plot!(
			    grouped_socv2.aleph, s_star.((0.6+ushift)/2, grouped_socv2.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[3], alpha=0.5
			)
		else
			plot!(
			    grouped_socv2.aleph, s_star.(0.6, grouped_socv2.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[3], alpha=0.5
			)
		end
		plot!(
			grouped_socv.aleph, grouped_socv.mean_s_median, 
			ribbon = (
			grouped_socv.mean_s_median .- grouped_socv.mean_s_lerror, 
			grouped_socv.mean_s_herror .- grouped_socv.mean_s_median
			),
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=2,
		)
		if envshift == 1
			plot!(
			    grouped_socv.aleph, s_star.((0.55+ushift)/2, grouped_socv.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[1], alpha=0.75
			)
		else
			plot!(
			    grouped_socv.aleph, s_star.(0.55, grouped_socv.aleph), 
			    fillalpha=0.2, label = "", lw=2, ls=:dash,
				color=palette(pal)[1], alpha=0.75
			)
		end
	
		peersens = plot(
			eld_peerinf,
			eld_sens,
			layout=(2,1),
			size=(500, 700),
			dpi=300,
			#margins=2Plots.mm
		)
	
		eldstake = plot(
			eld_inf,
			eld_stake,
			pb_plot,
			layout=(3,1),
			size=(500, 700),
			dpi=300
		)
		
		plot(
			peersens,
			eldstake,
			layout=(1,2),
			size=(600, 600),
			bottom_margin=6Plots.mm
		)
	end
	
	eldplot = plot_elder_portfolio( CSV.read("../data/analysis_1.csv", DataFrame) )

	savefig(eldplot, "../images/fig5_elders.pdf")

	eldplot
	
end

# ╔═╡ 326013e3-69fa-4ba2-9381-ccfadbf00613
begin
	modelio = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.55,
		aleph = 0.05,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 75548897,
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

	modelio2 = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.55,
		aleph = 0.5,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 755488978,
	)

	adata2, mdata2 = run!(
		modelio2, 
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

	modelio3 = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.55,
		aleph = 0.95,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 755488978,
	)

	adata3, mdata3 = run!(
		modelio3, 
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

	modelio4 = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.65,
		aleph = 0.05,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 7554889,
	)

	adata4, mdata4 = run!(
		modelio4, 
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

	modelio5 = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.65,
		aleph = 0.5,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 7554889,
	)

	adata5, mdata5 = run!(
		modelio5, 
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
	
	modelio6 = initialize_pessimistic_learning(
		N = 1000,
		T = 100,
		m = 10,
		n = 10,
		t = 15,
		sens = 0.0,
		soc_v = 0.0,
		#SINGLE POP
		u = 0.65,
		aleph = 0.95,
		envshift = 3000,
		peg_lambda = false,
		peg_aleph = true,
		u_shift = 0.65,
		aleph_shift = 0.05,
		#MIXED
		mixed_aleph1 = 0.05,
		mixed_aleph2 = 0.95,
		mixed_u1 = 0.55,
		mixed_u2 = 0.65,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.01,
		strategies = "UB&PB",
		mixed = false,
		mixed_freq = 0.5,
		parochial = false,
		periodic = true,
		selection = true,
		b_coeff = 1.0,
		seed = 7554889,
	)

	adata6, mdata6 = run!(
		modelio6, 
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

	md"""
	#### Figure 6 - Tails of risk taking
	"""
end

# ╔═╡ 6ac78a6a-aa27-4ea0-8e49-e469186dfce8
begin
	function plot_tails(dat; pal=:managua10)
		
		dat3 = dat[
				dat.time .== 2500 .&&
				dat.T .== 100 .&&
				dat.n .== 10 .&&
				dat.m .== 10 .&&
				dat.envshift .== 3000 .&&
				dat.u_shift .== 0.65
				,:]
		
		mdat3_1 = dat3[dat3.u .== 0.55, :]
		mdat3_2 = dat3[dat3.u .== 0.6, :]
		mdat3_3 = dat3[dat3.u .== 0.65, :]
		mdat3_4 = dat3[dat3.u .== 0.75, :]
	
		grouped_socv = combine(
			groupby(mdat3_1, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv2 = combine(
			groupby(mdat3_2, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv3 = combine(
			groupby(mdat3_3, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		grouped_socv4 = combine(
			groupby(mdat3_4, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb
		)
		
		plot(
			1:0,
			xlabelfontsize = 18,
			ylabel = L"\mathrm{stake\ } (s)",
			xlabel = L"\mathrm{wealth\ buffer\ } (\aleph)",
			ylabelfontsize = 18,
			title = "",
			legend = (0.2, 0.9),
			legendtitle = L"\epsilon",
			legendtitlefontsize=15,
			legendfontsize=12,
			label = "",
			ylim=(-0.05,0.5),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25], [L"0", L"0.25"]),
			size=(500, 300),
			dpi=300,
			margins=3Plots.mm
		)
		plot!(
		    grouped_socv4.aleph, grouped_socv4.mean_s_median, 
			ribbon = (
			grouped_socv4.mean_s_median .- grouped_socv4.mean_s_lerror, 
			grouped_socv4.mean_s_herror .- grouped_socv4.mean_s_median
			),
		    fillalpha=0.2, label = L"0.25", lw=3,
			color=palette(pal)[8],
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_s_median, 
			ribbon = (
			grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			),
		    fillalpha=0.2, label = L"0.15", lw=3,
			color=palette(pal)[6],
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_s_median, 
			ribbon = (
			grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			),
		    fillalpha=0.2, label = L"0.10", lw=3,
			color=palette(pal)[3],
		)
		plot!(
			grouped_socv.aleph, grouped_socv.mean_s_median, 
			ribbon = (
			grouped_socv.mean_s_median .- grouped_socv.mean_s_lerror, 
			grouped_socv.mean_s_herror .- grouped_socv.mean_s_median
			),
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=3,
		)
	end

	tailsplot = plot_tails(CSV.read("../data/analysis_1.csv", DataFrame))

	bins = 0.0:0.02:0.15
	bins2 = 0.0:0.05:0.4
	
	stakehist1 = histogram(
				[a.s_mean for a in allagents(modelio)],
				bins=bins,
				xlim=(-0.01,0.15),
				ylim=(0,1000),
				color=palette(:romaO10)[1],
				legend=false,
				yticks=(0:200:1000, [L"%$a" for a in 0:200:1000]),
				xtickfontsize=6,
				xticks=(0.0:0.05:0.15, [L"%$a" for a in 0.0:0.05:0.15]),
				ylabelfontsize = 18,
				ylabel=L"\epsilon = 0.05",
				title=L"\aleph = 0.05",
				alpha=0.85
			)
	vline!([s_star(0.55, 0.05)], lw=2, color="black", ls=:dash)

	stakehist2 = histogram(
				[a.s_mean for a in allagents(modelio2)],
				xlim=(-0.01,0.15),
				ylim=(0,1000),
				bins=bins,
				color=palette(:romaO10)[3],
				legend=false,
				xticks=(0.0:0.05:0.15, [L"%$a" for a in 0.0:0.05:0.15]),
				xtickfontsize=6,
				yticks=false,
				title=L"\aleph = 0.5",
				alpha=0.85
			)
	vline!([s_star(0.55, 0.5)], lw=2, color="black", ls=:dash)

	stakehist3 = histogram(
				[a.s_mean for a in allagents(modelio3)],
				xlim=(-0.01,0.15),
				ylim=(0,1000),
				color=palette(:romaO10)[6],
				legend=false,
				xticks=(0.0:0.05:0.15, [L"%$a" for a in 0.0:0.05:0.15]),
				xtickfontsize=6,
				yticks=false,
				bins=bins,
				title=L"\aleph = 0.95",
				alpha=0.85
			)
	vline!([s_star(0.55, 0.95)], lw=2, color="black", ls=:dash)

	stakehist4 = histogram(
				[a.s_mean for a in allagents(modelio4)],
				bins=bins2,
				color=palette(:romaO10)[1],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,1000),
				xticks=(0.0:0.1:0.4, [L"%$a" for a in 0.0:0.1:0.4]),
				xtickfontsize=6,
				yticks=(0:200:1000, [L"%$a" for a in 0:200:1000]),
				ylabelfontsize = 18,
				ylabel=L"\epsilon = 0.15",
				title=" ",
				alpha=0.85
			)
	vline!([s_star(0.65, 0.05)], lw=2, color="black", ls=:dash)

	stakehist5 = histogram(
				[a.s_mean for a in allagents(modelio5)],
				bins=bins2,
				color=palette(:romaO10)[3],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,1000),
				xticks=(0.0:0.1:0.4, [L"%$a" for a in 0.0:0.1:0.4]),
				xtickfontsize=6,
				yticks=false,
				xlabel=L"\mathrm{mean\ lifetime\ stake\ } (\bar{s}\ )",
				xlabelfontsize=18,
				title=" ",
				alpha=0.85
			)
	vline!([s_star(0.65, 0.5)], lw=2, color="black", ls=:dash)

	stakehist6= histogram(
				[a.s_mean for a in allagents(modelio6)],
				color=palette(:romaO10)[6],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,1000),
				xticks=(0.0:0.1:0.4, [L"%$a" for a in 0.0:0.1:0.4]),
				xtickfontsize=6,
				yticks=false,
				bins=bins2,
				title=" ",
				alpha=0.85
			)
	vline!([s_star(0.65, 0.95)], lw=2, color="black", ls=:dash)
	
	stakedist = plot(
		plot(
			stakehist1,
			stakehist2,
			stakehist3,
			layout=(1,3), link=:all, grid=false, size=(600,300)
		),
		plot(
			stakehist4,
			stakehist5,
			stakehist6,
			layout=(1,3), link=:all, grid=false, size=(600,400)
		),
		layout=(2,1), margin=3Plots.mm
	)
	
	staketails = plot(stakedist, tailsplot, layout=(1,2), size=(800, 400))

	savefig(staketails, "../images/fig6_staketails.pdf")

	staketails
	
end

# ╔═╡ 223e84bb-404e-4c66-bc3d-242d42c00bc9
md"""
#### Figure 7 - Environmental change
"""

# ╔═╡ 324d8681-82ae-42c0-8e23-91b7ab4a2cd4
begin
	envshift_plot = plot_elder_portfolio( CSV.read("../data/analysis_1.csv", DataFrame), envshift=1 )
	
	savefig(envshift_plot, "../images/fig7_envshift.pdf")
	
	envshift_plot
end

# ╔═╡ 88e18457-ae1b-4de6-958b-df4f407d0b98
md"""
#### Figure 8 - Life Trajectories
"""

# ╔═╡ 43b13c02-86ac-4620-b62b-a81e98a173a4
begin
	function plot_life_trajectory(model; c=palette(:romaO10)[1], up=false)
		s_vecs = [a.s_vec for a in allagents(model)]
		transposed = [getindex.(s_vecs, i) for i in 1:length(s_vecs[1])]
		mean_trajectory = mean.(transposed)

		inc = [1.0]

		if !up
			d = -0.02
		else
			d = 0.03
		end
		
		for i in 1:length(mean_trajectory)
			if i > 1
				push!(inc, inc[i-1]*(mean_trajectory[i]/mean_trajectory[i-1]))
			end
		end
		plot( 
			inc, 
			lw=2, color=c, 
			label=L"%$(model.aleph)", 
			legendtitle=L"\aleph", 
			ylim=(0.7, 1.1),
			xlabel=L"\mathrm{adult\ lifetime}",
			xlabelfontsize=18,
			ylabel=L"\mathrm{proportional\ stake\ change}",
			ylabelfontsize=13,
			yticks=(
			[0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6], 
			[L"%$a" for a in [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]]
			),
			xticks=(
			[0, 25, 50, 75, 100], 
			[L"%$a" for a in [0, 25, 50, 75, 100]]
			),
			grid=false
		)
		annotate!(
			[90], [inc[90]+d], 
			[text(L"\bar{s} = %$(round(mean(mean_trajectory), digits=2))", color=c)]
		)
	end

	function plot_life_trajectory!(model; c=palette(:romaO10)[1], up=false)
		s_vecs = [a.s_vec for a in allagents(model)]
		transposed = [getindex.(s_vecs, i) for i in 1:length(s_vecs[1])]
		mean_trajectory = mean.(transposed)

		inc = [1.0]

		if !up
			d = -0.02
		else
			d = 0.03
		end
		
		for i in 1:length(mean_trajectory)
			if i > 1
				push!(inc, inc[i-1]*(mean_trajectory[i]/mean_trajectory[i-1]))
			end
		end
		plot!( 
			inc, 
			lw=2, color=c, 
			label=L"%$(model.aleph)" 
		)
		annotate!(
			[90], [inc[90]+d], 
			[text(L"\bar{s} = %$(round(mean(mean_trajectory), digits=2))", 14, color=c)]
		)
	end

	function plot_increase(dat; pal=:managua10)
		dat3 = dat[
			dat.time .== 2500 .&&
			dat.T .== 100 .&&
			dat.n .== 10 .&&
			dat.m .== 10 .&&
			dat.envshift .== 3000 .&&
			dat.u_shift .== 0.65
			,:]
		
		mdat3_1 = dat3[dat3.u .== 0.55, :]
		mdat3_2 = dat3[dat3.u .== 0.6, :]
		mdat3_3 = dat3[dat3.u .== 0.65, :]
		mdat3_4 = dat3[dat3.u .== 0.75, :]
	
		grouped_socv = combine(
			groupby(mdat3_1, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb,
			:sbar => mean => :sbar,
			:mean_increment => mean => :mean_increment
		)
		grouped_socv2 = combine(
			groupby(mdat3_2, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb,
			:sbar => mean => :sbar,
			:mean_increment => mean => :mean_increment
		)
		grouped_socv3 = combine(
			groupby(mdat3_3, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb,
			:sbar => mean => :sbar,
			:mean_increment => mean => :mean_increment
		)
		grouped_socv4 = combine(
			groupby(mdat3_4, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_v_median => mean => :mean_soc_v_median,
			:soc_v_lerror => mean => :mean_soc_v_lerror,
			:soc_v_herror => mean => :mean_soc_v_herror,
			:sens_median => mean => :mean_sens_median,
			:sens_lerror => mean => :mean_sens_lerror,
			:sens_herror => mean => :mean_sens_herror,
			:s_median => mean => :mean_s_median,
			:s_ltail => mean => :mean_s_lerror,
			:s_htail => mean => :mean_s_herror,
			:s_young_median => mean => :mean_s_young_median,
			:s_young_lerror => mean => :mean_s_young_lerror,
			:s_young_herror => mean => :mean_s_young_herror,
			:Vbar => mean => :mean_Vbar,
			:freq_pb => mean => :freq_pb,
			:sbar => mean => :sbar,
			:mean_increment => mean => :mean_increment
		)

		increment = plot(
			1:0,
			xlabelfontsize = 18,
			ylabel = L"\mathrm{mean\ proportional\ stake\ change}",
			xlabel = L"\mathrm{wealth\ buffer\ } (\aleph)",
			ylabelfontsize = 10,
			title = "",
			legend = false,
			legendtitle = L"\mathrm{edge}",
			legendtitlefontsize=15,
			legendfontsize=12,
			label = "",
			ylim=(0.4, 1.25),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.5, 1.0], [L"0.5", L"1.0"]),
			size=(500, 300),
			dpi=300,
			#margins=5Plots.mm
		)
		plot!(
		    grouped_socv4.aleph, grouped_socv4.mean_increment, 
			#ribbon = (
			#grouped_socv4.mean_s_median .- grouped_socv4.mean_s_lerror, 
			#grouped_socv4.mean_s_herror .- grouped_socv4.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.25", lw=2,
			color=palette(pal)[8],
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_increment, 
			#ribbon = (
			#grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			#grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.15", lw=2,
			color=palette(pal)[6],
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_increment, 
			#ribbon = (
			#grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			#grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.10", lw=2,
			color=palette(pal)[3],
		)
		plot!(
			grouped_socv.aleph, grouped_socv.mean_increment, 
			#ribbon = (
			#grouped_socv.mean_s_median .- grouped_socv.mean_s_lerror, 
			#grouped_socv.mean_s_herror .- grouped_socv.mean_s_median
			#),
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=2,
		)
		hline!([1.0], ls=:dash, color=:black)

		sbar = plot(
			1:0,
			xlabelfontsize = 18,
			ylabel = L"\mathrm{mean\ lifetime\ stake\ } (\bar{s}\ )",
			#xlabel = L"\mathrm{wealth\ buffer\ } (\aleph)",
			ylabelfontsize = 12,
			title = "",
			legend = :topleft,
			legendtitle = L"\epsilon",
			legendtitlefontsize=12,
			legendfontsize=8,
			label = "",
			ylim=(-0.05, 0.5),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25], [L"0.0", L"0.25"]),
			size=(500, 300),
			dpi=300,
			#margins=5Plots.mm
		)
		plot!(
			grouped_socv.aleph, grouped_socv.sbar, 
			#ribbon = (
			#grouped_socv.mean_s_median .- grouped_socv.mean_s_lerror, 
			#grouped_socv.mean_s_herror .- grouped_socv.mean_s_median
			#),
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=2,
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.sbar, 
			#ribbon = (
			#grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			#grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.10", lw=2,
			color=palette(pal)[3],
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.sbar, 
			#ribbon = (
			#grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			#grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.15", lw=2,
			color=palette(pal)[6],
		)
		plot!(
		    grouped_socv4.aleph, grouped_socv4.sbar, 
			#ribbon = (
			#grouped_socv4.mean_s_median .- grouped_socv4.mean_s_lerror, 
			#grouped_socv4.mean_s_herror .- grouped_socv4.mean_s_median
			#),
		    fillalpha=0.2, label = L"0.25", lw=2,
			color=palette(pal)[8],
		)

		plot(sbar, increment, layout=(2,1), size=(600, 600))
		
	end
	
	lifeplot = plot_life_trajectory(modelio4, up=true)
	plot_life_trajectory!(modelio5, c=palette(:romaO10)[3], up=true)
	plot_life_trajectory!(modelio6, c=palette(:romaO10)[6])

	full_lifeplot = plot(
		lifeplot, 
		plot_increase(CSV.read("../data/analysis_1.csv", DataFrame)),
		layout=(1,2),
		size=(600, 500)
	)
	
	savefig(full_lifeplot, "../images/fig8_lifeplot.pdf")

	full_lifeplot
end

# ╔═╡ 80f6d4ac-3036-4e7a-9ab7-31d0644ec6ea
begin
	dat4 = CSV.read("../data/analysis_2.csv", DataFrame)
		
	dat5 = dat4[
		dat4.time .== 2500 .&&
		dat4.T .== 100 .&&
		dat4.mixed_u1 .== 0.75 .&&
		dat4.mixed_freq .== 0.5 .&&
		dat4.mixed_aleph1 .== 0.95 .&&
		dat4.n .== 15 .&&
		dat4.m .== 15
			,:]
	
	mdat5_1 = dat5[dat5.mixed_u2 .== 0.55, :]
	mdat5_2 = dat5[dat5.mixed_u2 .== 0.6, :]
	mdat5_3 = dat5[dat5.mixed_u2 .== 0.65, :]
	mdat5_4 = dat5[dat5.mixed_u2 .== 0.75, :]
	
	
	g0u75_g1u55 = combine(
		groupby(mdat5_1, :mixed_aleph2), 
	    :soc_h_median_g0 => mean => :mean_soc_h_median_g0,
		:soc_h_lerror_g0 => mean => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => mean => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => mean => :mean_soc_v_median_g0,
		:soc_v_lerror_g0 => mean => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => mean => :mean_soc_v_herror_g0,
		:sens_median_g0 => mean => :mean_sens_median_g0,
		:sens_lerror_g0 => mean => :mean_sens_lerror_g0,
		:sens_herror_g0 => mean => :mean_sens_herror_g0,
		:s_median_g0 => mean => :mean_s_median_g0,
		:s_lerror_g0 => mean => :mean_s_lerror_g0,
		:s_herror_g0 => mean => :mean_s_herror_g0,
		:s_young_median_g0 => mean => :mean_s_young_median_g0,
		:s_young_lerror_g0 => mean => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => mean => :mean_s_young_herror_g0,
		:Vbar_g0 => mean => :mean_Vbar_g0,
		:freq_pb_g0 => mean => :freq_pb_g0,
		:freq_parochial_g0 => mean => :freq_parochial_g0,
		:soc_h_median_g1 => mean => :mean_soc_h_median_g1,
		:soc_h_lerror_g1 => mean => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => mean => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => mean => :mean_soc_v_median_g1,
		:soc_v_lerror_g1 => mean => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => mean => :mean_soc_v_herror_g1,
		:sens_median_g1 => mean => :mean_sens_median_g1,
		:sens_lerror_g1 => mean => :mean_sens_lerror_g1,
		:sens_herror_g1 => mean => :mean_sens_herror_g1,
		:s_median_g1 => mean => :mean_s_median_g1,
		:s_lerror_g1 => mean => :mean_s_lerror_g1,
		:s_herror_g1 => mean => :mean_s_herror_g1,
		:s_young_median_g1 => mean => :mean_s_young_median_g1,
		:s_young_lerror_g1 => mean => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => mean => :mean_s_young_herror_g1,
		:Vbar_g1 => mean => :mean_Vbar_g1,
		:freq_pb_g1 => mean => :freq_pb_g1,
		:freq_parochial_g1 => mean => :freq_parochial_g1,
		:mixed_aleph1 => mean => :mixed_aleph1,
		:mixed_u1 => mean => :mixed_u1,
		:mixed_u2 => mean => :mixed_u2
	)
	
	g0u75_g1u6 = combine(
		groupby(mdat5_2, :mixed_aleph2), 
	    :soc_h_median_g0 => mean => :mean_soc_h_median_g0,
		:soc_h_lerror_g0 => mean => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => mean => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => mean => :mean_soc_v_median_g0,
		:soc_v_lerror_g0 => mean => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => mean => :mean_soc_v_herror_g0,
		:sens_median_g0 => mean => :mean_sens_median_g0,
		:sens_lerror_g0 => mean => :mean_sens_lerror_g0,
		:sens_herror_g0 => mean => :mean_sens_herror_g0,
		:s_median_g0 => mean => :mean_s_median_g0,
		:s_lerror_g0 => mean => :mean_s_lerror_g0,
		:s_herror_g0 => mean => :mean_s_herror_g0,
		:s_young_median_g0 => mean => :mean_s_young_median_g0,
		:s_young_lerror_g0 => mean => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => mean => :mean_s_young_herror_g0,
		:Vbar_g0 => mean => :mean_Vbar_g0,
		:freq_pb_g0 => mean => :freq_pb_g0,
		:freq_parochial_g0 => mean => :freq_parochial_g0,
		:soc_h_median_g1 => mean => :mean_soc_h_median_g1,
		:soc_h_lerror_g1 => mean => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => mean => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => mean => :mean_soc_v_median_g1,
		:soc_v_lerror_g1 => mean => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => mean => :mean_soc_v_herror_g1,
		:sens_median_g1 => mean => :mean_sens_median_g1,
		:sens_lerror_g1 => mean => :mean_sens_lerror_g1,
		:sens_herror_g1 => mean => :mean_sens_herror_g1,
		:s_median_g1 => mean => :mean_s_median_g1,
		:s_lerror_g1 => mean => :mean_s_lerror_g1,
		:s_herror_g1 => mean => :mean_s_herror_g1,
		:s_young_median_g1 => mean => :mean_s_young_median_g1,
		:s_young_lerror_g1 => mean => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => mean => :mean_s_young_herror_g1,
		:Vbar_g1 => mean => :mean_Vbar_g1,
		:freq_pb_g1 => mean => :freq_pb_g1,
		:freq_parochial_g1 => mean => :freq_parochial_g1,
		:mixed_aleph1 => mean => :mixed_aleph1,
		:mixed_u1 => mean => :mixed_u1,
		:mixed_u2 => mean => :mixed_u2
	)
	
	g0u75_g1u65 = combine(
		groupby(mdat5_3, :mixed_aleph2), 
	    :soc_h_median_g0 => mean => :mean_soc_h_median_g0,
		:soc_h_lerror_g0 => mean => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => mean => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => mean => :mean_soc_v_median_g0,
		:soc_v_lerror_g0 => mean => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => mean => :mean_soc_v_herror_g0,
		:sens_median_g0 => mean => :mean_sens_median_g0,
		:sens_lerror_g0 => mean => :mean_sens_lerror_g0,
		:sens_herror_g0 => mean => :mean_sens_herror_g0,
		:s_median_g0 => mean => :mean_s_median_g0,
		:s_lerror_g0 => mean => :mean_s_lerror_g0,
		:s_herror_g0 => mean => :mean_s_herror_g0,
		:s_young_median_g0 => mean => :mean_s_young_median_g0,
		:s_young_lerror_g0 => mean => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => mean => :mean_s_young_herror_g0,
		:Vbar_g0 => mean => :mean_Vbar_g0,
		:freq_pb_g0 => mean => :freq_pb_g0,
		:freq_parochial_g0 => mean => :freq_parochial_g0,
		:soc_h_median_g1 => mean => :mean_soc_h_median_g1,
		:soc_h_lerror_g1 => mean => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => mean => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => mean => :mean_soc_v_median_g1,
		:soc_v_lerror_g1 => mean => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => mean => :mean_soc_v_herror_g1,
		:sens_median_g1 => mean => :mean_sens_median_g1,
		:sens_lerror_g1 => mean => :mean_sens_lerror_g1,
		:sens_herror_g1 => mean => :mean_sens_herror_g1,
		:s_median_g1 => mean => :mean_s_median_g1,
		:s_lerror_g1 => mean => :mean_s_lerror_g1,
		:s_herror_g1 => mean => :mean_s_herror_g1,
		:s_young_median_g1 => mean => :mean_s_young_median_g1,
		:s_young_lerror_g1 => mean => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => mean => :mean_s_young_herror_g1,
		:Vbar_g1 => mean => :mean_Vbar_g1,
		:freq_pb_g1 => mean => :freq_pb_g1,
		:freq_parochial_g1 => mean => :freq_parochial_g1,
		:mixed_aleph1 => mean => :mixed_aleph1,
		:mixed_u1 => mean => :mixed_u1,
		:mixed_u2 => mean => :mixed_u2
	)
	g0u75_g1u75 = combine(
		groupby(mdat5_4, :mixed_aleph2), 
	    :soc_h_median_g0 => mean => :mean_soc_h_median_g0,
		:soc_h_lerror_g0 => mean => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => mean => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => mean => :mean_soc_v_median_g0,
		:soc_v_lerror_g0 => mean => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => mean => :mean_soc_v_herror_g0,
		:sens_median_g0 => mean => :mean_sens_median_g0,
		:sens_lerror_g0 => mean => :mean_sens_lerror_g0,
		:sens_herror_g0 => mean => :mean_sens_herror_g0,
		:s_median_g0 => mean => :mean_s_median_g0,
		:s_lerror_g0 => mean => :mean_s_lerror_g0,
		:s_herror_g0 => mean => :mean_s_herror_g0,
		:s_young_median_g0 => mean => :mean_s_young_median_g0,
		:s_young_lerror_g0 => mean => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => mean => :mean_s_young_herror_g0,
		:Vbar_g0 => mean => :mean_Vbar_g0,
		:freq_pb_g0 => mean => :freq_pb_g0,
		:freq_parochial_g0 => mean => :freq_parochial_g0,
		:soc_h_median_g1 => mean => :mean_soc_h_median_g1,
		:soc_h_lerror_g1 => mean => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => mean => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => mean => :mean_soc_v_median_g1,
		:soc_v_lerror_g1 => mean => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => mean => :mean_soc_v_herror_g1,
		:sens_median_g1 => mean => :mean_sens_median_g1,
		:sens_lerror_g1 => mean => :mean_sens_lerror_g1,
		:sens_herror_g1 => mean => :mean_sens_herror_g1,
		:s_median_g1 => mean => :mean_s_median_g1,
		:s_lerror_g1 => mean => :mean_s_lerror_g1,
		:s_herror_g1 => mean => :mean_s_herror_g1,
		:s_young_median_g1 => mean => :mean_s_young_median_g1,
		:s_young_lerror_g1 => mean => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => mean => :mean_s_young_herror_g1,
		:Vbar_g1 => mean => :mean_Vbar_g1,
		:freq_pb_g1 => mean => :freq_pb_g1,
		:freq_parochial_g1 => mean => :freq_parochial_g1,
		:mixed_aleph1 => mean => :mixed_aleph1,
		:mixed_u1 => mean => :mixed_u1,
		:mixed_u2 => mean => :mixed_u2
	)

md"""
#### Figure 9 - Mixed Populations
"""
	
end

# ╔═╡ 55c3f3f2-2e71-4fe3-b7a0-cdd9b0422816
begin
	function full_risk_profile(dat)
		eld = plot(
			dat.mixed_aleph2,
			dat.mean_soc_v_median_g0,
			ribbon=(
				dat.mean_soc_v_median_g0 .- dat.mean_soc_v_lerror_g0,
				dat.mean_soc_v_herror_g0 .- dat.mean_soc_v_median_g0
			),
			ylim=(-0,1),
			xlim=(0,1),
			lw=2,
			legend=false,
			label=L"0",
			c=palette(:roma)[1],
			ylab=L"\mathrm{elder\ influence\ } (\alpha)",
			xlab="",
			title="",
			ylabelfontsize=12,
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25, 0.5, 0.75, 1.0], 
			[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
			),
			grid=false,
		)
		plot!(
			dat.mixed_aleph2,
			dat.mean_soc_v_median_g1,
			ribbon=(
				dat.mean_soc_v_median_g1 .- dat.mean_soc_v_lerror_g1,
				dat.mean_soc_v_herror_g1 .- dat.mean_soc_v_median_g1
			),
			c=palette(:roma)[200],
			label=L"1",
			lw=2,
		)

		paroch = plot(
			dat.mixed_aleph2,
			dat.freq_parochial_g0,
			ylim=(-0.1,1),
			xlim=(0,1),
			lw=2,
			legend=:bottomright,
			label=L"c_0",
			c=palette(:roma)[1],
			ylab=L"\mathrm{parochialism}",
			xlab="",
			title="",
			ylabelfontsize=12,
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25, 0.5, 0.75, 1.0], 
			[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
			),
			grid=false,
			margins=3Plots.mm
		)
		plot!(
			dat.mixed_aleph2,
			dat.freq_parochial_g1,
			c=palette(:roma)[200],
			label=L"c_1",
			lw=2,
			#alpha=0.5,
			#ls=:dash
		)
		annotate!([1.2], [-0.5], text(L"\mathrm{wealth\ buffer\ of\ } c_1\ (\aleph_{c_1})", color="black", 17))

		pb = plot(
			dat.mixed_aleph2,
			dat.freq_pb_g0,
			ylim=(0,1),
			xlim=(0,1),
			lw=2,
			legend=false,
			legendtitle=L"\mathrm{group}",
			label=L"0",
			c=palette(:roma)[1],
			ylab=L"\mathrm{payoff\ bias}",
			xlab="",
			title="",
			ylabelfontsize=12,
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25, 0.5, 0.75, 1.0], 
			[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
			),
			grid=false,
			margins=3Plots.mm
		)
		plot!(
			dat.mixed_aleph2,
			dat.freq_pb_g1,
			c=palette(:roma)[200],
			label="",
			lw=2,
			#alpha=0.5,
			#ls=:dash
		)
	
		peer = plot(
			dat.mixed_aleph2,
			dat.mean_soc_h_median_g0,
			ribbon=(
				dat.mean_soc_h_median_g0 .- dat.mean_soc_h_lerror_g0,
				dat.mean_soc_h_herror_g0 .- dat.mean_soc_h_median_g0
			),
			ylim=(0,1),
			xlim=(0,1),
			lw=2,
			legend=false,
			label=L"g_0",
			c=palette(:roma)[1],
			ylab=L"\mathrm{peer\ reinforcement\ } (\beta)",
			xlab="",
			title="",
			ylabelfontsize=10,
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25, 0.5, 0.75, 1.0], 
			[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
			),
			grid=false,
		)
		plot!(
			dat.mixed_aleph2,
			dat.mean_soc_h_median_g1,
			ribbon=(
				dat.mean_soc_h_median_g1 .- dat.mean_soc_h_lerror_g1,
				dat.mean_soc_h_herror_g1 .- dat.mean_soc_h_median_g1
			),
			c=palette(:roma)[200],
			label=L"g_1",
			lw=2,
		)
	
		sens = plot(
			dat.mixed_aleph2,
			dat.mean_sens_median_g0,
			ribbon=(
				dat.mean_sens_median_g0 .- dat.mean_sens_lerror_g0,
				dat.mean_sens_herror_g0 .- dat.mean_sens_median_g0
			),
			c=palette(:roma)[1],
			legend=false,
			label=L"g_0",
			lw=2,
			ylab=L"\mathrm{sensitivity\ } (\delta)",
			xlab="",
			title="",
			ylabelfontsize=12,
			xlim=(0,1),
			ylim=(0,1),
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25, 0.5, 0.75, 1.0], 
			[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
			),
			grid=false,
		)
		plot!(
			dat.mixed_aleph2,
			dat.mean_sens_median_g1,
			ribbon=(
				dat.mean_sens_median_g1 .- dat.mean_sens_lerror_g1,
				dat.mean_sens_herror_g1 .- dat.mean_sens_median_g1
			),
			c=palette(:roma)[200],
			label=L"g_1",
			lw=2,
		)
	
		stake = plot(
			dat.mixed_aleph2,
			dat.mean_s_median_g0,
			ribbon=(
				dat.mean_s_median_g0 .- dat.mean_s_lerror_g0,
				dat.mean_s_herror_g0 .- dat.mean_s_median_g0
			),
			c=palette(:roma)[1],
			legend=false,
			label=L"g_0",
			lw=2,
			ylab=L"\mathrm{stake\ } (s)",
			xlab="",
			title="",
			ylabelfontsize=12,
			xlim=(0,1),
			ylim=(0,0.5),
			xticks=(
			[0.2, 0.4, 0.6, 0.8], 
			[L"0.2", L"0.4", L"0.6", L"0.8"]
			),
			yticks=(
			[0.0, 0.25], 
			[L"0", L"0.25"]
			),
			grid=false,
		)
		plot!(
			dat.mixed_aleph2,
			s_star.(dat.mixed_u1, dat.mixed_aleph1),
			c=palette(:roma)[1],
			label="",
			lw=2,
			alpha=0.5,
			ls=:dash
		)
		plot!(
			dat.mixed_aleph2,
			dat.mean_s_median_g1,
			ribbon=(
				dat.mean_s_median_g1 .- dat.mean_s_lerror_g1,
				dat.mean_s_herror_g1 .- dat.mean_s_median_g1
			),
			c=palette(:roma)[200],
			label=L"g_1",
			lw=2,
		)
		plot!(
			dat.mixed_aleph2,
			s_star.(dat.mixed_u2, dat.mixed_aleph2),
			c=palette(:roma)[200],
			label="",
			lw=2,
			alpha=0.5,
			ls=:dash
		)
		
		plot(
			plot(
				plot(
					peer, sens, paroch, layout = (3,1)
				),
				plot(
					eld, stake, pb, layout = (3,1)
				),
				layout = (1, 2)
			),
			top_margin=2Plots.mm,
			bottom_margin=9Plots.mm,
			right_margin=3Plots.mm,
			size=(600,600),
			dpi=300
		)
	end
	
	mixedplot = full_risk_profile(g0u75_g1u75)

	savefig(mixedplot, "../images/fig9_mixed.pdf")

	mixedplot
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
begin
	probruin_plot = plot(
		0.5:0.001:1.0,
		[probruin(l, 0.1, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", label=L"0.1", legendtitle=L"\aleph",
		xticks=([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [L"0.0", L"0.1", L"0.2", L"0.3", L"0.4", L"0.5"]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.5", L"0.75", L"1.0"]),
		xlabel=L"\mathrm{environmental\ edge\ } (\epsilon)",
		ylabel=L"p_{\textrm{ruin}}",
		grid=false
	)
	plot!(
		0.5:0.001:1.0,
		[probruin_numeric(l, 0.1, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", alpha=0.5, label=""
	)

	plot!(
		0.5:0.001:1.0,
		[probruin(l, 0.5, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", label=L"0.5", legendtitle=L"\aleph", ls=:dash
	)
	plot!(
		0.5:0.001:1.0,
		[probruin_numeric(l, 0.5, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", alpha=0.5, ls=:dash, label=""
	)
	
	plot!(
		0.5:0.001:1.0,
		[probruin(l, 0.95, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", label=L"0.95", legendtitle=L"\aleph", ls=:dashdotdot
	)
	plot!(
		0.5:0.001:1.0,
		[probruin_numeric(l, 0.95, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", alpha=0.5, ls=:dashdotdot, label=""
	)
	#=
	plot!(
		1.5:0.05:5,
		[probruin(l, 0.95, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", label="0.95", legendtitle=L"\aleph", ls=:dot
	)
	plot!(
		1.5:0.05:5,
		[probruin_numeric(l, 0.95, 0.5) for l in 1.5:0.05:5],
		lw=2, c="black", alpha=0.5, ls=:dot, label=""
	)
	=#

	savefig(probruin_plot, "../images/sup1_probruin.pdf")

	probruin_plot
end

# ╔═╡ c56c822a-ded8-4550-978c-2a037389a85a
md"""
#### Figure S2
"""

# ╔═╡ 2fbddc1c-6d2c-4195-9366-0e7a483450fc
begin
	trauma_plot = plot(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.0),
		color=palette(:managua10)[10],
		legend=(0.92, 0.915),
		legendtitle=L"\delta",
		legendtitlefontsize=9,
		legendfontsize=9,
		label=L"0",
		xticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.5", L"0.75", L"1.0"]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.5", L"0.75", L"1.0"]),
		ylab=L"\mathrm{social\ trauma\ response}",
		xlab=L"\mathrm{normalized\ mean\ stake\ difference\ from\ observed\ ruined\ peer}",
		lw=2,
		dpi=300,
		grid=false
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.1),
		color=palette(:managua10)[9],
		label=L"0.1",
		lw=2
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.2),
		color=palette(:managua10)[8],
		label=L"0.2",
		lw=2
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.5),
		color=palette(:managua10)[6],
		label=L"0.5",
		lw=2
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.8),
		color=palette(:managua10)[4],
		label=L"0.8",
		lw=2
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 0.9),
		color=palette(:managua10)[3],
		label=L"0.9",
		lw=2
	)
	plot!(
		0.0:0.0001:1.0,
		trauma.(0.0:0.0001:1.0, 1.0),
		color=palette(:managua10)[2],
		label=L"1",
		lw=2
	)

	savefig(trauma_plot, "../images/sup2_trauma.pdf")

	trauma_plot
end

# ╔═╡ 077c7aec-998a-4295-9426-85d254d251f7
md"""
#### Figure S3
"""

# ╔═╡ 3dc40146-3fdb-4d00-a288-571c47a45041
begin
	timeplot1 = plot(
		mdata.time[2:end],
		mdata.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		legend=false,
		ylabel=L"\epsilon = 0.05",
		xlabelfontsize=20,
		grid=false,
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		xticks=([0, 2500], [L"0", L"2500"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		title=L"\mathrm{\aleph = 0.05}"
	)
	plot!(
		mdata.time[2:end],
		mdata.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata.time[2:end],
		mdata.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata.time[2:end],
		mdata.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot2 = plot(
		mdata2.time[2:end],
		mdata2.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		legend=false,
		ylim=(0.0, 1.0),
		xlabel="",
		xlabelfontsize=25,
		grid=false,
		xticks=([0, 2500], [L"0", L"2500"]),
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		title=L"\mathrm{\aleph = 0.5}"
	)
	plot!(
		mdata2.time[2:end],
		mdata2.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata2.time[2:end],
		mdata2.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata2.time[2:end],
		mdata2.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot3 = plot(
		mdata3.time[2:end],
		mdata3.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		legend=false,
		#xlabel=L"t",
		xlabelfontsize=20,
		grid=false,
		xticks=([0, 2500], [L"0", L"2500"]),
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		title=L"\mathrm{\aleph = 0.95}"
		)
	plot!(
		mdata3.time[2:end],
		mdata3.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata3.time[2:end],
		mdata3.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata3.time[2:end],
		mdata3.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot4 = plot(
		mdata4.time[2:end],
		mdata4.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		legend=false,
		ylabel=L"\epsilon = 0.15",
		xlabelfontsize=20,
		grid=false,
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		xticks=([0, 2500], [L"0", L"2500"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		#title=L"\mathrm{\aleph = 0.05}"
	)
	plot!(
		mdata4.time[2:end],
		mdata4.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata4.time[2:end],
		mdata4.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata4.time[2:end],
		mdata4.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot5 = plot(
		mdata5.time[2:end],
		mdata5.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		legend=false,
		ylim=(0.0, 1.0),
		xlabel=L"\mathrm{generations}",
		xlabelfontsize=25,
		grid=false,
		xticks=([0, 2500], [L"0", L"2500"]),
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		#title=L"\mathrm{\aleph = 0.5}"
	)
	plot!(
		mdata5.time[2:end],
		mdata5.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata5.time[2:end],
		mdata5.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata5.time[2:end],
		mdata5.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot6 = plot(
		mdata6.time[2:end],
		mdata6.soc_v_median[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		legend=(0.55, 1.0),
		#xlabel=L"t",
		xlabelfontsize=20,
		grid=false,
		xticks=([0, 2500], [L"0", L"2500"]),
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		ytickfontsize=10,
		xtickfontsize=10,
		dpi=300,
		#title=L"\mathrm{\aleph = 0.95}"
		)
	plot!(
		mdata6.time[2:end],
		mdata6.soc_h_median[2:end],
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		mdata6.time[2:end],
		mdata6.sens_median[2:end],
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata6.time[2:end],
		mdata6.freq_pb[2:end],
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplots1 = plot(
		timeplot1,
		timeplot2, 
		timeplot3,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)

	timeplots2 = plot(
		timeplot4,
		timeplot5, 
		timeplot6,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)

	timeplots_full = plot(
		timeplots1,
		timeplots2,
		layout=(2,1),
		size=(700,600),
		dpi=300
	)
	
	savefig(timeplots_full, "../images/sup3_time.pdf")

	timeplots_full
end

# ╔═╡ 656b3a41-1ceb-4637-af43-81b3e4d25832
md"""
#### Figure S4
"""

# ╔═╡ 87c21ae5-79ec-4481-9c80-4e72c113a7be
begin
	function paroch_pay(dat; title=false, legend=false, xlab=false)
		paroch = plot(
				dat.mixed_aleph2,
				dat.freq_parochial_g0,
				ylim=(-0.1,1),
				xlim=(0,1),
				lw=2,
				legend=legend ? :bottomright : false,
				label=L"c_0",
				c=palette(:roma)[1],
				ylab="",
				xlab="",
				title=title ? L"\mathrm{parochialism}" : "",
				ylabelfontsize=12,
				xticks=(
				[0.2, 0.4, 0.6, 0.8], 
				[L"0.2", L"0.4", L"0.6", L"0.8"]
				),
				yticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
				),
				grid=false
			)
			plot!(
				dat.mixed_aleph2,
				dat.freq_parochial_g1,
				c=palette(:roma)[200],
				label=L"c_1",
				lw=2,
				#alpha=0.5,
				#ls=:dash
			)
		
		pb = plot(
				dat.mixed_aleph2,
				dat.freq_pb_g0,
				ylim=(0,1),
				xlim=(0,1),
				lw=2,
				legend=false,
				legendtitle=L"\mathrm{group}",
				label=L"0",
				c=palette(:roma)[1],
				title=title ? L"\mathrm{payoff\ bias}" : "",
				ylab=L"\epsilon_{c_1} = %$(round(dat.mixed_u2[1] - 0.5, digits=2))",
				xlab="",
				ylabelfontsize=12,
				xticks=(
				[0.2, 0.4, 0.6, 0.8], 
				[L"0.2", L"0.4", L"0.6", L"0.8"]
				),
				yticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"0", L"0.25", L"0.5", L"0.75", L"1.0"]
				),
				grid=false
			)
			plot!(
				dat.mixed_aleph2,
				dat.freq_pb_g1,
				c=palette(:roma)[200],
				label="",
				lw=2,
				#alpha=0.5,
				#ls=:dash
			)

		if xlab
			annotate!([1.1], [-0.35], text(L"\mathrm{wealth\ buffer\ of\ } c_1\ (\aleph_{c_1})", color="black", 15))
		end
	
		plot(pb, paroch, layout=(1,2))
	end
	
	pbparoch_plot = plot(
		paroch_pay(g0u75_g1u75, title=true, legend=true),
		paroch_pay(g0u75_g1u65),
		paroch_pay(g0u75_g1u6),
		paroch_pay(g0u75_g1u55, xlab=true),
		layout=(4,1),
		size=(600, 700),
		bottom_margin=4.75Plots.mm
	)

	savefig(pbparoch_plot, "../images/sup4_pbparoch.pdf")

	pbparoch_plot
end

# ╔═╡ f1d25cf9-b070-4acc-8431-3c24a928da62
md"""
#### Figure S6
"""

# ╔═╡ 7e61b751-7395-43a6-9f91-3d9151d0890e
begin
	function plot_opt_elders(
		dat,
		aleph; 
		ylim = (0.4, 1.15), 
		xlab = L"\mathrm{number\ of\ sampled\ peers\ } (n)",
		ylab = true,
		title = true,
		legend = true
		)
		
		dat1 = dat[
			dat.time .== 2500 .&&
			dat.aleph .== aleph
			,:]
	
		dat1_l2 = dat1[dat1.u .== 0.55, :]
		dat1_l3 = dat1[dat1.u .== 0.6, :]
		dat1_l6 = dat1[dat1.u .== 0.65, :]

		mdatl2_0 = dat1_l2[(dat1_l2.m .== 1), :]
		mdatl2_1 = dat1_l2[(dat1_l2.m .== 5), :]
		mdatl2_2 = dat1_l2[(dat1_l2.m .== 10), :]
		mdatl2_3 = dat1_l2[(dat1_l2.m .== 15), :]

		grouped_vbar0 = combine(
			groupby(mdatl2_0, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar1 = combine(
			groupby(mdatl2_1, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar2 = combine(
			groupby(mdatl2_2, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar3 = combine(
			groupby(mdatl2_3, :n), 
			:Vbar => mean => :mean_Vbar,
		)

		mdatl3_0 = dat1_l3[(dat1_l3.m .== 1), :]
		mdatl3_1 = dat1_l3[(dat1_l3.m .== 5), :]
		mdatl3_2 = dat1_l3[(dat1_l3.m .== 10), :]
		mdatl3_3 = dat1_l3[(dat1_l3.m .== 15), :]

		grouped_vbar02 = combine(
			groupby(mdatl3_0, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar4 = combine(
			groupby(mdatl3_1, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar5 = combine(
			groupby(mdatl3_2, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar6 = combine(
			groupby(mdatl3_3, :n), 
			:Vbar => mean => :mean_Vbar,
		)

		mdatl6_0 = dat1_l6[(dat1_l6.m .== 1), :]
		mdatl6_1 = dat1_l6[(dat1_l6.m .== 5), :]
		mdatl6_2 = dat1_l6[(dat1_l6.m .== 10), :]
		mdatl6_3 = dat1_l6[(dat1_l6.m .== 15), :]

		grouped_vbar03 = combine(
			groupby(mdatl6_0, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar7 = combine(
			groupby(mdatl6_1, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar8 = combine(
			groupby(mdatl6_2, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar9 = combine(
			groupby(mdatl6_3, :n), 
			:Vbar => mean => :mean_Vbar,
		)
		
		tauplot1 = plot(
		    grouped_vbar1.n, grouped_vbar0.mean_Vbar, 
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabel = ylab ? L"\bar{V} \left.\right|_{\aleph = %$(aleph)}" : "",
			ylabelfontsize = 15,
			legend = false,
			legendtitle = L"m",
		    label = L"5",
			lw = 2,
			grid = false,
			title = title ? L"\epsilon = 0.05" : "",
			xticks = ([1, 5, 10, 15], [L"1", L"5", L"10", L"15"]),
			yticks = ([0.5, 0.75, 1.0], [L"0.5", L"0.75", L"1.0"]),
			ylim = ylim
			)
		plot!(
		    grouped_vbar2.n, grouped_vbar1.mean_Vbar, 
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar3.n, grouped_vbar2.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
		plot!(
		    grouped_vbar0.n, grouped_vbar3.mean_Vbar,
		    fillalpha=0.2, label = L"1", lw=2,
		)
	
		tauplot2 = plot(
		    grouped_vbar4.n, grouped_vbar02.mean_Vbar, 
			xlabel = xlab,
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabelfontsize = 15,
			legend = false,
			legendtitle = L"n",
		    label = L"5",
			lw = 2,
			grid = false,
			title = title ? L"\epsilon = 0.10" : "",
			xticks = ([1, 5, 10, 15], [L"1", L"5", L"10", L"15"]),
			yticks = false,
			ylim = ylim
			)
		plot!(
		    grouped_vbar5.n, grouped_vbar4.mean_Vbar, 
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar6.n, grouped_vbar5.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
		plot!(
		    grouped_vbar02.n, grouped_vbar6.mean_Vbar,
		    fillalpha=0.2, label = L"1", lw=2,
		)
	
		tauplot3 = plot(
		    grouped_vbar7.n, grouped_vbar03.mean_Vbar, 
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabelfontsize = 15,
			legend = legend,
			legendtitle = L"m",
		    label = L"1",
			lw = 2,
			grid=false,
			title = title ? L"\epsilon = 0.15" : "",
			xticks = ([1, 5, 10, 15], [L"1", L"5", L"10", L"15"]),
			yticks = false,
			ylim = ylim
			)
		plot!(
		    grouped_vbar8.n, grouped_vbar7.mean_Vbar, 
		    fillalpha=0.2, label = L"5", lw=2,
		)
		plot!(
		    grouped_vbar9.n, grouped_vbar8.mean_Vbar,
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar03.n, grouped_vbar9.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
	
		return plot(
			tauplot1, tauplot2, tauplot3,
			layout = (1,3)
		)
		
	end
	
	modelnum_plot = plot(
		plot_opt_elders(
			CSV.read("../data/analysis_1.csv", DataFrame), 
			0.05, xlab="", legend=false
		),
		plot_opt_elders(
			CSV.read("../data/analysis_1.csv", DataFrame), 
			0.5, title=false, xlab="", legend=false
		),
		plot_opt_elders(
			CSV.read("../data/analysis_1.csv", DataFrame), 
			0.95, title=false, legend=:bottomright
		),
		layout=(3,1), size=(600, 600), bottom_margin=0Plots.mm
	)

	savefig(modelnum_plot, "../images/sup5_modelnum.pdf")

	modelnum_plot
end

# ╔═╡ f4ee25e4-626e-4d72-b4b5-dd1f9f389f8c
# ╠═╡ disabled = true
#=╠═╡
begin
	function plot_opt_peers(
		dat,
		aleph; 
		ylim = (0.0, 1.05), 
		xlab = L"\mathrm{risk-free\ period\ length\ } (\tau)",
		ylab = true,
		title = true,
		legend = true
		)
		
		dat1 = dat[
			dat.time .== 2500 .&&
			#dat.N .== 1000 .&& 
			dat.T .== 100 .&&
			dat.aleph .== aleph
			,:]
	
		dat1_l2 = dat1[dat1.u .== 0.55, :]
		dat1_l3 = dat1[dat1.u .== 0.6, :]
		dat1_l6 = dat1[dat1.u .== 0.65, :]

		mdatl2_0 = dat1_l2[(dat1_l2.n .== 1), :]
		mdatl2_1 = dat1_l2[(dat1_l2.n .== 5), :]
		mdatl2_2 = dat1_l2[(dat1_l2.n .== 10), :]
		mdatl2_3 = dat1_l2[(dat1_l2.n .== 15), :]

		grouped_vbar0 = combine(
			groupby(mdatl2_0, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar1 = combine(
			groupby(mdatl2_1, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar2 = combine(
			groupby(mdatl2_2, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar3 = combine(
			groupby(mdatl2_3, :t), 
			:Vbar => mean => :mean_Vbar,
		)

		mdatl3_0 = dat1_l3[(dat1_l3.n .== 1), :]
		mdatl3_1 = dat1_l3[(dat1_l3.n .== 5), :]
		mdatl3_2 = dat1_l3[(dat1_l3.n .== 10), :]
		mdatl3_3 = dat1_l3[(dat1_l3.n .== 15), :]

		grouped_vbar02 = combine(
			groupby(mdatl3_0, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar4 = combine(
			groupby(mdatl3_1, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar5 = combine(
			groupby(mdatl3_2, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar6 = combine(
			groupby(mdatl3_3, :t), 
			:Vbar => mean => :mean_Vbar,
		)

		mdatl6_0 = dat1_l6[(dat1_l6.n .== 1), :]
		mdatl6_1 = dat1_l6[(dat1_l6.n .== 5), :]
		mdatl6_2 = dat1_l6[(dat1_l6.n .== 10), :]
		mdatl6_3 = dat1_l6[(dat1_l6.n .== 15), :]

		grouped_vbar03 = combine(
			groupby(mdatl6_0, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar7 = combine(
			groupby(mdatl6_1, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar8 = combine(
			groupby(mdatl6_2, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		grouped_vbar9 = combine(
			groupby(mdatl6_3, :t), 
			:Vbar => mean => :mean_Vbar,
		)
		
		tauplot1 = plot(
		    grouped_vbar1.t, grouped_vbar0.mean_Vbar, 
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabel = ylab ? L"\bar{V} \left.\right|_{\aleph = %$(aleph)}" : "",
			ylabelfontsize = 15,
			legend = false,
			legendtitle = L"n",
		    label = L"5",
			lw = 2,
			grid = false,
			title = title ? L"\epsilon = 0.05" : "",
			xticks = ([5, 10, 15], [L"5", L"10", L"15"]),
			yticks = ([0.0, 0.25, 0.5, 0.75, 1.0], [L"0.0", L"0.25", L"0.5", L"0.75", L"1.0"]),
			ylim = ylim
			)
		plot!(
		    grouped_vbar2.t, grouped_vbar1.mean_Vbar, 
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar3.t, grouped_vbar2.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
		plot!(
		    grouped_vbar0.t, grouped_vbar3.mean_Vbar,
		    fillalpha=0.2, label = L"1", lw=2,
		)
	
		tauplot2 = plot(
		    grouped_vbar4.t, grouped_vbar02.mean_Vbar, 
			xlabel = xlab,
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabelfontsize = 15,
			legend = false,
			legendtitle = L"n",
		    label = L"5",
			lw = 2,
			grid = false,
			title = title ? L"\epsilon = 0.10" : "",
			xticks = ([5, 10, 15], [L"5", L"10", L"15"]),
			yticks = false,
			ylim = ylim
			)
		plot!(
		    grouped_vbar5.t, grouped_vbar4.mean_Vbar, 
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar6.t, grouped_vbar5.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
		plot!(
		    grouped_vbar02.t, grouped_vbar6.mean_Vbar,
		    fillalpha=0.2, label = L"1", lw=2,
		)
	
		tauplot3 = plot(
		    grouped_vbar7.t, grouped_vbar03.mean_Vbar, 
			xlabelfontsize = 15,
			palette=cgrad(:matter, 5, categorical = true)[2:end],
			ylabelfontsize = 15,
			legend = legend,
			legendtitle = L"n",
		    label = L"1",
			lw = 2,
			grid=false,
			title = title ? L"\epsilon = 0.15" : "",
			xticks = ([5, 10, 15], [L"5", L"10", L"15"]),
			yticks = false,
			ylim = ylim
			)
		plot!(
		    grouped_vbar8.t, grouped_vbar7.mean_Vbar, 
		    fillalpha=0.2, label = L"5", lw=2,
		)
		plot!(
		    grouped_vbar9.t, grouped_vbar8.mean_Vbar,
		    fillalpha=0.2, label = L"10", lw=2,
		)
		plot!(
		    grouped_vbar03.t, grouped_vbar9.mean_Vbar,
		    fillalpha=0.2, label = L"15", lw=2,
		)
	
		return plot(
			tauplot1, tauplot2, tauplot3,
			layout = (1,3)
		)
		
	end
	
	plot(
		plot_opt_peers(dat, 0.05, xlab="", legend=false),
		plot_opt_peers(dat, 0.5, title=false, xlab="", legend=false),
		plot_opt_peers(dat, 0.95, title=false, legend=:bottomright),
		layout=(3,1), size=(600, 600), bottom_margin=0Plots.mm
	)
end
  ╠═╡ =#

# ╔═╡ b9d9f726-e94e-4a95-815b-06a77e3c605b
# ╠═╡ disabled = true
#=╠═╡
begin
	modelio_mixed = initialize_pessimistic_learning(
			N = 2000,
			T = 500,
			m = 20,
			n = 20,
			t = 15,
			sens = 0.0,
			soc_v = 0.0,
			#SINGLE POP
			u = 0.55,
			aleph = 0.05,
			envshift = 3000,
			peg_lambda = false,
			peg_aleph = true,
			u_shift = 0.65,
			aleph_shift = 0.05,
			#MIXED
			mixed_aleph1 = 0.95,
			mixed_aleph2 = 0.05,
			mixed_u1 = 0.75,
			mixed_u2 = 0.55,
			#EVOLUTION PARAMETERS
			mu_std = 0.01,
			mu_soc_h = 0.001,
			mu_soc_v = 0.001,
			mu_sens = 0.001,
			mu_L = 0.01,
			mu_parochial = 0.01,
			strategies = "UB&PB",
			mixed = true,
			mixed_freq = 0.5,
			parochial = false,
			periodic = true,
			selection = true,
			b_coeff = 1.0,
			seed = 55686457456543,
		)
	
		adata7, mdata7 = run!(
			modelio_mixed, 
			2500,
			adata=[:s],
			mdata=[
				:Vbar, :s_median, :s_young_median, :s_child_median, :soc_v_median, 
				:soc_h_median, :sens_median, :soc_v_median_g0, :soc_h_median_g0, 
				:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
				:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
				:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
			]
		)

	md"""
	### Experimental Area
	"""
end
  ╠═╡ =#

# ╔═╡ ebf1dade-ce5e-42dd-a687-1bb8791eb7ab
#=╠═╡
begin
	plot(
		mdata7.time[2:end],
		mdata7.soc_v_median_g0[2:end],
		label=L"\alpha",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		xlabel=L"t",
		xlabelfontsize=20,
		#legend=false,
		grid=false,
		xticks=false,
		yticks=([0.0, 0.5, 1.0], [L"0", L"0.5", L"1"]),
		ytickfontsize=10
	)
	plot!(
		mdata7.time[2:end],
		mdata7.sens_median_g0[2:end],
		label=L"\delta",
		color="black",
		alpha=0.25,
		lw=2,
	)
	plot!(
		mdata7.time[2:end],
		mdata7.soc_v_median_g1[2:end],
		label="",
		color="black",
		lw=2,
		ls=:dot,
	)
	plot!(
		mdata7.time[2:end],
		mdata7.sens_median_g1[2:end],
		label="",
		color="black",
		alpha=0.25,
		lw=2,
		ls=:dot,
	)
end
  ╠═╡ =#

# ╔═╡ c8795305-655c-47bd-a385-e711f0ecc41d
#=╠═╡
begin
	plot(
		mdata7.time[2:end],
		mdata7.freq_pb_g0[2:end],
		label=L"\mathrm{PB}",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		xlabel=L"t",
		xlabelfontsize=20,
		grid=false
		)
	plot!(
		mdata7.time[2:end],
		mdata7.freq_pb_g1[2:end],
		label="",
		color="black",
		lw=2,
		ls=:dot,
	)
	plot!(
		mdata7.time[2:end],
		mdata7.soc_h_median_g0[2:end],
		label=L"\beta",
		color="black",
		lw=2,
		alpha=0.75
	)
	plot!(
		mdata7.time[2:end],
		mdata7.soc_h_median_g1[2:end],
		label="",
		color="black",
		lw=2,
		ls=:dot,
		alpha=0.75
	)
end
  ╠═╡ =#

# ╔═╡ dd70f9c5-1d26-4a0c-bcfa-106300110cbb
#=╠═╡
begin
	plot(
		mdata7.time[2:end],
		mdata7.freq_parochial_g0[2:end],
		label=L"\mathrm{Parochialism}",
		lw=2,
		color="black",
		ylim=(0.0, 1.0),
		xlabel=L"t",
		xlabelfontsize=20,
		grid=false
	)
	plot!(
		mdata7.time[2:end],
	 	mdata7.freq_parochial_g1[2:end],
		label="",
	 	color="black",
	 	lw=2,
		ls=:dot,
	)
end
  ╠═╡ =#

# ╔═╡ 896ff8a7-0f61-4be3-adf4-d8e55ad16874
# ╠═╡ disabled = true
#=╠═╡
begin
	env_plot = plot( 
		plot_powerdist.(
		[1.0, 1.5, 2.0, 3.0, 4.0, 5.0], n=1000)..., 
		layout=(2,3), 
		link=:all,
		margins=5Plots.mm
		#plot_title="power-distributed success rates (u)",
	)

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
	
	savefig(env_plot, "../images/fig1_env.pdf")
	
	plot(
		env_plot,
		varplot,
		size=(900, 400)
	)
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╟─cfb0a045-a40d-4019-987f-e4dc1c92074e
# ╟─1f7f1128-8eea-4695-b8cd-044d7c91890b
# ╟─ffd166fa-9d48-477a-8d16-f5e866551dfc
# ╟─b739131f-a173-4994-8143-d6c52ac07cb3
# ╟─fe9e037d-d510-4026-83e0-005a61fa005a
# ╟─21c738fa-0e6c-4592-a332-b0a564a23fa2
# ╟─f030487a-a4ed-461d-9405-fa69155f6e9e
# ╟─41f7d1c6-2a94-4fdf-a23e-6e93502cddd7
# ╟─5e5d671d-0b35-4404-b022-c2fe8bdde7c2
# ╟─8a2c7c0c-e59d-4727-ab06-c0367298fb8b
# ╟─f3fec0a1-40fc-4168-93a5-bb0648086d64
# ╟─aef1aaeb-5cd1-4121-81ab-490a24a89af0
# ╟─326013e3-69fa-4ba2-9381-ccfadbf00613
# ╟─6ac78a6a-aa27-4ea0-8e49-e469186dfce8
# ╟─223e84bb-404e-4c66-bc3d-242d42c00bc9
# ╟─324d8681-82ae-42c0-8e23-91b7ab4a2cd4
# ╟─88e18457-ae1b-4de6-958b-df4f407d0b98
# ╟─43b13c02-86ac-4620-b62b-a81e98a173a4
# ╟─80f6d4ac-3036-4e7a-9ab7-31d0644ec6ea
# ╟─55c3f3f2-2e71-4fe3-b7a0-cdd9b0422816
# ╟─ceacb32a-71ca-49c6-ab92-f5f33fc32b31
# ╟─a3dec80f-60df-44b1-adc5-855cf06b492a
# ╟─3a2082c4-ef09-4b53-af1b-411792729560
# ╟─c56c822a-ded8-4550-978c-2a037389a85a
# ╟─2fbddc1c-6d2c-4195-9366-0e7a483450fc
# ╟─077c7aec-998a-4295-9426-85d254d251f7
# ╟─3dc40146-3fdb-4d00-a288-571c47a45041
# ╟─656b3a41-1ceb-4637-af43-81b3e4d25832
# ╟─87c21ae5-79ec-4481-9c80-4e72c113a7be
# ╟─f1d25cf9-b070-4acc-8431-3c24a928da62
# ╟─7e61b751-7395-43a6-9f91-3d9151d0890e
# ╟─f4ee25e4-626e-4d72-b4b5-dd1f9f389f8c
# ╟─b9d9f726-e94e-4a95-815b-06a77e3c605b
# ╟─ebf1dade-ce5e-42dd-a687-1bb8791eb7ab
# ╟─c8795305-655c-47bd-a385-e711f0ecc41d
# ╟─dd70f9c5-1d26-4a0c-bcfa-106300110cbb
# ╟─896ff8a7-0f61-4be3-adf4-d8e55ad16874
