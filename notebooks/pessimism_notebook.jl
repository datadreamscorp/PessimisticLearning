### A Pluto.jl notebook ###
# v0.20.20

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
	## The development of risk behaviors and their cultural transmission
	
	#### Alejandro Pérez Velilla, Bret Beheim, Paul E. Smaldino.
	"""
end

# ╔═╡ 1f7f1128-8eea-4695-b8cd-044d7c91890b
begin
	function levelplot(
	    df::DataFrame,
	    xcol::Symbol, 
	    ycol::Symbol, 
	    zcol::Symbol;
	    title="",
	    xlab="",
		xlabelfontsize=15,
	    ylab="",
		ylabelfontsize=12,
		xticks=true,
		yticks=true,
		xtickfontsize=8,
		ytickfontsize=8,
	    colrange::Union{Nothing,Tuple{Float64,Float64}}=nothing,
	    colormap=cgrad(:grays, rev=true),
	    aspect_ratio::Symbol=:equal,
	    show_colorbar::Bool=true,
		cbarticks=[],
		titlefontsize::Int = 10,
		mask::Bool = true,
		delta::Float64 = 0.49,
	)
	
	    xvals = sort(unique(df[!, xcol]))
	    yvals = sort(unique(df[!, ycol]))
	
	    Z = Matrix{Float64}(undef, length(yvals), length(xvals))
	    
	    for row in eachrow(df)
	        xval = row[xcol]
	        yval = row[ycol]
	        zval = row[zcol]
	        # find indices in xvals, yvals
	        ix = findfirst(==(xval), xvals)
	        iy = findfirst(==(yval), yvals)
	        Z[iy, ix] = zval
	    end
	
	    hplot = heatmap(
	        xvals,       # x grid
	        yvals,       # y grid
	        Z;           # value matrix
	        title       = title,
			titlefontsize = titlefontsize,
	        xlabel      = xlab,
			xlabelfontsize = xlabelfontsize,
			ylabelfontsize = ylabelfontsize,
			xticks 		= xticks,
			xtickfontsize = xtickfontsize,
	        ylabel      = ylab,
			yticks 		= yticks,
			ytickfontsize = ytickfontsize,
	        #aspect_ratio= aspect_ratio,
	        framestyle  = :grid,
	        clims       = colrange,   
	        color       = colormap,
	        colorbar    = show_colorbar,
	    )
		
		ext = extrema(yvals)
		if mask
			for yi in range(ext[1], ext[2]+0.01, 20)
	    		annotate!(maximum(xvals) + delta, yi, text("██", :white, 30))
			end
		end
		if length(cbarticks) > 0
			annotate!(cbarticks[1], cbarticks[2], cbarticks[3])
		end
	
		hplot
	end

md"""
## Main text
"""
end

# ╔═╡ ffd166fa-9d48-477a-8d16-f5e866551dfc
md"""
#### Figure 1 - The effects of absorbing boundaries
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
#### Figure 2 - A tale of three brothers
"""

# ╔═╡ 21c738fa-0e6c-4592-a332-b0a564a23fa2
begin
	modelium = initialize_pessimistic_learning(N=3, u = 0.65, soc_h=0.25, aleph = 0.15, n=2, T=100, mu_sens = 1.0, selection=true, seed=58)

	svecs_young = [a.s_vec_young for a in allagents(modelium)|>collect]
	svecs = [a.s_vec for a in allagents(modelium)|>collect]
	pvecs = [a.payoff_vec for a in allagents(modelium)|>collect]
	ruin1 = (filter(x -> x > 0.0, pvecs[1])|>length) + 1
	ruin3 = (filter(x -> x > 0.0, pvecs[3])|>length) + 1
	
	svecs_youngplot = plot(
		svecs_young[1],
		ylabel=L"\mathrm{mean\ stake}",
		title=L"\mathrm{child\ timeline}",
		ylim=(-0.02, 0.85),
		xticks=([1, 17], [L"0", L"\tau"]),
		xtickfontsize=9,
		titlefontsize=12,
		yticks=([0.0, 0.25, 0.5, 0.75], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75]]),
		legend=false,
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(svecs_young[2], color=palette(:lipariS)[6], lw=2, label="")
	plot!(svecs_young[3], color=palette(:lipariS)[8], lw=2, label="")
	hline!([2*0.65 - 1], ls=:dash, color=:black, label="")
	hline!([0.0], color=:black, alpha=0.2, label="", ls=:dot)
	annotate!([15.5], [0.25], [text(L"s^*", 8, color="black")])
	
	svecsplot = plot(
		svecs[1][1:ruin1],
		#title=L"\mathrm{mean\ adult\ stake}",
		#xticks=false, 
		xticks=([0, 101], [L"\tau", L"T+\tau"]),
		yticks=false,
		#yticks=([0.0, 0.25, 0.5, 0.75], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75]]),
		ylim=(-0.02, 0.85),
		xtickfontsize=9,
		title=L"\mathrm{adult\ timeline}",
		titlefontsize=12,
		label=L"\mathrm{Juan}", 
		legendfontsize=7,
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(svecs[2], label=L"\mathrm{Roberto}", color=palette(:lipariS)[6], lw=2)
	plot!(svecs[3][1:ruin3], label=L"\mathrm{Gabriel}", color=palette(:lipariS)[8], lw=2)
	hline!([s_star(0.65, 0.15)], ls=:dash, color=:black, label=false)
	annotate!([90], [0.08], [text(L"s^*_{\mathrm{ruin}}", 8, color="black")])
	hline!([0.0], color=:black, alpha=0.2, label="", ls=:dot)
	scatter!(
		[ruin1, ruin3],
		[svecs[1][ruin1], svecs[3][ruin3]],
		label=L"\mathrm{ruin\ event}",
		markershape=:xcross,
		color="black"
	)
	
	pvecsplot = plot(
		vcat(repeat([log(1/(1 - 0.15))], 15), pvecs[1]), 
		xticks=([1, 16, 116], [L"0", L"\tau", L"T+\tau"]),
		yticks=([0, 1, 2], [L"%$a" for a in [0, 1, 2]]),
		ylabel=L"\mathrm{log\ wealth}",
		xlabel=L"\mathrm{lifetime}",
		xtickfontsize=12,
		xlabelfontsize=16,
		legend=false, 
		grid=false,
		lw=2,
		dpi=300,
		color=palette(:lipariS)[7]
	)
	plot!(vcat(repeat([log(1/(1 - 0.15))], 15), pvecs[2]), lw=2, color=palette(:lipariS)[6])
	plot!(vcat(repeat([log(1/(1 - 0.15))], 15), pvecs[3]), lw=2, color=palette(:lipariS)[8])
	scatter!(
		[ruin1+15, ruin3+15],
		[pvecs[1][ruin1], pvecs[3][ruin3]],
		markershape=:xcross,
		color="black"
	)
	hline!([0.0], color=:black, alpha=0.2, label="", ls=:dot, lw=2)

	broplot = plot(
		svecs_youngplot,
		svecsplot,
		layout=(1,2)
	)

	full_broplot = plot(
		broplot,
		pvecsplot,
		right_margin=2Plots.mm,
		layout=(2,1)
	)

	savefig(full_broplot, "../images/fig2_bros.pdf")

	full_broplot
end

# ╔═╡ f030487a-a4ed-461d-9405-fa69155f6e9e
begin
	sens = 0.5
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
#### Figure 3 - Population effects of social trauma
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
		color=palette(:tokyo10)[1],
	)
	histogram!( 
		[a.s_mean for a in allagents(model2)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.55, 0.05)], lw=2, color="black", ls=:dash, label="")
	
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
	vline!([s_star(0.65, 0.05)], lw=2, color="black", ls=:dash, label="")

	p3 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model5)|>collect]), bins=10, alpha=0.5, legend=false, xlim=(0,0.65), 
		title=L"\epsilon = 0.25", color=palette(:tokyo10)[1],
		xticks=false, yticks=false, grid=false 
	)
	histogram!( 
		[a.s_mean for a in allagents(model6)|>collect], 
		bins=25, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.75, 0.05)], lw=2, color="black", ls=:dash, label="")

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
	vline!([s_star(0.55, 0.5)], lw=2, color="black", ls=:dash, label="")

	p5 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model9)|>collect]), 
		bins=20, alpha=0.5, legend=false, xlim=(0,0.65), 
		xticks=false, yticks=false, grid=false, color=palette(:tokyo10)[1]
	)
	histogram!( 
		[a.s_mean for a in allagents(model10)|>collect], 
		bins=15, alpha=0.5, color=palette(:tokyo10)[7]
	)
	vline!([s_star(0.65, 0.5)], lw=2, color="black", ls=:dash, label="")

	p6 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model11)|>collect]), 
		bins=10, alpha=0.5, legend=false, xlim=(0,0.65), 
		xticks=false, yticks=false, grid=false, color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model12)|>collect], 
		bins=25, alpha=0.5, color=palette(:tokyo10)[7] 
	)
	vline!([s_star(0.75, 0.5)], lw=2, color="black", ls=:dash, label="")

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
	vline!([s_star(0.55, 0.95)], lw=2, color="black", ls=:dash, label="")

	p8 = histogram( 
		s.([a.α / (a.α + a.β) for a in allagents(model15)|>collect]), 
		bins=10, alpha=0.5, xlim=(0,0.65), yticks=false, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		grid=false, xtickfontsize=8, xlab=L"\mathrm{mean\ stake\ } (s)", xlabelfontsize=20, legend=false,
		color=palette(:tokyo10)[1] 
	)
	histogram!( 
		[a.s_mean for a in allagents(model16)|>collect], 
		bins=10, alpha=0.5, color=palette(:tokyo10)[7] )
	vline!([s_star(0.65, 0.95)], lw=2, color="black", ls=:dash, label="")

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
	vline!([s_star(0.75, 0.95)], lw=2, color="black", ls=:dash, label="")

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
		dat.time .== 2500 #.&&
		#dat.N .== 1000 .&& 
		#dat.T .== 100 .&& 
		#dat.n .== 10 .&&
		#dat.t .== 15 #.&&
		#dat.envshift .== 3000 .&&
		#dat.mu_soc_v .== 0.0
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
	#### Figure 5 - Evolution of learning strategies under static environments
	"""
end

# ╔═╡ fabf9049-9a39-4765-b74b-efced7fc73df
begin
	eldat = CSV.read("../data/analysis_1.csv", DataFrame)
	eldat = eldat[eldat.time .== 2500, :]
	eldat.soc_h = eldat.soc_h_median .* (1 .- eldat.soc_v_median)
	eldat = combine(
		groupby(eldat, [:aleph, :u]),
		:soc_h => median => :soc_h,
		:soc_v_median => median => :soc_v,
		:sens_median => median => :sens,
		:freq_pb => median =>  :pb,
		:s_median => median => :s,
	)
	
	low_left_plot = levelplot(
			eldat,
			:aleph,
			:u,
			:pb,
			title=L"\mathrm{C.\ payoff\ bias\ frequency}",
			colrange=(0.0,1.0),
			xticks=(0.05:0.3:0.95, [L"%$a" for a in 0.05:0.3:0.95]),
			yticks=(0.55:0.1:0.95, [L"%$(round(a - 0.5, digits=2))" for a in 0.55:0.1:0.95]),
			mask=true,
			delta=0.36,
		)
	annotate!([1.3], [0.44], text(L"\mathrm{wealth\ buffer\ } (\aleph)", 15))
	annotate!([-0.225], [1.075], text(L"\mathrm{environmental\ edge\ } (\epsilon)", 15, rotation=90))
	
	static_env = plot(
		levelplot(
			eldat,
			:aleph,
			:u,
			:soc_h,
			title=L"\mathrm{A.\ peer\ influence\ } (\tilde{\beta})",
			yticks=(0.55:0.1:0.95, [L"%$(round(a - 0.5, digits=2))" for a in 0.55:0.1:0.95]),
			xticks=false,
			colrange=(0.0,1.0),
			mask=true,
			delta=0.36,
			
		),
		levelplot(
			eldat,
			:aleph,
			:u,
			:sens,
			title=L"\mathrm{B.\ sensitivity\ } (\delta)",
			xticks=false,
			yticks=false,
			colrange=(0.0,1.0),
			mask=true,
			delta=0.36,
			cbarticks=(
				[1.18, 1.18, 1.18],
				[0.54, 0.75, 0.96],
				[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
			),
		),
		low_left_plot,
		levelplot(
			eldat,
			:aleph,
			:u,
			:soc_v,
			title=L"\mathrm{D.\ elder\ influence\ } (\alpha)",
			xticks=(0.05:0.3:0.95, [L"%$a" for a in 0.05:0.3:0.95]),
			yticks=false,
			colrange=(0.0,1.0),
			mask=true,
			delta=0.36,
			cbarticks=(
				[1.18, 1.18, 1.18],
				[0.54, 0.75, 0.96],
				[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
			),
		),
		bottom_margins=6Plots.mm,
		left_margins=6Plots.mm,
		dpi=300
	)

	savefig(static_env, "../images/fig5_static_env.pdf")

	static_env
end

# ╔═╡ 223e84bb-404e-4c66-bc3d-242d42c00bc9
md"""
#### Figure 6 - Environmental change
"""

# ╔═╡ 25a80f68-931e-4e2b-b591-316ab0df0863
begin
	dat_inc3 = CSV.read("../data/analysis_1d.csv", DataFrame)
	dat_inc3 = dat_inc3[dat_inc3.time .== 2500, :]
	dat_inc3.soc_h_median = dat_inc3.soc_h_median .* (1 .- dat_inc3.soc_v_median)
	dat_inc3.soc_h_lerror = dat_inc3.soc_h_lerror .* (1 .- dat_inc3.soc_v_lerror)
	dat_inc3.soc_h_herror = dat_inc3.soc_h_herror .* (1 .- dat_inc3.soc_v_herror)
	
	dat_inc3 = combine(
		groupby(dat_inc3, [:aleph, :λ, :μ]),
		:mean_increment => median => :inc,
		:sbar => median => :sbar,
		:soc_h_median => median => :soc_h,
		:soc_h_lerror => median => :soc_h_lerror,
		:soc_h_herror => median => :soc_h_herror,
		:soc_v_median => median => :soc_v,
		:soc_v_lerror => median => :soc_v_lerror,
		:soc_v_herror => median => :soc_v_herror,
		:sens_median => median => :sens,
		:sens_lerror => median => :sens_lerror,
		:sens_herror => median => :sens_herror,
		:freq_pb => median => :pb,
		:s_median => median => :s,
		:s_lerror => median => :s_lerror,
		:s_herror => median => :s_herror,
		:s_ltail => median => :s_ltail,
		:s_htail => median => :s_htail,
	)

	a05 = dat_inc3[dat_inc3.aleph .== 0.05, :]
	a50 = dat_inc3[dat_inc3.aleph .== 0.5, :]
	a95 = dat_inc3[dat_inc3.aleph .== 0.95, :]
	
	bottomcenter = levelplot(
				a50,
				:λ,
				:μ,
				:pb,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				xtickfontsize=6,
				xlab=" ",
				xlabelfontsize=10
			)
	annotate!([0.5], [-0.5], [text(L"\mathrm{aggregate\ uncertainty\ } (λ)")])
	
	instability_plot = plot(
		plot(
			levelplot(
				a05,
				:λ,
				:μ,
				:soc_h,
				colrange=(0.0,1.0),
				show_colorbar=false,
				xticks=false,
				yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				ytickfontsize=6,
				title=L"\aleph = 0.05",
				ylab=L"\mathrm{A.\ peer\ influence}",
				ylabelfontsize=8
			),
			levelplot(
				a50,
				:λ,
				:μ,
				:soc_h,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false,
				title=L"\aleph = 0.5"
			),
			levelplot(
				a95,
				:λ,
				:μ,
				:soc_h,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false,
				title=L"\aleph = 0.95"
			),
			layout=(1,3), margins=2Plots.mm
		),
		plot(
			levelplot(
				a05,
				:λ,
				:μ,
				:soc_v,
				colrange=(0.0,1.0),
				show_colorbar=false,
				xticks=false,
				yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				ytickfontsize=6,
				ylab=L"\mathrm{B.\ elder\ influence}",
				ylabelfontsize=8
			),
			levelplot(
				a50,
				:λ,
				:μ,
				:soc_v,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false
			),
			levelplot(
				a95,
				:λ,
				:μ,
				:soc_v,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false
			),
			layout=(1,3), margins=2Plots.mm
		),
		plot(
			levelplot(
				a05,
				:λ,
				:μ,
				:sens,
				colrange=(0.0,1.0),
				show_colorbar=false,
				xticks=false,
				yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				ytickfontsize=6,
				ylab=L"\mathrm{C.\ sensitivity}",
				ylabelfontsize=8
			),
			levelplot(
				a50,
				:λ,
				:μ,
				:sens,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false
			),
			levelplot(
				a95,
				:λ,
				:μ,
				:sens,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=false
			),
			layout=(1,3), margins=2Plots.mm
		),
		plot(
			levelplot(
				a05,
				:λ,
				:μ,
				:pb,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				ytickfontsize=6,
				xticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				xtickfontsize=6,
				ylab=L"\mathrm{D.\ payoff\ bias}",
				ylabelfontsize=8
			),
			bottomcenter,
			levelplot(
				a95,
				:λ,
				:μ,
				:pb,
				colrange=(0.0,1.0),
				show_colorbar=false,
				yticks=false,
				xticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
				xtickfontsize=6,
			),
			layout=(1,3), margins=2Plots.mm
		),
		layout=(4,1), size=(400,500)
	)

	# Create an empty spacer plot to serve as left margin
	spacer = plot(
	    1:0,
	    xlim = (0.9, 1), ylim = (0, 1),
	    axis = false, framestyle = :none, grid = false,
	    legend = false, ticks = false
	)
	annotate!([1.0], [0.5], [text(L"\mathrm{idiosyncratic\ uncertainty\ } (\mu)", rotation=90)])
	
	# Combine the spacer and your composite plot into a two-column layout.
	# The left column (spacer) gets a fixed fraction of the width.
	combined = plot(
	    spacer,
	    instability_plot,
	    layout = @layout([a{0.025w} b{0.9w}]),
	    size = (500,500)
	)
	
	savefig(combined, "../images/fig6_changing.pdf")

	combined
end

# ╔═╡ 80f6d4ac-3036-4e7a-9ab7-31d0644ec6ea
begin
	dat4 = CSV.read("../data/analysis_2.csv", DataFrame)
		
	dat5 = dat4[
		dat4.time .== 2500 #.&&
		,:]

	dat5.soc_h_median_g0 = (1 .- dat5.soc_v_median_g0) .* dat5.soc_h_median_g0
	dat5.soc_h_median_g1 = (1 .- dat5.soc_v_median_g1) .* dat5.soc_h_median_g1
	
	mdat5_1 = dat5[dat5.mixed_aleph2 .== 0.05, :]
	mdat5_2 = dat5[dat5.mixed_aleph2 .== 0.45, :]
	mdat5_3 = dat5[dat5.mixed_aleph2 .== 0.85, :]
	
	datseris = combine(
		groupby(mdat5_1, [:mixed_aleph2, :μ, :λ]), 
	    :soc_h_median_g0 => median => :soc_h_g0,
		:soc_h_lerror_g0 => median => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => median => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => median => :soc_v_g0,
		:soc_v_lerror_g0 => median => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => median => :mean_soc_v_herror_g0,
		:sens_median_g0 => median => :sens_g0,
		:sens_lerror_g0 => median => :mean_sens_lerror_g0,
		:sens_herror_g0 => median => :mean_sens_herror_g0,
		:s_median_g0 => median => :s_g0,
		:s_lerror_g0 => median => :mean_s_lerror_g0,
		:s_herror_g0 => median => :mean_s_herror_g0,
		:s_young_median_g0 => median => :mean_s_young_median_g0,
		:s_young_lerror_g0 => median => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => median => :mean_s_young_herror_g0,
		:Vbar_g0 => median => :Vbar_g0,
		:freq_pb_g0 => median => :pb_g0,
		:freq_parochial_g0 => median => :paroch_g0,
		:soc_h_median_g1 => median => :soc_h_g1,
		:soc_h_lerror_g1 => median => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => median => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => median => :soc_v_g1,
		:soc_v_lerror_g1 => median => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => median => :mean_soc_v_herror_g1,
		:sens_median_g1 => median => :sens_g1,
		:sens_lerror_g1 => median => :mean_sens_lerror_g1,
		:sens_herror_g1 => median => :mean_sens_herror_g1,
		:s_median_g1 => median => :s_g1,
		:s_lerror_g1 => median => :mean_s_lerror_g1,
		:s_herror_g1 => median => :mean_s_herror_g1,
		:s_young_median_g1 => median => :mean_s_young_median_g1,
		:s_young_lerror_g1 => median => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => median => :mean_s_young_herror_g1,
		:Vbar_g1 => median => :Vbar_g1,
		:freq_pb_g1 => median => :pb_g1,
		:freq_parochial_g1 => median => :paroch_g1,
	)
	
	datseris2 = combine(
		groupby(mdat5_2, [:mixed_aleph2, :μ, :λ]), 
	    :soc_h_median_g0 => median => :soc_h_g0,
		:soc_h_lerror_g0 => median => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => median => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => median => :soc_v_g0,
		:soc_v_lerror_g0 => median => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => median => :mean_soc_v_herror_g0,
		:sens_median_g0 => median => :sens_g0,
		:sens_lerror_g0 => median => :mean_sens_lerror_g0,
		:sens_herror_g0 => median => :mean_sens_herror_g0,
		:s_median_g0 => median => :s_g0,
		:s_lerror_g0 => median => :mean_s_lerror_g0,
		:s_herror_g0 => median => :mean_s_herror_g0,
		:s_young_median_g0 => median => :mean_s_young_median_g0,
		:s_young_lerror_g0 => median => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => median => :mean_s_young_herror_g0,
		:Vbar_g0 => median => :Vbar_g0,
		:freq_pb_g0 => median => :pb_g0,
		:freq_parochial_g0 => median => :paroch_g0,
		:soc_h_median_g1 => median => :soc_h_g1,
		:soc_h_lerror_g1 => median => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => median => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => median => :soc_v_g1,
		:soc_v_lerror_g1 => median => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => median => :mean_soc_v_herror_g1,
		:sens_median_g1 => median => :sens_g1,
		:sens_lerror_g1 => median => :mean_sens_lerror_g1,
		:sens_herror_g1 => median => :mean_sens_herror_g1,
		:s_median_g1 => median => :s_g1,
		:s_lerror_g1 => median => :mean_s_lerror_g1,
		:s_herror_g1 => median => :mean_s_herror_g1,
		:s_young_median_g1 => median => :mean_s_young_median_g1,
		:s_young_lerror_g1 => median => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => median => :mean_s_young_herror_g1,
		:Vbar_g1 => median => :Vbar_g1,
		:freq_pb_g1 => median => :pb_g1,
		:freq_parochial_g1 => median => :paroch_g1,
	)
	
	datseris3 = combine(
		groupby(mdat5_3, [:mixed_aleph2, :μ, :λ]), 
	    :soc_h_median_g0 => median => :soc_h_g0,
		:soc_h_lerror_g0 => median => :mean_soc_h_lerror_g0,
		:soc_h_herror_g0 => median => :mean_soc_h_herror_g0,
		:soc_v_median_g0 => median => :soc_v_g0,
		:soc_v_lerror_g0 => median => :mean_soc_v_lerror_g0,
		:soc_v_herror_g0 => median => :mean_soc_v_herror_g0,
		:sens_median_g0 => median => :sens_g0,
		:sens_lerror_g0 => median => :mean_sens_lerror_g0,
		:sens_herror_g0 => median => :mean_sens_herror_g0,
		:s_median_g0 => median => :s_g0,
		:s_lerror_g0 => median => :mean_s_lerror_g0,
		:s_herror_g0 => median => :mean_s_herror_g0,
		:s_young_median_g0 => median => :mean_s_young_median_g0,
		:s_young_lerror_g0 => median => :mean_s_young_lerror_g0,
		:s_young_herror_g0 => median => :mean_s_young_herror_g0,
		:Vbar_g0 => median => :Vbar_g0,
		:freq_pb_g0 => median => :pb_g0,
		:freq_parochial_g0 => median => :paroch_g0,
		:soc_h_median_g1 => median => :soc_h_g1,
		:soc_h_lerror_g1 => median => :mean_soc_h_lerror_g1,
		:soc_h_herror_g1 => median => :mean_soc_h_herror_g1,
		:soc_v_median_g1 => median => :soc_v_g1,
		:soc_v_lerror_g1 => median => :mean_soc_v_lerror_g1,
		:soc_v_herror_g1 => median => :mean_soc_v_herror_g1,
		:sens_median_g1 => median => :sens_g1,
		:sens_lerror_g1 => median => :mean_sens_lerror_g1,
		:sens_herror_g1 => median => :mean_sens_herror_g1,
		:s_median_g1 => median => :s_g1,
		:s_lerror_g1 => median => :mean_s_lerror_g1,
		:s_herror_g1 => median => :mean_s_herror_g1,
		:s_young_median_g1 => median => :mean_s_young_median_g1,
		:s_young_lerror_g1 => median => :mean_s_young_lerror_g1,
		:s_young_herror_g1 => median => :mean_s_young_herror_g1,
		:Vbar_g1 => median => :Vbar_g1,
		:freq_pb_g1 => median => :pb_g1,
		:freq_parochial_g1 => median => :paroch_g1,
	)

md"""
#### Figure 7 - Mixed populations
"""
	
end

# ╔═╡ f81601e9-a528-4359-8306-7111d542807d
begin
	function plot_mixed(datseris)
		
		delta_mixed = 0.6
		
		center_plot = levelplot(
					datseris,
					:λ,
					:μ,
					:soc_h_g0,
					colrange=(0.0,1.0),
					yticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
					colormap = cgrad(:YlOrRd),
					mask=true,
					delta=delta_mixed,
					ylab=L"\mathrm{E.\ peer\ influence}",
					ylabelfontsize=8,
					xticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
				)
		annotate!([1.4], [-0.5], text(L"\mathrm{aggregate\ uncertainty\ } (\lambda)", 15))
	
		mixedplot = plot(
			plot(
				levelplot(
					datseris,
					:λ,
					:μ,
					:paroch_g0,
					ylab=L"\mathrm{A.\ parochialism}",
					ylabelfontsize=8,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlOrRd),
					mask=true,
					delta=delta_mixed,
					xticks=false,
					yticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
					title=L"\mathrm{advantaged\ group}"
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:pb_g0,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlOrRd),
					mask=true,
					delta=delta_mixed,
					ylab=L"\mathrm{B.\ payoff\ bias}",
					ylabelfontsize=8,
					yticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
					xticks=false,
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:sens_g0,
					colrange=(0.0,1.0),
					yticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
					colormap = cgrad(:YlOrRd),
					mask=true,
					delta=delta_mixed,
					ylab=L"\mathrm{C.\ sensitivity}",
					ylabelfontsize=8,
					xticks=false,
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:soc_v_g0,
					colrange=(0.0,1.0),
					yticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
					colormap = cgrad(:YlOrRd),
					mask=true,
					delta=delta_mixed,
					ylab=L"\mathrm{D.\ elder\ influence}",
					ylabelfontsize=8,
					xticks=false,
				),
				center_plot,
				size=(1000,300), layout=(5,1), bottom_margin=6Plots.mm,
			),
			plot(
				levelplot(
					datseris,
					:λ,
					:μ,
					:paroch_g1,
					title=L"\mathrm{disadvantaged\ group}",
					ylabelfontsize=8,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=delta_mixed,
					cbarticks=(
						[1.4, 1.4, 1.4],
						[0.01, 0.5, 0.99],
						[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
					),
					xticks=false,
					yticks=false
					),
				levelplot(
					datseris,
					:λ,
					:μ,
					:pb_g1,
					colrange=(0.0,1.0),
					yticks=false,
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=delta_mixed,
					cbarticks=(
						[1.4, 1.4, 1.4],
						[0.01, 0.5, 0.99],
						[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
					),
					xticks=false,
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:sens_g1,
					colrange=(0.0,1.0),
					yticks=false,
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=delta_mixed,
					cbarticks=(
						[1.4, 1.4, 1.4],
						[0.01, 0.5, 0.99],
						[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
					),
					xticks=false,
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:soc_v_g1,
					colrange=(0.0,1.0),
					yticks=false,
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=delta_mixed,
					cbarticks=(
						[1.4, 1.4, 1.4],
						[0.01, 0.5, 0.99],
						[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
					),
					xticks=false,
				),
				levelplot(
					datseris,
					:λ,
					:μ,
					:soc_h_g1,
					colrange=(0.0,1.0),
					yticks=false,
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=delta_mixed,
					cbarticks=(
						[1.4, 1.4, 1.4],
						[0.01, 0.5, 0.99],
						[text(L"0.0", 8), text(L"0.5", 8), text(L"1.0", 8)]
					),
					xticks=(0.0:0.25:1.0, [L"%$a" for a in 0.0:0.25:1.0]),
				),
				size=(1000,300), bottom_margin=6Plots.mm, layout=(5,1)
			),
			layout=(1,2), size=(500,700), #left_margins=7Plots.mm
		)
	
		spacer2 = plot(
			1:0,
			xlim = (0.9, 1), ylim = (0, 1),
			axis = false, framestyle = :none, grid = false,
			legend = false, ticks = false
		)
		annotate!([0.9], [0.5], [text(L"\mathrm{idiosyncratic\ uncertainty\ } (\mu)", rotation=90)])
		
		mixedplot = plot(
			spacer2,
			mixedplot,
			layout = @layout([a{0.025w} b{0.9975w}]),
		)
	
	end

	mixedplot = plot_mixed(datseris)

	savefig(mixedplot, "../images/fig7_mixed.pdf")

	mixedplot
end

# ╔═╡ 59959294-880a-470e-b4c2-b2579a1c55fe
begin
	tticks = 2500
	
	modelio = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.55,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 75548897,
	)

	adata, mdata = run!(
		modelio, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio2 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.55,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 46577,
	)

	adata2, mdata2 = run!(
		modelio2, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio3 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.55,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 2445446,
	)

	adata3, mdata3 = run!(
		modelio3, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	modelio4 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.65,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 7554889,
	)

	adata4, mdata4 = run!(
		modelio4, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio5 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.65,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 1274245656,
	)

	adata5, mdata5 = run!(
		modelio5, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	modelio6 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.65,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 923445654,
	)

	adata6, mdata6 = run!(
		modelio6, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	md"""
	#### Figure 8 - Tails of risk taking
	"""
end

# ╔═╡ d9ed7f3b-1324-4468-aecd-1b6756e6f415
begin
	modelio7 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.75,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 7554889,
	)

	adata7, mdata7 = run!(
		modelio7, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio8 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.75,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 1274245656,
	)

	adata8, mdata8 = run!(
		modelio8, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	modelio9 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 10,
		u = 0.75,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 923445654,
	)

	adata9, mdata9 = run!(
		modelio9, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

md"""
#### Figure 4 - Population effects of elder influence
"""
end

# ╔═╡ 3ab29f6f-c949-43cf-a254-00d505a5313f
begin
	eldhist1 = histogram(
		[a.s_child for a in allagents(modelio)|>collect], 
		grid=false, 
		label="", 
		bins=10, xlim=(-0.01,0.65), 
		legendfontsize=12, 
		xticks=false,
		yticks=(
		[0, 200, 400, 600], 
		[L"%$(a)" for a in [0, 200, 400, 600]]
		),
		xlab="",
		xlabelfontsize=17,
		title=L"\epsilon = 0.05",
		ylab=L"\aleph = 0.05"
	)
	histogram!([a.s_young for a in allagents(modelio)|>collect], label="", alpha=0.7, bins=10)
	
	eldhist2 = histogram(
		[a.s_child for a in allagents(modelio2)|>collect], 
		grid=false, 
		legend=false,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=15, xlim=(-0.01,0.65), 
		legendfontsize=8, 
		xticks=false,
		yticks=(
		[0, 200, 400, 600], 
		[L"%$(a)" for a in [0, 200, 400, 600]]
		),
		#xlab=L"\mathrm{mean\ stake\ } (s)",
		xlabelfontsize=17,
		ylab=L"\aleph = 0.5"
	)
	histogram!([a.s_young for a in allagents(modelio2)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=3)

	eldhist3 = histogram(
		[a.s_child for a in allagents(modelio3)|>collect], 
		grid=false, 
		legend=:topright,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=10, xlim=(-0.01,0.65), 
		legendfontsize=8, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		yticks=(
		[0, 200, 400, 600], 
		[L"%$(a)" for a in [0, 200, 400, 600]]
		),
		xlab=" ",
		xlabelfontsize=17,
		ylab=L"\aleph = 0.95"
	)
	histogram!([a.s_young for a in allagents(modelio3)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=5)

	eldhist4 = histogram(
		[a.s_child for a in allagents(modelio4)|>collect], 
		grid=false, 
		label="", 
		bins=10, xlim=(-0.01,0.65), 
		legendfontsize=12, 
		xticks=false,
		yticks=false,
		xlab="",
		xlabelfontsize=17,
		title=L"\epsilon = 0.15"
	)
	histogram!([a.s_young for a in allagents(modelio4)|>collect], label="", alpha=0.7, bins=10)
	
	eldhist5 = histogram(
		[a.s_child for a in allagents(modelio5)|>collect], 
		grid=false, 
		legend=false,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=20, xlim=(-0.01,0.65), 
		legendfontsize=8, 
		xticks=false,
		yticks=false,
		#xlab=L"\mathrm{mean\ stake\ } (s)",
		xlabelfontsize=17,
		#title=L"\aleph = 0.5"
	)
	histogram!([a.s_young for a in allagents(modelio5)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=5)

	eldhist6 = histogram(
		[a.s_child for a in allagents(modelio6)|>collect], 
		grid=false, 
		legend=false,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=10, xlim=(-0.01,0.65), 
		legendfontsize=15, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		yticks=false,
		xlabelfontsize=20,
		#title=L"\aleph = 0.95",
		xlab=L"\mathrm{mean\ stake\ } (s)",
	)
	histogram!([a.s_young for a in allagents(modelio6)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=5)

	eldhist7 = histogram(
		[a.s_child for a in allagents(modelio7)|>collect], 
		grid=false, 
		label="", 
		bins=25, xlim=(-0.01,0.65), 
		legendfontsize=12, 
		xticks=false,
		yticks=false,
		xlab="",
		xlabelfontsize=17,
		title=L"\epsilon = 0.25"
	)
	histogram!([a.s_young for a in allagents(modelio7)|>collect], label="", alpha=0.7, bins=15)
	
	eldhist8 = histogram(
		[a.s_child for a in allagents(modelio8)|>collect], 
		grid=false, 
		legend=false,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=20, xlim=(-0.01,0.65), 
		legendfontsize=8, 
		xticks=false,
		yticks=false,
		#xlab=L"\mathrm{mean\ stake\ } (s)",
		xlabelfontsize=17,
		#title=L"\aleph = 0.5"
	)
	histogram!([a.s_young for a in allagents(modelio8)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=10)

	eldhist9 = histogram(
		[a.s_child for a in allagents(modelio9)|>collect], 
		grid=false, 
		legend=false,
		label=L"\mathrm{before\ elder\ influence}", 
		bins=15, xlim=(-0.01,0.65), 
		legendfontsize=15, 
		xticks=(0:0.25:0.5, [L"%$a" for a in 0:0.25:0.5]),
		yticks=false,
		xlab=" ",
		xlabelfontsize=17,
		#title=L"\aleph = 0.95"
	)
	histogram!([a.s_young for a in allagents(modelio9)|>collect], label=L"\mathrm{after\ elder\ influence}", alpha=0.7, bins=10)

	eld_pop = plot(
		plot(
			eldhist1,
			eldhist2,
			eldhist3,
			layout=(3,1),
			dpi=300, link=:all, 
			size=(600, 300), bottom_margins=2Plots.mm
		),
		plot(
			eldhist4,
			eldhist5,
			eldhist6,
			layout=(3,1),
			dpi=300, link=:all, 
			size=(600, 300), bottom_margins=2Plots.mm
		),
		plot(
			eldhist7,
			eldhist8,
			eldhist9,
			layout=(3,1),
			dpi=300, link=:all, 
			size=(600, 300), bottom_margins=2Plots.mm
		),
		layout=(1,3), link=:all, size=(650, 400)
	)

	savefig(eld_pop, "../images/fig4_eldpop.pdf")

	eld_pop
end

# ╔═╡ 6ac78a6a-aa27-4ea0-8e49-e469186dfce8
begin
	function plot_tails(dat; pal=:romaO10)
		
		dat3 = dat[
				dat.time .== 2500
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

		pbs = plot(
			1:0,
			xlabelfontsize = 18,
			ylabel = L"\mathrm{payoff\ bias\ frequency}",
			#xlabel = L"\mathrm{wealth\ buffer\ } (\aleph)",
			ylabelfontsize = 18,
			title = L"\mathrm{B.}",
			titlelocation=:left,
			legend = false,
			legendtitle = L"\epsilon",
			legendtitlefontsize=15,
			legendfontsize=12,
			label = "",
			ylim=(-0.01,0.5),
			grid=false,
			xticks=false,
			yticks=([0.0, 0.25], [L"0", L"0.25"]),
			size=(500, 300),
			dpi=300,
			margins=3Plots.mm
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.freq_pb, 
		    fillalpha=0.2, label = L"0.15", lw=3,
			color=palette(pal)[6],
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.freq_pb, 
		    fillalpha=0.2, label = L"0.10", lw=3,
			color=palette(pal)[3],
		)
		plot!(
			grouped_socv.aleph, grouped_socv.freq_pb, 
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=3,
		)
		
		stakes = plot(
			1:0,
			xlabelfontsize = 18,
			ylabel = L"\mathrm{stake\ } (s)",
			xlabel = L"\mathrm{wealth\ buffer\ } (\aleph)",
			ylabelfontsize = 18,
			title = L"\mathrm{C.}",
			titlelocation=:left,
			legend = (0.2, 0.9),
			legendtitle = L"\epsilon",
			legendtitlefontsize=15,
			legendfontsize=12,
			label = "",
			ylim=(-0.01,0.5),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25], [L"0", L"0.25"]),
			size=(500, 300),
			dpi=300,
			margins=3Plots.mm
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_s_median, 
			ribbon = (
			grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			),
		    fillalpha=0.6, label = L"0.15", lw=3,
			color=palette(pal)[6],
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_s_median, 
			ribbon = (
			grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			),
		    fillalpha=0.3, label = L"0.10", lw=3,
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

		plot(
			pbs,
			stakes,
			layout=(2,1)
		)
	end

	tailsplot = plot_tails(CSV.read("../data/analysis_1.csv", DataFrame))

	bins = 0.0:0.02:0.15
	bins2 = 0.0:0.05:0.4
	
	stakehist1 = histogram(
				[a.s_mean for a in allagents(modelio)],
				bins=bins,
				xlim=(-0.01,0.15),
				ylim=(0,800),
				color=palette(:managua10)[1],
				legend=false,
				yticks=(0:200:800, [L"%$a" for a in 0:200:800]),
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
				ylim=(0,800),
				bins=bins,
				color=palette(:managua10)[3],
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
				ylim=(0,800),
				color=palette(:managua10)[6],
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
				color=palette(:managua10)[1],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,800),
				xticks=(0.0:0.1:0.4, [L"%$a" for a in 0.0:0.1:0.4]),
				xtickfontsize=6,
				yticks=(0:200:800, [L"%$a" for a in 0:200:800]),
				ylabelfontsize = 18,
				ylabel=L"\epsilon = 0.15",
				title=" ",
				alpha=0.85
			)
	vline!([s_star(0.65, 0.05)], lw=2, color="black", ls=:dash)

	stakehist5 = histogram(
				[a.s_mean for a in allagents(modelio5)],
				bins=bins2,
				color=palette(:managua10)[3],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,800),
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
				color=palette(:managua10)[6],
				legend=false,
				xlim=(-0.025,0.4),
				ylim=(0,800),
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
			layout=(1,3), link=:all, grid=false, size=(600,300),
			plot_title=L"\mathrm{A.}", plot_titlelocation=:left
		),
		plot(
			stakehist4,
			stakehist5,
			stakehist6,
			layout=(1,3), link=:all, grid=false, size=(600,400)
		),
		layout=(2,1), margin=3Plots.mm
	)
	
	staketails = plot(stakedist, tailsplot, layout=(1,2), size=(900, 700))

	savefig(staketails, "../images/fig8_staketails.pdf")

	staketails
	
end

# ╔═╡ 88e18457-ae1b-4de6-958b-df4f407d0b98
md"""
#### Figure 9 - Life trajectories
"""

# ╔═╡ 43b13c02-86ac-4620-b62b-a81e98a173a4
begin
	model_life = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.05,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.0,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 45356784,
	)
	
	adata_life, mdata_life = run!(
		model_life, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	model_life2 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.5,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.0,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 3174556,
	)

	adata_life2, mdata_life2 = run!(
		model_life2, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	model_life3 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.95,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.0,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 56767564,
	)

	adata_life3, mdata_life3 = run!(
		model_life3, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	model_life4 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.05,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 45356784,
	)

	adata_life4, mdata_life4 = run!(
		model_life4, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	model_life5 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.5,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 3174556,
	)

	adata_life5, mdata_life5 = run!(
		model_life5, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	model_life6 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		aleph = 0.95,
		λ = 0.0,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 56767564,
	)

	adata_life6, mdata_life6 = run!(
		model_life6, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)
	
	function plot_life_trajectory(
		model; 
		c=palette(:managua10)[1], 
		up=false, 
		leg=false,
		xlab=L"\mathrm{adult\ timeline}",
		ylab=L"\mathrm{\Delta s}",
		xticks=(
			[0, 100], 
			[L"\tau", L"\tau + T"]
			),
		yticks=true
	)
		s_vecs = [a.s_vec for a in allagents(model)]
		transposed = [getindex.(s_vecs, i) for i in 1:length(s_vecs[1])]
		mean_trajectory = mean.(transposed)

		inc = [1.0]

		if !up
			d = -0.02
		else
			d = 0.03
		end
		
		for i in 2:length(mean_trajectory)
			if i > 1
				push!( inc, (mean_trajectory[i]/first(mean_trajectory)) )
			end
		end
		
		plot( 
			inc, 
			lw=2, color=c, 
			label=L"%$(model.aleph)",
			legend=leg,
			legendtitle=L"\aleph",
			legendtitlefontsize=12,
			legendfontsize=10,
			xlim=(-1,100),
			ylim=(0.2, 1.1),
			xlabel=xlab,
			xlabelfontsize=18,
			ylabel=ylab,
			ylabelfontsize=12,
			yticks= yticks ? (
			[0.25, 0.5, 0.75, 1.0], 
			[L"%$a" for a in [0.25, 0.5, 0.75, 1.0]]
			) : false,
			xticks=xticks,
			grid=false
		)
	end

	function plot_life_trajectory!(model; c=palette(:managua10)[1], up=false)
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
				push!(inc, (mean_trajectory[i]/first(mean_trajectory)))
			end
		end
		plot!( 
			inc, 
			lw=2, color=c, 
			label=L"%$(model.aleph)" 
		)
	end

	yeseld = plot_life_trajectory(model_life4, up=true, xlab="", ylab="", yticks=false),
	plot_life_trajectory!(model_life5, c=palette(:managua10)[3], up=true),
	plot_life_trajectory!(model_life6, c=palette(:managua10)[6])
	
	noeld = plot_life_trajectory(model_life, up=true, xlab="", leg=:right),
	plot_life_trajectory!(model_life2, c=palette(:managua10)[3], up=true),
	plot_life_trajectory!(model_life3, c=palette(:managua10)[6])
	annotate!([110], [0.1], [text(L"\mathrm{adult\ timeline}", 15)])

	full_lifeplot = plot(
		plot(
			noeld[1],
			title=L"\mathrm{A.\ no\ elder\ influence}",
			margins=4Plots.mm
		),
		plot(
			yeseld[1],
			title=L"\mathrm{B.\ with\ elder\ influence}",
			margins=4Plots.mm
		),
		bottom_margins=6Plots.mm, size=(650, 350),
		layout=(1,2)
	)

	savefig(full_lifeplot, "../images/fig9_lifeplot.pdf")

	full_lifeplot
end

# ╔═╡ cda2a3a4-3d1c-497b-9643-31ef1be4a370
md"""
#### Figure 10 - Poverty traps
"""

# ╔═╡ b5b58967-287f-47b5-99a4-81b008183204
begin
	tickers = 2500
	
	modelio14 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.1,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 75548897,
	)

	adata14, mdata14 = run!(
		modelio14, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio15 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.1,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 46577,
	)

	adata15, mdata15 = run!(
		modelio15, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio16 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.1,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 243534446,
	)

	adata16, mdata16 = run!(
		modelio16, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	beforebefore14 = median.([[[a.s_vec for a in allagents(modelio14)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.05)
	beforebefore15 = median.([[[a.s_vec for a in allagents(modelio15)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.5) 
	beforebefore16 = median.([[[a.s_vec for a in allagents(modelio16)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.95)

	step!(modelio14)

	step!(modelio15)

	step!(modelio16)

	before14 = median.([[[a.s_vec for a in allagents(modelio14)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.05)
	before15 = median.([[[a.s_vec for a in allagents(modelio15)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.5) 
	before16 = median.([[[a.s_vec for a in allagents(modelio16)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.95)

	modelio14.u = 0.95
	modelio14.aleph = 0.95
	step!(modelio14)

	modelio15.u = 0.95
	modelio15.aleph = 0.95
	step!(modelio15)

	modelio16.u = 0.95
	modelio16.aleph = 0.95
	step!(modelio16)

	after14 = median.([[[a.s_vec for a in allagents(modelio14)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.05)
	after15 = median.([[[a.s_vec for a in allagents(modelio15)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.5)
	after16 = median.([[[a.s_vec for a in allagents(modelio16)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.95)

	before_after10 = vcat(before14, after14)
	before_after11 = vcat(before15, after15)
	before_after12 = vcat(before16, after16)

	step!(modelio14)
	
	step!(modelio15)
	
	step!(modelio16)

	afterafter14 = median.([[[a.s_vec for a in allagents(modelio14)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.05)
	afterafter15 = median.([[[a.s_vec for a in allagents(modelio15)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.5)
	afterafter16 = median.([[[a.s_vec for a in allagents(modelio16)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.95)

	beaf14 = vcat(beforebefore14, before14, after14, afterafter14)
	beaf15 = vcat(beforebefore15, before15, after15, afterafter15)
	beaf16 = vcat(beforebefore16, before16, after16, afterafter16)

	poverty_trap_plot = plot( 
		beaf14, 
		legend=:topleft, 
		legendtitle=L"\aleph", 
		legendtitlefontsize=15,
		legendfontsize=13,
		label=L"0.05",
		color=palette(:managua10)[1],
		lw=2,
		ylab=L"\mathrm{median\ stake}",
		ylabelfontsize=15,
		xlab=L"\mathrm{adult\ lifetime\ (across\ generations)}",
		xlabelfontsize=15,
		xticks=([250, 750, 1250, 1750], [L"\epsilon = 0.15", L"\epsilon = 0.15", L"\epsilon = 0.45", L"\epsilon = 0.45"]),
		xtickfontsize=13,
		ylim=(0.0, 1.0),
		yticks=(0:0.2:1.4, [L"%$a" for a in 0:0.2:1.4]),
		grid=false,
		dpi=300
	)
	plot!( 
		beaf15, 
		label=L"0.5",
		color=palette(:managua10)[3],
		lw=2
	)
	plot!( 
		beaf16, 
		label=L"0.95",
		color=palette(:managua10)[6],
		lw=2, size=(600, 400)
	)
	annotate!([1250, 1750], [0.5, 0.5], [text(L"\aleph = 0.95"), text(L"\aleph = 0.95")])
	vline!([1000], lw=3, ls=:dash, color=:black, label="")
	vline!([501], lw=3, color=:gray, label="")
	vline!([1501], lw=3, color=:gray, label="")
	
	savefig(poverty_trap_plot, "../images/fig10_poverty.pdf")

	poverty_trap_plot
end

# ╔═╡ ec30c84e-42a0-420e-b717-1fb0f7f5defd
# ╠═╡ disabled = true
#=╠═╡
begin
	modelio17 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.95,
		λ = 0.3,
		μ = 0.1,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 75548897,
	)

	adata17, mdata17 = run!(
		modelio17, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio18 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.95,
		λ = 0.3,
		μ = 0.1,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 46577,
	)

	adata18, mdata18 = run!(
		modelio18, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio19 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.95,
		λ = 0.3,
		μ = 0.1,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tickers,
		seed = 243534446,
	)

	adata19, mdata19 = run!(
		modelio19, 
		tickers,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio17.λ = 0.0
	modelio18.λ = 0.0
	modelio19.λ = 0.0

	step!(modelio17, 2)
	step!(modelio18, 2)
	step!(modelio19, 2)
	
	beforebefore17 = modelio17.Vbar#median.([[[a.s_vec for a in allagents(modelio17)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.05)
	beforebefore18 = modelio18.Vbar#median.([[[a.s_vec for a in allagents(modelio18)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.5) 
	beforebefore19 = modelio19.Vbar#median.([[[a.s_vec for a in allagents(modelio19)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.95)

	step!(modelio17)
	step!(modelio18)
	step!(modelio19)

	before17 = modelio17.Vbar#median.([[[a.s_vec for a in allagents(modelio17)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.05)
	before18 = modelio18.Vbar#median.([[[a.s_vec for a in allagents(modelio18)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.5) 
	before19 = modelio19.Vbar#median.([[[a.s_vec for a in allagents(modelio19)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.65, 0.95)

	modelio17.u = 0.55
	modelio17.aleph = 0.05
	step!(modelio17)

	modelio18.u = 0.55
	modelio18.aleph = 0.05
	step!(modelio18)

	modelio19.u = 0.55
	modelio19.aleph = 0.05
	step!(modelio19)

	after17 = modelio17.Vbar#median.([[[a.s_vec for a in allagents(modelio17)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.05)
	after18 = modelio18.Vbar#median.([[[a.s_vec for a in allagents(modelio18)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.5)
	after19 = modelio19.Vbar#median.([[[a.s_vec for a in allagents(modelio19)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.95)

	before_after13 = vcat(before17, after17)
	before_after14 = vcat(before18, after18)
	before_after15 = vcat(before19, after19)

	step!(modelio17)
	step!(modelio18)
	step!(modelio19)

	afterafter17 = modelio17.Vbar#median.([[[a.s_vec for a in allagents(modelio17)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.05)
	afterafter18 = modelio18.Vbar#median.([[[a.s_vec for a in allagents(modelio18)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.5)
	afterafter19 = modelio19.Vbar#median.([[[a.s_vec for a in allagents(modelio19)|>collect][i][k] for i in 1:750] for k in 1:501]) #./ s_star.(0.95, 0.95)

	beaf17 = vcat(beforebefore17, before17, after17, afterafter17)
	beaf18 = vcat(beforebefore18, before18, after18, afterafter18)
	beaf19 = vcat(beforebefore19, before19, after19, afterafter19)

	poverty_trap_plot2 = plot( 
		beaf17, 
		legend=:topleft, 
		legendtitle=L"\aleph", 
		legendtitlefontsize=15,
		legendfontsize=13,
		label=L"0.05",
		color=palette(:managua10)[1],
		lw=2,
		ylab=L"\mathrm{median\ stake}",
		ylabelfontsize=15,
		xlab=L"\mathrm{adult\ lifetime\ (across\ generations)}",
		xlabelfontsize=15,
		xticks=([250, 750, 1250, 1750], [L"\epsilon = 0.15", L"\epsilon = 0.15", L"\epsilon = 0.45", L"\epsilon = 0.45"]),
		xtickfontsize=13,
		ylim=(0.0, 1.0),
		yticks=(0:0.2:1.4, [L"%$a" for a in 0:0.2:1.4]),
		grid=false,
		dpi=300
	)
	plot!( 
		beaf18, 
		label=L"0.5",
		color=palette(:managua10)[3],
		lw=2
	)
	plot!( 
		beaf19, 
		label=L"0.95",
		color=palette(:managua10)[6],
		lw=2, size=(600, 400)
	)
	annotate!([1250, 1750], [0.5, 0.5], [text(L"\aleph = 0.95"), text(L"\aleph = 0.95")])
	vline!([1000], lw=3, ls=:dash, color=:black, label="")
	vline!([501], lw=3, color=:gray, label="")
	vline!([1501], lw=3, color=:gray, label="")
	
	#savefig(poverty_trap_plot, "../images/fig10_poverty.pdf")

	#poverty_trap_plot
end
  ╠═╡ =#

# ╔═╡ 38ff7de5-a2dd-4e5e-9f05-494709d317d9
# ╠═╡ disabled = true
#=╠═╡
begin
	plot( 
		beaf17, 
		legend=:topleft, 
		legendtitle=L"\aleph", 
		legendtitlefontsize=10,
		legendfontsize=9,
		label=L"0.05",
		color=palette(:managua10)[1],
		lw=2,
		ylab=L"\mathrm{mean\ payoff}",
		ylabelfontsize=15,
		xlab=L"\mathrm{generations}",
		xlabelfontsize=15,
		xticks=([1.5, 2.5, 3.5], [L"\epsilon = 0.45", L"\epsilon = 0.05", L"\epsilon = 0.05"]),
		xtickfontsize=13,
		ylim=(0.6, 1.9),
		yticks=(0.8:0.2:1.4, [L"%$a" for a in 0.8:0.2:1.4]),
		grid=false,
		dpi=300
	)
	plot!( 
		beaf18, 
		label=L"0.5",
		color=palette(:managua10)[3],
		lw=2
	)
	plot!( 
		beaf19, 
		label=L"0.95",
		color=palette(:managua10)[6],
		lw=2, size=(600, 400)
	)
	annotate!([2.5, 3.5], [1.7, 1.7], [text(L"\aleph = 0.05"), text(L"\aleph = 0.05")])
	#vline!([1000], lw=3, ls=:dash, color=:black, label="")
	vline!([2], lw=3, ls=:dash, color=:black, label="")
	vline!([3], lw=3, color=:gray, label="")
end
  ╠═╡ =#

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
		[probruin_numeric(l, 0.1, 0.15, n=2000, seasons=2000) for l in 0.5:0.001:1.0],
		lw=2, c="black", alpha=0.5, label=""
	)

	plot!(
		0.5:0.001:1.0,
		[probruin(l, 0.5, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", label=L"0.5", legendtitle=L"\aleph", ls=:dash
	)
	plot!(
		0.5:0.001:1.0,
		[probruin_numeric(l, 0.5, 0.15, n=2000, seasons=2000) for l in 0.5:0.001:1.0],
		lw=2, c="black", alpha=0.5, ls=:dash, label=""
	)
	
	plot!(
		0.5:0.001:1.0,
		[probruin(l, 0.95, 0.15) for l in 0.5:0.001:1.0],
		lw=2, c="black", label=L"0.95", legendtitle=L"\aleph", ls=:dashdotdot
	)
	plot!(
		0.5:0.001:1.0,
		[probruin_numeric(l, 0.95, 0.15, n=2000, seasons=2000) for l in 0.5:0.001:1.0],
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

# ╔═╡ a0d7d4e6-e715-4aad-96d8-a1bc4ad7af88
begin
	incplot = plot(
		levelplot(
			a05,
			:λ,
			:μ,
			:inc,
			colrange=(0.5,1.1),
			show_colorbar=true,
			yticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
			),
			xticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
			),
			ytickfontsize=6,
			title=L"\aleph = 0.05",
			ylab=L"\mathrm{idiosyncratic\ uncertainty\ } (\mu)",
			ylabelfontsize=10,
			mask=true,
			delta=0.6,
			cbarticks=(
				[1.4, 1.425, 1.4],
				[0.0, 0.5, 1.0],
				[text(L"0.5", 8), text(L"0.75", 8), text(L"1.0", 8)]
			),
		),
		levelplot(
			a50,
			:λ,
			:μ,
			:inc,
			colrange=(0.0,1.1),
			show_colorbar=true,
			yticks=false,
			xticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
			),
			title=L"\aleph = 0.5",
			xlab=L"\mathrm{aggregate\ uncertainty\ } (\lambda)",
			xlabelfontsize=10,
			delta=0.6,
			cbarticks=(
				[1.4, 1.425, 1.4],
				[0.0, 0.5, 1.0],
				[text(L"0.5", 8), text(L"0.75", 8), text(L"1.0", 8)]
			),
		),
		levelplot(
			a95,
			:λ,
			:μ,
			:inc,
			colrange=(0.0,1.1),
			show_colorbar=true,
			yticks=false,
			xticks=(
				[0.0, 0.25, 0.5, 0.75, 1.0], 
				[L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]
			),
			title=L"\aleph = 0.95",
			cbarticks=(
				[1.4, 1.425, 1.4],
				[0.0, 0.5, 1.0],
				[text(L"0.5", 8), text(L"0.75", 8), text(L"1.0", 8)]
			),
			delta=0.6,
		),
		layout=(1,3), margins=5Plots.mm, size=(800,200)
	)

	savefig(incplot, "../images/sup2_incplot.pdf")

	incplot
end

# ╔═╡ 077c7aec-998a-4295-9426-85d254d251f7
md"""
#### Figure S3
"""

# ╔═╡ 9e373cb0-dc3c-4834-ac88-d12d13fd1fa0
begin
	savefig(
		plot_mixed(datseris2), 
		"../images/sup3_ineqplot2.pdf"
	)
	
	plot_mixed(datseris2)
end

# ╔═╡ 656b3a41-1ceb-4637-af43-81b3e4d25832
md"""
#### Figure S4
"""

# ╔═╡ f82a9c5f-b209-41f6-8360-7795b407f077
begin
	savefig(
		plot_mixed(datseris3), 
		"../images/sup4_ineqplot3.pdf"
	)
	
	plot_mixed(datseris3)
end

# ╔═╡ 721c05e8-3205-4d6e-af11-d7648e9f95fc
# ╠═╡ disabled = true
#=╠═╡
begin
	dat_inc1 = CSV.read("../data/analysis_1c.csv", DataFrame)
	dat_inc1 = dat_inc1[dat_inc1.time .== 2500, :]
	dat_inc1.soc_h_median = dat_inc1.soc_h_median .* (1 .- dat_inc1.soc_v_median)
	dat_inc1.soc_h_lerror = dat_inc1.soc_h_lerror .* (1 .- dat_inc1.soc_v_lerror)
	dat_inc1.soc_h_herror = dat_inc1.soc_h_herror .* (1 .- dat_inc1.soc_v_herror)
	dat_inc1 = combine(
		groupby(dat_inc1, [:aleph, :λ]),
		:mean_increment => median => :inc,
		:sbar => median => :sbar,
		:soc_h_median => median => :soc_h,
		:soc_h_lerror => median => :soc_h_lerror,
		:soc_h_herror => median => :soc_h_herror,
		:soc_v_median => median => :soc_v,
		:soc_v_lerror => median => :soc_v_lerror,
		:soc_v_herror => median => :soc_v_herror,
		:sens_median => median => :sens,
		:sens_lerror => median => :sens_lerror,
		:sens_herror => median => :sens_herror,
		:freq_pb => median => :pb,
		:s_median => median => :s,
		:s_lerror => median => :s_lerror,
		:s_herror => median => :s_herror,
		:s_ltail => median => :s_ltail,
		:s_htail => median => :s_htail,
	)
	
plot(
	levelplot(
		dat_inc1,
		:aleph,
		:λ,
		:soc_v,
		colormap = cgrad(:YlOrRd),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc1,
		:aleph,
		:λ,
		:soc_h,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc1,
		:aleph,
		:λ,
		:sens,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc1,
		:aleph,
		:λ,
		:pb,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),
	layout=(2,2)
)
end
  ╠═╡ =#

# ╔═╡ c67685eb-1018-4b18-86db-a8e5c7828e22
# ╠═╡ disabled = true
#=╠═╡
begin
	dat_inc2 = CSV.read("../data/analysis_1b.csv", DataFrame)
	dat_inc2 = dat_inc2[dat_inc2.time .== 2500, :]
	dat_inc2.soc_h_median = dat_inc2.soc_h_median .* (1 .- dat_inc2.soc_v_median)
	dat_inc2.soc_h_lerror = dat_inc2.soc_h_lerror .* (1 .- dat_inc2.soc_v_lerror)
	dat_inc2.soc_h_herror = dat_inc2.soc_h_herror .* (1 .- dat_inc2.soc_v_herror)
	dat_inc2 = combine(
		groupby(dat_inc2, [:aleph, :μ]),
		:mean_increment => median => :inc,
		:sbar => median => :sbar,
		:soc_h_median => median => :soc_h,
		:soc_h_lerror => median => :soc_h_lerror,
		:soc_h_herror => median => :soc_h_herror,
		:soc_v_median => median => :soc_v,
		:soc_v_lerror => median => :soc_v_lerror,
		:soc_v_herror => median => :soc_v_herror,
		:sens_median => median => :sens,
		:sens_lerror => median => :sens_lerror,
		:sens_herror => median => :sens_herror,
		:freq_pb => median => :pb,
		:s_median => median => :s,
		:s_lerror => median => :s_lerror,
		:s_herror => median => :s_herror,
		:s_ltail => median => :s_ltail,
		:s_htail => median => :s_htail,
	)
	
plot(
	levelplot(
		dat_inc2,
		:aleph,
		:μ,
		:soc_v,
		colormap = cgrad(:YlOrRd),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc2,
		:aleph,
		:μ,
		:soc_h,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc2,
		:aleph,
		:μ,
		:sens,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),

	levelplot(
		dat_inc2,
		:aleph,
		:μ,
		:pb,
		colormap = cgrad(:YlGnBu),
		colrange=(0.0,1.0)
	),
	layout=(2,2)
)
end
  ╠═╡ =#

# ╔═╡ 0133b3f6-b7e0-400b-a96f-6dfef71590c9
# ╠═╡ disabled = true
#=╠═╡
begin
	dat_agg = CSV.read("../data/analysis_1c_reduced.csv", DataFrame)
	dat_agg = dat_agg[dat_agg.time .== 2500, :]
	dat_agg.soc_h_median = dat_agg.soc_h_median .* (1 .- dat_agg.soc_v_median)
	dat_agg.soc_h_lerror = dat_agg.soc_h_lerror .* (1 .- dat_agg.soc_v_lerror)
	dat_agg.soc_h_herror = dat_agg.soc_h_herror .* (1 .- dat_agg.soc_v_herror)
	dat_agg = combine(
		groupby(dat_agg, [:aleph]),
		:mean_increment => mean => :inc,
		:sbar => median => :sbar,
		:soc_h_median => median => :soc_h,
		:soc_h_lerror => median => :soc_h_lerror,
		:soc_h_herror => median => :soc_h_herror,
		:soc_v_median => median => :soc_v,
		:soc_v_lerror => median => :soc_v_lerror,
		:soc_v_herror => median => :soc_v_herror,
		:sens_median => median => :sens,
		:sens_lerror => median => :sens_lerror,
		:sens_herror => median => :sens_herror,
		:freq_pb => median => :pb,
		:s_median => median => :s,
		:s_lerror => median => :s_lerror,
		:s_herror => median => :s_herror,
		:s_ltail => median => :s_ltail,
		:s_htail => median => :s_htail,
	)
	
	dat_id = CSV.read("../data/analysis_1b_reduced.csv", DataFrame)
	dat_id = dat_id[dat_id.time .== 2500, :]
	dat_id.soc_h_median = dat_id.soc_h_median .* (1 .- dat_id.soc_v_median)
	dat_id.soc_h_lerror = dat_id.soc_h_lerror .* (1 .- dat_id.soc_v_lerror)
	dat_id.soc_h_herror = dat_id.soc_h_herror .* (1 .- dat_id.soc_v_herror)
	dat_id = combine(
		groupby(dat_id, [:aleph]),
		:mean_increment => mean => :inc,
		:sbar => median => :sbar,
		:soc_h_median => median => :soc_h,
		:soc_h_lerror => median => :soc_h_lerror,
		:soc_h_herror => median => :soc_h_herror,
		:soc_v_median => median => :soc_v,
		:soc_v_lerror => median => :soc_v_lerror,
		:soc_v_herror => median => :soc_v_herror,
		:sens_median => median => :sens,
		:sens_lerror => median => :sens_lerror,
		:sens_herror => median => :sens_herror,
		:freq_pb => median => :pb,
		:s_median => median => :s,
		:s_lerror => median => :s_lerror,
		:s_herror => median => :s_herror,
		:s_ltail => median => :s_ltail,
		:s_htail => median => :s_htail,
	)

	function plot_agg(dat1, dat2)
		sens_change = plot(
			dat1.aleph,
			dat1.sens,
			ribbon=(
			dat1.sens .- dat1.sens_lerror,
			dat1.sens_herror .- dat1.sens
			),
			ylim=(0.0, 1.0),
			#xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			#yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
			xticks=false,
			yticks=false,
			lw=2,
			grid=false, label="",
			title=L"\mathrm{sensitivity\ } (\delta)",
			color=palette(:batlow10)[1]
		)
		plot!(
			dat2.aleph,
			dat2.sens,
			ribbon=(
			dat2.sens .- dat2.sens_lerror,
			dat2.sens_herror .- dat2.sens
			),
			ylim=(0.0, 1.0),
			lw=2,
			grid=false, label="",
			title=L"\mathrm{sensitivity\ } (\delta)",
			color=palette(:batlow10)[8]
		)

		soch_change = plot(
			dat1.aleph,
			dat1.soc_h,
			ribbon=(
			dat1.soc_h .- dat1.soc_h_lerror,
			dat1.soc_h_herror .- dat1.soc_h
			),
			ylim=(0.0, 1.0),
			#xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
			xticks=false,
			lw=2,
			grid=false, #label=L"\mathrm{aggregate\ risk}", 
			label="",
			title=L"\mathrm{peer\ influence\ } (\tilde\beta)",
			color=palette(:batlow10)[1]
		)
		plot!(
			dat2.aleph,
			dat2.soc_h,
			ribbon=(
			dat2.soc_h .- dat2.soc_h_lerror,
			dat2.soc_h_herror .- dat2.soc_h
			),
			lw=2,
			grid=false,
			label="",
			color=palette(:batlow10)[8]
		)

		socv_change = plot(
			dat1.aleph,
			dat1.soc_v,
			ribbon=(
			dat1.soc_v .- dat1.soc_v_lerror,
			dat1.soc_v_herror .- dat1.soc_v
			),
			ylim=(0.0, 1.0),
			#xticks=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			yticks=false,
			lw=2,
			grid=false, label="",
			title=L"\mathrm{elder\ influence\ } (\alpha)",
			color=palette(:batlow10)[1]
		)
		plot!(
			dat2.aleph,
			dat2.soc_v,
			ribbon=(
			dat2.soc_v .- dat2.soc_v_lerror,
			dat2.soc_v_herror .- dat2.soc_v
			),
			lw=2, label="",
			color=palette(:batlow10)[8]
		)

		pb_change = plot(
			dat1.aleph,
			dat1.pb,
			ylim=(0.0, 1.0),
			xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
			#yticks=false,
			lw=2,
			legendfontsize=10,
			grid=false, #label="",
			title=L"\mathrm{payoff\ bias\ frequency}",
			label=L"\mathrm{aggregate\ uncertainty}",
			color=palette(:batlow10)[1]
		)
		plot!(
			dat2.aleph,
			dat2.pb,
			ylim=(0.0, 1.0),
			xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			lw=2, #label="",
			label=L"\mathrm{idiosyncratic\ uncertainty}",
			color=palette(:batlow10)[8]
		)
		annotate!([1.05], [-0.225], [text(L"\mathrm{risk\ buffer\ } (\aleph)", 15)])
		
		plot(
			soch_change,
			sens_change,
			pb_change,
			socv_change,
			layout=(2,2), dpi=300,
			bottom_margins=5Plots.mm
		)
	end

	plotagg = plot_agg(dat_agg, dat_id)

	savefig(plotagg, "../images/sup3_agg_ind_uncertainty.pdf")
	
	plotagg
end
  ╠═╡ =#

# ╔═╡ f4979b22-a419-4473-809f-620ad3125057
# ╠═╡ disabled = true
#=╠═╡
begin
	socv_change = plot(
		dat_agg.aleph,
		dat_agg.soc_v,
		ribbon=(
		dat_agg.soc_v .- dat_agg.soc_v_lerror,
		dat_agg.soc_v_herror .- dat_agg.soc_v
		),
		ylim=(0.0, 1.0),
		#xticks=false,
		xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
		yticks=false,
		lw=2,
		grid=false,
		title=L"\mathrm{elder\ influence\ } (\alpha)",
		label=L"\mathrm{aggregate\ uncertainty}",
		legendfontsize=10,
		color=palette(:batlow10)[1]
	)
	plot!(
		dat_id.aleph,
		dat_id.soc_v,
		ribbon=(
		dat_id.soc_v .- dat_id.soc_v_lerror,
		dat_id.soc_v_herror .- dat_id.soc_v
		),
		lw=2, label=L"\mathrm{idiosyncratic\ uncertainty}",
		color=palette(:batlow10)[8]
	)

	soch_change = plot(
		dat_agg.aleph,
		dat_agg.soc_h,
		ribbon=(
		dat_agg.soc_h .- dat_agg.soc_h_lerror,
		dat_agg.soc_h_herror .- dat_agg.soc_h
		),
		ylim=(0.0, 1.0),
		xticks=([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
		yticks=([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
		lw=2,
		grid=false, #label=L"\mathrm{aggregate\ risk}", 
		label="",
		title=L"\mathrm{peer\ influence\ } (\tilde\beta)",
		color=palette(:batlow10)[1]
	)
	plot!(
		dat_id.aleph,
		dat_id.soc_h,
		ribbon=(
		dat_id.soc_h .- dat_id.soc_h_lerror,
		dat_id.soc_h_herror .- dat_id.soc_h
		),
		lw=2,
		grid=false,
		label="",
		color=palette(:batlow10)[8]
	)
	annotate!([1.025], [-0.1], [text(L"\mathrm{wealth\ buffer\ } (\aleph)", 15)])

	plot(soch_change, socv_change, bottom_margins=5Plots.mm, dpi=300)
end
  ╠═╡ =#

# ╔═╡ d5a60bb0-4e82-48d4-84da-0c762e5da660
# ╠═╡ disabled = true
#=╠═╡
begin
	function splot_agg(
		dat_inc_agg; 
		color1=palette(:batlow10)[1], 
		color2=:black, 
		colorline=palette(:batlow10)[1],
		title=true,
		xlab=true,
		ylab="",
		xlabheight=0.2
	)
		splot_agg = plot(
			dat_inc_agg.aleph,
			dat_inc_agg.s,
			ribbon = (
			dat_inc_agg.s .- dat_inc_agg.s_lerror,
			dat_inc_agg.s_herror .- dat_inc_agg.s
			),
			ylim = (0,1),
			ylab = ylab,
			lw = 3,
			color = color1,
			label = "", grid=false,
			xticks = ([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			yticks = ([0.0, 0.25, 0.5, 0.75, 1.0], [L"%$a" for a in [0.0, 0.25, 0.5, 0.75, 1.0]]),
			title = title ? L"\mathrm{mean\ stake\ } (\bar{s}\ )" : "",
			ylabelfontsize=12,
		)
		plot!(
			dat_inc_agg.aleph,
			dat_inc_agg.s,
			ribbon = (
			dat_inc_agg.s .- dat_inc_agg.s_ltail,
			dat_inc_agg.s_htail .- dat_inc_agg.s
			),
			lw = 3,
			color = color1,
			label = ""
		)
		plot!(
			dat_inc_agg.aleph,
			dat_inc_agg.s,
			lw=3,
			label="",
			color=colorline
		)
		if xlab annotate!([1.075], [-xlabheight], [text(L"\mathrm{risk\ buffer\ } (\aleph)", 17)]) end
	
		inc_agg = plot(
			dat_inc_agg.aleph,
			dat_inc_agg.inc,
			lw = 3,
			color = color2,
			label = "", grid=false,
			xticks = ([0.2, 0.4, 0.6, 0.8], [L"%$a" for a in [0.2, 0.4, 0.6, 0.8]]),
			yticks = ([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2], [L"%$a" for a in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]]),
			ylim = (0.49,1.2),
			title = title ? L"\mathrm{mean\ stake\ change\ } (\Delta \bar{s})" : "",
			ylabelfontsize=12,
		)
		hline!([1.0], ls=:dash, lw=2, color=:black, label="")
	
		plot(
			splot_agg,
			inc_agg,
			layout=(1,2),
			bottom_margins=6Plots.mm,
			size=(600, 300),
		)
	end

	splot1 = splot_agg(
		dat_agg,
		xlab=false,
		ylab=L"\mathrm{aggregate\ risk}"
	)

	splot2 = splot_agg(
		dat_id,
		ylab=L"\mathrm{idiosyncratic\ risk}",
		color1=palette(:batlow10)[8],
		color2=:black,
		colorline=palette(:batlow10)[8],
		title=false
	)

	plot(
		splot1,
		splot2,
		layout=(2,1),
		size=(600,500), dpi=300
	)
end
  ╠═╡ =#

# ╔═╡ 3dc40146-3fdb-4d00-a288-571c47a45041
# ╠═╡ disabled = true
#=╠═╡
begin
	
	function aggregate_adata(d)
		d[!, :pb] = d.L .== 3
		gadata = groupby(d, [:time])
		gadata = combine(
			gadata, 
			:soc_h => median => :soc_h,
			:soc_v => median => :soc_v,
			:sens => median => :sens,
			:pb => mean => :pb,
		)
	end
	
	
	timeplot1 = plot(
		aggregate_adata(adata).time,
		aggregate_adata(adata).soc_v,
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
		aggregate_adata(adata).time,
		aggregate_adata(adata).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata).time,
		aggregate_adata(adata).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata).time,
		aggregate_adata(adata).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot2 = plot(
		aggregate_adata(adata2).time,
		aggregate_adata(adata2).soc_v,
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
		aggregate_adata(adata2).time,
		aggregate_adata(adata2).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata2).time,
		aggregate_adata(adata2).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata2).time,
		aggregate_adata(adata2).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot3 = plot(
		aggregate_adata(adata3).time,
		aggregate_adata(adata3).soc_v,
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
		aggregate_adata(adata3).time,
		aggregate_adata(adata3).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata3).time,
		aggregate_adata(adata3).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata3).time,
		aggregate_adata(adata3).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot4 = plot(
		aggregate_adata(adata4).time,
		aggregate_adata(adata4).soc_v,
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
		aggregate_adata(adata4).time,
		aggregate_adata(adata4).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata4).time,
		aggregate_adata(adata4).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata4).time,
		aggregate_adata(adata4).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot5 = plot(
		aggregate_adata(adata5).time,
		aggregate_adata(adata5).soc_h,
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
		aggregate_adata(adata5).time,
		aggregate_adata(adata5).soc_v,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata5).time,
		aggregate_adata(adata5).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata5).time,
		aggregate_adata(adata5).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot6 = plot(
		aggregate_adata(adata6).time,
		aggregate_adata(adata6).soc_v,
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
		aggregate_adata(adata6).time,
		aggregate_adata(adata6).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata6).time,
		aggregate_adata(adata6).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata6).time,
		aggregate_adata(adata6).pb,
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
	
	#savefig(timeplots_full, "../images/sup3_time.pdf")

	timeplots_full
end
  ╠═╡ =#

# ╔═╡ ece62961-7745-49ac-9f89-3a37544c1786
# ╠═╡ disabled = true
#=╠═╡
begin
	lambda = 0.1
	
	modelio57 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = 0.0,
		λ = lambda,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 748897,
	)

	adata57, mdata57 = run!(
		modelio57, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio58 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = 0.0,
		λ = lambda,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 456745677,
	)

	adata58, mdata58 = run!(
		modelio58, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio59 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = 0.0,
		λ = lambda,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 244544646,
	)

	adata59, mdata59 = run!(
		modelio59, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio60 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.0,
		λ = lambda,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 43343897,
	)

	adata60, mdata60 = run!(
		modelio60, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio61 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.0,
		λ = lambda,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 460946577,
	)

	adata61, mdata61 = run!(
		modelio61, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio62 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = 0.0,
		λ = lambda,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 24446446,
	)

	adata62, mdata62 = run!(
		modelio62, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	timeplot13 = plot(
		aggregate_adata(adata57).time,
		aggregate_adata(adata57).soc_v,
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
		aggregate_adata(adata57).time,
		aggregate_adata(adata57).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata57).time,
		aggregate_adata(adata57).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata57).time,
		aggregate_adata(adata57).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot14 = plot(
		aggregate_adata(adata58).time,
		aggregate_adata(adata58).soc_v,
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
		aggregate_adata(adata58).time,
		aggregate_adata(adata58).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata58).time,
		aggregate_adata(adata58).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata58).time,
		aggregate_adata(adata58).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot15 = plot(
		aggregate_adata(adata59).time,
		aggregate_adata(adata59).soc_v,
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
		aggregate_adata(adata59).time,
		aggregate_adata(adata59).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata59).time,
		aggregate_adata(adata59).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata59).time,
		aggregate_adata(adata59).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot16 = plot(
		aggregate_adata(adata60).time,
		aggregate_adata(adata60).soc_v,
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
		aggregate_adata(adata60).time,
		aggregate_adata(adata60).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata60).time,
		aggregate_adata(adata60).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata60).time,
		aggregate_adata(adata60).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot17 = plot(
		aggregate_adata(adata61).time,
		aggregate_adata(adata61).soc_h,
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
		aggregate_adata(adata61).time,
		aggregate_adata(adata61).soc_v,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata61).time,
		aggregate_adata(adata61).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata61).time,
		aggregate_adata(adata61).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot18 = plot(
		aggregate_adata(adata62).time,
		aggregate_adata(adata62).soc_v,
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
		aggregate_adata(adata62).time,
		aggregate_adata(adata62).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata62).time,
		aggregate_adata(adata62).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata62).time,
		aggregate_adata(adata62).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplots5 = plot(
		timeplot13,
		timeplot14, 
		timeplot15,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)
	
	timeplots6 = plot(
		timeplot16,
		timeplot17, 
		timeplot18,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)
	
	timeplots_full3 = plot(
		timeplots5,
		timeplots6,
		layout=(2,1),
		size=(700,600),
		dpi=300
	)
	
	#savefig(timeplots_full, "../images/sup3_time.pdf")

	timeplots_full3

end
  ╠═╡ =#

# ╔═╡ 9e0cf19c-d66a-448b-b3fc-8ffbb99ce22a
# ╠═╡ disabled = true
#=╠═╡
begin
	mu = 0.25
	
	modelio50 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = mu,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 75548897,
	)

	adata50, mdata50 = run!(
		modelio50, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio52 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = mu,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 46577,
	)

	adata52, mdata52 = run!(
		modelio52, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio53 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.55,
		μ = mu,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 2445446,
	)

	adata53, mdata53 = run!(
		modelio53, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio54 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = mu,
		aleph = 0.05,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.01,
		mu_soc_v = 0.01,
		mu_sens = 0.01,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 75548897,
	)

	adata54, mdata54 = run!(
		modelio54, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio55 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = mu,
		aleph = 0.5,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 46577,
	)

	adata55, mdata55 = run!(
		modelio55, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	modelio56 = initialize_pessimistic_learning(
		N = 750,
		T = 500,
		m = 10,
		n = 10,
		t = 15,
		u = 0.65,
		μ = mu,
		aleph = 0.95,
		#EVOLUTION PARAMETERS
		mu_std = 0.01,
		mu_soc_h = 0.001,
		mu_soc_v = 0.001,
		mu_sens = 0.001,
		mu_L = 0.01,
		mu_parochial = 0.0,
		strategies = "UB&PB",
		total_ticks = tticks,
		seed = 2445446,
	)

	adata56, mdata56 = run!(
		modelio56, 
		tticks,
		adata=[:s, :soc_h, :soc_v, :sens, :L],
		mdata=[
			:Vbar, 
			:s_median, :s_lerror, :s_herror, :s_ltail, :s_htail,
			:s_young_median, :s_child_median, 
			:soc_v_median, :soc_v_lerror, :soc_v_herror, 
			:soc_h_median, :soc_h_lerror, :soc_h_herror,
			:sens_median, :sens_lerror, :sens_herror,
			:sbar, :mean_increment, :soc_v_median_g0, :soc_h_median_g0, 
			:sens_median_g0, :soc_v_median_g1, :soc_h_median_g1, :sens_median_g1,
			:freq_ub, :freq_pb, :freq_cb, :freq_cb_g0, :freq_cb_g1,
			:freq_ub_g0, :freq_ub_g1, :freq_pb_g0, :freq_pb_g1, :freq_parochial_g0, :freq_parochial_g1
		]
	)

	timeplot7 = plot(
		aggregate_adata(adata50).time,
		aggregate_adata(adata50).soc_v,
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
		aggregate_adata(adata50).time,
		aggregate_adata(adata50).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata50).time,
		aggregate_adata(adata50).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata50).time,
		aggregate_adata(adata50).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot8 = plot(
		aggregate_adata(adata52).time,
		aggregate_adata(adata52).soc_v,
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
		aggregate_adata(adata52).time,
		aggregate_adata(adata52).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata52).time,
		aggregate_adata(adata52).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata52).time,
		aggregate_adata(adata52).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot9 = plot(
		aggregate_adata(adata53).time,
		aggregate_adata(adata53).soc_v,
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
		aggregate_adata(adata53).time,
		aggregate_adata(adata53).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata53).time,
		aggregate_adata(adata53).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata53).time,
		aggregate_adata(adata53).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot10 = plot(
		aggregate_adata(adata54).time,
		aggregate_adata(adata54).soc_v,
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
		aggregate_adata(adata54).time,
		aggregate_adata(adata54).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata54).time,
		aggregate_adata(adata54).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata54).time,
		aggregate_adata(adata54).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
		
	timeplot11 = plot(
		aggregate_adata(adata55).time,
		aggregate_adata(adata55).soc_h,
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
		aggregate_adata(adata55).time,
		aggregate_adata(adata55).soc_v,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata55).time,
		aggregate_adata(adata55).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata55).time,
		aggregate_adata(adata55).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplot12 = plot(
		aggregate_adata(adata56).time,
		aggregate_adata(adata56).soc_v,
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
		aggregate_adata(adata56).time,
		aggregate_adata(adata56).soc_h,
		label=L"\beta",
		lw=2,
		color="brown",
		alpha=0.75,
	)
	plot!(
		aggregate_adata(adata56).time,
		aggregate_adata(adata56).sens,
		label=L"\delta",
		color="dark green",
		alpha=0.25,
		lw=2,
	)
	plot!(
		aggregate_adata(adata56).time,
		aggregate_adata(adata56).pb,
		label=L"\mathrm{PB}",
		color="purple",
		alpha=0.15,
		lw=2,
	)
	
	timeplots3 = plot(
		timeplot7,
		timeplot8, 
		timeplot9,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)
	
	timeplots4 = plot(
		timeplot10,
		timeplot11, 
		timeplot12,
		layout=(1,3),
		size=(1000,400),
		#margins=6Plots.mm
	)
	
	timeplots_full2 = plot(
		timeplots3,
		timeplots4,
		layout=(2,1),
		size=(700,600),
		dpi=300
	)
	
	#savefig(timeplots_full, "../images/sup3_time.pdf")

	timeplots_full2

end
  ╠═╡ =#

# ╔═╡ 8a2c7c0c-e59d-4727-ab06-c0367298fb8b
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ aef1aaeb-5cd1-4121-81ab-490a24a89af0
# ╠═╡ disabled = true
#=╠═╡
begin
	function plot_elder_portfolio(dat; pal=:managua10, stakeplot=true, n=true)

		if n
			dat3 = dat[
					dat.time .== 2500 .&&
					dat.n .== 10 .&&
					dat.m .== 10
					,:]
		else
			dat3 = dat[
					dat.time .== 2500
					,:]
		end

		dat3.soc_h_norm = dat3.soc_h_median .* (1 .- dat3.soc_v_median)
		dat3.soc_h_lerror_norm = dat3.soc_h_lerror .* (1 .- dat3.soc_v_lerror)
		dat3.soc_h_herror_norm = dat3.soc_h_herror .* (1 .- dat3.soc_v_herror)
		
		
		mdat3_1 = dat3[dat3.u .== 0.55, :]
		mdat3_2 = dat3[dat3.u .== 0.6, :]
		mdat3_3 = dat3[dat3.u .== 0.65, :]
		mdat3_4 = dat3[dat3.u .== 0.7, :]
	
		grouped_socv = combine(
			groupby(mdat3_1, :aleph), 
		    :soc_h_median => mean => :mean_soc_h_median,
			:soc_h_lerror => mean => :mean_soc_h_lerror,
			:soc_h_herror => mean => :mean_soc_h_herror,
			:soc_h_norm => mean => :soc_h_median,
			:soc_h_lerror_norm => mean => :soc_h_lerror,
			:soc_h_herror_norm => mean => :soc_h_herror,
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
			:soc_h_norm => mean => :soc_h_median,
			:soc_h_lerror_norm => mean => :soc_h_lerror,
			:soc_h_herror_norm => mean => :soc_h_herror,
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
			:soc_h_norm => mean => :soc_h_median,
			:soc_h_lerror_norm => mean => :soc_h_lerror,
			:soc_h_herror_norm => mean => :soc_h_herror,
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
			:soc_h_norm => mean => :soc_h_median,
			:soc_h_lerror_norm => mean => :soc_h_lerror,
			:soc_h_herror_norm => mean => :soc_h_herror,
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
		    grouped_socv4.aleph, grouped_socv4.soc_h_median, 
			ribbon = (
			grouped_socv4.soc_h_median .- grouped_socv4.soc_h_lerror, grouped_socv4.soc_h_herror .- grouped_socv4.soc_h_median
			),
			fillalpha=0.2,
			xlabelfontsize = 15,
			color=palette(pal)[8],
		    ylabel = L"\mathrm{peer\ influence\ } (\tilde{\beta})",
			ylabelfontsize = 12,
		    title = "",
			legend = false,
			legendtitle = L"\bar{u}",
		    label = L"0.6",
			lw=2,
			ylim=(0,0.5),
			grid=false,
			xticks=([0.2, 0.4, 0.6, 0.8], [L"0.2", L"0.4", L"0.6", L"0.8"]),
			yticks=([0.0, 0.25, 0.5], [L"0", L"0.25", L"0.5"])
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.soc_h_median, 
			ribbon = (
			grouped_socv3.soc_h_median .- grouped_socv3.soc_h_lerror, grouped_socv3.soc_h_herror .- grouped_socv3.soc_h_median
			),
			color=palette(pal)[6],
		    fillalpha=0.2, lw=2,
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.soc_h_median, 
			ribbon = (
			grouped_socv2.soc_h_median .- grouped_socv2.soc_h_lerror, grouped_socv2.soc_h_herror .- grouped_socv2.soc_h_median
			),
			color=palette(pal)[3],
		    fillalpha=0.2, lw=2,
		)
		plot!(
		    grouped_socv.aleph, grouped_socv.soc_h_median, 
			ribbon = (
			grouped_socv.soc_h_median .- grouped_socv.soc_h_lerror, grouped_socv.soc_h_herror .- grouped_socv.soc_h_median
			),
			color=palette(pal)[1],
		    fillalpha=0.2, lw=2,
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
			label = L"0.2",
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
		    fillalpha=0.2, label = L"0.1", lw=2,
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
			ylim=(-0.05,0.4),
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
		plot!(
			grouped_socv4.aleph, s_star.(0.7, grouped_socv4.aleph), 
			fillalpha=0.2, label = "", lw=2, ls=:dash,
			color=palette(pal)[8], alpha=0.5
		)
		plot!(
		    grouped_socv3.aleph, grouped_socv3.mean_s_median, 
			ribbon = (
			grouped_socv3.mean_s_median .- grouped_socv3.mean_s_lerror, 
			grouped_socv3.mean_s_herror .- grouped_socv3.mean_s_median
			),
		    fillalpha=0.2, label = L"0.15", lw=2,
			color=palette(pal)[6],
		)
		plot!(
			grouped_socv3.aleph, s_star.(0.65, grouped_socv3.aleph), 
			fillalpha=0.2, label = "", lw=2, ls=:dash,
			color=palette(pal)[6], alpha=0.5
		)
		plot!(
		    grouped_socv2.aleph, grouped_socv2.mean_s_median, 
			ribbon = (
			grouped_socv2.mean_s_median .- grouped_socv2.mean_s_lerror, 
			grouped_socv2.mean_s_herror .- grouped_socv2.mean_s_median
			),
		    fillalpha=0.2, label = L"0.10", lw=2,
			color=palette(pal)[3],
		)
		plot!(
			grouped_socv2.aleph, s_star.(0.6, grouped_socv2.aleph), 
			fillalpha=0.2, label = "", lw=2, ls=:dash,
			color=palette(pal)[3], alpha=0.5
		)
		plot!(
			grouped_socv.aleph, grouped_socv.mean_s_median, 
			ribbon = (
			grouped_socv.mean_s_median .- grouped_socv.mean_s_lerror, 
			grouped_socv.mean_s_herror .- grouped_socv.mean_s_median
			),
			fillalpha=0.3, color=palette(pal)[1], label = L"0.05", lw=2,
		)
		plot!(
			grouped_socv.aleph, s_star.(0.55, grouped_socv.aleph), 
			fillalpha=0.2, label = "", lw=2, ls=:dash,
			color=palette(pal)[1], alpha=0.75
		)
	
		peersens = plot(
			eld_peerinf,
			eld_sens,
			layout=(2,1),
			size=(500, 700),
			dpi=300,
			#margins=2Plots.mm
		)

		if stakeplot
			eldstake = plot(
				eld_inf,
				eld_stake,
				pb_plot,
				layout=(3,1),
				size=(500, 700),
				dpi=300
			)
		else
			eldstake = plot(
				eld_inf,
				pb_plot,
				layout=(2,1),
				size=(500, 700),
				dpi=300
			)
		end

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
  ╠═╡ =#

# ╔═╡ 2fbddc1c-6d2c-4195-9366-0e7a483450fc
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

# ╔═╡ f0f38b73-ea8f-404e-af4b-6539d1d78f54
# ╠═╡ disabled = true
#=╠═╡
begin
	dat6 = CSV.read("../data/analysis_1b.csv", DataFrame)
	dat6 = dat6[dat6.time .== 2500, :]
	
	datau = combine(
		groupby(dat6, [:aleph, :u, :envshift, :T]), 
		:soc_v_median => mean => :soc_v_median,
		:soc_h_median => mean => :soc_h_median,
		:sens_median => mean => :sens_median,
		:freq_pb => mean => :freq_pb
	)
	datau1 = datau[datau.u .== 0.55,:]
	datau2 = datau[datau.u .== 0.65,:]
	datau3 = datau[datau.u .== 0.75,:]
	datau4 = datau[datau.u .== 0.85,:]

	plot(
		plot(
			levelplot(
				datau1,
				:aleph,
				:envshift,
				:soc_v_median,
				colrange=(0.0,1.0),
				colormap = cgrad(:YlOrRd),
				mask=true,
				delta=0.4,
				xticks=false,
				#xticks=(0.1:0.1:0.9|>collect, [L"%$a" for a in 0.1:0.1:0.9|>collect]),
				yticks=(1:2:15|>collect, [L"%$a" for a in 1:2:15|>collect])
				),
			levelplot(
				datau2,
				:aleph,
				:envshift,
				:soc_v_median,
				colrange=(0.0,1.0),
				colormap = cgrad(:YlOrRd),
				mask=true,
				delta=0.4,
				xticks=false,
				yticks=false,
				#xticks=(0.1:0.1:0.9|>collect, [L"%$a" for a in 0.1:0.1:0.9|>collect]),
				#yticks=(1:10|>collect, [L"%$a" for a in 1:10|>collect])
				),
			levelplot(
				datau3,
				:aleph,
				:envshift,
				:soc_v_median,
				colrange=(0.0,1.0),
				colormap = cgrad(:YlOrRd),
				mask=true,
				delta=0.4,
				xticks=false,
				yticks=false,
				#xticks=(0.1:0.1:0.9|>collect, [L"%$a" for a in 0.1:0.1:0.9|>collect]),
				#yticks=(1:10|>collect, [L"%$a" for a in 1:10|>collect]),
				cbarticks=(
					[1.275, 1.275, 1.275],
					[1, 5.5, 10],
					[text(L"0.0", 10), text(L"0.5", 10), text(L"1.0", 10)]
				),
			),
			layout=(1,3)
		),
		plot(
			plot(
				levelplot(
					datau1,
					:aleph,
					:envshift,
					:soc_h_median,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=0.4,
					xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
					yticks=(1:2:15|>collect, [L"%$a" for a in 1:2:15|>collect])
					),
				xlab = " ",
			),
			plot(
				levelplot(
					datau2,
					:aleph,
					:envshift,
					:soc_h_median,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=0.4,
					xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
					yticks=false,
					#yticks=(1:10|>collect, [L"%$a" for a in 1:10|>collect])
					),
				xlab = L"\mathrm{wealth\ buffer\ } (\aleph)",
			),
			plot(
				levelplot(
					datau3,
					:aleph,
					:envshift,
					:soc_h_median,
					colrange=(0.0,1.0),
					colormap = cgrad(:YlGnBu),
					mask=true,
					delta=0.4,
					xticks=(0.0:0.25:1.0|>collect, [L"%$a" for a in 0.0:0.25:1.0|>collect]),
					yticks=false,
					#yticks=(1:10|>collect, [L"%$a" for a in 1:10|>collect]),
					cbarticks=(
						[1.275, 1.275, 1.275],
						[1, 5.5, 10],
						[text(L"0.0", 10), text(L"0.5", 10), text(L"1.0", 10)]
					),
				),
				xlab = " ",
			),
			layout=(1,3)
		),
		layout=(2,1), size=(1000,600), bottom_margin=6Plots.mm
	)

end
  ╠═╡ =#

# ╔═╡ 324d8681-82ae-42c0-8e23-91b7ab4a2cd4
# ╠═╡ disabled = true
#=╠═╡
begin
	envshift_plot = plot(
		plot_elder_portfolio( CSV.read("../data/analysis_1b.csv", DataFrame), envshift=1, stakeplot=false, n=false ),
		size=(500,430), margins=2Plots.mm
	)
	
	savefig(envshift_plot, "../images/fig7_envshift.pdf")
	
	envshift_plot
end
  ╠═╡ =#

# ╔═╡ 7e61b751-7395-43a6-9f91-3d9151d0890e
# ╠═╡ disabled = true
#=╠═╡
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
		#mdatl6_3 = dat1_l6[(dat1_l6.m .== 15), :]

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
		#grouped_vbar9 = combine(
			#groupby(mdatl6_3, :n), 
			#:Vbar => mean => :mean_Vbar,
		#)
		
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
  ╠═╡ =#

# ╔═╡ 55c3f3f2-2e71-4fe3-b7a0-cdd9b0422816
# ╠═╡ disabled = true
#=╠═╡
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
  ╠═╡ =#

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
# ╠═╡ disabled = true
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
# ╠═╡ disabled = true
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
# ╠═╡ disabled = true
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
# ╟─d9ed7f3b-1324-4468-aecd-1b6756e6f415
# ╟─3ab29f6f-c949-43cf-a254-00d505a5313f
# ╟─5e5d671d-0b35-4404-b022-c2fe8bdde7c2
# ╟─fabf9049-9a39-4765-b74b-efced7fc73df
# ╟─223e84bb-404e-4c66-bc3d-242d42c00bc9
# ╟─25a80f68-931e-4e2b-b591-316ab0df0863
# ╟─80f6d4ac-3036-4e7a-9ab7-31d0644ec6ea
# ╟─f81601e9-a528-4359-8306-7111d542807d
# ╟─59959294-880a-470e-b4c2-b2579a1c55fe
# ╟─6ac78a6a-aa27-4ea0-8e49-e469186dfce8
# ╟─88e18457-ae1b-4de6-958b-df4f407d0b98
# ╟─43b13c02-86ac-4620-b62b-a81e98a173a4
# ╟─cda2a3a4-3d1c-497b-9643-31ef1be4a370
# ╟─b5b58967-287f-47b5-99a4-81b008183204
# ╟─ec30c84e-42a0-420e-b717-1fb0f7f5defd
# ╟─38ff7de5-a2dd-4e5e-9f05-494709d317d9
# ╟─ceacb32a-71ca-49c6-ab92-f5f33fc32b31
# ╟─a3dec80f-60df-44b1-adc5-855cf06b492a
# ╟─3a2082c4-ef09-4b53-af1b-411792729560
# ╟─c56c822a-ded8-4550-978c-2a037389a85a
# ╟─a0d7d4e6-e715-4aad-96d8-a1bc4ad7af88
# ╟─077c7aec-998a-4295-9426-85d254d251f7
# ╟─9e373cb0-dc3c-4834-ac88-d12d13fd1fa0
# ╟─656b3a41-1ceb-4637-af43-81b3e4d25832
# ╟─f82a9c5f-b209-41f6-8360-7795b407f077
# ╟─721c05e8-3205-4d6e-af11-d7648e9f95fc
# ╟─c67685eb-1018-4b18-86db-a8e5c7828e22
# ╟─0133b3f6-b7e0-400b-a96f-6dfef71590c9
# ╟─f4979b22-a419-4473-809f-620ad3125057
# ╟─d5a60bb0-4e82-48d4-84da-0c762e5da660
# ╟─3dc40146-3fdb-4d00-a288-571c47a45041
# ╟─ece62961-7745-49ac-9f89-3a37544c1786
# ╟─9e0cf19c-d66a-448b-b3fc-8ffbb99ce22a
# ╟─8a2c7c0c-e59d-4727-ab06-c0367298fb8b
# ╟─aef1aaeb-5cd1-4121-81ab-490a24a89af0
# ╟─2fbddc1c-6d2c-4195-9366-0e7a483450fc
# ╟─f0f38b73-ea8f-404e-af4b-6539d1d78f54
# ╟─324d8681-82ae-42c0-8e23-91b7ab4a2cd4
# ╟─7e61b751-7395-43a6-9f91-3d9151d0890e
# ╟─55c3f3f2-2e71-4fe3-b7a0-cdd9b0422816
# ╟─f4ee25e4-626e-4d72-b4b5-dd1f9f389f8c
# ╟─b9d9f726-e94e-4a95-815b-06a77e3c605b
# ╟─ebf1dade-ce5e-42dd-a687-1bb8791eb7ab
# ╟─c8795305-655c-47bd-a385-e711f0ecc41d
# ╟─dd70f9c5-1d26-4a0c-bcfa-106300110cbb
# ╟─896ff8a7-0f61-4be3-adf4-d8e55ad16874
