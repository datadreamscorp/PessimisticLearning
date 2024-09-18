using CSV, DataFrames

function plot_powerdist(λ; n=1000, lw=3, legend=false)
	U = [ 1/rand( Pareto(λ) ) for i in 1:n ]
	
	hist = histogram(
		U,
		legend=legend,
		title=L"λ = "*"$λ",
		xlab= λ > 2.0 ? "rate of success "*L"(u)" : "",
		ylab= λ == 1.0 || λ == 3.0 ? "density" : "",
		#xlabelfontsize=8,
		normalize=true,
		color="white",
		titlefontsize=10,
		dpi=300
	)
	plot!(
		0:0.001:1,
		λ.*(0:0.001:1).^(λ - 1),
		lw=3,
		dpi=300,
		normalize=true,
		color="black"
	)
	vline!([mean(U)], lw=1, ls=:dash, color="black")

	return hist
end

function estimate_plots(λ, tries, n; title=false, label=false)
	
	estimates = calculate_estimates(n, tries, λ)
	estimated_stakes = calculate_stake.(estimates)
	
	estimateplot = histogram(
		estimates, 
		legend=false, 
		dpi=300, 
		title= title ? "success rate estimates (ū)" : "",
		color="white",
		ylab="λ = $λ"
	)
	vline!(
		[mean(estimates)], 
		#labels= label ? "mean" : "", 
		lw=3, 
		color="grey",
		alpha=0.75
	)
	
	stakeplot = histogram(
		estimated_stakes,
		legend=:topright,
		title= title ? "estimated stakes (s̄)" : "",
		label="",
		#color="light blue",
		bins=50,
		size=(700,500),
		dpi=300,
		color="white"
	)
	vline!(
		[mean(estimated_stakes)], 
		label= label ? "mean" : "",
		lw=3, 
		color="black",
		ls=:dash,
		alpha=0.75
	)
	vline!(
		[2*mean(estimates)-1],
		label= label ? "optimal stake" : "",
		lw=3,
		color="grey",
		alpha=0.75
	)
	
	return [estimateplot, stakeplot]
	
end

function simplots(λ, s, abarrier, n, aleph, legend, xlab, legend_alt)

	barrier_payoffs = [
	sim_payoffs(l, s, aleph, abarrier=abarrier, n=n)[2]
	for l in λ
	]
	
	non_barrier_opt = [
		(
			findall(x->x==round( optimal_stake(l), digits=2 ), 0:0.01:1)[1],
			round( optimal_stake(l), digits=2 )
			) 
		for l in λ
	]

	sim_plot = plot( 
		s, 
		barrier_payoffs[1],
		#label="$(λ[1])",
        label="",
		#legendtitle=L"λ",
		#legendtitle=L"\hat{s}",
        legendtitle=L"s^*",
        legendfontsize=7,
		legendtitlefontsize=12,
		lw=2,
		alpha=0.5,
		ylab=L"ℵ = "*"$aleph",
		xlab=xlab ? "stake "*L"(s)" : "",
		ylim=(0.9, 1.15),
		legend=legend,
		color=palette(:Dark2_5)[1],
		title=aleph == 0.65 ? L"G (s)"*" (w/ absorbing barrier)" : "",
		titlefontsize=10,
	)

	b_counter = 2
	for p in barrier_payoffs[2:length(barrier_payoffs)]

		plot!( 
			s, 
			barrier_payoffs[b_counter],
			#label="$(λ[b_counter])",
            label="",
			lw=2,
			alpha=0.5,
			color=palette(:Dark2_5)[b_counter]
		)
		scatter!(
			(s[findmax(barrier_payoffs[b_counter])[2]],
			findmax(barrier_payoffs[b_counter])[1]),
			label="",
			color="black",
			marker=:xcross
		)
		scatter!(
			[non_barrier_opt[b_counter][2]],
			[barrier_payoffs[b_counter][non_barrier_opt[b_counter][1]]],
			label="",
			color="black",
			marker=:circle
		)
		b_counter += 1
		
	end

	scatter!(
			(s[findmax(barrier_payoffs[1])[2]],
			findmax(barrier_payoffs[1])[1]),
			label = legend_alt ? "absorbing barrier" : "",
			color="black",
			marker=:xcross
	)
	scatter!(
			[non_barrier_opt[1][2]],
			[barrier_payoffs[1][non_barrier_opt[1][1]]],
			label = legend_alt ? "no barrier" : "",
			color="black",
			marker=:circle
		)
		hline!([1.0], lw=1, ls=:dash, color="black", label="")
	
	return sim_plot
	
end


function plot_resilience(l, e)
	data = CSV.read("../data/analysis_3.csv", DataFrame)

	dat = data[data.seed .== 1 .&& data.time .== 20 .&& data.aleph .== 0.65 .&& data.n .== 25 .&& data.lambda_shift .== l .&& data.envshift .== e, :]

	pal = cgrad(:matter, 4, categorical = true)[2:end]
	
	geometric_plot = plot(
		dat.init_soc_v,
		dat.gVbar,
		color=pal[1],
		lw=2,
		legendtitle=L"ℵ",
		label="0.25",
		title=L"\lambda_l"*" = $(dat.lambda_shift[1])"
	)

	colocount = 2
	for aleph in [0.8, 0.95]

		dat = data[data.seed .== 1 .&& data.time .== 20 .&& data.aleph .== aleph .&& data.n .== 25 .&& data.lambda_shift .== l .&& data.envshift .== e, :]
		
		plot!(
			dat.init_soc_v,
			dat.gVbar,
			lw=2,
			label="$aleph",
			color=pal[colocount]
		)
		
		colocount += 1
	end

	hline!([1.0], lw=1, ls=:dash, color="black", label="")

	plot( 
		geometric_plot, 
		xlabel="reliance on conservative learning "*L"(α)", 
		ylabel=L"\bar{V}_G \quad | \quad \kappa = "*"$e" 
	)
end


function development_plot(l; wb=[0.65, 0.8, 0.95], legend=:topright, xlabel="stake "*L"(s)", t=false)
	model0a = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[1], steps=2)
	model0b = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[1], steps=3)
	model0am = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[2], steps=2)
	model0bm = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[2], steps=3)
	model0ah = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[3], steps=2)
	model0bh = initialize_pessimistic_learning(λ=l, N=10000, n=25, t=2, T=1000, aleph=wb[3], steps=3)
		
	lowhist = histogram(
		[a.s for a in allagents(model0a)], 
		label="after juvenile stage", title=t ? L"ℵ = "*"$(wb[1])" : "", 
		xlabel=xlabel, ylabel=L"λ = "*"$l", 
		legend=false, ylim=(0, 1500))
	histogram!(
		[a.s for a in allagents(model0b)], 
		alpha=0.5, 
		label="end of lifetime"
	)
	
	medhist = histogram(
		[a.s for a in allagents(model0am)], 
		legend=legend, legendfontsize=3, 
		label="after juvenile stage",
		title=t ? L"ℵ = "*"$(wb[2])" : "", xlabel=xlabel, 
		ylim=(0, 1500)
	)
	histogram!(
		[a.s for a in allagents(model0bm)], 
		alpha=0.5, label="end of lifetime"
	)
	
	hihist = histogram(
		[a.s for a in allagents(model0ah)], 
		legend=false, title=t ? L"ℵ = "*"$(wb[3])" : "", 
		xlabel=xlabel, ylim=(0, 1500)
	)
	histogram!(
		[a.s for a in allagents(model0bh)], 
		alpha=0.5
	)
	
	plot(
		lowhist, medhist, hihist, 
		layout=(1,3), link=:all, size=(700, 300), 
		legendfontsize=6, margins=3Plots.mm, dpi=300
	)
end


function plot_conservative_attitudes(l; strat="UB")
	data = CSV.read("../data/analysis_2.csv", DataFrame)

	dat = data[data.seed .== 1 .&& data.aleph .== 0.65 .&& data.n .== 25 .&& data.t .== 25 .&& data.T .== 1000 .&& data.λ .== l .&& data.time .== 10 .&& data.strategies .== strat,:]

	pal = cgrad(:matter, 4, categorical = true)[2:end]
	
	conf_plot = plot(
		dat.init_soc_v,
		dat.s_median,
		ribbon=(dat.s_lerror, dat.s_herror),
		color=pal[1],
		lw=2,
		label="0.65",
		legendtitle=L"ℵ",
		legendtitlefontsize=8,
		legendfontsize=6,
		title=L"λ = "*"$l",
		xlabel="reliance on conservative learning "*L"(α)",
		xlabelfontsize=7
	)
	
	colocount = 2
	for aleph in [0.80, 0.95]
		dat = data[data.seed .== 1 .&& data.aleph .== aleph .&& data.n .== 25 .&& data.t .== 25 .&& data.T .== 1000 .&& data.λ .== l .&& data.time .== 10 .&& data.strategies .== strat ,:]

		plot!(
			dat.init_soc_v,
			dat.s_median,
			ribbon=(dat.s_lerror, dat.s_herror),
			color=pal[colocount],
			lw=2,
			label="$aleph",
			legendtitle=L"ℵ"
		)

		colocount += 1
	end

	return conf_plot
	
end


function plot_conservative_payoffs(l; pb=true)
	data2 = CSV.read("../data/analysis_2.csv", DataFrame)

	dat = data2[data2.seed .== 1 .&& data2.aleph .== 0.65 .&& data2.n .== 25 .&& data2.t .== 25 .&& data2.T .== 1000 .&& data2.λ .== l .&& data2.time .== 10 .&& data2.strategies .== "UB",:]

	pal = cgrad(:matter, 4, categorical = true)[2:end]
	
	conf_plot = plot(
		dat.init_soc_v,
		dat.Vbar,
		color=pal[1],
		lw=2,
		label="0.65",
		legendtitle=L"ℵ",
		legendtitlefontsize=8,
		legendfontsize=6,
		title=L"λ = "*"$l",
		xlabel="reliance on conservative learning "*L"(α)",
		xlabelfontsize=7
	)
	if pb
		pbdat = data2[data2.seed .== 1 .&& data2.aleph .== 0.65 .&& data2.n .== 25 .&& data2.t .== 25 .&& data2.T .== 1000 .&& data2.λ .== l .&& data2.time .== 10 .&& data2.strategies .== "PB",:]
		plot!(
			pbdat.init_soc_v,
			pbdat.Vbar,
			color=pal[1],
			lw=2,
			ls=:dash,
			alpha=0.5,
			label=""
		)
	end
	
	colocount = 2
	for aleph in [0.80, 0.95]
		dat = data2[data2.seed .== 1 .&& data2.aleph .== aleph .&& data2.n .== 25 .&& data2.t .== 25 .&& data2.T .== 1000 .&& data2.λ .== l .&& data2.time .== 10 .&& data2.strategies .== "UB",:]

		plot!(
			dat.init_soc_v,
			dat.Vbar,
			color=pal[colocount],
			lw=2,
			label="$aleph",
			legendtitle=L"ℵ"
		)

		if pb
			pbdat = data2[data2.seed .== 1 .&& data2.aleph .== aleph .&& data2.n .== 25 .&& data2.t .== 25 .&& data2.T .== 1000 .&& data2.λ .== l .&& data2.time .== 10 .&& data2.strategies .== "PB",:]
			plot!(
				pbdat.init_soc_v,
				pbdat.Vbar,
				color=pal[colocount],
				lw=2,
				ls=:dash,
				alpha=0.5,
				label=""
			)
		end

		colocount += 1
	end
	hline!([1.0], lw=1, ls=:dash, color="black", label="")

	return conf_plot
	
end


function plot_indv_learning(λ; n=25, aleph=0.65, t=2, seed=1)
	learndat = CSV.read("../data/analysis_0.csv", DataFrame)

	learndats = learndat[
		learndat.init_sens .== 0.0 .&& learndat.seed .== seed .&& learndat.n .== n .&& learndat.aleph .== aleph .&& learndat.λ .== λ[1] .&& learndat.t .== t,
		:]

	indv_plot = plot(
		learndats.init_soc_h,
		learndats.Vbar,
		palette=:Dark2_5,
		legendtitle=L"λ",
		legendtitlefontsize=7,
		lw=2,
		legendfontsize=6,
		label="$(λ[1])",
		ylim=(0.0, 1.3)
	)

	for l in λ[2:end]

		learndats = learndat[
		learndat.init_sens .== 0.0 .&& learndat.seed .== seed .&& learndat.n .== n .&& learndat.aleph .== aleph .&& learndat.λ .== l .&& learndat.t .== t,
		:]
		
		plot!(
			learndats.init_soc_h,
			learndats.Vbar,
			palette=:Dark2_5,
			label="$(l)",
			lw=2
		)
	end
	hline!([1.0], lw=1, ls=:dash, color="black", label="")

	return indv_plot
end


function plot_stake_sensitivity(dat)
	pal = cgrad(:matter, 4, categorical = true)[2:end]
	plot(
		dat[1].init_sens,
		dat[1].s_median,
		ribbon=(dat[1].s_lerror, dat[1].s_herror),
		ylim=(0,1),
		lw=2,
		c=pal[1],
		legend=first(dat[1].aleph)==0.65 ? true : false,
		ylab=first(dat[1].aleph)==0.65 ? "stake (s)" : "",
		xlab="sensitivity (δ)",
		title="ℵ = "*"$(first(dat[1].aleph))",
		legendtitle="λ",
		legendtitlefontsize=8,
		legenfontsize=6,
		label="1.5",
		dpi=300
	)
	
	scatter!(
		[dat[1].init_sens[ findmin( (dat[1].s_median .- mean(dat[1].opt_s)).^2 )[2] ]],
		[dat[1].s_median[ findmin( (dat[1].s_median .- mean(dat[1].opt_s)).^2 )[2] ]],
		lw=2,
		c=pal[1],
		ls=:dash,
		label=""
	)
	
	plot!(
		dat[2].init_sens,
		dat[2].s_median,
		ribbon=(dat[2].s_lerror, dat[2].s_herror),
		ylim=(0,1),
		lw=2,
		c=pal[2],
		label="3.5"
	)
	scatter!(
		[dat[2].init_sens[ findmin( (dat[2].s_median .- mean(dat[2].opt_s)).^2 )[2] ]],
		[dat[2].s_median[ findmin( (dat[2].s_median .- mean(dat[2].opt_s)).^2 )[2] ]],
		lw=2,
		ls=:dash,
		c=pal[2],
		label=""
	)
	
	plot!(
		dat[3].init_sens,
		dat[3].s_median,
		ribbon=(dat[3].s_lerror, dat[3].s_herror),
		ylim=(0,1),
		lw=2,
		c=pal[3],
		label="5.0"
	)
	scatter!(
		[dat[3].init_sens[ findmin( (dat[3].s_median .- mean(dat[3].opt_s)).^2 )[2] ]],
		[dat[3].s_median[ findmin( (dat[3].s_median .- mean(dat[3].opt_s)).^2 )[2] ]],
		lw=2,
		ls=:dash,
		c=pal[3],
		label=""
	)
	
end


function plot_vbar_sensitivity(dat)
	pal = palette(:Dark2_5)
	plot(
		dat[1].init_sens,
		dat[1].Vbar,
		ylim=(0.2, 1.25),
		lw=2,
		c=pal[1],
		legend= (first(dat[1].aleph), first(dat[1].n)) == (0.95, 25) ? :bottomright : false,
		ylab=first(dat[1].aleph)==0.65 ? L"\bar{V} \quad | \quad n = " * "$(first(dat[1].n))" : "",
		ylabelfontsize = 10,
		xlabelfontsize= 10,
		xlab=first(dat[1].n)==25 ? "sensitivity "*L"(δ)" : "",
		title=first(dat[1].n)==5 ? L"ℵ = "*"$(round(first(dat[1].aleph), digits=2))" : "",
		legendtitle=L"λ",
		legendtitlefontsize=7,
		legendfontsize=5,
		label="1.5",
		dpi=300
	)
	
	plot!(
		dat[2].init_sens,
		dat[2].Vbar,
		ylim=(0.2, 1.3),
		lw=2,
		c=pal[2],
		label="3.0"
	)
	
	plot!(
		dat[3].init_sens,
		dat[3].Vbar,
		ylim=(0.2, 1.3),
		lw=2,
		c=pal[3],
		label="6.0"
	)

	hline!([1.0], lw=1, ls=:dash, color="black", label="")
	
end


function plot_sensitivity(n, aleph, T, t; seed=1)
	sensdata = CSV.read("../data/analysis_1.csv", DataFrame)

	dat = [
		sensdata[
			sensdata.n .== n .&& sensdata.aleph .== aleph .&& sensdata.λ .== l .&& sensdata.T .== T .&& sensdata.t .== t .&& sensdata.seed .== seed, 
			:]
		for l in [1.5, 3.0, 6.0]
	]

	plot_vbar_sensitivity(dat)
end


function run_ABM_plot(;N = 10000, lamb = 2.0, aleph = 0.65, t = 2, envshift = 5, lambda_shift = 6.0, n = 25, mu=0.0, T=1000, strat="UB", legend=:topleft, plot_lab="(A)", seed=654123)
	
	model_CB = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lamb, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lambda_shift, mu_soc_v=0.0, T=T
	)
	model_UL = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lamb, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lambda_shift, mu_soc_v=0.0, init_soc_v=0.5, T=T
	)
	model_PEER = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lamb, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lambda_shift, init_soc_v=0.0, T=T
	)
	adataCB, mdataCB = run!(
		model_CB,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)
	adataUL, mdataUL = run!(
		model_UL,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)
	adataPEER, mdataPEER = run!(
		model_PEER,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)

	model_CB2 = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lambda_shift, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lamb, mu_soc_v=0.0, T=T
	)
	model_UL2 = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lambda_shift, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lamb, mu_soc_v=0.0, init_soc_v=0.5, T=T
	)
	model_PEER2 = initialize_pessimistic_learning(;
		N=N, n=n, seed=seed,  init_sens=0.1, 
		λ=lambda_shift, aleph=aleph, t=t, strategies=strat,
		envshift=envshift, lambda_shift=lamb, init_soc_v=0.0, T=T
	)
	adataCB2, mdataCB2 = run!(
		model_CB2,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)
	adataUL2, mdataUL2 = run!(
		model_UL2,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)
	adataPEER2, mdataPEER2 = run!(
		model_PEER2,
		10, mdata=[:opt_s, :opt_payoff, :s_median, :s_lerror, :s_herror, :Vbar]
	)

	
	timeplot1 = plot(
			mdataCB.time,
			mdataCB.Vbar,
			lw=2,
			label="1",
			color=palette(:Dark2_5)[1],
			legend=legend,
			legendtitle=L"\alpha",
			legendtitlefontsize=7,
			legendfontsize=5,
			xticks=0:10,
			#xlabel="time",
			ylabel=L"\bar{V}",
			#title=L"ℵ = "*"$aleph",
			title=plot_lab*" "*L"ℵ = "*"$aleph",
			titlefontsize=17,
			#title_color="dark green",
			titlelocation=:left,
			ylim=(0.7, 1.35)
		)
		plot!(
			mdataUL.time,
			mdataUL.Vbar,
			color=palette(:Dark2_5)[3],
			lw=2,
			label="0.5"
		)
		plot!(
			mdataPEER.time,
			mdataPEER.Vbar,
			lw=2,
			label="0",
			color=palette(:Dark2_5)[2]
		)
	
		hline!([1], c="black", alpha=0.5, ls=:dash, label="")
		vline!([5], c="black", alpha=0.5, label="")
	
		annotate!([5], [0.8], [text("→ less uncertainty", 8, color="dark green")])
	
		timeplot2 = plot(
			mdataUL.time,
			mdataUL.s_median,
			ribbon=(mdataUL.s_lerror, mdataUL.s_herror), lw=2,
			legend=false,
			xticks=0:10,
			xlabel="time (generations)",
			ylabel="stake "*L"(s)",
			color=palette(:Dark2_5)[3],
			ylim=(0,1)
		)
		plot!(
			mdataCB.time,
			mdataCB.s_median,
			ribbon=(mdataCB.s_lerror, mdataCB.s_herror), lw=2,
			color=palette(:Dark2_5)[1],
		)
		plot!(
			mdataPEER.time,
			mdataPEER.s_median,
			ribbon=(mdataPEER.s_lerror, mdataPEER.s_herror), lw=2,
			color=palette(:Dark2_5)[2],
		)
		vline!([5], c="black", alpha=0.5, label="")
		
		annotate!([5], [0.9], [text("→ less uncertainty", 8, color="dark green")])
		
		timeplot3 = plot(
			mdataUL2.time,
			mdataUL2.Vbar,
			color=palette(:Dark2_5)[3],
			lw=2,
			legend=false,
			legendfontsize=5,
			xticks=0:10,
			ylim=(0.7, 1.35),
			#title=text("→ more risky", 10, "dark red"),
			#title_color="dark red",
		)
		plot!(
			mdataCB2.time,
			mdataCB2.Vbar,
			lw=2,
			color=palette(:Dark2_5)[1],
		)
		plot!(
			mdataPEER2.time,
			mdataPEER2.Vbar,
			lw=2,
			color=palette(:Dark2_5)[2]
		)
		
		hline!([1], c="black", alpha=0.5, ls=:dash, label="")
		vline!([5], c="black", alpha=0.5, label="")

		annotate!([5], [0.8], [text("→ more uncertainty", 8, color="dark red")])
		
		timeplot4 = plot(
			mdataUL2.time,
			mdataUL2.s_median,
			ribbon=(mdataUL2.s_lerror, mdataUL2.s_herror), lw=2,
			legend=false,
			xticks=0:10,
			xlabel="time (generations)",
			#ylabel="stake (s)",
			color=palette(:Dark2_5)[3],
			ylim=(0,1)
		)
		plot!(
			mdataCB2.time,
			mdataCB2.s_median,
			ribbon=(mdataCB2.s_lerror, mdataCB2.s_herror), lw=2,
			color=palette(:Dark2_5)[1],
		)
		plot!(
			mdataPEER2.time,
			mdataPEER2.s_median,
			ribbon=(mdataPEER2.s_lerror, mdataPEER2.s_herror), lw=2,
			color=palette(:Dark2_5)[2],
		)
		vline!([5], c="black", alpha=0.5, label="")

		annotate!([5], [0.9], [text("→ more uncertainty", 8, color="dark red")])
		#=
		return plot(
			plot( timeplot1, timeplot3, link=:all ),
			plot( timeplot2, timeplot4, link=:all ), 
			plot_title = L"ℵ = "*"$aleph",
			layout=(2,1)
		)
		=#
		plot(
	    	plot(timeplot1, timeplot3, link=:all),
	    	plot(timeplot2, timeplot4, link=:all), 
	    	#plot_title = L"ℵ = "*"0.65",
	    	layout=(2, 1),
	    	#plot_titlefontsize = 14,
	    	#plot_titleposition = :top,
	    	#margin = (1Plots.mm)  # Adjust these margins as needed
		)

end


function mixed_pop_plot(;n=25, eshift=10, mshift=6.0, seed=1, freq=0.5, pay=:Vbar_g1, ls=:solid, leg=true)
	pal = cgrad(:matter, 4, categorical = true)[2:end]
	dat = CSV.read("../data/analysis_4.csv", DataFrame)
	data = dat[
		dat.seed .== seed .&& dat.parochial .== false .&& dat.mixed_λ2_shift .== mshift .&& dat.mixed_freq .== freq .&& dat.envshift .== eshift .&& dat.n .== n .&& dat.mixed_aleph2 .== 0.65, :]
	agg = []
	for i in 0:20
		push!(agg, data[data.time .== i, pay])
	end
	
	stab_plot = plot(
		0.0:0.05:1.0,
		geomean.( [ [a[i] for a in agg] for i in 1:21 ] ),
		ylab = L"\bar{V}_G \quad | \quad \omega = "*"$eshift",
		ylabelfontsize=8,
		xlab = "reliance on conservative learning "*L"(α)",
		xlabelfontsize=14,
		color=pal[1],
		lw=2,
		ls=ls,
		legend=leg,
		legendtitle=L"ℵ_D",
		legendtitlefontsize=6,
		legendfontsize=6,
		label="0.65",
		ylim=(0.9, 1.3)
	)

	colocount = 2
	for aleph in [0.8, 0.95]
		data = dat[
		dat.seed .== seed .&& dat.parochial .== false .&& dat.mixed_λ2_shift .== mshift .&& dat.mixed_freq .== freq .&& dat.envshift .== eshift .&& dat.n .== n .&& dat.mixed_aleph2 .== aleph, :]
		agg = []
		for i in 0:20
			push!(agg, data[data.time .== i, pay])
		end

		plot!(
			0.0:0.05:1.0,
			geomean.( [ [a[i] for a in agg] for i in 1:21 ] ),
			c=pal[colocount],
			lw=2,
			label="$aleph"
		)

		colocount += 1
	end
	
	colocount = 1
	for aleph in [0.65, 0.8, 0.95]
		data = dat[
		dat.seed .== seed .&& dat.parochial .== false .&& dat.mixed_λ2_shift .== mshift .&& dat.mixed_freq .== freq .&& dat.envshift .== eshift .&& dat.n .== n .&& dat.mixed_aleph2 .== aleph, :]
		agg = []
		for i in 0:20
			push!(agg, data[data.time .== i, :Vbar_g0])
		end

		plot!(
			0.0:0.05:1.0,
			geomean.( [ [a[i] for a in agg] for i in 1:21 ] ),
			c=pal[colocount],
			lw=2,
			ls=:dash,
			label="",
			alpha=0.5
		)
	
		colocount += 1
	end
	hline!([1.0], lw=1, ls=:dash, color="black", label="")
		
	return stab_plot
end

function aleph_plot(l; legend=false, ylab="")
	pal = palette(:Dark2_5)
	aleph_dat = CSV.read("../data/analysis_2_5.csv", DataFrame)
	aleph_dat = aleph_dat[aleph_dat.λ .== l, :]
	aleph_dat = aleph_dat[aleph_dat.time .== 5, :]
	aleph_dat = aleph_dat[aleph_dat.seed .== 1, :]
	
	aleph_dat_1 = aleph_dat[aleph_dat.init_soc_v .== 0.0, :]
	
	aleph_plot = plot(
		aleph_dat_1.aleph,
		aleph_dat_1.Vbar,
		lw=2,
		color=pal[2],
		legend=legend,
		label="0.0",
		legendtitle=L"α",
		xlab="wealth buffer "*L"(ℵ)",
		ylab=ylab,
		title=L"λ = "*"$l"
	)

	colocount = 1
	cols = [pal[3], pal[1]]
	for a in [0.5, 1.0]
		aleph_dat_iter = aleph_dat[aleph_dat.init_soc_v .== a, :]
		plot!(
			aleph_dat_iter.aleph,
			aleph_dat_iter.Vbar,
			lw=2,
			color=cols[colocount],
			label="$a"
		)

		colocount+=1
	end

	hline!([1.0], lw=1, ls=:dash, label="", color="black")

	aleph_plot
end

function aleph_geo_plot(l, e; legend=false, ylab="", xlab="", title="")
	pal = palette(:Dark2_5)
	aleph_dat = CSV.read("../data/analysis_3_5.csv", DataFrame)
	aleph_dat = aleph_dat[aleph_dat.lambda_shift .== l, :]
	aleph_dat = aleph_dat[aleph_dat.envshift .== e, :]
	aleph_dat = aleph_dat[aleph_dat.time .== 10, :]
	aleph_dat = aleph_dat[aleph_dat.seed .== 1, :]
	
	aleph_dat_1 = aleph_dat[aleph_dat.init_soc_v .== 0.0, :]
	
	aleph_plot = plot(
		aleph_dat_1.aleph,
		aleph_dat_1.gVbar,
		lw=2,
		color=pal[2],
		legend=legend ? :bottomright : false,
		legendfontsize=6,
		legendtitlefontsize=8,
		label="0.0",
		legendtitle=L"α",
		xlab=xlab,
		ylab=ylab,
		title=title,
		dpi=300
	)

	colocount = 1
	cols = [pal[3], pal[1]]
	for a in [0.5, 1.0]
		aleph_dat_iter = aleph_dat[aleph_dat.init_soc_v .== a, :]
		plot!(
			aleph_dat_iter.aleph,
			aleph_dat_iter.gVbar,
			lw=2,
			color=cols[colocount],
			label="$a",
			dpi=300
		)

		colocount+=1
	end

	hline!([1.0], lw=1, ls=:dash, label="", color="black")

	aleph_plot
end