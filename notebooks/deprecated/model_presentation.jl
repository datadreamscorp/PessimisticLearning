### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ aa91c8b2-f97d-11ed-285d-139b5d267c1d
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	#import VarianceAverseSocialLearning as VA
	using Distributions, StatsBase
	#using StatsPlots, Plots
	using Plots
	using PlutoUI, PlutoReport
	using Base.Threads
	include("../src/pessimistic_learning_Numeric.jl")
end

# ╔═╡ 59aed410-2c3d-4d2d-9c94-a64811ee25ee
@htl("""
<script id="first">

	var editor = document.querySelector("pluto-editor")
	var prev = document.querySelector("button.changeslide.prev")
	var next = document.querySelector("button.changeslide.next")
	
	const click_background = (e => {
		// debugger;
		if (editor != e.target) return;
		e.preventDefault();		
		console.log(e.button);
		if (e.button === 2 && prev) {
			prev.click();
		} else if (e.button === 0 && next) {
			next.click();
		} 
	})
	editor.addEventListener("click", click_background)
	editor.addEventListener("contextmenu", click_background)

	invalidation.then(() => { 
		editor.removeEventListener("click", click_background);
		editor.removeEventListener("contextmenu", click_background);
	})
	
	return true;
</script>
""")

# ╔═╡ 68cfee78-f9fd-403e-a85b-7afc5d83dac3
html"<style>
.markdown{
	font-size: 20px;
}
main {
	max-width: 80%;
}
pluto-output>div>img {
	margin: auto;
	display: block;
}
</style>"

# ╔═╡ 2f7d50cf-9246-4296-971f-369ada4bc47a
theme(:dark)

# ╔═╡ 71b3040d-d690-42b3-a764-bef5bd1857ae
@bind pcon presentation_controls(aside=false)

# ╔═╡ e78a57d1-c488-49d9-a0d9-8d596a87c174
@htl "$(apply_css_fixes()) $(presentation_ui(pcon))"

# ╔═╡ 9445caec-109b-446b-a480-b4d300c89f27
Title("Pessimistic social transmission and environmental instability", "An exploration of the effect of estimability of risks on uncertainty preferences and social learning strategies", "Alejandro Pérez Velilla, Supervised by Paul Smaldino", "Dept of Cognitive and Information Sciences - University of California, Merced") # TODO Add image

# ╔═╡ 774e2fe0-b40d-489e-bf20-bffea291ac3a
md"""
# Motivation

#### How do we incorporate risk and uncertainty attitudes into the ways in which we use social information?

#### How does this relate to social learning strategies as we study them in cultural evolution?

#### What do these things tell us about the evolution of risk and uncertainty attitudes as cultural norms?
"""

# ╔═╡ 2c086922-d6ab-44b8-9f2c-8366fcc1cb58
md"
# 0. Preliminaries: sampling, estimation and proportional gambling
"

# ╔═╡ 5a364700-a419-4026-9b51-05ea4168e467
TableOfContents()

# ╔═╡ 17528e3e-1053-4e58-a607-c7fb1bd66494
md"""
# What are we talking about when we talk about risk?

"(Exposure to) the possibility of loss, injury, or other adverse or welcome circumstance; a chance or situation involving such a possibility."

\- Oxford English Dictionary (Wikipedia)
"""

# ╔═╡ 3962efd9-a8d4-4a88-b424-40c4e132f3e2
begin
	histogram(
		rand(Normal(0,1), 10000),
		legend=false,
		lw=2
	)
	histogram!( rand(Normal(0.5,1.5), 10000), lw=2, size=(1000,600), dpi=200 )
	vline!( [-2.5] )
end

# ╔═╡ ae5e2863-36f1-4705-a7d6-161f05e91f1e
md"""
Formally speaking, risk involves an interplay between **means**, **variances** and **thresholds**.
"""

# ╔═╡ 95a4d640-dd39-49c8-881e-8b2d258ba29e
html"<hr>"

# ╔═╡ d2ab8266-63d1-4587-baa9-c4295caf3456
md"
# Proportional gambling and the Kelly criterion.
"

# ╔═╡ a6e2703a-c78f-41ba-aba8-a0a52185dd68
LocalResource("../images/proportional_betting.png")

# ╔═╡ 393875f0-fe27-4b2a-9f04-6e4063d3659b
md"""

#### When $T$ is large, an agent who chooses a *stake* $s_i$ (between zero and one), has an average resource growth rate of

$G_i = u \log (1 + s_i) + (1 - u) \log (1 - s_i)$

#### so that the total (lifetime) payoff is

$w_0 \cdot e^{G_i T}$


"""

# ╔═╡ 2b89c4ad-0c79-4b8c-8278-1d2d32a97f43
md"""
#### The optimal stake in this scenario is given by the **Kelly criterion**:

$s^* = 2 u - 1$

#### published by J.L. Kelly Jr. in 1956, he wanted to re-derive Shannon's Information Rate through a gambling interpretation.
"""

# ╔═╡ 31c29f28-0dc4-48c1-bb80-a75213ec4363
md"""
# Using **fractional** Kelly strategies leads to slower growth, but less volatility.
"""

# ╔═╡ fd4e8a0c-c010-4ddb-8e00-f8e8bab3159a
begin
	T2 = 200
	
	v_opt = Vector{Float64}()
	v_frac = Vector{Float64}()
	w1 = 1.0
	w2 = 1.0
	
	push!(v_opt, w1)
	push!(v_frac, w2)
	
	s_opt = 2*0.65 - 1
	s_frac = 0.9*s_opt
	
	for i in 1:T2
		if rand() < 0.65
			w1 = w1*(1+s_opt)
			push!(v_opt, w1)
		else
			w1 = w1*(1-s_opt)
			push!(v_opt, w1)
		end

		if rand() < 0.65
			w2 = w2*(1+s_frac)
			push!(v_frac, w2)
		else
			w2 = w2*(1-s_frac)
			push!(v_frac, w2)
		end
		
	end

	plot(1:T2+1, v_opt, label="Optimal Kelly", xlab="time", ylab="payoff", dpi=400)
	plot!(1:T2+1, v_frac, label="Fractional Kelly")
end

# ╔═╡ 7e0cbd7f-bae1-4eda-a6e0-7dee7fbe887d
html"<hr>"

# ╔═╡ 0d8755a0-ed65-440a-a101-01e9759b635b
begin
	T = 1000
	mem = 1000
	
md"
# Not all risks admit estimation through repeated sampling.
"
end

# ╔═╡ 75409435-88c1-4b8a-b721-197147397a02
@bind d Slider(1:10:101)

# ╔═╡ 829e5263-f0cd-4223-9d21-7c454be97404
begin
	tvec = Vector{Float64}()
		
	push!(tvec, 0.0)
			
	for i in 1:T
		v = rand( TDist(d) )
		
		if i < mem
			est = ( sum(tvec) + v ) / (i+1)
			push!(tvec, est)
		else
			est = ( sum( tvec[end-mem+1:end] ) + v ) / mem
			push!(tvec, est)
		end
	end
		
	tplot = plot(
		1:(T+1),
		tvec,
		label="average outcome",
		title= "t distribution"
	)
	
	md"""
	degrees of freedom for T distribution = $d
	"""
end

# ╔═╡ 7a1678f9-5892-4da8-9588-87f349888a75
@bind λ3 Slider(1:0.1:5)

# ╔═╡ 9ab2d60a-3031-412b-a359-4222c0b672aa
begin
	pvec = Vector{Float64}()
		
	push!(pvec, 1.0)
			
	for i in 1:T
		v = rand( Pareto(λ3) )
		
		if i < mem
			est = ( sum(pvec) + v ) / (i+1)
			push!(pvec, est)
		else
			est = ( sum( pvec[end-mem+1:end] ) + v ) / mem
			push!(pvec, est)
		end
	end
		
	pplot = plot(
			1:(T+1),
			pvec,
			label="average outcome",
			title="pareto distribution"
			)
	
	md"""
	λ for Pareto distribution = $λ3
	"""
end

# ╔═╡ 382395ba-9fd7-4d7b-943f-8fc101d48bd0
plot(tplot, pplot, size=(1000, 600), dpi=200)

# ╔═╡ ec192076-29de-4cc2-8ae6-9975ff18cd64
html"<hr>"

# ╔═╡ 28b6dea6-0a8a-406d-8966-df40bc6af05c
md"
## Proposed case: use Pareto distribution to model stochasticity of success rates.
"

# ╔═╡ 9c0a8741-d155-494c-847b-c080df4597f9
md"""
$u = \frac{1}{ρ}; \quad ρ \sim \text{Pareto}(λ)$
"""

# ╔═╡ 5e552a85-5786-4cb1-aaec-fc9ff8c062ec
LocalResource("../images/lambda.png", (:height => 440))

# ╔═╡ 50c40b95-8a3b-4f96-9970-231166ddf72c
@bind λ2 Slider(1:0.5:5)

# ╔═╡ d400383b-98c9-4010-8744-57a3d642954a
md"""
λ for Pareto distribution = $λ2
"""

# ╔═╡ e2a1b025-2f1c-474b-8cfe-6acb040535fb
begin
	U = [1/rand( Pareto(λ2) ) for i in 1:1000000]
	succrate_hist = histogram(
		U,
		legend=false,
		title="estimated success rates",
		normalize=true
	)
	plot!(
		0:0.001:1,
		λ2.*(0:0.001:1).^(λ2 - 1),
		lw=2,
		dpi=300,
		normalize=true
	)
	vline!([mean(U)], lw=2)
end

# ╔═╡ 65508f52-9207-4883-8f18-27d0710b00b7
html"<hr>"

# ╔═╡ e9735c5b-99a5-408f-aedd-bca2e72d5f36
md"# 1. Estimating stakes
"

# ╔═╡ a037b129-ca5d-4aac-be7f-ea6b22c17f55
LocalResource("../images/stake1.png", (:height => 440))

# ╔═╡ cae6945a-a713-49cb-b55c-0841e872d2a0
LocalResource("../images/stake2.png", (:height => 440))

# ╔═╡ a4a6ae38-f6d5-4e0e-8e81-7a4f18468315
LocalResource("../images/unknown_stake.png", (:height => 440))

# ╔═╡ fc231fa5-5d37-4c95-93de-d08356c27639
LocalResource("../images/estimating_stake1.png", (:height => 440))

# ╔═╡ 36661fea-6a69-4245-b8f4-b5fc5092ad75
LocalResource("../images/social_estimating_stake1.png", (:height => 440))

# ╔═╡ 211a41d6-cc88-4ec2-8611-ed7c0ba8c9ea
html"<hr>"

# ╔═╡ f82bccb0-44c8-4aab-83dc-b35fe63bbe40
md"""# Let's simulate. """

# ╔═╡ 27a0ac4c-3454-44d0-969a-13b7a7ef3833
@bind n Select(10:10:10000)

# ╔═╡ de408298-142e-4b66-ac8e-870e986d659c
md"""
number of agents = $n
"""

# ╔═╡ 57d8e1a7-2841-4dc5-9968-a7f0144243f8
@bind tries Select([1, 5, 10, 100, 500, 1000])

# ╔═╡ 19d58139-df9f-430c-87ee-6979042ec554
md"""
individual samples = $tries
"""

# ╔═╡ 0c0412bb-5d61-4692-acd4-43bad20ca75f
md"# 2. Probability-weighting and social transmission
"

# ╔═╡ 0a07114c-48ee-4367-9fb8-5cee83509af2
md"""
We can use **probability-weighting** as employed in Rank-Dependent Expected Utility Theory.
"""

# ╔═╡ e0771bd9-deb3-4177-b6c2-108f1d07d98d
LocalResource("../images/pessimistic_averaging.png", (:height => 640))

# ╔═╡ 509259f8-77e2-4394-ab8a-0de40cd118af
md""" # Let's visualize it. """

# ╔═╡ 49db51ab-492b-429e-9dbe-a0b530858c0a
@bind λ Select(1:0.05:10)

# ╔═╡ c42a1b72-ab64-4567-b9f6-7090dbf698fc
md"""
λ (environmental stability) = $λ
"""

# ╔═╡ 27dd10cc-882d-4717-9a55-de64e02ea9d1
begin

	u = 1.0
	
	#sim_rates = VA.simulate_success_rates(λ=λ, n=n, u=u)
	estimates = calculate_estimates(n, tries, λ)
	estimated_stakes = calculate_stake.(estimates)
	
	histogram(
		estimated_stakes,
		legend=:top,
		title="distribution of estimated stakes",
		label="",
		color="light blue",
		bins=50,
		size=(700,500),
		dpi=300
	)
	vline!([mean(estimated_stakes)], labels="population mean", lw=2)
	vline!([median(estimated_stakes)], labels="population median", lw=2)

	#expV = 2*(λ - 1)/λ - 1
	#vline!([expV > 0.0 ? expV : 0.0], labels="expected value", lw=2)
end

# ╔═╡ b935dfd4-2e4e-448d-b05d-687c7cc3c01a
histogram(estimates, legend=false, dpi=300)

# ╔═╡ 75c13362-d361-4f3e-aa25-a6f9046a2e87
begin
	prop0 = ( size( filter(x -> x == 0, estimated_stakes) ) ./ size(estimated_stakes) )[1]

	p0 = round(prop0[1], digits=3)

md"
the proportion of zero-valued estimated stakes is $p0
"
end

# ╔═╡ 65e9f00c-1416-4473-a0a4-d576a13097ef
md"""
λ (environmental stability) = $λ
"""

# ╔═╡ 0db63f9d-03c5-4781-8083-bbfbbd1db5ed
begin
	deltas = 1:0.05:10
	pweighted_stakes = [rdeu_power(estimated_stakes, d) for d in deltas]

	expected_stake = ( 2*( λ*( 1 - ( 1/2^(λ+1) ) ) / ( (λ+1)*( 1 - (1/2^λ) ) ) ) - 1 ) * ( 1 - (1/2^λ) )

	optimal_fraction = 1 / ( 1 + ( 2^(-λ) / (λ - 1) ) )

	optimal_delta = deltas[ findmin( ( (optimal_fraction * expected_stake) .- pweighted_stakes ).^2 )[2] ]
	
	plot(
		deltas,
		pweighted_stakes,
		label="",
		xlab="degree of pessimistic probability weighting (δ)",
		ylab="probability-weighted estimated stake",
		ylim=(0.0, 1.0),
		size=(800,600),
		lw=2
	)
	hline!([optimal_fraction * expected_stake], label = "optimal stake")
	vline!([optimal_delta], label="optimal degree of pessimism")
	
end

# ╔═╡ 53948d96-e299-4636-b1d4-67b0a1cf0417
md"
# 3. Iterated gambling and survival
"

# ╔═╡ c42d282e-01e6-4635-8b93-aafb9269472b
md"""
λ (environmental stability) = $λ
"""

# ╔═╡ 1fc4c47e-53d1-4e5c-b0d4-9273c752c34f
begin

	seasons = 2000
	rounds = 1
	N = n

	
	sims = [ 
	[simulate_gambles(u, λ, s, seasons=seasons, rounds=rounds, init_capital=0, abarrier=false) for i in 1:N] 
	for s in pweighted_stakes 
	]

	payoffs = [
		exp.(sim ./ seasons)
		for sim in sims
	]

	survival_payoffs = [
		filter(x -> x > 0, payoffs[i]) for i in 1:length(deltas)
	]

	average_payoffs = [
		mean(payoff)
		for payoff in payoffs
	]
	median_payoffs = [
		median(payoff)
		for payoff in payoffs
	]
	
	max_delta_median = deltas[findmax(median_payoffs)[2]]
	opt_stake_med = rdeu_power(estimated_stakes, max_delta_median)
	max_delta_mean = deltas[findmax(average_payoffs)[2]]
	opt_stake_mean = rdeu_power(estimated_stakes, max_delta_mean)
	optimal_average_payoff = exp( (λ/(λ+1))*log(1+opt_stake_mean) + ( 1 - (λ/(λ+1)) )*log(1-opt_stake_mean) )

	median_stake = median(estimated_stakes)
	median_stake_payoff = [
		simulate_gambles(u, λ, median_stake, seasons=seasons, rounds=rounds)
		for i in 1:N
	]

	survivals = [
		filter(x -> x > 0, sims[i]) |> length for i in 1:length(deltas)
	]
	
end

# ╔═╡ f9f324bf-f2d4-46ce-9875-0d712655c911
begin
	
	#payoff_hist = histogram( payoffs[1], alpha=0.5, label="δ = $(deltas[1])", xlab="average fitness" )
	#histogram!( payoffs[21], alpha=0.5, label="δ = $(deltas[21])" )
	#histogram!( payoffs[61], alpha=0.5, label="δ =  $(deltas[61])" )
	#histogram!( payoffs[141], alpha=0.5, label="δ =  $(deltas[141])" )
	#histogram!( exp.(sims[7]./seasons), alpha=0.5, label="δ =  $(deltas[7])" )
	#vline!([1.0], label="survival line", lw=2)
	#annotate!(0.975, 1000, text("λ = $λ", 12))

	payoff_hist = histogram( survival_payoffs[1], alpha=0.5, label="δ = $(deltas[1])", xlab="average fitness" )
	histogram!( survival_payoffs[21], alpha=0.5, label="δ = $(deltas[21])" )
	histogram!( survival_payoffs[61], alpha=0.5, label="δ =  $(deltas[61])" )
	histogram!( survival_payoffs[141], alpha=0.5, label="δ =  $(deltas[141])" )
	#histogram!( exp.(sims[7]./seasons), alpha=0.5, label="δ =  $(deltas[7])" )
	vline!([1.0], label="survival line", lw=2)
	annotate!(0.975, 1000, text("λ = $λ", 12))
	
	surv_plot = plot(
		deltas,
		survivals ./ N,
		legend=false,
		#xlab= "degree of pessimistic weighting (δ)",
		ylab="probability of survival",
		lw=2,
		ylim=(0.0, 1.0),
		color="pink"
	)
	hline!([0.5], color="dark green")
	annotate!( [(6, 0.45, text("danger zone", 10))] )

	payoff_plot = plot(
		deltas,
		average_payoffs,
		xlab="degree of pessimistic weighting (δ)",
		ylab="average fitness",
		label="mean",
		legend=:topright,
		lw=2,
		ylim=(1.0, optimal_average_payoff+0.005)
	)
	#plot!(
		#deltas,
		#median_payoffs,
		#label="median"
	#)
	#hline!([optimal_average_payoff], label="optimal fitness")
	vline!([optimal_delta], label="optimal degree of pessimism")
	#hline!([ exp.( mean(median_stake_payoff ./ seasons) ) ], label="fitness of median stake")
	
	l = @layout [
    a{0.5w} [grid(2,1)]
	]
	
	tri_plot = plot(
		payoff_hist, 
		surv_plot, 
		payoff_plot, 
		layout=l, 
		size=(900, 550),
		dpi=400
	)

end

# ╔═╡ 1cf593eb-821c-45b0-9f1a-c3dc36947b09
#savefig(tri_plot, "uncertain_env_light.pdf")

# ╔═╡ bdbc3867-248f-495d-8213-4c041a26f2ea
begin
	mean_stake = mean(estimated_stakes)
	
	md"""
	Mean stake: $mean_stake

	Expected stake: $expected_stake
	
	Median stake: $median_stake

	Optimal stake (mean): $opt_stake_mean

	Optimal stake (median): $opt_stake_med
	"""
end

# ╔═╡ 3e95992e-81b3-43d2-a782-b7024ff0e21c
optimal_expected_stake = 2*(λ/(λ+1)) - 1

# ╔═╡ c9c3a71f-6361-4b62-835b-f259e08a111e
optimal_fractional = optimal_expected_stake / expected_stake

# ╔═╡ 2f7e9ee1-a2b8-4057-ab41-ab2e0d129af1
md"
# 4. Social learning and the evolution of uncertainty-sensitive cultures
"

# ╔═╡ 1969e8e0-46b5-460b-8eea-4f7bf71c054d


# ╔═╡ 0a9f74aa-364f-455e-bda5-e0f6c1dec3c7
md"""
λ (environmental stability) = $λ
"""

# ╔═╡ ea1b9945-5964-4fcc-b561-027fcb9a3495
begin
	payoff_bias = 2000
	
	surv_payoffs = [
		filter(x -> x ≥ 1.0, payoff)
		for payoff in payoffs
	]
	surv_payoffs = [
		length(payoff) > 0 ? payoff : [0]
		for payoff in surv_payoffs
	]
	
	surv_payoffs_biased = [
		sp.^payoff_bias
		#payoff_bias.^sp
		for sp in surv_payoffs
	]
		
	surv_median_stake_payoff = filter(x -> x ≥ 1.0, median_stake_payoff)

	payoff_and_conf = mean.([
		( ( (length(sp) / n).^1 ) .* (sp) ).^payoff_bias
		for sp in surv_payoffs
	])
	
	payoff_biased = plot(
		deltas,
		[
			( mean(sp) / sum(mean.(surv_payoffs_biased)) )
			for sp in surv_payoffs_biased
		],
		label="",
		xlab="degree of pessimistic weighting (δ)",
		ylab="probability of acquisition",
		#ylab="aggregated payoff",
		legend=false,
		size=(300, 300),
		dpi=100,
		lw=2,
		xlim=(1,10)
	)
	vline!([optimal_delta], label="optimal degree of pessimism")
	annotate!(4, 0.075, "only payoff bias")

	conformity_plot = plot(
		deltas,
		[
			l / sum(payoff_and_conf)
			for l in payoff_and_conf
		],
		label="",
		xlab="degree of pessimistic weighting (δ)",
		dpi=100,
		lw=2,
		xlim=(1,10)
	)
	vline!([optimal_delta], label="optimal degree of pessimism")
	annotate!(4, 0.075, "with conformity")

	"""
	surv_payoff_conformity = plot(
		deltas,
		[
			( γ1 .* mean(sp) / sum(mean.(surv_payoffs)) ) .+ (1 - γ1)*( length(sp) / sum(length.(surv_payoffs)) )
			for sp in surv_payoffs
		],
		xlab="degree of pessimistic weighting (δ)",
		label="",
		legend=:topright,
		lw=2
	)
	vline!([optimal_delta], label="optimal degree of pessimism")
	
	plot(surv_payoff_plot, surv_payoff_conformity, size=(800,500), link=:all)
	"""

	conf_payoff_plot = plot(
		payoff_biased, 
		conformity_plot, 
		size=(800, 600), 
		dpi=300, 
		link=:all
	)
end

# ╔═╡ ffc8bf2f-f2d4-4d89-97d3-ed8b436d8602
#savefig(conf_payoff_plot, "conf_payoff_light.pdf")

# ╔═╡ 486cc5d7-e45d-4db5-91b4-063d093b679d
begin
	max_pdelta_median = deltas[ findmax(median.(surv_payoffs))[2] ]
	opt_pstake_med = rdeu_power(estimated_stakes, max_pdelta_median)
	max_pdelta_mean = deltas[ findmax(mean.(surv_payoffs))[2] ]
	opt_pstake_mean = rdeu_power(estimated_stakes, max_pdelta_mean)

	md"""
	Optimal perceived stake (mean): $opt_pstake_mean

	Optimal perceived stake (median): $opt_pstake_med
	"""
end

# ╔═╡ bbef9e41-3e84-491c-a474-617066ace59f
md""" # Having the right social learning strategy might be crucial to obtaining the right attitudes towards uncertainty. """

# ╔═╡ 5c1c7097-4921-4311-af2a-acfdd46b5712
LocalResource("../images/end.png", (:height => 640))

# ╔═╡ 78eb7149-f922-47fd-9671-10308a1f3fe6
md"""
# Frequency bias might be necessary for the cultural evolution of pessimistic weighting.
"""

# ╔═╡ 9a9d27c1-f75c-4eee-a228-7cf603614e8c
md"""
#### But we'll see. Evolutionary simulations are in order.

#### Some things I find interesting to ask:

- Can high frequency bias be adaptive under both environmental stability and instability?

- Are there situations in which high payoff bias can be more adaptive than high frequency bias?

- If so, how long does a population take to go from one regime to another?
"""

# ╔═╡ abb40e3b-1128-447b-a040-66306471f50c
md"
# Thanks!
"

# ╔═╡ d156f638-8e26-475b-8029-328b154cb227
md"
#### Critiques about the current work as well as suggestions for the next stage are **very** welcome.
"

# ╔═╡ Cell order:
# ╠═aa91c8b2-f97d-11ed-285d-139b5d267c1d
# ╟─e78a57d1-c488-49d9-a0d9-8d596a87c174
# ╟─59aed410-2c3d-4d2d-9c94-a64811ee25ee
# ╟─68cfee78-f9fd-403e-a85b-7afc5d83dac3
# ╟─2f7d50cf-9246-4296-971f-369ada4bc47a
# ╟─71b3040d-d690-42b3-a764-bef5bd1857ae
# ╟─9445caec-109b-446b-a480-b4d300c89f27
# ╟─774e2fe0-b40d-489e-bf20-bffea291ac3a
# ╟─2c086922-d6ab-44b8-9f2c-8366fcc1cb58
# ╟─5a364700-a419-4026-9b51-05ea4168e467
# ╟─17528e3e-1053-4e58-a607-c7fb1bd66494
# ╟─3962efd9-a8d4-4a88-b424-40c4e132f3e2
# ╟─ae5e2863-36f1-4705-a7d6-161f05e91f1e
# ╟─95a4d640-dd39-49c8-881e-8b2d258ba29e
# ╟─d2ab8266-63d1-4587-baa9-c4295caf3456
# ╟─a6e2703a-c78f-41ba-aba8-a0a52185dd68
# ╟─393875f0-fe27-4b2a-9f04-6e4063d3659b
# ╟─2b89c4ad-0c79-4b8c-8278-1d2d32a97f43
# ╟─31c29f28-0dc4-48c1-bb80-a75213ec4363
# ╟─fd4e8a0c-c010-4ddb-8e00-f8e8bab3159a
# ╟─7e0cbd7f-bae1-4eda-a6e0-7dee7fbe887d
# ╟─0d8755a0-ed65-440a-a101-01e9759b635b
# ╟─829e5263-f0cd-4223-9d21-7c454be97404
# ╟─75409435-88c1-4b8a-b721-197147397a02
# ╟─9ab2d60a-3031-412b-a359-4222c0b672aa
# ╟─7a1678f9-5892-4da8-9588-87f349888a75
# ╟─382395ba-9fd7-4d7b-943f-8fc101d48bd0
# ╟─ec192076-29de-4cc2-8ae6-9975ff18cd64
# ╟─28b6dea6-0a8a-406d-8966-df40bc6af05c
# ╟─9c0a8741-d155-494c-847b-c080df4597f9
# ╟─5e552a85-5786-4cb1-aaec-fc9ff8c062ec
# ╟─d400383b-98c9-4010-8744-57a3d642954a
# ╠═50c40b95-8a3b-4f96-9970-231166ddf72c
# ╠═e2a1b025-2f1c-474b-8cfe-6acb040535fb
# ╟─65508f52-9207-4883-8f18-27d0710b00b7
# ╟─e9735c5b-99a5-408f-aedd-bca2e72d5f36
# ╟─a037b129-ca5d-4aac-be7f-ea6b22c17f55
# ╟─cae6945a-a713-49cb-b55c-0841e872d2a0
# ╟─a4a6ae38-f6d5-4e0e-8e81-7a4f18468315
# ╟─fc231fa5-5d37-4c95-93de-d08356c27639
# ╟─36661fea-6a69-4245-b8f4-b5fc5092ad75
# ╟─211a41d6-cc88-4ec2-8611-ed7c0ba8c9ea
# ╟─f82bccb0-44c8-4aab-83dc-b35fe63bbe40
# ╟─c42a1b72-ab64-4567-b9f6-7090dbf698fc
# ╟─de408298-142e-4b66-ac8e-870e986d659c
# ╟─27a0ac4c-3454-44d0-969a-13b7a7ef3833
# ╟─19d58139-df9f-430c-87ee-6979042ec554
# ╟─57d8e1a7-2841-4dc5-9968-a7f0144243f8
# ╠═b935dfd4-2e4e-448d-b05d-687c7cc3c01a
# ╟─27dd10cc-882d-4717-9a55-de64e02ea9d1
# ╟─75c13362-d361-4f3e-aa25-a6f9046a2e87
# ╟─0c0412bb-5d61-4692-acd4-43bad20ca75f
# ╟─0a07114c-48ee-4367-9fb8-5cee83509af2
# ╟─e0771bd9-deb3-4177-b6c2-108f1d07d98d
# ╟─509259f8-77e2-4394-ab8a-0de40cd118af
# ╟─65e9f00c-1416-4473-a0a4-d576a13097ef
# ╟─49db51ab-492b-429e-9dbe-a0b530858c0a
# ╠═0db63f9d-03c5-4781-8083-bbfbbd1db5ed
# ╟─53948d96-e299-4636-b1d4-67b0a1cf0417
# ╟─c42d282e-01e6-4635-8b93-aafb9269472b
# ╠═1fc4c47e-53d1-4e5c-b0d4-9273c752c34f
# ╠═f9f324bf-f2d4-46ce-9875-0d712655c911
# ╟─1cf593eb-821c-45b0-9f1a-c3dc36947b09
# ╟─bdbc3867-248f-495d-8213-4c041a26f2ea
# ╟─3e95992e-81b3-43d2-a782-b7024ff0e21c
# ╟─c9c3a71f-6361-4b62-835b-f259e08a111e
# ╟─2f7e9ee1-a2b8-4057-ab41-ab2e0d129af1
# ╟─1969e8e0-46b5-460b-8eea-4f7bf71c054d
# ╟─0a9f74aa-364f-455e-bda5-e0f6c1dec3c7
# ╠═ea1b9945-5964-4fcc-b561-027fcb9a3495
# ╟─ffc8bf2f-f2d4-4d89-97d3-ed8b436d8602
# ╟─486cc5d7-e45d-4db5-91b4-063d093b679d
# ╟─bbef9e41-3e84-491c-a474-617066ace59f
# ╟─5c1c7097-4921-4311-af2a-acfdd46b5712
# ╟─78eb7149-f922-47fd-9671-10308a1f3fe6
# ╟─9a9d27c1-f75c-4eee-a228-7cf603614e8c
# ╟─abb40e3b-1128-447b-a040-66306471f50c
# ╟─d156f638-8e26-475b-8029-328b154cb227
