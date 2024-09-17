### A Pluto.jl notebook ###
# v0.19.40

using Markdown
using InteractiveUtils

# ╔═╡ 44c0fc58-d33c-11ee-2853-85a28adbbb59
begin
	using Pkg
	Pkg.activate("../..")
	using Random, Distributions, StatsBase, Plots, Base.Threads
	include("../../src/deprecated/pessimistic_learning_Numeric.jl")
end

# ╔═╡ 5e6a7ed3-06b5-4267-8226-8628fdbe32cc
begin
	begin
		λ = 1.0:1.0:5.0
		s = 0:0.01:1 |> collect
		#u = 1.0
		#n = 1000
		tries = [3, 5, 10, 100]
		#seasons = 2000
		init_capital = [1, 2, 3]
	end
	
	md"""
## The Development of Risk Attitudes and their Cultural Transmission.

### Alejandro Pérez Velilla, Bret Beheim, Paul E. Smaldino.
"""
end

# ╔═╡ 7fa16c52-e357-4d85-8460-3c979e09f629
md"""
*Preferences are central to economic theory,
yet little is known about them. Because
of this ignorance, economists have worked
hard to make their assumptions about preferences 
as mild as possible, and economic
theory has grown more abstract and general
over the years. This has cushioned the theory 
against violations of assumptions but
has not increased its power. Economics
would gain much from a theory of preferences. 
The theory of evolutionary genetics
may be useful in this regard. If genes affect
preferences, then an evolutionary model
may succeed in predicting them.*

- Alan R. Rogers (1994)
"""

# ╔═╡ 0c679c2a-5d68-4efe-ba74-86873f96594b
md"""
#### Introduction

Risk attitudes develop throughout life, a result of inherited knowledge from seniors, personal experiences from exploration of local environments and social influence from interaction with peers. In a way, risk attitudes embody the experiences people go through during critical stages of development, aggregating an individual's accumulated information on how risky the environment they find themselves in is. As such, growing up in a safe space that encourages risk-free exploration can imbue people with the notion that the world is less dangerous than it actually might be outside of such a bubble, while growing up in an environment where dangers are evident (or made evident by senior members of the communities we belong to) can lead to the development of behavioral safeguards that might operate even when in a relatively safe environment.

The notion that peoples' attitudes toward risk depend on individual life trajectories as well as social factors is uncontroversial, and since the original development of the theory of choice (CITE VN-M) (and throughout is various iterations and alternatives (CITE PROSPECT)) myriad work in decision-making, both theoretical and empirical, attributes individual differences in behavior to individual differences in preferences, particularly risk preferences. Evolutionary models of decision-making preferences have made progress in showing how they can come to be through selection borne from environmental pressures (CITE ROGERS, INGELA), with risk preferences being acutely affected by the structure of uncertainty in the environment (CITE HOUSTON-MCNAMARA, PROB HAVING DESCENDANTS). The role of social influence and culture on risk attitudes, however, has not in general been explored outside of empirical work, and calls for modeling work to fill this gap are not new, but remain current (CITE WEBER AND HSEE).

In this short essay we explore such a model of how risk attitudes are developed, and how social learning and environmental uncertainty impact this development. We show that the use of social information in order to inform what risk attitudes to adopt is not a straightforward business, that dangerously optimistic biases can creep into social learning processes at different stages, and that these biases are exacerbated as individual risk and environmental uncertainty increase. Thus, as the risks and uncertainties decision-makers face increase, pessimistic risk atittudes emerge as solutions to help correct the potential optimistic biases in (otherwise useful) information obtained through individual learning and social pooling. Moreover, these attitudes will be implemented through different processes depending on the social information being used: when attempting to infer the best attitude to adopt from environmental feedback and social pooling of information, being sensitive to peers' catastrophic losses can help adjust one's own preferences before catastrophe has a chance to hit; in other words, it is a form of adaptive pessimism. On the other hand, when attempting to learn risk attitudes by observing the outcomes of previous generations, adaptive pessimism is manifested through increasingly frequency-biased social learning (also known as conformist transmission) in particular when attempting to find the best-paying strategy in a demographically-filtered sample of potential targets for imitation (CITE DEFFNER-MCELREATH). We finish by discussing the links that emerge from this modeling exercise between the theories of decision-making under uncertainty, reinforcement learning and cultural evolution, as well as the roles that environmental uncertainty, sensitivity to losses and conformity play in the evolution and reproduction of risk-averse culture. 

To maximize clarity, we choose to explore the model in a progressive fashion, starting off with a simple dynamic encoding a risk-reward tradeoff and adding complexity as we go along, showing how factors that introduce risk and uncertainty affect the optimal attitude towards risk and the possible solutions that can be deployed in order to counteract the negative consequences of these factors.

In order to do so, we start out with a simple version of Kelly's proportional betting model (Kelly, 1956). In this model, a player starting with wealth $w_0$ chooses a stake $s \in (0,1)$, representing the fraction of their wealth they are betting away, and then proceeds to throw a biased coin every time period (for a total of $T$ periods), with every period resulting in a success with probability $u$ (also known as the rate of success) and a failure with probability $1-u$. If successful, the player increases their wealth proportionally to $1 + s$, and if not successful, their wealth decreases by $1 - s$. At its score, this is a simple model of risky capital investment: the more one places at risk, the more there is to lose and the more there is to win. The less one stakes, the less one can win, but the smaller potential losses will be.

We can write the payoff of a player at time T in the following manner:

$V_T(s) = w_0 \prod_{t=1}^T e^{g_t(s)}$

with $g_t (s) = \text{log}(1 + s)$ if successful, and $g_t (s) = \text{log}(1 - s)$ otherwise. It can be shown that in the limit of large $T$, there is an optimal stake that can be easily calculated and that ensures the best rate of growth for wealth. Known as the Kelly criterion, it can be written as

$s^* = \left\{
     \begin{array}{@{}l@{\thinspace}l}
       2u - 1  &: u > \frac{1}{2}\\
       0 &: u \leq \frac{1}{2}\\

     \end{array}
   \right.$

This means that a decision-maker facing a scenario of proportional betting can know the optimal stake if they know the rate of success $u$. In what remains of this theoretical exercise, we will examine what we think are the main obstacles that learners have to face when trying to be optimal proportional bettors. Throughout life we constantly face decisions that require us to make tradeoffs between risk and reward. The proportional betting model captures this idea with minimal mathematical structure, allowing us to explore how players fare when different forms of uncertainty come into the mix.

We will see that using individual experience and social information can lead to biased information, requiring pessimistic biases (risk aversion) to deal with potentially dangerous optimism that creeps in. Learning from those who came before us can also lead us to choose behaviors that are too risky when we use pure payoff-biased learning and/or when the information we obtain about others exhibits a success bias (successful agents are more visibile, or in the most extreme case, the only visible learning models).

"""

# ╔═╡ 6a33e703-427f-400c-aa85-eae291126584
md"""
#### Part One: The Development of Risk Attitudes.

As story to go along with the argument, we can put ourselves in the shoes of Niko. Niko recently arrived in Liberty City, motivated by the stories of opportunity his cousin Roman wrote about in his letters. Finding himself in a new environment, Niko does not particularly know how to act or what risks to avoid, and he does not want to solely rely on Cousin Roman, who has a weakness for overblown claims. In order for Niko to get a notion of how much it is worth to put his resources at risk, he must have some notion of the city environment's rate of success, $u$.

Let us say life in Liberty City is benevolent enough that Niko can sample the environment for a while before settling on a stake. He comes in with enough money to last him a couple weeks, and he does not waste his time. He uses this time to get some sense of how much the city's environment favors (or disfavors) those who take risks, by trying out different activities and noting down his successes and his failures. Since he does not want to rely purely on his individual impressions, he figures that at the end of his testing period he can meet up with other recent arrivals and everyone can share their own impressions of the city. Using this mix of individual sampling and social learning, he should be able to get an estimate of $u$, which we call $\bar{u}$, which his cognitive machinery transforms into an estimate of the optimal stake one should use ($\bar{s}$).

In doing so, there are several things he must contend with.

"""

# ╔═╡ cd4f3d9d-e0f3-4b52-ab69-827d77637a56
md"""
##### Individual learning: imperfect information and random success rates lead to potentially dangerous errors.

**Success rates can be random**. This means that every time period Niko samples the city, the chance of success he obtains can be different from the last. Thus, every success or failure he gets might have a different success rate generating it, drawn from a common distribution representing the city's stability. Environments where success rates can vary considerably are said to be more *unstable* or *uncertain* than environments where it varies less. If the city is a crime-ridden dystopia governed by a council of mafiosos, Niko's luck might vary significantly on a day to day basis, and he should take care not to put himself in danger. If, on the other hand, the city is a secure place with several safety nets in place, then his chances of success can stay consistently favorable, and he can allow himself to take higher risks.

In general, instead of knowing their chances of success, individuals must sample their environments and try to inferr them. As environments become increasingly unstable, having an accurate estimate requires longer periods of inidividual sampling, which might be infeasible. For this reason, individuals may need to complement their own inferences about the environment with their peers' own estimates, lest they end up staking too little or too much based on luck alone.
"""

# ╔═╡ 5407a245-750c-4a11-9e4a-49632cb41f88
begin
	Random.seed!(12345)
	plot( 
		plot_powerdist.(
		[1.0, 1.5, 2.0, 3.0, 4.0, 5.0], n=1000)..., 
		layout=(2,3), 
		link=:all,
		plot_title="power-distributed success rates (u)"
	)
end

# ╔═╡ 64abbea3-237f-4524-a7a5-2d1c2aec74ec
md"""
We use a Power distribution to model this situation. This means we take the standard Kelly betting model and adds uncertainty in the rate of succes $u$. Specifically, the *inverse* of the probability of winning a bet follows the Pareto distribution with a minimum value of $1$. We will call this quantity $V$ varying between 1 and infinity, and say

$P(V \leq v) = 1 - (m/v)^\lambda$

where $m = 1$ and $\lambda > 1$.

If $V$ is Pareto-distributed, the random variable $U = 1/V$ follows a Power distribution (which *not* the same thing as a "power law"), and varies between 0 ($V$ is very very large) and $m^{-1}$ ($V$ is close to $m = 1$) (Dallas, 1973):

$P(U \leq u) = (mu)^\lambda = u^\lambda$

with probability density function

$f(u) = \lambda u^{\lambda - 1}$

and an expected value given by

$E(U) = \frac{\lambda}{\lambda + 1}$

This probability density function is plotted above (smooth line) alongside samples from the distribution (histogram) and the expected value (vertical line) for different values of $\lambda$, in order to show how $\lambda$ works as a parameter controlling the environment's stability/degree of uncertainty. When $\lambda = 1$, the environment is fully uncertain: one can go from having very good fortune to having catatrophically bad luck. As $\lambda$ grows towards positive infinity, environments grow more consistently favorable, such that the ocassional high failure probability gets more easily outweighted by strings of good fortune.

"""

# ╔═╡ a1051807-7284-48fb-8ade-d347f771be21
begin
	xaxis2 = 0.001:0.001:1.0
	payplot = plot(
		xaxis2,
		g.(xaxis2, l=λ[1]),
		label = "λ = $(λ[1])",
		lw = 2,
		xlab = "stake (s)",
		ylab = "geometric mean payoff",
		alpha = 0.5,
		ylim=(0.9, 1.35)
	)
	
	for l in λ[2:length(λ)]
		plot!(
		xaxis2,
		g.(xaxis2, l=l),
		label = "λ = $l",
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

	hline!([1.0], color="black", alpha=0.2, label="")
	
	payplot
end

# ╔═╡ 026cc528-79c0-418c-84ed-d245960d136c
md"""
It can be shown that when success rates are random, then the best value one can use when choosing a stake is the expected success rate, $E(U) = \frac{\lambda}{1 + \lambda}$. The figure above shows the mean payoffs for different stakes under different values for $\lambda$. The optimal stake for each $\lambda$ is marked by the black dot on the peak of each curve, and is given by $s^* = 2 \frac{\lambda}{1+\lambda} - 1$. The horizontal line marks a growth rate of 1; haivng a payoff below this point means that the individual is incurring in losses rather than gains. It is worth taking not that as $\lambda$ grows smaller, the region around the optimal stake that supports growth also grows smaller. This means that as environments grow more unstable, there is less of an acceptable margin of error around the stake an individual chooses. **Being overly optimisitc can lead to losses**.

This all means that if Niko knows the city's $\lambda$, then he can consistently choose the best stake for his adventures. However, since he does not know it, he must estimate it by sampling the environment before choosing his stake. Over $\tau$ time periods, Niko samples the environment by trying out some activity which yields him either a success or a failure, each activity entailing a sample from $U$. At the end of the $\tau$ test periods, he sums up his successes and divides the sum by $\tau$. This yields him an estimate of $E(U)$, which we call $\bar{u}$. Niko then calculates his estimate of the optimal stake, $\bar{s} = 2 \bar{u} - 1$.

Below, we simulate many Nikos in order to get an idea of the error rate of such a process, starting at $\tau = 5$ and ending at $\tau = 95$. The grey horizontal line represents the optimal stake given the value of $\lambda$. Unsurprisingly, when $\tau$ is low the probability that Niko ends up with a very bad estimate is high. The error in estimation decreases as $\tau$ grows larger, giving Niko a better chance at getting a good estimate. However, even at large $\tau$, there is still considerable variance around the estimates, which can be dangerous, particularly in more unstable environments where there is less tolerance for bad estimates (see above). 
"""

# ╔═╡ 2f776994-4f8c-4e0e-97a5-f551b9eb2fe4
begin
	Random.seed!(12345)
	error_plot1 = scatter(
		get_power_samples(1.5, 5:5:100, 1000), 
		alpha=0.02, 
		markershape=:cross,
		color="black",
		#xlab="number of individual samples (τ) at λ = 1.5",
		#ylab="estimated stake (s̄)",
		legend=false,
		title="λ = 1.5",
		titlefontsize=8
	)
	hline!([2*(1.5 / (1 + 1.5)) - 1], alpha=0.75, lw=2, color="grey")
	
	error_plot2 = scatter(
		get_power_samples(3.0, 5:5:100, 1000), 
		alpha=0.02, 
		markershape=:cross,
		color="black",
		#xlab="number of individual samples (τ) at λ = 3.0",
		ylab="estimated stake (s̄)",
		legend=false,
		title="λ = 3.0",
		titlefontsize=8
	)
	hline!([2*(3.0 / (1 + 3.0)) - 1], alpha=0.75, lw=2, color="grey")
	
	error_plot3 = scatter(
		get_power_samples(5.0, 5:5:100, 1000), 
		alpha=0.02, 
		markershape=:cross,
		color="black",
		xlab="number of individual samples (τ)",
		#ylab="estimated stake (s̄)",
		legend=false,
		title="λ = 5.0",
		titlefontsize=8
	)
	hline!([2*(5.0 / (1 + 5.0)) - 1], alpha=0.75, lw=2, color="grey")

	plot(error_plot1, error_plot2, error_plot3, layout=(3,1), dpi=300)
end

# ╔═╡ 815ac5e7-91ec-49e3-84ae-d6cd2676411d
md"""
In order to make up for the error rate of individual learning, Niko can turn to social learning in order to get an estimate closer to optimal. We know this from Boyd and Richerson: **when individual learning is prone to error, social learning can save the day**. But there are caveats.
"""

# ╔═╡ 299e9851-3bc3-46c4-a270-e2fd56e2c3d3
md"""

##### Horizontal transmission: indirect sampling of environmental information leads to unintended optimism.

**Socially-sampled information might be biased**. When Niko speaks to other immigrants in order to get their impressions, he does not get access to their full experiences, but rather rationalized impressions of them. He now has the daunting task of inferring experiences from behavior. 

In general, individuals are not exposed to others' direct experiences. As with Niko, they must use peers' behavior in order to inferr those experiences, and *then* use that information to complement their own sampling. However, behavior is often a non-linear function of experience. Individuals who, based on their individual experiences, judge the environment to be disfavorable for risk-taking will propose not putting anything at stake. On the other hand, individuals who judge the environment as providing favorable chances will propose staking something. In a sufficiently uncertain environment, after individual sampling, there will be a mix of individuals who propose not staking anything and individuals who propose staking something. For a focal individual who wants to use this information to know what to stake, integrating information from these two groups of peers introduces bias in the estimated stakes (see averaging convex functions of random variables and Jensen's inequality).
"""

# ╔═╡ 2ce26997-43a2-4757-aaf4-b9863ac9ac5a
begin
	expected_plot = plot(
		1:0.01:5,
		expected_stake.(1:0.01:5),
		label="expected stake",
		ylab="stake (s)",
		#xlab="environmental stability (λ)",
		lw=2,
		ls=:dash,
		color="black"
	)
	plot!(
		1:0.01:5,
		optimal_fraction.(1:0.01:5) .* expected_stake.(1:0.01:5),
		label="optimal stake",
		lw=2,
		color="black"
	)

	optimal_pessimism_plot = plot(
		optimal_fraction, 1.0, 5.0,
		color="black",
		lw = 2,
		#legend=false,
		xlab="environmental stability (λ)",
		ylab="credible weight (a)",
		label="optimal credible weight"
	)
	hline!([1.0], lw=2, ls=:dash, label="neutral weight", color="black")

	plot(expected_plot, optimal_pessimism_plot, layout=(2,1))
end

# ╔═╡ 50272356-0411-4728-86cc-313787f04aed
md"""

When Niko speaks to another newcomer $i$, he does not see their $\bar{u}_i$. Rather, he sees their estimate of the optimal stake, $\bar{s}_i$, which is a convex function of $\bar{u}_i$. Jensen's inequality tells us that the distribution of estimated stakes has an expected value that is *higher* than optimal. This means that if Niko tries to hone his estimate by averaging it along with his peers' estimates, he will end up with an overestimate of the optimal stake. In other words, **using social information leads to an optimistic bias**. The above plot (upper pane) shows the difference between $\hat{s} = E(2U - 1)$, the expected stake after averaging social information (dashed line), and $s^* = 2 E(U) - 1$, the actual optimal stake. **As environments grow more unstable, this difference becomes more prominent**. This means that social information must be weighted down by a factor $a \in (0,1)$, which we call a **credible weight**, such that $\alpha \hat{s} = s^*$. This gives us

$a^* = \frac{s^*}{\hat{s}}$

for the optimal credible weight, which we plot above (lower pane). We can think of $\alpha$ as a measure of an individual's optimal aversion to uncertainty. An $a$ closer to 0 indicates individual should be less confident in obtained estimates, whether they come from individual learning or horizontal transmission, whereas an $a$ of 1 indicates they should exhibit full confidence. It is thus a form of optimal *pessimism*, itself a manifestation of optimal risk aversion.

The above plots assume that each individual draws a value of $u$ from $U$, which they go on to use in calculating their observable estimate. In practice, however, what each individual draws is an estimate of $E(U)$, with the variance of these draws depending on $\tau$, as we saw above. In order to visualize the dependence on $\tau$, we simulate many immigrants coming into Liberty City, each sampling the city for $\tau$ time periods and then sharing their estimated stakes $\bar{s}$ among one another.

"""

# ╔═╡ 6145ba3c-1f69-4f9e-b850-89bbee75c57f
begin
	Random.seed!(12345)
	est01 = estimate_plots(1.0, tries[3], 1000, title=true, label=true)
	est02 = estimate_plots(1.5, tries[3], 1000, title=false)
	est03 = estimate_plots(2.0, tries[3], 1000, title=false)
	plot(
		[est01..., 
		est02..., 
		est03...]..., 
		layout=(3,2),
		link=:all
	)
end

# ╔═╡ 51df7295-a53d-44f9-837d-6337f45050b3
md"""
### IMPORTANT: MEDIAN DOESN'T WORK
"""

# ╔═╡ 78c9f173-1f49-49c1-bae9-ec40cdd4b492
md"""

The above plots are simulations for $\tau = 5$. The histograms on the left column show the distributions of $\bar{u}$ among immigrants (not observable) while the right column shows the (observable) distribution of $\bar{s}$. The solid line shows the optimal stake for the given value of $\lambda$, while the dashed line shows the stake an immigrant like Niko would arrive at if they simply averaged their own estimated stake with the one obtained from their peers. Note how the optimism is induced purely by the averaging process, without a need for individuals to have an optimistic preference themselves.

"""

# ╔═╡ dd260c75-0e31-4a9d-9dc2-f8079e5c98e2
begin
	Random.seed!(12345)
	jensen_gap = Vector{Vector{Float64}}()
	optimal_pessimism = Vector{Vector{Float64}}()
	
	for t in tries
		mstake = meanstake.(20000, t, 1:0.01:5)
		smean = stakemean.(20000, t, 1:0.01:5)
		
		push!( jensen_gap, mstake .- smean )
		push!( optimal_pessimism, smean ./ mstake )
	end
	
	jensen_plot = plot(
		1:0.01:5,
		jensen_gap[1],
		label = "$(tries[1])",
		legendtitle="τ",
		legendtitlefontsize=8,
		legendfontsize=6,
		#xlab = "environmental stability (λ)",
		ylab = "Jensen gap (ŝ-s*)",
		lw=2
		)
	counter = 2
	for p in jensen_gap[2:length(jensen_gap)]
		plot!(
		1:0.01:5,
		p,
		label = "$(tries[counter])",
		#xlab = "environmental stability (λ)",
		#ylab = "Jensen gap (ŝ - s*)",
		lw=2
		)
		global counter += 1
	end
	
	pessimism_plot = plot(
		1:0.01:5,
		optimal_pessimism[1],
		legend=false,
		label = "$(tries[1])",
		xlab = "environmental stability (λ)",
		ylab = "credible weight (s*/ŝ)",
		lw=2
	)
	counter = 2
	for p in optimal_pessimism[2:length(optimal_pessimism)]
		plot!(
		1:0.01:5,
		p,
		label = "$(tries[counter])",
		#xlab = "environmental stability (λ)",
		#ylab = "Jensen gap (ŝ - s*)",
		lw=2
		)
		global counter += 1
	end

	plot(jensen_plot, pessimism_plot, layout=(2,1))
end

# ╔═╡ 63aa936d-7955-46b0-9ac1-c1d23268079b
md"""
The above plots show the Jensen gap (the difference $\hat{s} - s^*$ between estimated stakes and optimal stakes) and the optimal credible weight ($a = \frac{s^*}{\hat{s}}$) for different values of $\tau$. It is clear that if test periods are short ($\tau$ small) then individuals should apply a more aggresive credible weight, which is to say that they should be more pessimistic. This makes sense, as the quality of social information also depends on the quality of peers' bouts of individual learning. And even when $\tau$ is large, more unstable environments require higher degrees of pessimism. **As environments become more unstable, at some point higher degrees of pessimism will be required to mitigate optimisitic biases, regardless of the length of individual learning periods.**

At this point it is clear that being pessimistic can be advantageous through the uncertainty introduced by random success rates and the difference between unobservable experiences and the observable behaviors they lead to. Since staking too much can lead to losses, Niko should be pessimistic in one way or another when dealing with the information he gathers. We now proceed to examine what happens if individuals can lose everything when they hit an absorbing barrier, destroying their ability to bounce back from losses that are too great.
"""

# ╔═╡ 02b671bc-b9d9-45f5-a3ac-ef4d8a413459
md"""
##### Initial wealth and the looming threat of absorbing barriers: the possibility of ruin intensifies the need for pessimism.

In real life, there are hidden risks that are difficult to account for when choosing a proper course of action. An common manifestation of this is an **absorbing barrier: a "point of no return" for a payoff trajectory**. For example, if payoffs are meant to represent a strategy's reproductive success, then obtaining a payoff of zero in any given generation is an absorbing barrier: having no offspring in one generation means there are no offpsring in any subsequent generations. Similarly, if payoffs represent wealth, then negative wealth growth at any given period can lead to bankruptcy. On the other hand, individuals can also come in with some initial advantage into the betting dynamic, given by an initial level of wealth. Starting off with a higher level of wealth can make it harder to hit that absorbing barrier, by providing more possibilities for bounce-back in case of losses.

Rememeber that at a given time $T$, the payoff of an individual staking $s$ is

$V_T (s) = w_0 \prod_{t=1}^T e^{g_t (s)}$

Since we can write $w_0$ as $w_0 = e^{g_0}$, the above expression is the same as

$V_T (s) = e^{\sum_{t=0}^T g_t (s)} = e^{\nu + \sum_{t=1}^T g_t (s)}$

where $\nu = g_0$ is our measure of initial (log) wealth.

We can then simulate the effects of absorbing barrier and initial wealth by setting an absorbing barrier at $V = 0.1$. This means that if the total payoff at any point of the payoff trajectory becomes lower than 0.1 (which is the same as saying that the agent has lost most of their wealth or soma), then payoffs are set to zero, a state representing total ruin.
"""

# ╔═╡ 97af9b4a-cb54-4592-9fb1-496d0cb5d298
begin
	Random.seed!(123456)
	splots_barrier = [
		simplots(
			λ, s, 
			true, 2000, 
			cap, cap == 1 ? true : false, 
			cap == 3 ? true : false
		) 
		for cap in init_capital
	]

	plot(
		splots_barrier...,
		layout=(3,1),
		plot_title="mean payoffs under absorbing barriers",
		link=:all,
		ylim=(0.9, 1.3)
	)
end

# ╔═╡ b729db3a-6dcf-427b-b6ff-8b8ceadbc6e6
plot(splots_barrier..., layout=(3,1), ylim=(0.9, 1.3))

# ╔═╡ 92dcf70c-a58b-481b-b473-e7c0e5c180b3
md"""

The above plot shows what happens to mean payoffs under these new dynamics. The effects of absorbing barriers are dire: optimal stakes are lower than without barriers, and when individuals start out poor ($\nu = 1$) the optimal stakes can be quite conservative in unstable environments, with very little margin for error. As initial wealth levels increase, so do optimal stakes. However, these are generally lower than the optimal stakes in a dynamic without absorbing barriers, even at very large levels of intial wealth ($\nu = 5$). Since the initial learning process for estimating success rates is free of risk itself (remember we learn about the environment in relatively risk-free testing periods, like childhood and adolescence), then it cannot represent the effect of absorbing barriers, and thus we can conclude that **absorbing barriers induce a further need for pessimistic weighting, independent of the learning strategy employed to estimate environmental instability.**

"""

# ╔═╡ 197c9fc9-8b1c-484e-8f0e-cffa08375296
md"""
### CLAVE: SOCIAL INFO ON CATASTROPHES AS A DRIVER FOR INCREASING GROUP SIZE
"""

# ╔═╡ 0ff38bba-587e-44b4-a65c-b724bf8c89ca
md"""

**Key takeaways #1**: 

- Individual learning becomes more unreliable as learning periods become shorter and as environments become more unstable. Even long learning periods can lead to significant error when environments are very unstable. Decision-makers thus should be pessimistic about their own estimates after individual learning in order to be on the safe side of the error, as being too optimistic can lead to ruin, while being too pessimistic will only lead to slower payoff growth (or no growth at all in the extreme case of full pessimism, $a = 0$).

- Precautions must be taken when using horizontally-transmitted social information to complement individual learning as environmental stability decreases and environments become more uncertain. An optimistic bias can creep into estimates without any sort of mechanism explicitly promoting it in individuals' cognitive apparatus. Decision-makers have to use corrective cognitive mechanisms, such as implementing pessimistic credible weights on the information they receive (risk aversion), in order to avoid potentially dangerous unintended optimism.

- The presence of absorbing barriers and the possibility of ruin lead to the need for pessimism regardless of the combination of individual and social learning an individual uses. This makes it even harder to find the appropriate level of pessimism that an individual should use.

Acquiring the right pessimistic cognitive strategy is thus the first-order correction problem of risky decision-making in uncertain environments, applying to individual learning and horizontal transmission. Given that in real life there are always probabilities of ruin for those who take excessive risks, there are clear evolutionary pressures favoring the emergence of a pessimistic bias in decision-making. We argue that this pessimistic bias is more likely to arise from cultural evolutionary processes rather than through genetic evolution, as human environments can change from relatively stable to relative unstable in the span of mere generations, prompting decision-makers to use learning as an adaptive tool. Once test periods are finished, learning is not necessarily finished: as we observe the successes, losses and ruin of others around us, more information is revealed about the environment than that which we can get during sheltered testing periods. For example, a sensibility to observed ruin events can use information about visible catastrophic failures without having to be a victim of one. By using this information to adjust pessimistic credible weights as a function of how frequent ruin is in the environment, decision-makers could arrive at the optimal degree of pessimism, although some of them will inevitably fail.

The kind of learning described until now is good for when environmental conditions change from one generation to another. But when environments maintain the same level of initial wealth and environmental stability for many generations, learners need not expose themselves to the possible errors of individual learning and horizontal transmission. Instead, vertical and oblique transmission can cue learners into what has been working for the past generations, who already performed in the environment. Thus, learners can rely on culture as an adaptive tool. This leads to the second order problem of risky decision-making in uncertain environments, which we examine below.
"""

# ╔═╡ 28414f1a-83d5-4a96-b1b2-b43bd599f7ed
md"""

#### Part Two: The Cultural Inheritance of Risk Attitudes.

Niko knows there are many unknowns about the information he has managed to get about life in Liberty City, and that he should be careful. But being too careful might lead him to passing on actual good opportunities. Luckily for him, his is not the first wave of immigrants to arrive at the metropolis. He sets out to find those who came before him and hear their stories of success and their advice on how pessimistic to be.

Niko speaks to Uncle Yuri, the owner of a successful deli who tells him how he put all of his savings behind his then-nascent business. On TV, he watches broadcasts of immigrant success stories like that of millionaire Edna, who says she found a welcoming city full of opportunity. This fills Niko with a sense of confidence; surely it is not necessary to be that pessimistic! In paying attention to these success stories, Niko inadvertedly exposes himself to danger yet again, as he fails to consider information about those who failed and experienced ruin. While Uncle Yuri's and Edna's successes are commendable, they are also more visible and salient than the failures of those who might have been using equally optimistic credible weights.

"""

# ╔═╡ 5571c8e1-21bb-4b6e-80bf-4c811eae7aa3
md"""

##### Success-biased social sampling hides failures, leading to dangerous optimism.

Success bias is not the same as payoff bias. While payoff bias looks at a distribution of strategies alongside their payoffs and preferentially chooses among the strategies with the highest associated payoffs, **success bias is about visibility: agents that did well enough to be thought of a successful are those who will be considered as possible targets for learning**. Thus, success bias works at the level of the learning pool, and only thereafter can another learning strategy such as unbiased learning or payoff bias act upon the pool of possible targets.

When successes are disproportionally visible (or, in the most extreme case, the only visible behavior), then the optimal degree of pessimism can actually appear suboptimal, while dangerously-optimistic weighting can seem like the highest-paying solution. This happens because the higher variability of optimistic weighting leads to long tails in payoffs, which under success-biased sampling can end up generating skewed estimates of the average payoff of a weighting strategy. Success bias is likely more common today than in days past, due to mass media that overblows the visibility of highly successful individuals, to the point of having [ad campaigns that have recognized individuals vouch for behaviors that are completely alien to the source of their success](https://theintercept.com/2022/10/26/matt-damon-crypto-commercial/). This is an example of how, just like the estimation error in the previous section, an optimistic bias can arise without there being mechanisms generating it at the level of an individual's cognition. Instead, success bias can arise purely from external processes, such as market competition between mass media outlets.

When looking at previous generations in order to choose a the right degree of pessimism, success bias can paint a distorted picture of what works best. This is due to the fact that the least pessimistic strategies also have the highest payoff variance, so that the few non-pessimistic individuals who are lucky enough to succeed can also happen to do so with very high payoffs. Thus, the average surviving payoff of a given level of pessimism can look very different than the actual average payoff, which includes the (zero) payoffs of ruined individuals.

"""

# ╔═╡ 42a59a13-22e0-40ab-92a3-00db2b04249f
plot_meanconf([1.5, 3.0, 4.5], [1, 2, 3])

# ╔═╡ f201eb67-f78b-47ae-ab06-9c612c5e2c0b
#=
we show the result of simulating payoffs from the very start of the estimation process. $N$ agents test out the environment for $\tau$ time periods and produce individual estimates $\hat{s}$. These estimates are then averaged by a each agent and weighted down by a credible weight $\alpha$. The agents then go on to play the proportional betting game using their weighted estimates, and their time-averaged payoffs are then averaged across all $N$ agents to produce the mean payoff, while for the success-biased mean payoff only average payoffs higher than 1 are kept and averaged. 
=#
md"""
In the above plot, bets are simulated for values of $s$ going from 0 to 1, in steps of 0.01. The dashed line shows the success-biased mean payoff at a given $s$ (in which only payoffs greater than 0 are considered), while the solid grey line shows the full mean payoff. It is evident that in most cases these two payoffs will diverge at some point as $s$ goes from 0 to 1, which is meant to show how much success bias can distort the payoff landscape for learners. 

The grey solid line is what decision-makers want to optimize for, so its considerable difference from what is actually visible (the dashed line) poses a significant challenge, except perhaps when learner populations start very rich and environments are very stable. Very little changes by making individual learning periods longer, as can be seen in the plot below. How, then, can the optimal strategy be recovered from information that has been sifted through a success-biased filter?
"""

# ╔═╡ 68bfe7f9-d733-41b3-9f0e-cdca87bcc894
md"""
##### Payoff-biased social learning also leads to optimism due to the higher variance of less pessimistic strategies.

**Even in the absence of success bias, payoff bias on its own can still pose problems**. What appears as a handy learning strategy where we focus on the highest observed payoffs and adopt the strategies associated to them can lead us to make fatal mistakes when the strategies considered have to do with risk management. However, it is likely that success bias and payoff bias often occur together.
"""

# ╔═╡ ff971851-a7f3-4cda-a1e0-793a075c2893
begin
	Random.seed!(12345)
	n_samples = 200
	
	w1 = 0.45
	pop1 = [
			d[2] 
			for d in simpop(
				2.0, 1,
				n_samples = n_samples, w_constant=true, w=w1
			)[2]
		]
	prop_p1 = length(pop1) / n_samples

	w2 = 0.15
	pop2 = [
			d[2] 
			for d in simpop(
				2.0, 1,
				n_samples = n_samples, w_constant=true, w=w2
			)[2]
		]
	prop_p2 = length(pop2) / n_samples
	
	histogram( 
		pop1, 
		label="$w1", alpha=0.3,
		legendtitle="s",
		color="dark green",
		xlab="distribution of success-biased mean payoffs"
	)
	annotate!(
		1.03, n_samples*0.15, 
		text("prob. survival\n$prop_p1", 7, "dark green" ),
		alpha=0.5
	)

	annotate!(
		1.075, n_samples*0.06, 
		text("prob. survival\n$prop_p2", 7, "dark red" ),
		alpha=0.5
	)
	
	histogram!(
		pop2, 
		label="$w2", alpha=0.5
	)
end

# ╔═╡ 9cfc79d3-f9aa-488b-832a-e02bf711ca19
md"""
The above figure is an illustration of how payoff bias can make agents fall for one of the classic blunders. Plotted for $\lambda = 2$, $\nu = 1$ and $\tau = 100$, the figure shows the payoffs of 500 agents who go through the proportional betting game for 2000 time periods. This procedure is done for $s \in [0.25, 0.45]$. It is evident from looking at the histograms that 0.25 is the better-performing stake, both in terms of modal payoff and probability of survival (which is just the complement of the probability of ruin). However, the higher variance of the less pessimistic stake leads to a few payoffs that are considerably higher than the more pessimistic strategy. This right tail of high payoffs will be especially salient to payoff-biased learners, nudging them towards suboptimal (even dangerous) levels of pessimism.

Paying special attention to the highest payoffs of the previous generation leads decision-makers to honing in on the tails of the observed payoff distribution. This can lead them to preferentially choose overly-optimisitic strategies that worked well for a few individuals, but that will lead to bad payoffs or even ruin for most individuals who use them. Both with success-biased social sampling and without, only conformity/frequency-biased social learning, in which information about frequency is used to penalize rare successes before searching for the highest payoffs, will steer decision-makers away from disaster, while still managing to use payoff information strategically.
"""

# ╔═╡ d9f28433-f818-42c2-8eb0-aa844fa3eb1a
md"""
#### Frequency biases can recover optimal pessimism from success-biased model pools, while reaping the rewards from payoff biased strategies.
"""

# ╔═╡ 8565e558-d7e3-443b-a56f-a82ffae661ce
md"""
When looking at the previous generation and observing their strategies and payoffs, one can be conformist by looking at the median strategy present in the population. If a decision-maker is fully conformist, then they would immediately choose the median strategy. In a fortunate turn of events, this works well in the presence of success bias: since the riskier strategies are less represented in the learning model pool, being conformist will avoid these. But perhaps there are circumstances in which incorporating payoff information might still be useful. Must we choose between one strategy or the other? Not necessarily. For a set of observed payoffs $v \in V$, each associated to an observed strategy $s \in S$, we can write down a hybrid frequency-payoff biased learning rule like so:

$w (s, v) = Κ_S(s) \cdot v$

where $w \in W$ is then a frequency-weighted payoff, and

$K_S(s) = \text{exp} \bigg( \frac{(s - M[S])^2}{\zeta} \bigg)$

is a weight determining the effect of frequency bias on payoffs, such that payoffs associated with deviations from the median observed strategy $M[S]$ are penalized. The degree to which they are penalized depends on $\zeta$, which is the parameter controlling frequency bias. When $\zeta$ approaches 0 from the right, learners approach full conformity, choosing purely based on how close a strategy is to the median observed strategy. On the other hand, as $\zeta$ grows large, all weights approach 1 regardless of distance from the median. This means that if one chooses the highest payoff from the set of weighted payoffs $W$, one can get pure conformity, pure payoff bias, or anything in between by varying the $\zeta$ parameter.

"""

# ╔═╡ 7781b6d1-8618-4ec4-bc0d-c5589c2c55b8
function plot_confpanels(
	λ, C, n; 
	α = 1, β = 1, 
	w_constant=false, w=0:0.05:1|>collect, 
	n_samples = 5,
	T = 5000, seed=574869
)
	seeders = get_seeds(λ, C, seed=seed)
	
	scatterconfs = [
		plot_scatterconf(
			l, c,
			n = n,
			T = T,
			w_constant = w_constant,
			w = w,
			var=0.025,
			n_samples=n_samples,
			xlab = c == first(C) ? "λ = $l" : ( c == last(C) ? "stake (s)" : "" ),
			ylab = l == first(λ) ? "ν = $c" : "",
			legend= (c, l) == (last(C), first(λ)) ? :topright : false,
			gp = c == last(C) ? :bottom : :top,
			alpha=0.1
		)
		for l in λ,
		c in C
	]
	
	plot(
		vec(scatterconfs)...,
		layout = (length(C), length(λ)),
		link = :all,
		plot_title = "observed stakes vs. observed payoffs",
		ylim = (0.95, 1.35),
		dpi = 300
	)
end

# ╔═╡ 0e9f8365-c8ec-4f45-a5be-9accb1a3e049
begin
	Random.seed!(123456)
	plot_confpanels(
		[1.5, 3.0, 4.5], 
		[1, 5], 
		100,
		T=1000
	)
end

# ╔═╡ 2197f0d3-6aca-412f-86f8-c9f9ed54ad7e
md"""
The above figure shows scatterplots representing what a learner sees when they see the previous generation of decision-makers. For each plot, 100 decision-makers are simulated using strategies drawn from a Beta(1, 1) distribution. The focal agent then sees both the weight they used as well as the associated payoff, for all agents who managed to survive. The dashed line is the stake a learner using pure payoff bias would choose: the one associated to the highest observable payoff. The solid line represents the median stake among survivors: the $s$ that a purely frequency-biased learner would end up choosing. Between these two, $\zeta$ can be tweaked to obtain a form of hybrid frequency-payoff bias. The blue notch on the x-axis of the plots is the optimal stake, given $\lambda$ and $\nu$. Thus, we arrive at the following conclusions:

- In very unstable environments (here given by $\lambda = 1.5$), one should be fully conformist, regardless of initial wealth. Even by being conformist, a degree of optimism cannot be fully avoided in low wealth populations, but conformity does help mitigate the dangers of overconfidence. In environments like these with low initial wealth, one should be additionally pessimistic on top of what conformity leads to, although higher initial wealth lessens the need for extra pessimism. At very high wealth, payoff bias and conformity will tend to converge, so one can be either one or the other.

- As environments become more stable, but still fluctuate considerably (here given by $\lambda = 3$), hybrid conformity-payoff bias becomes optimal, especially as initial wealth increases. Everyone is best by being conformist here, but the wealthy can afford to be less conformist than the downtrodden.

- As environments stabilize further (here given by $\lambda = 5$), the optimal learning strategy will still be an implementation of hybrid frequency-payoff bias, except for wealthy populations. The degree of frequency bias depends on decision-makers' intial wealth. If the learner population starts out with scarce resources, then being more conformist is more optimal. At the other end of the spectrum, if the learner population starts out wealthy, then conformity should be abandoned as the optimal observed stake approaches the one that pure payoff bias would choose.
"""

# ╔═╡ 308a5b66-0dcf-4fee-a36c-68b68e6dcf4b
md"""
**Key takeaways #2**: When using social learning to find the right degree of risk aversion by looking at the performance of previous generations, individuals are prone to a second-order correction problem, analogous to the first-order problem described above. Using payoff information can lead to a biased estimate of what pessimistic strategy works best, as the attractiveness of a strategy will be a non-linear (convex) function of its payoffs. Pure payoff bias can lead individuals to choose strategies that worked very well for the few, but spelled disaster for the many. This problem grows more dire in the presence of low initial wealth and as environments grow more unstable and uncertain. Incorporating frequency information through a cognitive strategy of conformist bias (weighting the payoffs of strategies by their distance to the median surviving strategy) is a possible solution to this problem.

These results imply that natural selection will favor the cultural evolution of frequency-biased learning, especially in environments that have a continued generational history of instability. Only in the combination of highly initially wealth populations in particularly stable environments does conformity start losing its edge, with pure payoff bias leading to the best performance. Risk aversion should always be present in one way or another, although it should decrease as environments become more stable. Thus, pessimism/risk aversion should co-occur with conformity, and this association should be stronger among decision-makers of lower initial wealth. If decision-makers come from wealthier backgrounds, then this association should weaken in stable environments, but remain strong in unstable environments.
"""

# ╔═╡ 4b24bff6-0a95-4763-b8da-76cf0884f543
md"""
### Final Remarks

Alan Moore's Watchmen, cooperation and uncertainty.

Elites and political manipulation through narratives that offload uncertainty onto outgroups, distract from actual root causes.

Mass media, social media platforms and biases towards success-related narratives. Conformity and risk aversion as possible protective measures against hype leading to risky investment, and against potential dangers to mental health.

What do we give away? Flexibility when it comes to exploring our environments, social or otherwise. If we wish to maintain this flexibility, we should concentrate on building societies where people feel safe enough to take risks when investing their time, resources and learning opportunities.

"""

# ╔═╡ ec990ffb-727d-4419-9550-ad7f7a61bf85
md"""
#### Experimental area
"""

# ╔═╡ cb2f27e7-4001-4f8f-83a9-d26a166be9b5
function pull_stake(stakes, sens)
	for i in 1:length(stakes)
		if sens[i] > 0.0
			stakes[i] = clamp(
				stakes[i] - rand( Exponential(sens[i]) ),
				0.0, 1.0
			)
		end
	end
	return stakes
end

# ╔═╡ d621aecd-79eb-4b69-add8-59de52d47d05
function simulate_gambles_sensitive(λ, stake, n;
	sens=0.01,
	u=1.0,
	init_capital=1.0,
	seasons=2000,
	rounds=1,
	abarrier=true,
	succ_bias=false
	)

	sens = length(sens) == 1 ? repeat([sens], n) : sens
	stakes = repeat([stake], n)
	log_capitals = repeat([init_capital], n)

	for i in 1:seasons

		results = [
			rand() < u / rand( Pareto(λ) ) ? log(1 + stakes[k]) : log(1 - stakes[k]) 
			for k in 1:n
		]
		
		log_capitals = log_capitals .+ results
	
		if abarrier
			
			for i in 1:n
				if log_capitals[i] != -Inf
					if log_capitals[i] < 0.0
						log_capitals[i] = -Inf
						stakes = pull_stake(stakes, sens)
					end
				end
			end
		end

	end

	
	if succ_bias
		dead = findall(c -> c == -Inf, log_capitals)
		log_capitals = [log_capitals[i] for i in 1:n if i ∉ dead]
		stakes = [stakes[i] for i in 1:n if i ∉ dead]
	end

	return (log_capitals, stakes)

end

# ╔═╡ a0f6d5a5-f8be-4651-a9a7-d52b7ad8a6e0
begin
	Random.seed!(123456)
	log_payoffs, stakes = simulate_gambles_sensitive(2.0, 0.5, 50)
	pays_adaptive = exp.(log_payoffs ./ 2000)
	pays = [exp( simulate_gambles(1.0, 2.0, 0.5) ./ 2000 ) for i in 1:50]

	mean_adapt = round(mean( exp.(log_payoffs ./ 2000) ), digits=2)
	mean_no = round(mean( pays ), digits=2)
	
	histogram(
		filter(x -> x > 0.0, pays_adaptive),
		#legendtitle="",
		label="adaptive pessimism",
		xlab="mean payoff (survivors only)"
	)
	histogram!(
		filter(x -> x > 0.0, pays), alpha=0.5,
		label="insensitive"
	)
	annotate!(1.07, 13, text("mean payoff\n$(mean_adapt)", 8, color="blue"))
	annotate!(1.03, 15, text("mean payoff\n$(mean_no)", 8, color="red"))
end

# ╔═╡ 77ae48e8-36ac-4909-932a-aacfb33ff659
function compare_learning(
	λ, cap, T, tries, n_samples, sens;
	total_samples=1000, seasons=5000, n_opt=5000
	)
	
	opt_surv = (0:0.01:1|>collect)[
		findmax(
			sim_payoffs(λ, 0:0.01:1, cap, seasons=seasons, n=n_opt
			)[2])[2]
	]

	payoffs = Vector{Float64}()
	meds = Vector{Float64}()
	med_dev = Vector{Float64}()
	pays = Vector{Float64}()
	pay_dev = Vector{Float64}()
	
	for i in 1:total_samples
		
		meaner = mean(calculate_stake.( calculate_estimates(n_samples, tries, λ) ))
		lpays, stakers = simulate_gambles_sensitive(
			λ, meaner, n_samples, 
			sens=sens, init_capital=cap, seasons=T
		)

		med = median(stakers)
		mdev = med - opt_surv
		pay = stakers[findmax(lpays)[2]]
		pdev = pay - opt_surv
		mean_payoff = mean(exp.(lpays ./ T))

		push!(payoffs, mean_payoff)
		push!(meds, med)
		push!(med_dev, mdev)
		push!(pays, pay)
		push!(pay_dev, pdev)
		
	end

	return (payoffs, meds, med_dev, pays, pay_dev)
	
end

# ╔═╡ e205678e-e9ae-46ed-ab94-7d827bdaabc0
function plot_confcomparison(
	λ, ν, T, tries, N, sens; 
	ylim=(-0.5, 0.5), xlab=100, ylab=0, yticks=true, lquant=0.25, hquant=0.75
)

	pfs, medbias, mdev, paybias, pdev = compare_learning(
		first(λ), ν, T, tries, N, sens
	)

	medbias_vec = [medbias]
	paybias_vec = [paybias]
	lambdas = [first(λ)]

	median_mdev = median(mdev)
	med_ϵ⁻ = median_mdev - quantile(mdev, lquant)
	med_ϵ⁺ = quantile(mdev, hquant) - median_mdev

	median_pdev = median(pdev)
	pay_ϵ⁻ = median_pdev - quantile(pdev, lquant)
	pay_ϵ⁺ = quantile(pdev, hquant) - median_pdev
	
	comparison_scatter = scatter(
				[first(λ)],
				[median_mdev],
				yerr=[(med_ϵ⁻, med_ϵ⁺)],
				c="pink",
				legend=false,
				ylim=ylim,
				xlim=(first(λ)-0.5, last(λ)+0.5),
				xlab= xlab ? "environmental stability (λ)" : "",
				xlabelfontsize = 6,
				xtickfontsize = 6,
				ytickfontsize = 6,
				ylab= ylab == ν ? "deviation from optimal stake" : "",
				xticks=(minimum(λ):0.5:maximum(λ)),
				yticks=yticks,
				alpha=0.75,
				grid=false,
				xrotation=90,
			)
	scatter!(
		[first(λ)],
		[median_pdev],
		yerr=[(pay_ϵ⁻, pay_ϵ⁺)],
		c="dark blue",
		alpha=0.5
	)
	hline!([0.0], ls=:dash, lw=2, color="black", alpha=0.75)
	
	if length(λ) > 1
		@threads for lamb in λ[2:end]
			pfs, medbias, mdev, paybias, pdev = compare_learning(
				lamb, ν, T, tries, N, sens
			)

			push!(medbias_vec, medbias)
			push!(paybias_vec, paybias)
			push!(lambdas, lamb)
		
			median_mdev = median(mdev)
			med_ϵ⁻ = median_mdev - quantile(mdev, lquant)
			med_ϵ⁺ = quantile(mdev, hquant) - median_mdev
		
			median_pdev = median(pdev)
			pay_ϵ⁻ = median_pdev - quantile(pdev, lquant)
			pay_ϵ⁺ = quantile(pdev, hquant) - median_pdev
			
			scatter!(
				[lamb],
				[median_mdev],
				yerr=[(med_ϵ⁻, med_ϵ⁺)],
				c="pink",
				alpha=0.75,
			)
			scatter!(
				[lamb],
				[median_pdev],
				yerr=[(pay_ϵ⁻, pay_ϵ⁺)],
				c="dark blue",
				alpha=0.5
			)
		end
	end

	return [comparison_scatter, medbias_vec, paybias_vec, lambdas]
	
end

# ╔═╡ 92563edb-058f-4247-b8e5-fd5c24fb4838
function plot_payoff_confcomparison(
	λ,
	ν,
	medbias,
	paybias;
	ylab=1,
	title=true,
	lquant=0.25,
	hquant=0.75,
	seasons=1000,
	n=5000,
	leg=false,
	xticks=true,
	yticks=true,
	ylim=(0.4, 1.3)
)
	
	comparison_scatter = scatter(
				[], [],
				c="pink",
				legend=leg,
				label="conformity",
				legendfontsize=6,
				xlim=(minimum(λ)-0.5, maximum(λ)+0.5),
				ylim=ylim,
				xlab=title ? "ν = $ν" : "",
				guide_position=:top,
				ylab= ylab == ν ? "geometric mean payoff" : "",
				xticks=xticks ? (minimum(λ):0.5:maximum(λ)) : false,
				ytickfontsize = 6,
				yticks=yticks,
				alpha=0.75,
				grid=false,
				xtickfontsize = 6,
				xrotation=90
			)
	scatter!(
		[], [],
		c="dark blue",
		label="payoff bias"
	)
	hline!([1.0], alpha=0.75, lw=2, color="black", ls=:dash, label="")
	
		@threads for h in zip(medbias, paybias, λ) |> collect
			
			survpays_med, fullpays_med, probsurv_med = sim_payoffs(
				h[3], h[1], ν, 
				seasons=seasons, n=n
			)
			survpays_pay, fullpays_pay, probsurv_mpay = sim_payoffs(
				h[3], h[2], ν,
				seasons=seasons, n=n
			)
			
			median_medpay = median(fullpays_med)
			med_ϵ⁻ = median_medpay - quantile(fullpays_med, lquant)
			med_ϵ⁺ = quantile(fullpays_med, hquant) - median_medpay

			median_paypay = median(fullpays_pay)
			pay_ϵ⁻ = median_paypay - quantile(fullpays_pay, lquant)
			pay_ϵ⁺ = quantile(fullpays_pay, hquant) - median_paypay
			
			scatter!(
				[h[3]],
				[median_medpay],
				yerror=[(med_ϵ⁻, med_ϵ⁺)],
				c=["pink"],
				label="",
				alpha=0.75,
			)
			scatter!(
				[h[3]],
				[median_paypay],
				yerror=[(pay_ϵ⁻, pay_ϵ⁺)],
				c="dark blue",
				label="",
				alpha=0.5,
			)
		end
	
	return comparison_scatter
	
end

# ╔═╡ 4af45122-41e8-4ec4-8ac9-b56cd24996ee
function full_comparison(
	λ, ν, T, tries, n_community, sens;
	xlab=true, ylab=1, ylim1=(-0.5, 0.5), ylim2=(0.4, 1.3),
	leg=false, yticks=true, lquant=0.1, hquant=0.9,
)
	
	comparison_plot, medbias, paybias, lambdas = plot_confcomparison(λ, ν, T, tries, n_community, sens, xlab=xlab, ylab=ylab, ylim=ylim1, yticks=yticks)
	
	payoff_comparison_plot = plot_payoff_confcomparison(lambdas, ν, medbias, paybias, leg=leg, ylab=ylab, yticks=yticks, ylim=ylim2)
	
	plot(
		payoff_comparison_plot, 
		comparison_plot, 
		layout=(2,1), 
		ylabelfontsize=8
	)
end

# ╔═╡ 59937c6e-ae84-4264-b88c-2c787087b712
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(54321)
	compare_tau5_N10 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 5, 10, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0.5, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau5N10panel = plot(compare_tau5_N10..., layout=(1,4), dpi=300)
	savefig(tau5N10panel, "comparison_tau5_N10.pdf")
	
	compare_tau10_N10 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 10, 10, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0.5, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau10N10panel = plot(compare_tau10_N10..., layout=(1,4), dpi=300)
	savefig(tau10N10panel, "comparison_tau10_N10.pdf")
	
	compare_tau100_N10 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 100, 10, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0.5, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau100N10panel = plot(compare_tau100_N10..., layout=(1,4), dpi=300)
	savefig(tau100N10panel, "comparison_tau100_N10.pdf")
end
  ╠═╡ =#

# ╔═╡ 3574fdad-f923-4755-8ad7-325634d8f6c3
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(948576)
	compare_tau5_N25 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 5, 25, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau5N25panel = plot(compare_tau5_N25..., layout=(1,4), dpi=300)
	savefig(tau5N25panel, "comparison_tau5_N25.pdf")
	
	compare_tau10_N25 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 10, 25, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau10N25panel = plot(compare_tau10_N25..., layout=(1,4), dpi=300)
	savefig(tau10N25panel, "comparison_tau10_N25.pdf")
	
	compare_tau100_N25 = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 100, 25, 0.05, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau100N25panel = plot(compare_tau100_N25..., layout=(1,4), dpi=300)
	savefig(tau100N25panel, "comparison_tau100_N25.pdf")
end
  ╠═╡ =#

# ╔═╡ 04924952-b873-4c5d-88c1-c11ad767d100
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(123654)
	compare_tau5_N25_ls = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 5, 25, 0.01, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau5N25panel_ls = plot(compare_tau5_N25_ls..., layout=(1,4), dpi=300)
	savefig(tau5N25panel_ls, "comparison_tau5_N25_ls.pdf")
	
	compare_tau10_N25_ls = [
			full_comparison(
			1.5:0.5:5.0, i[1], 1000, 10, 25, 0.01, 
			ylab=0, leg=i[2], yticks=i[3], ylim1=(-0.55, 0.55), ylim2=(0.1, 1.5)
			)
			for i in [
				(0, false, true), 
				(1, false, false), 
				(2, false, false), 
				(3, :bottomright, false)
			]
		]
	tau10N25panel_ls = plot(compare_tau10_N25_ls..., layout=(1,4), dpi=300)
	savefig(tau10N25panel_ls, "comparison_tau10_N25_ls.pdf")

	compare_tau100_N25_ls = [
		full_comparison(
		1.5:0.5:5.0, i[1], 1000, 100, 25, 0.01, 
		ylab=0, leg=i[2], yticks=i[3]
		)
		for i in [
			(0, false, true), 
			(1, false, false), 
			(2, false, false), 
			(3, :bottomright, false)
		]
	]
	tau100N25panel_ls = plot(compare_tau100_N25_ls..., layout=(1,4), dpi=300)
	savefig(tau100N25panel_ls, "comparison_tau100_N25_ls.pdf")
end
  ╠═╡ =#

# ╔═╡ 9cb067fa-4ab2-41cf-9688-eff29b8271ef
function plot_sensitivity(λ, cap, N, tries, T; succ_bias=false, n_samples=1000, xlab=true, ylab=true)

	lambdas = Vector{Float64}()
	apays = Vector{Vector{Float64}}()
	
	@threads for lp in λ
		adapt_payoffs = median.(
		zip([
				[
				mean( exp.( p[1] ./ T ) )
				for p in [
						simulate_gambles_sensitive(
							lp, staker, 
							N, 
							sens=s , 
							init_capital=cap, 
							succ_bias=false
						) 
						for s in 0.0:0.01:0.15
						]
				]
			for staker in [
				mean(calculate_stake.( calculate_estimates(N, tries, lp) ))
				for j in 1:n_samples
				]
		]...)
		)

		push!(lambdas, lp)
		push!(apays, adapt_payoffs)
	end

	
	sens_plot = plot(
		0.0:0.01:0.15,
		apays[1],
		lw=1,
		label="$(lambdas[1])",
		xlab= xlab ? "pessimism sensitivity" : "",
		ylab= ylab == first(λ) ? "mean payoff (ν = $cap)" : "",
	)

	if length(lambdas) > 1
		for i in 2:length(lambdas)
		plot!(
			0.0:0.01:0.15,
			apays[i],
			lw=1,
			label="$(lambdas[i])",
		)
		end
	end

	return sens_plot
end

# ╔═╡ a2999be0-7804-4b3a-9545-7e0549a4618d
# ╠═╡ disabled = true
#=╠═╡
sensiplots_cap_50 = [ 
	plot_sensitivity(
		[1.5, 2.0, 3.0, 4.0, 5.0], cap, 50, 10, 1000; 
		succ_bias=false, n_samples=1000,
		xlab=true, ylab=1.5
	)
	for cap in [0, 1, 2, 3]
]
  ╠═╡ =#

# ╔═╡ 87dc18e4-ec34-4c8a-abef-da0a631fcfaa
sensplot(sensiplots) = plot(
	plot(sensiplots[1], xlabel="", ylabel="average growth rate", ylabelfontsize=9, legend=false, title="ν = 0", titlefontsize=10),
	plot(sensiplots[2], xlabel="", legend=false, title="ν = 1", titlefontsize=10), 
	plot(sensiplots[3], legend=false, title="ν = 2", titlefontsize=10, ylabel="average growth rate", ylabelfontsize=9),
	plot(sensiplots[4], title="ν = 3", titlefontsize=10, legendfontsize=6, legendtitle="λ", legendtitlefontsize=6),
	layout=(2,2), link=:all, dpi=300
)

# ╔═╡ d459c06a-d858-45f6-a794-654b7ef3ba63
function payoffs_sensitivity(λ, ν, n_community, sens, tries; T=1000, n_samples=1000)
	
	paysims = []
	for i in 1:n_samples
		est_stake = calculate_stake.( calculate_estimates(n_community, tries, λ) ) |> mean
		paysim = simulate_gambles_sensitive(λ, est_stake, n_community, sens=sens, init_capital=ν, seasons=T)
		push!(paysims, exp.( paysim[1] ./ T )[1])
	end
	
	probdeath = length( filter(x -> x == 0, paysims) ) / n_samples
	
	return (paysims, (1 .- probdeath))

end

# ╔═╡ 8e921465-ee2a-4ee4-be0b-5f9b5666b95e
function normalize_measures(m1, m2)
	meds = median.(m1)
	centered_meds = meds #.- 1.0
	normed_meds = centered_meds .^ 1 #./ maximum(centered_meds)
	
	centered_m2 = m2 #.- minimum(m2)
	normed_m2 = centered_m2 #./ maximum(centered_m2)

	return(normed_meds, normed_m2)
end

# ╔═╡ a3dfb647-a4f8-4c23-a4a4-e452922c6eb3
function plot_sens(pays1, pays2, pays3)
	payoffers1 = [p[1] for p in pays1]
	probsurvs1 = [p[2] for p in pays1]
	
	normed_meds1, normed_probsurvs1 = normalize_measures(payoffers1, probsurvs1)

	payoffers2 = [p[1] for p in pays2]
	probsurvs2 = [p[2] for p in pays2]
	
	normed_meds2, normed_probsurvs2 = normalize_measures(payoffers2, probsurvs2)

	payoffers3 = [p[1] for p in pays3]
	probsurvs3 = [p[2] for p in pays3]
	
	normed_meds3, normed_probsurvs3 = normalize_measures(payoffers3, probsurvs3)
	
	axis1 = 0.0:0.005:0.2

	plot(
		axis1,
		normed_meds1 .* normed_probsurvs1,
		lw=2,
		legend=false,
		ylim=(0.6, 1.3),
		colors = palette(:Dark2_5)[1:3]'
	)
	
	plot!(
		axis1,
		normed_meds2 .* normed_probsurvs2,
		lw=2,
		legend=false
	)

	plot!(
		axis1,
		normed_meds3 .* normed_probsurvs3,
		lw=2,
		legend=false
	)

end

# ╔═╡ a424dce5-edb2-4018-a335-6d6e6b5d66e8
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(98765)
	sens_payoffs_c1 = []
	@threads for l in [2.0, 3.0, 5.0]
		sens_payoffs = [ 
			payoffs_sensitivity.(l, 1, n_com, 0.0:0.005:0.2, 5; T=1000, n_samples=10000)
			for n_com in [5, 10, 25]
		]
		push!(sens_payoffs_c1, (l, sens_payoffs))
	end
end
  ╠═╡ =#

# ╔═╡ ac8e2795-64a3-4dbe-9739-bb651cb67ccb
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(45678)
	sens_payoffs_c2 = []
	@threads for l in [2.0, 3.0, 5.0]
		sens_payoffs = [ 
			payoffs_sensitivity.(l, 2, n_com, 0.0:0.005:0.2, 5; T=1000, n_samples=10000)
			for n_com in [5, 10, 25]
		]
		push!(sens_payoffs_c2, (l, sens_payoffs))
	end
end
  ╠═╡ =#

# ╔═╡ 0e1c7c04-dd3f-4d1a-ba2f-0a607ec3bd97
# ╠═╡ disabled = true
#=╠═╡
begin
	Random.seed!(567432)
	sens_payoffs_c3 = []
	@threads for l in [2.0, 3.0, 5.0]
		sens_payoffs = [ 
			payoffs_sensitivity.(l, 3, n_com, 0.0:0.005:0.2, 5; T=1000, n_samples=10000)
			for n_com in [5, 10, 25]
		]
		push!(sens_payoffs_c3, (l, sens_payoffs))
	end
end
  ╠═╡ =#

# ╔═╡ 25f14d59-7fa2-4ed1-b511-0993b487bd26
#=╠═╡
plot(
	
	plot(
		plot_sens(sort(sens_payoffs_c1, by=first)[1][2]...),
		plot_sens(sort(sens_payoffs_c2, by=first)[1][2]...),
		plot_sens(sort(sens_payoffs_c3, by=first)[1][2]...),
		layout=(1,3), link=:all
	),
	plot(
		plot_sens(sort(sens_payoffs_c1, by=first)[2][2]...),
		plot_sens(sort(sens_payoffs_c2, by=first)[2][2]...),
		plot_sens(sort(sens_payoffs_c3, by=first)[2][2]...),
		layout=(1,3), link=:all
	),
	plot(
		plot_sens(sort(sens_payoffs_c1, by=first)[3][2]...),
		plot_sens(sort(sens_payoffs_c2, by=first)[3][2]...),
		plot_sens(sort(sens_payoffs_c3, by=first)[3][2]...),
		layout=(1,3), link=:all
	),
	layout=(3,1), link=:all
	
)
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═44c0fc58-d33c-11ee-2853-85a28adbbb59
# ╠═5e6a7ed3-06b5-4267-8226-8628fdbe32cc
# ╟─7fa16c52-e357-4d85-8460-3c979e09f629
# ╠═0c679c2a-5d68-4efe-ba74-86873f96594b
# ╠═6a33e703-427f-400c-aa85-eae291126584
# ╠═cd4f3d9d-e0f3-4b52-ab69-827d77637a56
# ╠═5407a245-750c-4a11-9e4a-49632cb41f88
# ╟─64abbea3-237f-4524-a7a5-2d1c2aec74ec
# ╠═a1051807-7284-48fb-8ade-d347f771be21
# ╟─026cc528-79c0-418c-84ed-d245960d136c
# ╠═2f776994-4f8c-4e0e-97a5-f551b9eb2fe4
# ╠═815ac5e7-91ec-49e3-84ae-d6cd2676411d
# ╟─299e9851-3bc3-46c4-a270-e2fd56e2c3d3
# ╠═2ce26997-43a2-4757-aaf4-b9863ac9ac5a
# ╟─50272356-0411-4728-86cc-313787f04aed
# ╠═6145ba3c-1f69-4f9e-b850-89bbee75c57f
# ╠═51df7295-a53d-44f9-837d-6337f45050b3
# ╟─78c9f173-1f49-49c1-bae9-ec40cdd4b492
# ╠═dd260c75-0e31-4a9d-9dc2-f8079e5c98e2
# ╟─63aa936d-7955-46b0-9ac1-c1d23268079b
# ╟─02b671bc-b9d9-45f5-a3ac-ef4d8a413459
# ╠═97af9b4a-cb54-4592-9fb1-496d0cb5d298
# ╠═b729db3a-6dcf-427b-b6ff-8b8ceadbc6e6
# ╟─92dcf70c-a58b-481b-b473-e7c0e5c180b3
# ╠═a0f6d5a5-f8be-4651-a9a7-d52b7ad8a6e0
# ╟─197c9fc9-8b1c-484e-8f0e-cffa08375296
# ╟─0ff38bba-587e-44b4-a65c-b724bf8c89ca
# ╟─28414f1a-83d5-4a96-b1b2-b43bd599f7ed
# ╟─5571c8e1-21bb-4b6e-80bf-4c811eae7aa3
# ╠═42a59a13-22e0-40ab-92a3-00db2b04249f
# ╟─f201eb67-f78b-47ae-ab06-9c612c5e2c0b
# ╟─68bfe7f9-d733-41b3-9f0e-cdca87bcc894
# ╠═ff971851-a7f3-4cda-a1e0-793a075c2893
# ╟─9cfc79d3-f9aa-488b-832a-e02bf711ca19
# ╟─d9f28433-f818-42c2-8eb0-aa844fa3eb1a
# ╟─8565e558-d7e3-443b-a56f-a82ffae661ce
# ╠═7781b6d1-8618-4ec4-bc0d-c5589c2c55b8
# ╠═0e9f8365-c8ec-4f45-a5be-9accb1a3e049
# ╟─2197f0d3-6aca-412f-86f8-c9f9ed54ad7e
# ╟─308a5b66-0dcf-4fee-a36c-68b68e6dcf4b
# ╟─4b24bff6-0a95-4763-b8da-76cf0884f543
# ╟─ec990ffb-727d-4419-9550-ad7f7a61bf85
# ╠═cb2f27e7-4001-4f8f-83a9-d26a166be9b5
# ╠═d621aecd-79eb-4b69-add8-59de52d47d05
# ╠═77ae48e8-36ac-4909-932a-aacfb33ff659
# ╠═e205678e-e9ae-46ed-ab94-7d827bdaabc0
# ╠═92563edb-058f-4247-b8e5-fd5c24fb4838
# ╠═4af45122-41e8-4ec4-8ac9-b56cd24996ee
# ╠═59937c6e-ae84-4264-b88c-2c787087b712
# ╠═3574fdad-f923-4755-8ad7-325634d8f6c3
# ╠═04924952-b873-4c5d-88c1-c11ad767d100
# ╠═9cb067fa-4ab2-41cf-9688-eff29b8271ef
# ╠═a2999be0-7804-4b3a-9545-7e0549a4618d
# ╠═87dc18e4-ec34-4c8a-abef-da0a631fcfaa
# ╠═d459c06a-d858-45f6-a794-654b7ef3ba63
# ╠═8e921465-ee2a-4ee4-be0b-5f9b5666b95e
# ╠═a3dfb647-a4f8-4c23-a4a4-e452922c6eb3
# ╠═a424dce5-edb2-4018-a335-6d6e6b5d66e8
# ╠═ac8e2795-64a3-4dbe-9739-bb651cb67ccb
# ╠═0e1c7c04-dd3f-4d1a-ba2f-0a607ec3bd97
# ╠═25f14d59-7fa2-4ed1-b511-0993b487bd26
