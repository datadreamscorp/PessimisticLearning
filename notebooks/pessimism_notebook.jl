### A Pluto.jl notebook ###
# v0.19.42

using Markdown
using InteractiveUtils

# ╔═╡ cfb0a045-a40d-4019-987f-e4dc1c92074e
begin
	using Pkg
	Pkg.activate("..")
	using Revise
	using StatsBase, Random, Distributions, Agents, Plots, CSV, DataFrames
	using PlutoUI
	using LaTeXStrings
	include("../src/pessimistic_learning_Numeric.jl")
	include("../src/pessimistic_learning_ABM.jl")

	md"""
	## The Development of Risk Attitudes and their Cultural Transmission
	
	#### Alejandro Pérez Velilla, Bret Beheim, Paul E. Smaldino.
	"""
end

# ╔═╡ ff915f4b-e120-43c6-90d3-d9e4b7659994
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

# ╔═╡ ad3d1e83-ee2d-484d-bd3e-359135e105ad
md"""
### 0. Motivation

This will eventually be an introduction. But really, what's the deal with risk attitudes? We know that risk aversion has an important role to play in evolutionary dynamics in stochastic environments and is closely related to the concept of geometric mean fitness. Lots of good models out there in Biology and Economics, which we will cite. But not everyone is equally risk averse, not within or between societies, or even within individual lifetimes. There is quite a lot of variation. Researchers in economics and psychology find diverse risk attitudes all the time, within/between populations and across the lifespan. So what's the deal?

Learning and cultural transmission is the deal. We propose risk preferences are mainly the causal product of: **a)** risk-free interactions with the environment during early developmental periods, **b)** peer influence borne from social interaction with same-cohort individuals (henceforth intra-generational learning), **c)** influence from previous-cohort individuals in processes of vertical and oblique transmission (henceforth inter-generational learning) and **d)** adjustments made throughout the lifetime in response to signals of excessive risk-taking. They are therefore dependent on the social learning strategies individuals are using, and can change throughout the lifetime. Can they be genetically-transmitted? Sure, who knows. But it is evident that at least a significant part of the variation comes from developmental trajectories and cultural inheritance. Our humble mission is to model these processes and come out with testable hypotheses.

The resulting model generates agents that start off optimistic (less risk averse than they should be) due to juvenile risk-free learning and peer influence, and then grow more risk averse during the lifetime due to exposure to adulthood risks (in this case, the can observe the bankruptcy of generational peers who are using similar risk attitudes). Agents can also pass on their acquired risk attitude to the next generation, and next-generation agents who learn from those who came before (conservative learners) will, on average, start off more risk averse than those who learn mostly from individual experience and generational peers (exploratory learners). Exploratory learners are better equipped to handle improving environments, while conservative learners will take longer to react to improving conditions, but hedge better for environments that become riskier. Generally speaking, conservative-leaning learning strategies do better than more exploratory learning strategies, although the difference in performance decreases as baseline wealth grows and environmental uncertainty decreases. That is, low exposure to risk makes exploratory learning strategies more viable.

We also examine what happens in the case of two groups within a mixed population and under different conditions of risk, in order to explore the effect of structural inequalities and economic stratification on a population's risk attitudes. Structural inequalities, such as pronounced differences in baseline wealth and/or experienced uncertainty by members of different groups (like socioeconomic strata) encourages the use of in-group biases in learning (parochialism) alongside conservative learning strategies. For advantaged groups (high wealth, low uncertainty), this in-group bias can be simply expressed as a payoff bias, as observed payoffs will be highly correlated with group membership. Disadvantaged groups (low wealth, high uncertainty) are locked out of payoff-biased learning, and must use explicit group markers instead. The combination of highly-conservative parochial learning is effective for disadvantaged groups so long as group inequality is in place, but can generate poverty traps for these groups if the inequalities get resolved and several generations must pass for learning strategies (and, consequently, risk aversion) to adjust. Allowing for more exploratory learning greatly increases the disadvantaged group's capacity to respond to improving environments, but is highly non-viable when stark inequalities are in place. In contrast, the adoption of conservative payoff-biased learning by the advantaged group makes them very capable to track improving environments while remaining conservative, but is ill-suited to deal with worsening conditions (like economic collapse, environmental catastrophe, or both) as agents end up learning from low risk aversion, high payoff individuals, even when their outcomes are not representative of what happens to most individuals exhibiting similar risk attitudes. Thus, the effects of structural inequalities may persist (or even self-reinforce) beyond the presence of group inequalities themselves through the cultural transmission of risk aversion, and can weaken groups' adaptive cultural responses to environmental change.
"""

# ╔═╡ 13636d74-91c4-4a5b-ba09-5ec872ea2058
md"""

### 1. Preliminaries

We start out with a single-asset risky investment dynamic (Kelly, 1956). In this model, a player starting with initial wealth $w_0$ chooses a stake $s \in (0,1)$, representing the fraction of their wealth they are betting away, and then proceeds to throw a biased coin every time period (for a total of $T$ periods), with every period resulting in a success with probability $u$ (also known as the rate of success) and a failure with probability $1-u$. If successful, the player increases their wealth (proportionally) by $1 + s$, and if not successful, their wealth decreases by $1 - s$. The mission of these agents is to **maximize the growth rate of their wealth**.

At its core, this is a simple model of risky capital investment: the more one places at risk, the more there is to lose and the more there is to win. The less one stakes, the less one can win, but the smaller potential losses will be. The behaviors represented by $s$ can be anything that involves such a risk-reward tradeoff, and we assume that $s$ itself represents an individual's global risk-taking parameter, which affects the strategies they choose in different situations across the lifetime (i.e. a forager's tendency to decide between exploiting known patches versus exploring promising but potentially-dangerous distant patches). Likewise, the currency in this model could in principle be any form of somatic investment that exhibits such a risk-reward trade-off, although we use wealth because it provides a natural way to interpret results and wealth (and/or its growth rate) is a plausible currency for cultural evolutionary processes.

We assume that individuals start with an initial amount of wealth, given by $w_0 = \text{Exp}[\log w_0] = \text{Exp}(\nu)$. This makes $\nu$ the initial log wealth of an agent, which we will use as the mean measure of initial wealth. We can write the payoff of a player at time T in the following manner:

$V_T(s) = w_0 \prod_{t=1}^T e^{g_t(s)} = \prod_{t=0}^T e^{g_t(s)}$

with $g_0 = \nu$ and $g_t (s) = \text{log}(1 + s)$ if successful or $g_t (s) = \text{log}(1 - s)$ otherwise. It can be shown that in the limit of large $T$, there is an optimal stake that maximizes the expected growth rate of wealth, and that can be easily calculated. Commonly known as the Kelly criterion, it can be written as

$s^* = \left\{
     \begin{array}{@{}l@{\thinspace}l}
       2u - 1  &: u \geq \frac{1}{2}\\
       0 &: u < \frac{1}{2}\\

     \end{array}
   \right.$

This means that a decision-maker facing such an investment scenario can know the optimal stake to adopt if they know the rate of success $u$. It can also be shown easily that under such a dynamic an equal number of wins and losses yields an overall loss in the agent's wealth. Losses carry more weight than wins in this binary game, so it is important to be playing under favorable odds. This can be seen reflected in the expression for $s^*$: a success rate of 1/2 yields an optimal Kelly bet of 0. We identify the stake agents choose as their **risk attitude**.
"""

# ╔═╡ f944d9d1-8804-4fd8-9611-a03d03b296ef
md"""
**Measure of success: the mean time-average growth rate of wealth.** When an agent $i$ plays the game with any particular stake, we summarize their experience using their average growth rate. This is given by

$W^i_T (s | \nu, \lambda) = \text{Exp}\left[ \frac{1}{T+1} \sum_{t=0}^{T} g^i_t (s|\nu, \lambda)  \right]$

where the notation is meant to indicate that the growth rate of a stake is conditional on the agent's initial wealth and the uncertainty in their environment. In this model, growth rate is not independent of initial wealth, because starting off with a higher degree of wealth decreases the probabilities of falling into ruin, as we will see soon. Because luck plays a part in this growth process for every individual trajectory, we simulate $N$ agents for every condition, and take the average growth rate as the measure of success of a particular strategy:

$\bar{V}(s | \nu, \lambda) = \frac{1}{N} \sum_{i=1}^N W^i_T (s|\nu, \lambda)$

This gives us the **average growth rate** of agents using a stake $s$, with initial wealth $\nu$ in environment with uncertainty given by $\lambda$.
"""

# ╔═╡ 600fc17f-7db8-4f18-b319-acdf28bd7644
md"""
**Assumption: agents are boundedly rational**. *We assume that agents, at baseline, start out with CRRA utility functions with η → 1. This is equivalent to logarithmic utility, and amounts to assuming that evolution has encoded the solution to the one asset model, under known probabilities, in agents' psychologies (for example, having a logarithmic utility in fertility outcomes is mathematically equivalent to geometric mean fitness maximization). We believe this is a sensible assumption, as it leads to evolutionarily-plausible decision-makers that will refuse to bet anything if they know, with certainty, that the odds are not in their favor. However, we do not assume that the shape of utility functions is constant throughout life: changes in agents' preferred risk-taking borne from experiences in life will necessarily lead to changes in their preferred stake, which itself implies a change in their utility function, yielding some other function in the CRRA family. Therefore, we assume agents are always trying to maximize the growth rate of their wealth with respect to the information they have about their chances of success, and they start off with a modest degree of adaptive risk aversion. Heterogeneity in risk preferences must then be a result of the constraints agents face when gathering information. If all agents have perfect information about the environments thay are performing in, they would all share the same risk preference: the one that maximizes the objective growth rate of their wealth, and that would result in the agent acting as a Kelly maximizer. This means agents are boundedly-rational due to information constraints.*
"""

# ╔═╡ de81b0f9-d5e9-4443-a64a-a104cdf8f442
md"""
**Absorbing boundaries:** we assume there is an absorbing boundary for wealth $V_{\text{Boundary}}$ such that if wealth ever falls under this value during the $T$ time periods of an agent's lifetime, it is set to zero. We say that such an agent is **ruined**. We use $V_{\text{Boundary}} = 1$ as the boundary, as it is the payoff achieved by having a zero growth rate. However, the placement of the absorbing barrier is an arbitrary manner. What is important for agents is not the (absolute) location of the barrier, but rather their (relative) distance from it.
"""

# ╔═╡ 7ee8dab2-d381-4064-b4cc-4f69cbc127b6
md"""
**Uncertain environments:** the environments agents face do not offer certain success rates, they are sampled every time period from an environmental distribution $\mathbf{U}$. This means that for every time period (in total $T$) that agents play their string of gambles, they get a different $u$ which they sample from the environment. Samples are independent across agents in a population, and environments can be more or less stable. More stable environments offer better success rates more reliably, while less stable environments can provide success rates that can easily go from very good one day to very bad in a subsequent time period. We parameterize this by assuming $\mathbf{U} \sim \text{Power}(\lambda)$. Then $\lambda$ describes the shape of the environmental distribution, going from a uniform distribution at $\lambda = 1$, and converging to a certain success rate of 1 as $\lambda \rightarrow +\infty$. The figure below shows it.
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

# ╔═╡ 59cd6ed6-53cc-45f1-9a16-b62a72a00243
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
	
	savefig(varplot, "../images/sup1_var.pdf")
	varplot
end

# ╔═╡ f052e274-ccb2-4d54-b6a2-1975a5809124
md"""
It turns out that in this uncertain scenario, and assuming that there are no effects from absorbing boundaries, knowing an environment's $\lambda$ allows an agent to choose an optimal stake: it is simply $s^*$ evaluated at $\mathbb{E}(\mathbf{U}) = \frac{\lambda}{1 + \lambda}$. The rest of this model is about the strategies agents use to infer $\lambda$ under constraints. Where can we find information about our environments, what kind of strategies we use to procure it, and what kind of biases do we encounter on the way? What sort of risk attitudes and benefits do different learning strategies lead to? We say an agent is **optimistic** if they are above the Kelly-optimal stake for their environment, **pessimistic** if they are under it (corresponding to a fractional Kelly strategy), and **optimal** if they bet with the Kelly-optimal stake. In the next figure, we can see $s^*$, as black dots, for different values of $\lambda$.
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
		legendtitlefontsize=12,
		lw = 2,
		xlab = "stake "*L"(s)",
		ylab = L"G (s)"*" (expected growth rate)",
		alpha = 0.5,
		ylim=(0.9, 1.35),
		dpi=300,
		palette=:Dark2_5
	)
	
	for l in λ[2:length(λ)]
		plot!(
		xaxis2,
		g.(xaxis2, l=l),
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

	hline!([1.0], color="black", alpha=0.2, label="")

	savefig(payplot, "../images/fig2_kelly.pdf")

	payplot
end

# ╔═╡ 0d2af086-3c43-46cb-86a9-d4ac583ab688
md"""
When there are absorbing boundaries, the optimal Kelly stake is generally lower than for the case with no absorbing boundaries. But having high initial wealth leads to optimal stakes that approach the no-boundary dynamics. This is intuitive: if an agent starts off far away from the absorbing boundary, they are less likely from the get-go to ever feel its effects. We can think of the no-boundary dynamics as the limit of the boundary dynamics when initial wealth goes to infinity. So, in this game, initial wealth matters. It allows agents to take on more risk, and thus to get higher lifetime growth. The next figure is of numerically-simulated growth rates, hence the noise. You can see that $s^*$ decreases as we go to lower initial wealth $\nu$.
"""

# ╔═╡ eb3d9026-c2c4-4882-9338-14c9e2852cce
begin
	Random.seed!(123456)
	splots_barrier = [
		simplots(
			1.0:1.0:5.0, 0:0.01:1, 
			true, 2000, 
			aleph, aleph == 0.65 ? true : false, 
			aleph == 0.95 ? true : false
		) 
		for aleph in [0.65, 0.8, 0.95]
	]

	splot = plot(
		splots_barrier...,
		layout=(3,1),
		#plot_title="mean payoffs under absorbing barriers",
		link=:all,
		ylim=(0.9, 1.3),
		dpi=300
	)

	savefig(splot, "../images/fig3_kelly_barrier.pdf")

	splots_barrier = nothing
	GC.gc()
	
	splot
end

# ╔═╡ 77d56680-5a71-40cb-98ae-eeb91a8ff2bc
md"""
It is evident that the risk of ruin (that is, the risk of hitting the absorbing boundary at any point of the trajectory) of an agent is largely determined by $\lambda$ and $\nu$. It is also imaginable that the values of these parameters will be correlated in real-life scenarios, even though here we model them as independent. For convenience, we call scenarios of high uncertainty and low initial wealth **high risk**, while scenarios of low uncertainty and high initial wealth are **low risk**. Scenarios in between we call **intermediate risk**. This is only to simplify verbal reasoning, since every conclusion will be backed by precise values for these parameters.
"""

# ╔═╡ 3a2082c4-ef09-4b53-af1b-411792729560
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

	savefig(probruin_plot, "../images/sup2_probruin.pdf")

	probruin_plot
end

# ╔═╡ 04452287-07ab-45b5-9b8f-f175ddc43d61
md"""
#### 2. The developmental trajectory of risk attitudes (in the absence of cultural inheritance) 
"""

# ╔═╡ 96cdc247-40df-4538-be8a-7465ca638bf2
md"""
How do risk attitudes develop in the absence of cultural transmission? Here we construct a process meant to represent the acquisition of risk attitudes out of juvenile risk-free exploration of the environment (individual learning), information-sharing between generational peers (horizontal social learning) and socio-environmental feedback from actual risky performance in the environment during adulthood. Agents start off optimistic due to risk-free learning and peer social reinforcement, and become more pessimistic during their lifetime as they observe similar peers get ruined.
"""

# ╔═╡ 9eb10b7b-b282-4b10-831a-ea00fb722538
md"""
##### Exploration phase: acquiring risk attitudes using intra-generational learning

We start with a population of $N$ individuals in a young generational cohort. Maybe they are the children in a population of migrants who have entered mysterious new lands full of opportunity and, potentially, danger. Since the previous generation is also new to this environment, there is no use in learning the risk atittudes of elders. Everyone must interact with the environment to learn from it.

In this juvenile phase, agents can sample the environment without actually staking anything. This risk-free sampling is meant to represent individual learning in early development: juveniles get to "play" in risk-free settings and use that information to perform in settings where stakes actually matter. The objective: estimate $\mathbb{E}(\mathbf{U})$. We divide the **juvenile learning phase** into three key processes:

- **Individual learning.** Agents draw $\tau$ samples from the environment. Every time period of their juvenile stage, agents throw a biased coin, where the probability of success is a sample from $\mathbf{U}$. The total number of samples $\tau$ represents the length of juvenile periods, and it is imaginable that longer periods will be more costly, so that agents who start off more wealthy can also have access to longer bouts of individual learning, although we do not explicitly model costs. After drawing the samples, agents calculate the sample mean and use it to get their estimate of $\mathbb{E}(\mathbb{U})$, which we call $\hat{u}$ and that they express as an **estimated stake**

$\hat{s} = \left\{
     \begin{array}{@{}l@{\thinspace}l}
       2\hat{u} - 1  &: \hat{u} \geq \frac{1}{2}\\
       0 &: \hat{u} < \frac{1}{2}\\

     \end{array}
   \right.$

Thus, we assume that an agent's overall individual learning experience is not publicly expressed as their estimate of $\mathbb{E}(\mathbb{U})$ (namely $\hat{u}$), which is unobservable, but by the optimal Kelly stake implied by that estimate, $\hat{s}$.

- **Peer influence, or intra-generational learning**: agents can observe $n$ other agents within their cohort (henceforth their peer learning set) after individual learning is finished, and have their estimates influenced by what they observe. We assume such peer influence takes the form of an averaging rule: agent $i$'s estimate after horizontal learning will be given by $\bar{s}_i = (1 - \beta) \hat{s}_i + \beta \frac{1}{n} \sum_{j=1}^{n} \hat{s}_j$, where $j$ indexes the peers in $i$'s peer learning set, and $\beta$ gives the overall weight of peer influence on $i$'s estimated stake.

- **Elder influence, or inter-generational learning**: when they are not the first generation in a new environment, agents also get to use information from the previous generation. A basic process the assumes equal weights from all learning targets is blending inheritance, where the focal agent averages the stakes of $n$ agents from the previous generation (henceforth the elder learning set). It is what we assumed for horizontal learning as well. Agents ignore information from ruined individuals, a form of cultural viability selection. This amounts to asserting that humans do not generally learn risky behaviors from those who have evidently lost everything. The resulting aggregated risk attitude $s^{\text{old}}$, representing the elder learning set's knowledge, is then integrated with the estimate obtained after individual and social learning like so:

$s_i = (1 - \alpha) \bar{s}_i + \alpha s^{\text{old}}_i$

and $\alpha \in [0,1]$ tells us how much agents rely on intra- versus inter-generational learning. We also assume that when agents learn from the previous generation, they learn from individuals who also relied on inter-generational learning in the same way as them (same value for $\alpha$). Thus, we will call intra-generational learning strategies **exploratory**, while the inter-generational learning strategies we just described will be called **conservative**. This means $\alpha$ measures agents' reliance on conservative learning strategies (vs. explorative ones), and any strategy that is not purely exploratory ($\alpha = 0$) or purely conservative ($\alpha = 1$) is known as a **mixed** strategy.

As we established above, the first generation cannot rely on conservative learning, so they do fully explorative learning to start. We start then by examining fully exploratory learning first and describing its characteristics.

"""

# ╔═╡ 3e4b23c8-7a02-45f3-9753-1b2dbaba0fa9
md"""

###### Peer influence leads to optimistic estimates due to the way behavior transforms information

The fact that intra-generational learning works does not mean it leads to good estimates of $s^*$. In fact, on average it leads to overly optimistic stakes that overshoot $s^*$. In this model, this arises from the mathematical structure of learning: since social learners average over their peers' observed behavior (their estimated stakes) instead of their unobservable direct experience (their estimate of $\mathbb{E}(\mathbf{U})$), they are averaging over a convex function of this experience, leading to optimistic stakes (Jensen's inequality). Thus, our model carries a hidden assumption: the stakes of peers who had favorable experiences (and thus arrived at estimates of $\mathbb{E}(\mathbf{U}) > \frac{1}{2}$) have more weight in social learning than those of peers who had disfavorable experiences (and thus arrived at estimates of $\mathbb{E}(\mathbf{U}) \leq \frac{1}{2}$).

"""

# ╔═╡ ea485e6d-2cfa-48b7-b02d-d86578ca0d37
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

	savefig(jenplot, "../images/fig4_jensen.pdf")

	jenplot
end

# ╔═╡ e7c9aed3-b053-4052-9337-be5b9a9b8880
md"""
The above plot shows this particular case of biased inference, in the high wealth limit (no absorbing boundaries). Agents observe a large number of peers and average their estimated stakes, leading to optimistically-biased estimates of $s^*$. The measure of bias is given by the Jensen gap, which is just the averaged stake minus the optimal stake. The analytical results in the topmost plot show the optimal stake ($s^* = 2 \mathbb{E}(\mathbf{U}) - 1$, solid line) and the expected value of $2u - 1$ (dashed line), where $u$ is a draw of the environmental distribution $\mathbf{U}$. This assumes agents draw a $u$, use it to estimate the optimal stake, and then get observed by their peers. We also simulate the process where, instead of sampling a particular $u$, agents sample wins and losses for $\tau$ time periods 9as a string of 1s and 0s) and average it. The distributional assumptions are different, but the results are fairly similar. The bottom plot shows the Jensen gap for the first process (solid black) and for the second process under different juvenile period lengths $\tau$. 
"""

# ╔═╡ 188478f3-c4a4-49d2-a5d9-e018486b3842
md"""
While all of this might sound fairly arcane, this is actually a desirable property of our model. Empirical research on risk attitudes suggests that people start out life less wary of risk, and grow more risk averse as they age. A proposed explanation of this is that during early stages, humans tend to develop in shielded environments, as parents and guardians provide controlled conditions for individuals to grow and learn about the world before being sent out to face it. If there is no inter-generational learning involved, this leaves juvenile humans with a worldview that ignores important risks (such as the effects of absorbing boundaries, which agents never get to observe during the juvenile stage), and this optimism is reinforced by the sharing of information between juveniles. Our model can thus capture the effects of such a process. It predicts individuals will come out of juvenile stages more optimistic than they should, as they are ignorant of the effects of absorbing barriers and they have given more weight to peers who might have been luckier in their sampling. If agents are in a fairly stable environment (high $\lambda$) this does not pose as much of a problem, but in very uncertain environments it can become a big problem. Again, having longer individual learning periods can help.
"""

# ╔═╡ a074cc0f-9484-4771-8078-2581aaa17f78
md"""
###### Reliance on peer influence is favored over pure individual learning when environments are not too uncertain or when juvenile periods are long

We simulate 10,000 agents going through the juvenile learning process and then playing uncertain coin tosses for 1000 time periods. We take the average of their growth rates for different values of $\nu$, $\lambda$, $n$, $\tau$ and $\beta$. The value of $\alpha$ does not matter for the first generation, since there is no previous cohort. We can appreciate the following:

- If environments are too uncertain and juvenile periods are short, individual learning is favored, because the additional optimistic bias introduced by peer influence becomes higher as environments become more uncertain. As environments become more favorable and/or juvenile periods grow longer, this relationship flips and heavy reliance on peer influence is favored.

- Results are robust to changes in size of peer learning sets. But the larger the peer learning sets ($n$ larger), the better social learners do on average. There is incentive to learn from ever larger groups of peers, assuming that there is no per-capita cost of social learning.

- Higher risk situations (lower $\nu$, higher $\lambda$) reduce the gap between learning styles if peer learning sets are large enough. Because of the bias introduced by peer influence, its strength as a learning strategy is reduced when environments are very uncertain.
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
			layout=(1,3)
		),
		layout=(2,1),
		link=:all,
		margins=2Plots.mm
	)
	
	savefig(indv_plot, "../images/fig5_indv.pdf")

	indv_plot
end

# ╔═╡ 1fc1da0c-4ccd-442d-903f-c26b43a3f94c
md"""
##### Gameplay phase: adjusting risk attitudes in response to observed ruin

We have seen how agents can start life out optimistic. Now we explore how they can grow more risk averse throughout the lifetime. Once agents have passed the juvenile stage, they are thrust into real life and forced to fend for themselves with only their estimated stakes in tow. They play $T$ gambles, while keeping an eye on their peer learning set. If at any point in the gambling process they hit the absorbing boundary (if their cumulative log-growth rate becomes less than zero), then they become ruined, growth rate is set to zero, and they do not play anymore.

We say agents can have a **sensitivity** $\delta \in [0, + \infty)$ such that during gameplay they focus on the agents of their peer learning set who are within $\pm \delta$ of their own risk attitude and, if they happen to observe any of them hit the absorbing boundary during the gambling process, they adjust their risk attitude according to the following recursion

$s_{i, r+1} = s_{i, r} - R_{r+1}$

where $r$ indexes an observed peer ruin event, $s_{i, 0}$ is the risk attitude just before gameplay starts, and

$R_r \sim \text{Exponential}(\delta)$

is a draw from an Exponential distribution with expected value $\delta$. If $s_{i, r} - R_{r+1} < 0$, then $s_{i, r+1} = 0$.

This process is meant to capture some assumptions about how sensitivity mediates agents' responses to their peers' ruin. First, agents who are more sensitive will be affected by the ruin of a wider range of peers. Second, agents' actual response will not be constant, and can instead vary from no response at all ($R_r = 0$), to becoming maximally risk averse ($s_i = 0$). Third, less sensitive agents will more often have smaller responses, while more sensitive agents have a higher probability of responding too strongly, and becoming too risk averse or even maximally risk averse (akin to getting traumatized after witnessing an esteemed peer go bankrupt). For simplicity, we use a single parameter to encode these behaviors, but in principle the similarity tolerance and the average size of responses to ruin can be independent of each other.
"""

# ╔═╡ fc9b6c6b-6832-4842-a246-3492ce9e61e2
md"""
###### Non-zero sensitivity is always favored, even if responses to peer ruin can lead to pessimistic overshoot
"""

# ╔═╡ d008b259-7a84-4f7e-99ed-e4b283775591
begin
	sens_plot = plot([ 
		plot( plot_sensitivity.([5, 25], i, 1000, 25; seed=1)..., layout=(2,1) )
		for i in [0.65, 0.8, 0.95]
		]...,
		layout=(1,3)
	)
	savefig(sens_plot, "../images/fig6_sensitivity.pdf")
	sens_plot
end

# ╔═╡ babba0ec-d157-443f-b3da-789d248a0f71
md"""
###### High risk scenarios lead to higher risk aversion by end of life

We set $\delta = 0.1$ for the rest of the analyses that remain.
"""

# ╔═╡ 71768709-5e37-43fc-91d9-cd47e659e1d8
begin
	dev_plot = plot(
		plot( development_plot(2.0, xlabel="", t=true, legend=false), xaxis=false, xlim=(0,1)),
		plot( development_plot(3.0, xlabel="", legend=false), xaxis=false, xlim=(0,1)),
		development_plot(6.0, legend=:topleft),
		layout=(3,1), link=:all, legendfontsize=6, ylim=(0, 1200), size=(800, 800), margins=1Plots.mm
	)
	savefig(dev_plot, "../images/fig7_development.pdf")
	dev_plot
end

# ╔═╡ b97683e7-9187-49d2-baaa-301327bf65ab
md"""
### 3. The cultural inheritance of risk attitudes

Exploratory learning during juvenile periods is a source of optimistic variation, while adjustment during gameplay can provide some pessimistic vaiation due to potentially-large responses to observed ruin. This leads to population risk-attitude distributions at the end of a generation that can range from wide and highly-pessimistic in high-risk scenarios, to only slightly different from juvenile distributions in low-risk scenarios. Now we examine what happens when this variation can be inherited by the next generation (and any subsequent generation).

In the following, we simulate large populations of agents (again, $N = 10,000$) where agents go through the juvenile learning process and the gameplay phase every generation. If they are not the first generation, then they also use inter-generational (conservative) learning during their juvenile learning process, with the weight of this learning on the attitude they adopt by the end of their juvenile period given by $\alpha$. In the following figures we use $n = 25$, $\tau = 25$, $\delta = 0.1$ and $\beta = 1$. We take the average growth rate at the fifth generation.
"""

# ╔═╡ 256f53ce-c5a0-413d-9de9-3fa15e51289e
md"""
###### Conservative learning leads to the inheritance of higher risk aversion

Conservatively-biased strategies ($\alpha > 0.5$) can still retain a considerable degree of (mostly pessimistic) variation. Full conservatism reduces variation considerably. Plotting medians along with 10%- and 90%-quantiles for the variation. 

Important to note that environmental risk is the main driver of risk attitude. In order of importance, the drivers of differences in risk attitudes are: 1) environmental uncertainty, 2) baseline wealth, and 3) conservativeness in learning.
"""

# ╔═╡ aebf5e0d-7520-4f6e-9dff-493c4e7a5bb0
begin
	attitude_plot = plot(
		plot( plot_conservative_attitudes(1.5), ylabel="median stake "*L"(s)", xlabel="" ),
		plot( plot_conservative_attitudes(3.0), legend=false, xlabelfontsize=12 ),
		plot( plot_conservative_attitudes(6.0), legend=false, xlabel="" ),
		layout=(1,3), link=:all, size=(700, 300), margins=4Plots.mm
	)
	savefig(attitude_plot, "../images/fig8_attitudes.pdf")
	attitude_plot
end

# ╔═╡ 009223ce-5095-491d-89b8-4e372e025446
md"""
###### High-risk scenarios demand more conservative learning

Featuring the power of cumulative cultural evolution. Although the growth rate difference decreases the lower the population's risk (the higher the wealth and the lower the uncertainty).
"""

# ╔═╡ 6e37d178-4026-4dce-94bf-8b8eb3f1d6fb
begin
	conservative_payoff_plot = plot(
		plot( plot_conservative_payoffs(1.5), ylabel=L"\bar{V}", xlabel="" ),
		plot( plot_conservative_payoffs(3.0), legend=false, xlabelfontsize=12 ),
		plot( plot_conservative_payoffs(6.0), legend=false, xlabel="" ),
		layout=(1,3), link=:all, size=(700, 300), margins=4Plots.mm
	)
	savefig(conservative_payoff_plot, "../images/fig9_payoffs.pdf")
	conservative_payoff_plot
end

# ╔═╡ 127fbf7b-4b8e-49e1-8239-b9bb6698d7ce
md"""
###### Conservative learners hedge for when environments worsen, explorative learners better track improving environments

Here we simulate population trajectories for ten generations, starting with a fully-exploratory first generation at time 0, and inducing a change in environmental uncertainty at time 5. In both panels, environments go from $\lambda = 2.5$ to $\lambda = 6$ on the left column, and from $\lambda = 6$ to $\lambda = 2.5$ in the right column. We show trajectories for $\alpha \in \{0, 0.5, 1\}$, equivalent to fully exploratory, fully mixed and fully conservative learning respectively.

The effect mediated by initial wealth. Low wealth populations have a higher incentive to remain conservative, unless environments are only improving. High wealth populations can be more explorative after environments improve, and have less of an incentive to turn or remain conservative after an environments turn more uncertain.
"""

# ╔═╡ 7300889d-373f-424d-9cdc-f28856cb0f23
begin
	abmplot = plot(
		run_ABM_plot(N=10000, aleph=0.65, t=25),
		run_ABM_plot(N=10000, aleph=0.95, t=25, legend=false),
		layout=(1,2), size=(800,400), 
		margins=3Plots.mm, xtickfontsize=5, xrotation=90
	)
	
	savefig(abmplot, "../images/fig10_time.pdf")

	GC.gc()
	
	abmplot
end

# ╔═╡ 189cca21-b09b-4b56-ab6c-82f3120cd31f
begin
	abmplot_sup = plot(
		run_ABM_plot(aleph=0.65, t=2),
		run_ABM_plot(aleph=0.95, t=2, legend=false),
		layout=(1,2), size=(800,400), 
		margins=3Plots.mm, xtickfontsize=5, xrotation=90
	)
	
	savefig(abmplot_sup, "../images/sup3_envchange.pdf")

	GC.gc()
	
	abmplot_sup
end

# ╔═╡ 019af979-9993-462c-831f-41e50c141ce5
md"""
###### In environments that oscillate between high and low uncertainty, conservative learning is generally favored

Here we take the geometric mean of the average growth rates of agent populations across generations, for a total of 20 generations. Populations start at comfortable environments with $\lambda_h = 6$, which oscillate between this initial value and a lower $\lambda_l$ value every $K$ time periods. we use $K = 2$ for the plot. The difference $\lambda_h - \lambda_l$ is the peak-to-peak amplitude of the oscillatory wave representing environmental change between generations.

As we can see, conservative learning is still favored, although more exploratory learning can do better when oscillations are wide, environments change frequently and learners have high baseline wealth. This is because these wealthy learners have more padding during high-uncertainty periods and can adapt faster to low-uncertainty periods once they start. Again, exploratory learning is a luxury of the wealthy.
"""

# ╔═╡ 01a70f2d-7c1d-404f-8501-82b7d4a44d10
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
			plot( plot_resilience(4.0, 10), ylabel="", title="", titlefontsize=10, xlabelfontsize=12 ),
			plot( plot_resilience(5.0, 10), legend=false, ylabel="", xlabel="", title="", titlefontsize=10 ),
			layout=(1,3), link=:all, size=(700, 300), margins=2Plots.mm, legendfontsize=4, legendtitlefontsize=5
		),
		layout=(3,1), size=(600, 500), xtickfontsize=4, ytickfontsize=6
	)
	savefig(resilience_plot, "../images/fig11_resilience.pdf")
	resilience_plot
end

# ╔═╡ 502806bf-06c1-4c3a-aba9-719e4cba2990
md"""
There is an assymmetry in this problem that is not present in the Rogers model and in Deffner and McElreath's treatment of age-structured social learning. While more frequent environmental change does weaken selection for conservative learning with respect to exploratory learning, the fact is that environmental change here does not carry the same effects all the time, since there are environmental states that are clearly more favorable than others. Oscillating between a good and bad state (low uncertainty versus higher uncertainty) means that conservative learners that adapt to the bad state will not perform optimally in the good state, but will also not perform as badly as when exploratory learners who do not count with a safety net are quick to adapt to the good state and then have to face the bad state. Older age classes, by virtue of being more pessimistic than younger age classes, always hold behavior that is likely to perform well enough, in the sense that it will likely avoid ruin.
"""

# ╔═╡ e8ad4bb4-ee44-4df4-9aed-db6d764a56b9
md"""
### 4. Structural inequalities, parochialism, poverty traps and payoff bias
"""

# ╔═╡ b2481d42-e3c5-471e-9277-59fd2b53aec1
md"""

###### Payoff bias retains optimistic variation, even when highly conservative

We also examine what happens if instead of a blending rule for cultural inheritance agents use a payoff-biased rule: selecting the agent with the highest growth rate from their elder learning set and imitating their risk attitude. Payoff bias is beloved by cultural evolutionists and often used as a basic assumption for modeling cultural change. Here we show why this might not be the best practice.

Payoff bias does not seem to ever perform better than the blending rule, but it does  have interesting properties. First, it is much better at retaining optimistic variation in risk attitudes, even when highly conservative. This is due to payoff-biased learners adopting risk attitudes from the right tail of the attitude distribution, since some individuals using excessively-high stakes will have been lucky enough to not go bankrupt, and instead will have attained very high payoffs. This is the main reason the payoffs of payoff-biased individuals are hampered with respect to agents using blending inheritance. Second, conservative payoff bias, like an inverted mirror of its blending counterpart, does better than more exploratory learning at adapting to improving environments, but much worse at hedging for worsening environments.

"""

# ╔═╡ da7818d8-3692-4896-9c85-0cb439da6626
md"""

If payoff bias does not perform better, why would someone ever use it? This comes at odds with how freely payoff-biased copying is used in the modeling literature, and calls for careful consideration when attempting to model the evolution of traits that exhibit risk-reward tradeoffs. Payoff-biased learning can be an intuitive assumption, but might not be a justifiable one.

There is way in which payoff bias can regain relevance in the context of risk attitudes, and that is through group inequality. When different groups in a mixed society are facing unequal scenarios, groups in lowest-risk conditions that preferentially learn from high-payoff individuals are more likely to be learning from in-group members.

To study this, we go beyond the single-group case, and see what happens when the population is subdivided into groups, each facing different scenarios of risk (as happens in a socio-economically stratified society). We model a two-group case to illustrate the effects of socio-economic hierarchy on learning strategies and, in consequence, risk attitudes.

"""

# ╔═╡ da6dbb32-fcee-43d4-b38e-a360de2ea54f
md"""
###### In a mixed-group setting, advantaged and disadvantaged groups need different learning strategies
"""

# ╔═╡ 6ea0d83d-b3df-443a-9dd0-baf314ded951
begin
	inequality_plot = plot(
		plot(
			plot(stab_plot_low(mshift=6.0, n=25, eshift=2, freq=0.25), leg=:bottomleft, title=L"f = 0.75"),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=2, freq=0.5, leg=false), ylab="", title=L"f = 0.5"),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=2, freq=0.75, leg=false), ylab="", title=L"f = 0.25"),
			layout=(1,3), xlab=""
		),
		plot(
			stab_plot_low(mshift=6.0, n=25, eshift=5, freq=0.25, leg=false),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=5, freq=0.5, leg=false), ylabel=""),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=5, freq=0.75, leg=false), ylabel=""),
			layout=(1,3),
			xlabel=""
		),
		plot(
			plot(stab_plot_low(mshift=6.0, n=25, eshift=10, freq=0.25, leg=false), xlab=""),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=10, freq=0.5, leg=false), ylabel=""),
			plot(stab_plot_low(mshift=6.0, n=25, eshift=10, freq=0.75, leg=false), ylabel="", xlab=""),
			layout=(1,3)
		),
		layout=(3,1), link=:all, size=(800, 600), margins=5Plots.mm
	)

	savefig(inequality_plot, "../images/fig12_inequality.pdf")

	inequality_plot
end

# ╔═╡ 5f2c213c-2d15-47ef-8b08-b86b3d1101c2
md"""
What works best for a disadvantaged group (solid line, $\lambda = 2$) is not what works best for an advantaged group (dashed line, $\lambda = 6$). 

In situations like these, learners can end up learning from individuals from the out-group. This is dangerous for individuals in the disadvantaged group, as they can end up acquiring risk attitudes from individuals who developed in much more favorable environments. For the advantaged group, it is best to use strategies that retain exploratory potential, such as exploratory learning or conservative payoff bias. We have seen that payoff bias retains optimistic variation, but in the mixed-group setting it is also a form of parochial learning for advantaged groups. But it is clear too that using the "wrong" learning strategy means very different things for individuals of each group. While anything but a very conservative learning style leaves disadvantaged agents with a significant risk of ruin, advantaged agents at works grow their wealth somewhat less if they are conservative and non-payoff-biased.

Payoff bias is also a form of in-group-biased (parochial) learning for advantaged groups, and parochial social learning is good for unequal groups in the mixed group scenario, because it directs learning towards individuals in similar coonditions while avoiding those who are under different conditions.
"""

# ╔═╡ 8412ef2a-8429-4398-ad2b-6461c2071f57
md"""
In the sense that parochialism relies on observable markers of identity, the effectiveness of parochial strategies will depend on how reliable those markers are. In an economically-stratified society, markers can emerge from observable behaviors associated with wealth, and individuals might find benefits in adopting them even if they do not come from wealthy backgrounds, as they serve as signals of cultural capital (Bourdieu) that can earn them opportunities in the social environment. Markers indicating a high position in the socioeconomic hierarchy, like the ones we have implemented here, have to be highly-associated with payoffs in order to remain reliable (the more associated with wealth, the costlier the signal). The bottom line is that in an economically-stratified society, parochialism with respect to class markers will be functionally equivalent to payoff bias, but only for the wealthy classes in low-risk environments. Those not at the top the hierarchy will have to choose arbitrary (but more accessible) markers of group identity, which after the fact might become associated with their wealth class. The accessibility of these markers might also make them less reliable in general, as wealthy individuals might also be able to use them, while the markers of wealth remain less accessible to anyone not in the wealthiest class.
"""

# ╔═╡ 8d568cf3-fb47-4f77-b603-ea99271f6bfe
md"""
###### Structural inequalities can lead to parochial poverty traps and fragile responses to disaster

The effects of structural inequalities might persist even after the inequalities themselves disappear. Here we look at a group-structured population, where both groups use different strategies. We only plot parochial strategies (payoff bias and explicit parochialism) for the advantaged group. Dashed lines represent explicitly-parochial strategies for both groups. We look at two scenarios: the **cultural poverty trap** and the **fragile response to disaster**. They are meant to illustrate how the long-run effects of persistent structural inequalities might make populations less resilient to change.
"""

# ╔═╡ 9ff6bb98-6e7b-4ebd-9a35-9b1d15144658
md"""
In the **cultural poverty trap** case, the group facing high-risk conditions is strongly incentivized to clamp up by using highly-conservative, in-group biased (parochial) social learning. Less restrictive strategies run the risk of having agents adopt risk attitudes that are too optimistic for their conditions, either from exploratory in-group agents or from out-group agents facing much better environments (and thus having developed less risk aversion). However, if conditions change and groups end up facing equally low-risk conditions, what was the best strategy becomes the worst perfomer, as conservative parochialism is the least sensitive to improving conditions. While this provides an incentive for agents in the (formerly) high-risk group to adopt more exploratory, less in-groupy learning strategies (and become less risk averse in the process), the time required for such cultural evolutionary change might see group differences in performance remain for several generations before groups close their risk attitude gap. Meanwhile, the compunding generatinal effects of differential performance (which we do not model here) might contribute to persistent wealth inequality, even after the risk attitude gap has been fully closed. In individuals as in populations, an initial advantage can remain an advantage forever.
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
		label="0.95", legendtitle=L"α",
		xlab="time (generations)",
		ylab=L"\bar{V}"
	)
	plot!(
		0:20, mdat2.Vbar_g0,
		ls=:dash, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="0.75"
	)
	plot!(
		0:20, mdat3.Vbar_g0, 
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="0.5"
	)
	plot!(
		0:20, mdat4.Vbar_g0, 
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="0.0"
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


	hline!(
		[1.0], ls=:dash, color="black", label="", alpha=0.75
	)
	vline!(
		[10], ls=:solid, color="black", label="", alpha=0.5
	)

	savefig(poverty_trap_plot, "../images/fig13_poverty.pdf")

	mdat = nothing
	mdat2 = nothing
	mdat3 = nothing
	mdat4 = nothing
	GC.gc()
	
	poverty_trap_plot
end

# ╔═╡ b01f4747-bdc0-497c-8361-760de4627a63
begin
	_, mdat5 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.95), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat6 = run!( initialize_pessimistic_learning(mixed=true, parochial=true, mixed_L1=1, mixed_L2=1, envshift=10, init_soc_v=0.95), 20, mdata=[:Vbar_g0, :Vbar_g1] )
	_, mdat7 = run!( initialize_pessimistic_learning(mixed=true, parochial=false, mixed_L1=3, mixed_L2=3, envshift=10, init_soc_v=0.5), 20, mdata=[:Vbar_g0, :Vbar_g1] )

	poverty_trap_plot_ingroup = plot(
		0:20, mdat5.Vbar_g0, 
		ls=:solid, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="non-parochial", legendtitle="learning strategies "*L"(\alpha = 0.95)",
		legendtitlefontsize=8,
		legendfontsize=6,
		xlab="time (generations)",
		ylab=L"\bar{V}"
	)
	plot!(
		0:20, mdat5.Vbar_g1, 
		lw=2, color=palette(:Dark2_5)[2], 
		label="", ls=:solid, alpha=0.75
	)
	plot!(
		0:20, mdat7.Vbar_g0,
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="non-parochial, payoff bias"
	)
	plot!(
		0:20, mdat7.Vbar_g1, 
		ls=:dot, lw=2, 
		color=palette(:Dark2_5)[2], 
		label=""
	)
	plot!(
		0:20, mdat6.Vbar_g0,
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[1], 
		label="parochial"
	)
	plot!(
		0:20, mdat6.Vbar_g1, 
		ls=:dashdotdot, lw=2, 
		color=palette(:Dark2_5)[2], 
		label=""
	)

	hline!(
		[1.0], ls=:dash, color="black", label="", alpha=0.75
	)
	vline!(
		[10], ls=:solid, color="black", label="", alpha=0.5
	)

	savefig(poverty_trap_plot_ingroup, "../images/sup4_parochial.pdf")

	mdat5 = nothing
	mdat6 = nothing
	mdat7 = nothing
	GC.gc()
	
	poverty_trap_plot_ingroup
end

# ╔═╡ f5064507-d46f-4f68-bc34-6a576e5a0da9
md"""
In the **fragile response** case, the best performing strategies for the low-risk group when inequality is present (parochial and/or payoff-biased strategies) are the worst performers after environments worsen, and groups find themselves in equally high-risk scenarios. Parochiality, either explicitly-implemented through markers or indirectly implemented through a payoff bias, fails the (formerly) low-risk group spectacularly, leaving it in worse conditions than the better-adapted, high-risk group. The previously-advantaged group will be stuck in its own version of a poverty trap, one in which their response to disaster is fragile, as it leads to overly optimistic risk attitudes and behaviors. Overcoming their parochialism/payoff bias becomes the only way to achieve better conditions.
"""

# ╔═╡ c29662bb-5990-438a-b53b-450fec8fddf2
md"""
There is a discussion to be had about short-term versus long-term adaptation here if environments oscillate. Disadvantaged groups seem to be better prepared for the long haul, while advantaged groups might do better in the short-term but might not do so well in the long-term, as they do not hedge very well for the times when environments grow very uncertain.
"""

# ╔═╡ e4e70596-db0f-495b-90be-a2f7c301e5c2
md"""
#### (the eventual) Discussion
"""

# ╔═╡ 073a5341-b1b2-4f15-8e35-5d4714590bd8
md"""
The balance between conservative and explorative learning strategies can be thought of as the action of the steering wheel guiding the cultural evolution of risk attitudes, as well as their capacity for adaptation to changing environments. Highly-explorative strategies lead to a wheel that is very responsive to the environment's twists and turns, and especially sprone to optimistic overshoots. Highly-conservative strategies are prone to keeping the wheel in the first suitable direction and  preferentially accept steering in pessimistic/risk averse directions.
"""

# ╔═╡ 94ea2626-d03d-43a6-b4cb-7e3981491d30
md"""
Being young and poor puts you at higher risk of optimism? Maybe this is why being young, poor and lacking parental figures are all risk factors for criminal behavior. I never bought the common explanation from life history theory. This model suggests this relationship comes from how cultural transmission works for risky behavior, as highly explorative learning will lead to excessively-optimistic stakes, and even more so in high-risk scenarios.
"""

# ╔═╡ b2dee4c9-dd07-46c4-8757-77529c023662
md"""
A lot of politically-conservative talking points seem to be risk-related. Coincidence?! I think not.
"""

# ╔═╡ a65eec8d-46e7-4c53-b7d8-e49956d9e45b
md"""
There is this paper by Gopnik suggesting that people who grow up in harsher environments have to mature earlier. There is an analogue of that in this model, as harsher conditions (more uncertain environments and/or lower baseline wealth) will lead agents to witness more ruin events earlier on during their gameplay phase, forcing earlier adjustments into more risk averse attitudes. 
"""

# ╔═╡ 4d76bc17-5d8a-4e36-bc0d-781f9a17a8a2
md"""
Maybe the WEIRD cluster is just the cluster of people who are high wealth (relative to rest of the world) and have been experiencing progressively less uncertain environments, encouraging less risk aversion and, in consequence, more explorative learning and less reliance on strict tradition and/or past ideas about risk. I like this thought, because it provides an explanation for the emergence of WEIRD that does not explicitly invoke ethnicity (Western) or political system (Democratic). University students in rich, urbanized areas are also a good example of very long juvenile learning periods. I honestly just hate the WEIRD acronym and I'm always on the lookout for better ways to look at the sampling problem.
"""

# ╔═╡ c5312b27-4984-4def-889a-c033321a0202
md"""
Traumatic events like economic catastrophe, climate collapse, warfare and colonization can not only set groups back through their direct effects. If a population is at a low baseline wealth level (or there is a correlated loss of wealth across a population), or if some traumatic event hits one group but not another in a mixed society (as, for example, when a group gets colonized by another), this trauma can persist even after conditions improve through the conservative cultural inheritance of high risk aversion. On the other hand, advantaged groups that have been able to relax their learning strategies and risk attitudes might be ill-equipped to deal with environments that suddenly grow more hostile and uncertain.
"""

# ╔═╡ 6af88939-505d-415d-8d36-f1c28e7f1f4e
md"""
Generational trauma can lead to persistent inequalities. Parochialism can be seen as a result of the generational transmission of trauma experienced by a disadvantaged group, especially when said trauma destabilizes the environment of said group and/or impoverishes its members. Peoples' need to be mindful of who they learn from can aid them in dealing with traumatic correlated shocks, but it might also engender conditions that make the seizing of new opportunities when environments improve more difficult and/or slower. So-called poverty traps are examples of these scenarios, and our theory can explain how they come to be through instances group-level trauma and/or persistent economic stratification. As the effects of poverty traps themselves might feed back into, and aid the persistence of, group-level wealth inequality, poverty traps that arise through these sorts of processes might be self-reinforcing. Das ist nicht so gut.
"""

# ╔═╡ Cell order:
# ╟─cfb0a045-a40d-4019-987f-e4dc1c92074e
# ╟─ff915f4b-e120-43c6-90d3-d9e4b7659994
# ╟─ad3d1e83-ee2d-484d-bd3e-359135e105ad
# ╟─13636d74-91c4-4a5b-ba09-5ec872ea2058
# ╟─f944d9d1-8804-4fd8-9611-a03d03b296ef
# ╟─600fc17f-7db8-4f18-b319-acdf28bd7644
# ╟─de81b0f9-d5e9-4443-a64a-a104cdf8f442
# ╟─7ee8dab2-d381-4064-b4cc-4f69cbc127b6
# ╟─896ff8a7-0f61-4be3-adf4-d8e55ad16874
# ╟─59cd6ed6-53cc-45f1-9a16-b62a72a00243
# ╟─f052e274-ccb2-4d54-b6a2-1975a5809124
# ╟─b739131f-a173-4994-8143-d6c52ac07cb3
# ╟─0d2af086-3c43-46cb-86a9-d4ac583ab688
# ╟─eb3d9026-c2c4-4882-9338-14c9e2852cce
# ╟─77d56680-5a71-40cb-98ae-eeb91a8ff2bc
# ╟─3a2082c4-ef09-4b53-af1b-411792729560
# ╟─04452287-07ab-45b5-9b8f-f175ddc43d61
# ╟─96cdc247-40df-4538-be8a-7465ca638bf2
# ╟─9eb10b7b-b282-4b10-831a-ea00fb722538
# ╟─3e4b23c8-7a02-45f3-9753-1b2dbaba0fa9
# ╟─ea485e6d-2cfa-48b7-b02d-d86578ca0d37
# ╟─e7c9aed3-b053-4052-9337-be5b9a9b8880
# ╟─188478f3-c4a4-49d2-a5d9-e018486b3842
# ╟─a074cc0f-9484-4771-8078-2581aaa17f78
# ╟─aa8d557f-6436-407f-a0f7-afa823a35129
# ╟─1fc1da0c-4ccd-442d-903f-c26b43a3f94c
# ╟─fc9b6c6b-6832-4842-a246-3492ce9e61e2
# ╟─d008b259-7a84-4f7e-99ed-e4b283775591
# ╟─babba0ec-d157-443f-b3da-789d248a0f71
# ╟─71768709-5e37-43fc-91d9-cd47e659e1d8
# ╟─b97683e7-9187-49d2-baaa-301327bf65ab
# ╟─256f53ce-c5a0-413d-9de9-3fa15e51289e
# ╟─aebf5e0d-7520-4f6e-9dff-493c4e7a5bb0
# ╟─009223ce-5095-491d-89b8-4e372e025446
# ╟─6e37d178-4026-4dce-94bf-8b8eb3f1d6fb
# ╟─127fbf7b-4b8e-49e1-8239-b9bb6698d7ce
# ╟─7300889d-373f-424d-9cdc-f28856cb0f23
# ╟─189cca21-b09b-4b56-ab6c-82f3120cd31f
# ╟─019af979-9993-462c-831f-41e50c141ce5
# ╟─01a70f2d-7c1d-404f-8501-82b7d4a44d10
# ╟─502806bf-06c1-4c3a-aba9-719e4cba2990
# ╟─e8ad4bb4-ee44-4df4-9aed-db6d764a56b9
# ╟─b2481d42-e3c5-471e-9277-59fd2b53aec1
# ╟─da7818d8-3692-4896-9c85-0cb439da6626
# ╟─da6dbb32-fcee-43d4-b38e-a360de2ea54f
# ╟─6ea0d83d-b3df-443a-9dd0-baf314ded951
# ╟─5f2c213c-2d15-47ef-8b08-b86b3d1101c2
# ╟─8412ef2a-8429-4398-ad2b-6461c2071f57
# ╟─8d568cf3-fb47-4f77-b603-ea99271f6bfe
# ╟─9ff6bb98-6e7b-4ebd-9a35-9b1d15144658
# ╟─26ea2476-a0da-4f33-8e4b-6d3619db30b6
# ╟─b01f4747-bdc0-497c-8361-760de4627a63
# ╟─f5064507-d46f-4f68-bc34-6a576e5a0da9
# ╟─c29662bb-5990-438a-b53b-450fec8fddf2
# ╟─e4e70596-db0f-495b-90be-a2f7c301e5c2
# ╟─073a5341-b1b2-4f15-8e35-5d4714590bd8
# ╟─94ea2626-d03d-43a6-b4cb-7e3981491d30
# ╟─b2dee4c9-dd07-46c4-8757-77529c023662
# ╟─a65eec8d-46e7-4c53-b7d8-e49956d9e45b
# ╟─4d76bc17-5d8a-4e36-bc0d-781f9a17a8a2
# ╟─c5312b27-4984-4def-889a-c033321a0202
# ╟─6af88939-505d-415d-8d36-f1c28e7f1f4e
