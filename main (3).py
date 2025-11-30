"""
11/25 Model – AV Pricing with MEC Scaling, Mis-spec Robustness, and Diagnostics

- Two parallel BPR links
- Live Pigouvian tolls (unscaled MEC, or scaled by effective compliance p_eff)
- Class-specific logit route choice (AV vs Human)
- Time-sliced peak demand; partial compliance via eta
- Sweep over AV penetration p, compare welfare curves
- Robustness sweep over controller capacity mis-specification

Creates:
  data/sweep_core.csv
  data/sweep_mispec_p*.csv
  figs/welfare_vs_p.png
  figs/welfare_vs_mispec_p*.png
  figs/tolls_*.png
  figs/route_shares_*.png
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# -----------------------------
# Parameters
# -----------------------------
@dataclass
class DemandSpec:
	T_minutes: int = 120
	dt_sec: int = 20
	Qmax_per_h: float = 2400.0
	peak_min: float = 60.0
	sigma_min: float = 20.0
	noise_frac: float = 0.0  # 0.05 for CI runs

@dataclass
class LinkSpec:
	t0_min: float
	C_per_h: float

@dataclass
class BPRSpec:
	alpha: float = 0.15
	beta: float = 4.0

@dataclass
class ChoiceSpec:
	theta_av: float = 0.4   # 1/min (route-choice sensitivity for AVs)
	theta_h: float  = 0.4   # for humans
	eta: float = 0.0        # human toll sensitivity multiplier in [0,1]
	equalize_when_tau_zero: bool = True  # if True, use a common θ when both tolls are ~0
	# --- Mixed logit options (optional; leave sim_R=0 to disable) ---
	sim_R: int = 0  # if >0, use R simulated draws per class per bin (continuous mixed logit)
	ln_gamma_mu_av: float = math.log(2.5)
	ln_gamma_sigma_av: float = 0.0
	ln_theta_mu_av: float = math.log(0.6)
	ln_theta_sigma_av: float = 0.0
	rho_av: float = 0.0
	ln_gamma_mu_h: float = math.log(2.5)
	ln_gamma_sigma_h: float = 0.0
	ln_theta_mu_h: float = math.log(0.6)
	ln_theta_sigma_h: float = 0.0
	rho_h: float = 0.0
	# Route-choice inertia across bins: fraction of travelers who reconsider each bin
	share_relax: float = 0.4

@dataclass
class TollSpec:
	gamma_min_per_dollar: float = 60.0/24.0  # VOT=$24/h => 2.5 min/$
	smoothing_lambda: float = 0.5            # low-pass filter on τ
	tau_cap: float | None = None             # hard cap on τ [$/veh]
	rate_cap_per_min: float = 0.5            # max |Δτ| per minute (anti-oscillation)
	mec_ema_alpha: float = 0.3               # EMA smoothing for MEC target

@dataclass
class Scenario:
	p_av: float = 0.3                  # AV penetration
	controller: str = "scaled"         # "scaled" or "unscaled"
	capacity_mispec_frac: float = 0.0  # +0.1 or -0.1 if controller mis-specifies C
	inner_iters: int = 4               # within-bin fixed-point iterations
	inner_relax: float = 0.3           # MSA relaxation for flows (0<ω≤1)

# Defaults
LINK1 = LinkSpec(t0_min=10.0, C_per_h=800.0)   # fast
LINK2 = LinkSpec(t0_min=15.0, C_per_h=1600.0)   # slow
BPR   = BPRSpec(alpha=0.15, beta=4.0)
CHOICE= ChoiceSpec(theta_av=0.4, theta_h=0.4, eta=0.0)
TOLL  = TollSpec(gamma_min_per_dollar=60.0/24.0, smoothing_lambda=0.5, tau_cap=None)
DEMAND= DemandSpec(T_minutes=120, dt_sec=20, Qmax_per_h=2400.0, peak_min=60.0, sigma_min=20.0, noise_frac=0.0)

# -----------------------------
# Helpers: Demand, Network, Controller, Choice
# -----------------------------

def time_grid_and_demand(spec: DemandSpec, p_av: float, rng: np.random.Generator):
	K = int(spec.T_minutes*60/spec.dt_sec)
	dt_h = spec.dt_sec/3600.0
	t_min = np.arange(K)*spec.dt_sec/60.0
	# Gaussian peak around peak_min
	Q = spec.Qmax_per_h*np.exp(-0.5*((t_min - spec.peak_min)/spec.sigma_min)**2)
	if spec.noise_frac > 0:
		eps = rng.normal(0.0, spec.noise_frac, size=K)
		eps = np.clip(eps, -2*spec.noise_frac, 2*spec.noise_frac)
		Q = Q*(1.0+eps)
		Q = np.clip(Q, 0.0, None)
	Q_av = p_av*Q
	Q_h  = (1.0-p_av)*Q
	return K, dt_h, t_min, Q, Q_av, Q_h

# BPR travel time (minutes) given per-bin flow (veh/h)
def bpr_travel_time(q: float, t0: float, C: float, a: float, b: float) -> float:
	x = max(q/C, 0.0) if C>0 else 0.0
	return t0*(1.0 + a*(x**b))

# MEC in minutes (marginal external delay) for BPR
def bpr_mec_minutes(q: float, t0: float, C: float, a: float, b: float) -> float:
	if q <= 0.0 or C <= 0.0:
		return 0.0
	x = q/C
	return q * t0 * a * b * (1.0/C) * (x**(b-1.0))

# Effective compliance share among marginal responders
def effective_compliance(p: float, eta: float, theta_h: float, theta_av: float) -> float:
	"""Effective compliance share.

	Interpret eta as the fraction of humans who effectively internalize the toll.
	So p_eff is simply the AV share plus eta times the human share:
	  - eta=0   => only AVs respond (p_eff = p)
	  - eta=1   => everyone responds (p_eff = 1)
	"""
	peff = p + eta*(1.0 - p)
	return max(1e-6, min(1.0, peff))

# Tolls (dollars)
def toll_unscaled_mec(mec_minutes: float, gamma: float) -> float:
	return mec_minutes/gamma

def toll_scaled_mec(mec_minutes: float, gamma: float, p_eff: float) -> float:
	return mec_minutes/(gamma*max(p_eff, 1e-6))

def smooth_tau(prev: float, target: float, lam: float, cap: float|None,
			   rate_cap_per_min: float, dt_min: float) -> float:
	"""One-step τ update with low-pass smoothing and rate limiting.
	rate_cap_per_min limits absolute change per minute; applied per time bin of length dt_min.
	"""
	tau_unsat = (1.0 - lam) * prev + lam * target
	max_delta = rate_cap_per_min * dt_min
	delta = np.clip(tau_unsat - prev, -max_delta, +max_delta)
	tau = prev + delta
	if cap is not None:
		tau = min(tau, cap)
	return max(0.0, tau)

# Logit splits for a 2-route choice
def logit_split(costs_min: np.ndarray, theta: float) -> np.ndarray:
	# subtract min for numerical stability
	z = np.exp(-theta*(costs_min - costs_min.min()))
	s = z / z.sum()
	return s


def _mixed_logit_share(t_min: np.ndarray, tau_eff: np.ndarray,
					   ln_gamma_mu: float, ln_gamma_sigma: float,
					   ln_theta_mu: float, ln_theta_sigma: float,
					   rho: float, R: int, rng: np.random.Generator) -> np.ndarray:
	"""Return 2-route shares under simulated mixed logit with random gamma, theta.
	gamma ~ lognormal(ln_gamma_mu, ln_gamma_sigma)
	theta ~ lognormal(ln_theta_mu, ln_theta_sigma)
	corr(ln gamma, ln theta) = rho
	"""
	if R <= 0 or (ln_gamma_sigma == 0.0 and ln_theta_sigma == 0.0):
		gamma = math.exp(ln_gamma_mu)
		theta = math.exp(ln_theta_mu)
		C = t_min + gamma * tau_eff
		return logit_split(C, theta)
	cov = np.array([[ln_gamma_sigma**2, rho*ln_gamma_sigma*ln_theta_sigma],
					[rho*ln_gamma_sigma*ln_theta_sigma, ln_theta_sigma**2]], dtype=float)
	try:
		L = np.linalg.cholesky(cov + 1e-12*np.eye(2))
	except np.linalg.LinAlgError:
		L = np.zeros((2,2))
		if ln_gamma_sigma>0: L[0,0]=ln_gamma_sigma
		if ln_theta_sigma>0: L[1,1]=ln_theta_sigma
	z = rng.standard_normal(size=(2, R))
	draws = (np.array([[ln_gamma_mu],[ln_theta_mu]]) + L @ z)
	ln_gammas, ln_thetas = draws[0], draws[1]
	gammas = np.exp(ln_gammas)
	thetas = np.exp(ln_thetas)
	costs0 = t_min[0] + gammas * tau_eff[0]
	costs1 = t_min[1] + gammas * tau_eff[1]
	cmin = np.minimum(costs0, costs1)
	e0 = np.exp(-thetas * (costs0 - cmin))
	e1 = np.exp(-thetas * (costs1 - cmin))
	s0 = e0 / (e0 + e1)
	s1 = 1.0 - s0
	return np.array([s0.mean(), s1.mean()])


def class_splits(t_min: np.ndarray, tau_dollars: np.ndarray, gamma: float,
				 choice: ChoiceSpec, rng: np.random.Generator):
	"""Return (s_av, s_h, C_av, C_h) route shares and perceived costs.

	Uses a logit route choice model, optionally with mixed-logit heterogeneity
	(random VOT and sensitivity) when choice.sim_R > 0. Humans can have
	attenuated toll sensitivity via eta.
	"""
	t_min = np.asarray(t_min, dtype=float)
	tau_dollars = np.asarray(tau_dollars, dtype=float)

	C_av = t_min + gamma * tau_dollars
	C_h  = t_min + choice.eta * gamma * tau_dollars

	# If tolls ~ 0 and we want equivalence, use a common θ and time-only costs
	if choice.equalize_when_tau_zero and np.allclose(tau_dollars, 0.0, atol=1e-9):
		theta_common = 0.5 * (choice.theta_av + choice.theta_h)
		s = logit_split(t_min, theta_common)
		return s, s.copy(), C_av, C_h

	if choice.sim_R and choice.sim_R > 0:
		s_av = _mixed_logit_share(t_min, tau_dollars,
								  choice.ln_gamma_mu_av, choice.ln_gamma_sigma_av,
								  choice.ln_theta_mu_av, choice.ln_theta_sigma_av,
								  choice.rho_av, choice.sim_R, rng)
		s_h  = _mixed_logit_share(t_min, choice.eta * tau_dollars,
								  choice.ln_gamma_mu_h, choice.ln_gamma_sigma_h,
								  choice.ln_theta_mu_h, choice.ln_theta_sigma_h,
								  choice.rho_h, choice.sim_R, rng)
	else:
		s_av = logit_split(C_av, choice.theta_av)
		s_h  = logit_split(C_h,  choice.theta_h)

	return s_av, s_h, C_av, C_h


# -----------------------------
# Simulation
# -----------------------------

def run_once(
	link1: LinkSpec = LINK1,
	link2: LinkSpec = LINK2,
	bpr: BPRSpec = BPR,
	choice: ChoiceSpec = CHOICE,
	toll: TollSpec = TOLL,
	demand: DemandSpec = DEMAND,
	scenario: Scenario = Scenario(),
	seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
	"""Simulate one day for given scenario; return bin-level and aggregated dataframes.

	Key design choices:
	  - Physics (travel time) always uses TRUE capacities.
	  - Controller MEC calculation uses controller-believed capacities (with mis-spec).
	  - MEC EMA is updated once per time bin, after inner physics convergence.
	  - effective_compliance uses p_eff = p + eta*(1-p).
	"""
	rng = np.random.default_rng(seed)
	K, dt_h, t_min, Q, Q_av, Q_h = time_grid_and_demand(demand, scenario.p_av, rng)

	# True capacities vs controller's (mis-spec allowed for MEC calc only)
	C1_true, C2_true = link1.C_per_h, link2.C_per_h
	C1_ctrl = C1_true*(1.0 + scenario.capacity_mispec_frac)
	C2_ctrl = C2_true  # keep slow link correct by default

	# Initialize state
	q1 = q2 = 0.0  # veh/h last-bin flows
	tau1 = tau2 = 0.0  # dollars (controller state)

	records = []

	# Anti-oscillation helpers
	dt_min = demand.dt_sec/60.0
	mec1_ema = 0.0
	mec2_ema = 0.0
	inner_iters = getattr(scenario, "inner_iters", 6)
	inner_relax = getattr(scenario, "inner_relax", 0.4)
	# Route-choice inertia across time bins: only a fraction of travelers reconsider each bin
	share_relax = max(0.0, min(1.0, getattr(choice, "share_relax", 0.4)))
	s_av_prev = np.array([0.5, 0.5], dtype=float)
	s_h_prev  = np.array([0.5, 0.5], dtype=float)

	for k in range(K):
		# ------------------------------
		# A. Inner loop: physics equilibrium with fixed tolls this bin
		# ------------------------------
		q1_loc, q2_loc = q1, q2
		tau_vec = np.array([tau1, tau2], dtype=float)

		for _ in range(max(1, inner_iters)):
			# 1) link times from current local flows (physics uses TRUE capacities)
			t1 = bpr_travel_time(q1_loc, link1.t0_min, C1_true, bpr.alpha, bpr.beta)
			t2 = bpr_travel_time(q2_loc, link2.t0_min, C2_true, bpr.alpha, bpr.beta)
			t_vec = np.array([t1, t2], dtype=float)

			# 2) Class splits with current τ (held fixed within bin)
			s_av, s_h, C_av, C_h = class_splits(
				t_vec, tau_vec, toll.gamma_min_per_dollar,
				choice, rng
			)
			q1_new = float(s_av[0]*Q_av[k] + s_h[0]*Q_h[k])
			q2_new = float(s_av[1]*Q_av[k] + s_h[1]*Q_h[k])

			# 3) MSA relaxation of flows
			q1_loc = (1 - inner_relax) * q1_loc + inner_relax * q1_new
			q2_loc = (1 - inner_relax) * q2_loc + inner_relax * q2_new

		# Commit physics state for this bin
		q1, q2 = q1_loc, q2_loc
		# Recompute times at converged flows
		t1 = bpr_travel_time(q1, link1.t0_min, C1_true, bpr.alpha, bpr.beta)
		t2 = bpr_travel_time(q2, link2.t0_min, C2_true, bpr.alpha, bpr.beta)
		t_vec = np.array([t1, t2], dtype=float)

		# Final splits for logging, based on tolls actually applied this bin
		s_av_star, s_h_star, C_av, C_h = class_splits(
			t_vec, tau_vec, toll.gamma_min_per_dollar,
			choice, rng
		)

		# Route-choice inertia across bins
		if k == 0 or share_relax >= 1.0:
			s_av_eff = s_av_star
			s_h_eff  = s_h_star
		else:
			s_av_eff = (1.0 - share_relax)*s_av_prev + share_relax*s_av_star
			s_h_eff  = (1.0 - share_relax)*s_h_prev  + share_relax*s_h_star
		s_av_prev = s_av_eff
		s_h_prev  = s_h_eff

		# Realized flows this bin
		q1 = float(s_av_eff[0]*Q_av[k] + s_h_eff[0]*Q_h[k])
		q2 = float(s_av_eff[1]*Q_av[k] + s_h_eff[1]*Q_h[k])

		# ------------------------------
		# B. Controller update: once per bin, after physics settles
		#     MEC uses controller's (possibly mis-specified) capacities
		# ------------------------------
		mec1_inst = bpr_mec_minutes(q1, link1.t0_min, C1_ctrl, bpr.alpha, bpr.beta)
		mec2_inst = bpr_mec_minutes(q2, link2.t0_min, C2_ctrl, bpr.alpha, bpr.beta)
		mec1_ema = (1 - TOLL.mec_ema_alpha) * mec1_ema + TOLL.mec_ema_alpha * mec1_inst
		mec2_ema = (1 - TOLL.mec_ema_alpha) * mec2_ema + TOLL.mec_ema_alpha * mec2_inst

		p = scenario.p_av
		p_eff = effective_compliance(p, choice.eta, choice.theta_h, choice.theta_av)
		if scenario.controller == "scaled":
			tau1_target = toll_scaled_mec(mec1_ema, toll.gamma_min_per_dollar, p_eff)
			tau2_target = toll_scaled_mec(mec2_ema, toll.gamma_min_per_dollar, p_eff)
		elif scenario.controller == "unscaled":
			tau1_target = toll_unscaled_mec(mec1_ema, toll.gamma_min_per_dollar)
			tau2_target = toll_unscaled_mec(mec2_ema, toll.gamma_min_per_dollar)
		else:
			raise ValueError("controller must be 'scaled' or 'unscaled'")

		# Smooth + rate-limit τ toward target (controller state for next bin)
		tau1 = smooth_tau(tau1, tau1_target, toll.smoothing_lambda, toll.tau_cap,
						  toll.rate_cap_per_min, dt_min)
		tau2 = smooth_tau(tau2, tau2_target, toll.smoothing_lambda, toll.tau_cap,
						  toll.rate_cap_per_min, dt_min)

		records.append({
			"k": k,
			"time_min": float(t_min[k]),
			"p": scenario.p_av,
			"controller": scenario.controller,
			"Q": float(Q[k]),
			"Q_av": float(Q_av[k]),
			"Q_h": float(Q_h[k]),
			"q1": q1, "q2": q2,
			"q1_av": float(s_av_eff[0]*Q_av[k]), "q1_h": float(s_h_eff[0]*Q_h[k]),
			"q2_av": float(s_av_eff[1]*Q_av[k]), "q2_h": float(s_h_eff[1]*Q_h[k]),
			"t1": float(t1), "t2": float(t2),
			"tau1": float(tau1), "tau2": float(tau2),
			"C1_av": float(C_av[0]), "C2_av": float(C_av[1]),
			"C1_h": float(C_h[0]),   "C2_h": float(C_h[1]),
		})

	df_bins = pd.DataFrame.from_records(records)

	# ------------------------------
	# Aggregate to day-level metrics
	# ------------------------------
	dt_h = DEMAND.dt_sec/3600.0
	# Total travel time minutes across all vehicles over the day
	TT_min_total = ((df_bins["q1"]*df_bins["t1"] + df_bins["q2"]*df_bins["t2"]) * dt_h).sum()
	# Toll revenue (dollars) — AVs pay full, humans pay eta fraction
	Rev_total = (((df_bins["q1_av"]*df_bins["tau1"] + df_bins["q2_av"]*df_bins["tau2"]) +
				  CHOICE.eta*(df_bins["q1_h"]*df_bins["tau1"] + df_bins["q2_h"]*df_bins["tau2"])) * dt_h).sum()
	# Welfare in dollars (no rebate): - time cost (minutes)/gamma + revenue
	W_norebate = -(TT_min_total / TOLL.gamma_min_per_dollar) + Rev_total
	# Revenue-neutral welfare: rebating tolls lump-sum (same choices), equals just - time cost term
	W_withrebate = -(TT_min_total / TOLL.gamma_min_per_dollar)

	# Average delay proxy (vs free-flow)
	TT_ff_min_total = (((df_bins["q1"]*LINK1.t0_min) + (df_bins["q2"]*LINK2.t0_min)) * dt_h).sum()
	avg_delay_min = (TT_min_total - TT_ff_min_total) / max(((df_bins["q1"]+df_bins["q2"]) * dt_h).sum(), 1e-9)

	# Simple p95 travel time proxy (per-bin AV generalized time)
	costs_av_min = (df_bins[["C1_av","C2_av"]].min(axis=1)).to_numpy()
	p95_tt_min = np.percentile(costs_av_min, 95)

	fast_share_av = (df_bins["q1_av"]*dt_h).sum() / max((df_bins["Q_av"]*dt_h).sum(), 1e-9)
	fast_share_h  = (df_bins["q1_h"]*dt_h).sum()  / max((df_bins["Q_h"]*dt_h).sum(),  1e-9)

	day = pd.DataFrame.from_records([{
		"p": scenario.p_av,
		"controller": scenario.controller,
		"TT_min_total": TT_min_total,
		"Revenue_total": Rev_total,
		"W_norebate": W_norebate,
		"W_withrebate": W_withrebate,
		"avg_delay_min": avg_delay_min,
		"p95_tt_min": float(p95_tt_min),
		"fast_share_av": float(fast_share_av),
		"fast_share_h": float(fast_share_h),
	}])

	return df_bins, day

# -----------------------------
# Experiment sweep & plotting
# -----------------------------

def plot_route_choice_timeseries(df_bins: pd.DataFrame, title: str, out_path: str|None=None):
	"""Plot the fast-route share over time for AVs and Humans, plus overall share."""
	fig, ax = plt.subplots(figsize=(7.5,4.2))
	t = df_bins['time_min'].to_numpy()
	Q  = df_bins['Q'].to_numpy()
	Q_av = df_bins['Q_av'].to_numpy()
	Q_h  = df_bins['Q_h'].to_numpy()

	eps = 1e-9
	share_av = np.where(Q_av > eps, (df_bins['q1_av'].to_numpy()) / (Q_av + eps), 0.0)
	share_h  = np.where(Q_h  > eps, (df_bins['q1_h'].to_numpy())  / (Q_h  + eps), 0.0)
	share_all= np.where(Q   > eps, (df_bins['q1'].to_numpy())     / (Q   + eps), 0.0)

	ax.plot(t, share_av, label='Fast-route share (AV)', linewidth=1.8)
	ax.plot(t, share_h,  label='Fast-route share (Human)', linewidth=1.8, linestyle='--')
	ax.plot(t, share_all, label='Fast-route share (All)', linewidth=1.2, alpha=0.6)

	ax2 = ax.twinx()
	ax2.plot(t, Q, alpha=0.3, label='Demand Q(t)')
	ax2.set_ylabel('Demand Q(t) [veh/h]', alpha=0.8)
	ax2.grid(False)

	ax.set_ylim(-0.02, 1.02)
	ax.set_xlabel('Time [min]')
	ax.set_ylabel('Share on fast route')
	ax.set_title(title)
	ax.legend(loc='best')
	fig.tight_layout()
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig.savefig(out_path, dpi=160)
	return ax


def plot_tolls_timeseries(df_bins: pd.DataFrame, title: str, out_path: str|None=None):
	fig, ax = plt.subplots(figsize=(7.5,4.2))
	t = df_bins['time_min']
	l1, = ax.plot(t, df_bins['tau1'], label='τ₁ (fast route)', linewidth=1.8)
	l2, = ax.plot(t, df_bins['tau2'], label='τ₂ (slow route)', linewidth=1.8, linestyle='--')

	ax2 = ax.twinx()
	ax2.plot(t, df_bins['Q'], alpha=0.35, label='Demand Q(t)')
	ax2.set_ylabel('Demand Q(t) [veh/h]', alpha=0.8)
	ax2.grid(False)

	ax.set_xlabel('Time [min]')
	ax.set_ylabel('Toll [dollars]')
	ax.set_title(title)
	ax.legend(handles=[l1, l2], loc='upper left')
	fig.tight_layout()
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig.savefig(out_path, dpi=160)
	return ax


def sweep_over_p(p_list: List[float], controllers=("unscaled","scaled"), seeds: List[int]|None=None) -> pd.DataFrame:
	if seeds is None:
		seeds = [0]
	all_days = []
	for ctrl in controllers:
		for p in p_list:
			for sd in seeds:
				scen = Scenario(p_av=p, controller=ctrl, capacity_mispec_frac=0.0)
				_, day = run_once(scenario=scen, seed=sd)
				day["seed"] = sd
				all_days.append(day)
	return pd.concat(all_days, ignore_index=True)


def plot_welfare_vs_p(df_days: pd.DataFrame, out_path: str|None=None):
	fig, ax = plt.subplots(figsize=(7.2,4.6))
	ycol = "W_withrebate"  # revenue-neutral welfare: time-only
	yvals: list[float] = []
	for ctrl, label, style in [("unscaled","Unscaled MEC (naïve)","-"),("scaled","Scaled MEC / p_eff","--")]:
		sub = df_days[df_days["controller"]==ctrl]
		grp = sub.groupby("p", as_index=False)[ycol].mean()
		ax.plot(grp["p"], grp[ycol], linestyle=style, marker="o", label=label)
		yvals.extend(grp[ycol].tolist())
	if yvals:
		y_min = min(yvals)
		y_max = max(yvals)
		pad = 0.05 * max(1.0, abs(y_max - y_min))
		ax.set_ylim(y_min - pad, y_max + pad)
	ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
	ax.set_xlabel("AV penetration p")
	ax.set_ylabel("Revenue-neutral welfare (time only) [dollars]")
	ax.set_title("Welfare vs AV share (revenue-neutral): Scaled vs Unscaled MEC")
	ax.legend()
	fig.tight_layout()
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig.savefig(out_path, dpi=160)
	return ax

# -----------------------------
# Robustness to capacity mis-specification
# -----------------------------

def sweep_over_mispec(p: float,
					  mispec_list,
					  controllers=("unscaled","scaled"),
					  seeds: List[int] | None = None) -> pd.DataFrame:
	"""Sweep welfare vs controller capacity mis-specification at a fixed AV share p."""
	if seeds is None:
		seeds = [0]
	all_days = []
	for ctrl in controllers:
		for m in mispec_list:
			for sd in seeds:
				scen = Scenario(p_av=p, controller=ctrl, capacity_mispec_frac=m)
				_, day = run_once(scenario=scen, seed=sd)
				day["seed"] = sd
				day["capacity_mispec_frac"] = m
				all_days.append(day)
	return pd.concat(all_days, ignore_index=True)


def plot_welfare_vs_mispec(df_days: pd.DataFrame,
						   p: float,
						   out_path: str | None = None):
	"""Plot revenue-neutral welfare vs capacity mis-specification, for a fixed p."""
	fig, ax = plt.subplots(figsize=(7.2, 4.6))
	ycol = "W_withrebate"
	yvals: list[float] = []
	for ctrl, label, style in [
		("unscaled", "Unscaled MEC (naïve)", "-"),
		("scaled", "Scaled MEC / p_eff", "--"),
	]:
		sub = df_days[df_days["controller"] == ctrl]
		grp = sub.groupby("capacity_mispec_frac", as_index=False)[ycol].mean()
		grp = grp.sort_values("capacity_mispec_frac")
		ax.plot(grp["capacity_mispec_frac"], grp[ycol],
				linestyle=style, marker="o", label=label)
		yvals.extend(grp[ycol].tolist())
	if yvals:
		y_min = min(yvals)
		y_max = max(yvals)
		pad = 0.05 * max(1.0, abs(y_max - y_min))
		ax.set_ylim(y_min - pad, y_max + pad)
	ax.axhline(0.0, color="k", linewidth=0.8, alpha=0.6)
	ax.set_xlabel("Capacity mis-specification fraction (controller belief)")
	ax.set_ylabel("Revenue-neutral welfare (time only) [dollars]")
	ax.set_title(f"Welfare vs capacity mis-specification (p = {p})")
	ax.legend()
	fig.tight_layout()
	if out_path:
		os.makedirs(os.path.dirname(out_path), exist_ok=True)
		fig.savefig(out_path, dpi=160)
	return ax


# -----------------------------
# Main: run core & robustness sweeps
# -----------------------------
if __name__ == "__main__":
	os.makedirs("data", exist_ok=True)
	os.makedirs("figs", exist_ok=True)

	# 1) Core welfare vs p
	P = [round(x,2) for x in np.linspace(0.0, 1.0, 21)]
	days = sweep_over_p(P, controllers=("unscaled","scaled"), seeds=[0])

	out_csv = os.path.join("data","sweep_core.csv")
	days.to_csv(out_csv, index=False)
	plot_welfare_vs_p(days, out_path=os.path.join("figs","welfare_vs_p.png"))

	# 2) Time-series diagnostics for representative p values
	P_SHOW = [0.2, 0.5, 0.8]
	for ctrl in ("unscaled","scaled"):
		for p in P_SHOW:
			scen = Scenario(p_av=p, controller=ctrl, capacity_mispec_frac=0.0)
			df_bins, _ = run_once(scenario=scen, seed=0)
			title = f"Tolls over time — controller: {ctrl}, p={p}"
			fname = f"tolls_{ctrl}_p{str(p).replace('.','p')}.png"
			plot_tolls_timeseries(df_bins, title, out_path=os.path.join("figs", fname))
			title_rc = f"Fast-route shares over time — controller: {ctrl}, p={p}"
			fname_rc = f"route_shares_{ctrl}_p{str(p).replace('.','p')}.png"
			plot_route_choice_timeseries(df_bins, title_rc, out_path=os.path.join("figs", fname_rc))

	# 3) Robustness to capacity mis-specification at a few AV shares
	MIS = [-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
	P_ROB = [0.0, 0.2, 0.5, 0.81, 1.0]
	for p in P_ROB:
		df_mis = sweep_over_mispec(p, MIS, controllers=("unscaled","scaled"), seeds=[0])
		p_tag = str(p).replace(".", "p")
		out_csv_mis = os.path.join("data", f"sweep_mispec_p{p_tag}.csv")
		df_mis.to_csv(out_csv_mis, index=False)
		out_fig_mis = os.path.join("figs", f"welfare_vs_mispec_p{p_tag}.png")
		plot_welfare_vs_mispec(df_mis, p, out_path=out_fig_mis)

	print("Wrote core and mis-spec robustness results (data/*.csv, figs/*.png)")
