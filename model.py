"""Game model: users decide to share data; the organization sets the reward.

User i (privacy sensitivity theta_i in [0,1]) with x = fraction of users sharing:
    U_share(i, x) = r + beta*x - theta_i * L * (1 + gamma*x)
    U_not_share   = 0
  r      reward offered          beta  service benefit from a bigger data pool
  L      base privacy loss       gamma extra risk as more data is pooled (breach/re-identification)
User i shares  <=>  theta_i < T(x) = (r + beta*x) / (L*(1 + gamma*x)).
Nash equilibrium (population):  x* = F(T(x*)), F = CDF of sensitivities.
Organization (Stackelberg leader): profit(r) = N * x*(r) * (v - r).
"""
import numpy as np
import pandas as pd

DEFAULTS = dict(N=1000, seed=42, a=2.0, b=3.0, v=8.0, L=10.0, beta=8.0, gamma=0.8, r=2.0)


def make_users(N, seed, a, b, **_):
    rng = np.random.default_rng(seed)
    return rng.beta(a, b, N)


def threshold(x, r, p):
    return (r + p["beta"] * x) / (p["L"] * (1 + p["gamma"] * x))


def equilibria(theta, r, p, grid=801):
    """All pure population equilibria as a list of (x, stable)."""
    th = np.sort(theta)
    x = np.linspace(0, 1, grid)
    g = np.searchsorted(th, threshold(x, r, p), side="left") / len(th)
    h = g - x
    roots = []
    if h[0] <= 0:
        roots.append((0.0, True))
    for i in np.where(h[:-1] * h[1:] < 0)[0]:
        x0 = x[i] + (x[i + 1] - x[i]) * h[i] / (h[i] - h[i + 1])
        roots.append((float(x0), bool(h[i] > 0)))  # + to - crossing => stable
    if h[-1] >= 0:
        roots.append((1.0, True))
    if not roots:  # exact grid touch: take the closest fixed point
        roots.append((float(x[np.argmin(np.abs(h))]), True))
    return roots, x, g


def best_equilibrium(theta, r, p):
    roots, *_ = equilibria(theta, r, p)
    stable = [x for x, s in roots if s]
    return max(stable) if stable else roots[-1][0]


def outcome(theta, r, p):
    x = best_equilibrium(theta, r, p)
    t = threshold(x, r, p)
    share = theta < t
    util = np.where(share, r + p["beta"] * x - theta * p["L"] * (1 + p["gamma"] * x), 0.0)
    part = share.mean()
    return dict(x=x, T=t, participation=part, profit=len(theta) * part * (p["v"] - r),
                user_welfare=util.sum(), avg_utility=util.mean(), share=share, util=util)


def sweep_rewards(theta, p, r_max=None, n=60):
    r_max = r_max or p["v"] * 1.2
    rows = []
    for r in np.linspace(0, r_max, n):
        o = outcome(theta, r, p)
        rows.append(dict(reward=r, participation=o["participation"], profit=o["profit"],
                         user_welfare=o["user_welfare"], avg_utility=o["avg_utility"]))
    return pd.DataFrame(rows)


def user_table(theta, r, p):
    o = outcome(theta, r, p)
    return pd.DataFrame({
        "user": np.arange(len(theta)), "sensitivity": theta,
        "min_reward_needed": theta * p["L"] * (1 + p["gamma"] * o["x"]) - p["beta"] * o["x"],
        "shares_data": o["share"], "utility": o["util"]}), o


def mixed_strategy(theta_i, r, p):
    """Symmetric mixed equilibrium: prob. p* at which type theta_i is indifferent."""
    c = theta_i * p["L"]
    denom = p["beta"] - c * p["gamma"]
    pstar = (c - r) / denom if abs(denom) > 1e-12 else np.nan
    return pstar, denom  # denom<0 => congestion-like, mixed point is stable


def participation_heatmap(theta, p, rewards, scales):
    Z = np.zeros((len(scales), len(rewards)))
    for i, s in enumerate(scales):
        t = np.clip(theta * s, 0, 1)
        for j, r in enumerate(rewards):
            Z[i, j] = (t < threshold(best_equilibrium(t, r, p), r, p)).mean()
    return Z
