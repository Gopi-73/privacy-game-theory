import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import model as m

st.set_page_config(page_title="Privacy Data-Sharing Game", layout="wide")
st.title("🔐 Game-Theoretic Privacy-Preserving Data Sharing")
st.caption("Users weigh reward vs. privacy risk; the organization sets the reward; we solve for Nash equilibrium.")

D = m.DEFAULTS
with st.sidebar:
    st.header("Population")
    N = st.slider("Users", 200, 5000, D["N"], 100)
    seed = st.number_input("Random seed", 0, 9999, D["seed"])
    a = st.slider("Sensitivity Beta(a, ·)", 0.5, 8.0, D["a"], 0.5)
    b = st.slider("Sensitivity Beta(·, b)", 0.5, 8.0, D["b"], 0.5)
    st.header("Game parameters")
    L = st.slider("Base privacy loss L", 1.0, 20.0, D["L"], 0.5)
    beta = st.slider("Service benefit β (network effect)", 0.0, 20.0, D["beta"], 0.5)
    gamma = st.slider("Risk growth γ", 0.0, 3.0, D["gamma"], 0.1)
    v = st.slider("Org. value per user v", 1.0, 20.0, D["v"], 0.5)
    r = st.slider("Reward r", 0.0, 20.0, D["r"], 0.25)

p = dict(N=N, a=a, b=b, L=L, beta=beta, gamma=gamma, v=v, r=r)
theta = m.make_users(N, seed, a, b)
o = m.outcome(theta, r, p)
sweep = m.sweep_rewards(theta, p)
best = sweep.loc[sweep.profit.idxmax()]

c = st.columns(4)
c[0].metric("Equilibrium participation", f"{o['participation']:.1%}")
c[1].metric("Org. profit", f"{o['profit']:,.0f}")
c[2].metric("Avg user utility", f"{o['avg_utility']:.2f}")
c[3].metric("Profit-maximizing reward", f"{best.reward:.2f}", f"{best.participation:.0%} participate")

t1, t2, t3, t4, t5 = st.tabs(["Nash equilibrium", "Reward strategy", "Users: privacy vs reward",
                              "Participation prediction", "Mixed strategies"])

with t1:
    roots, x, g = m.equilibria(theta, r, p)
    fig = go.Figure()
    fig.add_scatter(x=x, y=g, name="Best response F(T(x))", line=dict(width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="45° line (x = F(T(x)))", line=dict(dash="dash", color="gray"))
    for xe, st_ in roots:
        fig.add_scatter(x=[xe], y=[xe], mode="markers", name="Stable NE" if st_ else "Unstable NE",
                        marker=dict(size=14, symbol="circle" if st_ else "x", color="green" if st_ else "red"))
    fig.update_layout(xaxis_title="Fraction of others sharing (x)", yaxis_title="Fraction who want to share",
                      height=450)
    st.plotly_chart(fig, use_container_width=True)
    st.write("**Equilibria found:** " + ", ".join(f"{xe:.1%} ({'stable' if s else 'unstable'})" for xe, s in roots))
    st.info("A user shares iff sensitivity < T(x) = (r + βx) / (L(1 + γx)). Where the best-response curve crosses "
            "the 45° line, no user wants to deviate — a Nash equilibrium. Multiple stable crossings mean "
            "coordination matters; we report the highest (Pareto-better) one.")

with t2:
    fig = go.Figure()
    fig.add_scatter(x=sweep.reward, y=sweep.profit, name="Org. profit", yaxis="y")
    fig.add_scatter(x=sweep.reward, y=sweep.participation, name="Participation", yaxis="y2")
    fig.add_scatter(x=sweep.reward, y=sweep.avg_utility, name="Avg user utility (satisfaction)", yaxis="y3")
    fig.add_vline(x=best.reward, line_dash="dot", annotation_text="best reward")
    fig.update_layout(height=450, xaxis_title="Reward r", xaxis_domain=[0, 0.88],
                      yaxis=dict(title="Profit"),
                      yaxis2=dict(title="Participation", overlaying="y", side="right", range=[0, 1]),
                      yaxis3=dict(title="Avg utility", overlaying="y", side="right", anchor="free", position=1.0))
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Stackelberg solution:** the organization anticipates user equilibrium and picks "
             f"**r\\* = {best.reward:.2f}** → profit {best.profit:,.0f}, participation {best.participation:.1%}, "
             f"average user utility {best.avg_utility:.2f}.")
    st.dataframe(sweep.round(3), use_container_width=True, height=250)

with t3:
    df, _ = m.user_table(theta, r, p)
    fig = go.Figure()
    for flag, name, col in [(True, "Shares", "#2a9d8f"), (False, "Declines", "#e76f51")]:
        d = df[df.shares_data == flag].sample(min(800, (df.shares_data == flag).sum()), random_state=1) \
            if (df.shares_data == flag).any() else df.iloc[:0]
        fig.add_scatter(x=d.sensitivity, y=d.min_reward_needed, mode="markers", name=name,
                        marker=dict(color=col, size=5, opacity=.7))
    fig.add_hline(y=r, line_dash="dash", annotation_text=f"offered reward = {r}")
    fig.update_layout(height=420, xaxis_title="Privacy sensitivity θ",
                      yaxis_title="Reward needed to be indifferent")
    st.plotly_chart(fig, use_container_width=True)
    h = go.Figure(go.Histogram(x=df.sensitivity, nbinsx=40))
    h.add_vline(x=o["T"], line_dash="dash", annotation_text="equilibrium threshold")
    h.update_layout(height=300, xaxis_title="Sensitivity distribution")
    st.plotly_chart(h, use_container_width=True)
    st.dataframe(df.round(3).head(200), use_container_width=True, height=250)
    st.download_button("Download users CSV", df.to_csv(index=False), "users.csv")

with t4:
    st.subheader("Will a given user participate?")
    th_in = st.slider("User privacy sensitivity θ", 0.0, 1.0, 0.3, 0.01)
    u = r + beta * o["x"] - th_in * L * (1 + gamma * o["x"])
    st.write(f"Predicted: **{'SHARES' if u > 0 else 'DECLINES'}** (utility from sharing = {u:.2f}, "
             f"given equilibrium participation {o['x']:.1%}).")
    st.subheader("Participation map: reward × population sensitivity")
    rewards = np.linspace(0, v * 1.2, 20)
    scales = np.linspace(0.5, 1.5, 15)
    Z = m.participation_heatmap(theta, p, rewards, scales)
    hm = go.Figure(go.Heatmap(z=Z, x=rewards.round(2), y=scales.round(2), colorscale="Viridis",
                              colorbar=dict(title="Participation")))
    hm.update_layout(height=420, xaxis_title="Reward r", yaxis_title="Sensitivity scale (1 = baseline)")
    st.plotly_chart(hm, use_container_width=True)

with t5:
    st.subheader("Mixed-strategy equilibrium")
    th_m = st.slider("Representative user sensitivity θ", 0.0, 1.0, float(np.median(theta)), 0.01, key="mix")
    ps = np.linspace(0, 1, 101)
    U = r + beta * ps - th_m * L * (1 + gamma * ps)
    pstar, denom = m.mixed_strategy(th_m, r, p)
    fig = go.Figure()
    fig.add_scatter(x=ps, y=U, name="U(share | others share w.p. p)")
    fig.add_scatter(x=ps, y=0 * ps, name="U(not share) = 0", line=dict(dash="dash"))
    if 0 < pstar < 1:
        fig.add_scatter(x=[pstar], y=[0], mode="markers", marker=dict(size=14, color="red"), name="Mixed NE p*")
    fig.update_layout(height=400, xaxis_title="Probability others share", yaxis_title="Payoff")
    st.plotly_chart(fig, use_container_width=True)
    if U[0] > 0 and U[-1] > 0:
        st.success("Sharing is a dominant (pure) strategy for this user.")
    elif U[0] < 0 and U[-1] < 0:
        st.error("Not sharing is a dominant (pure) strategy for this user.")
    elif 0 < pstar < 1:
        kind = "stable (congestion-like: more sharers lower the payoff)" if denom < 0 else \
               "unstable (coordination game: it separates two pure equilibria)"
        st.warning(f"Mixed equilibrium: share with probability **p\\* = {pstar:.2f}** — {kind}.")
