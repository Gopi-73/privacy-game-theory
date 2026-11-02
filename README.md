# Game-Theoretic Privacy-Preserving Data Sharing Platform
Run: `pip install -r requirements.txt && streamlit run app.py`

- `model.py` – game logic (utilities, Nash equilibrium solver, Stackelberg reward sweep, mixed strategy)
- `app.py` – Streamlit dashboard (5 tabs)

Model: user i shares iff `r + βx − θᵢL(1+γx) > 0`, x = fraction sharing. Equilibrium: x* = F(T(x*)).
The organization chooses r to maximize `N·x*(r)·(v − r)`.
