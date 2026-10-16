"""Generates deck.json from the engine so every number in the PPT matches the app."""
import json, numpy as np, leak_engine as le
A = le.get(); k = A["k"]; m = A["m"]; C = A["camp"]; L = A["lab"]; P = A["products"]; ac = A["actions"]; S = A["stock"]
def f(x):
    x = float(x); s = "-" if x < 0 else ""; x = abs(x)
    return f"{s}{x/1e6:.2f}M" if x >= 1e6 else f"{s}{x/1e3:.0f}K" if x >= 1e4 else f"{s}{x:,.0f}"
comp = m[m.status == "Completed"]
cat_loss = comp[comp.margin < 0].groupby("category").margin.sum().abs().sort_values(ascending=False).round(0)
worst = P.sort_values("margin").head(8)
nb = int(P.below_cost.sum()); top20 = P.sort_values("margin").head(20).margin.sum() / P[P.margin < 0].margin.sum()
best = L["best"]; seg = A["seg_prof"]
slides = [
 dict(id=1, type="title", title="LeakFinder", subtitle="Finds where a business loses money, proves it, and says what to fix first",
      meta="Track 4: Business | The Money Leak | Datathon 2026 | [Team name: members + reg no]"),
 dict(id=2, type="bullets_cards", title="The Problem", subtitle="Sales look healthy. Profit does not.",
      cards=[dict(head="Who", body=["Business owners and retail managers", "Category and operations managers"]),
             dict(head="Pain", body=[f"{f(k['booked_rev'])} booked, only {f(k['realized_rev'])} realized", f"{k['failed_rev_pct']:.1%} of revenue is cancelled or returned", f"Realized margin is just {k['realized_margin_pct']:.1%}"]),
             dict(head="Gap today", body=["Signals sit in 5 separate tables", "Dashboards show sales, not causes", "Managers cannot tell real leaks from noise"])]),
 dict(id=3, type="stats_chart", title="What the Data Says", subtitle="18,000 orders, 250 products, 3,000 customers, 1,200 campaigns",
      stats=[dict(label="Products priced below cost", value=f"{nb} of {len(P)}"), dict(label="Lost on below-cost sales", value=f(k["list_below_cost_loss"])),
             dict(label="Cancelled + returned revenue", value=f(k["failed_rev"])), dict(label="Marketing spend vs realized margin", value=f"{C['spend_vs_margin']:.1f}x")],
      chart=dict(type="bar", title="Money leaks (revenue or margin at stake)", labels=["Sold below cost", "Cancelled", "Returned", "Stuck pending 30d+"],
                 values=[round(abs(k["list_below_cost_loss"])), round(k["cancelled_rev"]), round(k["returned_rev"]), round(k["stale_pending_rev"])])),
 dict(id=4, type="usp", title="Our Solution and USP", usp="The only tool that separates real leaks from noise: every alert comes with proof, a confidence level, and a ranked action.",
      cards=[dict(head="What it does", body=["Finds each leak in revenue, price, stock and ads", "Ranks fixes by money at stake", "Explains why each item is flagged"]),
             dict(head="Why it is different", body=["Tests 10 possible drivers; says 'noise' when it is noise", "Refuses to sell a fake risk score", "All ML written from scratch, scored on future data"]),
             dict(head="Who uses it", body=["Manager opens Command Center", "Sees top 5 actions with impact", "Downloads repricing list and action plan"])]),
 dict(id=5, type="mvp", title="MVP: What Works Today", subtitle="Built and running on the provided data",
      steps=[dict(n="1", head="Data", body=f"5 tables joined. {len(m):,} orders checked. 7 data issues found and documented."),
             dict(n="2", head="Models", body="From-scratch logistic, random forest, gradient boosting, KMeans, robust anomaly alerts, regression forecast."),
             dict(n="3", head="App", body="8 tabs: Command Center, Investigate, Pricing, Stock, Campaigns, Forecast, ML Lab, Action Plan.")],
      demo="Demo path (60s): Command Center > pricing leak > what-if reprice > Investigate shows 'no real driver' > campaigns > action plan"),
 dict(id=6, type="chart_insights", title="The Biggest Leak: Price Below Cost", subtitle="Not customers. Not channels. The price list.",
      chart=dict(type="bar", title="Loss on below-cost completed orders, by category", labels=list(cat_loss.index), values=[float(v) for v in cat_loss.values]),
      insights=[dict(head="Loss", body=f"{f(abs(k['list_below_cost_loss']))} lost on completed orders because list price is under unit cost"),
                dict(head="Concentrated", body=f"The 20 worst products cause {top20:.0%} of the loss: fix these first"),
                dict(head="Not discounts", body=f"Discounts add only {f(abs(k['discount_only_loss']))} extra loss; unit_price already includes discount")]),
 dict(id=7, type="table", title="Honest ML: Failed Orders Are Not Predictable Here", subtitle="Time-aware test: train Jan-Sep, test Oct-Dec on unseen orders",
      table=dict(columns=["Model (from scratch)", "AUC", "Accuracy", "Always-'no' baseline", "Top-decile lift"],
                 rows=[[r.model, f"{r.AUC:.3f}", f"{r.accuracy:.3f}", f"{r.majority_baseline_acc:.3f}", f"{r.top_decile_lift:.2f}"] for r in L["failure"].itertuples()]),
      takeaway=f"AUC near 0.50 means a coin flip. No category, region, channel, segment, discount or time group fails more than chance explains (0 of 10 drivers real). We report this instead of a fake risk score. Fix: record cancel/return reasons."),
 dict(id=8, type="bullets_cards", title="Technical Approach", subtitle="From-scratch ML, validated and explained",
      cards=[dict(head="Pipeline", body=["Join 5 tables; unit_price already net of discount", "Completed = realized; cancelled/returned = not", "Time-aware split; robust (median/MAD) alerts"]),
             dict(head="Models (NumPy, written by us)", body=["Logistic, random forest, gradient boosting", "KMeans customer segments (4)", "Trend + seasonality regression forecast"]),
             dict(head="Explainability", body=["Every flag shows evidence + p-value", "Stock risk = stock, demand, restock time", "Confidence label on every action"])]),
 dict(id=9, type="actions", title="Action Plan", subtitle="Ranked by money at stake and how sure we are",
      table=dict(columns=["#", "Action", "Impact (up to)", "Confidence"], rows=[[str(int(r["rank"])), r["action"], f(r["impact"]), r["confidence"]] for _, r in ac.iterrows()]),
      takeaway="Impacts overlap and are upper bounds. Do not add them up."),
 dict(id=10, type="bullets_cards", title="Data Quality and Assumptions", subtitle="Documented before recommending anything",
      cards=[dict(head="Issues found", body=[f"{nb} products have list price below cost", f"{C['invalid_rows']} impossible campaign rows (clicks > impressions or conversions > clicks)", "Stockout flag disagrees with stock level"]),
             dict(head="Assumptions", body=["Discount already inside unit_price", "Cancelled/returned revenue not realized", "Pending older than 30 days = stuck", "Currency not stated"]),
             dict(head="Limits", body=["Synthetic data: no real-world causes claimed", "Spend does not predict conversions (r = %.2f)" % C["corr_spend_conv"], "Costs need finance confirmation"])]),
 dict(id=11, type="bullets_cards", title="Impact, Deployment and Round 2", subtitle="Ships as one public link",
      cards=[dict(head="Impact", body=[f"Repricing: up to {f(ac.iloc[0].impact)} margin gain", f"Stuck orders: {f(ac.iloc[1].impact)} recoverable", "Marketing: audit tracking, test budget moves"]),
             dict(head="Deployment", body=["Streamlit app on a public URL", "Repo with README and from-scratch ML", "Swap in new CSVs to refresh"]),
             dict(head="By 4 PM", body=["Polish UI and rehearse 60s demo", "Final checks on every number", "Submit repo + model details"])]),
]
json.dump(dict(meta=dict(title="LeakFinder", event="Datathon 2026", track="Business: The Money Leak", team="[Team name]"), slides=slides), open("deck.json", "w"), indent=1, default=str)
print("slides:", len(slides)); print(json.dumps(slides[2]["stats"], indent=0)); print(cat_loss.to_dict(), round(top20, 2))
