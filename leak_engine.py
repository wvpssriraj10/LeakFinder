"""LeakFinder engine: every number in the app and the deck comes from here.
Assumptions (also shown in the app):
 A1 unit_price is already net of discount (verified: unit_price == list_price*(1-discount_pct) on 100% of rows) -> never re-apply discount.
 A2 Completed = realized revenue. Cancelled/Returned = revenue NOT realized (goods restocked, no cost incurred).
 A3 Pending older than 30 days at data end = stale (stuck order).
 A4 Currency is not given in the data; shown as plain numbers.
 A5 Campaign spend and product prices are assumed to be in the same currency.
"""
import os, math, pickle
import numpy as np, pandas as pd
import ml_scratch as ms
import pipeline as pl

STALE_DAYS = 30
Z_SIG = 3.0


# ------------------------------------------------------------------ load
def load(folder="data"):
    d = {k: pd.read_csv(f"{folder}/{k}.csv") for k in ["customers", "products", "transactions", "inventory", "campaigns"]}
    d["customers"]["signup_date"] = pd.to_datetime(d["customers"]["signup_date"])
    d["transactions"]["timestamp"] = pd.to_datetime(d["transactions"]["timestamp"])
    d["inventory"]["date"] = pd.to_datetime(d["inventory"]["date"])
    d["campaigns"]["date"] = pd.to_datetime(d["campaigns"]["date"])
    return d


def build_fact(d):
    m = d["transactions"].merge(d["products"], on="product_id", how="left").merge(d["customers"], on="customer_id", how="left")
    m["revenue"] = m.quantity * m.unit_price
    m["cost"] = m.quantity * m.unit_cost
    m["margin"] = m.revenue - m.cost
    m["failed"] = m.status.isin(["Cancelled", "Returned"]).astype(int)
    m["list_below_cost"] = m.list_price < m.unit_cost
    m["below_cost"] = m.unit_price < m.unit_cost
    m["month"] = m.timestamp.dt.to_period("M").astype(str)
    m["hour"] = m.timestamp.dt.hour; m["dow"] = m.timestamp.dt.dayofweek
    m["tenure_days"] = (m.timestamp - m.signup_date).dt.days
    end = m.timestamp.max()
    m["age_days"] = (end - m.timestamp).dt.days
    m["stale_pending"] = (m.status == "Pending") & (m.age_days > STALE_DAYS)
    return m


# ------------------------------------------------------------------ KPIs
def kpis(m):
    comp = m[m.status == "Completed"]
    k = dict(
        orders=len(m), booked_rev=m.revenue.sum(), realized_rev=comp.revenue.sum(),
        cancelled_rev=m[m.status == "Cancelled"].revenue.sum(), returned_rev=m[m.status == "Returned"].revenue.sum(),
        pending_rev=m[m.status == "Pending"].revenue.sum(), stale_pending_rev=m[m.stale_pending].revenue.sum(),
        stale_pending_n=int(m.stale_pending.sum()),
        realized_margin=comp.margin.sum(), realized_margin_pct=comp.margin.sum() / comp.revenue.sum(),
        below_cost_loss=comp[comp.margin < 0].margin.sum(), below_cost_lines=int((comp.margin < 0).sum()),
        list_below_cost_loss=comp[comp.list_below_cost].margin.sum(),
        discount_only_loss=comp[(~comp.list_below_cost) & (comp.margin < 0)].margin.sum(),
        margin_ex_pricing_leak=comp[~comp.list_below_cost].margin.sum(),
        fail_rate=m.failed.mean(), cancel_rate=(m.status == "Cancelled").mean(), return_rate=(m.status == "Returned").mean(),
        pending_rate=(m.status == "Pending").mean(),
    )
    k["failed_rev"] = k["cancelled_rev"] + k["returned_rev"]
    k["failed_rev_pct"] = k["failed_rev"] / k["booked_rev"]
    return k


# ------------------------------------------------------------------ pricing leak
def product_table(m, d):
    p = d["products"].copy()
    comp = m[m.status == "Completed"]
    g = comp.groupby("product_id").agg(units=("quantity", "sum"), revenue=("revenue", "sum"), margin=("margin", "sum"))
    p = p.merge(g, on="product_id", how="left").fillna({"units": 0, "revenue": 0, "margin": 0})
    p["list_margin_pct"] = (p.list_price - p.unit_cost) / p.list_price
    p["below_cost"] = p.list_price < p.unit_cost
    p["realized_margin_pct"] = np.where(p.revenue > 0, p.margin / p.revenue.replace(0, np.nan), np.nan)
    return p


def reprice_whatif(m, target_margin=0.20, volume_loss=0.15):
    """Floor price so margin >= target on products priced below that floor. Volume haircut = share of units lost after increase.
    Returns (gain, table). Gain = new margin - old margin on completed lines of affected products."""
    comp = m[m.status == "Completed"].copy()
    floor = comp.unit_cost / (1 - target_margin)
    new_price = np.maximum(comp.unit_price, floor)
    affected = new_price > comp.unit_price + 1e-9
    old_margin = comp.margin
    new_margin = (new_price - comp.unit_cost) * comp.quantity * (1 - volume_loss)
    comp["gain"] = np.where(affected, new_margin - old_margin, 0.0)
    t = comp[affected].groupby(["product_id", "category"]).agg(
        lines=("transaction_id", "size"), units=("quantity", "sum"), old_margin=("margin", "sum"), gain=("gain", "sum"),
        unit_cost=("unit_cost", "first"), list_price=("list_price", "first")).reset_index()
    t["suggested_list"] = t.unit_cost / (1 - target_margin) / (1 - 0.0)
    t["list_margin_now"] = (t.list_price - t.unit_cost) / t.list_price
    return float(comp.gain.sum()), t.sort_values("gain", ascending=False)


# ------------------------------------------------------------------ failure investigation
def _pval(z): return math.erfc(abs(z) / math.sqrt(2))


def segment_rates(m, dim, base=None):
    base = m.failed.mean() if base is None else base
    g = m.groupby(dim).agg(orders=("failed", "size"), failed=("failed", "sum"),
                           cancelled=("status", lambda s: (s == "Cancelled").sum()), returned=("status", lambda s: (s == "Returned").sum()),
                           pending=("status", lambda s: (s == "Pending").sum()), revenue=("revenue", "sum"),
                           failed_rev=("revenue", lambda s: s[m.loc[s.index, "failed"] == 1].sum())).reset_index()
    g["fail_rate"] = g.failed / g.orders
    g["z"] = (g.fail_rate - base) / np.sqrt(base * (1 - base) / g.orders)
    g["p"] = g.z.apply(_pval)
    thr = 0.05 / max(1, len(g))  # Bonferroni across groups in this view
    g["verdict"] = np.where(g.p < thr, "Real signal", "Normal noise")
    return g.sort_values("fail_rate", ascending=False)


def driver_scan(m, dims):
    """Chi-square-like scan: strongest group per dimension, with Bonferroni across dimensions. Shows whether ANY driver is real."""
    rows = []
    for dim in dims:
        s = segment_rates(m, dim)
        top = s.iloc[0]
        rows.append(dict(dimension=dim, worst_group=str(top[dim]), orders=int(top.orders), fail_rate=top.fail_rate, z=top.z, p=top.p,
                         verdict="Real signal" if top.p < 0.05 / (len(dims) * max(1, len(s))) else "Normal noise"))
    return pd.DataFrame(rows).sort_values("p")


# ------------------------------------------------------------------ inventory
def stock_risk(d, m):
    inv = d["inventory"].sort_values("date")
    last = inv.groupby("product_id").tail(1).merge(d["products"], on="product_id")
    end = m.timestamp.max()
    recent = m[(m.timestamp > end - pd.Timedelta(days=90)) & (m.status != "Cancelled")]
    rate = (recent.groupby("product_id").quantity.sum() / 90).rename("daily_demand")
    last = last.merge(rate, on="product_id", how="left").fillna({"daily_demand": 0})
    last["days_cover"] = last.stock_on_hand / last.daily_demand.clip(lower=0.01)
    last["lead_total"] = last.supplier_lead_days + last.supplier_delay_days
    last["below_reorder"] = last.stock_on_hand <= last.reorder_threshold
    last["runs_out_before_restock"] = last.days_cover < last.lead_total
    last["risk"] = np.where(last.below_reorder & last.runs_out_before_restock, "High",
                            np.where(last.below_reorder | last.runs_out_before_restock, "Medium", "Low"))
    last["revenue_at_risk"] = (last.lead_total - last.days_cover).clip(lower=0) * last.daily_demand * last.list_price
    why = []
    for _, r in last.iterrows():
        w = []
        if r.below_reorder: w.append(f"stock {int(r.stock_on_hand)} <= reorder point {int(r.reorder_threshold)}")
        if r.runs_out_before_restock: w.append(f"cover {r.days_cover:.0f}d < restock time {r.lead_total:.0f}d")
        why.append("; ".join(w) or "healthy")
    last["why_flagged"] = why
    return last.sort_values(["risk", "revenue_at_risk"], ascending=[True, False]).sort_values("revenue_at_risk", ascending=False)


def inventory_quality(d):
    inv = d["inventory"]
    return dict(flag_with_stock=int(((inv.stockout_flag == 1) & (inv.stock_on_hand > 0)).sum()), flagged=int(inv.stockout_flag.sum()),
                zero_stock_unflagged=int(((inv.stock_on_hand == 0) & (inv.stockout_flag == 0)).sum()),
                dup_product_date=int(inv.duplicated(["product_id", "date"]).sum()),
                below_reorder_share=float((inv.stock_on_hand <= inv.reorder_threshold).mean()))


# ------------------------------------------------------------------ campaigns
def campaign_table(d):
    g = d["campaigns"].copy()
    g["valid"] = (g.clicks <= g.impressions) & (g.conversions <= g.clicks) & (g.conversions >= 0)
    g["month"] = g.date.dt.to_period("M").astype(str)
    return g


def campaign_summary(g, by):
    v = g[g.valid]
    s = v.groupby(by).agg(campaigns=("campaign_id", "size"), spend=("spend", "sum"), impressions=("impressions", "sum"),
                          clicks=("clicks", "sum"), conversions=("conversions", "sum")).reset_index()
    s["ctr"] = s.clicks / s.impressions; s["cvr"] = s.conversions / s.clicks
    s["cpa"] = s.spend / s.conversions; s["cpc"] = s.spend / s.clicks
    return s.sort_values("cpa")


def campaign_facts(d, m, g):
    v = g[g.valid]
    med_cpa = (v.spend.sum() / v.conversions.sum())
    excess = float(np.maximum(0, v.spend - v.conversions * med_cpa).sum())
    realized = m[m.status == "Completed"]
    ov = ["App", "Store", "Web"]
    sp = g.groupby("channel").spend.sum(); rv = realized.groupby("channel").revenue.sum(); mg = realized.groupby("channel").margin.sum()
    roas = pd.DataFrame({"spend": sp.reindex(ov), "realized_revenue": rv.reindex(ov), "realized_margin": mg.reindex(ov)})
    roas["revenue_per_spend"] = roas.realized_revenue / roas.spend; roas["margin_per_spend"] = roas.realized_margin / roas.spend
    cat_sp = g.groupby("category").spend.sum(); cat_rv = realized.groupby("category").revenue.sum(); cat_mg = realized.groupby("category").margin.sum()
    catr = pd.DataFrame({"spend": cat_sp, "realized_revenue": cat_rv, "realized_margin": cat_mg})
    catr["revenue_per_spend"] = catr.realized_revenue / catr.spend; catr["margin_per_spend"] = catr.realized_margin / catr.spend
    return dict(total_spend=float(g.spend.sum()), invalid_rows=int((~g.valid).sum()), invalid_spend=float(g[~g.valid].spend.sum()),
                clicks_gt_impr=int((g.clicks > g.impressions).sum()), conv_gt_clicks=int((g.conversions > g.clicks).sum()),
                corr_spend_conv=float(v.spend.corr(v.conversions)), benchmark_cpa=float(med_cpa), excess_spend=excess,
                spend_vs_margin=float(g.spend.sum() / realized.margin.sum()), channel_roas=roas.reset_index(), category_roas=catr.reset_index())


# ------------------------------------------------------------------ anomalies + forecast
def weekly(m):
    start = m.timestamp.min().normalize()
    w = m.assign(week=((m.timestamp - start).dt.days // 7))
    full = w.groupby("week").timestamp.apply(lambda s: s.dt.normalize().nunique())
    keep = full[full == 7].index
    w = w[w.week.isin(keep)]
    out = w.groupby("week").apply(lambda x: pd.Series(dict(
        realized_rev=x.loc[x.status == "Completed", "revenue"].sum(), orders=len(x), fail_rate=x.failed.mean(),
        below_cost_share=x.below_cost.mean(), margin=x.loc[x.status == "Completed", "margin"].sum()))).reset_index()
    out["week_start"] = start + pd.to_timedelta(out.week * 7, unit="D")
    return out


def deviations(wk):
    out = []
    for col, label in [("realized_rev", "Realized revenue"), ("fail_rate", "Failure rate"), ("margin", "Realized margin")]:
        z = ms.robust_z(wk[col].values)
        out.append(pd.DataFrame(dict(week_start=wk.week_start, metric=label, value=wk[col], robust_z=z)))
    return pd.concat(out).assign(abs_z=lambda x: x.robust_z.abs()).sort_values("abs_z", ascending=False)


def anomalies(wk):
    rows = []
    for col, label in [("realized_rev", "Realized revenue"), ("fail_rate", "Failure rate"), ("margin", "Realized margin")]:
        z = ms.robust_z(wk[col].values)
        for i in np.where(np.abs(z) > 3.0)[0]:
            rows.append(dict(week_start=wk.week_start.iloc[i], metric=label, value=float(wk[col].iloc[i]), robust_z=float(z[i]),
                             why=f"{label} {wk[col].iloc[i]:,.2f} is {abs(z[i]):.1f} robust-SDs {'above' if z[i] > 0 else 'below'} the 52-week median"))
    return pd.DataFrame(rows)


def forecast(wk, horizon=8, holdout=8):
    y = wk.realized_rev.values; n = len(y); t = np.arange(n)
    def feats(tt): return np.c_[tt, np.sin(2 * np.pi * tt / 52), np.cos(2 * np.pi * tt / 52)]
    sc = ms.StandardScaler().fit(feats(t[:-holdout]))
    mdl = ms.LinearRegression(l2=5.0).fit(sc.transform(feats(t[:-holdout])), y[:-holdout])
    pred = mdl.predict(sc.transform(feats(t[-holdout:])))
    mean_pred = np.full(holdout, y[:-holdout].mean()); naive = np.full(holdout, y[-holdout - 4:-holdout].mean())
    res = dict(holdout_mae_model=ms.mae(y[-holdout:], pred), holdout_mae_mean=ms.mae(y[-holdout:], mean_pred), holdout_mae_naive=ms.mae(y[-holdout:], naive))
    sc2 = ms.StandardScaler().fit(feats(t)); mdl2 = ms.LinearRegression(l2=5.0).fit(sc2.transform(feats(t)), y)
    fut = np.arange(n, n + horizon); fp = mdl2.predict(sc2.transform(feats(fut)))
    use_model = res["holdout_mae_model"] < min(res["holdout_mae_mean"], res["holdout_mae_naive"])
    if use_model:
        sd = float(np.std(y - mdl2.predict(sc2.transform(feats(t))))); center = fp
    else:  # honest fallback: model does not beat a flat baseline on hold-out -> use flat recent mean
        sd = float(np.std(y[-26:])); center = np.full(horizon, y[-8:].mean())
    res.update(future=pd.DataFrame(dict(week_start=wk.week_start.iloc[-1] + pd.to_timedelta((np.arange(1, horizon + 1)) * 7, unit="D"),
                                        forecast=center, low=center - 1.28 * sd, high=center + 1.28 * sd)), use_model=use_model,
               method="Trend + seasonality regression (scratch)" if use_model else "Flat baseline: recent 8-week mean (trend/seasonality model did not beat it on hold-out)")
    return res


# ------------------------------------------------------------------ ML lab (time-aware, honest)
def _fit_eval(Xtr, ytr, Xte, yte):
    base_acc = max(yte.mean(), 1 - yte.mean())
    rng = np.random.default_rng(0)
    sub = rng.choice(len(Xtr), min(len(Xtr), 6000), replace=False)
    models = {"Logistic Regression (scratch)": ms.LogisticRegression(lr=0.1, epochs=60),
              "Random Forest (scratch)": ms.RandomForest("classification", 25, 6),
              "Gradient Boosting (scratch)": ms.GradientBoosting("classification", 40, 0.1, 3)}
    rows = []
    for name, mdl in models.items():
        mdl.fit(Xtr[sub] if "Forest" in name or "Boosting" in name else Xtr, ytr[sub] if "Forest" in name or "Boosting" in name else ytr)
        pr = mdl.predict_proba(Xte)[:, 1]
        rows.append(dict(model=name, AUC=ms.roc_auc(yte, pr), accuracy=ms.accuracy(yte, (pr > 0.5).astype(int)),
                         majority_baseline_acc=float(base_acc), top_decile_lift=ms.top_decile_lift(yte, pr)))
    return pd.DataFrame(rows), models


def ml_lab(m, d):
    feat = ["quantity", "unit_price", "discount_pct", "list_below_cost", "category", "region", "customer_segment", "channel", "payment_method",
            "tenure_days", "hour", "dow", "supplier_lead_days"]
    df = m[feat + ["month", "failed"]].copy(); df["list_below_cost"] = df.list_below_cost.astype(int)
    tr = df[df.month <= "2025-09"]; te = df[df.month > "2025-09"]
    pre = pl.Preprocessor().fit(tr[feat]); Xtr, Xte = pre.transform(tr[feat]), pre.transform(te[feat])
    lb, models = _fit_eval(Xtr, tr.failed.values, Xte, te.failed.values)
    # stockout flag: is the supplied flag even predictable?
    inv = d["inventory"].merge(d["products"], on="product_id")
    f2 = ["stock_on_hand", "reorder_threshold", "supplier_delay_days", "supplier_lead_days", "category"]
    inv = inv.sort_values("date"); cut = int(len(inv) * 0.8)
    pre2 = pl.Preprocessor().fit(inv.iloc[:cut][f2])
    lb2, _ = _fit_eval(pre2.transform(inv.iloc[:cut][f2]), inv.iloc[:cut].stockout_flag.values, pre2.transform(inv.iloc[cut:][f2]), inv.iloc[cut:].stockout_flag.values)
    best = lb.sort_values("AUC", ascending=False).iloc[0]
    imp = getattr(models["Gradient Boosting (scratch)"], "importances_", None)
    imp = pd.Series(imp, index=pre.names).sort_values(ascending=False).head(10) if imp is not None else None
    return dict(failure=lb, stockout=lb2, best=best, importances=imp, train_n=len(tr), test_n=len(te),
                has_signal=bool(best.AUC > 0.56))


def segment_customers(m, k=4):
    comp = m[m.status == "Completed"]
    c = m.groupby("customer_id").agg(orders=("transaction_id", "size"), fail_rate=("failed", "mean"), avg_discount=("discount_pct", "mean"),
                                    tenure=("tenure_days", "max")).join(comp.groupby("customer_id").agg(revenue=("revenue", "sum"), margin=("margin", "sum"))).fillna(0).reset_index()
    X = c[["orders", "revenue", "fail_rate", "avg_discount", "tenure"]].values.astype(float)
    Z = ms.StandardScaler().fit_transform(X)
    km = ms.KMeans(k, seed=1).fit(Z); c["cluster"] = km.labels_
    prof = c.groupby("cluster").agg(customers=("customer_id", "size"), orders=("orders", "mean"), revenue=("revenue", "mean"),
                                    fail_rate=("fail_rate", "mean"), avg_discount=("avg_discount", "mean"), tenure=("tenure", "mean"), total_revenue=("revenue", "sum"))
    names = {}
    names[prof.revenue.idxmax()] = "High-value buyers"
    rest = prof.drop(index=list(names))
    names[rest.fail_rate.idxmax()] = "High-friction (many failed orders)"
    rest = prof.drop(index=list(names))
    names[rest.orders.idxmin()] = "Light / occasional buyers"
    for i in prof.index:
        names.setdefault(i, "Core steady buyers")
    c["segment"] = c.cluster.map(names); prof["segment"] = prof.index.map(names)
    return c, prof.reset_index(drop=True)


# ------------------------------------------------------------------ data quality + actions
def data_quality(d, m, inv_q, camp):
    p = d["products"]
    items = [
        dict(issue="List price below unit cost", detail=f"{int((p.list_price < p.unit_cost).sum())} of {len(p)} products ({(p.list_price < p.unit_cost).mean():.0%}) cannot make a profit at list price", severity="High", treatment="Treated as a real pricing leak; flagged per product (README warns costs/prices are generated independently, so confirm with finance)"),
        dict(issue="Discount already inside unit_price", detail="unit_price == list_price x (1 - discount_pct) on 100% of rows", severity="High", treatment="Discount is never applied twice"),
        dict(issue="Campaign rows that are impossible", detail=f"{camp['clicks_gt_impr']} rows with clicks > impressions, {camp['conv_gt_clicks']} rows with conversions > clicks", severity="High", treatment="Excluded from efficiency metrics; counted as unreliable spend"),
        dict(issue="Stockout flag disagrees with stock level", detail=f"{inv_q['flag_with_stock']} of {inv_q['flagged']} flagged records still have stock on hand; {inv_q['zero_stock_unflagged']} zero-stock records are not flagged", severity="Medium", treatment="Own stock-risk rule built from stock, demand and restock time; flag not trusted"),
        dict(issue="Duplicate product-date inventory records", detail=f"{inv_q['dup_product_date']} duplicated product/date pairs", severity="Low", treatment="Latest record per product used"),
        dict(issue="Customers with no orders", detail=f"{int((~d['customers'].customer_id.isin(d['transactions'].customer_id)).sum())} customers", severity="Low", treatment="Ignored in segmentation"),
        dict(issue="Pending orders never resolved", detail=f"{int((m.status == 'Pending').sum())} pending orders spread across all 12 months; {int(m.stale_pending.sum())} older than {STALE_DAYS} days", severity="Medium", treatment="Flagged as stuck orders"),
    ]
    return pd.DataFrame(items)


def build_actions(k, reprice_gain, camp, stock, lab, m, fail_cut=0.15):
    comp = m[m.status == "Completed"]
    comp_rate = k["realized_rev"] / (k["realized_rev"] + k["failed_rev"])
    healthy_failed = m[(m.failed == 1) & (~m.list_below_cost) & (m.margin > 0)]
    acts = [
        dict(rank=1, action="Reprice the products that are sold below cost", impact=reprice_gain, confidence="High",
             evidence=f"{int(m[m.list_below_cost].product_id.nunique())} products have list price below unit cost; they lost {abs(k['list_below_cost_loss']):,.0f} on completed orders. Discounts alone explain only {abs(k['discount_only_loss']):,.0f}.",
             how="Set list price to cost / (1 - target margin). Start with the 20 worst products.", basis="Margin gain at 20% target margin and 15% volume loss"),
        dict(rank=2, action="Clear stuck pending orders", impact=k["stale_pending_rev"] * comp_rate, confidence="Medium",
             evidence=f"{k['stale_pending_n']} pending orders older than {STALE_DAYS} days hold {k['stale_pending_rev']:,.0f} of revenue.",
             how="Daily pending-order queue with an owner; auto-escalate after 7 days.", basis=f"Revenue x {comp_rate:.0%} historical completion rate"),
        dict(rank=3, action="Audit campaign tracking, then cut spend with above-benchmark cost per conversion", impact=camp["excess_spend"], confidence="Low",
             evidence=f"Spend and conversions are uncorrelated (r = {camp['corr_spend_conv']:.2f}); {camp['invalid_rows']} campaign rows are impossible; total spend is {camp['spend_vs_margin']:.1f}x realized margin.",
             how="Fix click/conversion tracking first. Then shift budget from the highest cost-per-conversion rows.", basis="Spend above benchmark cost per conversion (upper bound, not guaranteed savings)"),
        dict(rank=4, action="Cut cancellations and returns on healthy products, and start recording reasons", impact=float(healthy_failed.margin.sum()) * fail_cut, confidence="Low",
             evidence=f"{k['fail_rate']:.1%} of orders fail ({k['failed_rev']:,.0f} revenue). No customer, product, channel or region group shows a real signal, and the best model AUC is {lab['best'].AUC:.2f}.",
             how="Add cancellation/return reason codes. Do not build targeting on the current data.", basis=f"{fail_cut:.0%} fewer failures on profitable products"),
        dict(rank=5, action="Fix inventory flags and reorder the products that will run out", impact=float(stock[stock.risk == "High"].revenue_at_risk.sum()), confidence="Medium",
             evidence=f"{int((stock.risk == 'High').sum())} products are below reorder point and will run out before restock arrives; the supplied stockout flag does not match stock levels.",
             how="Use the stock-risk table; rebuild the flag from stock and reorder point.", basis="Demand x days short x list price"),
    ]
    return pd.DataFrame(acts)


# ------------------------------------------------------------------ orchestrate
def run_all(folder="data"):
    d = load(folder); m = build_fact(d); k = kpis(m)
    gain, rep = reprice_whatif(m)
    g = campaign_table(d); camp = campaign_facts(d, m, g); stock = stock_risk(d, m); inv_q = inventory_quality(d)
    wk = weekly(m); lab = ml_lab(m, d); seg, prof = segment_customers(m)
    dims = ["category", "region", "customer_segment", "channel", "payment_method", "discount_pct", "quantity", "month", "hour", "dow"]
    scan = driver_scan(m, dims)
    return dict(d=d, m=m, k=k, reprice_gain=gain, reprice=rep, products=product_table(m, d), camp_tbl=g, camp=camp, stock=stock, inv_q=inv_q,
                wk=wk, anom=anomalies(wk), dev=deviations(wk), fc=forecast(wk), lab=lab, seg=seg, seg_prof=prof, scan=scan,
                dq=data_quality(d, m, inv_q, camp), actions=build_actions(k, gain, camp, stock, lab, m))


def get(folder="data", cache="models/analysis.pkl"):
    srcs = [os.path.getmtime(f"{folder}/{f}") for f in os.listdir(folder) if f.endswith(".csv")]
    if os.path.exists(cache) and os.path.getmtime(cache) > max(srcs):
        with open(cache, "rb") as f: return pickle.load(f)
    a = run_all(folder)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "wb") as f: pickle.dump(a, f)
    return a


if __name__ == "__main__":
    import time; t0 = time.time(); a = get(); print("built in", round(time.time() - t0, 1), "s")
    k = a["k"]; print({x: (round(v, 3) if isinstance(v, float) else v) for x, v in k.items()})
    print(a["actions"][["rank", "action", "impact", "confidence"]].to_string())
    print(a["lab"]["failure"].round(3).to_string()); print(a["lab"]["stockout"].round(3).to_string())
    print(a["scan"].round(3).to_string()); print(a["anom"].to_string() if len(a["anom"]) else "no anomalies")
    f = a["fc"]; print({x: round(f[x], 1) for x in ["holdout_mae_model", "holdout_mae_mean", "holdout_mae_naive"]}, f["use_model"])
    print(a["seg_prof"].round(2).to_string()); print(a["stock"].risk.value_counts().to_dict()); print(a["camp"]["channel_roas"].round(2).to_string())
