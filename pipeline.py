"""Auto pipeline for any tabular dataset. Run:  python pipeline.py data/yourfile.csv target_column
Trains from-scratch models + sklearn baselines, picks best by CV-free holdout, saves models/bundle.pkl"""
import sys, pickle, time
import numpy as np, pandas as pd
import ml_scratch as ms

MAX_ONEHOT = 15


class Preprocessor:
    def fit(self, df):
        self.num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        self.cat = [c for c in df.columns if c not in self.num and df[c].nunique() <= MAX_ONEHOT]
        self.med = {c: df[c].median() for c in self.num}
        self.levels = {c: sorted(df[c].dropna().astype(str).unique()) for c in self.cat}
        self.dropped = [c for c in df.columns if c not in self.num and c not in self.cat]
        Xn = self._raw(df)
        self.scaler = ms.StandardScaler().fit(Xn)
        self.names = self.num + [f"{c}={l}" for c in self.cat for l in self.levels[c]]
        return self

    def _raw(self, df):
        parts = [df[c].fillna(self.med[c]).astype(float).values[:, None] for c in self.num]
        for c in self.cat:
            v = df[c].astype(str).values
            for l in self.levels[c]:
                parts.append((v == l).astype(float)[:, None])
        return np.hstack(parts) if parts else np.zeros((len(df), 0))

    def transform(self, df): return self.scaler.transform(self._raw(df))


def detect_task(y):
    if pd.api.types.is_numeric_dtype(y) and y.nunique() > 15: return "regression"
    return "classification"


def build_models(task, n_classes):
    if task == "regression":
        return {
            "Linear Regression (scratch)": ms.LinearRegression(),
            "Ridge (scratch)": ms.LinearRegression(l2=10),
            "KNN (scratch)": ms.KNN(7, "regression"),
            "Decision Tree (scratch)": ms.DecisionTree("regression", 8),
            "Random Forest (scratch)": ms.RandomForest("regression", 40, 8),
            "Gradient Boosting (scratch)": ms.GradientBoosting("regression", 80, 0.1, 3),
        }
    m = {
        "Logistic Regression (scratch)": ms.LogisticRegression(lr=0.2, epochs=150),
        "KNN (scratch)": ms.KNN(7),
        "Decision Tree (scratch)": ms.DecisionTree("classification", 8),
        "Random Forest (scratch)": ms.RandomForest("classification", 40, 8),
    }
    if n_classes == 2: m["Gradient Boosting (scratch)"] = ms.GradientBoosting("classification", 80, 0.1, 3)
    return m


def sklearn_baselines(task):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if task == "regression":
        return {"sklearn LinearRegression": LinearRegression(), "sklearn RandomForest": RandomForestRegressor(100, random_state=0, n_jobs=-1)}
    return {"sklearn LogisticRegression": LogisticRegression(max_iter=500), "sklearn RandomForest": RandomForestClassifier(100, random_state=0, n_jobs=-1)}


def run(df, target, max_rows=20000, log=print):
    df = df.dropna(subset=[target]).copy()
    if len(df) > max_rows: df = df.sample(max_rows, random_state=0)
    y_raw = df[target]; Xdf = df.drop(columns=[target])
    # drop id-like columns (unique per row)
    idlike = [c for c in Xdf.columns if Xdf[c].nunique() == len(Xdf) and not pd.api.types.is_float_dtype(Xdf[c])]
    Xdf = Xdf.drop(columns=idlike)
    task = detect_task(y_raw)
    pre = Preprocessor().fit(Xdf)
    X = pre.transform(Xdf)
    if task == "classification":
        labels = sorted(y_raw.astype(str).unique()); li = {l: i for i, l in enumerate(labels)}
        y = y_raw.astype(str).map(li).values
    else:
        labels, y = None, y_raw.astype(float).values
    Xtr, Xte, ytr, yte = ms.train_test_split(X, y, 0.2, 42, stratify=(task == "classification"))
    n_classes = len(labels) if labels else 0
    rows, fitted = [], {}
    for name, mdl in build_models(task, n_classes).items():
        t = time.time(); mdl.fit(Xtr, ytr); dt = time.time() - t
        p = mdl.predict(Xte)
        rows.append(_row(name, task, yte, p, dt, "scratch")); fitted[name] = mdl; log(f"{name}: done {dt:.1f}s")
    try:
        for name, mdl in sklearn_baselines(task).items():
            t = time.time(); mdl.fit(Xtr, ytr); dt = time.time() - t
            rows.append(_row(name, task, yte, mdl.predict(Xte), dt, "sklearn baseline")); fitted[name] = mdl
    except Exception as e:
        log(f"sklearn skipped: {e}")
    lb = pd.DataFrame(rows)
    key = "R2" if task == "regression" else "F1 (macro)"
    scratch = lb[lb["type"] == "scratch"].sort_values(key, ascending=False)
    best_name = scratch.iloc[0]["model"]  # best FROM-SCRATCH model is the deployed one (rubric)
    best = fitted[best_name]
    metric = ms.r2 if task == "regression" else ms.f1_macro
    imp = getattr(best, "importances_", None)
    if imp is None or len(imp) != X.shape[1] or np.allclose(imp, 0):
        imp = ms.permutation_importance(best, Xte[:1000], yte[:1000], metric)
    meta = {}
    for c in Xdf.columns:
        meta[c] = ({"kind": "num", "min": float(Xdf[c].min()), "max": float(Xdf[c].max()), "default": float(Xdf[c].median())}
                   if c in pre.num else {"kind": "cat", "levels": pre.levels.get(c, []), "default": (pre.levels.get(c) or [""])[0]} if c in pre.cat else {"kind": "skip"})
    pred_te = best.predict(Xte)
    bundle = dict(task=task, target=target, labels=labels, pre=pre, model=best, best_name=best_name, leaderboard=lb,
                  importances=dict(zip(pre.names, map(float, imp))), meta=meta, dropped=pre.dropped + idlike,
                  y_test=yte, y_pred=pred_te, n_rows=len(df), n_features=len(pre.names), key_metric=key)
    if task == "classification":
        bundle["cm"] = ms.confusion_matrix(yte, pred_te, list(range(n_classes)))
    return bundle


def _row(name, task, y, p, dt, typ):
    if task == "regression":
        return dict(model=name, type=typ, R2=ms.r2(y, p), RMSE=ms.rmse(y, p), MAE=ms.mae(y, p), seconds=round(dt, 2))
    return dict(model=name, type=typ, Accuracy=ms.accuracy(y, p), **{"F1 (macro)": ms.f1_macro(y, p)}, seconds=round(dt, 2))


def predict_one(bundle, row: dict):
    df = pd.DataFrame([row])
    for c, m in bundle["meta"].items():
        if c not in df: df[c] = m.get("default", 0)
    X = bundle["pre"].transform(df)
    mdl = bundle["model"]
    p = mdl.predict(X)[0]
    if bundle["task"] == "regression": return float(p), None
    proba = mdl.predict_proba(X)[0] if hasattr(mdl, "predict_proba") else None
    return bundle["labels"][int(p)], (dict(zip(bundle["labels"], map(float, proba))) if proba is not None else None)


def save(bundle, path="models/bundle.pkl"):
    with open(path, "wb") as f: pickle.dump(bundle, f)


def load(path="models/bundle.pkl"):
    with open(path, "rb") as f: return pickle.load(f)


if __name__ == "__main__":
    csv, target = sys.argv[1], sys.argv[2]
    b = run(pd.read_csv(csv), target)
    save(b)
    print(b["leaderboard"].to_string(index=False)); print("Deployed:", b["best_name"])
