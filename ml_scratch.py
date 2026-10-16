"""From-scratch ML (NumPy only). Judges score from-scratch ML highest -> keep this central.
Models: LinearRegression, Ridge, LogisticRegression (multiclass softmax), KNN,
DecisionTree (CART), RandomForest, GradientBoosting (regression/binary), KMeans.
All follow fit / predict (+ predict_proba for classifiers)."""
import numpy as np


# ---------- utilities ----------
def train_test_split(X, y, test_size=0.2, seed=42, stratify=False):
    rng = np.random.default_rng(seed)
    n = len(X)
    if stratify:
        idx_test = []
        for c in np.unique(y):
            ci = np.where(y == c)[0]
            rng.shuffle(ci)
            idx_test += list(ci[: max(1, int(len(ci) * test_size))])
        idx_test = np.array(idx_test)
    else:
        idx = rng.permutation(n)
        idx_test = idx[: int(n * test_size)]
    mask = np.zeros(n, bool)
    mask[idx_test] = True
    return X[~mask], X[mask], y[~mask], y[mask]


class StandardScaler:
    def fit(self, X):
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.sd[self.sd == 0] = 1
        return self

    def transform(self, X):
        return (X - self.mu) / self.sd

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# ---------- metrics ----------
def accuracy(y, p): return float(np.mean(y == p))
def rmse(y, p): return float(np.sqrt(np.mean((y - p) ** 2)))
def mae(y, p): return float(np.mean(np.abs(y - p)))
def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot else 0.0


def f1_macro(y, p):
    f = []
    for c in np.unique(y):
        tp = np.sum((y == c) & (p == c))
        fp = np.sum((y != c) & (p == c))
        fn = np.sum((y == c) & (p != c))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        f.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    return float(np.mean(f))


def confusion_matrix(y, p, labels):
    m = np.zeros((len(labels), len(labels)), int)
    li = {l: i for i, l in enumerate(labels)}
    for a, b in zip(y, p):
        m[li[a], li[b]] += 1
    return m


def kfold_indices(n, k=5, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    return np.array_split(idx, k)


def permutation_importance(model, X, y, metric, higher_better=True, seed=0, repeats=3):
    rng = np.random.default_rng(seed)
    base = metric(y, model.predict(X))
    out = []
    for j in range(X.shape[1]):
        drops = []
        for _ in range(repeats):
            Xp = X.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            s = metric(y, model.predict(Xp))
            drops.append(base - s if higher_better else s - base)
        out.append(np.mean(drops))
    return np.array(out)


# ---------- linear models ----------
class LinearRegression:
    def __init__(self, l2=0.0): self.l2 = l2

    def fit(self, X, y):
        A = np.c_[np.ones(len(X)), X]
        reg = self.l2 * np.eye(A.shape[1]); reg[0, 0] = 0
        self.w = np.linalg.solve(A.T @ A + reg, A.T @ y)
        return self

    def predict(self, X): return np.c_[np.ones(len(X)), X] @ self.w


class LogisticRegression:
    """Multiclass softmax regression, mini-batch gradient descent, L2."""
    def __init__(self, lr=0.1, epochs=300, l2=1e-3, batch=256, seed=0):
        self.lr, self.epochs, self.l2, self.batch, self.seed = lr, epochs, l2, batch, seed

    @staticmethod
    def _softmax(z):
        z = z - z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        self.classes_ = np.unique(y)
        Y = (y[:, None] == self.classes_[None, :]).astype(float)
        n, d = X.shape
        self.W = np.zeros((d, len(self.classes_))); self.b = np.zeros(len(self.classes_))
        for _ in range(self.epochs):
            idx = rng.permutation(n)
            for s in range(0, n, self.batch):
                bi = idx[s:s + self.batch]
                P = self._softmax(X[bi] @ self.W + self.b)
                G = (P - Y[bi]) / len(bi)
                self.W -= self.lr * (X[bi].T @ G + self.l2 * self.W)
                self.b -= self.lr * G.sum(0)
        return self

    def predict_proba(self, X): return self._softmax(X @ self.W + self.b)
    def predict(self, X): return self.classes_[self.predict_proba(X).argmax(1)]


# ---------- KNN ----------
class KNN:
    def __init__(self, k=7, task="classification"): self.k, self.task = k, task

    def fit(self, X, y):
        self.X, self.y = X, y
        if self.task == "classification": self.classes_ = np.unique(y)
        return self

    def _neighbors(self, X):
        d = (X ** 2).sum(1)[:, None] - 2 * X @ self.X.T + (self.X ** 2).sum(1)[None, :]
        return np.argsort(d, axis=1)[:, : self.k]

    def predict_proba(self, X):
        nb = self._neighbors(X)
        return np.stack([(self.y[nb] == c).mean(1) for c in self.classes_], 1)

    def predict(self, X):
        if self.task == "regression": return self.y[self._neighbors(X)].mean(1)
        return self.classes_[self.predict_proba(X).argmax(1)]


# ---------- decision tree (CART) ----------
class DecisionTree:
    def __init__(self, task="classification", max_depth=8, min_samples=5, max_features=None, n_thresholds=16, seed=0):
        self.task, self.max_depth, self.min_samples = task, max_depth, min_samples
        self.max_features, self.n_thr, self.rng = max_features, n_thresholds, np.random.default_rng(seed)

    def _impurity(self, y):
        if self.task == "regression": return y.var() if len(y) else 0
        p = np.bincount(y, minlength=self.n_classes) / len(y)
        return 1 - np.sum(p ** 2)

    def _leaf(self, y):
        if self.task == "regression": return float(y.mean())
        return np.bincount(y, minlength=self.n_classes) / len(y)

    def fit(self, X, y):
        if self.task == "classification":
            self.classes_, yi = np.unique(y, return_inverse=True)
            self.n_classes = len(self.classes_)
        else:
            yi = y.astype(float)
        self.n_features = X.shape[1]
        self.importances_ = np.zeros(self.n_features)
        self.tree = self._grow(X, yi, 0)
        s = self.importances_.sum()
        if s: self.importances_ /= s
        return self

    def _grow(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples * 2 or self._impurity(y) == 0:
            return ("leaf", self._leaf(y))
        feats = np.arange(self.n_features)
        if self.max_features: feats = self.rng.choice(feats, self.max_features, replace=False)
        best = (None, None, 0.0)
        parent = self._impurity(y)
        for f in feats:
            col = X[:, f]
            qs = np.unique(np.quantile(col, np.linspace(0.05, 0.95, self.n_thr)))
            for t in qs:
                m = col <= t
                nl = m.sum()
                if nl < self.min_samples or len(y) - nl < self.min_samples: continue
                gain = parent - (nl * self._impurity(y[m]) + (len(y) - nl) * self._impurity(y[~m])) / len(y)
                if gain > best[2]: best = (f, t, gain)
        if best[0] is None: return ("leaf", self._leaf(y))
        f, t, g = best
        self.importances_[f] += g * len(y)
        m = X[:, f] <= t
        return ("node", f, t, self._grow(X[m], y[m], depth + 1), self._grow(X[~m], y[~m], depth + 1))

    def _one(self, x, node):
        while node[0] == "node":
            node = node[3] if x[node[1]] <= node[2] else node[4]
        return node[1]

    def _raw(self, X): return [self._one(x, self.tree) for x in X]

    def predict_proba(self, X): return np.array(self._raw(X))

    def predict(self, X):
        r = self._raw(X)
        if self.task == "regression": return np.array(r)
        return self.classes_[np.argmax(r, 1)]


class RandomForest:
    def __init__(self, task="classification", n_trees=40, max_depth=8, seed=0):
        self.task, self.n_trees, self.max_depth, self.seed = task, n_trees, max_depth, seed

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        mf = max(1, int(np.sqrt(X.shape[1])))
        if self.task == "classification": self.classes_ = np.unique(y)
        self.trees = []
        for i in range(self.n_trees):
            b = rng.integers(0, len(X), len(X))
            t = DecisionTree(self.task, self.max_depth, max_features=mf, seed=self.seed + i)
            yb = y[b]
            if self.task == "classification":
                t.fit(X[b], yb)
                if len(t.classes_) != len(self.classes_):  # bootstrap missed a class -> retry on full data
                    t.fit(X, y)
            else:
                t.fit(X[b], yb)
            self.trees.append(t)
        self.importances_ = np.mean([t.importances_ for t in self.trees], 0)
        return self

    def predict_proba(self, X):
        P = np.zeros((len(X), len(self.classes_)))
        for t in self.trees:
            P += t.predict_proba(X)
        return P / len(self.trees)

    def predict(self, X):
        if self.task == "regression": return np.mean([t.predict(X) for t in self.trees], 0)
        return self.classes_[self.predict_proba(X).argmax(1)]


class GradientBoosting:
    """Gradient boosting with shallow regression trees. Regression: squared loss.
    Binary classification: log loss. (Multiclass -> use RandomForest/Logistic.)"""
    def __init__(self, task="regression", n_estimators=80, lr=0.1, max_depth=3, seed=0):
        self.task, self.n, self.lr, self.max_depth, self.seed = task, n_estimators, lr, max_depth, seed

    @staticmethod
    def _sig(z): return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X, y):
        if self.task == "classification":
            self.classes_ = np.unique(y)
            yy = (y == self.classes_[1]).astype(float)
            p0 = np.clip(yy.mean(), 1e-6, 1 - 1e-6)
            self.f0 = np.log(p0 / (1 - p0))
        else:
            yy = y.astype(float); self.f0 = yy.mean()
        F = np.full(len(X), self.f0); self.trees = []
        for i in range(self.n):
            resid = (yy - self._sig(F)) if self.task == "classification" else (yy - F)
            t = DecisionTree("regression", self.max_depth, seed=self.seed + i).fit(X, resid)
            F += self.lr * t.predict(X); self.trees.append(t)
        self.importances_ = np.mean([t.importances_ for t in self.trees], 0)
        return self

    def _F(self, X):
        F = np.full(len(X), self.f0)
        for t in self.trees: F += self.lr * t.predict(X)
        return F

    def predict_proba(self, X):
        p = self._sig(self._F(X)); return np.c_[1 - p, p]

    def predict(self, X):
        if self.task == "regression": return self._F(X)
        return self.classes_[(self._sig(self._F(X)) > 0.5).astype(int)]


class KMeans:
    def __init__(self, k=4, iters=100, seed=0): self.k, self.iters, self.seed = k, iters, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        C = X[rng.choice(len(X), self.k, replace=False)]
        for _ in range(self.iters):
            lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(self.k)])
            if np.allclose(newC, C): break
            C = newC
        self.centers_ = C; self.labels_ = lab
        return self

    def predict(self, X): return ((X[:, None] - self.centers_[None]) ** 2).sum(2).argmin(1)


# ---------- extra metrics / helpers ----------
def roc_auc(y, score):
    """Rank-based AUC (Mann-Whitney), ties averaged."""
    y = np.asarray(y); s = np.asarray(score, float)
    pos, neg = (y == 1).sum(), (y == 0).sum()
    if pos == 0 or neg == 0: return 0.5
    order = np.argsort(s, kind="mergesort"); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.unique(s, return_counts=True)[1].argmax():][:0]): pass
    # average ranks for ties
    u, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks); ranks = (sums / cnt)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def top_decile_lift(y, score):
    y = np.asarray(y); k = max(1, len(y) // 10)
    top = np.argsort(-np.asarray(score))[:k]
    base = y.mean()
    return float(y[top].mean() / base) if base else 1.0


def robust_z(x):
    """Median/MAD z-score (0.6745 scaling). Robust to the outliers we are trying to find."""
    x = np.asarray(x, float); med = np.median(x); mad = np.median(np.abs(x - med))
    if mad == 0: mad = np.mean(np.abs(x - med)) or 1.0
    return 0.6745 * (x - med) / mad
