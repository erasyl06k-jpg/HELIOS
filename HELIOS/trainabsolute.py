"""Обучение классификатора цикла E02 (complete / incomplete).
Данные: data/index.csv с колонками path,volunteer_id,label (complete|incomplete);
каждый path указывает на CSV с колонками t (мс), angle (градусы) - один цикл.
Такие файлы создаёт кнопка «Экспорт циклов» в app.html.
Запуск: python train.py
"""
import json, numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix

def features(t, a):
    t, a = np.asarray(t, float), np.asarray(a, float)
    dt = np.maximum(np.diff(t), 1) / 1000
    vel = np.abs(np.diff(a)) / dt
    return [a.max() - a.min(), a.max(), a.min(), (t[-1] - t[0]) / 1000, vel.max()]

def rule(x):  # базовый классификатор: те же пороги, что в app.html
    rom, mx = x[0], x[1]
    return int(rom >= 90 and mx >= 150)

idx = pd.read_csv("data/index.csv")
X = np.array([features(*pd.read_csv("data/" + p)[["t", "angle"]].values.T) for p in idx.path])
y = (idx.label == "complete").astype(int).values
g = idx.volunteer_id.values

# строгое разбиение по ID добровольца: train / val / test не пересекаются по людям
tv, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(X, y, g))
tr, va = next(GroupShuffleSplit(1, test_size=0.25, random_state=0).split(X[tv], y[tv], g[tv]))
tr, va = tv[tr], tv[va]
assert not (set(g[tr]) & set(g[va]) | set(g[tr]) & set(g[te]) | set(g[va]) & set(g[te]))

sc = StandardScaler().fit(X[tr])
best = max(([C, balanced_accuracy_score(y[va], LogisticRegression(C=C, class_weight="balanced")
            .fit(sc.transform(X[tr]), y[tr]).predict(sc.transform(X[va])))] for C in [.01, .1, 1, 10]),
           key=lambda r: r[1])
clf = LogisticRegression(C=best[0], class_weight="balanced").fit(sc.transform(X[tr]), y[tr])

def report(pred):
    return {"balanced_accuracy": balanced_accuracy_score(y[te], pred),
            "f1_complete": f1_score(y[te], pred),
            "confusion_matrix [[TN,FP],[FN,TP]]": confusion_matrix(y[te], pred).tolist()}

res = {"model": report(clf.predict(sc.transform(X[te]))),
       "baseline_rule": report([rule(x) for x in X[te]]),
       "n_test_cycles": int(len(te)), "n_test_volunteers": int(len(set(g[te]))), "C": best[0]}
json.dump(res, open("metrics.json", "w"), indent=2, default=float)
json.dump({"mean": sc.mean_.tolist(), "scale": sc.scale_.tolist(),
           "coef": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0])}, open("model.json", "w"))
print(json.dumps(res, indent=2, default=float))
