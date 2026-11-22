"""Эталонная проверка задания этапа 5: честный сигнал против ловушки утечки."""
import glob, sys, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

root = sys.argv[1] if len(sys.argv) > 1 else 'out_s'
churn = pd.read_csv(f'{root}/reference/churn_truth.csv').set_index('subscriber_key')
parts = sorted(glob.glob(f'{root}/raw/cdr/dt=*/cdr.csv'))
H = len(parts) * 2 // 3                       # горизонт: метка — уход после этой отсечки

def load(ps, cols):
    out = []
    for f in ps:
        try: out.append(pd.read_csv(f, usecols=cols, low_memory=False))
        except Exception: pass
    return pd.concat(out, ignore_index=True).dropna(subset=['subscriber_key'])

half = H // 2
prev = load(parts[:half], ['subscriber_key'])
last = load(parts[half:H], ['subscriber_key', 'duration_sec', 'bytes_down'])
last['bytes_down'] = last['bytes_down'].clip(lower=0)   # очистка дефекта диапазона
f = pd.concat([prev.groupby('subscriber_key').size().rename('n_prev'),
               last.groupby('subscriber_key').size().rename('n_last'),
               last.groupby('subscriber_key').bytes_down.sum().rename('bytes_last')],
              axis=1).fillna(0)
f['trend'] = (f.n_last + 1) / (f.n_prev + 1)

tic = pd.concat([pd.read_csv(x) for x in sorted(glob.glob(f'{root}/raw/tickets/dt=*/tickets.csv'))[:H]],
                ignore_index=True).dropna(subset=['subscriber_key'])
g = tic.groupby('subscriber_key')
f = f.join(pd.DataFrame({
    'tickets': g.size(),
    'quality': g.category.apply(lambda s: (s == 'quality').sum()),
    'phone':   g.channel.apply(lambda s: (s.astype(str).str.strip().str.lower() == 'phone').sum()),
    'retention_contact': g.retention_contact.max().astype(float),
})).fillna(0).join(churn)

f['label'] = (f.churn_day_index >= H).astype(int)
f = f[(f.churn_day_index < 0) | (f.churn_day_index >= H)]
# счётные признаки логарифмируем и масштабируем: без этого объём трафика
# в сотнях миллионов подавляет остальные слагаемые линейной модели
# дефект «объём вне допустимого диапазона» даёт отрицательные значения:
# без очистки логарифмирование обращается в минус бесконечность
f['bytes_last'] = f['bytes_last'].clip(lower=0)
for c in ('n_prev', 'n_last', 'bytes_last', 'tickets', 'quality', 'phone'):
    f[c] = np.log1p(f[c])
idx = np.random.default_rng(1).permutation(len(f))
tr, te = f.iloc[idx[:int(len(f) * .7)]], f.iloc[idx[int(len(f) * .7):]]

honest = ['n_prev', 'n_last', 'trend', 'bytes_last', 'tickets', 'quality', 'phone']
print(f'абонентов {len(f):,}, доля ушедших в горизонте {f.label.mean()*100:.1f} %'.replace(',', ' '))
res = {}
for name, cols in (('честные признаки', honest), ('+ ловушка утечки', honest + ['retention_contact'])):
    m = make_pipeline(StandardScaler(),
                      LogisticRegression(max_iter=2000)).fit(tr[cols], tr.label)
    a = roc_auc_score(tr.label, m.predict_proba(tr[cols])[:, 1])
    b = roc_auc_score(te.label, m.predict_proba(te[cols])[:, 1])
    res[name] = (a, b)
    print(f'  {name:<22} AUC обучение {a:.3f} | контроль {b:.3f}')
print(f'\nспад активности:      ушедшие {f.loc[f.label==1,"trend"].mean():.2f}  '
      f'оставшиеся {f.loc[f.label==0,"trend"].mean():.2f}')
print(f'обращений на абонента: ушедшие {f.loc[f.label==1,"tickets"].mean():.2f}  '
      f'оставшиеся {f.loc[f.label==0,"tickets"].mean():.2f}')
print(f'признак удержания:     ушедшие {f.loc[f.label==1,"retention_contact"].mean()*100:.1f} %  '
      f'оставшиеся {f.loc[f.label==0,"retention_contact"].mean()*100:.1f} %')
