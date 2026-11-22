"""Прогноз будущих событий деградации сети; исходные файлы не изменяются."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', str(Path.cwd() / '.mplconfig'))
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger('telemetry_pipeline')
CONFIG_PATH = Path(__file__).resolve().with_name('config.json')


def days_delta(value):
    return pd.Timedelta(int(value), unit='D')


def load_config(path=CONFIG_PATH):
    """Load and minimally validate the experiment configuration."""
    config = json.loads(Path(path).read_text(encoding='utf-8'))
    required = {'random_state', 'history_window_days', 'prediction_horizon_days', 'split_days',
                'features', 'search_count', 'xgboost_search_space', 'baseline_xgboost',
                'logistic_regression', 'xgboost_runtime', 'alert_policy', 'cooldown_days',
                'shap_sample_size', 'paths'}
    missing = required - config.keys()
    if missing:
        raise ValueError(f'В config.json отсутствуют обязательные параметры: {sorted(missing)}')
    if sum(len(config['split_days'].get(k, [])) for k in ('train','validation','test')) == 0:
        raise ValueError('В config.json не заданы календарные срезы')
    if int(config['history_window_days']) <= 0 or int(config['prediction_horizon_days']) <= 0:
        raise ValueError('Окно истории и горизонт прогноза должны быть положительными')
    if int(config['search_count']) < 1:
        raise ValueError('search_count должен быть не меньше 1')
    if not 0 < config['alert_policy']['max_alerts_per_1000'] <= 1000:
        raise ValueError('alert_policy.max_alerts_per_1000 должен быть в интервале (0, 1000]')
    return config


def apply_config(config):
    """Expose configured values to the existing pure feature/model helpers."""
    global SEED, WINDOW, HORIZON, SPLIT_DAYS, FEATURES, SEARCH_SPACE, POLICY, DEFINITIONS
    SEED = int(config['random_state'])
    WINDOW = int(config['history_window_days'])
    HORIZON = int(config['prediction_horizon_days'])
    SPLIT_DAYS = {k: list(v) for k, v in config['split_days'].items()}
    FEATURES = list(config['features'])
    SEARCH_SPACE = dict(config['xgboost_search_space'])
    POLICY = dict(config['alert_policy'])
    if 'DEFINITIONS' in globals():
        DEFINITIONS = {
            key: (name, source, description.replace('исторические 7 суток', f'исторические {WINDOW} суток')
                  .replace('за 7 суток', f'за {WINDOW} суток')
                  .replace('7-дневном окне', f'{WINDOW}-дневном окне')
                  .replace('8 при отсутствии ERROR', f'{WINDOW + 1} при отсутствии ERROR'))
            for key, (name, source, description) in BASE_DEFINITIONS.items()
        }


CONFIG = load_config()
apply_config(CONFIG)
BASE_DEFINITIONS = {
 'mean_load': ('Средняя загрузка', 'load_pct', 'Среднее за исторические 7 суток, %'),
 'max_load': ('Максимальная загрузка', 'load_pct', 'Максимум за исторические 7 суток, %'),
 'std_load': ('Изменчивость загрузки', 'load_pct', 'Выборочное стандартное отклонение, %'),
 'mean_latency': ('Средняя задержка', 'latency_ms', 'Среднее за 7 суток, мс'),
 'max_latency': ('Максимальная задержка', 'latency_ms', 'Максимум за 7 суток, мс'),
 'std_latency': ('Изменчивость задержки', 'latency_ms', 'Выборочное стандартное отклонение, мс'),
 'mean_error_rate': ('Средняя доля ошибок', 'error_rate', 'Среднее за 7 суток, доля 0…1'),
 'max_error_rate': ('Максимальная доля ошибок', 'error_rate', 'Максимум за 7 суток, доля 0…1'),
 'error_rate_slope': ('Тренд доли ошибок', 'error_rate, ts', 'Наклон МНК по времени, доля за сутки; NaN при <2 различных времён'),
 'error_count': ('Исторические ERROR', 'level', 'Число ERROR строго до cutoff в 7-дневном окне'),
 'warning_count': ('Исторические WARN', 'level', 'Число WARN строго до cutoff в 7-дневном окне'),
 'measurement_count': ('Число измерений', 'ts', 'Число записей в историческом окне'),
 'days_since_error': ('Давность ERROR в окне', 'level, ts', 'Сутки от последнего ERROR в окне; 8 при отсутствии ERROR'),
 'days_since_measurement': ('Давность измерения', 'ts', 'Сутки от последнего исторического измерения'),
 'observed_days': ('Сутки с измерениями', 'ts', 'Число различных календарных суток в историческом окне')}
DEFINITIONS = BASE_DEFINITIONS.copy()
apply_config(CONFIG)
def dump(path, obj):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [clean(x) for x in v]
        if isinstance(v, (np.ndarray, pd.Series)): return clean(v.tolist())
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.bool_): return bool(v)
        return v
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, default=str,
                                    allow_nan=False) + '\n', encoding='utf-8')


def atomic_dump(path, obj):
    """Replace JSON metadata only after the complete file has been written."""
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.tmp')
    dump(temporary, obj)
    os.replace(temporary, path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda: stream.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def preserve(base, out):
    dest = out / 'previous_files.json'
    if dest.exists(): return
    files = [p for p in base.iterdir() if p.is_file() and p.suffix in {'.py', '.md', '.ipynb', '.json', '.txt', '.zip'}
             and 'failure' not in p.name.lower()]
    files += list((base / 'tests').glob('test_*.py'))
    files += [p for p in (base / 'processed/telemetry').rglob('*') if p.is_file()]
    dump(dest, {str(p.relative_to(base)): sha256(p) for p in sorted(files)})


def load_sources(dataset, out):
    # Переиспользуются проверенные правила разбора, но не признаки за весь период.
    from telemetry_hw import discover, load_telemetry, normalize
    files = discover(dataset)
    if not files: raise ValueError('Нет исходных JSONL телеметрии; подмена учебными данными запрещена.')
    raw, audit = load_telemetry(files, out)
    exact = int(raw.duplicated().sum())
    frame, normalization = normalize(raw, source_tz='UTC')
    level_messages = raw.groupby(['level', 'message']).size().reset_index(name='count').to_dict('records')
    schema = list(raw.columns)
    del raw
    frame = frame.loc[frame.timestamp.notna() & frame.node_id.notna()].copy()
    frame = frame.sort_values(['node_id', 'timestamp'], kind='stable').reset_index(drop=True)
    # Отсутствующие или неизвестные level нельзя молча считать INFO.
    if not frame.level.isin(['info', 'warn', 'error']).all():
        raise ValueError('Неизвестная семантика level: требуется пересмотр прокси-разметки.')
    frame['day'] = frame.timestamp.dt.floor('D')
    grouped = frame.groupby(['day', 'level']).size().unstack(fill_value=0).reset_index()
    grouped.to_csv(out / 'events_by_day.csv', index=False)
    source_info = dict(files=len(files), physical_lines=audit['physical_lines'],
        accepted=audit['physical_lines'] - audit['rejected_lines'], rejected=audit['rejected_lines'],
        usable_records=len(frame), nodes=int(frame.node_id.nunique()), schema=schema,
        exact_duplicates=exact, duplicate_node_times=int(frame.duplicated(['node_id', 'timestamp']).sum()),
        start=frame.timestamp.min(), end=frame.timestamp.max(), timezone='UTC (допущение для наивных дат)',
        normalization=normalization, levels=frame.level.value_counts().to_dict(),
        level_messages=level_messages, error_days=grouped.loc[grouped.get('error', 0) > 0, 'day'].astype(str).tolist(),
        late_files=sum('late_delivery' in p.parts for p in files),
        late_rows=sum(x['accepted'] for x in audit['manifest'] if 'late_delivery' in x['path']),
        label_search={
            'reference_files': sorted(str(p.relative_to(dataset)) for p in (dataset / 'reference').glob('*')),
            'ticket_schema': list(pd.read_csv(next((dataset/'raw/tickets').glob('dt=*/*.csv')), nrows=0).columns)
                if list((dataset/'raw/tickets').glob('dt=*/*.csv')) else [],
            'independent_network_incidents': False,
            'reason': 'В reference нет журнала сетевых инцидентов; tickets привязаны к абонентам, не к узлам. '
                      'passport.broken_day_index — сценарий генератора, не операционный журнал; не используется в X/y.'})
    manifest = [{**x, 'path': str(Path(x['path']).relative_to(dataset))} for x in audit['manifest']]
    for p in [dataset/'reference/passport.json', Path(__file__).parent/'generator/tdg/facts.py',
              Path(__file__).parent/'generator/README.md']:
        if p.exists(): manifest.append({'path': str(p.relative_to(dataset)) if p.is_relative_to(dataset) else str(p.name),
                                       'sha256': sha256(p), 'bytes': p.stat().st_size})
    dump(out / 'input_manifest.json', manifest)
    dump(out / 'source_profile.json', source_info)
    return frame, source_info


def build_snapshot(frame, cutoff, window=WINDOW, horizon=HORIZON):
    """[c−W,c) → c → [c,c+H). Состав популяции зависит только от прошлого."""
    cutoff = pd.Timestamp(cutoff)
    hist = frame.loc[frame.timestamp.ge(cutoff - days_delta(window)) & frame.timestamp.lt(cutoff)].copy()
    future = frame.loc[frame.timestamp.ge(cutoff) & frame.timestamp.lt(cutoff + days_delta(horizon))].copy()
    hist['is_error'] = hist.level.eq('error').astype(int)
    hist['is_warning'] = hist.level.eq('warn').astype(int)
    hist['day'] = hist.timestamp.dt.floor('D')
    hist['t'] = (hist.timestamp - (cutoff - days_delta(window))).dt.total_seconds()/86400
    hist['tt'] = hist.t**2
    hist['ty'] = hist.t * hist.error_rate
    grouped = hist.groupby('node_id', sort=True)
    f = grouped.agg(mean_load=('load_pct', 'mean'), max_load=('load_pct', 'max'), std_load=('load_pct', 'std'),
        mean_latency=('latency_ms', 'mean'), max_latency=('latency_ms', 'max'), std_latency=('latency_ms', 'std'),
        mean_error_rate=('error_rate', 'mean'), max_error_rate=('error_rate', 'max'),
        error_count=('is_error', 'sum'), warning_count=('is_warning', 'sum'),
        measurement_count=('timestamp', 'size'), hist_min=('timestamp', 'min'), hist_max=('timestamp', 'max'),
        observed_days=('day', 'nunique'))
    pairs = hist.loc[hist.error_rate.notna()].groupby('node_id').agg(
        n=('t', 'size'), t=('t', 'sum'), tt=('tt', 'sum'), ty=('ty', 'sum'), y=('error_rate', 'sum'))
    den = pairs.n * pairs.tt - pairs.t**2
    slope = (pairs.n * pairs.ty - pairs.t * pairs.y) / den.where(den > 1e-12)
    f['error_rate_slope'] = slope
    previous_error = hist.loc[hist.is_error.eq(1)].groupby('node_id').timestamp.max()
    f['days_since_error'] = ((cutoff - previous_error).dt.total_seconds()/86400).reindex(f.index).fillna(window+1)
    f['days_since_measurement'] = (cutoff - f.hist_max).dt.total_seconds()/86400
    target_events = future.loc[future.level.isin(['warn', 'error'])].groupby('node_id').agg(
        future_events=('timestamp', 'size'), label_min=('timestamp', 'min'), label_max=('timestamp', 'max'))
    f = f.join(target_events)
    f['future_events'] = f.future_events.fillna(0).astype(int)
    f['future_measurements'] = future.groupby('node_id').size().reindex(f.index).fillna(0).astype(int)
    f['target'] = f.future_events.gt(0).astype('int8')
    f['cutoff'] = cutoff
    f['history_start'] = cutoff - days_delta(window)
    f['label_end'] = cutoff + days_delta(horizon)
    f = f.reset_index()
    f['node_code'] = f.node_id.map(lambda s: 'УЗ-' + hashlib.sha256(('telemetry-study:'+s).encode()).hexdigest()[:12])
    f['row_id'] = f.node_code + '/' + cutoff.strftime('%Y-%m-%d')
    assert f.hist_max.lt(cutoff).all() and f.hist_min.ge(cutoff-days_delta(window)).all()
    assert f.loc[f.target.eq(1), 'label_min'].ge(cutoff).all()
    assert f.loc[f.target.eq(1), 'label_max'].lt(cutoff+days_delta(horizon)).all()
    return f


def build_dataset(frame, out):
    origin = frame.timestamp.min().floor('D')
    end = frame.timestamp.max().floor('D') + days_delta(1)
    parts = []
    for split, days in SPLIT_DAYS.items():
        for day in days:
            cutoff = origin + days_delta(day)
            assert cutoff+days_delta(HORIZON) <= end
            f = build_snapshot(frame, cutoff)
            f['split'] = split
            parts.append(f)
            print(f'{split}: {cutoff.date()} N={len(f)}, y=1: {f.target.sum()}', flush=True)
    data = pd.concat(parts, ignore_index=True)
    assert data.row_id.is_unique
    split_info = []
    for split in SPLIT_DAYS:
        f = data.loc[data.split.eq(split)]
        split_info.append(dict(split=split, first_cutoff=f.cutoff.min(), last_cutoff=f.cutoff.max(),
            history_start=f.history_start.min(), label_end=f.label_end.max(), observations=len(f),
            positives=int(f.target.sum()), positive_rate=float(f.target.mean()), nodes=int(f.node_id.nunique()),
            no_future_measurements=int(f.future_measurements.eq(0).sum()),
            unseen_nodes=int((~f.node_id.isin(data.loc[data.split.eq('train'), 'node_id'])).sum())))
    for previous, following in zip(split_info, split_info[1:]):
        assert pd.Timestamp(previous['label_end']) <= pd.Timestamp(following['history_start'])
    for split, part in data.groupby('split'):
        cuts = sorted(part.cutoff.unique())
        assert all(b-a >= days_delta(HORIZON) for a,b in zip(cuts,cuts[1:]))
        assert part.target.nunique() == 2, f'Оба класса необходимы: {split}'
    data.to_parquet(out / 'node_time_dataset.parquet', index=False)
    dump(out / 'split_summary.json', split_info)
    return data, split_info


def predictors(frame):
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan).astype(float)


def metrics(y, p, threshold=.5):
    y, p = np.asarray(y), np.asarray(p)
    d = p >= threshold
    tn, fp, fn, tp = confusion_matrix(y, d, labels=[0,1]).ravel()
    return dict(threshold=float(threshold), n=len(y), positives=int(y.sum()), TP=int(tp), TN=int(tn),
        FP=int(fp), FN=int(fn), precision=float(precision_score(y,d,zero_division=0)),
        total_alerts=int(tp+fp), false_alerts=int(fp),
        recall=float(recall_score(y,d,zero_division=0)), f1=float(f1_score(y,d,zero_division=0)),
        accuracy=float(accuracy_score(y,d)), roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
        average_precision=float(average_precision_score(y,p)) if y.sum() else None,
        alerts_per_1000=float(1000*d.mean()), false_alerts_per_1000=float(1000*fp/len(y)))


def threshold_table(y, p):
    # Полный перебор уникальных score: честный дискретный бюджет без случайного отсечения ties.
    y,p = np.asarray(y), np.asarray(p)
    order = np.argsort(-p, kind='stable'); sy,sp=y[order],p[order]
    end = np.r_[np.flatnonzero(np.diff(sp)!=0), len(sp)-1]
    tp=np.cumsum(sy)[end]; count=end+1; fp=count-tp; fn=y.sum()-tp; tn=len(y)-y.sum()-fp
    precision=tp/count; recall=tp/y.sum()
    rows=pd.DataFrame(dict(threshold=sp[end], TP=tp, TN=tn, FP=fp, FN=fn,
        precision=precision, recall=recall, f1=2*tp/np.maximum(2*tp+fp+fn,1),
        alerts_per_1000=1000*count/len(y), false_alerts_per_1000=1000*fp/len(y)))
    # При score=1 порог 1 тоже создаёт тревоги; пустое решение требует строгого превышения.
    empty_threshold=max(1.,float(np.nextafter(p.max(),np.inf)))
    empty=pd.DataFrame([dict(threshold=empty_threshold,TP=0,TN=int((y==0).sum()),FP=0,FN=int(y.sum()),
        precision=0.,recall=0.,f1=0.,alerts_per_1000=0.,false_alerts_per_1000=0.)])
    table=pd.concat([empty, rows], ignore_index=True)
    table['total_alerts']=table.TP+table.FP
    table['false_alerts']=table.FP
    return table


def choose_threshold(frame, probabilities, policy=POLICY):
    if not frame.split.eq('validation').all(): raise ValueError('Порог выбирается только на validation')
    table = threshold_table(frame.target.to_numpy(), probabilities)
    eligible = table.loc[table.alerts_per_1000.le(policy['max_alerts_per_1000'])]
    best = eligible.sort_values(['recall','FP','threshold'], ascending=[False,True,False], kind='stable').iloc[0]
    return float(best.threshold), table, bool(best.recall >= policy['desired_recall'])


def xgb_model(params, config=CONFIG):
    from xgboost import XGBClassifier
    return make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True),
        XGBClassifier(objective='binary:logistic', tree_method=config['xgboost_runtime']['tree_method'], eval_metric='logloss',
                      random_state=config['random_state'],
                      n_jobs=config['xgboost_runtime']['n_jobs'], verbosity=0, **params))


def randomized_search(train, validation, out, count=None, config=CONFIG):
    if not train.split.eq('train').all() or not validation.split.eq('validation').all():
        raise ValueError('Поиск получает только train и validation')
    assert train.label_end.max() <= validation.history_start.min()
    ratio = float((train.target==0).sum()/train.target.sum())
    count = int(count if count is not None else config['search_count'])
    space = {**config['xgboost_search_space'], 'scale_pos_weight': [1., float(np.sqrt(ratio)), ratio]}
    rows=[]; start=time.perf_counter()
    for i,params in enumerate(ParameterSampler(space, n_iter=count, random_state=config['random_state'])):
        model=xgb_model(params, config); fit_start=time.perf_counter()
        model.fit(predictors(train), train.target)
        fit_seconds=time.perf_counter()-fit_start
        p=model.predict_proba(predictors(validation))[:,1]
        row={'attempt':i+1, 'params':params, 'fit_seconds':fit_seconds,
             'validation_ap':float(average_precision_score(validation.target,p)),
             'validation_roc_auc':float(roc_auc_score(validation.target,p))}
        rows.append(row)
        print(f'Поиск {i+1}/{count}: AP={row["validation_ap"]:.6f}, fit={fit_seconds:.2f}с',flush=True)
    elapsed=time.perf_counter()-start
    winner=max(rows, key=lambda r: (r['validation_ap'], -r['attempt']))
    dump(out/'search_results.json', rows)
    return winner, dict(space=space, attempts=len(rows), seed=config['random_state'], metric='validation_average_precision',
        strategy='Фиксированный chronological holdout; train не расширяется validation; refit=False внутри поиска',
        selection_splits=['train','validation'], test_used=False, search_seconds=elapsed,
        sum_candidate_fit_seconds=sum(r['fit_seconds'] for r in rows), refit_in_search=False,
        best=winner)


def suppression(frame, p, threshold, cooldown_days=14):
    """Симуляция по score и прошлым оповещениям, без обращения к меткам."""
    p=np.asarray(p); decision=p>=threshold
    kept=np.zeros(len(frame),dtype=bool); previous={}
    for idx in np.argsort(frame.cutoff.to_numpy(),kind='stable'):
        if not decision[idx]: continue
        node=frame.node_code.iloc[idx]; cutoff=frame.cutoff.iloc[idx]
        if node not in previous or (cutoff-previous[node]).total_seconds()>=cooldown_days*86400:
            kept[idx]=True; previous[node]=cutoff
    return kept


def decision_metrics(y, d):
    # Для подавления probability уже не является score: считаем только решения.
    tn,fp,fn,tp=confusion_matrix(y,d,labels=[0,1]).ravel()
    return dict(TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
        precision=float(precision_score(y,d,zero_division=0)),recall=float(recall_score(y,d,zero_division=0)),
        f1=float(f1_score(y,d,zero_division=0)),alerts_per_1000=float(1000*np.mean(d)),
        false_alerts_per_1000=float(1000*fp/len(y)))


def diagnostic_tables(test, p, threshold, out):
    f=test.copy(); f['score']=p
    f['outcome']=np.select([(f.target==1)&(p>=threshold),(f.target==0)&(p>=threshold),
                           (f.target==1)&(p<threshold)],['TP','FP','FN'],default='TN')
    summary=f.groupby('outcome').agg(observations=('target','size'), mean_load=('mean_load','mean'),
        mean_latency=('mean_latency','mean'), mean_error_rate=('mean_error_rate','mean'),
        history_measurements=('measurement_count','mean'), prior_WARN=('warning_count','mean'),
        prior_ERROR=('error_count','mean'), no_future_measurements=('future_measurements',lambda x: int(x.eq(0).sum())))
    summary.to_csv(out/'error_groups.csv')
    # Пороги диагностических срезов заданы заранее; они не оптимизируют модель.
    slice_masks={
        'Пик загрузки ≥80%': f.max_load.ge(80), 'Пик загрузки <80%': f.max_load.lt(80),
        'Пик задержки ≥40 мс': f.max_latency.ge(40), 'Пик задержки <40 мс': f.max_latency.lt(40),
        'Средняя доля ошибок ≥0,01': f.mean_error_rate.ge(.01),
        'Средняя доля ошибок <0,01': f.mean_error_rate.lt(.01),
        'В прошлом были WARN': f.warning_count.gt(0), 'В прошлом не было WARN': f.warning_count.eq(0),
        'История ≤2 измерений': f.measurement_count.le(2), 'История >2 измерений': f.measurement_count.gt(2),
        'Нет будущих измерений': f.future_measurements.eq(0)}
    rows=[]
    for name,mask in slice_masks.items():
        if mask.sum(): rows.append({'slice':name,'n':int(mask.sum()),**decision_metrics(f.loc[mask,'target'],p[mask]>=threshold)})
    pd.DataFrame(rows).to_csv(out/'error_slices.csv',index=False)
    cols=['node_code','cutoff','target','score','outcome',*FEATURES,'future_measurements']
    f.loc[f.outcome.eq('FP'),cols].sort_values('score',ascending=False).head(12).to_csv(out/'false_positive_examples.csv',index=False)
    return f, summary.reset_index().to_dict('records'), rows


def calibration_diagnostics(frame, probabilities, train_prevalence):
    """Описательная диагностика score; модель и порог не изменяются."""
    p=np.asarray(probabilities,dtype=float);y=frame.target.to_numpy()
    bins=pd.qcut(pd.Series(p),q=10,duplicates='drop')
    rows=[]
    for interval,indices in bins.groupby(bins,observed=True).groups.items():
        idx=np.asarray(indices,dtype=int)
        rows.append(dict(bin=str(interval),n=len(idx),mean_score=float(p[idx].mean()),
                         event_rate=float(y[idx].mean())))
    return dict(n=len(y),brier=float(np.mean((p-y)**2)),
        train_prior_brier=float(np.mean((train_prevalence-y)**2)),
        train_prevalence=float(train_prevalence),mean_score=float(p.mean()),
        event_rate=float(y.mean()),bins=rows,
        purpose='Диагностика после фиксации; калибратор не обучен, настройки не меняются')


def shap_analysis(model, test, out, count=None, config=CONFIG):
    import shap
    # Простая случайная выборка всей test-популяции, без балансировки классов.
    count = int(count if count is not None else config['shap_sample_size'])
    rng=np.random.default_rng(config['random_state'])
    idx=np.sort(rng.choice(len(test),min(count,len(test)),replace=False))
    sample=test.iloc[idx]
    x=model[0].transform(predictors(sample))
    explainer=shap.TreeExplainer(model[-1], model_output='raw', feature_perturbation='tree_path_dependent')
    values=explainer(x,check_additivity=True)
    sv=np.asarray(values.values); bases=np.broadcast_to(values.base_values,(len(sample),))
    margin=model[-1].predict(x, output_margin=True)
    probability=model.predict_proba(predictors(sample))[:,1]
    np.testing.assert_allclose(bases+sv.sum(axis=1),margin,atol=2e-5,rtol=2e-5)
    np.testing.assert_allclose(expit(bases+sv.sum(axis=1)),probability,atol=2e-6)
    importance=np.mean(np.abs(sv),axis=0); order=np.argsort(-importance,kind='stable')
    rows=[dict(feature=FEATURES[i],name=DEFINITIONS[FEATURES[i]][0],mean_abs_shap=float(importance[i])) for i in order]
    pd.DataFrame(rows).to_csv(out/'shap_importance.csv',index=False)
    high_idx=int(np.argmax(model.predict_proba(predictors(test))[:,1])); high=test.iloc[[high_idx]]
    hx=model[0].transform(predictors(high)); hv=explainer(hx)
    local_base=float(np.asarray(hv.base_values).ravel()[0]); local_sv=np.asarray(hv.values)[0]
    local_p=float(model.predict_proba(predictors(high))[0,1])
    assert abs(float(expit(local_base+local_sv.sum()))-local_p)<2e-6
    local=dict(row_id=high.row_id.iloc[0],target=int(high.target.iloc[0]),score=local_p,base=local_base,
        margin=float(local_base+local_sv.sum()),contributions=[dict(feature=k,value=float(hx[0,i]),
        shap=float(local_sv[i])) for i,k in enumerate(FEATURES)])
    np.savez_compressed(out/'shap_values.npz',values=sv,base=bases,x=x,y=sample.target.to_numpy(),
        probability=probability,margin=margin,row_indices=idx)
    details=dict(sample_size=len(sample),positive_labels=int(sample.target.sum()),seed=config['random_state'],
        sample_method='Простая случайная выборка test без балансировки, после фиксации модели и порога',
        scale='raw margin / логарифм шансов',perturbation='tree_path_dependent',
        max_additivity_error=float(np.max(np.abs(bases+sv.sum(axis=1)-margin))),
        top3=[r['feature'] for r in rows[:3]],importance=rows,local=local)
    dump(out/'shap_summary.json',details)
    return details


def target_associations(train, out):
    rows=[]; bins=[]
    for feature in FEATURES:
        pair=train[[feature,'target']].dropna(); x=pair[feature]
        corr=float(pair.corr().iloc[0,1]) if x.nunique()>1 else None
        rows.append(dict(feature=feature,correlation=corr,n=len(pair),
            mean_positive=float(pair.loc[pair.target.eq(1),feature].mean()),
            mean_negative=float(pair.loc[pair.target.eq(0),feature].mean())))
        if x.nunique()>1:
            b=pd.qcut(x,q=4,duplicates='drop')
            for interval,g in pair.groupby(b,observed=True):
                bins.append(dict(feature=feature,bin=str(interval),n=len(g),positive_rate=float(g.target.mean())))
    pd.DataFrame(rows).to_csv(out/'train_target_associations.csv',index=False)
    pd.DataFrame(bins).to_csv(out/'train_target_bins.csv',index=False)
    return rows,bins


def load_data(dataset, out):
    """Read telemetry sources and return normalized records plus a source audit."""
    return load_sources(Path(dataset).resolve(), Path(out))


def preprocess(frame, out, config=CONFIG):
    """Create leakage-safe node/time snapshots and chronological partitions."""
    apply_config(config)
    data, split_info = build_dataset(frame, Path(out))
    partitions = {name: data.loc[data.split.eq(name)].reset_index(drop=True)
                  for name in ('train', 'validation', 'test')}
    return data, split_info, partitions


def evaluate(models, frame, split_name='test', threshold=.5, timing=None):
    """Evaluate fitted models on one untouched temporal partition."""
    probabilities, scores = {}, {}
    for name, model in models.items():
        started = time.perf_counter()
        p = model.predict_proba(predictors(frame))[:, 1]
        elapsed = time.perf_counter() - started
        probabilities[name] = p
        scores[name] = metrics(frame.target, p, threshold)
        if timing is not None:
            timing.setdefault(name, {})[f'{split_name}_prediction_seconds'] = elapsed
        LOGGER.info('%s %s: AP=%.6f ROC-AUC=%s', split_name, name,
                    scores[name]['average_precision'], scores[name]['roc_auc'])
    return probabilities, scores


def train_model(train, validation, out, search_count=None, config=CONFIG):
    """Fit baselines, select XGBoost hyperparameters on validation AP, freeze threshold."""
    apply_config(config)
    out = Path(out)
    ratio = float((train.target == 0).sum() / train.target.sum())
    lr = config['logistic_regression']
    logistic = make_pipeline(SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True),
        StandardScaler(), LogisticRegression(class_weight=lr['class_weight'], max_iter=lr['max_iter'],
                                             random_state=config['random_state']))
    baseline_params = {**config['baseline_xgboost'], 'scale_pos_weight': ratio}
    models = {'LogisticRegression': logistic,
              'XGBoost baseline': xgb_model(baseline_params, config)}
    timing = {}
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(predictors(train), train.target)
        fit_seconds = time.perf_counter() - started
        timing[name] = dict(fit_seconds=fit_seconds)
    val_probs, validation_metrics = evaluate(models, validation, 'validation')
    baseline_winner = max(validation_metrics, key=lambda n: validation_metrics[n]['average_precision'])
    best, search = randomized_search(train, validation, out, search_count, config)
    # Refit only on train: validation remains independent for threshold selection.
    started = time.perf_counter()
    final = xgb_model(best['params'], config)
    final.fit(predictors(train), train.target)
    final_refit = time.perf_counter() - started
    models['XGBoost tuned'] = final
    started = time.perf_counter()
    val_probs['XGBoost tuned'] = final.predict_proba(predictors(validation))[:, 1]
    final_val_pred = time.perf_counter() - started
    validation_metrics['XGBoost tuned'] = metrics(validation.target, val_probs['XGBoost tuned'])
    timing['XGBoost tuned'] = dict(fit_seconds=final_refit,
                                   validation_prediction_seconds=final_val_pred)
    threshold, curve, recall_feasible = choose_threshold(validation, val_probs['XGBoost tuned'], config['alert_policy'])
    threshold_info=dict(threshold=threshold,policy=POLICY,desired_recall_reached=recall_feasible,
        selection_split='validation',test_used=False,model_training_split='train',
        validation=metrics(validation.target,val_probs['XGBoost tuned'],threshold),
        f1_threshold=float(curve.sort_values(['f1','threshold'],ascending=[False,False]).iloc[0].threshold))
    # Фиксация настроек записана ДО первого обращения к test-метрикам.
    dump(out/'selection_frozen.json',dict(threshold=threshold_info,search=search,features=FEATURES,
        split_days=config['split_days'],baseline_winner=baseline_winner,baseline_params=baseline_params))
    curve.to_csv(out/'validation_thresholds.csv',index=False)
    candidates=sorted(set([0.,.01,.02,.05,.1,.2,.3,.4,.5,.6,.8,.95,1.,threshold,threshold_info['f1_threshold']]))
    threshold_candidates=[metrics(validation.target,val_probs['XGBoost tuned'],v) for v in candidates]
    dump(out/'threshold_candidates.json',threshold_candidates)
    suppression_rows=[]
    vp=val_probs['XGBoost tuned']
    for rule,decision in [('Без подавления',vp>=threshold),
        (f'Cooldown {config["cooldown_days"]} суток',suppression(validation,vp,threshold,config['cooldown_days'])),
        ('Не менее двух измерений в истории',(vp>=threshold)&validation.measurement_count.ge(2).to_numpy())]:
        suppression_rows.append(dict(rule=rule,**decision_metrics(validation.target,decision)))
    dump(out/'validation_suppression.json',suppression_rows)
    return dict(models=models, timing=timing, val_probs=val_probs,
        validation_metrics=validation_metrics, baseline_winner=baseline_winner,
        baseline_params=baseline_params, train_class_ratio=ratio, best=best, search=search,
        final_refit=final_refit, threshold=threshold, threshold_info=threshold_info,
        curve=curve, threshold_candidates=threshold_candidates,
        suppression=suppression_rows)


def configure_logging(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    LOGGER.handlers.clear()
    LOGGER.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    for handler in (logging.StreamHandler(), logging.FileHandler(out / 'pipeline.log', encoding='utf-8')):
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def input_fingerprint(dataset, config):
    """Fingerprint data, effective config, and pipeline sources for safe caching."""
    from telemetry_hw import discover
    dataset = Path(dataset).resolve()
    files = discover(dataset)
    if not files:
        raise ValueError(f'Не найдены JSONL-файлы телеметрии в {dataset}')
    base = Path(__file__).resolve().parent
    sources = [Path(__file__), base / 'telemetry_hw.py']
    digest_payload = {
        'config': config,
        'data': {str(path.relative_to(dataset)) if path.is_relative_to(dataset) else path.name: sha256(path)
                 for path in files},
        'code': {path.name: sha256(path) for path in sources if path.is_file()},
    }
    encoded = json.dumps(digest_payload, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()
    return hashlib.sha256(encoded).hexdigest(), digest_payload


def run(dataset, out, search_count=None, config=None, force=False):
    config = dict(config or CONFIG)
    if search_count is not None:
        config['search_count'] = int(search_count)
    search_count = int(config['search_count'])
    apply_config(config)
    base=Path(__file__).resolve().parent; out=Path(out).resolve(); out.mkdir(parents=True,exist_ok=True)
    configure_logging(out)
    fingerprint, fingerprint_inputs = input_fingerprint(dataset, config)
    manifest_path = out / 'run_manifest.json'
    evaluation_path = out / 'evaluation.json'
    if not force and manifest_path.is_file() and evaluation_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding='utf-8'))
        if previous.get('fingerprint') == fingerprint:
            LOGGER.info('Конфигурация, данные и код не изменились; повторно использую готовые результаты.')
            return json.loads(evaluation_path.read_text(encoding='utf-8'))
    # Do not leave a stale success marker if forced execution fails part-way through.
    manifest_path.unlink(missing_ok=True)
    preserve(base,out)
    started=time.perf_counter(); frame,source=load_data(dataset,out)
    LOGGER.info('Загружено %d измерений с %d узлов', len(frame), source['nodes'])
    data,splits,partitions=preprocess(frame,out,config); del frame
    preprocessing_seconds=time.perf_counter()-started
    train, validation, test = (partitions[name] for name in ('train','validation','test'))
    training = train_model(train, validation, out, search_count, config)
    models=training['models']; timing=training['timing']; val_probs=training['val_probs']
    validation_metrics=training['validation_metrics']; baseline_winner=training['baseline_winner']
    baseline_params=training['baseline_params']; ratio=training['train_class_ratio']
    search=training['search']; final_refit=training['final_refit']; threshold=training['threshold']
    threshold_info=training['threshold_info']; threshold_candidates=training['threshold_candidates']
    suppression_rows=training['suppression']; vp=val_probs['XGBoost tuned']
    final=models['XGBoost tuned']
    # Далее test только для окончательной оценки и диагностики; никаких изменений настроек.
    test_probs,test_metrics=evaluate(models,test,'test',timing=timing)
    prediction_frame=test[['row_id','node_code','cutoff','target',*FEATURES,'future_measurements']].copy()
    val_frame=validation[['row_id','node_code','cutoff','target',*FEATURES,'future_measurements']].copy()
    for name,p in test_probs.items():
        prediction_frame[name]=p
        val_frame[name]=val_probs[name]
    threshold_info['test']=metrics(test.target,test_probs['XGBoost tuned'],threshold)
    calibration={split:calibration_diagnostics(part,p,float(train.target.mean()))
        for split,part,p in [('validation',validation,vp),('test',test,test_probs['XGBoost tuned'])]}
    dump(out/'calibration_diagnostics.json',calibration)
    diag,error_groups,error_slices=diagnostic_tables(test,test_probs['XGBoost tuned'],threshold,out)
    prediction_frame['outcome']=diag.outcome
    prediction_frame.to_parquet(out/'test_predictions.parquet',index=False)
    val_frame.to_parquet(out/'validation_predictions.parquet',index=False)
    alerts=[]
    for split,f,probs in [('validation',validation,vp),('test',test,test_probs['XGBoost tuned'])]:
        d=probs>=threshold
        counts=f.loc[d].groupby('node_code').size()
        for cutoff,g in f.groupby('cutoff'):
            ix=g.index.to_numpy();n_alert=int(d[ix].sum())
            alerts.append(dict(split=split,cutoff=cutoff,n=len(g),alerts=n_alert,unique_nodes=n_alert,
                repeated_alert_rows_in_split=int((counts-1).clip(lower=0).sum())))
    dump(out/'alert_load.json',alerts)
    t=time.perf_counter();shap_info=shap_analysis(final,test,out,config=config);shap_seconds=time.perf_counter()-t
    associations,bins=target_associations(train,out)
    for name,model in models.items():
        joblib.dump(model,out/({'LogisticRegression':'logistic','XGBoost baseline':'xgb_baseline','XGBoost tuned':'xgb_tuned'}[name]+'.joblib'))
    # Persist the native Booster directly; this is stable across sklearn tag API changes.
    final[-1].get_booster().save_model(out/'xgb_tuned.ubj')
    versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','xgboost',
        'shap','matplotlib','seaborn','joblib','pyarrow','nbformat','nbclient','pytest']}
    dump(out/'package_versions.json',dict(python=platform.python_version(),platform=platform.platform(),packages=versions))
    summary=dict(source=source,splits=splits,window_days=WINDOW,horizon_days=HORIZON,
        features=FEATURES,definitions=DEFINITIONS,seed=SEED,observations=len(data),positives=int(data.target.sum()),
        positive_rate=float(data.target.mean()),validation_metrics=validation_metrics,test_metrics=test_metrics,
        timing=timing,search=search,threshold=threshold_info,baseline_params=baseline_params,
        baseline_winner=baseline_winner,train_class_ratio=ratio,shap=shap_info,associations=associations,
        target_bins=bins,error_groups=error_groups,error_slices=error_slices,
        alert_load=alerts,suppression=suppression_rows,threshold_candidates=threshold_candidates,
        calibration=calibration,
        versions=versions,preprocessing_seconds=preprocessing_seconds,shap_seconds=shap_seconds,
        total_model_fit_seconds=sum(v['fit_seconds'] for v in timing.values())+search['sum_candidate_fit_seconds'],
        xgb_tuned_workflow_seconds=search['search_seconds']+final_refit,
        total_analysis_seconds=time.perf_counter()-started,
        completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),
        leakage_checks=dict(features_strictly_before_cutoff=True,label_windows_disjoint=True,
            cross_split_history_and_labels_disjoint=True,preprocessing_fit_train_only=True,
            hyperparameters_test_used=False,threshold_test_used=False,node_id_in_features=False,
            event_availability_verified=False),
        test_sample_sha256=hashlib.sha256('\n'.join(test.row_id).encode()).hexdigest())
    dump(evaluation_path,summary)
    atomic_dump(manifest_path,dict(fingerprint=fingerprint,inputs=fingerprint_inputs,
        completed_at_utc=pd.Timestamp.now(tz='UTC').isoformat(),output=str(out)))
    LOGGER.info('Полный расчёт завершён; результаты: %s', out)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=CONFIG_PATH,help='JSON-конфигурация конвейера')
    parser.add_argument('--dataset',type=Path,help='Переопределить каталог входных JSONL')
    parser.add_argument('--output',type=Path,help='Переопределить каталог артефактов')
    parser.add_argument('--search-count',type=int,help='Переопределить число кандидатов XGBoost')
    parser.add_argument('--force',action='store_true',help='Переобучить модель и обновить артефакты')
    args=parser.parse_args()
    config_path=args.config.resolve()
    config=load_config(config_path)
    base=config_path.parent
    dataset=args.dataset or (base / config['paths']['dataset'])
    output=args.output or (base / config['paths']['output'])
    LOGGER.info('Конфигурация: %s; вход: %s; результаты: %s', config_path, dataset, output)
    run(dataset,output,args.search_count,config,args.force)


if __name__=='__main__':main()
