"""Пять заданий по телеметрии: исходные файлы никогда не изменяются."""
from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parent / '.mplconfig'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

METRICS = ['load_pct', 'latency_ms', 'error_rate']
FEATURES = ['avg_load', 'max_load', 'avg_error_rate', 'avg_latency', 'failure_count']
NULL_WORDS = {'null', 'none', 'nan', 'n/a', 'na', 'unknown', '-', '?', '<na>', 'nat'}
TITLES = {'load_pct': 'Загрузка узлов', 'latency_ms': 'Задержка сети', 'error_rate': 'Доля ошибок'}
UNITS = {'load_pct': '%', 'latency_ms': 'мс', 'error_rate': 'доля от 0 до 1'}
REPORT_NAMES = ['telemetry_profiling.md', 'telemetry_trends.md', 'telemetry_integration.md',
                'telemetry_features.md', 'telemetry_outliers.md']
CHART_NAMES = ['load_by_hour.png', 'load_by_day.png', 'error_rate_by_day.png'] + [f'{c}_boxplot.png' for c in METRICS]


def log(message):
    print(message, flush=True)


def dump(path, value):
    def default(x):
        if isinstance(x, np.generic):
            return x.item()
        return str(x)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=default, allow_nan=False) + '\n', encoding='utf-8')


def fmt(value):
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return '—'
    if isinstance(value, (float, np.floating)):
        return f'{value:.8g}'
    return str(value).replace('|', '\\|').replace('\n', ' ')


def table(rows):
    if isinstance(rows, pd.DataFrame):
        rows = rows.to_dict('records')
    if not rows:
        return 'Нет наблюдений.\n'
    keys = list(rows[0])
    return ('| ' + ' | '.join(keys) + ' |\n| ' + ' | '.join(['---'] * len(keys)) + ' |\n' +
            '\n'.join('| ' + ' | '.join(fmt(row.get(k)) for k in keys) + ' |' for row in rows) + '\n\n')


def relative_link(path, report_dir):
    return Path(os.path.relpath(Path(path).resolve(), Path(report_dir).resolve())).as_posix()


def discover(root, name='telemetry', extension='jsonl'):
    """Accept a dataset root, raw root, source directory, single file, or glob."""
    import glob
    root = Path(root).expanduser()
    if any(x in str(root) for x in '*?['):
        return sorted({Path(p).resolve() for p in glob.glob(str(root), recursive=True)
                       if Path(p).is_file() and Path(p).suffix == f'.{extension}'})
    if root.is_file():
        return [root.resolve()] if root.suffix == f'.{extension}' else []
    candidates = []
    for pattern in [f'raw/{name}/dt=*/*.{extension}', f'late_delivery/{name}/dt=*/*.{extension}',
                    f'{name}/dt=*/*.{extension}', f'dt=*/*.{extension}', f'*.{extension}']:
        candidates.extend(root.glob(pattern))
    return sorted({p.resolve() for p in candidates if p.is_file()})


def default_dataset(base):
    # Prefer conventions from data_preparation_hw.py; select one root, never union copies.
    for path in [base / 'dataset', base / 'out', base, Path('/content/out')]:
        if discover(path):
            return path.resolve()
    return (base / 'dataset').resolve()


def load_telemetry(files, output):
    """Read JSONL with bounded decoding buffers; keep every decoded object, audit rejects."""
    parts, pending, manifest = [], [], []
    quarantine = output / 'telemetry_rejected_lines.jsonl'
    physical = rejected = blank = 0
    with quarantine.open('w', encoding='utf-8') as bad:
        for path in files:
            digest = hashlib.sha256()
            count = accepted = 0
            with path.open('rb') as stream:
                for number, raw_line in enumerate(stream, 1):
                    digest.update(raw_line)
                    physical += 1
                    count += 1
                    try:
                        if not raw_line.strip():
                            blank += 1
                            raise ValueError('Пустая физическая строка')
                        item = json.loads(raw_line.decode('utf-8'))
                        if not isinstance(item, dict):
                            raise ValueError('JSON не является объектом')
                        # Keep nested extra fields as deterministic JSON, suitable for duplicated().
                        item = {k: json.dumps(v, sort_keys=True, ensure_ascii=False) if isinstance(v, (dict, list)) else v
                                for k, v in item.items()}
                    except (ValueError, UnicodeDecodeError) as exc:
                        rejected += 1
                        bad.write(json.dumps({'file': str(path), 'line': number, 'reason': str(exc),
                                              'raw_base64': base64.b64encode(raw_line).decode('ascii')}, ensure_ascii=False) + '\n')
                        continue
                    pending.append(item)
                    accepted += 1
                    if len(pending) >= 100_000:
                        parts.append(pd.DataFrame(pending))
                        pending = []
            manifest.append({'path': str(path), 'sha256': digest.hexdigest(), 'bytes': path.stat().st_size,
                             'physical_lines': count, 'accepted': accepted, 'rejected': count - accepted})
    if pending:
        parts.append(pd.DataFrame(pending))
    frame = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    assert physical == len(frame) + rejected
    return frame, {'files': len(files), 'physical_lines': physical, 'rejected_lines': rejected,
                   'blank_lines': blank, 'manifest': manifest}


def clean_text(series, case=None):
    s = series.astype('string').str.replace('\u00a0', ' ', regex=False).str.strip()
    s = s.mask(s.str.lower().isin(NULL_WORDS) | s.eq(''))
    return s.str.upper() if case == 'upper' else s.str.lower() if case == 'lower' else s


def normalize_node(series, pattern=r'ND[0-9]{6}'):
    s = clean_text(series, 'upper')
    return s.where(s.str.fullmatch(pattern, na=False)) if pattern else s


def parse_times(series, source_tz='UTC'):
    """Naive values use the configured source zone; offset-bearing values keep their instant."""
    s = clean_text(series)
    # Explicitly reject bare numeric epochs: their units are unknown.
    s = s.mask(s.str.fullmatch(r'[+-]?\d+(\.\d+)?', na=False))
    aware = s.str.contains(r'(?:Z|[+-]\d{2}:?\d{2})$', case=False, regex=True, na=False)
    result = pd.Series(pd.NaT, index=s.index, dtype='datetime64[ns, UTC]')
    for dayfirst, mask in [(True, s.str.match(r'^\d{2}\.\d{2}\.\d{4}', na=False)),
                           (False, ~s.str.match(r'^\d{2}\.\d{2}\.\d{4}', na=False))]:
        naive_mask = mask & ~aware
        dates = pd.to_datetime(s.loc[naive_mask], format='mixed', errors='coerce', dayfirst=dayfirst)
        result.loc[naive_mask] = dates.dt.tz_localize(source_tz, ambiguous='NaT', nonexistent='NaT').dt.tz_convert('UTC')
        aware_mask = mask & aware
        result.loc[aware_mask] = pd.to_datetime(s.loc[aware_mask], format='mixed', errors='coerce', utc=True, dayfirst=dayfirst)
    return result


def missing_summary(raw):
    rows = []
    basic = raw.isna().sum()
    for col in raw:
        text = raw[col].astype('string')
        blanks = text.str.strip().eq('').fillna(False)
        placeholders = text.str.strip().str.lower().isin(NULL_WORDS)
        total = int((raw[col].isna() | blanks | placeholders).sum())
        rows.append({'Столбец': col, 'NULL/NaN': int(basic[col]), 'Пустые/пробельные': int(blanks.sum()),
                     'Строки-маркеры NULL': int(placeholders.sum()), 'Всего пропусков': total,
                     'Доля, %': 100 * total / len(raw) if len(raw) else 0.0})
    return rows


def normalize(raw, timestamp_column=None, source_tz='UTC', node_pattern=r'ND[0-9]{6}'):
    candidates = ['ts', 'timestamp', 'event_ts', 'datetime', 'time', 'event_time']
    if timestamp_column and timestamp_column not in raw:
        raise ValueError(f'Столбец времени {timestamp_column!r} отсутствует')
    ts_col = timestamp_column or next((c for c in candidates if c in raw), None)
    required = ['node_id', 'level'] + METRICS
    absent = [c for c in required if c not in raw]
    df = pd.DataFrame(index=raw.index)
    for col in required:
        values = raw[col] if col in raw else pd.Series(pd.NA, index=raw.index, dtype='string')
        df[col] = clean_text(values)
    original_node = df['node_id']
    df['node_id'] = normalize_node(original_node, node_pattern)
    audit = {'missing_columns': absent, 'timestamp_column': ts_col, 'source_timezone': source_tz,
             'node_pattern': node_pattern, 'missing_node_id': int(original_node.isna().sum()),
             'invalid_node_id': int((original_node.notna() & df.node_id.isna()).sum()),
             'numeric': {}}
    df['level'] = clean_text(df.level, 'lower')
    for col in METRICS:
        text = df[col]
        numeric = pd.to_numeric(text, errors='coerce').astype('float64')
        malformed = text.notna() & numeric.isna()
        nonfinite = numeric.notna() & ~np.isfinite(numeric)
        finite = numeric.where(np.isfinite(numeric))
        invalid = finite.lt(0)
        if col in ['load_pct', 'error_rate']:
            invalid |= finite.gt(100 if col == 'load_pct' else 1)
        audit['numeric'][col] = {'source_missing': int(text.isna().sum()), 'malformed': int(malformed.sum()),
                                 'nonfinite': int(nonfinite.sum()), 'domain_invalid': int(invalid.sum()),
                                 'usable': int((finite.notna() & ~invalid).sum())}
        df[col] = finite.mask(invalid)
    times = raw[ts_col] if ts_col else pd.Series(pd.NA, index=raw.index, dtype='string')
    df['timestamp'] = parse_times(times, source_tz)
    audit['missing_timestamp'] = int(clean_text(times).isna().sum())
    audit['invalid_timestamp'] = int((clean_text(times).notna() & df.timestamp.isna()).sum())
    return df, audit


def iqr_stats(series):
    s = pd.to_numeric(series, errors='coerce')
    s = s.loc[np.isfinite(s)].astype('float64')
    empty = {'n': 0, 'q1': None, 'median': None, 'q3': None, 'iqr': None, 'lower': None,
             'upper': None, 'outliers': 0, 'outlier_pct': 0.0, 'min': None, 'max': None}
    if s.empty:
        return empty
    q1, med, q3 = (float(x) for x in s.quantile([.25, .5, .75]))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_outliers = int((s.lt(lower) | s.gt(upper)).sum())
    return dict(n=len(s), q1=q1, median=med, q3=q3, iqr=iqr, lower=lower, upper=upper,
                outliers=n_outliers, outlier_pct=100 * n_outliers / len(s), min=float(s.min()), max=float(s.max()))


def build_timeseries(df):
    valid = df.loc[df.timestamp.notna()].set_index('timestamp').sort_index()
    hourly = valid.resample('h').agg(avg_load=('load_pct', 'mean'), records=('load_pct', 'count'))
    daily = valid.resample('D').agg(avg_load=('load_pct', 'mean'), avg_error_rate=('error_rate', 'mean'),
                                   load_records=('load_pct', 'count'), error_records=('error_rate', 'count'))
    return hourly, daily


def detect_anomalies(hourly, daily):
    rows, rules = [], []
    def check(series, key, mode):
        stats = iqr_stats(series)
        rules.append({'Правило': key, 'N': stats['n'], 'Q1': stats['q1'], 'Q3': stats['q3'],
                      'IQR': stats['iqr'], 'Нижняя граница': stats['lower'], 'Верхняя граница': stats['upper']})
        if not stats['n']:
            return
        for ts, value in series.dropna().items():
            low = value < stats['lower'] and mode == 'both'
            high = value > stats['upper']
            if low or high:
                rows.append({'Время UTC': ts.isoformat(), 'Тип': key + (' — снижение' if low else ' — превышение'),
                             'Значение': float(value), 'Граница': stats['lower'] if low else stats['upper']})
    check(hourly.avg_load, 'Часовая загрузка, %', 'both')
    # Resampling inserts gaps: diff() stays null across missing intervals.
    check(hourly.avg_load.diff().abs(), 'Модуль часового изменения, п.п.', 'high')
    check(daily.avg_error_rate, 'Суточная доля ошибок', 'high')
    check(daily.avg_error_rate.diff().clip(lower=0), 'Суточный прирост доли ошибок', 'high')
    return sorted(rows, key=lambda r: (r['Время UTC'], r['Тип'])), rules


def node_features(df, cutoff=None):
    selected = df.loc[df.node_id.notna()].copy()
    if cutoff is not None:
        selected = selected.loc[selected.timestamp.notna() & selected.timestamp.lt(cutoff)].copy()
    selected['is_error'] = selected.level.eq('error').fillna(False).astype('int64')
    result = selected.groupby('node_id', sort=True, observed=True).agg(
        avg_load=('load_pct', 'mean'), max_load=('load_pct', 'max'), avg_error_rate=('error_rate', 'mean'),
        avg_latency=('latency_ms', 'mean'), failure_count=('is_error', 'sum')).reset_index()
    result['failure_count'] = result.failure_count.astype('int64')
    assert result.node_id.is_unique and result.node_id.notna().all()
    assert list(result) == ['node_id'] + FEATURES
    return result


def telemetry_daily(df):
    valid = df.loc[df.node_id.notna() & df.timestamp.notna()].copy()
    valid['event_day'] = valid.timestamp.dt.floor('D') + pd.Timedelta(days=1)
    result = valid.groupby(['node_id', 'event_day'], observed=True, sort=True).agg(
        avg_load=('load_pct', 'mean'), avg_error_rate=('error_rate', 'mean'), avg_latency=('latency_ms', 'mean'),
        measurement_count=('load_pct', 'size')).reset_index()
    assert not result.duplicated(['node_id', 'event_day']).any()
    return result


def left_join_cdr(cdr, daily, source_tz='UTC', node_pattern=r'ND[0-9]{6}'):
    left = cdr.copy()
    left['node_id'] = normalize_node(left.node_id, node_pattern)
    left['event_time'] = parse_times(left.event_ts, source_tz)
    left['event_day'] = left.event_time.dt.floor('D')
    # Drop null right keys explicitly: pandas unlike SQL may match null to null.
    right = daily.dropna(subset=['node_id', 'event_day'])
    result = left.merge(right, how='left', on=['node_id', 'event_day'], validate='many_to_one', indicator=True, sort=False)
    assert len(result) == len(cdr)
    return result


def iter_cdr_batches(source, audit, batch_size=100_000):
    """Prefer existing clean Parquet; fall back to all raw/late CSV with physical validation."""
    import pyarrow.dataset as ds
    source = Path(source)
    if source.suffix == '.parquet' or (source.is_dir() and list(source.glob('*.parquet'))):
        dataset = ds.dataset(source, format='parquet', exclude_invalid_files=True)
        audit.update(format='parquet', files=len(dataset.files), source_files=sorted(dataset.files))
        columns = [c for c in ['record_id', 'node_id', 'event_ts'] if c in dataset.schema.names]
        if not {'node_id', 'event_ts'} <= set(columns):
            raise ValueError('Для интеграции CDR необходимы node_id и event_ts')
        for batch in dataset.scanner(columns=columns, batch_size=batch_size).to_batches():
            yield batch.to_pandas()
        return
    import csv
    paths = discover(source, 'cdr', 'csv')
    audit['format'] = 'csv'
    audit['files'] = len(paths)
    for path in paths:
        pending = []
        with path.open('rb') as stream:
            header = next(csv.reader([next(stream).decode('utf-8').strip()]))
            if not {'node_id', 'event_ts', 'record_id'} <= set(header):
                raise ValueError(f'Неподдерживаемая схема CDR: {path}')
            for line_no, line in enumerate(stream, 2):
                try:
                    text = line.decode('utf-8')
                    values = next(csv.reader([text]))
                    if len(values) != len(header) or not line.endswith(b'\n') or not values[0].isdigit():
                        raise ValueError('Неверное число полей, record_id или оборванная строка')
                except (UnicodeDecodeError, ValueError, csv.Error) as exc:
                    audit.setdefault('rejected', []).append({'file': str(path), 'line': line_no, 'reason': str(exc),
                                                            'raw_base64': base64.b64encode(line).decode('ascii')})
                    continue
                record = dict(zip(header, values))
                pending.append({c: record[c] for c in ['record_id', 'node_id', 'event_ts']})
                if len(pending) >= batch_size:
                    yield pd.DataFrame(pending)
                    pending = []
        if pending:
            yield pd.DataFrame(pending)


def validate_integration(df, source, output, source_tz='UTC', node_pattern=r'ND[0-9]{6}'):
    if source is None or not Path(source).exists():
        return {'available': False, 'reason': 'CDR не найден; интеграция не выполнялась.'}
    daily = telemetry_daily(df)
    daily.to_parquet(output / 'telemetry_previous_day.parquet', index=False)
    stats = {'available': True, 'source': str(Path(source).resolve()), 'rows_before': 0, 'rows_after': 0,
             'matched': 0, 'missing_node': 0, 'invalid_node': 0, 'invalid_time': 0,
             'unknown_node_rows': 0, 'outside_date_range': 0, 'matched_missing_load': 0,
             'cdr_min': None, 'cdr_max': None, 'right_rows': len(daily), 'right_duplicate_keys': 0}
    physical_audit, ids, sample = {}, [], []
    nodes = set(df.node_id.dropna())
    for batch in iter_cdr_batches(source, physical_audit):
        joined = left_join_cdr(batch, daily, source_tz, node_pattern)
        matched = joined['_merge'].eq('both')
        stats['rows_before'] += len(batch)
        stats['rows_after'] += len(joined)
        if stats['rows_before'] // 1_000_000 != (stats['rows_before'] - len(batch)) // 1_000_000:
            log(f'Проверено CDR: {stats["rows_before"]:,} строк')
        stats['matched'] += int(matched.sum())
        stats['missing_node'] += int(clean_text(batch.node_id).isna().sum())
        stats['invalid_node'] += int((clean_text(batch.node_id).notna() & normalize_node(batch.node_id, node_pattern).isna()).sum())
        stats['invalid_time'] += int(joined.event_time.isna().sum())
        stats['unknown_node_rows'] += int((joined.node_id.notna() & ~joined.node_id.isin(nodes)).sum())
        stats['matched_missing_load'] += int((matched & joined.avg_load.isna()).sum())
        if not daily.empty:
            stats['outside_date_range'] += int((joined.event_day.lt(daily.event_day.min()) | joined.event_day.gt(daily.event_day.max())).sum())
        valid_time = joined.event_time.dropna()
        if not valid_time.empty:
            stats['cdr_min'] = min(filter(lambda x: x is not None, [stats['cdr_min'], valid_time.min()]))
            stats['cdr_max'] = max(filter(lambda x: x is not None, [stats['cdr_max'], valid_time.max()]))
        if 'record_id' in batch:
            ids.append(clean_text(batch.record_id).dropna().to_numpy(dtype=str))
        if sum(len(x) for x in sample) < 20:
            sample.append(joined.head(20 - sum(len(x) for x in sample)).drop(columns='_merge'))
    if not stats['rows_before']:
        return {'available': False, 'reason': 'Источник CDR не содержит доступных строк.', 'source': str(source)}
    if ids:
        all_ids = np.concatenate(ids)
        stats['duplicate_id_excess'] = len(all_ids) - len(np.unique(all_ids))
    stats['match_pct'] = 100 * stats['matched'] / stats['rows_before']
    stats['unmatched'] = stats['rows_after'] - stats['matched']
    stats['raw_csv_rejected'] = len(physical_audit.get('rejected', [])) if physical_audit.get('format') == 'csv' else None
    dump(output / 'cdr_input_audit.json', physical_audit)
    pd.concat(sample, ignore_index=True).to_parquet(output / 'cdr_telemetry_sample.parquet', index=False)
    assert stats['rows_before'] == stats['rows_after']
    return stats


def generate_charts(df, hourly, daily, output):
    sns.set_theme(style='whitegrid', font='DejaVu Sans', context='notebook')
    for data, col, filename, title, unit in [
        (hourly, 'avg_load', 'load_by_hour.png', 'Средняя загрузка узлов по хронологическим часам', '%'),
        (daily, 'avg_load', 'load_by_day.png', 'Средняя загрузка узлов по календарным дням', '%'),
        (daily, 'avg_error_rate', 'error_rate_by_day.png', 'Средняя доля ошибок по календарным дням', 'доля от 0 до 1')]:
        fig, ax = plt.subplots(figsize=(13, 4.5))
        if data[col].notna().any():
            ax.plot(data.index, data[col], color='#24689b', linewidth=1.1)
            locator = mdates.AutoDateLocator(minticks=5, maxticks=10)
            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(
                locator, formats=['%Y', '%m.%Y', '%d.%m', '%H:%M', '%H:%M', '%S'],
                zero_formats=['', '%Y', '%m.%Y', '%d.%m', '%H:%M', '%H:%M'],
                offset_formats=['', '%Y', '%m.%Y', '%d.%m.%Y', '%d.%m.%Y', '%d.%m.%Y %H:%M']))
        else:
            ax.text(.5, .5, 'Нет пригодных наблюдений', ha='center', transform=ax.transAxes)
        ax.set(title=title, xlabel='Время (UTC)', ylabel=f'Среднее значение, {unit}')
        ax.grid(True, alpha=.35)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=200)
        plt.close(fig)
    for col in METRICS:
        fig, ax = plt.subplots(figsize=(11, 4))
        values = df[col].dropna()
        if not values.empty:
            sns.boxplot(x=values, ax=ax, color='#82b5cf', whis=1.5,
                        flierprops={'markersize': 2, 'alpha': .15, 'marker': '.'})
        else:
            ax.text(.5, .5, 'Нет пригодных наблюдений', ha='center', transform=ax.transAxes)
        ax.set(title=f'{TITLES[col]}: распределение и выбросы по IQR', xlabel=f'{TITLES[col]}, {UNITS[col]}', ylabel='Все наблюдения')
        ax.grid(True, axis='x', alpha=.35)
        fig.tight_layout()
        fig.savefig(output / f'{col}_boxplot.png', dpi=200)
        plt.close(fig)


def duplicate_summary(raw, normalized):
    result = {'exact': int(raw.duplicated().sum()), 'composite_excess': None, 'event_id_excess': None}
    for key in ['event_id', 'measurement_id', 'record_id']:
        if key in raw:
            valid = clean_text(raw[key]).dropna()
            result.update(event_id_column=key, event_id_excess=int(valid.duplicated().sum()))
            break
    valid = normalized.loc[normalized.node_id.notna() & normalized.timestamp.notna()]
    result['composite_excess'] = int(valid.duplicated(['node_id', 'timestamp']).sum())
    return result


def describe_numeric(df):
    rows = []
    for col in METRICS:
        s = df[col].dropna()
        rows.append({'Метрика': col, 'N': len(s), 'Минимум': s.min(), 'Среднее': s.mean(), 'Медиана': s.median(),
                     'Максимум': s.max(), 'Станд. отклонение': s.std(), 'P01': s.quantile(.01),
                     'P95': s.quantile(.95), 'P99': s.quantile(.99)})
    return rows


def outlier_table(stats):
    return table([{'Метрика': c, 'N': s['n'], 'Q1': s['q1'], 'Медиана': s['median'], 'Q3': s['q3'],
                   'IQR': s['iqr'], 'Нижняя граница': s['lower'], 'Верхняя граница': s['upper'],
                   'Выбросов': s['outliers'], 'Доля, %': s['outlier_pct'], 'Минимум': s['min'], 'Максимум': s['max']}
                  for c, s in stats.items()])


def write_reports(raw, df, inventory, audit, duplicate, outliers, hourly, daily, anomalies, rules,
                  features, integration, output, report_dir, cutoff):
    n = len(raw)
    link = lambda name: relative_link(output / name, report_dir)
    available = bool(n)
    status = (f'Выполнено на предоставленных исходных файлах: **{n:,} записей**, файлов: **{inventory["files"]}**.' if available else
              '**Вычисление на данных не выполнено:** пригодные JSONL-записи не найдены. Численные результаты, PNG и признаки требуют запуска с исходным набором; отсутствующие результаты не заменены вымышленными.')
    common = ('\n\n' + status + '\n\n'
              'Контекст: учебный проект по аналитике телекоммуникационной сети. Локальный комплект описан в '
              '`generator/README.md` как синтетический учебный набор; его показатели не являются измерениями '
              'действующей сети Ростелекома. Анализ использует уже существующие файлы, генератор не запускается. '
              'Для другого входного каталога происхождение данных следует проверять отдельно.\n\n'
              f'Время приведено к UTC. У наивных дат принята исходная зона `{audit["source_timezone"]}` '
              '(настройка `--source-timezone`); это явное допущение, поскольку смещения в исходной схеме не указаны.\n')
    period = f'{fmt(df.timestamp.min())} — {fmt(df.timestamp.max())}' if df.timestamp.notna().any() else 'не определён'
    policy = ('\n## Общие правила подготовки\n\n'
              'Сырые файлы и дубликаты сохранены. Для аналитической копии пробелы и маркеры '
              '`null/none/nan/n/a/na/unknown/-/?/<na>/nat` превращены в пропуски. '
              'Числа разобраны через `pd.to_numeric(errors="coerce")`; ошибки преобразования учтены отдельно. '
              'Допустимые области: `load_pct ∈ [0,100]`, `error_rate ∈ [0,1]`, `latency_ms ≥ 0`; '
              'бесконечности недопустимы. Ошибочные значения заменены на NaN только в соответствующем '
              'аналитическом столбце. Другие значения той же записи сохранены. Пропуски не заполнены нулями. '
              'Статистика, IQR, признаки и графики используют эту аналитическую копию. '
              'Записи без времени участвуют в общем профиле и в признаках без cutoff, но не во временных рядах. '
              'Записи без корректного узла участвуют в общем профиле и временных рядах, но не в группировке по узлам.\n')
    validation_rows = [{'Метрика': c, 'Пропуски источника': a['source_missing'], 'Неверное число': a['malformed'],
                        'Бесконечность': a['nonfinite'], 'Вне области': a['domain_invalid'], 'Пригодно': a['usable']}
                       for c, a in audit['numeric'].items()]
    schema = [{'Столбец': c, 'Тип при чтении pandas': str(raw[c].dtype),
               'Аналитический тип': str(df[c].dtype) if c in df else ('datetime64[ns, UTC] → timestamp' if c == audit['timestamp_column'] else 'сохранён только в исходном профиле')}
              for c in raw]
    missing = missing_summary(raw)
    exact_pct = 100 * duplicate['exact'] / n if n else 0.0
    bad_numeric = sum(a['malformed'] + a['nonfinite'] + a['domain_invalid'] for a in audit['numeric'].values())
    body = '# Задание 1. Профилирование телеметрии\n' + common
    body += '\n## Обзор набора\n\n' + table([
        {'Показатель': 'Загружено файлов', 'Значение': inventory['files']},
        {'Показатель': 'Физических строк JSONL', 'Значение': inventory['physical_lines']},
        {'Показатель': 'Принято JSON-объектов', 'Значение': n},
        {'Показатель': 'Не принято физических строк (включая пустые)', 'Значение': inventory['rejected_lines']},
        {'Показатель': 'Пустых физических строк', 'Значение': inventory['blank_lines']},
        {'Показатель': 'Столбцов в объединённой исходной схеме', 'Значение': len(raw.columns)},
        {'Показатель': 'Поле времени', 'Значение': audit['timestamp_column']},
        {'Показатель': 'Период UTC', 'Значение': period},
        {'Показатель': 'Различных непустых исходных node_id', 'Значение': raw.node_id.nunique() if 'node_id' in raw else 0},
        {'Показатель': 'Корректных нормализованных узлов', 'Значение': df.node_id.nunique()},
        {'Показатель': 'Отсутствующие обязательные столбцы', 'Значение': ', '.join(audit['missing_columns']) or 'нет'}])
    body += '\nОбъединяются все партиции raw и late_delivery выбранного каталога; копии набора из других каталогов автоматически не добавляются. '
    body += f'Состав и SHA256 каждого файла: [input_manifest.json]({link("input_manifest.json")}). '
    body += f'Неразобранные строки с путём, номером строки и исходными байтами в Base64: [аудит JSONL]({link("telemetry_rejected_lines.jsonl")}).\n'
    body += '\n## Схема\n\n' + table(schema) + '\n## Пропуски\n\n'
    body += 'Базовый подсчёт — `df.isna().sum()` на исходном объединённом DataFrame. Дополнительно учтены пустые строки и строковые маркеры; категории в таблице не пересекаются. Знаменатель доли — число принятых JSON-объектов.\n\n' + table(missing)
    body += '\n## Дубликаты\n\n'
    body += f'Полные повторы по всем исходным столбцам: **{duplicate["exact"]}** ({exact_pct:.6f}%); применяется `raw.duplicated().sum()`, считаются экземпляры сверх первого. Служебный путь файла в сравнение не входит. '
    body += f'Повторы нормализованной пары `(node_id, timestamp)` среди корректных ключей: **{duplicate["composite_excess"]}**. Такая пара — кандидатный ключ измерения, но разные показания в одну секунду могут быть самостоятельными событиями; автоматически удалять их нельзя. '
    body += (f'Повторы идентификатора `{duplicate["event_id_column"]}`: {duplicate["event_id_excess"]}.\n' if duplicate.get('event_id_column') else 'Отдельного идентификатора события в схеме нет.\n')
    body += policy + '\n## Контроль качества значений\n\n' + table(validation_rows)
    body += f'Пустых/NULL узлов: **{audit["missing_node_id"]}**; непустых идентификаторов вне шаблона `{audit["node_pattern"]}`: **{audit["invalid_node_id"]}**. '
    body += f'Пропущено время: **{audit["missing_timestamp"]}**; не разобрано непустое время: **{audit["invalid_timestamp"]}**. '
    body += f'Примеры проблемных значений с индексами исходного DataFrame: [validation_samples.json]({link("validation_samples.json")}).\n'
    body += '\n## IQR-выбросы\n\n'
    body += 'Для каждого показателя: `Q1=P25`, `Q3=P75`, `IQR=Q3−Q1`, `L=Q1−1,5·IQR`, `U=Q3+1,5·IQR`. '
    body += 'Используется линейная интерполяция квантилей pandas. Выброс — строго `x<L` или `x>U`; границы включены в обычные наблюдения. '
    body += 'Знаменатель процента — N пригодных значений соответствующей метрики, а не все строки. Физически неверные значения показаны отдельно выше.\n\n' + outlier_table(outliers)
    body += '\n## Описательная статистика\n\n' + table(describe_numeric(df))
    body += '\n## Выводы\n\n'
    if available:
        missing_total = sum(r['Всего пропусков'] for r in missing)
        body += f'1. В исходных {len(raw.columns)} столбцах обнаружено {missing_total} пропущенных ячеек из {n * len(raw.columns)}; после проверки чисел дополнительно выявлено {bad_numeric} недопустимых значений в трёх метриках.\n'
        body += f'2. Полных повторов {duplicate["exact"]}; повторов пары узел×время сверх первого — {duplicate["composite_excess"]}. '
        body += ('Повторные полные записи могут смещать средние и failure_count; их удаление требует подтверждённого ключа.\n' if duplicate['exact'] else
                 'Точное дублирование не обнаружено; совпадения узла и секунды требуют проверки дискретности времени и различий показаний.\n')
        body += f'3. Пригодность времени: {int(df.timestamp.notna().sum())} из {n}; пригодность узла: {int(df.node_id.notna().sum())} из {n}. '
        body += ('По этим двум критериям все измерения доступны для временной и узловой агрегации.\n' if df.timestamp.notna().all() and df.node_id.notna().all() else
                 'Записи с непригодным временем или ключом ограничивают соответствующие виды агрегации.\n')
        body += '4. ' + '; '.join(f'{c}: {s["outliers"]} IQR-выбросов ({s["outlier_pct"]:.4f}%)' for c, s in outliers.items()) + '. Экстремумы задержки и ошибок могут отражать сбои; статистический критерий сам по себе не доказывает аварию.\n'
        body += '5. Рекомендуется сохранять исходный слой, контролировать полноту новых загрузок, согласовать зону времени и идентификаторы, маркировать выбросы и проверять их по журналу инцидентов. Автоматическое удаление допустимых экстремумов не выполняется.\n'
    else:
        body += 'Содержательные выводы о полноте и сетевых сбоях требуют исходных измерений. Нулевые счётчики выше описывают пустой вход, а не качество действующей сети.\n'
    (report_dir / REPORT_NAMES[0]).write_text(body, encoding='utf-8')

    body = '# Задание 2. Анализ временных рядов телеметрии\n' + common
    body += f'\n## Период и методика\n\nПериод наблюдений: **{period}**. '
    body += f'Из временного анализа исключено {int(df.timestamp.isna().sum())} записей без пригодной даты; их метрики остаются в общем профиле. '
    body += 'Средние рассчитаны по отдельным измерениям, без предварительного усреднения узлов. Поэтому узлы с более частыми измерениями имеют больший вес. '
    body += 'Интервалы — `[начало, конец)`, метка слева; `resample("h")` означает последовательные часы всего периода, `resample("D")` — календарные сутки UTC. '
    body += 'Пустые интервалы сохранены как NaN, без заполнения и интерполяции; линия на таком интервале прерывается. '
    body += 'Недопустимые числа исключаются только из среднего соответствующей метрики; число пригодных значений сохранено в агрегатах.\n\n'
    body += table([{'Показатель': 'Часовых интервалов', 'Значение': len(hourly)},
                   {'Показатель': 'Часов без измерений load_pct', 'Значение': int(hourly.avg_load.isna().sum())},
                   {'Показатель': 'Календарных дней', 'Значение': len(daily)},
                   {'Показатель': 'Дней без измерений error_rate', 'Значение': int(daily.avg_error_rate.isna().sum())}])
    body += '\n## Графики\n\n'
    for filename, text in [
        ('load_by_hour.png', 'Средняя загрузка по последовательным часам показывает локальные пики, провалы и резкие переходы.'),
        ('load_by_day.png', 'Средняя загрузка по суткам сглаживает внутрисуточные колебания и показывает междневные изменения.'),
        ('error_rate_by_day.png', 'Средняя доля ошибок по суткам показывает ухудшения качества. Значение 0,01 соответствует 1% ошибок.')]:
        body += f'### {text}\n\n'
        body += f'![{text}]({link(filename)})\n\n' if available else f'`{filename}` будет создан после загрузки измерений.\n\n'
    if available:
        extremes = []
        for label, series in [('Часовая загрузка, %', hourly.avg_load), ('Суточная загрузка, %', daily.avg_load), ('Суточная доля ошибок', daily.avg_error_rate)]:
            s = series.dropna()
            if not s.empty:
                extremes.append({'Ряд': label, 'Минимум': s.min(), 'Время минимума UTC': s.idxmin(),
                                 'Максимум': s.max(), 'Время максимума UTC': s.idxmax()})
        body += table(extremes)
    body += '\n## Объяснимое обнаружение аномалий\n\n'
    body += 'Критерий один для всех рядов: границы Тьюки `Q1−1,5·IQR` и `Q3+1,5·IQR`. '
    body += 'Для часовой средней нагрузки проверяются обе границы; для модуля её изменения — верхняя. '
    body += 'Разность вычисляется между соседними хронологическими часами; разрыв с NaN не соединяется. '
    body += 'Для суточной доли ошибок проверяются высокий уровень и положительное изменение `max(e[d]−e[d−1],0)`. '
    body += 'При IQR=0 граница равна квартилю: это свойство критерия требует осторожной интерпретации. '
    body += 'При отсутствии пригодных наблюдений правило не выдаёт сигналов. Это ретроспективный критерий всего периода; для онлайн-мониторинга пороги нужно оценивать только на прошлых данных.\n\n' + table(rules)
    body += f'\n### Обнаруженные сигналы: {len(anomalies)}\n\n'
    body += table(anomalies)
    body += '\nЗначение в строке изменения — модуль изменения в п.п. либо положительный прирост доли ошибок; значение в строке уровня — само среднее. '
    body += 'Отсутствие сигналов по отдельному правилу означает, что порог не пересечён; обычный максимум не объявляется аномалией.\n'
    body += '\n## Интерпретация и ограничения\n\n'
    if available and daily.avg_error_rate.notna().any():
        errors = daily.avg_error_rate.dropna()
        peak, baseline = errors.max(), errors.median()
        body += (f'Наиболее заметный суточный всплеск ошибок приходится на **{errors.idxmax().date()}**: '
                 f'средняя доля ошибок **{peak:.8f}**, то есть **{100 * peak:.4f}%**, '
                 f'при медиане суточного ряда **{baseline:.8f}** ({100 * baseline:.4f}%). ')
        if baseline > 0:
            body += f'Отношение пика к медиане — **{peak / baseline:.2f} раза**. '
        body += ('Это наблюдаемый результат; причину всплеска следует проверять отдельно. '
                 'Небольшое превышение порога прироста в другие дни не равнозначно масштабному сбою.\n\n')
    if anomalies:
        counts = pd.Series([row['Тип'] for row in anomalies]).value_counts().sort_index()
        body += table([{'Тип сигнала': name, 'Количество': int(count)} for name, count in counts.items()])
    body += 'Высокая загрузка может быть связана с ростом спроса или недостатком пропускной способности; снижение — с падением активности, отключением узла или изменением состава измерений. '
    body += 'Совместный рост ошибок и задержки требует проверки журналов оборудования. Эти причины — гипотезы, не доказанные данным агрегированием. '
    body += 'Среднее по сети скрывает отдельные перегруженные узлы; отсутствие телеметрии не означает отсутствие неисправностей. '
    body += 'Глобальные IQR-пороги не учитывают сезонность, множественные проверки и смену числа наблюдаемых узлов. '
    body += 'Неизвестна фактическая зона наивных дат; поздняя доставка не имеет отдельного ingest timestamp. Дубликаты не удалены.\n'
    body += f'\n## Вывод\n\nПолучено {len(hourly)} часовых и {len(daily)} суточных интервалов; правилом выявлено {len(anomalies)} сигналов. '
    body += 'Проверять следует указанные интервалы и узлы в них, сохраняя допустимые экстремумы для диагностики. '
    body += f'Полные воспроизводимые таблицы: [часы]({link("telemetry_hourly.csv")}), [сутки]({link("telemetry_daily.csv")}), [аномалии]({link("telemetry_anomalies.csv")}).\n' if available else 'Итоговые значения требуют исполнения на исходных данных.\n'
    (report_dir / REPORT_NAMES[1]).write_text(body, encoding='utf-8')

    body = '# Задание 3. Интеграция телеметрии с CDR\n' + common
    body += '\n## Источники и ключ\n\n'
    body += 'В проекте существуют `dataset/raw/cdr`, `dataset/late_delivery/cdr`, модуль `data_preparation_hw.py` и очищенный `processed/cdr_clean.parquet`. '
    body += 'Схема CDR содержит `record_id`, `event_ts`, `ingest_ts`, `subscriber_key`, `node_id` и показатели услуг. '
    body += 'Существующая функция `integrate_frames` связывает телеметрию предыдущих суток; новый пример сохраняет эту договорённость. '
    body += 'По умолчанию используется готовый очищенный CDR. Если его нет, доступны все исходные CSV raw/late_delivery с проверкой физической структуры. '
    body += 'Статистика очищенного CDR не выдаётся за аудит исходных CSV; правила предыдущей очистки изложены в `cleaning_report.md`.\n\n'
    body += f'В обоих источниках `node_id` приводится к строке, обрезаются пробелы (включая NBSP), регистр становится верхним; маркеры NULL превращаются в пропуск. '
    body += f'Принят шаблон существующего проекта `{audit["node_pattern"]}`; его можно изменить через `--node-pattern`. '
    body += 'Ведущие нули не удаляются. Преобразование вроде `1 → ND000001` без таблицы соответствий запрещено. '
    body += 'Из правой таблицы удаляются пустые ключи, поскольку pandas может соединять NULL с NULL, в отличие от SQL.\n'
    body += '\n## Безопасная стратегия\n\n'
    body += 'Прямое соединение всех записей только по node_id даёт для узла `n_CDR × n_telemetry` строк и искажает суммы. Используется следующий порядок:\n\n'
    body += '1. Выбрать аналитическое окно. В выполненной проверке использован весь доступный CDR; измерения ограничены загруженной телеметрией. Не покрытые наблюдения остаются в LEFT JOIN.\n'
    body += '2. Агрегировать телеметрию до одной строки на `(node_id, measurement_day)` в UTC.\n'
    body += '3. Связать CDR дня `d` с телеметрией дня `d−1`: `event_day = measurement_day + 1 день`. Все измерения правой строки предшествуют началу дня CDR.\n'
    body += '4. Выполнить LEFT JOIN по `(node_id, event_day)` с `validate="many_to_one"`. Правая сторона обязана быть уникальной; дубликат ключа вызывает ошибку.\n'
    body += '5. Сравнить количество строк до/после, долю совпадений, пустые показатели и временное покрытие. Совпадение определять по индикатору соединения, а не по ненулевому значению нагрузки.\n\n'
    body += 'Пример исполняется функциями `telemetry_daily`, `left_join_cdr`, `validate_integration` в `telemetry_hw.py`:\n\n'
    body += '```python\nfrom telemetry_hw import telemetry_daily, left_join_cdr\n'
    body += '# telemetry — нормализованный DataFrame с timestamp в UTC\n'
    body += 'daily = telemetry_daily(telemetry)\n'
    body += 'joined = left_join_cdr(cdr, daily, source_tz="UTC")\n'
    body += 'assert len(joined) == len(cdr)\n'
    body += 'coverage = joined["_merge"].eq("both").mean()\n```\n'
    body += '\n## Проверка на доступных данных\n\n'
    if integration['available']:
        body += f'Источник: `{integration["source"]}`. Обработан весь источник пакетами до 100 000 строк; в память не загружается вся CDR.\n\n'
        labels = {'rows_before': 'Строк CDR до', 'rows_after': 'Строк после LEFT JOIN', 'matched': 'Совпавших CDR', 'unmatched': 'CDR без совпадения',
                  'match_pct': 'Покрытие, %', 'missing_node': 'Пустых node_id', 'invalid_node': 'Некорректных непустых node_id',
                  'invalid_time': 'Непригодных дат CDR', 'unknown_node_rows': 'CDR с узлом вне наблюдённой телеметрии',
                  'outside_date_range': 'CDR вне диапазона дат доступных правых агрегатов', 'matched_missing_load': 'Совпавших CDR с пустой нагрузкой',
                  'cdr_min': 'Первое событие CDR, UTC', 'cdr_max': 'Последнее событие CDR, UTC', 'right_rows': 'Суточных агрегатов telemetry',
                  'right_duplicate_keys': 'Повторов правого составного ключа', 'duplicate_id_excess': 'Повторов record_id сверх первого',
                  'raw_csv_rejected': 'Физически непригодных строк при чтении raw CSV'}
        body += table([{'Показатель': label, 'Значение': integration.get(k)} for k, label in labels.items()])
        body += 'Причины несовпадения в таблице могут пересекаться; их нельзя складывать. '
        body += f'Правые агрегаты: [Parquet]({link("telemetry_previous_day.parquet")}); первые 20 строк результата: [пример]({link("cdr_telemetry_sample.parquet")}); '
        body += f'аудит входа CDR: [JSON]({link("cdr_input_audit.json")}). Полная широкая объединённая таблица повторно не материализуется; проверяется каждая строка.\n'
        body += f'Строки сохранены: {integration["rows_before"]} → {integration["rows_after"]}. Покрытие {integration["match_pct"]:.4f}%; '
        body += f'{integration["unmatched"]} событий требуют обработки пропущенной сетевой информации.\n'
    else:
        body += integration.get('reason', 'Интеграция не выполнялась без данных.') + ' Успешное соединение не заявляется; код выше готов для запуска после подключения источников.\n'
    body += '\n## Проблемы, последствия и решения\n\n' + table([
        {'Проблема': 'Пустой node_id', 'Последствие': 'Ложные совпадения NULL или потеря CDR', 'Решение': 'Сохранять CDR слева, исключить NULL справа, добавить флаг отсутствия связи'},
        {'Проблема': 'Повторы telemetry', 'Последствие': 'Смещение средних и счётчиков отказов', 'Решение': 'Аудит полного/составного ключа, дедупликация только по договорённости; сейчас повторы сохранены'},
        {'Проблема': 'Повторный/конфликтный record_id CDR', 'Последствие': 'Повторный учёт услуг и неверные признаки', 'Решение': 'Сверить ingest_ts и бизнес-правила выбора версии, сохранить конфликтные варианты; использовать проверенный clean слой'},
        {'Проблема': 'Соединение многие-ко-многим', 'Последствие': 'Размножение строк и сумм', 'Решение': 'Уникальный узел×день справа, validate=many_to_one, обязательное равенство числа строк'},
        {'Проблема': 'Разный формат идентификаторов', 'Последствие': 'Не найденные либо ложные узлы', 'Решение': 'Единые string/trim/upper, шаблон и утверждённый справочник алиасов'},
        {'Проблема': 'Разные даты/часовые пояса', 'Последствие': 'Сдвиг на соседний час или день', 'Решение': 'Явно задать source-timezone, разобрать ISO и DD.MM.YYYY, привести к UTC до группировки'},
        {'Проблема': 'Разные периоды', 'Последствие': 'Падение покрытия и смещение выборки', 'Решение': 'Сравнить min/max дат, согласовать окно и прогрев предыдущего дня, сохранять события вне покрытия'},
        {'Проблема': 'Узел вне telemetry', 'Последствие': 'Нет сетевого контекста для части услуг', 'Решение': 'LEFT JOIN, аудит справочника/охвата; неизвестный узел не заменять произвольным'},
        {'Проблема': 'Пропуски после JOIN', 'Последствие': 'Нет информации либо все значения метрики непригодны', 'Решение': 'Разделять отсутствие ключа и пустую метрику; импутацию оценивать только на train и сохранять флаги'},
        {'Проблема': 'Утечка будущего', 'Последствие': 'Завышенная оценка прогноза оттока', 'Решение': 'Только предыдущие закрытые интервалы; отсечение по event и ingest, временное разбиение train/test'}])
    body += '\n## Ограничения и вывод\n\n'
    body += 'Агрегат того же полного дня содержит будущие для утреннего CDR измерения, поэтому для прогнозных признаков он не используется. '
    body += 'Предыдущие сутки устраняют утечку по времени события, но не доказывают известность к моменту прогноза: в telemetry нет ingest_ts, '
    body += 'а late_delivery содержит опоздавшие измерения. Для production нужен журнал поступления и фильтр `ingest_ts < cutoff`. '
    body += 'При отсутствии дат можно построить только агрегат по узлу: это описательный сетевой контекст всего окна, который нельзя безопасно использовать для прошлых прогнозов. '
    body += 'Наличие нормализованного node_id позволяет контролируемое обогащение CDR, но не устанавливает причинную связь сетевых показателей с оттоком.\n'
    (report_dir / REPORT_NAMES[2]).write_text(body, encoding='utf-8')

    body = '# Задание 4. Конструирование признаков телеметрии\n' + common
    body += '\n## Объект и окно агрегации\n\nОдна строка соответствует одному корректному нормализованному `node_id`. '
    body += f'Число рассчитанных признаков: **5**; число строк набора: **{len(features)}**. '
    body += (f'Использованы только измерения `timestamp < {cutoff.isoformat()}`; строки с неизвестной датой исключены из признаков.\n' if cutoff is not None else
             'В этом запуске cutoff не задан: признаки описывают весь доступный период и не предназначены для прогнозов на даты внутри него. При отсутствии cutoff запись с неизвестным временем участвует в узловых показателях.\n')
    body += policy
    body += '\n## Определения\n\nПусть `Vₘ(n)` — пригодные непустые измерения метрики m узла n в выбранном окне, `R(n)` — все записи узла. '
    body += 'Среднее делится на число непустых значений соответствующей метрики. Если их нет, результат — NaN. '
    body += '`failure_count` считает записи с нормализованным уровнем строго `error`; WARN, INFO, пропуск и неизвестная категория не считаются ошибкой. '
    body += 'Нулевой счётчик означает отсутствие зарегистрированных ERROR, а не гарантированное отсутствие отказов.\n\n'
    body += table([
        {'Признак': 'avg_load', 'Тип': 'float64', 'Формула': 'Σ(load_pctᵢ), i∈V_load(n) / |V_load(n)|', 'Python': 'groupby("node_id")["load_pct"].mean()', 'Смысл': 'Средняя загрузка узла, %', 'Связь с качеством/оттоком': 'Хронически высокая загрузка может ухудшать доступность и пользовательский опыт'},
        {'Признак': 'max_load', 'Тип': 'float64', 'Формула': 'max(load_pctᵢ), i∈V_load(n)', 'Python': 'groupby("node_id")["load_pct"].max()', 'Смысл': 'Пиковая загрузка, %', 'Связь с качеством/оттоком': 'Выявляет периоды насыщения, скрытые средним значением'},
        {'Признак': 'avg_error_rate', 'Тип': 'float64', 'Формула': 'Σ(error_rateᵢ), i∈V_error(n) / |V_error(n)|', 'Python': 'groupby("node_id")["error_rate"].mean()', 'Смысл': 'Средняя доля ошибок, 0…1', 'Связь с качеством/оттоком': 'Повышенный уровень отражает нестабильность обслуживания'},
        {'Признак': 'avg_latency', 'Тип': 'float64', 'Формула': 'Σ(latency_msᵢ), i∈V_latency(n) / |V_latency(n)|', 'Python': 'groupby("node_id")["latency_ms"].mean()', 'Смысл': 'Средняя задержка, мс', 'Связь с качеством/оттоком': 'Характеризует отклик сервиса и возможное недовольство абонентов'},
        {'Признак': 'failure_count', 'Тип': 'int64', 'Формула': 'Σ 𝟙(lower(trim(levelᵢ)) = error), i∈R(n)', 'Python': 'assign(is_error=level.eq("error").fillna(False)).groupby("node_id")["is_error"].sum()', 'Смысл': 'Число зарегистрированных ERROR-записей', 'Связь с качеством/оттоком': 'Отражает частоту сообщений о сбоях; зависит от частоты измерений'}])
    body += '\n## Реализация\n\n```python\n'
    body += 'selected = telemetry[telemetry["node_id"].notna()].copy()\n'
    body += 'if cutoff is not None:\n    selected = selected[selected["timestamp"].notna() & (selected["timestamp"] < cutoff)].copy()\n'
    body += 'selected["is_error"] = selected["level"].eq("error").fillna(False).astype("int64")\n'
    body += 'features = selected.groupby("node_id").agg(\n'
    body += '    avg_load=("load_pct", "mean"), max_load=("load_pct", "max"),\n'
    body += '    avg_error_rate=("error_rate", "mean"), avg_latency=("latency_ms", "mean"),\n'
    body += '    failure_count=("is_error", "sum"),\n).reset_index()\n'
    body += 'assert features["node_id"].is_unique\n```\n'
    body += '\nКод вызывается функцией `node_features` из `telemetry_hw.py`. Дополнительные показатели не добавлены в обязательный набор. '
    body += f'В исходных данных {int(df.level.isna().sum())} записей без уровня, нормализованных ERROR — {int(df.level.eq("error").sum())}.\n'
    if available:
        body += f'\n## Полученный набор\n\n[telemetry_node_features.parquet]({link("telemetry_node_features.parquet")}) '
        body += f'содержит ровно {len(features)} строк и 6 столбцов: ключ и пять признаков. Ключ уникален и не содержит NULL.\n\n'
        body += 'Первые 10 строк в лексикографическом порядке узла:\n\n' + table(features.head(10))
        body += '\nПропуски в рассчитанных признаках:\n\n' + table([{'Признак': c, 'Тип': str(features[c].dtype), 'Пропусков': int(features[c].isna().sum())} for c in FEATURES])
        body += f'\nСумма failure_count по набору: **{int(features.failure_count.sum())}**. '
        body += 'Она может быть меньше числа ERROR в исходном массиве из-за отсутствующего узла или ограничения cutoff.\n'
    else:
        body += '\nParquet не создан: исходные измерения отсутствуют. После подключения данных команда из README создаст требуемый набор.\n'
    body += '\n## Предотвращение утечки и бизнес-вывод\n\n'
    body += 'Для прогноза на момент T задать `--feature-cutoff T` и использовать только `timestamp < T`; cutoff интерпретируется в исходной зоне, если смещение не указано. '
    body += 'Для обучающей выборки на нескольких датах признаки пересчитываются отдельно для каждой даты; общий агрегат периода нельзя присоединять к историческим целям. '
    body += 'При наличии времени поступления дополнительно требуется `ingest_ts < T`; для late_delivery без такого поля доступность признака неизвестна. '
    body += 'Для сравнимости узлов следует согласовать длину окна и частоту регистрации. Связь с абонентом строится через CDR, затем качество прогноза проверяется на временном holdout. '
    body += 'Пять признаков описывают нагрузку, ошибки, задержки и зарегистрированные отказы, но сами по себе не доказывают причину оттока и не заменяют обучение модели.\n'
    (report_dir / REPORT_NAMES[3]).write_text(body, encoding='utf-8')

    body = '# Задание 5. Визуализация выбросов телеметрии\n' + common + policy
    body += '\n## Методика и результаты\n\n'
    body += 'Коробка ограничена Q1 и Q3; линия внутри — медиана. `IQR=Q3−Q1`, `L=Q1−1,5·IQR`, `U=Q3+1,5·IQR`. '
    body += 'Усы доходят до крайних наблюдений внутри границ; точки строго за границами считаются статистическими выбросами. '
    body += 'Знаменатель доли выбросов — N пригодных значений данной метрики. Отрицательная расчётная нижняя граница не означает допустимость отрицательного измерения. '
    body += 'В таблице минимумы и максимумы относятся к пригодным значениям; недопустимые измерения отдельно подсчитаны ниже. '
    body += 'Эта таблица строится из того же объекта статистики, что и таблица задания 1.\n\n' + outlier_table(outliers)
    body += '\n## Физически и технически недопустимые значения\n\n' + table(validation_rows)
    for col in METRICS:
        body += f'\n## {TITLES[col]}\n\n'
        body += f'![{TITLES[col]}: коробчатая диаграмма]({link(col + "_boxplot.png")})\n\n' if available else f'График `{col}_boxplot.png` ожидает исходных данных.\n\n'
        s = outliers[col]
        body += f'Пригодных значений: {s["n"]}; IQR-выбросов: {s["outliers"]} ({s["outlier_pct"]:.4f}%). '
        if col == 'load_pct':
            body += 'Рекомендация для допустимых экстремумов: **сохранить (Keep)**. Загрузка около 100% возможна и полезна для поиска нехватки ёмкости. Значения вне 0…100 исключать только из аналитической метрики с сохранением исходной строки. '
        elif col == 'latency_ms':
            body += 'Рекомендация для допустимых экстремумов: **сохранить (Keep)**. Большая положительная задержка может быть реальным ухудшением сервиса. Отрицательные и бесконечные значения исключать из аналитической метрики. '
        else:
            body += 'Рекомендация для допустимых экстремумов: **сохранить (Keep)**. Доля ошибок до 1 включительно может указывать на настоящий отказ. Значения вне 0…1 исключать из аналитической метрики; долю не путать с процентом. '
        body += 'Для чувствительной к хвостам модели допустима отдельная версия с winsorization (Cap), но пороги оцениваются только на train и сравниваются на временной валидации. Автоматическое удаление (Remove) только по IQR не рекомендуется.\n'
    body += '\n## Что действительно преобразовано\n\n'
    body += 'Выполнены разбор типов, нормализация строк и замена непригодных значений на NaN в аналитической копии; исходные файлы не изменены. '
    body += 'Удаление статистических выбросов и winsorization не выполнялись. Полные дубликаты также оставлены. '
    body += 'Рекомендации по отдельной модельной версии являются предложением, а не заявлением о сделанном преобразовании.\n'
    body += '\n## Вывод\n\n' + ('; '.join(f'{c}: {s["outliers"]} допустимых экстремумов по IQR' for c, s in outliers.items()) + '. ' if available else '')
    body += 'Выбросы нужно сопоставлять со временем, узлом и эксплуатационными журналами. Сохранение допустимых хвостов поддерживает диагностику качества и последующее построение признаков оттока.\n'
    (report_dir / REPORT_NAMES[4]).write_text(body, encoding='utf-8')
    append_instructor_feedback(raw, df, daily, output, report_dir)


def append_instructor_feedback(raw, df, daily, output, report_dir):
    """Дополняет три отчёта, не меняя прежние числа и графики.

    Функцию можно вызвать отдельно на тех же исходных данных; повторный вызов
    заменяет только собственное дополнение. Основной генератор вызывает её всегда.
    """
    output, report_dir = Path(output), Path(report_dir)
    marker = '\n## Дополнение по замечаниям преподавателя\n'
    def append(name, text):
        path = report_dir / name
        if path.exists():
            before = path.read_text(encoding='utf-8').split(marker)[0]
            path.write_text(before + marker + '\n' + text, encoding='utf-8')

    rows = []
    numeric = {'load_pct', 'active_sessions', 'error_rate', 'latency_ms', 'temperature_c'}
    for col in raw:
        source = raw[col]
        r = {'Поле': col, 'Непустых N': int(source.notna().sum()), 'Различных': int(source.nunique()),
             'Минимум / начало': '—', 'Максимум / конец': '—', 'Среднее': '—', 'Медиана': '—',
             'Наиболее частое значение': '—', 'Частота моды': '—'}
        if col in numeric:
            values = pd.to_numeric(source, errors='coerce')
            values = values.loc[np.isfinite(values)]
            if len(values):
                r.update({'Минимум / начало': values.min(), 'Максимум / конец': values.max(),
                          'Среднее': values.mean(), 'Медиана': values.median()})
        elif col in ['ts', 'timestamp', 'event_ts'] and 'timestamp' in df:
            r.update({'Минимум / начало': str(df.timestamp.min()), 'Максимум / конец': str(df.timestamp.max())})
        elif source.notna().any():
            frequencies = source.value_counts()
            r.update({'Наиболее частое значение': str(frequencies.index[0]), 'Частота моды': int(frequencies.iloc[0])})
        rows.append(r)
    append(REPORT_NAMES[0], '### Самодостаточная статистика всех исходных полей\n\n' +
           table(rows) + '\nЧисла в этом дополнении относятся к исходным полям до доменных ограничений; '
           'нечисловые и бесконечные значения не участвуют в числовых средних. Для времени использован '
           'тот же разбор UTC, что в основном профиле. Категориальным полям не приписываются средние. '
           'Ранее приведённая статистика пригодных аналитических значений не изменена.\n')
    append(REPORT_NAMES[4], table([
        {'Метрика': 'load_pct', 'Решение для допустимых IQR-экстремумов': 'Сохранить (Keep)', 'Причина': 'Нагрузка около 100% физически допустима и важна для диагностики'},
        {'Метрика': 'latency_ms', 'Решение для допустимых IQR-экстремумов': 'Сохранить (Keep)', 'Причина': 'Большая положительная задержка может характеризовать инцидент'},
        {'Метрика': 'error_rate', 'Решение для допустимых IQR-экстремумов': 'Сохранить (Keep)', 'Причина': 'Допустимые высокие доли ошибок нельзя удалять только по IQR'}]) +
        '\nОграничить (Cap) — только отдельный эксперимент модели с порогами train; '
        'удалить (Remove) — только физически неверное значение из аналитической метрики '
        'с сохранением исходной записи. В текущем расчёте Cap и удаление IQR-экстремумов не выполняются.\n')
    if df.empty or not daily.avg_error_rate.notna().any():
        return
    peak = daily.avg_error_rate.idxmax()
    x = df.loc[df.timestamp.notna() & df.node_id.notna() & df.error_rate.notna()].copy()
    day = x.timestamp.dt.floor('D')
    now = x.loc[day.eq(peak)].groupby('node_id').error_rate.agg(['mean', 'count'])
    before = x.loc[day.ge(peak - pd.Timedelta(days=7)) & day.lt(peak)].groupby('node_id').error_rate.agg(['mean', 'count'])
    matched = now.join(before, how='inner', lsuffix='_peak', rsuffix='_before')
    if matched.empty:
        return
    matched['difference'] = matched.mean_peak - matched.mean_before
    matched.reset_index().to_csv(output / 'peak_node_diagnostic.csv', index=False)
    summary = [
        {'Показатель': 'Дата пика UTC', 'Значение': str(peak.date())},
        {'Показатель': 'Узлов с измерениями и в пик, и в предыдущие 7 суток', 'Значение': len(matched)},
        {'Показатель': 'Среднее по этим узлам до пика (равные веса узлов)', 'Значение': matched.mean_before.mean()},
        {'Показатель': 'Среднее по этим узлам в пик (равные веса узлов)', 'Значение': matched.mean_peak.mean()},
        {'Показатель': 'Узлов с ростом, N', 'Значение': int(matched.difference.gt(0).sum())},
        {'Показатель': 'Узлов с ростом, %', 'Значение': 100 * matched.difference.gt(0).mean()},
        {'Показатель': 'Медиана изменения доли ошибок по узлам', 'Значение': matched.difference.median()},
        {'Показатель': 'Узлов с ростом более 0,05 (5 п.п.)', 'Значение': int(matched.difference.gt(.05).sum())}]
    top = matched.sort_values('difference', ascending=False).head(10).reset_index().rename(columns={
        'node_id': 'Узел', 'mean_peak': 'Средняя доля ошибок в пик', 'count_peak': 'N в пик',
        'mean_before': 'Средняя доля ошибок до пика', 'count_before': 'N до пика', 'difference': 'Разность'})
    append(REPORT_NAMES[1], '### Проверка гипотезы о составе узлов и массовом ухудшении\n\n'
        '**Гипотеза:** пик может объясняться изменением состава измеряемых узлов либо ростом ошибок '
        'внутри тех же узлов. Сопоставлены узлы, наблюдавшиеся в сутки пика и в предыдущие семь календарных суток. '
        'В каждом периоде сначала усредняются измерения узла, затем все совпавшие узлы получают равный вес.\n\n' +
        table(summary) + '\nДесять наибольших приростов:\n\n' + table(top) +
        '\n**Предварительное решение:** изменение состава узлов не является достаточным объяснением, '
        'если рост сохраняется при фиксированном составе и равных весах; проверить журналы инцидентов '
        'и согласованность показаний ERROR/задержки. Числа показывают внутригрупповую ассоциацию, '
        'но не причину. Многие узлы имеют мало измерений за сутки, поэтому экстремальные индивидуальные '
        'разности нестабильны. Отсутствующие узлы и неодинаковая частота измерений остаются ограничениями. '
        'Полная таблица: `processed/telemetry/peak_node_diagnostic.csv`. Старые значения суточного ряда не изменены.\n')


def save_validation_samples(raw, df, audit, output):
    result = {}
    for col in METRICS:
        if col in raw:
            bad = clean_text(raw[col]).notna() & df[col].isna()
            result[col] = [{'row_index': int(i), 'raw_value': fmt(raw.at[i, col])} for i in raw.index[bad][:20]]
    if 'node_id' in raw:
        bad = clean_text(raw.node_id).notna() & df.node_id.isna()
        result['node_id'] = [{'row_index': int(i), 'raw_value': fmt(raw.at[i, 'node_id'])} for i in raw.index[bad][:20]]
    ts = audit['timestamp_column']
    if ts:
        bad = clean_text(raw[ts]).notna() & df.timestamp.isna()
        result['timestamp'] = [{'row_index': int(i), 'raw_value': fmt(raw.at[i, ts])} for i in raw.index[bad][:20]]
    dump(output / 'validation_samples.json', result)


def verify_artifacts(output, report_dir, features, expect_data):
    from PIL import Image
    import pyarrow.parquet as pq
    result = {'reports': [], 'charts': [], 'feature_rows': 0, 'broken_links': []}
    for name in REPORT_NAMES:
        path = report_dir / name
        text = path.read_text(encoding='utf-8')
        assert len(text) > 1000, name
        result['reports'].append(name)
        for target in re.findall(r'\]\(([^)]+)\)', text):
            if '://' not in target and not (report_dir / target).exists():
                result['broken_links'].append(target)
    assert not result['broken_links'], result['broken_links']
    if expect_data:
        for name in CHART_NAMES:
            path = output / name
            with Image.open(path) as img:
                img.load()
                assert min(img.size) >= 700 and path.stat().st_size > 5000, name
                pixels = np.asarray(img.convert('RGB'))
                assert pixels.std() > 2, name
                result['charts'].append({'name': name, 'pixels': list(img.size), 'bytes': path.stat().st_size})
        stored = pd.read_parquet(output / 'telemetry_node_features.parquet')
        pd.testing.assert_frame_equal(stored, features)
        assert stored.node_id.is_unique and stored.node_id.notna().all()
        assert pq.read_schema(output / 'telemetry_node_features.parquet').names == ['node_id'] + FEATURES
        result['feature_rows'] = len(stored)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description='Пять заданий по анализу телеметрии, без генерации данных')
    parser.add_argument('--dataset', type=Path, help='Корень dataset/out, каталог telemetry, JSONL или glob')
    parser.add_argument('--output', type=Path, help='Каталог результатов; по умолчанию processed/telemetry')
    parser.add_argument('--reports', type=Path, help='Каталог пяти Markdown; по умолчанию рядом со скриптом')
    parser.add_argument('--cdr', type=Path, help='CDR Parquet/CSV или корень raw/late_delivery; по умолчанию clean слой проекта')
    parser.add_argument('--skip-cdr', action='store_true', help='Не выполнять проверку CDR')
    parser.add_argument('--timestamp-column', help='Имя поля времени, если оно отличается от известных вариантов')
    parser.add_argument('--source-timezone', default='UTC', help='Исходная зона наивных дат, по умолчанию UTC')
    parser.add_argument('--node-pattern', default=r'ND[0-9]{6}', help='Regex корректного узла; пустая строка разрешает любой непустой ключ')
    parser.add_argument('--feature-cutoff', help='В признаки включать только события строго раньше этой даты')
    args = parser.parse_args(argv)
    base = Path(__file__).resolve().parent
    dataset = args.dataset or default_dataset(base)
    output = (args.output or base / 'processed/telemetry').resolve()
    report_dir = (args.reports or base).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    files = discover(dataset)
    log(f'Найдено файлов телеметрии: {len(files)}; вход: {dataset}')
    raw, inventory = load_telemetry(files, output)
    dump(output / 'input_manifest.json', inventory)
    log(f'Загружено записей: {len(raw):,}; непрочитанных строк: {inventory["rejected_lines"]}')
    df, audit = normalize(raw, args.timestamp_column, args.source_timezone, args.node_pattern)
    cutoff = None
    if args.feature_cutoff:
        cutoff = parse_times(pd.Series([args.feature_cutoff]), args.source_timezone).iloc[0]
        if pd.isna(cutoff):
            raise ValueError('Не удалось разобрать --feature-cutoff')
    duplicate = duplicate_summary(raw, df)
    outliers = {col: iqr_stats(df[col]) for col in METRICS}
    save_validation_samples(raw, df, audit, output)
    hourly, daily = build_timeseries(df)
    anomalies, rules = detect_anomalies(hourly, daily)
    features = node_features(df, cutoff)
    integration = {'available': False, 'reason': 'CDR не проверялся: нет телеметрии либо задан --skip-cdr.'}
    if len(raw):
        hourly.to_csv(output / 'telemetry_hourly.csv', index_label='timestamp_utc')
        daily.to_csv(output / 'telemetry_daily.csv', index_label='timestamp_utc')
        pd.DataFrame(anomalies, columns=['Время UTC', 'Тип', 'Значение', 'Граница']).to_csv(output / 'telemetry_anomalies.csv', index=False)
        features.to_parquet(output / 'telemetry_node_features.parquet', index=False)
        log('Создание шести графиков и Parquet с пятью признаками...')
        generate_charts(df, hourly, daily, output)
        if not args.skip_cdr:
            cdr_source = args.cdr
            if cdr_source is None:
                # A custom dataset must not silently join unrelated local CDR.
                project_dataset = Path(dataset).resolve() == (base / 'dataset').resolve()
                clean = base / 'processed/cdr_clean.parquet'
                cdr_source = clean if project_dataset and clean.exists() else dataset
            log(f'Пакетная проверка CDR: {cdr_source}')
            integration = validate_integration(df, cdr_source, output, args.source_timezone, args.node_pattern)
    write_reports(raw, df, inventory, audit, duplicate, outliers, hourly, daily, anomalies, rules,
                  features, integration, output, report_dir, cutoff)
    result = {'status': 'completed' if len(raw) else 'awaiting_data', 'records': len(raw), 'raw_columns': list(raw),
              'period_utc': {'min': str(df.timestamp.min()), 'max': str(df.timestamp.max())},
              'unique_valid_nodes': int(df.node_id.nunique()), 'normalization': audit, 'duplicates': duplicate,
              'missing': missing_summary(raw), 'outliers': outliers, 'anomaly_count': len(anomalies), 'anomaly_rules': rules,
              'feature_count': 5, 'feature_rows': len(features), 'feature_cutoff': str(cutoff) if cutoff is not None else None,
              'integration': integration, 'environment': {'python': platform.python_version(), 'execution': 'local' if 'google.colab' not in sys.modules else 'Colab',
                  'packages': {name: importlib.metadata.version(name) for name in ['pandas', 'numpy', 'matplotlib', 'seaborn', 'pyarrow']}}}
    result['validation'] = verify_artifacts(output, report_dir, features, bool(len(raw)))
    dump(output / 'telemetry_metrics.json', result)
    log(f'Готово: {len(raw):,} записей, {len(features):,} узлов, {len(anomalies)} сигналов, пять отчётов.')
    return result


if __name__ == '__main__':
    main()
