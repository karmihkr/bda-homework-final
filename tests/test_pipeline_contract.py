import json

import pandas as pd

import telemetry_failure_modeling as pipeline


def test_config_has_reproducible_temporal_setup():
    config = pipeline.load_config()
    assert config['history_window_days'] == 7
    assert config['prediction_horizon_days'] == 7
    assert config['split_days']['train']
    assert config['split_days']['validation']
    assert config['split_days']['test']


def test_snapshot_uses_history_before_cutoff_and_future_label_after_it():
    start = pd.Timestamp('2026-01-01', tz='UTC')
    rows = []
    for day in range(15):
        rows.append({
            'node_id': 'ND000001', 'timestamp': start + pd.Timedelta(int(day), unit='D'),
            'level': 'info', 'load_pct': 10 + day, 'latency_ms': 20 + day,
            'error_rate': day / 100,
        })
    rows.append({
        'node_id': 'ND000001', 'timestamp': start + pd.Timedelta(10, unit='D') + pd.Timedelta(1, unit='h'),
        'level': 'warn', 'load_pct': 99, 'latency_ms': 99, 'error_rate': .99,
    })
    rows.append({
        'node_id': 'ND000001', 'timestamp': start + pd.Timedelta(13, unit='D'),
        'level': 'error', 'load_pct': 100, 'latency_ms': 100, 'error_rate': 1,
    })
    result = pipeline.build_snapshot(pd.DataFrame(rows), start + pd.Timedelta(10, unit='D'), window=7, horizon=3)
    assert len(result) == 1
    assert result.loc[0, 'target'] == 1
    assert result.loc[0, 'max_load'] == 19
    assert result.loc[0, 'error_count'] == 0
    assert result.loc[0, 'future_events'] == 1


def test_fingerprint_changes_when_source_bytes_change(tmp_path):
    source = tmp_path / 'raw' / 'telemetry' / 'dt=2026-01-01' / 'events.jsonl'
    source.parent.mkdir(parents=True)
    source.write_text('{"node_id":"ND000001"}\n', encoding='utf-8')
    first, _ = pipeline.input_fingerprint(tmp_path, pipeline.CONFIG)
    source.write_text('{"node_id":"ND000002"}\n', encoding='utf-8')
    second, _ = pipeline.input_fingerprint(tmp_path, pipeline.CONFIG)
    assert first != second


def test_config_is_json():
    # Keep this small test independent of environment-specific package versions.
    assert json.loads(pipeline.CONFIG_PATH.read_text(encoding='utf-8'))['paths']['dataset'] == 'dataset'
