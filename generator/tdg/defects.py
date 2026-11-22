# -*- coding: utf-8 -*-
"""Внедрение дефектов качества и ведомость.

Ведомость хранится как правило воспроизведения (тип, целевая и фактическая доля,
начальное состояние), а не как поимённый перечень записей: при 300 млн записей
перечень затронутых идентификаторов сам по себе занял бы сотни мегабайт.
Восстановить перечень позволяет verify.py по тем же правилам.
"""
import numpy as np
import pandas as pd
from . import config as C


class Ledger:
    """Накопитель фактических долей дефектов по всему периоду."""

    def __init__(self):
        self.counts = {k: 0 for k in C.DEFECTS}
        self.totals = {"F1": 0, "F2": 0}

    def add(self, name, k):
        self.counts[name] += int(k)

    def add_total(self, obj, k):
        self.totals[obj] += int(k)

    def to_frame(self):
        rows = []
        for name, spec in C.DEFECTS.items():
            base = sum(self.totals[o] for o in spec["obj"].split(","))
            rows.append(dict(
                defect=name,
                description=spec["desc"],
                object=spec["obj"],
                target_rate=spec["rate"],
                actual_count=self.counts[name],
                base_rows=base,
                actual_rate=round(self.counts[name] / base, 6) if base else 0.0,
            ))
        return pd.DataFrame(rows)


def _mask(rng, n, rate):
    return rng.random(n) < rate


def inject_cdr(df, rng, ledger, tariff_ids):
    """Дефекты в записях о соединениях. Порядок важен: дубликаты добавляются
    последними, чтобы попасть в выгрузку вместе с уже искажёнными полями."""
    n = len(df)
    if n == 0:
        return df
    ledger.add_total("F1", n)

    # пропуски в необязательных полях
    m = _mask(rng, n, C.DEFECTS["null_optional"]["rate"])
    df.loc[m, "partner_code"] = None
    ledger.add("null_optional", m.sum())

    # пропуск в обязательном поле
    m = _mask(rng, n, C.DEFECTS["null_required"]["rate"])
    df.loc[m, "subscriber_key"] = None
    ledger.add("null_required", m.sum())

    # тарифный план отсутствует в справочнике
    m = _mask(rng, n, C.DEFECTS["orphan_tariff"]["rate"])
    if m.any():
        df.loc[m, "tariff_id"] = np.array(
            [f"T{x:02d}" for x in rng.integers(90, 99, m.sum())], dtype=object)
    ledger.add("orphan_tariff", m.sum())

    # отрицательная длительность
    m = _mask(rng, n, C.DEFECTS["negative_duration"]["rate"])
    df.loc[m, "duration_sec"] = -df.loc[m, "duration_sec"].abs() - 1
    ledger.add("negative_duration", m.sum())

    # объём трафика вне допустимого диапазона
    m = _mask(rng, n, C.DEFECTS["bytes_out_of_range"]["rate"])
    df.loc[m, "bytes_down"] = -1
    ledger.add("bytes_out_of_range", m.sum())

    # экстремальные выбросы
    m = _mask(rng, n, C.DEFECTS["extreme_outlier"]["rate"])
    if m.any():
        df.loc[m, "bytes_down"] = (rng.integers(5 * 10 ** 11, 9 * 10 ** 12, m.sum())).astype(np.int64)
    ledger.add("extreme_outlier", m.sum())

    # несогласованные написания справочных значений
    m = _mask(rng, n, C.DEFECTS["inconsistent_enum"]["rate"])
    if m.any():
        alt = {"voice": "VOICE", "sms": "Sms", "data": "DATA "}
        df.loc[m, "service_type"] = df.loc[m, "service_type"].map(lambda v: alt.get(v, v))
    ledger.add("inconsistent_enum", m.sum())

    # смешанные форматы даты: часть записей — в ином представлении
    m = _mask(rng, n, C.DEFECTS["mixed_date_format"]["rate"])
    df["event_ts"] = df["event_ts"].astype(str)
    if m.any():
        alt = pd.to_datetime(df.loc[m, "event_ts"]).dt.strftime("%d.%m.%Y %H:%M:%S")
        df.loc[m, "event_ts"] = alt
    ledger.add("mixed_date_format", m.sum())

    # полные дубликаты строк
    m = _mask(rng, n, C.DEFECTS["dup_full"]["rate"])
    k = int(m.sum())
    if k:
        df = pd.concat([df, df[m]], ignore_index=True)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    ledger.add("dup_full", k)
    return df


def inject_tickets(df, rng, ledger):
    n = len(df)
    if n == 0:
        return df
    ledger.add_total("F2", n)

    m = _mask(rng, n, C.DEFECTS["null_optional"]["rate"])
    df.loc[m, "resolution_code"] = None
    ledger.add("null_optional", m.sum())

    # закрытие раньше регистрации
    m = _mask(rng, n, C.DEFECTS["closed_before_open"]["rate"])
    if m.any():
        df.loc[m, "closed_ts"] = df.loc[m, "created_ts"] - pd.to_timedelta(
            rng.integers(60, 7200, m.sum()), unit="s")
    ledger.add("closed_before_open", m.sum())

    m = _mask(rng, n, C.DEFECTS["inconsistent_enum"]["rate"])
    if m.any():
        alt = {"phone": "PHONE", "cabinet": "Cabinet", "chat": "CHAT ", "office": "Office"}
        df.loc[m, "channel"] = df.loc[m, "channel"].map(lambda v: alt.get(v, v))
    ledger.add("inconsistent_enum", m.sum())

    # дубли по бизнес-ключу с расхождением атрибутов
    m = _mask(rng, n, C.DEFECTS["dup_business_key"]["rate"])
    k = int(m.sum())
    if k:
        dup = df[m].copy()
        dup["resolution_code"] = "RC9"
        dup["closed_ts"] = dup["closed_ts"] + pd.to_timedelta(
            rng.integers(60, 86400, k), unit="s")
        df = pd.concat([df, dup], ignore_index=True)
        df = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    ledger.add("dup_business_key", k)
    return df
