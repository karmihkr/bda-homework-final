# -*- coding: utf-8 -*-
"""Справочники (измерения) с историей изменений и веса перекоса.

Веса строятся конструктивно: целевой профиль перекоса выполняется по построению,
а не подбором показателя степенного закона.
"""
import numpy as np
import pandas as pd
from . import config as C


# ──────────────────────────────────────────────────────── веса перекоса
def _mixture_weights(n, rng, groups):
    """groups: [(доля объектов, доля веса), …]. Внутри группы — мягкий степенной хвост."""
    w = np.zeros(n, dtype=np.float64)
    start = 0
    for frac, share in groups:
        cnt = max(1, int(round(n * frac)))
        cnt = min(cnt, n - start)
        if cnt <= 0:
            break
        raw = 1.0 / np.power(np.arange(1, cnt + 1), 0.6)
        w[start:start + cnt] = raw / raw.sum() * share
        start += cnt
    if start < n:                       # остаток — равномерно, вес уже распределён
        w[start:] = w[start:].sum()
    total = w.sum()
    return w / total


def subscriber_weights(n_subs, rng):
    """Верхние 0,1 % → 35 %, следующие 9,9 % → 40 %, остальные → 25 %.
    Первые SKEW['mega_subs'] абонентов получают долю свыше 1 % каждый."""
    s = C.SKEW
    mega_n, mega_share = s["mega_subs"], s["mega_share"]
    mega_total = mega_n * mega_share

    top_n = max(mega_n + 1, int(round(n_subs * s["top_sub_frac"])))
    mid_n = int(round(n_subs * s["mid_sub_frac"]))
    tail_n = n_subs - top_n - mid_n
    if tail_n < 1:
        raise ValueError("слишком мало абонентов для заданного профиля перекоса")

    w = np.empty(n_subs, dtype=np.float64)
    w[:mega_n] = mega_share
    rest_top = top_n - mega_n
    raw = 1.0 / np.power(np.arange(1, rest_top + 1), 0.7)
    w[mega_n:top_n] = raw / raw.sum() * (s["top_sub_share"] - mega_total)
    raw = 1.0 / np.power(np.arange(1, mid_n + 1), 0.5)
    w[top_n:top_n + mid_n] = raw / raw.sum() * s["mid_sub_share"]
    tail_share = 1.0 - s["top_sub_share"] - s["mid_sub_share"]
    raw = rng.random(tail_n) + 0.15
    w[top_n + mid_n:] = raw / raw.sum() * tail_share

    perm = rng.permutation(n_subs)       # чтобы «горячие» не стояли подряд по ключу
    return w[np.argsort(perm)], perm


def region_weights(rng):
    """Два крупнейших региона дают около 30 % записей."""
    n = C.N_REGIONS
    w = np.empty(n)
    w[0] = w[1] = C.SKEW["top_region_share"]
    rest = 1.0 - 2 * C.SKEW["top_region_share"]
    raw = 1.0 / np.power(np.arange(1, n - 1), 0.8)
    w[2:] = raw / raw.sum() * rest
    return w


def node_weights(node_region, reg_w, rng):
    """Внутри каждого региона верхний 1 % узлов обслуживает 45 % его нагрузки.
    Доля региона в итоге равна reg_w по построению."""
    s = C.SKEW
    w = np.zeros(len(node_region), dtype=np.float64)
    for r in range(C.N_REGIONS):
        idx = np.flatnonzero(node_region == r)
        if idx.size == 0:
            continue
        k = max(1, int(round(idx.size * s["top_node_frac"])))
        order = rng.permutation(idx.size)
        hot, cold = idx[order[:k]], idx[order[k:]]
        w[hot] = s["top_node_share"] / k
        if cold.size:
            raw = rng.random(cold.size) + 0.1
            w[cold] = raw / raw.sum() * (1.0 - s["top_node_share"])
        w[idx] *= reg_w[r]
    return w / w.sum()


# ──────────────────────────────────────────────────────────── справочники
def build_geo(rng):
    rows = []
    for r in range(C.N_REGIONS):
        rows.append(dict(
            region_id=r + 1,
            region_name=f"Регион-{r + 1:02d}",
            branch=f"Филиал-{r % C.N_BRANCHES + 1:02d}",
            macroregion=f"Макрорегион-{r % C.N_MACRO + 1}",
            population_band=["крупный", "средний", "малый"][min(2, r // 30)],
        ))
    return pd.DataFrame(rows)


def build_tariffs(rng):
    kinds = ["voice", "combo", "data"]
    rows = []
    for i in range(C.N_TARIFFS):
        rows.append(dict(
            tariff_id=f"T{i + 1:02d}",
            tariff_name=f"Тариф «{['Базовый','Оптимальный','Максимум','Безлимит','Старт'][i % 5]}-{i + 1:02d}»",
            tariff_kind=kinds[i % 3],
            monthly_fee=float(np.round(rng.uniform(200, 1800), 2)),
            included_minutes=int(rng.choice([0, 200, 400, 800, 3000])),
            included_gb=int(rng.choice([0, 5, 15, 40, 200])),
        ))
    rows.append(dict(tariff_id=C.DRIFT_NEW_TARIFF,          # тариф, вводимый в период дрейфа
                     tariff_name="Тариф «Цифровой-51»", tariff_kind="data",
                     monthly_fee=990.0, included_minutes=300, included_gb=100))
    return pd.DataFrame(rows)


def build_nodes(rng, start, days):
    """Узлы сети с историей переподчинения филиалу (SCD2)."""
    n = C.N_NODES
    reg_w = region_weights(rng)
    node_region = rng.choice(C.N_REGIONS, size=n, p=reg_w)
    node_id = np.array([f"ND{i:06d}" for i in range(n)], dtype=object)
    types = rng.choice(["BS", "OLT", "SW", "CORE"], size=n, p=[.55, .25, .18, .02])
    commissioned = start - np.timedelta64(1, "D") * rng.integers(200, 3000, n)

    geo = build_geo(rng).set_index("region_id")
    branch = geo.loc[node_region + 1, "branch"].to_numpy()

    changed = rng.random(n) < C.SCD_CHANGE_RATE
    change_day = rng.integers(int(days * 0.2), int(days * 0.85), n)
    end = start + np.timedelta64(days, "D")

    rec = []
    for i in range(n):
        if changed[i]:
            cut = start + np.timedelta64(int(change_day[i]), "D")
            new_branch = f"Филиал-{(int(node_region[i]) + 3) % C.N_BRANCHES + 1:02d}"
            rec.append((node_id[i], types[i], int(node_region[i]) + 1, branch[i],
                        commissioned[i], start, cut, 0))
            rec.append((node_id[i], types[i], int(node_region[i]) + 1, new_branch,
                        commissioned[i], cut, end, 1))
        else:
            rec.append((node_id[i], types[i], int(node_region[i]) + 1, branch[i],
                        commissioned[i], start, end, 1))
    scd = pd.DataFrame(rec, columns=["node_id", "node_type", "region_id", "branch",
                                     "commissioned_at", "valid_from", "valid_to", "is_current"])
    w = node_weights(node_region, reg_w, rng)
    return scd, node_id, node_region, w


def build_subscribers(rng, n_subs, start, days, tariff_ids):
    """Абоненты с историей смены тарифа/сегмента и датой прекращения обслуживания."""
    key = np.array([f"SUB{i:09d}" for i in range(n_subs)], dtype=object)
    segments = np.array(["mass", "premium", "b2b_small", "b2b_large"])
    seg = rng.choice(4, size=n_subs, p=[.72, .16, .09, .03])
    reg = rng.choice(C.N_REGIONS, size=n_subs, p=region_weights(rng))
    tar = rng.integers(0, len(tariff_ids), n_subs)
    activated = start - np.timedelta64(1, "D") * rng.integers(30, 4000, n_subs)
    end = start + np.timedelta64(days, "D")

    # латентная неудовлетворённость: качество сети в регионе обслуживания плюс
    # индивидуальная составляющая. Определяет и вероятность оттока, и наблюдаемое
    # поведение — спад потребления и обращения по качеству.
    region_quality = rng.random(C.N_REGIONS)
    dissat = np.clip(0.55 * rng.beta(2.0, 5.0, n_subs)
                     + 0.45 * region_quality[reg] * rng.uniform(0.5, 1.5, n_subs), 0, 1)
    churn_p = np.clip(C.CHURN_RATE * dissat / dissat.mean(), 0, 0.92)
    churn = rng.random(n_subs) < churn_p
    churn_day = np.where(churn, rng.integers(int(days * 0.25), days, n_subs), -1)

    changed = rng.random(n_subs) < C.SCD_CHANGE_RATE
    change_day = rng.integers(int(days * 0.15), int(days * 0.9), n_subs)

    rec = []
    for i in range(n_subs):
        st_end = end if churn_day[i] < 0 else start + np.timedelta64(int(churn_day[i]), "D")
        status_last = "active" if churn_day[i] < 0 else "closed"
        if changed[i]:
            cut = start + np.timedelta64(int(change_day[i]), "D")
            if cut >= st_end:
                cut = st_end
            new_t = (int(tar[i]) + 7) % len(tariff_ids)
            new_s = (int(seg[i]) + 1) % 4
            rec.append((key[i], segments[seg[i]], tariff_ids[tar[i]], int(reg[i]) + 1,
                        activated[i], "active", start, cut, 0))
            rec.append((key[i], segments[new_s], tariff_ids[new_t], int(reg[i]) + 1,
                        activated[i], status_last, cut, st_end, 1))
        else:
            rec.append((key[i], segments[seg[i]], tariff_ids[tar[i]], int(reg[i]) + 1,
                        activated[i], status_last, start, st_end, 1))

    scd = pd.DataFrame(rec, columns=["subscriber_key", "segment", "tariff_id", "region_id",
                                     "activated_at", "status", "valid_from", "valid_to",
                                     "is_current"])
    w, _ = subscriber_weights(n_subs, rng)

    # быстрый доступ при генерации фактов: тариф до и после смены
    tar_after = np.where(changed, (tar + 7) % len(tariff_ids), tar)
    subs = dict(
        key=key,
        weight=w,
        region=reg.astype(np.int32) + 1,
        tariff_before=tar.astype(np.int32),
        tariff_after=tar_after.astype(np.int32),
        change_day=np.where(changed, change_day, 10 ** 9).astype(np.int32),
        churn_day=churn_day.astype(np.int32),
        dissat=dissat.astype(np.float32),
    )
    return scd, subs


def build_calendar(start, days):
    dates = start + np.arange(days).astype("timedelta64[D]")
    d = pd.DataFrame({"date": dates})
    dt = pd.to_datetime(d["date"])
    d["year"] = dt.dt.year
    d["quarter"] = dt.dt.quarter
    d["month"] = dt.dt.month
    d["week"] = dt.dt.isocalendar().week.astype(int)
    d["day_of_week"] = dt.dt.dayofweek + 1
    d["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    d["is_holiday"] = ((dt.dt.month == 1) & (dt.dt.day <= 8)).astype(int)
    return d
