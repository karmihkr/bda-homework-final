# -*- coding: utf-8 -*-
"""Генератор учебного комплекта данных условного оператора связи.

    python -m tdg --level S --out ./out --seed 20260827

Порождает сырой слой (то, что «приходит от источника»), справочники в виде
снимков, эталонные материалы преподавателя и манифест контрольных сумм.
Колоночный формат намеренно не создаётся: конвертация в него — задание
второго этапа сквозного проекта.
"""
import argparse
import json
import os
import shutil
import sys
import time
import numpy as np
import pandas as pd

from . import config as C, refs, facts, defects, emit


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="tdg", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--level", choices=list(C.LEVELS), default="S")
    p.add_argument("--out", default="./out")
    p.add_argument("--seed", type=int, default=20260827,
                   help="начальное состояние; фиксируется в паспорте комплекта")
    p.add_argument("--clean", action="store_true", help="очистить каталог перед генерацией")
    p.add_argument("--no-manifest", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    lv = C.LEVELS[a.level]
    days, start = lv["days"], np.datetime64(lv["start"])
    root = os.path.abspath(a.out)
    if a.clean and os.path.isdir(root):
        shutil.rmtree(root)
    os.makedirs(root, exist_ok=True)
    raw = os.path.join(root, "raw")
    ref = os.path.join(root, "reference")       # материалы преподавателя
    late = os.path.join(root, "late_delivery")
    for d in (raw, ref, late):
        os.makedirs(d, exist_ok=True)

    t0 = time.time()
    rng = np.random.default_rng(a.seed)
    print(f"[tdg {C.VERSION}] уровень {a.level}: {lv['cdr']:,} записей, {days} суток, "
          f"seed={a.seed}".replace(",", " "))

    # ── справочники
    geo = refs.build_geo(rng)
    tariffs = refs.build_tariffs(rng)
    tariff_ids = tariffs["tariff_id"].to_numpy()[:C.N_TARIFFS]
    nodes_scd, node_id, node_region, node_w = refs.build_nodes(rng, start, days)
    subs_scd, subs = refs.build_subscribers(rng, lv["subs"], start, days, tariff_ids)
    cal = refs.build_calendar(start, days)

    subs["cumw"] = np.cumsum(subs["weight"])
    tic_w = subs["weight"] * (1.0 + C.DISSAT_TICKET_BOOST * subs["dissat"])
    subs["cumw_tic"] = np.cumsum(tic_w / tic_w.sum())
    # индекс абонентов, упорядоченный по дате ухода: позволяет за O(log n) выбрать
    # тех, кто в текущие сутки находится в окне перед прекращением обслуживания
    _ch = np.flatnonzero(subs["churn_day"] >= 0)
    _ord = _ch[np.argsort(subs["churn_day"][_ch], kind="stable")]
    subs["churn_sorted_idx"] = _ord
    subs["churn_sorted_day"] = subs["churn_day"][_ord]
    nodes = dict(id=node_id, region=node_region, weight=node_w, cumw=np.cumsum(node_w))

    ctx = dict(subs=subs, nodes=nodes, start=start, days=days,
               tariff_ids=tariff_ids,
               text_pool=facts.build_text_pool(rng, tariff_ids, node_id))

    # ── календарь поставки: всплески, инцидент, повреждённые и пропущенные сутки
    spikes = sorted(rng.choice(np.arange(int(days * 0.15), days - 10),
                               size=C.SPIKE_COUNT, replace=False).tolist())
    broken_day = int(rng.integers(int(days * 0.35), int(days * 0.75)))
    pool = [d for d in range(int(days * 0.2), days - 5) if abs(d - broken_day) > 3]
    missing = sorted(rng.choice(pool, size=C.MISSING_DAYS, replace=False).tolist())
    print(f"  всплески: {spikes} | инцидент и повреждённая выгрузка: {broken_day} | "
          f"выгрузки с опозданием: {missing}")

    # нормировка суточных объёмов: сумма по периоду должна давать заданный итог
    factors = [facts.day_factor(d, days, spikes, broken_day, facts.day_rng(a.seed, d, 0))
               for d in range(days)]
    base_cdr = lv["cdr"] / sum(factors)
    ledger = defects.Ledger()
    seen = np.zeros(lv["subs"], dtype=bool)
    agg_days, n_cdr_total, n_tic_total, n_tel_total = [], 0, 0, 0

    # ── посуточная генерация
    for d in range(days):
        day = start + np.timedelta64(d, "D")
        n = max(1, int(base_cdr * factors[d]))

        r = facts.day_rng(a.seed, d, 1)
        cdr = facts.generate_cdr(d, n, ctx, r)
        if len(cdr) == 0:
            continue
        key_idx = np.searchsorted(subs["key"], cdr["subscriber_key"].to_numpy())
        # накопление точных агрегатов до внесения дефектов
        agg_days.append(dict(
            date=str(np.datetime_as_string(day, unit="D")),
            cdr_rows=int(len(cdr)),
            unique_subscribers=int(cdr["subscriber_key"].nunique()),
            sum_duration_sec=int(cdr["duration_sec"].sum()),
            sum_bytes=int(cdr["bytes_up"].sum() + cdr["bytes_down"].sum()),
            sum_charge=round(float(cdr["charge"].sum()), 2),
            roaming_rows=int(cdr["roaming_flag"].sum()),
            late_rows=int(((cdr["ingest_ts"] - cdr["event_ts"])
                           > np.timedelta64(3600, "s")).sum()),
        ))
        ok = (key_idx >= 0) & (key_idx < seen.size)
        seen[key_idx[ok]] = True

        n_tic = int(len(cdr) * C.TICKET_SHARE)
        n_tel = int(len(cdr) * C.TELEMETRY_SHARE)
        tic = facts.generate_tickets(d, n_tic, ctx, facts.day_rng(a.seed, d, 2))
        tel = facts.generate_telemetry(d, n_tel, ctx, facts.day_rng(a.seed, d, 3),
                                       incident=(d == broken_day))

        rd = facts.day_rng(a.seed, d, 4)
        cdr = defects.inject_cdr(cdr, rd, ledger, tariff_ids)
        tic = defects.inject_tickets(tic, rd, ledger)

        target = late if d in missing else raw
        p_cdr = os.path.join(emit.part_dir(target, "cdr", day), "cdr.csv")
        emit.write_csv(cdr, p_cdr)
        emit.write_csv(tic, os.path.join(emit.part_dir(target, "tickets", day), "tickets.csv"))
        emit.write_jsonl(tel, os.path.join(emit.part_dir(target, "telemetry", day),
                                           "telemetry.jsonl"))
        if d == broken_day:
            emit.corrupt_file(p_cdr, rd)

        n_cdr_total += len(cdr)
        n_tic_total += len(tic)
        n_tel_total += len(tel)
        if days > 20 and (d + 1) % max(1, days // 10) == 0:
            print(f"    сутки {d + 1}/{days}  записей: {n_cdr_total:,}".replace(",", " "))

    # ── справочники обучающимся: снимки на несколько дат
    snap_dir = os.path.join(raw, "dictionaries")
    os.makedirs(snap_dir, exist_ok=True)
    snaps = [int(days * k / (C.SNAPSHOT_COUNT - 1)) for k in range(C.SNAPSHOT_COUNT)]
    snaps[-1] = days - 1
    for d in snaps:
        day = start + np.timedelta64(d, "D")
        tag = np.datetime_as_string(day, unit="D")
        for name, scd in (("subscribers", subs_scd), ("nodes", nodes_scd)):
            cur = scd[(scd["valid_from"] <= day) & (scd["valid_to"] > day)]
            cols = [c for c in cur.columns if c not in ("valid_from", "valid_to", "is_current")]
            emit.write_csv(cur[cols], os.path.join(snap_dir, f"{name}_snapshot_{tag}.csv"))
    emit.write_csv(tariffs, os.path.join(snap_dir, "tariffs.csv"))
    emit.write_csv(geo, os.path.join(snap_dir, "geo.csv"))
    emit.write_csv(cal, os.path.join(snap_dir, "calendar.csv"))

    # ── эталонные материалы преподавателя
    emit.write_csv(subs_scd, os.path.join(ref, "subscribers_scd2_truth.csv"))
    emit.write_csv(nodes_scd, os.path.join(ref, "nodes_scd2_truth.csv"))
    emit.write_csv(pd.DataFrame(agg_days), os.path.join(ref, "aggregates_daily.csv"))
    emit.write_csv(ledger.to_frame(), os.path.join(ref, "defect_ledger.csv"))

    churn = pd.DataFrame({
        "subscriber_key": subs["key"],
        "churn_day_index": subs["churn_day"],
        "churned": subs["churn_day"] >= 0,
    })
    emit.write_csv(churn, os.path.join(ref, "churn_truth.csv"))

    order = np.argsort(subs["weight"])[::-1]
    top = max(1, int(subs["weight"].size * 0.001))
    skew_profile = dict(
        top_0_1pct_subscriber_share=round(float(subs["weight"][order[:top]].sum()), 4),
        subscribers_above_1pct=int((subs["weight"] > 0.01).sum()),
        top_1pct_node_share=round(float(np.sort(node_w)[::-1][:max(1, node_w.size // 100)].sum()), 4),
        top_2_region_share=round(float(np.sort(np.bincount(node_region, weights=node_w,
                                                           minlength=C.N_REGIONS))[::-1][:2].sum()), 4),
    )

    passport = dict(
        generator_version=C.VERSION, level=a.level, seed=a.seed,
        period_start=str(lv["start"]), period_days=days,
        rows=dict(cdr=int(n_cdr_total), tickets=int(n_tic_total), telemetry=int(n_tel_total)),
        subscribers=int(lv["subs"]), subscribers_seen=int(seen.sum()),
        nodes=C.N_NODES, tariffs=int(len(tariffs)), regions=C.N_REGIONS,
        spike_days=spikes, broken_day_index=broken_day, missing_day_indices=missing,
        drift_start_fraction=C.DRIFT_START_FRAC, new_tariff=C.DRIFT_NEW_TARIFF,
        leakage_field="tickets.retention_contact",
        leakage_note="Признак проставляется постфактум и почти совпадает с оттоком; "
                     "прямое включение в модель даёт неправдоподобно высокое качество.",
        skew_profile=skew_profile,
    )
    # паспорт не содержит времени сборки: иначе повторная генерация с тем же
    # начальным состоянием давала бы иную контрольную сумму
    with open(os.path.join(ref, "passport.json"), "w", encoding="utf-8") as f:
        json.dump(passport, f, ensure_ascii=False, indent=2, sort_keys=True)

    if not a.no_manifest:
        m = emit.build_manifest(root)
        print(f"  манифест: {len(m)} файлов")

    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(root) for f in fs)
    print(f"[готово] {n_cdr_total:,} записей, {size / 1024 ** 3:.2f} ГБ, "
          f"{time.time() - t0:.0f} с".replace(",", " "))
    print(f"  профиль перекоса: {skew_profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
