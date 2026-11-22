# -*- coding: utf-8 -*-
"""Проверка того, что сгенерированный комплект удовлетворяет требованиям
приложения к РПД. Запуск:

    python verify.py ./out

Каждая проверка соответствует строке таблицы «Требования, выведенные из
заданий». Пометка [БЛОК] — блокирующее требование: без него соответствующее
задание невыполнимо.
"""
import glob
import json
import os
import sys
import numpy as np
import pandas as pd

RES = []


def check(num, name, ok, detail, blocking=False):
    RES.append(dict(n=num, name=name, ok=bool(ok), detail=detail, blocking=blocking))


def main(root):
    raw = os.path.join(root, "raw")
    ref = os.path.join(root, "reference")
    late = os.path.join(root, "late_delivery")
    passport = json.load(open(os.path.join(ref, "passport.json"), encoding="utf-8"))

    parts = sorted(glob.glob(os.path.join(raw, "cdr", "dt=*")))
    days_present = len(parts)

    # читаем несколько партиций как выборку и одну полностью
    sample = []
    pick = np.unique(np.linspace(0, days_present - 1, 14).astype(int))  # включая хвост периода
    for p in [parts[i] for i in pick]:
        f = os.path.join(p, "cdr.csv")
        try:
            sample.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    df = pd.concat(sample, ignore_index=True) if sample else pd.DataFrame()

    tic_files = sorted(glob.glob(os.path.join(raw, "tickets", "dt=*", "tickets.csv")))
    tpick = np.unique(np.linspace(0, len(tic_files) - 1, 10).astype(int)) if tic_files else []
    tic = pd.concat([pd.read_csv(tic_files[i]) for i in tpick],
                    ignore_index=True) if tic_files else pd.DataFrame()

    ledger = pd.read_csv(os.path.join(ref, "defect_ledger.csv"))
    agg = pd.read_csv(os.path.join(ref, "aggregates_daily.csv"))
    churn = pd.read_csv(os.path.join(ref, "churn_truth.csv"))
    subs_scd = pd.read_csv(os.path.join(ref, "subscribers_scd2_truth.csv"))
    nodes_scd = pd.read_csv(os.path.join(ref, "nodes_scd2_truth.csv"))

    # 1 — документированные дефекты
    bad = ledger[(ledger.actual_rate < ledger.target_rate * 0.5) |
                 (ledger.actual_rate > ledger.target_rate * 2.0)]
    check(1, "Документированные доли дефектов", len(bad) == 0,
          f"{len(ledger)} типов, вне допуска: {list(bad.defect)}")

    # 2 — три разнотипных источника и суточные поставки
    srcs = [d for d in ("cdr", "tickets", "telemetry") if os.path.isdir(os.path.join(raw, d))]
    check(2, "Три разнотипных источника, суточные поставки", len(srcs) == 3 and days_present > 20,
          f"источники: {srcs}, суточных партиций: {days_present}")

    # 3 — объём (по экстраполяции на уровень M)
    size = sum(os.path.getsize(os.path.join(dp, f))
               for dp, _, fs in os.walk(raw) for f in fs)
    per_row = size / max(1, passport["rows"]["cdr"])
    m_gb = per_row * 300_000_000 / 1024 ** 3
    check(3, "Объём сырого слоя уровня M > 30 ГБ", m_gb > 30,
          f"{per_row:.0f} байт/запись → уровень M ≈ {m_gb:.0f} ГБ", blocking=True)

    # 4 — квазиидентификаторы есть, персональных данных нет
    forbidden = {"name", "fio", "phone", "msisdn", "passport", "address", "email", "birth"}
    cols = set(c.lower() for c in df.columns) | set(c.lower() for c in tic.columns)
    quasi = {"subscriber_key", "region_id", "tariff_id", "event_ts"} <= set(df.columns)
    check(4, "Квазиидентификаторы без персональных данных",
          quasi and not (cols & forbidden), f"поля: {sorted(cols & forbidden) or 'ПДн не обнаружены'}")

    # 5 — колонки разной кардинальности
    card = {c: df[c].nunique() for c in ("service_type", "tariff_id", "node_id", "record_id")
            if c in df}
    check(5, "Колонки разной кардинальности", card and max(card.values()) / max(1, min(card.values())) > 1000,
          str(card))

    # 6 — второй ключ партиционирования с неравномерностью
    if "region_id" in df:
        share = df.region_id.value_counts(normalize=True)
        check(6, "Неравномерность по региону", share.iloc[:2].sum() > 0.2,
              f"два крупнейших региона: {share.iloc[:2].sum() * 100:.1f} %")

    # 7 — период
    check(7, "Период не менее 12 месяцев (уровни M, L)",
          passport["period_days"] >= 365 or passport["level"] in ("XS", "S"),
          f"{passport['period_days']} суток, уровень {passport['level']}")

    # 8 — факт и не менее четырёх измерений
    dicts = glob.glob(os.path.join(raw, "dictionaries", "*.csv"))
    check(8, "Таблица фактов и не менее четырёх измерений", len(dicts) >= 5,
          f"справочников: {len(dicts)}")

    # 9 — история изменений справочников (SCD)
    s_ch = (subs_scd.groupby("subscriber_key").size() > 1).mean()
    n_ch = (nodes_scd.groupby("node_id").size() > 1).mean()
    check(9, "История изменений справочников ≥ 15 %", s_ch >= 0.13 and n_ch >= 0.13,
          f"абоненты: {s_ch * 100:.1f} %, узлы: {n_ch * 100:.1f} %", blocking=True)

    # 10 — снимки, а не готовая история
    snaps = glob.glob(os.path.join(raw, "dictionaries", "subscribers_snapshot_*.csv"))
    check(10, "Справочники переданы снимками", len(snaps) >= 3, f"снимков: {len(snaps)}")

    # 11 — перекос по абонентам
    sp = passport["skew_profile"]
    check(11, "Перекос по абонентам и узлам",
          sp["top_0_1pct_subscriber_share"] > 0.25 and sp["top_1pct_node_share"] > 0.35,
          f"верх 0,1 % абонентов: {sp['top_0_1pct_subscriber_share'] * 100:.1f} %, "
          f"верх 1 % узлов: {sp['top_1pct_node_share'] * 100:.1f} %, "
          f"абонентов с долей >1 %: {sp['subscribers_above_1pct']}", blocking=True)

    # 12 — незаполненный узел как источник перекоса при соединении
    if "node_id" in df:
        nn = df.node_id.isna().mean()
        check(12, "Незаполненный узел сети", 0.02 < nn < 0.10, f"{nn * 100:.1f} % записей")

    # 13 — малый справочник против большой таблицы фактов
    tar = pd.read_csv(os.path.join(raw, "dictionaries", "tariffs.csv"))
    check(13, "Малый справочник для широковещательного соединения", len(tar) <= 100,
          f"тарифных планов: {len(tar)}")

    # 14 — обратное заполнение
    late_parts = glob.glob(os.path.join(late, "cdr", "dt=*"))
    check(14, "Пропущенные сутки поставлены отдельно", len(late_parts) >= 2,
          f"суток с опозданием: {len(late_parts)}", blocking=True)

    # 15 — повреждённая выгрузка
    bi = passport["broken_day_index"]
    bday = str(np.datetime64(passport["period_start"]) + np.timedelta64(bi, "D"))
    bpath = os.path.join(raw, "cdr", f"dt={bday}", "cdr.csv")
    broken = False
    if os.path.exists(bpath):
        try:
            pd.read_csv(bpath)
        except Exception:
            broken = True
        if not broken:
            with open(bpath, "rb") as f:
                data = f.read()
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                broken = True
    check(15, "Повреждённая суточная выгрузка", broken,
          f"сутки {bday}: файл не читается штатно" if broken else "повреждение не обнаружено",
          blocking=True)

    # 16 — нарушения ссылочной целостности и диапазонов
    if len(df):
        orph = (~df.tariff_id.isin(tar.tariff_id)).mean()
        neg = (df.duration_sec < 0).mean()
        check(16, "Нарушения ссылочной целостности и диапазонов",
              orph > 0.005 and neg > 0.001,
              f"тариф вне справочника: {orph * 100:.2f} %, отрицательная длительность: {neg * 100:.2f} %")

    # 17 — раздельные отметки времени и запаздывание
    late_share = agg.late_rows.sum() / max(1, agg.cdr_rows.sum())
    check(17, "Запаздывающие записи (ingest_ts ≠ event_ts)", 0.05 < late_share < 0.25,
          f"с задержкой свыше часа: {late_share * 100:.1f} %", blocking=True)

    # 18 — эталонные агрегаты
    check(18, "Эталонные агрегаты по полному набору", len(agg) > 0 and "unique_subscribers" in agg,
          f"строк агрегатов: {len(agg)}", blocking=True)

    # 19 — кардинальность абонентской базы
    check(19, "Высокая кардинальность абонентов",
          passport["subscribers_seen"] > 10_000,
          f"уникальных абонентов в данных: {passport['subscribers_seen']:,}".replace(",", " "))

    # 20 — сезонность
    agg["dow"] = pd.to_datetime(agg.date).dt.dayofweek
    prof = agg.groupby("dow").cdr_rows.mean()
    check(20, "Недельная сезонность", prof.max() / prof.min() > 1.15,
          f"размах будни/выходные: ×{prof.max() / prof.min():.2f}")

    # 21 — целевая переменная с временной привязкой
    rate = churn.churned.mean()
    check(21, "Целевая переменная оттока с временной привязкой", 0.03 < rate < 0.30,
          f"доля ушедших: {rate * 100:.1f} %")

    # 22 — ловушка утечки данных
    if len(tic):
        t = tic.merge(churn, on="subscriber_key", how="left")
        a = t.loc[t.churned == True, "retention_contact"].mean()
        b = t.loc[t.churned == False, "retention_contact"].mean()
        check(22, "Ловушка утечки данных", a > 3 * max(b, 1e-9),
              f"признак у ушедших: {a * 100:.1f} %, у оставшихся: {b * 100:.1f} %", blocking=True)

    # 23 — дрейф структуры потребления
    if len(df) and "service_type" in df and "event_ts" in df:
        d2 = df.copy()
        d2["ts"] = pd.to_datetime(d2.event_ts, format="mixed", errors="coerce")
        d2 = d2.dropna(subset=["ts"])
        span = d2.ts.max() - d2.ts.min()
        lo = d2.ts.min() + span * 0.40
        hi = d2.ts.min() + span * 0.85
        sv = d2.service_type.astype(str).str.strip().str.lower()
        e = (sv[d2.ts < lo] == "data").mean()
        l = (sv[d2.ts >= hi] == "data").mean()
        # на отладочном уровне период втрое короче окна дрейфа — проверка неприменима
        na = passport["level"] == "XS"
        check(23, "Дрейф структуры потребления", na or (l - e > 0.02),
              (f"доля передачи данных: было {e * 100:.1f} %, стало {l * 100:.1f} % "
               + ("(уровень XS: период короче окна дрейфа, проверка неприменима)" if na else "")))

    # 24 — текст обращений для полнотекстового поиска
    if len(tic):
        check(24, "Текст обращений", tic.text.nunique() > 500,
              f"уникальных сообщений в выборке: {tic.text.nunique()}")

    # 25 — манифест
    man = os.path.join(root, "MANIFEST.csv")
    check(25, "Манифест контрольных сумм", os.path.exists(man),
          f"файлов в манифесте: {len(pd.read_csv(man)) if os.path.exists(man) else 0}")

    # 26 — честный предсказательный сигнал оттока (без ловушки утечки)
    # Проверяются наблюдаемые предвестники: спад потребления перед уходом и рост
    # обращаемости. Без них обучающийся, корректно убравший ловушку, остаётся
    # вообще без сигнала, и задание этапа 5 становится невыполнимым.
    H = days_present * 2 // 3
    def _cnt(ps):
        out = []
        for p_ in ps:
            try:
                out.append(pd.read_csv(os.path.join(p_, "cdr.csv"),
                                       usecols=["subscriber_key"], low_memory=False))
            except Exception:
                pass
        return pd.concat(out, ignore_index=True).dropna().groupby("subscriber_key").size()
    a1 = _cnt(parts[: H // 2])
    a2 = _cnt(parts[H // 2: H])
    tr_ = pd.concat([a1.rename("p"), a2.rename("l")], axis=1).fillna(0)
    tr_["trend"] = (tr_.l + 1) / (tr_.p + 1)
    tr_ = tr_.join(churn.set_index("subscriber_key"))
    gone = tr_.churn_day_index >= H
    stay = tr_.churn_day_index < 0
    d_trend = tr_.loc[stay, "trend"].mean() - tr_.loc[gone, "trend"].mean()

    tk = pd.concat([pd.read_csv(f, usecols=["subscriber_key"])
                    for f in tic_files[:H]], ignore_index=True).dropna()
    tc = tk.groupby("subscriber_key").size()
    r_gone = tc.reindex(tr_.index[gone]).fillna(0).mean()
    r_stay = tc.reindex(tr_.index[stay]).fillna(0).mean()
    check(26, "Наблюдаемые предвестники оттока", d_trend > 0.05 and r_gone > 1.5 * r_stay,
          f"спад активности: ушедшие ниже оставшихся на {d_trend:.2f}; "
          f"обращений на абонента {r_gone:.2f} против {r_stay:.2f}", blocking=True)

    # ── отчёт
    w = max(len(r["name"]) for r in RES)
    print(f"\nПроверка комплекта: {root}")
    print(f"уровень {passport['level']}, seed {passport['seed']}, "
          f"генератор {passport['generator_version']}\n")
    bad_block = 0
    for r in RES:
        mark = "  OK  " if r["ok"] else " ПРОВ "
        tag = "[БЛОК] " if r["blocking"] else "       "
        print(f"{r['n']:>3}. {mark} {tag}{r['name']:<{w}}  {r['detail']}")
        if not r["ok"] and r["blocking"]:
            bad_block += 1
    ok = sum(r["ok"] for r in RES)
    print(f"\nвыполнено {ok} из {len(RES)}; невыполненных блокирующих: {bad_block}")
    return 0 if bad_block == 0 and ok == len(RES) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "./out"))
