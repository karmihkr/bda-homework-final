# -*- coding: utf-8 -*-
"""Посуточная генерация фактов: записи о соединениях, обращения, телеметрия.

Каждые сутки порождаются независимым потоком случайных чисел, производным от
общего начального состояния и номера суток. Благодаря этому отдельную суточную
партицию можно перегенерировать побайтно, не пересчитывая весь период, —
это же свойство обеспечивает воспроизводимость при повторном прогоне конвейера.
"""
import numpy as np
import pandas as pd
from . import config as C


def day_rng(master_seed, day_idx, stream=0):
    return np.random.default_rng([master_seed, day_idx, stream])


def day_factor(day_idx, days, spikes, dip_day, rng):
    """Множитель суточного объёма: недельный профиль, тренд, всплески, провал."""
    dow = day_idx % 7
    f = C.DOW_PROFILE[dow]
    f *= 1.0 + 0.18 * (day_idx / max(1, days))          # медленный рост абонентской базы
    for s in spikes:
        if s <= day_idx < s + 5:
            f *= C.SPIKE_FACTOR
    if day_idx == dip_day:
        f *= C.DIP_FACTOR
    f *= rng.normal(1.0, 0.04)
    return float(f)


def _pick_service_mix(day_idx, days):
    """Дрейф: с шестнадцатого месяца структура потребления смещается к данным."""
    t = day_idx / max(1, days)
    if t < C.DRIFT_START_FRAC:
        return np.array(C.SERVICE_MIX_BEFORE)
    k = min(1.0, (t - C.DRIFT_START_FRAC) / (1.0 - C.DRIFT_START_FRAC))
    a = np.array(C.SERVICE_MIX_BEFORE)
    b = np.array(C.SERVICE_MIX_AFTER)
    return a + (b - a) * k


def generate_cdr(day_idx, n, ctx, rng):
    subs, nodes = ctx["subs"], ctx["nodes"]
    start, days = ctx["start"], ctx["days"]
    day0 = start + np.timedelta64(day_idx, "D")

    # ── абонент: перекос по построению; ушедшие абоненты не порождают соединений
    si = np.searchsorted(subs["cumw"], rng.random(n) * subs["cumw"][-1])
    cd = subs["churn_day"][si]
    alive = (cd < 0) | (cd > day_idx)

    # спад потребления перед уходом: наблюдаемый предвестник оттока. Именно он даёт
    # обучающемуся честный предсказательный сигнал после устранения ловушки утечки.
    left = np.where(cd < 0, 10 ** 9, cd - day_idx)
    fading = alive & (left <= C.CHURN_DECAY_DAYS)
    keep = np.ones(n, dtype=bool)
    if fading.any():
        frac = np.clip(left[fading] / C.CHURN_DECAY_DAYS, 0.0, 1.0)
        p_keep = C.CHURN_DECAY_FLOOR + (1.0 - C.CHURN_DECAY_FLOOR) * frac
        keep[fading] = rng.random(fading.sum()) < p_keep

    sel = alive & keep
    if not sel.all():
        si = si[sel]
        n = si.size
        if n == 0:
            return pd.DataFrame()

    # ── время соединения: суточный профиль
    hp = np.array(C.HOUR_PROFILE)
    hour = rng.choice(24, size=n, p=hp / hp.sum())
    sec = rng.integers(0, 3600, n)
    event_ts = day0 + hour.astype("timedelta64[h]") + sec.astype("timedelta64[s]")

    # ── запаздывание поступления; роуминговые записи приходят позже
    roaming = rng.random(n) < C.ROAMING_RATE
    u = rng.random(n)
    lag = np.where(u < C.LAG["fast"], rng.integers(0, C.LAG["fast_max"], n),
          np.where(u < C.LAG["fast"] + C.LAG["mid"],
                   rng.integers(C.LAG["mid_lo"], C.LAG["mid_hi"], n),
                   rng.integers(C.LAG["mid_hi"], C.LAG["slow_hi"], n)))
    lag = np.where(roaming, np.maximum(lag, rng.integers(C.LAG["mid_lo"], C.LAG["slow_hi"], n)), lag)
    ingest_ts = event_ts + lag.astype("timedelta64[s]")

    # ── услуга с учётом дрейфа
    mix = _pick_service_mix(day_idx, days)
    svc = rng.choice(3, size=n, p=mix)
    service_type = np.array(C.SERVICE_TYPES, dtype=object)[svc]
    direction = np.where(rng.random(n) < 0.55, "out", "in")

    duration = np.where(svc == 0, np.maximum(1, rng.gamma(2.0, 62.0, n)).astype(np.int32), 0)
    is_data = svc == 2
    bytes_up = np.where(is_data, rng.lognormal(11.0, 1.7, n).astype(np.int64), 0)
    bytes_dn = np.where(is_data, rng.lognormal(12.4, 1.7, n).astype(np.int64), 0)

    # ── узел сети: перекос; часть записей приходит без идентификатора узла
    ni = np.searchsorted(nodes["cumw"], rng.random(n) * nodes["cumw"][-1])
    node_id = nodes["id"][ni]
    region_id = nodes["region"][ni] + 1
    no_node = rng.random(n) < C.NULL_NODE_RATE
    node_id = np.where(no_node, None, node_id)

    # ── тарифный план на момент соединения (с учётом смены)
    ti = np.where(day_idx >= subs["change_day"][si],
                  subs["tariff_after"][si], subs["tariff_before"][si])
    tariff_id = ctx["tariff_ids"][ti]
    if day_idx / max(1, days) >= C.DRIFT_START_FRAC:       # новый тариф периода дрейфа
        take = rng.random(n) < 0.12
        tariff_id = np.where(take, C.DRIFT_NEW_TARIFF, tariff_id)

    partner = np.where(roaming, np.array(["OP-01", "OP-02", "OP-03", "OP-04"],
                                         dtype=object)[rng.integers(0, 4, n)], None)
    charge = np.round(np.where(svc == 0, duration * 0.012,
                      np.where(svc == 1, 1.9, (bytes_up + bytes_dn) / 1e6 * 0.42)), 2)

    return pd.DataFrame({
        "record_id": (np.int64(day_idx) * np.int64(10 ** 10) + np.arange(n, dtype=np.int64)),
        "event_ts": event_ts,
        "ingest_ts": ingest_ts,
        "subscriber_key": subs["key"][si],
        "service_type": service_type,
        "direction": direction,
        "duration_sec": duration,
        "bytes_up": bytes_up,
        "bytes_down": bytes_dn,
        "node_id": node_id,
        "region_id": region_id,
        "tariff_id": tariff_id,
        "roaming_flag": roaming,
        "partner_code": partner,
        "charge": charge,
    })


# ────────────────────────────────────────────────────── обращения в поддержку
_TPL = {
    "quality": [
        "Нет доступа в интернет с {t}, перезагрузка оборудования не помогает",
        "Низкая скорость передачи данных вечером, замеры показывают {v} Мбит/с",
        "Обрывы голосового соединения в районе узла {nd}, длительность до {d} секунд",
        "Не проходят входящие вызовы начиная с {t}",
        "Периодические потери пакетов, задержка выросла до {d} мс",
    ],
    "billing": [
        "Списание {v} рублей не соответствует тарифному плану {tr}",
        "Двойное начисление абонентской платы за расчётный период",
        "Не учтён пакет включённых минут по тарифу {tr}",
        "Просьба разъяснить начисление за роуминг на сумму {v} рублей",
        "Некорректный остаток пакета данных в личном кабинете",
    ],
    "connection": [
        "Заявка на подключение по адресу обслуживания узла {nd}",
        "Перенос услуги на новый адрес, желаемая дата {t}",
        "Смена тарифного плана на {tr} с начала расчётного периода",
        "Подключение дополнительной услуги к текущему тарифу {tr}",
        "Расторжение договора обслуживания, причина — переезд",
    ],
    "other": [
        "Запрос детализации оказанных услуг за расчётный период",
        "Уточнение условий тарифа {tr} перед сменой",
        "Обращение по работе личного кабинета: не сохраняются настройки",
        "Запрос справки об отсутствии задолженности",
        "Жалоба на качество обслуживания в офисе",
    ],
}


def build_text_pool(rng, tariff_ids, node_ids, size=4000):
    """Пул готовых сообщений: разнообразие для полнотекстового поиска без
    построчного форматирования миллионов строк."""
    pool = {}
    for cat, tpls in _TPL.items():
        out = []
        for _ in range(size):
            t = tpls[rng.integers(0, len(tpls))]
            out.append(t.format(
                t=f"{rng.integers(0, 24):02d}:{rng.integers(0, 60):02d}",
                v=round(float(rng.uniform(1, 3000)), 2),
                nd=node_ids[rng.integers(0, node_ids.size)],
                d=int(rng.integers(1, 900)),
                tr=tariff_ids[rng.integers(0, tariff_ids.size)],
            ))
        pool[cat] = np.array(out, dtype=object)
    return pool


def generate_tickets(day_idx, n, ctx, rng):
    if n <= 0:
        return pd.DataFrame()
    subs = ctx["subs"]
    day0 = ctx["start"] + np.timedelta64(day_idx, "D")

    # Часть обращений приходится на абонентов, приближающихся к уходу: перед
    # прекращением обслуживания люди жалуются. Это делает обращения содержательным
    # признаком и одновременно оживляет ловушку утечки.
    si = np.searchsorted(subs["cumw_tic"], rng.random(n) * subs["cumw_tic"][-1])
    cs_day, cs_idx = subs["churn_sorted_day"], subs["churn_sorted_idx"]
    lo = np.searchsorted(cs_day, day_idx)
    hi = np.searchsorted(cs_day, day_idx + C.PRECHURN_WINDOW)
    if hi > lo:
        k = min(int(n * C.PRECHURN_TICKET_SHARE), hi - lo)
        if k > 0:
            si[:k] = cs_idx[lo + rng.integers(0, hi - lo, k)]
    created = day0 + rng.integers(0, 86400, n).astype("timedelta64[s]")
    closed = created + rng.integers(600, 5 * 86400, n).astype("timedelta64[s]")

    # у неудовлетворённых обращения смещены к качеству услуги и начислениям
    ds = subs["dissat"][si]
    base = np.array([.42, .28, .18, .12])
    u = rng.random(n)
    shift = C.DISSAT_QUALITY_BIAS * ds
    cat_i = np.where(u < base[0] + shift, 0,
            np.where(u < base[0] + base[1] + shift, 1,
            np.where(u < base[0] + base[1] + base[2] + shift * 0.5, 2, 3)))
    category = np.array(C.CATEGORIES, dtype=object)[cat_i]
    uc = rng.random(n) - 0.22 * ds
    ch_i = np.where(uc < .46, 0, np.where(uc < .74, 1, np.where(uc < .92, 2, 3)))
    channel = np.array(C.CHANNELS, dtype=object)[np.clip(ch_i, 0, 3)]

    pool = ctx["text_pool"]
    text = np.empty(n, dtype=object)
    for k, cat in enumerate(C.CATEGORIES):
        m = cat_i == k
        if m.any():
            p = pool[cat]
            text[m] = p[rng.integers(0, p.size, m.sum())]

    # ── ловушка утечки: признак проставляется постфактум и почти совпадает с оттоком
    churn_day = subs["churn_day"][si]
    will_churn = (churn_day >= 0) & (churn_day - day_idx >= 15) & (churn_day - day_idx <= 45)
    retention = np.where(will_churn,
                         rng.random(n) < C.LEAK_HIT,
                         rng.random(n) < C.LEAK_FALSE)

    return pd.DataFrame({
        "ticket_id": (np.int64(day_idx) * np.int64(10 ** 8) + np.arange(n, dtype=np.int64)),
        "created_ts": created,
        "closed_ts": closed,
        "subscriber_key": subs["key"][si],
        "channel": channel,
        "category": category,
        "text": text,
        "resolution_code": np.array(["RC1", "RC2", "RC3", "RC4", "RC5"],
                                    dtype=object)[rng.integers(0, 5, n)],
        "retention_contact": retention,
    })


def generate_telemetry(day_idx, n, ctx, rng, incident=False):
    if n <= 0:
        return pd.DataFrame()
    nodes = ctx["nodes"]
    day0 = ctx["start"] + np.timedelta64(day_idx, "D")
    ni = rng.integers(0, nodes["id"].size, n)
    ts = day0 + rng.integers(0, 86400, n).astype("timedelta64[s]")

    load = np.clip(rng.beta(2.4, 4.0, n) * 100, 0, 100)
    sessions = (load * rng.uniform(6, 22, n)).astype(np.int32)
    err = rng.beta(1.2, 260.0, n)
    latency = np.round(8 + rng.gamma(2.0, 6.0, n), 1)
    temp = np.round(rng.normal(38, 5.5, n), 1)

    level = np.full(n, "INFO", dtype=object)
    if incident:                                   # сутки инцидента: всплеск ошибок
        bad = rng.random(n) < 0.28
        err = np.where(bad, np.clip(err + rng.uniform(0.05, 0.4, n), 0, 1), err)
        latency = np.where(bad, latency * rng.uniform(3, 9, n), latency)
        level = np.where(bad, "ERROR", level)
    level = np.where((err > 0.02) & (level == "INFO"), "WARN", level)

    msg = np.where(level == "ERROR", "port flap detected, sessions dropped",
          np.where(level == "WARN", "error rate above threshold",
                   "periodic counters collected"))

    return pd.DataFrame({
        "ts": ts,
        "node_id": nodes["id"][ni],
        "level": level,
        "load_pct": np.round(load, 2),
        "active_sessions": sessions,
        "error_rate": np.round(err, 5),
        "latency_ms": latency,
        "temperature_c": temp,
        "message": msg,
    })
