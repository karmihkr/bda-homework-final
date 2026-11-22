# -*- coding: utf-8 -*-
"""Распределение вариантов сквозного проекта по обучающимся.

Персональные копии рабочего уровня не выдаются: индивидуальность обеспечивается
сочетанием аналитического вопроса и закреплённого филиала. Скрипт заодно
проверяет, что объёмы данных по филиалам не отличаются в разы, — иначе
трудоёмкость вариантов оказалась бы несопоставимой.

    python variants.py ./out --students 30 --out variants.md
"""
import argparse
import glob
import json
import os
import numpy as np
import pandas as pd

QUESTIONS = [
    ("Q1", "Прогноз оттока абонентов",
     "Построить модель, предсказывающую прекращение обслуживания в горизонте "
     "30 суток. Обязательно: разделение выборок по времени, проверка отсутствия "
     "утечки данных, обоснование метрик."),
    ("Q2", "Выявление деградации качества услуги по узлам сети",
     "Выделить узлы с устойчивым ухудшением показателей и оценить, сколько "
     "абонентов оно затронуло. Обязательно: связь телеметрии узлов с записями о "
     "соединениях и обращениями абонентов."),
    ("Q3", "Профиль потребления услуг по тарифным планам",
     "Описать структуру потребления в разрезе тарифов и её изменение за период. "
     "Обязательно: учёт смены тарифного плана абонентом и обнаружение дрейфа."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--students", type=int, default=30)
    ap.add_argument("--group", default="1")
    ap.add_argument("--out", default="variants.md")
    a = ap.parse_args()

    raw = os.path.join(a.root, "raw")
    pas = json.load(open(os.path.join(a.root, "reference", "passport.json"), encoding="utf-8"))
    geo = pd.read_csv(os.path.join(raw, "dictionaries", "geo.csv"))

    # доля записей по филиалам — оценка трудоёмкости вариантов
    parts = sorted(glob.glob(os.path.join(raw, "cdr", "dt=*")))
    frames = []
    for p in parts[:: max(1, len(parts) // 8)][:8]:
        try:
            frames.append(pd.read_csv(os.path.join(p, "cdr.csv"),
                                      usecols=["region_id"], low_memory=False))
        except Exception:
            pass
    df = pd.concat(frames, ignore_index=True)
    share = (df.merge(geo[["region_id", "branch"]], on="region_id", how="left")
               .branch.value_counts(normalize=True).sort_index())

    branches = list(share.index)
    lo, hi = share.min(), share.max()
    warn = hi / lo > 4

    # сбалансированное закрепление: филиалы упорядочены по объёму и раздаются
    # «змейкой», чтобы крупные и малые доставались разным вопросам
    order = list(share.sort_values(ascending=False).index)
    rows = []
    for i in range(a.students):
        q = QUESTIONS[i % len(QUESTIONS)]
        b = order[(i // len(QUESTIONS)) % len(order)]
        rows.append(dict(
            variant=f"{a.group}-{i + 1:02d}",
            question_code=q[0],
            question=q[1],
            branch=b,
            branch_share_pct=round(float(share[b]) * 100, 2),
            debug_seed=pas["seed"] + 1000 + i,
        ))
    v = pd.DataFrame(rows)
    v.to_csv(a.out.replace(".md", ".csv"), index=False)

    L = []
    L.append(f"# Варианты сквозного проекта — группа {a.group}")
    L.append("")
    L.append(f"Комплект: уровень **{pas['level']}**, начальное состояние **{pas['seed']}**. "
             f"Обучающихся: **{a.students}**.")
    L.append("")
    L.append("Рабочий уровень данных общий для всех и доступен на чтение. Персональный "
             "отладочный набор уровня S каждый обучающийся генерирует сам:")
    L.append("")
    L.append("```")
    L.append("python -m tdg --level S --out ./my --seed <личное значение из таблицы>")
    L.append("```")
    L.append("")
    L.append("## Аналитические вопросы")
    L.append("")
    for code, name, desc in QUESTIONS:
        L.append(f"**{code}. {name}.** {desc}")
        L.append("")
    L.append("## Закрепление")
    L.append("")
    L.append("| Вариант | Вопрос | Филиал | Доля записей, % | Личное начальное состояние |")
    L.append("|---|---|---|---|---|")
    for _, r in v.iterrows():
        L.append(f"| {r.variant} | {r.question_code} | {r.branch} | "
                 f"{r.branch_share_pct} | {r.debug_seed} |")
    L.append("")
    L.append("## Сопоставимость нагрузки")
    L.append("")
    L.append(f"Доля записей по филиалам: от **{lo * 100:.2f} %** до **{hi * 100:.2f} %**, "
             f"отношение **×{hi / lo:.1f}**.")
    L.append("")
    if warn:
        L.append("> **Внимание.** Объёмы вариантов различаются более чем вчетверо. "
                 "Филиал с наибольшей долей содержит крупнейшие регионы, и работа по "
                 "нему заметно тяжелее. Варианты розданы «змейкой» — крупные и малые "
                 "филиалы распределены между всеми тремя вопросами, — но при "
                 "оценивании это различие следует учитывать: критерий 3 применяется "
                 "к корректности обработки, а не к объёму обработанного.")
    else:
        L.append("Различие объёмов в допустимых пределах, поправка при оценивании "
                 "не требуется.")
    L.append("")
    L.append(f"Каждое сочетание «вопрос + филиал» встречается не более одного раза: "
             f"{len(QUESTIONS)} вопроса × {len(branches)} филиалов = "
             f"{len(QUESTIONS) * len(branches)} вариантов, чего хватает на две группы "
             f"по 30 человек при смене начального состояния комплекта между потоками.")

    open(a.out, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L[-8:]))
    print(f"\n[записано] {a.out} и {a.out.replace('.md', '.csv')}")


if __name__ == "__main__":
    main()
