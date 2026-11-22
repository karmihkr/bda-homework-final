# -*- coding: utf-8 -*-
"""Запись суточных партиций, аварийные ситуации поставки и манифест."""
import hashlib
import os
import numpy as np
import pandas as pd


def part_dir(root, obj, day):
    p = os.path.join(root, obj, f"dt={np.datetime_as_string(day, unit='D')}")
    os.makedirs(p, exist_ok=True)
    return p


def write_csv(df, path):
    df.to_csv(path, index=False, encoding="utf-8")


def write_jsonl(df, path):
    df.to_json(path, orient="records", lines=True, force_ascii=False,
               date_format="iso")


def corrupt_file(path, rng):
    """Повреждение суточной выгрузки биллинга. Три независимых симптома, каждый
    из которых встречается в практике: часть строк со сдвигом колонок (теряется
    разделитель), фрагмент в иной кодировке и обрыв файла на середине записи."""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    head, body = lines[0], [l for l in lines[1:] if l]

    # сдвиг колонок: в части строк теряется поле, число колонок перестаёт сходиться
    idx = rng.choice(len(body), size=max(1, int(len(body) * 0.015)), replace=False)
    for i in idx:
        parts = body[i].split(",")
        if len(parts) > 4:
            body[i] = ",".join(parts[1:])

    cut = int(len(body) * 0.6)
    head_b = (head + "\n" + "\n".join(body[:cut]) + "\n").encode("utf-8")
    # фрагмент выгрузки в кодировке cp1251: служебная пометка биллинга
    frag = "# ВЫГРУЗКА ПРЕРВАНА, ПОВТОРИТЕ ЗАПРОС\n" + "\n".join(body[cut:cut + 40])
    frag_b = frag.encode("cp1251")
    tail_b = ("\n" + "\n".join(body[cut + 40:])).encode("utf-8")

    out = head_b + frag_b + tail_b
    out = out[: int(len(out) * 0.985)]          # обрыв на середине записи
    with open(path, "wb") as f:
        f.write(out)


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_manifest(root):
    rows = []
    for dirpath, _, files in os.walk(root):
        for fn in sorted(files):
            if fn == "MANIFEST.csv":
                continue
            full = os.path.join(dirpath, fn)
            rows.append(dict(path=os.path.relpath(full, root).replace(os.sep, "/"),
                             bytes=os.path.getsize(full),
                             sha256=sha256(full)))
    df = pd.DataFrame(sorted(rows, key=lambda r: r["path"]))
    df.to_csv(os.path.join(root, "MANIFEST.csv"), index=False)
    return df
