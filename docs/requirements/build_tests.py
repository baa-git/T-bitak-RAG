"""Build the test overview workbook (testler.xlsx) from the test set, run results and reviews.

Usage:
    pytest tests/unit -q --junitxml=artifacts/junit.xml
    python docs/requirements/build_tests.py [--root <repo>] [--junit <xml>] [--out <xlsx>]

Inputs (all read-only):
    eval/testset_v1.jsonl                         104 test questions
    artifacts/runs/<cfg>_<stamp>/per_question.jsonl, metrics.json   latest run per config
    artifacts/judge/<cfg>_claude.json, <cfg>_rest_claude.json        Claude (Opus) review
    junit.xml                                      pytest tests/unit --junitxml output
    tests/unit/**/test_*.py                        test docstrings (AST, no import)

The in-run ``judge`` field (automatic qwen3:4b judge) is never read.
Styling follows docs/requirements/build.py (Arial, navy headers, thin borders).
"""

from __future__ import annotations

import argparse
import ast
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

DEFAULT_ROOT = Path(__file__).resolve().parents[2]

FONT = "Arial"
NAVY = "1F3A5F"
SUB = "D6DCE4"
GREY = "F2F2F2"
LINE = "BFBFBF"
GREEN, AMBER, ORANGE, RED, DARKRED, LGREY = (
    "D9EAD3",
    "FFF2CC",
    "FCE4D6",
    "F4CCCC",
    "EA9999",
    "E7E6E6",
)
thin = Side(style="thin", color=LINE)
BORDER = Border(left=thin, right=thin, top=thin, bottom=thin)

CONFIGS = ["v0", "v0r", "v1", "v2", "v3", "v3b"]
METHODS = {
    "v0": "sabit pencere (512/64), dense arama",
    "v0r": "sabit pencere (512/64), dense arama + reranker",
    "v1": "yapı bazlı parçalama + bağlam başlığı, dense arama",
    "v2": "yapı bazlı parçalama + bağlam başlığı, hibrit arama (BM25 + dense, RRF)",
    "v3": "yapı bazlı parçalama + bağlam başlığı, hibrit arama (BM25 + dense, RRF) + reranker",
    "v3b": "v3 ile aynı, embedding modeli bge-m3",
}


def label(cfg: str) -> str:
    return f"{cfg} – {METHODS[cfg]}"


ANSWERABLE = ["tek_belge", "cok_belge", "surum", "belirsiz"]
UNANSWERABLE = ["cevapsiz", "kapsam_disi"]
TUR = {
    "tek_belge": "Tek belge",
    "cok_belge": "Çok belge",
    "surum": "Sürüm",
    "belirsiz": "Belirsiz",
    "cevapsiz": "Cevapsız",
    "kapsam_disi": "Kapsam dışı",
}
TUR_DEF = {
    "tek_belge": "Cevap tek belgede (oran, süre, tutar, koşul)",
    "cok_belge": "İki belgenin birleşimi gerekir",
    "surum": "Eski ve yeni değer var; beklenen en güncel kaynak",
    "belirsiz": "Program belirtilmemiş; cevap programlara ayrılmalı",
    "cevapsiz": "Konu kapsamda ama cevabı corpus'ta yok",
    "kapsam_disi": "TÜBİTAK/Ar-Ge ile ilgisiz",
}
ZORLUK = {"kolay": "Kolay", "orta": "Orta", "zor": "Zor"}
EXPECT = {"cevapsiz": "Ret (bilgi bulunamadı)", "kapsam_disi": "Ret (kapsam dışı)"}

STATUS = {
    ("cevaplandi", None): "Cevaplandı",
    ("kapsam_disi", "kapsam_disi"): "Ret: kapsam dışı",
    ("bilgi_bulunamadi", "cevaplanamaz"): "Ret: bilgi yok (model)",
    ("bilgi_bulunamadi", "dayanaksiz_sayi"): "Ret: dayanaksız sayı",
    ("bilgi_bulunamadi", "esik_alti"): "Ret: skor eşik altı",
    ("bilgi_bulunamadi", "sonuc_yok"): "Ret: arama sonucu yok",
    ("atif_reddi", "atifsiz_cevap"): "Ret: atıfsız cevap",
    ("atif_reddi", "gecersiz_atif"): "Ret: geçersiz atıf",
}
PLACE = {"retrieval": "arama", "uretim": "üretim", "belirsiz": "belirsiz"}
KARAR = {"dogru_ret": "Doğru ret", "uydurma": "Uydurma"}


def font(**kw) -> Font:
    return Font(name=FONT, size=kw.pop("size", 10), **kw)


def style(cell, *, bold=False, wrap=False, center=False, fill=None, fmt=None, color=None) -> None:
    cell.font = font(bold=bold, color=color) if color else font(bold=bold)
    cell.border = BORDER
    cell.alignment = Alignment(
        vertical="top" if not center else "center",
        horizontal="center" if center else "left",
        wrap_text=wrap,
    )
    if fill:
        cell.fill = PatternFill("solid", fgColor=fill)
    if fmt:
        cell.number_format = fmt


def header_row(ws, row: int, labels: list[str], start: int = 1, fill: str = NAVY) -> None:
    for col, text in enumerate(labels, start=start):
        cell = ws.cell(row=row, column=col, value=text)
        cell.font = font(bold=True, color="FFFFFF" if fill == NAVY else "000000")
        cell.fill = PatternFill("solid", fgColor=fill)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER


def widths(ws, values: list[float]) -> None:
    for i, w in enumerate(values, start=1):
        ws.column_dimensions[get_column_letter(i)].width = w


def section(ws, row: int, text: str) -> None:
    ws.cell(row=row, column=1, value=text).font = font(size=12, bold=True, color=NAVY)


# ----------------------------------------------------------------------------- data


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def latest_run(root: Path, cfg: str) -> Path:
    sample = root / "docs" / "sample-reports" / f"{cfg}.metrics.json"
    if sample.exists():
        run_id = json.loads(sample.read_text(encoding="utf-8"))["run_id"]
        path = root / "artifacts" / "runs" / run_id
        if path.exists():
            return path
    runs = sorted((root / "artifacts" / "runs").glob(f"{cfg}_2*"))
    if not runs:
        raise FileNotFoundError(f"no run for {cfg}")
    return runs[-1]


def load_data(root: Path) -> dict:
    questions = read_jsonl(root / "eval" / "testset_v1.jsonl")
    judge = root / "artifacts" / "judge"
    runs: dict[str, dict] = {}
    for cfg in CONFIGS:
        run_dir = latest_run(root, cfg)
        rows = {r["id"]: r for r in read_jsonl(run_dir / "per_question.jsonl")}
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        answers = {
            str(x["id"]): x
            for x in json.loads((judge / f"{cfg}_claude.json").read_text(encoding="utf-8"))
        }
        others = {
            str(x["id"]): x
            for x in json.loads((judge / f"{cfg}_rest_claude.json").read_text(encoding="utf-8"))
        }
        # same staleness check as reporting/review.py: reviewed answers must be this run's answers
        norm = lambda s: " ".join(str(s or "").split())  # noqa: E731
        for suffix, field in (("_items.json", "cevap"), ("_rest.json", "durum")):
            snap = judge / f"{cfg}{suffix}"
            if snap.exists():
                for item in json.loads(snap.read_text(encoding="utf-8")):
                    row = rows.get(str(item["id"]))
                    if row is None or norm(item.get(field)) != norm(row["answer"].get(field)):
                        raise ValueError(
                            f"{cfg}: review snapshot does not match run {run_dir.name} ({item['id']})"
                        )
        runs[cfg] = {
            "dir": run_dir,
            "rows": rows,
            "metrics": metrics,
            "answers": answers,
            "others": others,
        }
    return {"questions": questions, "runs": runs}


def is_answered_answerable(row: dict) -> bool:
    return row["tur"] in ANSWERABLE and row["answer"]["durum"] == "cevaplandi"


def status_label(row: dict) -> str:
    ans = row["answer"]
    key = (ans["durum"], ans.get("red_nedeni"))
    return STATUS.get(
        key,
        "Cevaplandı"
        if ans["durum"] == "cevaplandi"
        else f"Ret: {ans.get('red_nedeni') or ans['durum']}",
    )


def verdict(run: dict, row: dict) -> tuple[str, str, str]:
    """(Claude kararı, Sadık, not) for one row of one run."""
    qid = row["id"]
    if is_answered_answerable(row):
        v = run["answers"][qid]
        karar = "Doğru cevap" if v.get("dogru") is True else "Yanlış cevap"
        sadik = {True: "Evet", False: "Hayır"}.get(v.get("sadik"), "Kararsız")
        return karar, sadik, str(v.get("not") or "")
    v = run["others"][qid]
    if v["karar"] == "gereksiz_ret":
        karar = f"Gereksiz ret ({PLACE.get(v.get('hata_yeri'), v.get('hata_yeri') or '?')})"
    else:
        karar = KARAR.get(v["karar"], v["karar"])
    return karar, "", str(v.get("not") or "")


def is_scope_rejected(row: dict) -> bool:
    return row["tur"] in ANSWERABLE and row["answer"].get("red_nedeni") == "kapsam_disi"


# ----------------------------------------------------------------------------- sheets


def build_testset(wb: Workbook, questions: list[dict]) -> int:
    ws = wb.create_sheet("Test Seti")
    cols = [
        ("ID", 8),
        ("Tür", 12),
        ("Zorluk", 9),
        ("Soru", 55),
        ("Beklenen cevap", 55),
        ("Kanıt alıntıları", 62),
        ("Beklenen belgeler", 30),
        ("Beklenen sayılar", 14),
        ("Program kodu", 12),
        ("Cevaplanmalı mı", 22),
        ("İnsan incelemesi", 15),
        ("Yazar notu", 40),
    ]
    header_row(ws, 1, [c for c, _ in cols])
    widths(ws, [w for _, w in cols])
    ws.row_dimensions[1].height = 30
    for r, q in enumerate(questions, start=2):
        quotes = "\n".join(f"[{k['doc_id']}] {k['alinti']}" for k in q.get("kanit_alintilari", []))
        should = "Evet" if q["tur"] in ANSWERABLE else f"Hayır – {EXPECT[q['tur']].lower()}"
        values = [
            q["id"],
            TUR[q["tur"]],
            ZORLUK.get(q["zorluk"], q["zorluk"]),
            q["soru"],
            q["beklenen_cevap"],
            quotes,
            "\n".join(q.get("beklenen_belgeler", [])),
            ", ".join(q.get("beklenen_sayilar", [])),
            ", ".join(q.get("program_kodu", [])),
            should,
            "Doğrulandı" if q.get("dogrulandi") else "Doğrulanmadı",
            q.get("review_notes", ""),
        ]
        for c, v in enumerate(values, start=1):
            style(
                ws.cell(row=r, column=c, value=v),
                bold=(c == 1),
                wrap=c in (4, 5, 6, 7, 10, 12),
                center=c in (2, 3, 8, 9, 11),
            )
    last = len(questions) + 1
    ws.conditional_formatting.add(
        f"J2:J{last}",
        FormulaRule(formula=['LEFT($J2,5)="Hayır"'], fill=PatternFill("solid", fgColor=LGREY)),
    )
    ws.freeze_panes = "B2"
    ws.auto_filter.ref = f"A1:L{last}"
    ws.sheet_view.zoomScale = 90
    return last


def group_header(ws, cfg_start: int, per: int, sub: list[str], fixed: list[str]) -> None:
    header_row(ws, 1, fixed)
    header_row(ws, 2, fixed)
    for c in range(1, len(fixed) + 1):
        ws.merge_cells(start_row=1, start_column=c, end_row=2, end_column=c)
    for i, cfg in enumerate(CONFIGS):
        col = cfg_start + i * per
        header_row(ws, 1, [label(cfg)] + [""] * (per - 1), start=col)
        ws.merge_cells(start_row=1, start_column=col, end_row=1, end_column=col + per - 1)
        header_row(ws, 2, sub, start=col, fill=SUB)
    ws.row_dimensions[1].height = 48
    ws.row_dimensions[2].height = 30


def build_retrieval(wb: Workbook, data: dict) -> dict:
    ws = wb.create_sheet("Arama Sonuçları")
    fixed = ["ID", "Tür", "Soru", "Alıntı sayısı"]
    sub = ["İlk isabet sırası", "R@5", "R@10", "MRR katkısı (1/sıra)"]
    per, start = len(sub), len(fixed) + 1
    group_header(ws, start, per, sub, fixed)
    widths(ws, [8, 11, 48, 9] + [10, 7, 7, 10] * len(CONFIGS))
    first = 3
    for r, q in enumerate(data["questions"], start=first):
        n_quotes = len(q.get("kanit_alintilari", []))
        for c, v in enumerate([q["id"], TUR[q["tur"]], q["soru"], n_quotes or "—"], start=1):
            style(
                ws.cell(row=r, column=c, value=v), bold=(c == 1), wrap=(c == 3), center=c in (2, 4)
            )
        for i, cfg in enumerate(CONFIGS):
            row = data["runs"][cfg]["rows"][q["id"]]
            col = start + i * per
            if q["tur"] not in ANSWERABLE:
                vals = ["ölçülmez", None, None, None]
            elif is_scope_rejected(row):
                vals = ["kapsam reddi", None, None, None]
            else:
                hits = row["evidence_hits"]
                rank = row["first_hit_rank"]
                vals = [
                    rank if rank else "—",
                    sum(1 for h in hits if h and h <= 5) / len(hits),
                    sum(1 for h in hits if h and h <= 10) / len(hits),
                    1 / rank if rank else 0,
                ]
            for k, v in enumerate(vals):
                style(
                    ws.cell(row=r, column=col + k, value=v),
                    center=True,
                    fmt=("0.00" if k in (1, 2) else "0.000" if k == 3 else None),
                )
    last = first + len(data["questions"]) - 1
    avg = last + 2
    ws.cell(row=avg, column=1, value="Ortalama (ölçülen sorular)")
    ws.merge_cells(start_row=avg, start_column=1, end_row=avg, end_column=4)
    cnt = avg + 1
    ws.cell(row=cnt, column=1, value="Ölçülen soru sayısı")
    ws.merge_cells(start_row=cnt, start_column=1, end_row=cnt, end_column=4)
    for rr in (avg, cnt):
        for c in range(1, 5):
            style(ws.cell(row=rr, column=c), bold=True, fill=GREY)
    refs = {}
    for i, cfg in enumerate(CONFIGS):
        col = start + i * per
        L = [get_column_letter(col + k) for k in range(per)]
        style(ws.cell(row=avg, column=col, value="—"), center=True, fill=GREY)
        for k in (1, 2, 3):
            style(
                ws.cell(row=avg, column=col + k, value=f"=AVERAGE({L[k]}{first}:{L[k]}{last})"),
                bold=True,
                center=True,
                fill=GREY,
                fmt="0.000",
            )
        style(
            ws.cell(row=cnt, column=col, value=f"=COUNT({L[1]}{first}:{L[1]}{last})"),
            bold=True,
            center=True,
            fill=GREY,
        )
        for k in (1, 2, 3):
            style(ws.cell(row=cnt, column=col + k), fill=GREY)
        refs[cfg] = {
            "rank": L[0],
            "r5": L[1],
            "r10": L[2],
            "rr": L[3],
            "avg_row": avg,
            "cnt_row": cnt,
        }
        rng = f"{L[0]}{first}:{L[0]}{last}"
        tl = f"{L[0]}{first}"
        for formula, color in (
            (f"AND(ISNUMBER({tl}),{tl}<=5)", GREEN),
            (f"AND(ISNUMBER({tl}),{tl}<=10)", AMBER),
            (f"ISNUMBER({tl})", ORANGE),
            (f'{tl}="—"', RED),
            (f'{tl}="kapsam reddi"', RED),
            (f'{tl}="ölçülmez"', LGREY),
        ):
            ws.conditional_formatting.add(
                rng,
                FormulaRule(
                    formula=[formula], fill=PatternFill("solid", fgColor=color), stopIfTrue=True
                ),
            )
        r5 = f"{L[1]}{first}:{L[1]}{last}"
        ws.conditional_formatting.add(
            r5,
            FormulaRule(
                formula=[f"AND(ISNUMBER({L[1]}{first}),{L[1]}{first}=1)"],
                fill=PatternFill("solid", fgColor=GREEN),
            ),
        )
        ws.conditional_formatting.add(
            r5,
            FormulaRule(
                formula=[f"AND(ISNUMBER({L[1]}{first}),{L[1]}{first}>0,{L[1]}{first}<1)"],
                fill=PatternFill("solid", fgColor=AMBER),
            ),
        )
    legend = cnt + 2
    notes = [
        "İlk isabet sırası: kanıt alıntılarından herhangi birini tam içeren ilk parçanın 30 aday içindeki sırası. "
        'Yeşil 1–5 (dil modeline giden parçalar), sarı 6–10, turuncu 11–30, kırmızı "—" = 30 adayda hiç yok.',
        "R@5 / R@10: o sorunun alıntılarından ilk 5 / 10 parçada bulunanların oranı (2 alıntıdan 1'i bulunduysa 0,50).",
        '"kapsam reddi": kapsam kontrolü soruyu yanlışlıkla durdurdu, arama yapılmadı; ortalamaya katılmaz (cevaplanabilir 79 sorudan 77\'si ölçülür).',
        '"ölçülmez": cevapsız ve kapsam dışı soruların kanıt alıntısı yoktur.',
    ]
    for k, note in enumerate(notes, start=legend):
        ws.cell(row=k, column=1, value=note).font = font(color="404040", size=9)
    ws.freeze_panes = "E3"
    ws.auto_filter.ref = f"A2:{get_column_letter(start + per * len(CONFIGS) - 1)}{last}"
    ws.sheet_view.zoomScale = 90
    return {"first": first, "last": last, "cols": refs}


def build_answers(wb: Workbook, data: dict) -> dict:
    ws = wb.create_sheet("Cevap Sonuçları")
    fixed = ["ID", "Tür", "Soru", "Beklenen davranış"]
    sub = ["Sistem durumu", "Claude kararı", "Kaynağa sadık", "Beklenen sayılar cevapta"]
    per, start = len(sub), len(fixed) + 1
    group_header(ws, start, per, sub, fixed)
    note_cols = ["v3", "v3b"]
    note_start = start + per * len(CONFIGS)
    for j, cfg in enumerate(note_cols):
        c = note_start + j
        header_row(ws, 1, [f"Claude notu – {cfg}"], start=c)
        header_row(ws, 2, [""], start=c)
        ws.merge_cells(start_row=1, start_column=c, end_row=2, end_column=c)
    widths(ws, [8, 11, 48, 16] + [15, 15, 9, 10] * len(CONFIGS) + [42, 42])
    first = 3
    for r, q in enumerate(data["questions"], start=first):
        exp = "Cevap" if q["tur"] in ANSWERABLE else EXPECT[q["tur"]]
        for c, v in enumerate([q["id"], TUR[q["tur"]], q["soru"], exp], start=1):
            style(
                ws.cell(row=r, column=c, value=v), bold=(c == 1), wrap=c in (3, 4), center=(c == 2)
            )
        for i, cfg in enumerate(CONFIGS):
            run = data["runs"][cfg]
            row = run["rows"][q["id"]]
            karar, sadik, _ = verdict(run, row)
            numbers = row["rule_check"].get("sayilar_bulundu")
            nums = (
                ""
                if q["tur"] not in ANSWERABLE or not q.get("beklenen_sayilar")
                else ("Evet" if numbers else "Hayır")
            )
            col = start + i * per
            for k, v in enumerate([status_label(row), karar, sadik, nums]):
                style(ws.cell(row=r, column=col + k, value=v), center=True, wrap=k < 2)
        for j, cfg in enumerate(note_cols):
            run = data["runs"][cfg]
            _, _, note = verdict(run, run["rows"][q["id"]])
            style(ws.cell(row=r, column=note_start + j, value=note), wrap=True)
    last = first + len(data["questions"]) - 1
    refs = {}
    for i, cfg in enumerate(CONFIGS):
        col = start + i * per
        L = [get_column_letter(col + k) for k in range(per)]
        refs[cfg] = {"status": L[0], "karar": L[1], "sadik": L[2], "nums": L[3]}
        rng = f"{L[1]}{first}:{L[1]}{last}"
        tl = f"{L[1]}{first}"
        for formula, color in (
            (f'{tl}="Doğru cevap"', GREEN),
            (f'{tl}="Doğru ret"', GREEN),
            (f'{tl}="Yanlış cevap"', RED),
            (f'{tl}="Uydurma"', DARKRED),
            (f'LEFT({tl},12)="Gereksiz ret"', AMBER),
        ):
            ws.conditional_formatting.add(
                rng, FormulaRule(formula=[formula], fill=PatternFill("solid", fgColor=color))
            )
        st = f"{L[0]}{first}"
        ws.conditional_formatting.add(
            f"{L[0]}{first}:{L[0]}{last}",
            FormulaRule(formula=[f'LEFT({st},4)="Ret:"'], fill=PatternFill("solid", fgColor=LGREY)),
        )
        sd = f"{L[2]}{first}"
        ws.conditional_formatting.add(
            f"{L[2]}{first}:{L[2]}{last}",
            FormulaRule(formula=[f'{sd}="Hayır"'], fill=PatternFill("solid", fgColor=ORANGE)),
        )
        nm = f"{L[3]}{first}"
        ws.conditional_formatting.add(
            f"{L[3]}{first}:{L[3]}{last}",
            FormulaRule(formula=[f'{nm}="Evet"'], fill=PatternFill("solid", fgColor=GREEN)),
        )
        ws.conditional_formatting.add(
            f"{L[3]}{first}:{L[3]}{last}",
            FormulaRule(formula=[f'{nm}="Hayır"'], fill=PatternFill("solid", fgColor=ORANGE)),
        )
    legend = last + 2
    notes = [
        "Claude kararı: 624 cevabın Claude (Opus) ile tek tek incelenmesi. Cevaplanabilir soru cevaplandıysa Doğru cevap / Yanlış cevap ve kaynağa sadık mı; "
        "cevaplanabilir soru reddedildiyse Gereksiz ret (hatanın yeri: arama = kanıt ilk 5'te yok, üretim = kanıt ilk 5'te vardı, belirsiz); "
        'cevapsız / kapsam dışı soruda Doğru ret ya da Uydurma (cevap metninde "kaynakta yok" diyen cevap da doğru ret sayılır).',
        "Sistem durumu: koşunun kaydettiği durum ve ret nedeni (kapsam kontrolü, skor eşiği, sayı denetimi, atıf denetimi ya da modelin kendisi).",
        "Beklenen sayılar cevapta: kural tabanlı kontrol; beklenen her sayı cevapta tam sayı olarak geçiyor mu (beklenen sayısı olan 69 soru; ret = Hayır).",
        'Kaynağa sadık "Kararsız": inceleme sadakat konusunda karar vermedi; sadakat oranına katılmaz.',
    ]
    for k, note in enumerate(notes, start=legend):
        ws.cell(row=k, column=1, value=note).font = font(color="404040", size=9)
    ws.freeze_panes = "E3"
    ws.auto_filter.ref = f"A2:{get_column_letter(note_start + len(note_cols) - 1)}{last}"
    ws.sheet_view.zoomScale = 90
    return {"first": first, "last": last, "cols": refs}


# --- curated failure cases (docs/RAG-GELISTIRME-REHBERI.md §6), checked against artifacts below
FAILURES = [
    (
        "q001",
        ["v0"],
        'Getirilen pencere yanlış madde etiketi taşıyor: %75 cümlesi 28. sıradaki "Madde 5" etiketli pencerede (Madde 5–7 birlikte).',
        "1507 Uygulama Esasları Madde 7: her dönem için en fazla %75.",
        "Arama",
        "Sabit pencere başladığı yerin konumunu taşır; üç madde tek embedding'de ortalanır.",
        "Çözüldü",
        "v1+ yapı bazlı parçalama: her parça kendi maddesini taşır.",
    ),
    (
        "q010",
        ["v0", "v0r"],
        'v0: kanıt 29. sırada; cevap "en fazla 12 ay" (Aşama 1 çağrısından).',
        "Ek süreler dahil en fazla 18 ay (1512 Esasları).",
        "Arama",
        'Dense arama "1512" ve "Aşama" geçen her parçayı yakın buldu.',
        "Çözüldü",
        "v0r: reranker kanıtı 2. sıraya çıkardı, cevap 18 ay.",
    ),
    (
        "q033",
        ["v1", "v2", "v3"],
        'v1: ilk 5\'te 4005/4004/4007 çağrılarının benzer bölümleri, kanıt 16. sırada; cevap "en az 1 gün".',
        "En az 4 gün (4008 çağrı metni).",
        "Arama",
        "Bilim ve toplum çağrıları neredeyse aynı şablonla yazılmış.",
        "Çözüldü",
        "v2: BM25 kanıtı 4. sıraya taşıdı; v3'te 1. sırada.",
    ),
    (
        "q059",
        ["v0", "v1"],
        'v0: 2024 tarihli üst limit tablosu geldi, güncel 2026-3 çağrısı 30 adayda yok; cevap "5.000.000 TL".',
        "10.000.000 TL (güncel 1707 çağrısı).",
        "Arama",
        "Eski sürüm (tablo) öne çıktı, güncel belge havuza girmedi.",
        "Çözüldü",
        "v1'den itibaren güncel çağrı ilk 5'te.",
    ),
    (
        "q053",
        ["v3", "v2"],
        "v3: süre kanıtı 3., KOBİ oranı kanıtı 6. sırada; model \"kaynaklarda yok\" dedi. v2: %100'ü (kamu oranı) KOBİ'ye mal etti.",
        "KOBİ için %75'e kadar; süre en fazla 48 ay.",
        "Arama",
        "Dil modeline yalnız ilk 5 parça gider; iki parçalı soruda bir kanıt dışarıda kaldı.",
        "Açık",
        "Alt sorulara bölme ya da çok parçalı soruda daha fazla parça.",
    ),
    (
        "q036",
        ["v3"],
        "v3: İtiraz Yönetmeliği süre maddesi 3. sırada; 4007/4008 çağrıları ilk 5'i doldurdu, model 4007 çağrısına atıf yaptı (sayı doğru; Claude: doğru cevap).",
        "15 gün; genel kural İtiraz Yönetmeliği'nde (tebliğden itibaren).",
        "Arama",
        "Tekrar eden içerik ilk 5'i dolduruyor; kaynak doğru ama dar.",
        "Açık",
        "Belge başına parça sınırı / çeşitlendirme (MMR).",
    ),
    (
        "q069",
        ["v3"],
        'v3: rastgele programların "destek süresi" maddeleri geldi; beklenen kanıtların hiçbiri 30 adayda yok.',
        "Programa göre değişir: 1001'de 36 ay, 1005'te 18 ay, 1002-A'da 12 ay.",
        "Arama",
        "Belirsiz soru tek bir sorgu olarak arandı.",
        "Açık",
        "Belirsizliği fark edip programlara göre alt sorgu ya da kullanıcıya sorma.",
    ),
    (
        "q052",
        ["v3"],
        'v3: 3. ve 4. kaynağın belge adı "Sıkça sorulan sorular için tıklayınız."',
        "Belge adı program/SSS başlığı olmalı.",
        "Arama (veri/başlık)",
        'Başlık bağlantı metninden alınmış; 7.017 parçanın 204\'ünde "tıklayınız".',
        "Açık",
        "Manifestte başlıklar elle düzeltilmeli.",
    ),
    (
        "q083",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        'Altı koşunun altısında tutar uyduruldu (v3: "ayda 7.000 TL", v0/v0r/v3b: "32.500 TL", v2: "60.000 ABD doları").',
        "Bilgi bulunamadı (2211 aylık burs tutarı corpus'ta yok).",
        "Denetim (eşik + sayı denetimi)",
        "Tutarlar kaynakta proje bursiyeri limiti olarak geçiyor; sayı denetimi yalnız varlığı kontrol ediyor, eşik (−2) kapıyı açtı.",
        "Açık",
        "Anlam düzeyinde denetim; eşiğin doğrulama kümesiyle ayarlanması.",
    ),
    (
        "q091",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        'Altı koşunun altısında uydurma: beşinde "43.500 TL" (kaynakta doktora sonrası araştırmacı bursu), v0r\'de "25.000 TL".',
        "Bilgi bulunamadı (1001 PTİ üst sınırı corpus'ta yok).",
        "Denetim (sayı denetimi)",
        "Sayı kaynakta geçiyor ama başka anlamda.",
        "Açık",
        "Anlam düzeyinde denetim.",
    ),
    (
        "q011",
        ["v3"],
        'v3: "%3 hisse karşılığında 900.000 TL" (eski SSS değeri).',
        "%3 pay karşılığında 1.350.000 TL; kanıt 3. sıradaydı.",
        "Üretim",
        "Arama doğruydu, sürüm tarihleri prompt'taydı; model eski değeri seçti.",
        "Açık",
        "Değerler çatışınca en yeni tarihli kaynağı öne alan kural.",
    ),
    (
        "q066",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        'Altı koşunun hepsinde 900.000 TL; sorudaki "eski tabloda 900 bin" varsayımı onaylandı.',
        "Güncel tutar 1.350.000 TL; 900.000 TL önceki dönemler.",
        "Üretim",
        "Kullanıcının verdiği sayı kaynakta da geçtiği için model onu doğruladı.",
        "Açık",
        "",
    ),
    (
        "q079",
        ["v0r", "v1", "v2", "v3", "v3b"],
        "v0r ve v1: güncel 3.500.000 TL limiti 2027'ye mal edildi; v2, v3, v3b doğru reddetti.",
        "Bilgi bulunamadı (2027 limiti yok).",
        "Üretim",
        "Sorudaki yıl varsayımı mevcut değere bağlandı.",
        "Açık",
        "v2–v3b'de doğru ret.",
    ),
    (
        "q052",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        'v3: "yılda en fazla 5 başvuru … Ulusal patent başvuruları hâlâ desteklenmektedir." Altı koşunun altısında.',
        "Yılda 5 doğru; ama 2019'dan beri ulusal patent desteği başvuruya kapalı.",
        "Üretim",
        'Model "hâlâ … mu?" kalıbına olumlu cevap verdi; 4. kaynak tersini söylüyordu.',
        "Açık",
        "Güçlü model ya da evet/hayır cevaplarını kaynağa karşı denetleme.",
    ),
    (
        "q028",
        ["v3"],
        'v3: "Evet, aynı yıl içinde ikinci kez yararlanabilirsiniz" (4001 kitapçığından).',
        "Hayır; o yıl 2224-A'dan yararlanmamış olmak başvuru koşulu.",
        "Üretim",
        "1. ve 3. sıradaki 4001 kuralı 2224-A'ya uygulandı.",
        "Açık",
        "",
    ),
    (
        "q003",
        ["v3"],
        'v3: lisans mezunu için başka programın "10 katı / 6 katı" değerleri yazıldı.',
        "Doktoralı 12 kat, lisans 8 kat (kanıt 1. sıradaydı).",
        "Üretim",
        "Programlar karıştırıldı.",
        "Açık",
        "",
    ),
    (
        "q070",
        ["v1", "v2", "v3", "v3b"],
        "\"Destek oranı %75'tir [1].\" (1. sıradaki 1507 SSS'i genel kural gibi verildi)",
        "Programa göre değişir: 1501 büyük %60 / KOBİ %75; 1507 her dönem en fazla %75.",
        "Üretim",
        "Belirsiz 10 sorunun hiçbirinde, hiçbir koşuda cevap programlara ayrılmadı.",
        "Açık",
        "Program sorulmalı ya da cevap programlara göre listelenmeli.",
    ),
    (
        "q039",
        ["v0r", "v1", "v2", "v3", "v3b"],
        "Kanıt 1. sırada (v1–v3b) olduğu hâlde cevap sayı denetimiyle reddedildi (v0r–v3b).",
        '%95 ("yüzde doksan beşi").',
        "Denetim (sayı denetimi)",
        'Desen ekli sayı sözcüğünü ("beşi") tanımıyordu: kaynaktan 90, cevaptan 95.',
        "Kodda çözüldü",
        "Desen artık ekleri kabul ediyor; altı koşu düzeltmeden önce yapıldı.",
    ),
    (
        "q032",
        ["v1", "v2", "v3"],
        "v1: formül 1. sıradaydı, model doğru hesapladı (22.000 TL) ama sayı ilk 5'te geçmediği için reddedildi.",
        "22.000 TL; v2 ve v3'te örnek tutar ilk 5'e girdi ve cevap geçti.",
        "Denetim (sayı denetimi)",
        "Denetim hesaplanmış sayıyı tanımıyor.",
        "Açık",
        "",
    ),
    (
        "q027",
        ["v3", "v3b"],
        "v3 ve v3b: kanıt 1. sıradaydı, model atıf koymadı, atıf denetimi cevabı reddetti.",
        "Toplam 24 aya kadar burs.",
        "Denetim (atıf denetimi)",
        "Atıfsız cevap reddedilir.",
        "Açık",
        "",
    ),
    (
        "q012",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        'Altı koşunun hepsinde "kapsam dışı"; arama hiç yapılmadı.',
        "Her boyuttan (10 üzerinden) en az 7 puan.",
        "Denetim (kapsam kontrolü)",
        'Sorudaki terimler sözlükte yok ("BiGG"); LLM sınıflandırıcı da kapsam dışı dedi.',
        "Açık",
        "Sözlüğe program adları; emin olunamazsa aramaya bırakmak.",
    ),
    (
        "q077",
        ["v0", "v0r", "v1", "v2", "v3", "v3b"],
        '"Değerlendirmede geçme puanı kaç?" kapsam dışı sayıldı.',
        "Soru bağlamsız; doğru davranış netleştirme sormaktı.",
        "Denetim (kapsam kontrolü)",
        "Bağlamsız soru kapsam dışı sınıflandırıldı.",
        "Açık",
        "",
    ),
]


def evidence_text(row: dict) -> str:
    if row["tur"] not in ANSWERABLE:
        return "alıntı yok"
    if is_scope_rejected(row):
        return "arama yok"
    hits = [str(h) if h else "—" for h in row["evidence_hits"]]
    return "kanıt sırası " + "/".join(hits)


def build_failures(wb: Workbook, data: dict) -> int:
    ws = wb.create_sheet("Hata Örnekleri")
    cols = [
        ("No", 5),
        ("ID", 7),
        ("Soru", 44),
        ("Yapılandırma", 30),
        ("Sistem ne yaptı", 50),
        ("Doğrusu", 40),
        ("Hata türü", 16),
        ("Neden", 40),
        ("Durum", 13),
        ("Çözüm / öneri", 32),
        ("Artefakt kontrolü (koşu dosyası + Claude kararı)", 60),
    ]
    header_row(ws, 1, [c for c, _ in cols])
    widths(ws, [w for _, w in cols])
    ws.row_dimensions[1].height = 30
    qmap = {q["id"]: q for q in data["questions"]}
    for r, (qid, cfgs, did, right, kind, why, state, fix) in enumerate(FAILURES, start=2):
        checks = []
        for cfg in cfgs:
            run = data["runs"][cfg]
            row = run["rows"][qid]
            karar, _, _ = verdict(run, row)
            checks.append(f"{cfg}: {evidence_text(row)}; {status_label(row)}; Claude: {karar}")
        cfg_text = "\n".join(label(c) for c in cfgs[:2]) + (
            f"\n(+{len(cfgs) - 2} koşu daha)" if len(cfgs) > 2 else ""
        )
        if len(cfgs) == 6:
            cfg_text = "Altı koşunun hepsi (v0 … v3b)"
        values = [
            r - 1,
            qid,
            qmap[qid]["soru"],
            cfg_text,
            did,
            right,
            kind,
            why,
            state,
            fix,
            "\n".join(checks),
        ]
        for c, v in enumerate(values, start=1):
            style(
                ws.cell(row=r, column=c, value=v), bold=(c == 2), wrap=c >= 3, center=c in (1, 2, 9)
            )
    last = len(FAILURES) + 1
    for text, color in (("Çözüldü", GREEN), ("Kodda çözüldü", AMBER), ("Açık", RED)):
        ws.conditional_formatting.add(
            f"I2:I{last}",
            FormulaRule(formula=[f'$I2="{text}"'], fill=PatternFill("solid", fgColor=color)),
        )
    ws.cell(
        row=last + 2,
        column=1,
        value=(
            "Kaynak: docs/RAG-GELISTIRME-REHBERI.md §6. Son sütun koşu dosyalarından (per_question.jsonl) ve Claude incelemesinden "
            "otomatik üretilir: kanıt sırası = her kanıt alıntısının 30 aday içindeki sırası (— = yok)."
        ),
    ).font = font(color="404040", size=9)
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:K{last}"
    ws.sheet_view.zoomScale = 90
    return last


# --- unit tests


RESULT_PASS, RESULT_XFAIL, RESULT_SKIP, RESULT_FAIL, RESULT_DESEL = (
    "Geçti",
    "Beklenen hata (xfail)",
    "Atlandı",
    "Başarısız",
    "Seçilmedi (integration)",
)


def docstrings(test_file: Path) -> dict[str, str]:
    try:
        tree = ast.parse(test_file.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return {}
    out: dict[str, str] = {}

    def visit(node, prefix=""):
        for child in getattr(node, "body", []):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(child)
                if doc:
                    out[prefix + child.name] = " ".join(doc.strip().splitlines()[0].split())
            elif isinstance(child, ast.ClassDef):
                visit(child, prefix + child.name + ".")

    visit(tree)
    return out


def load_unit_tests(root: Path, junit: Path, deselected: list[str]) -> list[dict]:
    tests: list[dict] = []
    doc_cache: dict[Path, dict[str, str]] = {}
    for case in ET.parse(junit).getroot().iter("testcase"):
        parts = case.get("classname", "").split(".")
        name = case.get("name", "")
        # longest prefix that is a file
        file_parts, cls = parts, []
        while file_parts and not (root.joinpath(*file_parts[:-1], file_parts[-1] + ".py")).exists():
            cls.insert(0, file_parts[-1])
            file_parts = file_parts[:-1]
        path = root.joinpath(*file_parts[:-1], file_parts[-1] + ".py")
        if case.find("failure") is not None or case.find("error") is not None:
            result = RESULT_FAIL
        elif (sk := case.find("skipped")) is not None:
            result = RESULT_XFAIL if "xfail" in (sk.get("type") or "") else RESULT_SKIP
        else:
            result = RESULT_PASS
        tests.append(
            {
                "path": path,
                "cls": ".".join(cls),
                "name": name,
                "result": result,
                "time": float(case.get("time") or 0),
                "reason": (sk.get("message") if result == RESULT_XFAIL else "") or "",
            }
        )
    for nodeid in deselected:
        file_part, _, name = nodeid.partition("::")
        tests.append(
            {
                "path": root / file_part,
                "cls": "",
                "name": name,
                "result": RESULT_DESEL,
                "time": None,
                "reason": "integration işaretli: ayrıştırılmış corpus gerekir (-m integration ile koşulur)",
            }
        )
    for t in tests:
        if t["path"] not in doc_cache:
            doc_cache[t["path"]] = docstrings(t["path"])
        key = (t["cls"] + "." if t["cls"] else "") + t["name"].split("[")[0]
        t["doc"] = doc_cache[t["path"]].get(key, "")
        rel = t["path"].relative_to(root).as_posix()
        t["module"] = rel.rsplit("/", 1)[0]
        t["file"] = rel.rsplit("/", 1)[1]
    return tests


def build_unit_tests(wb: Workbook, tests: list[dict]) -> int:
    ws = wb.create_sheet("Birim Testleri")
    cols = [
        ("Modül (paket yolu)", 34),
        ("Dosya", 34),
        ("Sınıf", 16),
        ("Test adı", 62),
        ("Sonuç", 22),
        ("Süre (s)", 9),
        ("Açıklama (docstring) / xfail nedeni", 70),
    ]
    header_row(ws, 1, [c for c, _ in cols])
    widths(ws, [w for _, w in cols])
    ws.row_dimensions[1].height = 28
    for r, t in enumerate(tests, start=2):
        desc = t["doc"]
        if t["reason"]:
            desc = (desc + " — " if desc else "") + t["reason"]
        values = [t["module"], t["file"], t["cls"], t["name"], t["result"], t["time"], desc]
        for c, v in enumerate(values, start=1):
            style(
                ws.cell(row=r, column=c, value=v),
                wrap=c in (4, 7),
                center=c in (5, 6),
                fmt="0.000" if c == 6 else None,
            )
    last = len(tests) + 1
    for text, color in (
        (RESULT_PASS, GREEN),
        (RESULT_XFAIL, AMBER),
        (RESULT_SKIP, LGREY),
        (RESULT_FAIL, RED),
        (RESULT_DESEL, LGREY),
    ):
        ws.conditional_formatting.add(
            f"E2:E{last}",
            FormulaRule(formula=[f'$E2="{text}"'], fill=PatternFill("solid", fgColor=color)),
        )
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = f"A1:G{last}"
    ws.sheet_view.zoomScale = 90
    return last


def build_unit_summary(wb: Workbook, tests: list[dict], last: int) -> dict:
    ws = wb.create_sheet("Birim Testleri Özeti")
    ws.sheet_view.showGridLines = False
    widths(ws, [40, 10, 10, 10, 14, 10, 12, 16])
    ws["A1"] = "Birim testleri – modüle göre özet"
    ws["A1"].font = font(size=14, bold=True, color=NAVY)
    ws["A2"] = (
        "Komut: python -m pytest tests/unit -q -rA --junitxml=artifacts/junit.xml (çevrimdışı; GPU, Docker ve API çağrısı yok)."
    )
    ws["A2"].font = font(color="595959", size=9)
    labels = [
        "Modül",
        "Dosya",
        "Test",
        "Geçti",
        "xfail",
        "Atlandı",
        "Başarısız",
        "Seçilmedi (integration)",
    ]
    header_row(ws, 4, labels)
    ws.row_dimensions[4].height = 30
    sheet = "'Birim Testleri'"
    mod, res = f"{sheet}!$A$2:$A${last}", f"{sheet}!$E$2:$E${last}"
    files = defaultdict(set)
    for t in tests:
        files[t["module"]].add(t["file"])
    modules = sorted(files)
    for i, m in enumerate(modules, start=5):
        row = [m, len(files[m]), f"=COUNTIF({mod},$A{i})"] + [
            f'=COUNTIFS({mod},$A{i},{res},"{x}")'
            for x in (RESULT_PASS, RESULT_XFAIL, RESULT_SKIP, RESULT_FAIL, RESULT_DESEL)
        ]
        for c, v in enumerate(row, start=1):
            style(ws.cell(row=i, column=c, value=v), center=c > 1)
    end = 4 + len(modules)
    tot = end + 1
    style(ws.cell(row=tot, column=1, value="Toplam"), bold=True, fill=GREY)
    for c in range(2, 9):
        col = get_column_letter(c)
        style(
            ws.cell(row=tot, column=c, value=f"=SUM({col}5:{col}{end})"),
            bold=True,
            center=True,
            fill=GREY,
        )
    for c in (4, 5, 7):
        col = get_column_letter(c)
        color = {4: GREEN, 5: AMBER, 7: RED}[c]
        ws.conditional_formatting.add(
            f"{col}5:{col}{end}",
            FormulaRule(formula=[f"{col}5>0"], fill=PatternFill("solid", fgColor=color)),
        )
    note = tot + 2
    for k, text in enumerate(
        [
            'xfail (beklenen hata): bilinen ve açıkça kayıtlı bir açığı belgeleyen test; test başarısız olmalıdır ve olduğu için "beklenen" sayılır. '
            'Buradaki 12 xfail: sorudaki bir sayı (yüzde, tutar, süre, tarih) kaynakta olmadan cevabı "dayanaklı" gösterebiliyor (grounding_tokens).',
            "Seçilmedi (integration): pyproject.toml 'not integration' ile varsayılan koşudan çıkarılır; ayrıştırılmış corpus (data/raw + parse önbelleği) ister.",
        ],
        start=note,
    ):
        ws.cell(row=k, column=1, value=text).font = font(color="404040", size=9)
    return {"total_row": tot, "sheet": "'Birim Testleri Özeti'"}


def build_glossary(wb: Workbook) -> None:
    ws = wb.create_sheet("Açıklamalar")
    widths(ws, [30, 110])
    header_row(ws, 1, ["Terim", "Açıklama"])
    terms = [
        (
            "Test seti (testset_v1)",
            "104 soru: 79 cevaplanabilir (tek belge 43, çok belge 16, sürüm 10, belirsiz 10) ve 25 reddedilmesi gereken (cevapsız 15, kapsam dışı 10). Sorular ayrıştırılmış belgeler okunarak yazıldı; insan incelemesi henüz yapılmadı (dogrulandi = false).",
        ),
        (
            "Kanıt alıntısı",
            "Cevabı kanıtlayan, belgede birebir geçen kısa cümle (soru başına 1–3, toplam 117). Getirilen parça alıntının belgesinden ise ve alıntının tamamını içeriyorsa isabettir. Parça kimliğine değil belgeye bağlı olduğu için farklı parçalama yöntemleri aynı setle adil ölçülür; hakeme bağlı değildir.",
        ),
        (
            "İlk isabet sırası",
            "Herhangi bir kanıt alıntısını içeren ilk parçanın 30 aday içindeki sırası. Dil modeline yalnız ilk 5 parça gider.",
        ),
        (
            "Recall@k (R@5, R@10)",
            "Bir sorunun kanıt alıntılarından kaçının ilk k parçada bulunduğu (0–1); cevaplanabilir sorular üzerinden ortalaması alınır. Kapsam kontrolünün yanlışlıkla durdurduğu 2 soru ölçüme katılmaz (77 soru).",
        ),
        (
            "MRR",
            "Mean Reciprocal Rank: ilk isabetin sırasının tersi (1. sıra 1; 2. sıra 0,5; 3. sıra 0,33; 30 adayda yoksa 0), sorular üzerinden ortalama.",
        ),
        (
            "Belge isabeti@k",
            "Beklenen belgelerden en az birinin ilk k parçada olması; aynı bilgi başka belgede de geçtiğinde alıntı ölçümünü tamamlar.",
        ),
        (
            "Claude incelemesi",
            "Altı koşunun 624 cevabı Claude (Opus) ile tek tek okundu: soru, beklenen cevap, sistemin cevabı ve atıf yapılan kaynak metinleri verildi. Bu çalışma kitabındaki doğruluk, sadakat ve ret sayıları yalnız bu incelemeden gelir; koşu içindeki otomatik hakem sonuçları kullanılmaz.",
        ),
        (
            "Doğru cevap / Yanlış cevap",
            "Cevaplanabilir ve cevaplanmış soruda cevap beklenen cevapla özünde aynı mı. Doğruluk = Doğru cevap sayısı / 79 (gereksiz ret yanlış sayılır).",
        ),
        (
            "Kaynağa sadık (sadakat)",
            "Cevaptaki her iddia atıf yapılan kaynakta geçiyor mu. Sadakat = Evet / (Evet + Hayır), cevaplanmış cevaplanabilir sorularda. Kaynakların ilk ~1.000 karakteri gösterildiği için temkinli okunmalı.",
        ),
        (
            "Doğru ret",
            'Cevapsız veya kapsam dışı soruda sistemin cevap vermemesi (cevap metninde "kaynaklarda bu bilgi yok" demesi de doğru ret sayılır). Oran = Doğru ret / 25.',
        ),
        (
            "Gereksiz ret",
            'Cevabı corpus\'ta olan soruya "bilgi bulunamadı" ya da "kapsam dışı" denmesi. Hatanın yeri: arama (kanıt ilk 5\'te yok), üretim (kanıt ilk 5\'te vardı; model ya da denetim reddetti), belirsiz. Yanlış ret oranı = gereksiz ret / 79.',
        ),
        (
            "Uydurma",
            "Cevabı corpus'ta olmayan soruya dayanaksız cevap verilmesi (ör. başka bir tutarın soruya mal edilmesi).",
        ),
        (
            "Kural tabanlı kontroller",
            "Hakem gerektirmeyen, kodla hesaplanan kontroller: beklenen sayılar cevapta mı (69 soru), atıf kesinliği, belirsiz soruda programlara ayırma, sayı denetimiyle ret, sistem hatası.",
        ),
        (
            "Sistem durumu / ret nedeni",
            'Cevaplandı; Ret: kapsam dışı (kapsam kontrolü); Ret: skor eşik altı (reranker kapısı); Ret: dayanaksız sayı (cevaptaki sayı kaynakta yok); Ret: atıfsız cevap / geçersiz atıf (atıf denetimi); Ret: bilgi yok (model kendisi "bilgi bulunamadı" dedi).',
        ),
        (
            "Birim testi",
            "tests/unit altındaki, sahte (fake) bileşenlerle çevrimdışı koşan pytest testleri; GPU, Docker, ağ ya da API gerektirmez.",
        ),
        (
            "xfail",
            '"Expected failure": bilinen bir açığı belgeleyen test. Test başarısız olur ve bu beklendiği için koşu kırılmaz; açık kapatıldığında test geçer ve işaret kaldırılır.',
        ),
        (
            "Seçilmedi (integration)",
            "integration işaretli testler varsayılan koşudan çıkarılır (gerçek corpus ya da servis ister).",
        ),
        (
            "Ortak ayarlar",
            "Altı koşuda aynı test seti ve aynı cevap modeli (qwen3:4b, temperature 0); varsayılan embedding EmbeddingGemma 2, reranker bge-reranker-v2-m3; 30 aday, dil modeline ilk 5 parça. Gecikmeler RTX 3080 GPU'da ölçüldü.",
        ),
    ]
    terms += [(label(c).split(" – ")[0], label(c)) for c in CONFIGS]
    for r, (term, desc) in enumerate(terms, start=2):
        style(ws.cell(row=r, column=1, value=term), bold=True, wrap=True)
        style(ws.cell(row=r, column=2, value=desc), wrap=True)
    ws.freeze_panes = "A2"


def build_summary(wb: Workbook, data: dict, ts_last: int, ret: dict, ans: dict, unit: dict) -> dict:
    ws = wb.active
    ws.title = "Özet"
    ws.sheet_view.showGridLines = False
    widths(ws, [46, 14, 12, 12, 12, 12, 12, 12, 12, 14])
    ws["A1"] = "TÜBİTAK Mevzuatı RAG — Testler"
    ws["A1"].font = font(size=16, bold=True, color=NAVY)
    run_ids = ", ".join(data["runs"][c]["dir"].name for c in CONFIGS)
    ws["A2"] = f"Oluşturma: {date.today().isoformat()} · Koşular: {run_ids}"
    ws["A2"].font = font(color="595959", size=9)
    ws["A3"] = (
        "Kaynaklar: eval/testset_v1.jsonl, artifacts/runs/*/per_question.jsonl, artifacts/judge/*_claude.json (Claude incelemesi), "
        "tests/unit (pytest). Bu dosya build_tests_xlsx.py ile üretilir; tablolardaki değerler diğer sayfalara formülle bağlıdır."
    )
    ws["A3"].font = font(italic=True, color="595959", size=9)

    section(ws, 5, "Neler, nasıl test edildi?")
    lines = [
        "1. Arama testi: cevaplanabilir her sorunun kanıt alıntısı (belgede birebir geçen cümle) getirilen 30 parçada aranır → Recall@5, Recall@10, MRR. Hakemden bağımsızdır.",
        "2. Cevap testi: altı yapılandırmanın 624 cevabı Claude (Opus) ile tek tek okundu → doğru / yanlış, kaynağa sadık, doğru ret, gereksiz ret, uydurma.",
        "3. Kural kontrolleri: beklenen sayılar cevapta mı, sayı ve atıf denetimleriyle ret, belirsiz soruda programlara ayırma (kodla hesaplanır).",
        "4. Birim testleri: tests/unit altında çevrimdışı pytest testleri (sahte bileşenlerle; GPU, Docker, API yok).",
        "Yapılandırmalar aynı test setini ve aynı cevap modelini (qwen3:4b) kullanır; her basamak bir öncekine göre tek bir şeyi değiştirir.",
    ]
    for k, text in enumerate(lines, start=6):
        ws.cell(row=k, column=1, value=text).font = font(color="404040")

    # --- test set by type
    r = 12
    section(ws, r, "Test seti – soru türlerine göre")
    header_row(ws, r + 1, ["Tür", "Soru", "Kolay", "Orta", "Zor", "Beklenen davranış"])
    ts = "'Test Seti'"
    tur_rng, z_rng = f"{ts}!$B$2:$B${ts_last}", f"{ts}!$C$2:$C${ts_last}"
    start = r + 2
    for i, t in enumerate(ANSWERABLE + UNANSWERABLE, start=start):
        vals = [TUR[t], f"=COUNTIF({tur_rng},$A{i})"] + [
            f'=COUNTIFS({tur_rng},$A{i},{z_rng},"{z}")' for z in ("Kolay", "Orta", "Zor")
        ]
        vals.append("Doğru ve kaynaklı cevap" if t in ANSWERABLE else EXPECT[t])
        for c, v in enumerate(vals, start=1):
            style(ws.cell(row=i, column=c, value=v), center=1 < c < 6)
    end = start + 5
    tot = end + 1
    style(ws.cell(row=tot, column=1, value="Toplam"), bold=True, fill=GREY)
    for c in range(2, 6):
        col = get_column_letter(c)
        style(
            ws.cell(row=tot, column=c, value=f"=SUM({col}{start}:{col}{end})"),
            bold=True,
            center=True,
            fill=GREY,
        )
    style(
        ws.cell(
            row=tot,
            column=6,
            value=f'=COUNTIF({ts}!$K$2:$K${ts_last},"Doğrulandı")&" / "&B{tot}&" insan incelemesinden geçti"',
        ),
        fill=GREY,
    )
    ws.merge_cells(start_row=r + 1, start_column=6, end_row=r + 1, end_column=8)
    for i in range(start, tot + 1):
        ws.merge_cells(start_row=i, start_column=6, end_row=i, end_column=8)

    # --- retrieval
    r = tot + 3
    section(ws, r, "Arama testi – yapılandırmaya göre (kanıt alıntısı, 77 soru)")
    header_row(
        ws,
        r + 1,
        [
            "Yapılandırma (yöntem)",
            "R@5",
            "R@10",
            "MRR",
            "1. sırada",
            "2–5",
            "6–30",
            "Yok (30 adayda)",
            "Ölçülen soru",
        ],
    )
    asheet = "'Arama Sonuçları'"
    rf, rl = ret["first"], ret["last"]
    ret_rows = {}
    for i, cfg in enumerate(CONFIGS, start=r + 2):
        c = ret["cols"][cfg]
        rank = f"{asheet}!${c['rank']}${rf}:${c['rank']}${rl}"
        vals = [
            label(cfg),
            f"={asheet}!{c['r5']}{c['avg_row']}",
            f"={asheet}!{c['r10']}{c['avg_row']}",
            f"={asheet}!{c['rr']}{c['avg_row']}",
            f"=COUNTIF({rank},1)",
            f'=COUNTIFS({rank},">=2",{rank},"<=5")',
            f'=COUNTIFS({rank},">=6",{rank},"<=30")',
            f'=COUNTIF({rank},"—")',
            f"={asheet}!{c['rank']}{c['cnt_row']}",
        ]
        for k, v in enumerate(vals, start=1):
            style(
                ws.cell(row=i, column=k, value=v),
                wrap=(k == 1),
                center=k > 1,
                fmt="0.000" if k in (2, 3, 4) else None,
            )
        ret_rows[cfg] = i
    ws.conditional_formatting.add(f"B{r + 2}:D{r + 7}", _scale())
    r = r + 9
    section(ws, r, "Recall@5 – soru türüne göre")
    header_row(ws, r + 1, ["Tür"] + CONFIGS)
    for i, t in enumerate(ANSWERABLE, start=r + 2):
        style(ws.cell(row=i, column=1, value=f"{TUR[t]}"))
        for j, cfg in enumerate(CONFIGS, start=2):
            c = ret["cols"][cfg]
            f = f"=IFERROR(AVERAGEIFS({asheet}!${c['r5']}${rf}:${c['r5']}${rl},{asheet}!$B${rf}:$B${rl},$A{i}),0)"
            style(ws.cell(row=i, column=j, value=f), center=True, fmt="0.000")
    ws.conditional_formatting.add(f"B{r + 2}:G{r + 5}", _scale())

    # --- answers
    r = r + 7
    section(ws, r, "Cevap testi – Claude (Opus) incelemesi")
    header_row(
        ws,
        r + 1,
        [
            "Yapılandırma (yöntem)",
            "Doğru cevap",
            "Doğruluk (/79)",
            "Sadakat",
            "Doğru ret (/25)",
            "Gereksiz ret",
            "Yanlış ret oranı",
            "Uydurma",
            "Gecikme p50 (ms)",
        ],
    )
    cs = "'Cevap Sonuçları'"
    af, al = ans["first"], ans["last"]
    exp_rng = f"{cs}!$D${af}:$D${al}"
    ans_rows = {}
    for i, cfg in enumerate(CONFIGS, start=r + 2):
        c = ans["cols"][cfg]
        kar = f"{cs}!${c['karar']}${af}:${c['karar']}${al}"
        sad = f"{cs}!${c['sadik']}${af}:${c['sadik']}${al}"
        p50 = data["runs"][cfg]["metrics"]["latency_ms"]["total_p50"]
        vals = [
            label(cfg),
            f'=COUNTIF({kar},"Doğru cevap")',
            f'=B{i}/COUNTIF({exp_rng},"Cevap")',
            f'=COUNTIF({sad},"Evet")/(COUNTIF({sad},"Evet")+COUNTIF({sad},"Hayır"))',
            f'=COUNTIF({kar},"Doğru ret")/COUNTIF({exp_rng},"Ret*")',
            f'=COUNTIF({kar},"Gereksiz ret*")',
            f'=F{i}/COUNTIF({exp_rng},"Cevap")',
            f'=COUNTIF({kar},"Uydurma")',
            round(p50),
        ]
        for k, v in enumerate(vals, start=1):
            style(
                ws.cell(row=i, column=k, value=v),
                wrap=(k == 1),
                center=k > 1,
                fmt="0.000" if k in (3, 4, 5, 7) else "#,##0" if k == 9 else None,
            )
        ans_rows[cfg] = i
    ws.conditional_formatting.add(f"C{r + 2}:E{r + 7}", _scale())
    r = r + 9
    section(ws, r, "Doğru davranış sayısı – soru türüne göre (Claude)")
    header_row(ws, r + 1, ["Tür"] + CONFIGS)
    tur_ans = f"{cs}!$B${af}:$B${al}"
    for i, t in enumerate(ANSWERABLE + UNANSWERABLE, start=r + 2):
        target = "Doğru cevap" if t in ANSWERABLE else "Doğru ret"
        style(
            ws.cell(
                row=i,
                column=1,
                value=f"{TUR[t]} ({'doğru cevap' if t in ANSWERABLE else 'doğru ret'})",
            )
        )
        for j, cfg in enumerate(CONFIGS, start=2):
            c = ans["cols"][cfg]
            f = f'=COUNTIFS({tur_ans},"{TUR[t]}",{cs}!${c["karar"]}${af}:${c["karar"]}${al},"{target}")'
            style(ws.cell(row=i, column=j, value=f), center=True)
    r = r + 9
    section(ws, r, "Gereksiz retlerde hatanın yeri (Claude)")
    header_row(ws, r + 1, ["Hatanın yeri"] + CONFIGS)
    for i, place in enumerate(["arama", "üretim", "belirsiz"], start=r + 2):
        style(ws.cell(row=i, column=1, value=f"Gereksiz ret ({place})"))
        for j, cfg in enumerate(CONFIGS, start=2):
            c = ans["cols"][cfg]
            style(
                ws.cell(
                    row=i,
                    column=j,
                    value=f"=COUNTIF({cs}!${c['karar']}${af}:${c['karar']}${al},$A{i})",
                ),
                center=True,
            )

    # --- rule checks
    r = r + 6
    section(ws, r, "Kural tabanlı kontroller")
    header_row(ws, r + 1, ["Kontrol"] + CONFIGS)
    checks = [
        (
            "Beklenen sayılar cevapta (69 soru)",
            lambda cfg, c: f'=COUNTIF({cs}!${c["nums"]}${af}:${c["nums"]}${al},"Evet")',
            None,
        ),
        (
            "Sayı denetimiyle ret",
            lambda cfg, c: (
                f'=COUNTIF({cs}!${c["status"]}${af}:${c["status"]}${al},"Ret: dayanaksız sayı")'
            ),
            None,
        ),
        (
            "Kapsam kontrolüyle ret (cevaplanabilir)",
            lambda cfg, c: (
                f'=COUNTIFS({cs}!${c["status"]}${af}:${c["status"]}${al},"Ret: kapsam dışı",{exp_rng},"Cevap")'
            ),
            None,
        ),
        (
            "Atıf kesinliği (koşu dosyası)",
            lambda cfg, c: data["runs"][cfg]["metrics"]["generation"].get("citation_precision"),
            "0.000",
        ),
        (
            "Programlara ayırma, belirsiz 10 soru (koşu dosyası)",
            lambda cfg, c: sum(
                1
                for row in data["runs"][cfg]["rows"].values()
                if row["tur"] == "belirsiz" and row["rule_check"].get("programlara_ayrildi")
            ),
            None,
        ),
        (
            "Sistem hatası (koşu dosyası)",
            lambda cfg, c: sum(
                1
                for row in data["runs"][cfg]["rows"].values()
                if row["answer"].get("red_nedeni") == "llm_hatasi"
            ),
            None,
        ),
    ]
    for i, (name, fn, fmt) in enumerate(checks, start=r + 2):
        style(ws.cell(row=i, column=1, value=name), wrap=True)
        for j, cfg in enumerate(CONFIGS, start=2):
            style(ws.cell(row=i, column=j, value=fn(cfg, ans["cols"][cfg])), center=True, fmt=fmt)

    # --- unit tests
    r = r + 2 + len(checks) + 1
    section(ws, r, "Birim testleri (tests/unit)")
    header_row(ws, r + 1, ["Gösterge", "Değer"])
    us, ut = unit["sheet"], unit["total_row"]
    rows = [
        ("Toplam test", f"={us}!C{ut}"),
        ("Geçti", f"={us}!D{ut}"),
        ("Beklenen hata (xfail)", f"={us}!E{ut}"),
        ("Atlandı", f"={us}!F{ut}"),
        ("Başarısız", f"={us}!G{ut}"),
        ("Seçilmedi (integration)", f"={us}!H{ut}"),
        (
            "Geçme oranı (xfail hariç)",
            f"=IFERROR(B{r + 3}/(B{r + 2}-B{r + 4}-B{r + 5}-B{r + 7}),0)",
        ),
    ]
    for i, (name, f) in enumerate(rows, start=r + 2):
        style(ws.cell(row=i, column=1, value=name))
        style(
            ws.cell(row=i, column=2, value=f),
            bold=True,
            center=True,
            fmt="0.0%" if "oranı" in name else None,
        )
    ws.cell(
        row=r + 2 + len(rows), column=1, value="Modül kırılımı: 'Birim Testleri Özeti' sayfası."
    ).font = font(color="595959", size=9)

    r = r + 4 + len(rows)
    section(ws, r, "Nasıl okumalı?")
    for k, text in enumerate(
        [
            "79 cevaplanabilir soruda 1 soru ≈ 1,3 puan, 25 ret sorusunda 4 puandır; v3 ile v3b arasındaki farklar bu belirsizliğin içindedir.",
            "Renkler: 'Arama Sonuçları' yeşil = kanıt ilk 5'te, sarı = 6–10, turuncu = 11–30, kırmızı = yok; 'Cevap Sonuçları' yeşil = doğru / doğru ret, kırmızı = yanlış / uydurma, sarı = gereksiz ret.",
            "Seçilmiş başarısızlık örnekleri ve durumları 'Hata Örnekleri' sayfasında; terimler 'Açıklamalar' sayfasında.",
        ],
        start=r + 1,
    ):
        ws.cell(row=k, column=1, value=text).font = font(color="404040")
    return {"ret_rows": ret_rows, "ans_rows": ans_rows}


def _scale():
    from openpyxl.formatting.rule import ColorScaleRule

    return ColorScaleRule(
        start_type="min", start_color="F8F8F8", end_type="max", end_color="A9D08E"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--junit", type=Path, default=DEFAULT_ROOT / "artifacts" / "junit.xml")
    ap.add_argument("--out", type=Path, default=DEFAULT_ROOT / "docs" / "testler.xlsx")
    ap.add_argument(
        "--deselected",
        nargs="*",
        default=[
            "tests/unit/evaluation/test_testset.py::test_repo_testset_quotes_match_parsed_corpus"
        ],
        help="node ids deselected by the default marker expression (pytest --collect-only -m integration)",
    )
    args = ap.parse_args()
    root = args.root

    data = load_data(root)
    tests = load_unit_tests(root, args.junit, args.deselected)
    wb = Workbook()
    ts_last = build_testset(wb, data["questions"])
    ret = build_retrieval(wb, data)
    ans = build_answers(wb, data)
    build_failures(wb, data)
    unit_last = build_unit_tests(wb, tests)
    unit = build_unit_summary(wb, tests, unit_last)
    build_glossary(wb)
    build_summary(wb, data, ts_last, ret, ans, unit)
    wb.move_sheet("Özet", offset=-wb.index(wb["Özet"]))
    wb.active = 0
    wb.save(args.out)
    print(f"{len(data['questions'])} questions, {len(tests)} unit tests -> {args.out}")


if __name__ == "__main__":
    main()
