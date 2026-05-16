# src/kif_parser.py

import re
from pathlib import Path
import pandas as pd


def parse_kif(filepath: str) -> dict:
    """
    KIFファイルをパースしてヘッダ情報を返す
    消費時間は対象外（全ファイル未記録のため）
    """
    result = {
        "filename":           Path(filepath).name,
        "start_datetime":     None,
        "end_datetime":       None,
        "event":              None,
        "time_limit_minutes": None,
        "handicap":           None,
        "sente":              None,
        "gote":               None,
        "opening":            None,
        "total_moves":        None,
        "termination":        None,
        "winner":             None,  # "sente" / "gote"
    }

    header_map = {
        "開始日時": "start_datetime",
        "終了日時": "end_datetime",
        "棋戦":    "event",
        "手合割":  "handicap",
        "先手":    "sente",
        "後手":    "gote",
        "戦型":    "opening",
    }

    with open(filepath, encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    in_moves = False
    for line in lines:
        line = line.rstrip("\n")

        if not in_moves:
            for key, field in header_map.items():
                if line.startswith(f"{key}："):
                    result[field] = line.split("：", 1)[1].strip()
                    break

            if line.startswith("持ち時間："):
                val = line.split("：", 1)[1].strip()
                result["time_limit_minutes"] = _parse_time_limit(val)

            if line.startswith("手数"):
                in_moves = True
            continue

        # 指し手行（消費時間あり・なし両対応）
        m = re.match(r"\s*(\d+)\s+(.+?)(?:\s+\([\d:]+/[\d:]+\))?$", line)
        if not m:
            continue

        move_num = int(m.group(1))
        move_str = m.group(2).strip()

        if move_str in ("投了", "中断", "持将棋", "千日手"):
            result["termination"] = move_str
            result["total_moves"] = move_num - 1
            if move_str == "投了":
                # 投了したのは手番側（奇数手目=先手番）
                result["winner"] = "gote" if move_num % 2 == 1 else "sente"
            break

    return result


def _parse_time_limit(val: str) -> float | None:
    """「4時間」「各5分」「10分」→ 分（float）"""
    m = re.search(r"(\d+)時間", val)
    if m:
        hours = int(m.group(1))
        m2 = re.search(r"(\d+)分", val)
        minutes = int(m2.group(1)) if m2 else 0
        return hours * 60 + minutes
    m = re.search(r"(\d+)分", val)
    if m:
        return float(m.group(1))
    return None


def build_game_df(kif_dir: str) -> pd.DataFrame:
    """
    ディレクトリ内の全KIFをパースしてDataFrameを返す
    勝敗不明（投了以外の終局 or パース失敗）は除外
    """
    records = []
    skipped = []

    for path in sorted(Path(kif_dir).glob("**/*.kif")):
        r = parse_kif(str(path))
        if r["winner"] is None:
            skipped.append(r["filename"])
            continue
        records.append(r)

    if skipped:
        print(f"[skipped] {len(skipped)} files (winner=None): {skipped[:5]} ...")

    df = pd.DataFrame(records)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], errors="coerce")
    df["end_datetime"]   = pd.to_datetime(df["end_datetime"],   errors="coerce")
    return df