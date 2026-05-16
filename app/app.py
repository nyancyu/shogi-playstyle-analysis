import sys
import pickle
import numpy as np
import streamlit as st

sys.path.insert(0, "../src")

# モデル読み込み
@st.cache_resource
def load_artifacts():
    with open("models/lgbm_artifacts.pkl", "rb") as f:
        return pickle.load(f)

artifacts = load_artifacts()
model       = artifacts["model"]
winrate_map = artifacts["winrate_map"]
games_map   = artifacts["games_map"]
le          = artifacts["le"]
time_median = artifacts["time_median"]
feature_cols = artifacts["feature_cols"]
valid_players = artifacts["valid_players"]
openings      = artifacts["openings"]

# ページ設定
st.set_page_config(page_title="将棋勝敗予測", page_icon="♟️", layout="centered")
st.title("♟️ 将棋勝敗予測モデル")
st.caption("対局前の事前情報から先手/後手の勝率を予測します")

st.divider()

# 入力フォーム
col1, col2 = st.columns(2)
with col1:
    st.subheader("▲ 先手")
    sente = st.selectbox("棋士を選択", valid_players, key="sente")
    sente_wr  = winrate_map.get(sente, np.nan)
    sente_g   = games_map.get(sente, 0)
    st.metric("勝率", f"{sente_wr:.1%}" if not np.isnan(sente_wr) else "不明")
    st.metric("対局数", f"{sente_g}局")

with col2:
    st.subheader("△ 後手")
    gote_options = [p for p in valid_players if p != sente]
    gote = st.selectbox("棋士を選択", gote_options, key="gote")
    gote_wr = winrate_map.get(gote, np.nan)
    gote_g  = games_map.get(gote, 0)
    st.metric("勝率", f"{gote_wr:.1%}" if not np.isnan(gote_wr) else "不明")
    st.metric("対局数", f"{gote_g}局")

st.divider()

col3, col4 = st.columns(2)
with col3:
    opening = st.selectbox("戦型", openings)
with col4:
    time_limit = st.selectbox(
        "持ち時間",
        [5, 10, 30, 60, 120, 240, 300, 360, 540],
        index=5,
        format_func=lambda x: f"{x}分" if x < 60 else f"{x//60}時間{x%60}分" if x % 60 else f"{x//60}時間"
    )

st.divider()

# 予測
if st.button("予測する", type="primary", use_container_width=True):
    try:
        opening_enc = le.transform([opening])[0]
    except ValueError:
        opening_enc = le.transform(["不明"])[0]

    features = np.array([[
        winrate_map.get(sente, np.nan),
        winrate_map.get(gote, np.nan),
        winrate_map.get(sente, np.nan) - winrate_map.get(gote, np.nan)
            if not np.isnan(winrate_map.get(sente, np.nan)) and not np.isnan(winrate_map.get(gote, np.nan))
            else np.nan,
        games_map.get(sente, 0),
        games_map.get(gote, 0),
        0,  # h2h_games
        opening_enc,
        time_limit,
    ]])

    prob_sente = model.predict_proba(features)[0][1]
    prob_gote  = 1 - prob_sente

    st.subheader("予測結果")
    col5, col6 = st.columns(2)
    with col5:
        st.metric("▲ 先手勝率", f"{prob_sente:.1%}",
                  delta=f"{prob_sente - 0.5:+.1%} vs 五分")
    with col6:
        st.metric("△ 後手勝率", f"{prob_gote:.1%}",
                  delta=f"{prob_gote - 0.5:+.1%} vs 五分")

    winner = sente if prob_sente >= 0.5 else gote
    side   = "先手" if prob_sente >= 0.5 else "後手"
    prob   = max(prob_sente, prob_gote)
    st.success(f"**{side}（{winner}）** の勝利を予測　（確信度 {prob:.1%}）")

    # 判断根拠
    with st.expander("予測の根拠"):
        diff = winrate_map.get(sente, np.nan) - winrate_map.get(gote, np.nan)
        if not np.isnan(diff):
            st.write(f"- 実力差（winrate_diff）: {diff:+.3f}")
        st.write(f"- 先手勝率: {winrate_map.get(sente, np.nan):.3f}" if not np.isnan(winrate_map.get(sente, np.nan)) else "- 先手勝率: 不明")
        st.write(f"- 後手勝率: {winrate_map.get(gote, np.nan):.3f}" if not np.isnan(winrate_map.get(gote, np.nan)) else "- 後手勝率: 不明")
        st.write(f"- 持ち時間: {time_limit}分")
        st.write(f"- 戦型: {opening}")

st.divider()
st.caption("データ: 将棋DB2 | モデル: LightGBM (AUC=0.781, 5-fold CV)")
