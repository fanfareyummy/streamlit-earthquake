import os
import tempfile
import urllib.request
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 슈팅스타팩트 & 우주 스타티니핑 커스텀 스타일링 (CSS)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 - 지진 탐지 시스템", page_icon="✨", layout="wide")

st.markdown(
    """
    <style>
    /* 전체 배경을 깊은 우주 테마로 변경 */
    .main {
        background: radial-gradient(circle at center, #110022 0%, #050011 70%, #000000 100%);
        color: #ffffff;
    }
    
    /* 슈팅스타팩트 상단 마법 스크린 헤더 */
    .shooting-star-header {
        background: linear-gradient(135deg, #ff66b2 0%, #9933ff 50%, #33ccff 100%);
        padding: 30px;
        border-radius: 24px;
        color: #fff;
        text-align: center;
        box-shadow: 0 0 30px rgba(255, 102, 178, 0.6);
        border: 3px solid #ffccff;
        margin-bottom: 25px;
    }
    .shooting-star-header h1 {
        margin: 0;
        font-size: 36px;
        font-weight: 900;
        text-shadow: 0 0 15px #fff, 0 0 20px #ff3399;
    }
    .shooting-star-header p {
        margin: 10px 0 0;
        font-size: 16px;
        color: #ffffcc;
        font-weight: 600;
    }
    
    /* 스타팩트 마법 등급 배지 */
    .risk-badge {
        display: inline-block;
        padding: 20px;
        border-radius: 20px;
        color: #fff;
        font-size: 24px;
        font-weight: 900;
        width: 100%;
        text-align: center;
        border: 2px solid #fff;
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    .fact-high { background: radial-gradient(circle, #ff3333, #990000); box-shadow: 0 0 25px #ff3333; }
    .fact-mid  { background: radial-gradient(circle, #ff9933, #994c00); box-shadow: 0 0 25px #ff9933; }
    .fact-low  { background: radial-gradient(circle, #33cc33, #006600); box-shadow: 0 0 25px #33cc33; }
    .fact-none { background: radial-gradient(circle, #778899, #445566); box-shadow: 0 0 15px #778899; }
    
    /* 스타팩트 내부 홀로그램 정보 카드 */
    .fact-card {
        background: rgba(15, 0, 30, 0.7);
        border: 2px solid #ff66b2;
        border-radius: 18px;
        padding: 20px;
        color: #fff;
        box-shadow: inset 0 0 15px rgba(255, 102, 178, 0.3), 0 0 15px rgba(51, 204, 255, 0.2);
    }
    .fact-card b { color: #ff99cc; }
    
    /* 스트림릿 기본 위젯 내부 커스텀 */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid #ff66b2;
        border-radius: 16px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(255, 102, 178, 0.2);
    }
    div[data-testid="stMetric"] label { color: #ffccff !important; font-weight: bold; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00ffff !important; font-weight: 800; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 한글 폰트 설정
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def setup_korean_font():
    mpl.rcParams["axes.unicode_minus"] = False
    local_candidates = [
        "C:/Windows/Fonts/malgun.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for path in local_candidates:
        if os.path.exists(path):
            try:
                fm.fontManager.addfont(path)
                name = fm.FontProperties(fname=path).get_name()
                mpl.rc("font", family=name)
                return name
            except Exception:
                pass
    return None

KOREAN_FONT = setup_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 데이터 생성 및 머신러닝 로직 (기존 구조 강건하게 유지)
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_data():
    # 주어질 quake.csv 대용 샘플 (파일이 없을 경우 대비)
    np.random.seed(42)
    num_samples = 5000
    df = pd.DataFrame({
        '위도': np.random.uniform(-60, 60, num_samples),
        '경도': np.random.uniform(-180, 180, num_samples),
        '규모': np.random.uniform(2.0, 7.5, num_samples),
        '진원깊이': np.random.uniform(0, 600, num_samples),
        '영향도': np.random.uniform(10, 100, num_samples),
    })
    return df

df_path = os.path.join(APP_DIR, "quake.csv")
if os.path.exists(df_path):
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            df = pd.read_csv(df_path, encoding=enc)
            break
        except: continue
else:
    df = load_sample_data()

# 간단 모델링 빌드 (K-means, K=3)
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit(X_scaled)

# 위험 등급 자동 매핑
agg = df.groupby("cluster")[FEATURES].mean()
score = ((agg["규모"] - agg["규모"].min()) / (agg["규모"].max() - agg["규모"].min() + 1e-5) * 0.5 + 
         (1 - (agg["진원깊이"] - agg["진원깊이"].min()) / (agg["진원깊이"].max() - agg["진원깊이"].min() + 1e-5)) * 0.5)
order = score.sort_values(ascending=False).index.tolist()
labels = ["고위험", "중위험", "저위험"]
grade_map = {int(c): labels[i] for i, c in enumerate(order)}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# 스타핑 스타일 위험도 디자인 가이드
RISK_STYLE = {
    "고위험": ("fact-high", "⭐🚨", "슈팅스타 에너지 감지! 파괴적인 강진이 빈번한 고위험 레이더 존입니다!"),
    "중위험": ("fact-mid", "⭐⚡", "진원이 다소 깊지만, 에너지가 축적된 중간 위험도 레이더 존입니다."),
    "저위험": ("fact-low", "⭐✨", "우주 별빛처럼 안정적이고 평화로운 저위험 레이더 존입니다."),
    "판단보류": ("fact-none", "⭐💫", "스타팩트 레이더 반경 내에 지진 에너지가 포착되지 않은 고요한 공간입니다."),
}

# ═════════════════════════════════════════════════════════════
# 헤더 : 슈팅스타팩트 마법 도구 연출
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="shooting-star-header">
        <h1>✨ SHOOTING STAR PACT : 스타티니핑 지진 레이더 ✨</h1>
        <p>시즌 5 슈팅스타팩트 전개! 우주 위성 주파수를 동기화하여 전 세계 지진 군집을 3D 홀로그램으로 분석합니다.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# 사이드바 컨트롤러 (팩트 조작 버튼 스타일)
st.sidebar.header("🎵 스타팩트 컨트롤 패널")
st.sidebar.caption("스타티니핑 주파수 동기화")
K_NEIGHBORS = st.sidebar.slider("레이더 동기화 별빛 노드 개수 (k)", 5, 50, 20)
st.sidebar.markdown("---")
st.sidebar.subheader("🌟 스페이스 포스 데이터")
st.sidebar.write(f"📡 수신된 전세계 스타 노드: {len(df):,}개")

# ═════════════════════════════════════════════════════════════
# 입력창 : 슈팅스타팩트 좌표 패드
# ═════════════════════════════════════════════════════════════
st.subheader("🔮 스타 레이더 타겟 좌표 입력")
c1, c2 = st.columns(2)
with c1:
    lat = st.number_input("팩트 타겟 위도 (Latitude)", -90.0, 90.0, 37.5, 0.1, format="%.3f")
with c2:
    lon = st.number_input("팩트 타겟 경도 (Longitude)", -180.0, 180.0, 127.0, 0.1, format="%.3f")

# ═════════════════════════════════════════════════════════════
# 마법 레이더 가동 및 위험도 판단
# ═════════════════════════════════════════════════════════════
if st.button("🪄 슈팅스타 팩트 가동! (지진 에너지 스캔)", type="primary", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    order = np.argsort(dist)[:K_NEIGHBORS]
    near = df.iloc[order].copy()
    nearest_km = float(dist[order[0]])

    if nearest_km > 1500:
        grade = "판단보류"
        dom_cluster = None
    else:
        cw = {}
        for idx, c in zip(order, near["cluster"].values):
            cw[int(c)] = cw.get(int(c), 0.0) + 1.0 / (dist[idx] + 50.0)
        dom_cluster = int(max(cw, key=cw.get))
        grade = grade_map.get(dom_cluster, "저위험")

    css_cls, emoji, desc = RISK_STYLE[grade]
    st.divider()

    # 슈팅스타팩트 LCD 내부 디스플레이 화면 느낌의 배치
    r1, r2 = st.columns([1, 1.4])
    with r1:
        sub = f"스타 클러스터 {dom_cluster}번 연동" if dom_cluster is not None else "우주 미개척 구역"
        st.markdown(
            f'<div class="risk-badge {css_cls}">{emoji} {grade}<br>'
            f'<span style="font-size:14px; font-weight:600; opacity:0.8;">{sub}</span></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="fact-card"><b>🔮 슈팅스타팩트 분석 리포트</b><br>{desc}<br><br>'
            f'<b>🎯 레이더 동기화 좌표</b> : 위도 {lat:.3f}°, 경도 {lon:.3f}°<br>'
            f'<b>🌌 가장 가까운 코어 엔티티</b> : {nearest_km:,.1f} km 거리 분리됨</div>',
            unsafe_allow_html=True,
        )

    # 하단 인터랙티브 메트릭 스크린
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("🔮 주변 평균 지진 규모", f"Mag {near['규모'].mean():.1f}")
    m2.metric("🌌 포착된 평균 진원 깊이", f"{near['진원깊이'].mean():.0f} km")
    m3.metric("✨ 스타 이펙트 영향도", f"{near['영향도'].mean():.1f} pt")

    # ═════════════════════════════════════════════════════════════
    # 🛰️ 슈팅스타팩트 내부 스크린: 3D 우주 위성 맵 (Pydeck 지구본 뷰)
    # ═════════════════════════════════════════════════════════════
    st.subheader("📺 슈팅스타팩트 내부 메인 홀로그램 스크린 (3D 우주 뷰)")
    
    # 샘플링 및 네온 컬러 칩 부여
    show = df.sample(min(4000, len(df)), random_state=42).copy()
    
    # 고위험: 네온 핫핑크 / 중위험: 일렉트릭 오렌지 / 저위험: 민트 그린
    COLOR_MAP = {"고위험": [255, 20, 147, 180], "중위험": [255, 165, 0, 180], "저위험": [0, 255, 200, 180]}
    show['color'] = show['cluster'].map(lambda c: COLOR_MAP.get(grade_map.get(int(c)), [150, 150, 150, 150]))

    # 타겟 마커 데이터 (슈팅스타 모양 대용의 강력한 마법 교차점)
    target_df = pd.DataFrame([{"lon": lon, "lat": lat}])

    layers = [
        # 전세계 지진 분출구 (별빛 네온 포인트)
        pdk.Layer(
            'ScatterplotLayer',
            data=show,
            get_position='[경도, 위도]',
            get_color='color',
            get_radius=40000,
            pickable=True
        ),
        # 슈팅스타 마법 조준점 (황금빛 스타 아우라 펄스)
        pdk.Layer(
            'ScatterplotLayer',
            data=target_df,
            get_position='[lon, lat]',
            get_color=[255, 255, 0, 255],
            get_radius=150000,
            filled=True,
            stroked=True,
            line_width_min_pixels=3,
            get_line_color=[255, 102, 218, 255] # 핑크 테두리 아우라
        )
    ]

    # 우주선/스타팩트 안에서 지구를 내려다보는 각도 (Pitch=50, Bearing=20)
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=2.8,
        pitch=50,
        bearing=20
    )

    # Mapbox의 완전 다크 스페이스 스타일 맵 연동하여 우주 느낌 극대화
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "스타 등급: {cluster}\n위도: {위도}°\n경도: {경도}°"}
    )
    st.pydeck_chart(r)

    # ═════════════════════════════════════════════════════════════
    # 스타티니핑 매직 2D 산점도 (경도 vs 위도)
    # ═════════════════════════════════════════════════════════════
    st.subheader("📊 우주 스타 레이더 매트릭스")
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor='#050011')
    ax.set_facecolor('#0f0022')
    
    # 차트 한글 깨짐 및 테마 색상 수정
    if KOREAN_FONT:
        mpl.rc('font', family=KOREAN_FONT)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('#ff66b2')

    HEX_MAP = {"고위험": "#ff1493", "중위험": "#ffa500", "저위험": "#00ffc8"}
    for c in sorted(df["cluster"].unique()):
        s_ = df[df["cluster"] == c]
        g_name = grade_map.get(int(c))
        ax.scatter(s_["경도"], s_["위도"], s=6, alpha=0.4,
                   color=HEX_MAP.get(g_name, "#ffffff"),
                   label=f"클러스터 {c} ({g_name})")
                   
    # 마법 별모양 타겟 포인트 마커
    ax.scatter(lon, lat, c="#ffff00", s=300, marker="*",
               edgecolors="#ff66b2", linewidths=2, zorder=5, label="팩트 타겟 포인트")
               
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("우주 경도선 (Longitude)")
    ax.set_ylabel("우주 위도선 (Latitude)")
    ax.set_title("✨ 슈팅스타팩트 레이더 매핑 도표", fontsize=14, fontweight='bold')
    
    legend = ax.legend(fontsize=9, loc="lower left")
    plt.setp(legend.get_texts(), color='white')
    legend.get_frame().set_facecolor('#110022')
    legend.get_frame().set_edgecolor('#ff66b2')
    
    ax.grid(color='#ff66b2', alpha=0.15)
    st.pyplot(fig)
    plt.close(fig)
