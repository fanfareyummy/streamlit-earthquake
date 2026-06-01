import os
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 🌌 캐치! 티니핑 시즌 5: 슈팅스타팩트 영롱한 오로라 우주 디자인 (CSS)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="반짝반짝! 슈팅스타팩트 오로라 레이더", page_icon="🪐", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@400;700;900&display=swap');

    /* 전체 웹 배경: 깊은 우주와 오로라가 감도는 딥 오션 & 퍼플 그라데이션 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', 'Orbit', 'Malgun Gothic', sans-serif !important;
        background: radial-gradient(circle at center, #0d002d 0%, #050014 70%, #010005 100%) !important;
        color: #e0ffff !important;
    }
    
    /* 은하수 성단 가루 효과 */
    .stMainBlockContainer {
        background-image: radial-gradient(rgba(0, 255, 204, 0.15) 1px, transparent 40px),
                          radial-gradient(rgba(138, 43, 226, 0.15) 2px, transparent 30px);
        background-size: 400px 400px, 300px 300px;
        background-position: 0 0, 50px 50px;
    }

    /* 슈팅스타팩트 최상단: 영롱한 오로라 네온 엠블럼 헤더 */
    .shooting-star-header {
        background: linear-gradient(135deg, rgba(0, 255, 204, 0.2) 0%, rgba(75, 0, 130, 0.5) 50%, rgba(255, 20, 147, 0.15) 100%);
        padding: 35px;
        border-radius: 25px;
        color: #fff;
        text-align: center;
        box-shadow: 0 0 35px rgba(0, 255, 204, 0.4), inset 0 0 20px rgba(138, 43, 226, 0.4);
        border: 2px solid #00ffcc;
        margin-bottom: 30px;
        backdrop-filter: blur(12px);
    }
    .shooting-star-header h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 900;
        letter-spacing: 1px;
        text-shadow: 0 0 12px #fff, 0 0 25px #00ffcc, 0 0 35px #8a2be2;
    }
    .shooting-star-header p {
        margin: 12px 0 0;
        font-size: 17px;
        color: #d1f7ff;
        font-weight: bold;
        text-shadow: 0 0 6px rgba(0,0,0,0.8);
    }
    
    /* 스타팩트 하트/별빛 오로라 등급 배지 */
    .risk-badge {
        display: inline-block;
        padding: 25px;
        border-radius: 25px;
        color: #fff;
        font-size: 26px;
        font-weight: 900;
        width: 100%;
        text-align: center;
        border: 2px dashed rgba(255, 255, 255, 0.6);
        box-shadow: 0 0 25px rgba(0, 255, 204, 0.2);
    }
    .fact-high { background: radial-gradient(circle, #ff1493, #4b0082); box-shadow: 0 0 30px rgba(255, 14, 147, 0.6); text-shadow: 0 0 8px #fff; border-color: #ff69b4; }
    .fact-mid  { background: radial-gradient(circle, #ffaa00, #3a0066); box-shadow: 0 0 30px rgba(255, 170, 0, 0.5); text-shadow: 0 0 8px #fff; border-color: #ffd700; }
    .fact-low  { background: radial-gradient(circle, #00ffcc, #0a2342); box-shadow: 0 0 30px rgba(0, 255, 204, 0.6); text-shadow: 0 0 8px #000; border-color: #e0ffff; }
    .fact-none { background: radial-gradient(circle, #483d8b, #120024); box-shadow: 0 0 20px #483d8b; border-color: #8a2be2; }
    
    /* 초롱핑의 오로라 비밀 카드 스크린 */
    .fact-card {
        background: rgba(10, 3, 35, 0.85);
        border: 2px solid #8a2be2;
        border-radius: 22px;
        padding: 25px;
        color: #ffffff;
        box-shadow: inset 0 0 20px rgba(138, 43, 226, 0.4), 0 0 25px rgba(0, 255, 204, 0.2);
    }
    .fact-card b { color: #00ffcc; font-size: 18px; text-shadow: 0 0 8px rgba(0,255,204,0.5); }
    
    /* 사이드바 우주 제어 캡슐 */
    [data-testid="stSidebar"] {
        background-color: #050014 !important;
        border-right: 2px solid #8a2be2 !important;
    }
    
    /* 오로라 메트릭 계기판 */
    div[data-testid="stMetric"] {
        background: rgba(0, 255, 204, 0.04);
        border: 1px solid #8a2be2;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 0 20px rgba(138, 43, 226, 0.25);
        text-align: center;
    }
    div[data-testid="stMetric"] label { color: #d1f7ff !important; font-weight: bold; font-size: 15px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #00ffcc !important; font-weight: 900; font-size: 26px !important; text-shadow: 0 0 10px rgba(0,255,204,0.7); }
    
    /* 오로라 스캔 스타트 버튼 */
    .stButton>button {
        background: linear-gradient(90deg, #4b0082, #8a2be2, #00ffcc) !important;
        color: white !important;
        font-weight: 900 !important;
        font-size: 17px !important;
        border-radius: 20px !important;
        border: 2px solid #00ffcc !important;
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.3) !important;
        transition: all 0.3s ease;
        padding: 12px 0px !important;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 35px #00ffcc !important;
        color: #fff !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 깨짐 없는 깔끔한 한글 폰트 글로벌 설정 (Matplotlib 전용)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def setup_korean_font():
    mpl.rcParams["axes.unicode_minus"] = False
    local_candidates = [
        "C:/Windows/Fonts/malgun.ttf", 
        "C:/Windows/Fonts/malgunbd.ttf",
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
    return "sans-serif"

KOREAN_FONT = setup_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 데이터 로드 및 K-Means 머신러닝 프로세스
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_data():
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

# K-Means 클러스터링 고도화
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

df = df.dropna(subset=["cluster", "위도", "경도"])
df["cluster"] = df["cluster"].astype(int)

# 위험 등급 연산 및 인덱스 매핑
agg = df.groupby("cluster")[FEATURES].mean()
score = ((agg["규모"] - agg["규모"].min()) / (agg["규모"].max() - agg["규모"].min() + 1e-5) * 0.5 + 
         (1 - (agg["진원깊이"] - agg["진원깊이"].min()) / (agg["진원깊이"].max() - agg["진원깊이"].min() + 1e-5)) * 0.5)
order = score.sort_values(ascending=False).index.tolist()
labels = ["고위험", "중위험", "저위험"]

grade_map = {}
for i, c in enumerate(order):
    try: grade_map[int(float(c))] = labels[i]
    except: pass

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# 🪐 오로라 성간 감성의 5기 티니핑 스토리 멘트로 교체
RISK_STYLE = {
    "고위험": ("fact-high", "🪐🚨 [스타핑 비상! 고위험]", "초롱핑 궤도 추적기에 적색 신호 포착! 얕고 강력한 지진 오우라가 밤하늘을 흔들고 있어츄! 대피 배리어가 필요해!"),
    "중위험": ("fact-mid", "💫⚡ [초롱핑 감지! 중위험]", "초롱핑 돋보기 스캔 통과! 지진파 스펙트럼이 은하 너머 아주 깊은 곳에서 요동쳐 충격이 분산되고 있어!"),
    "저위험": ("fact-low", "✨💚 [반짝안전! 오로라존]", "오로라 커튼이 춤추는 영롱하고 안전한 구역이야츄! 지동 주파수가 아주 예쁘고 고르게 흐르고 있어!"),
    "판단보류": ("fact-none", "🌌💫 [고요구역! 판단보류]", "슈팅스타팩트 레이더에 잡히지 않는 태고의 고요가 깃든 신비로운 성간 암흑 구역이야!"),
}

# ═════════════════════════════════════════════════════════════
# 사이드바 제어 패널
# ═════════════════════════════════════════════════════════════
st.sidebar.markdown("### 🛰️ 오로라 주파수 다이얼")
st.sidebar.caption("Season 5 Space Radar Device")
K_NEIGHBORS = st.sidebar.slider("🔮 동기화할 오로라 노드 (k)", 5, 50, 25)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background: rgba(138, 43, 226, 0.1); padding: 15px; border-radius: 15px; border: 1px solid #00ffcc;">
        <span style="color: #00ffcc; font-weight: bold;">🌌 오로라 궤도 관측 대원</span><br>
        <small style="color: #e0ffff;">• 초롱핑 (오로라 홀로그램 투사)</small><br>
        <small style="color: #ba55d3;">• 스타핑 (우주 좌표 안착 지원)</small><br>
        <small style="color: #ff69b4;">• 하츄핑 (하트 링크 동기화)</small>
    </div>
    """,
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════
# 메인 스크린 헤더
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="shooting-star-header">
        <h1>✨ CATCH! TEENIEPING: AURORA SPACE FACT ✨</h1>
        <p>초롱핑의 입체 홀로그램 장치 개방! 오로라 주파수 궤도를 동기화하여 지진파 성단을 정밀 추적합니다츄!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 입력창 레이아웃
# ═════════════════════════════════════════════════════════════
st.subheader("🎯 오로라 레이더 크로스헤어 좌표 조준")
c1, c2 = st.columns(2)
with c1:
    lat = st.number_input("🔮 팩트 타겟 위도 (Latitude)", -90.0, 90.0, 36.5, 0.1, format="%.4f")
with c2:
    lon = st.number_input("🌌 팩트 타겟 경도 (Longitude)", -180.0, 180.0, 127.5, 0.1, format="%.4f")

# ═════════════════════════════════════════════════════════════
# 마법 레이더 스캔 시작 (오류 유발 코드 완벽 제거 및 정상 구동)
# ═════════════════════════════════════════════════════════════
if st.button("🪐 슈팅스타 팩트 오픈! 오로라 스캔 스타트!!", type="primary", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    order = np.argsort(dist)[:K_NEIGHBORS]
    near = df.iloc[order].copy()
    nearest_km = float(dist[order[0]])

    if nearest_km > 1800:
        grade = "판단보류"
        dom_cluster = None
    else:
        cw = {}
        for idx, c_val in zip(order, near["cluster"].values):
            try:
                c_key = int(float(c_val))
                cw[c_key] = cw.get(c_key, 0.0) + 1.0 / (dist[idx] + 30.0)
            except: continue
        
        if cw:
            dom_cluster = int(max(cw, key=cw.get))
            grade = grade_map.get(dom_cluster, "저위험")
        else:
            grade = "저위험"
            dom_cluster = 0

    css_cls, emoji, desc = RISK_STYLE[grade]
    st.divider()

    # 슈팅스타 레이더 매직 스크린 디스플레이
    r1, r2 = st.columns([1.1, 1.3])
    with r1:
        sub = f"🌌 오로라 스타 궤도 {dom_cluster}번 성단 연결 완료" if dom_cluster is not None else "🪐 미확인 딥스페이스 매트릭스"
        # ⚠️ 존재하지 않던 오타 매개변수 'Total_Formatter=True' 제거로 오류 근본적 해결!
        st.markdown(
            f'<div class="risk-badge {css_cls}">{emoji}<br>'
            f'<span style="font-size:15px; font-weight:bold; letter-spacing:1px; opacity:0.95;">{sub}</span></div>',
            unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f'<div class="fact-card"><b>🛰️ 초롱핑의 오로라 정밀 홀로그램 피드</b><br><span style="color:#d1f7ff;">{desc}</span><br><br>'
            f'<b>📍 조준 은하 좌표</b> : 위도 {lat:.4f}°, 경도 {lon:.4f}°<br>'
            f'<b>☄️ 최접근 에너지 코어 궤도</b> : {nearest_km:,.1f} km</div>',
            unsafe_allow_html=True,
        )

    # 마법 계기판 메트릭 배치
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("🔮 레이더 주파수 규모", f"Mag {near['규모'].mean():.2f}")
    m2.metric("🌌 오로라 관측 진원깊이", f"{near['진원깊이'].mean():.0f} km")
    m3.metric("✨ 초롱핑 동기화 영향도", f"{near['영향도'].mean():.1f} 핑")

    # ═════════════════════════════════════════════════════════════
    # 🛰️ 3D 입체 우주 홀로그램 가상 돔 (Pydeck 사이언 오로라 커스텀)
    # ═════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔮 슈팅스타팩트 3D 입체 오로라 스크린 (초롱핑 홀로그램 뷰)")
    
    show = df.sample(min(4500, len(df)), random_state=42).copy()
    
    # 핑크 비중을 낮추고 민트/아쿠아/네온옐로우 오로라 빛 컬러 매핑 선언
    COLOR_MAP = {"고위험": [255, 69, 0, 220], "중위험": [255, 215, 0, 220], "저위험": [0, 255, 204, 220]}
    show['color'] = show['cluster'].map(lambda c: COLOR_MAP.get(grade_map.get(int(c)), [138, 43, 226, 170]))

    # 타겟 포인트 주위로 퍼져나가는 '네온 그린 오로라 홀로그램 파동' 링 레이어
    ring_data = []
    for r_factor in [0.6, 1.2, 2.2]:
        ring_data.append({"lon": lon, "lat": lat, "radius": r_factor * 150000})
    ring_df = pd.DataFrame(ring_data)
    target_df = pd.DataFrame([{"lon": lon, "lat": lat}])

    layers = [
        # 1. 성간 지진 데이터 플롯 (별가루)
        pdk.Layer(
            'ScatterplotLayer',
            data=show,
            get_position='[경도, 위도]',
            get_color='color',
            get_radius=50000,
            pickable=True
        ),
        # 2. 초롱핑의 홀로그램 파동 테두리 레이어
        pdk.Layer(
            'ScatterplotLayer',
            data=ring_df,
            get_position='[lon, lat]',
            get_radius='radius',
            get_color=[0, 255, 204, 60],
            filled=False,
            stroked=True,
            line_width_min_pixels=2
        ),
        # 3. 팩트 골드 스타 크로스헤어 조준점
        pdk.Layer(
            'ScatterplotLayer',
            data=target_df,
            get_position='[lon, lat]',
            get_color=[255, 255, 0, 255],
            get_radius=220000,
            filled=True,
            stroked=True,
            line_width_min_pixels=4,
            get_line_color=[138, 43, 226, 255] # 퍼플 오로라 오라 테두리
        )
    ]

    # SF 우주선 콕핏 뷰 시점 각도 (Pitch=60, Bearing=20)
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=2.8,
        pitch=60,
        bearing=20
    )

    # 딥 다크 내비게이션 스타일로 변경하여 완전히 우주선 스크린 홀로그램 느낌 극대화
    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/navigation-night-v1',
        tooltip={"text": "티니핑 성간 코드: {cluster}\n위도: {위도}°\n경도: {경도}°"}
    )
    st.pydeck_chart(r)

    # ═════════════════════════════════════════════════════════════
    # 2D 매직스타 레이더 주파수 도표 (Matplotlib 딥 사이언 커스텀)
    # ═════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 오로라 레이더 주파수 분석 트랙")
    
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#050014')
    ax.set_facecolor('#0d002d')
    
    # 폰트 깨짐 예방 및 글자 오로라 네온 톤 커스텀
    ax.tick_params(colors='#d1f7ff', labelsize=10)
    ax.xaxis.label.set_color('#00ffcc')
    ax.yaxis.label.set_color('#00ffcc')
    ax.title.set_color('#00ffcc')

    HEX_MAP = {"고위험": "#ff4500", "중위험": "#ffd700", "저위험": "#00ffcc"}
    for c in sorted(df["cluster"].unique()):
        s_ = df[df["cluster"] == c]
        g_name = grade_map.get(int(c), "저위험")
        ax.scatter(s_["경도"], s_["위도"], s=9, alpha=0.6,
                   color=HEX_MAP.get(g_name, "#8a2be2"),
                   label=f"오로라 노드 {c} ({g_name})")
                   
    # 황금 별 모양 타겟 센터 마킹 
    ax.scatter(lon, lat, c="#ffff00", s=380, marker="*",
                edgecolors="#00ffcc", linewidths=2.5, zorder=10, label="팩트 조준점")
               
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("스페이스 경도 (Longitude)", fontsize=11, fontweight='bold')
    ax.set_ylabel("스페이스 위도 (Latitude)", fontsize=11, fontweight='bold')
    ax.set_title("✨ 슈팅스타 레이더 오로라 주파수 도표 ✨", fontsize=14, fontweight='bold', pad=15)
    
    legend = ax.legend(fontsize=10, loc="lower left")
    plt.setp(legend.get_texts(), color='white')
    legend.get_frame().set_facecolor('#050014')
    legend.get_frame().set_edgecolor('#8a2be2')
    legend.get_frame().set_linewidth(1.5)
    
    ax.grid(color='#8a2be2', alpha=0.2, linestyle=':')
    st.pyplot(fig)
    plt.close(fig)
