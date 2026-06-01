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
# 💖 캐치! 티니핑 시즌 5: 스타티니핑 & 슈팅스타팩트 러블리 우주 디자인 (CSS)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="반짝반짝! 슈팅스타팩트 지진 레이더", page_icon="⭐", layout="wide")

st.markdown(
    """
    <style>
    /* 전체 배경: 러블리 가득한 밤하늘 은하수 테마 */
    .main {
        background: radial-gradient(circle at center, #2a0845 0%, #170022 60%, #050014 100%);
        color: #ffffff;
    }
    
    /* 슈팅스타팩트 최상단 마법 엠블럼 헤더 */
    .shooting-star-header {
        background: linear-gradient(135deg, #ff7ebb 0%, #b259ff 50%, #61dbff 100%);
        padding: 35px;
        border-radius: 30px;
        color: #fff;
        text-align: center;
        box-shadow: 0 0 35px rgba(255, 105, 180, 0.7), inset 0 0 15px rgba(255, 255, 255, 0.4);
        border: 4px solid #fff0f5;
        margin-bottom: 25px;
    }
    .shooting-star-header h1 {
        margin: 0;
        font-size: 38px;
        font-weight: 900;
        font-family: 'Malgun Gothic', sans-serif;
        text-shadow: 0 0 10px #fff, 0 0 20px #ff1493, 0 0 30px #ff69b4;
    }
    .shooting-star-header p {
        margin: 12px 0 0;
        font-size: 17px;
        color: #ffffdd;
        font-weight: bold;
        text-shadow: 0 0 5px #000;
    }
    
    /* 스타팩트 하트/별빛 마법 등급 배지 */
    .risk-badge {
        display: inline-block;
        padding: 22px;
        border-radius: 25px;
        color: #fff;
        font-size: 26px;
        font-weight: 900;
        width: 100%;
        text-align: center;
        border: 3px dashed #ffffff;
        box-shadow: 0 0 25px rgba(255, 255, 255, 0.3);
    }
    .fact-high { background: radial-gradient(circle, #ff2a7f, #b30047); box-shadow: 0 0 30px #ff2a7f; text-shadow: 0 0 8px #fff; }
    .fact-mid  { background: radial-gradient(circle, #ffaa00, #b36600); box-shadow: 0 0 30px #ffaa00; text-shadow: 0 0 8px #fff; }
    .fact-low  { background: radial-gradient(circle, #00ffcc, #008066); box-shadow: 0 0 30px #00ffcc; text-shadow: 0 0 8px #000; }
    .fact-none { background: radial-gradient(circle, #a1a8bd, #51586b); box-shadow: 0 0 20px #a1a8bd; }
    
    /* 스타핑의 홀로그램 비밀 카드 상자 */
    .fact-card {
        background: rgba(45, 15, 65, 0.75);
        border: 3px solid #ff7ebb;
        border-radius: 22px;
        padding: 22px;
        color: #fff;
        box-shadow: inset 0 0 20px rgba(255, 126, 187, 0.4), 0 0 20px rgba(97, 219, 255, 0.3);
    }
    .fact-card b { color: #ffb3d1; font-size: 18px; text-shadow: 0 0 5px rgba(255,179,209,0.5); }
    
    /* 귀염뽀짝 메트릭 스크린 (계기판) */
    div[data-testid="stMetric"] {
        background: rgba(255, 126, 187, 0.12);
        border: 2px solid #ff7ebb;
        border-radius: 20px;
        padding: 18px;
        box-shadow: 0 0 15px rgba(255, 126, 187, 0.3);
        text-align: center;
    }
    div[data-testid="stMetric"] label { color: #ffebf2 !important; font-weight: bold; font-size: 15px; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #61dbff !important; font-weight: 900; font-size: 26px !important; text-shadow: 0 0 8px rgba(97,219,255,0.6); }
    
    /* 버튼 스타일 수정 (핑크빛 마법 발사 버튼) */
    .stButton>button {
        background: linear-gradient(90deg, #ff69b4, #da70d6) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        border: 2px solid #fff !important;
        box-shadow: 0 0 15px #ff69b4 !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px #ff1493 !important;
    }
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

# K-Means 클러스터링 기반 다지기
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

# 에러 원천 차단: 클러스터 컬럼 정수화 처리 및 결측치 제거
df = df.dropna(subset=["cluster", "위도", "경도"])
df["cluster"] = df["cluster"].astype(int)

# 위험 등급 매핑 연산
agg = df.groupby("cluster")[FEATURES].mean()
score = ((agg["규모"] - agg["규모"].min()) / (agg["규모"].max() - agg["규모"].min() + 1e-5) * 0.5 + 
         (1 - (agg["진원깊이"] - agg["진원깊이"].min()) / (agg["진원깊이"].max() - agg["진원깊이"].min() + 1e-5)) * 0.5)
order = score.sort_values(ascending=False).index.tolist()
labels = ["고위험", "중위험", "저위험"]

# 여기서 예외처리 강화하여 확실하게 딕셔너리 구축
grade_map = {}
for i, c in enumerate(order):
    try:
        grade_map[int(float(c))] = labels[i]
    except:
        pass

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# 스타티니핑 맞춤형 귀염뽀짝 코멘트
RISK_STYLE = {
    "고위험": ("fact-high", "💖🚨 [쿠릉발생! 고위험]", "슈팅스타 파워 적색경보츄! 강력하고 얕은 지진 에너지가 뭉쳐있어 주의가 필요해!!"),
    "중위험": ("fact-mid", "💛⚡ [찌릿감지! 중위험]", "지진 에너지가 감지되었지만 진원이 깊어서 스타 아우라가 충격을 막아주고 있어!"),
    "저위험": ("fact-low", "💚✨ [반짝안전! 저위험]", "우주 별빛처럼 안전하고 평화로운 구역이야! 안심해도 좋아츄~"),
    "판단보류": ("fact-none", "🤍💫 [빛바램! 판단보류]", "스타팩트 반경 내에 지진 기록이 거의 없는 아주 신비롭고 고요한 곳이야!"),
}

# ═════════════════════════════════════════════════════════════
# 사이드바 컨트롤러 (매직 스타 다이얼 스타일)
# ═════════════════════════════════════════════════════════════
st.sidebar.markdown("### 🔮 스타팅 제어 패널")
st.sidebar.caption("시즌 5 스타티니핑 시스템")
K_NEIGHBORS = st.sidebar.slider("동기화할 별빛 노드 개수 (k)", 5, 50, 20)
st.sidebar.markdown("---")
st.sidebar.markdown("💟 **오늘의 요원 파트너**\n- 하츄핑, 빛나핑, 반짝핑과 함께 행성을 지켜요!")

# ═════════════════════════════════════════════════════════════
# 메인 헤더
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="shooting-star-header">
        <h1>💫 캐치! 티니핑 5기 : 슈팅스타 스페이스 팩트 💫</h1>
        <p>반짝이는 우주 에너지 개방! 지정 좌표를 탐색하여 전 세계 지진을 귀엽고 안전하게 분석합니다츄!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 입력창 : 하트 패드 조작
# ═════════════════════════════════════════════════════════════
st.subheader("🎯 스타 레이더 조준점 입력")
c1, c2 = st.columns(2)
with c1:
    lat = st.number_input("💖 하트 위도 (Latitude)", -90.0, 90.0, 37.5, 0.1, format="%.3f")
with c2:
    lon = st.number_input("💙 스타 경도 (Longitude)", -180.0, 180.0, 127.0, 0.1, format="%.3f")

# ═════════════════════════════════════════════════════════════
# 🪄 마법 레이더 스캔 시작 (예측 및 에러 전면 수정)
# ═════════════════════════════════════════════════════════════
if st.button("🪄 슈팅스타 팩트 오픈! 마법 스캔 스타트!!", type="primary", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    order = np.argsort(dist)[:K_NEIGHBORS]
    near = df.iloc[order].copy()
    nearest_km = float(dist[order[0]])

    if nearest_km > 1500:
        grade = "판단보류"
        dom_cluster = None
    else:
        # 가중치 딕셔너리 연산 에러 완전 해결 코드
        cw = {}
        for idx, c_val in zip(order, near["cluster"].values):
            try:
                c_key = int(float(c_val)) # 정수 변환 안전장치
                cw[c_key] = cw.get(c_key, 0.0) + 1.0 / (dist[idx] + 50.0)
            except:
                continue
        
        if cw:
            dom_cluster = int(max(cw, key=cw.get))
            grade = grade_map.get(dom_cluster, "저위험")
        else:
            grade = "저위험"
            dom_cluster = 0

    css_cls, emoji, desc = RISK_STYLE[grade]
    st.divider()

    # 슈팅스타팩트 LCD 연출 화면
    r1, r2 = st.columns([1.1, 1.3])
    with r1:
        sub = f"🔮 스페이스 스타 링크 {dom_cluster}번 연결 완료" if dom_cluster is not None else "🪐 미지의 스타더스트 영역"
        st.markdown(
            f'<div class="risk-badge {css_cls}">{emoji}<br>'
            f'<span style="font-size:14px; font-weight:bold; opacity:0.9;">{sub}</span></div>',
            unsafe_allow_html=True,
        )
    with r2:
        st.markdown(
            f'<div class="fact-card"><b>💫 슈팅스타 비밀 분석 레포트</b><br>{desc}<br><br>'
            f'<b>📍 팩트 조준 좌표</b> : 위도 {lat:.3f}°, 경도 {lon:.3f}°<br>'
            f'<b>🛸 가장 가까운 지진 에너지 코어</b> : {nearest_km:,.1f} km</div>',
            unsafe_allow_html=True,
        )

    # 하단 팅팅탱탱 계기판 메트릭
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("⭐ 레이더 평균 규모", f"Mag {near['규모'].mean():.1f}")
    m2.metric("🪐 우주 진원 깊이", f"{near['진원깊이'].mean():.0f} km")
    m3.metric("✨ 반짝반짝 영향도", f"{near['영향도'].mean():.1f} 핑")

    # ═════════════════════════════════════════════════════════════
    # 🛰️ 3D 입체 우주 홀로그램 스크린 (Pydeck 지구본 테마)
    # ═════════════════════════════════════════════════════════════
    st.subheader("🔮 슈팅스타팩트 3D 홀로그램 스크린 (우주 위성 뷰)")
    
    show = df.sample(min(4000, len(df)), random_state=42).copy()
    
    # 컬러맵도 티니핑 5기 우주 컬러로 블렌딩 (고위험: 핫핑크, 중위험: 옐로우골드, 저위험: 민트아쿠아)
    COLOR_MAP = {"고위험": [255, 20, 147, 195], "중위험": [255, 215, 0, 195], "저위험": [0, 255, 204, 195]}
    show['color'] = show['cluster'].map(lambda c: COLOR_MAP.get(grade_map.get(int(c)), [200, 200, 200, 150]))

    target_df = pd.DataFrame([{"lon": lon, "lat": lat}])

    layers = [
        # 지진 데이터 플롯 (은하수 별빛 가루)
        pdk.Layer(
            'ScatterplotLayer',
            data=show,
            get_position='[경도, 위도]',
            get_color='color',
            get_radius=45000,
            pickable=True
        ),
        # 팩트 조준 포인트 (빛나는 황금 왕별 크로스헤어 효과)
        pdk.Layer(
            'ScatterplotLayer',
            data=target_df,
            get_position='[lon, lat]',
            get_color=[255, 255, 0, 255],
            get_radius=180000,
            filled=True,
            stroked=True,
            line_width_min_pixels=4,
            get_line_color=[255, 20, 147, 255] # 핑크 오라 테두리
        )
    ]

    # 우주 시점 극대화 각도 (Pitch=55, Bearing=15)
    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=2.6,
        pitch=55,
        bearing=15
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10', # 딥 다크 스페이스 테마
        tooltip={"text": "티니핑 스타 등급: {cluster}\n위도: {위도}°\n경도: {경도}°"}
    )
    st.pydeck_chart(r)

    # ═════════════════════════════════════════════════════════════
    # 2D 매직스타 레이더 산점도
    # ═════════════════════════════════════════════════════════════
    st.subheader("📊 스타 레이더 트래킹 맵")
    fig, ax = plt.subplots(figsize=(12, 5), facecolor='#170022')
    ax.set_facecolor('#2a0845')
    
    if KOREAN_FONT:
        mpl.rc('font', family=KOREAN_FONT)
        
    ax.tick_params(colors='#ffebf2')
    ax.xaxis.label.set_color('#ffebf2')
    ax.yaxis.label.set_color('#ffebf2')
    ax.title.set_color('#ff7ebb')

    HEX_MAP = {"고위험": "#ff1493", "중위험": "#ffd700", "저위험": "#00ffcc"}
    for c in sorted(df["cluster"].unique()):
        s_ = df[df["cluster"] == c]
        g_name = grade_map.get(int(c), "저위험")
        ax.scatter(s_["경도"], s_["위도"], s=7, alpha=0.5,
                   color=HEX_MAP.get(g_name, "#ffffff"),
                   label=f"스타 노드 {c} ({g_name})")
                   
    # 황금 별 모양 타겟 마킹 
    ax.scatter(lon, lat, c="#ffff00", s=350, marker="*",
               edgecolors="#ff1493", linewidths=2.5, zorder=5, label="팩트 조준점")
               
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("스페이스 경도 (Longitude)")
    ax.set_ylabel("스페이스 위도 (Latitude)")
    ax.set_title("✨ 슈팅스타 레이더 주파수 도표 ✨", fontsize=13, fontweight='bold')
    
    legend = ax.legend(fontsize=9, loc="lower left")
    plt.setp(legend.get_texts(), color='white')
    legend.get_frame().set_facecolor('#170022')
    legend.get_frame().set_edgecolor('#ff7ebb')
    
    ax.grid(color='#ff7ebb', alpha=0.15)
    st.pyplot(fig)
    plt.close(fig)
