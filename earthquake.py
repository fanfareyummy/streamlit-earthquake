import os
import tempfile
import urllib.request
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 🌌 구글 폰트 실시간 다운로드로 한글 깨짐 근본적 차단 (Matplotlib용)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_google_korean_font():
    mpl.rcParams["axes.unicode_minus"] = False
    url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ttf") as f:
            f.write(urllib.request.urlopen(url).read())
            font_path = f.name
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        mpl.rc("font", family=font_name)
        return font_name
    except Exception as e:
        for backup in ["Malgun Gothic", "Apple SD Gothic Neo", "NanumGothic"]:
            if backup in [f.name for f in fm.fontManager.ttflist]:
                mpl.rc("font", family=backup)
                return backup
    return "sans-serif"

KOREAN_FONT = load_google_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# ✨ 캐치! 티니핑 시즌 5: 슈팅스타팩트 파스텔 오로라 디바이스 테마 (CSS)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 오로라 레이더", page_icon="🪐", layout="wide")

# 모든 단일 중괄호 {}를 f-string 문법에 맞게 이중 중괄호 {{}}로 수정 완료
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@700&family=Orbit&family=Pretendard:wght@500;800&display=swap');

    /* 글로벌 폰트 및 파스텔톤 은하수 오로라 배경 애니메이션 패널 */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Pretendard', 'Orbit', sans-serif !important;
        background: linear-gradient(125deg, #161032 0%, #1a1c4b 35%, #152e46 70%, #1b1235 100%) !important;
        color: #e3f2fd !important;
    }}
    
    /* 파스텔 오로라 성운 필터 */
    .stMainBlockContainer {{
        background: radial-gradient(circle at 20% 30%, rgba(200, 160, 255, 0.15) 0%, transparent 45%),
                    radial-gradient(circle at 80% 70%, rgba(150, 230, 220, 0.15) 0%, transparent 50%);
    }}

    /* 최상단 슈팅스타 오로라 팩트 타이틀 박스 */
    .shooting-star-header {{
        background: rgba(255, 255, 255, 0.05);
        padding: 30px;
        border-radius: 30px;
        text-align: center;
        box-shadow: 0 10px 40px rgba(162, 126, 255, 0.15);
        border: 2px solid rgba(200, 180, 255, 0.25);
        margin-bottom: 35px;
        backdrop-filter: blur(20px);
    }}
    .shooting-star-header h1 {{
        margin: 0;
        font-size: 36px;
        font-weight: 800;
        color: #e8dbff;
        font-family: 'Orbit', sans-serif;
        text-shadow: 0 0 15px rgba(216, 191, 216, 0.6), 0 0 30px rgba(176, 224, 230, 0.4);
    }}
    .shooting-star-header p {{
        margin: 10px 0 0;
        font-size: 16px;
        color: #bce9ff;
        font-weight: 500;
    }}
    
    /* ⭐ 슈팅스타팩트 실물 디바이스 프레임 구현 ⭐ */
    .star-fact-hardware-case {{
        background: radial-gradient(circle at center, #fdf6ff 0%, #ecd3ff 70%, #d8b4f8 100%);
        border: 8px solid #ffffff;
        border-radius: 40px;
        padding: 25px;
        box-shadow: 0 25px 60px rgba(0,0,0,0.4), inset 0 0 20px rgba(255,255,255,0.8), 0 0 30px rgba(186, 85, 211, 0.2);
        margin: 20px auto;
        position: relative;
    }}
    
    .star-fact-hardware-case::before {{
        content: "⭐ SHOOTING STAR FACT ⭐";
        display: block;
        text-align: center;
        font-weight: 900;
        font-size: 13px;
        color: #b584e6;
        letter-spacing: 3px;
        margin-bottom: 12px;
        text-shadow: 0 1px 2px #fff;
    }}

    /* 팩트 내부 LCD 스크린 (3D 맵 투사존) */
    .star-fact-lcd-screen {{
        background: #090620 !important;
        border: 4px solid #b584e6;
        border-radius: 25px;
        padding: 10px;
        box-shadow: inset 0 0 30px rgba(0, 255, 204, 0.2), 0 0 20px rgba(162, 126, 255, 0.4);
        overflow: hidden;
    }}
    
    /* 파스텔 레이더 결과 배지 정보창 */
    .risk-badge {{
        display: inline-block;
        padding: 22px;
        border-radius: 22px;
        color: #fff;
        font-size: 24px;
        font-weight: 800;
        width: 100%;
        text-align: center;
        border: 2px dashed rgba(255, 255, 255, 0.4);
    }}
    .fact-high {{ background: linear-gradient(135deg, #ff9ebb, #d476aa); box-shadow: 0 8px 25px rgba(255, 158, 187, 0.4); }}
    .fact-mid  {{ background: linear-gradient(135deg, #ffcf87, #d99c4c); box-shadow: 0 8px 25px rgba(255, 207, 135, 0.4); }}
    .fact-low  {{ background: linear-gradient(135deg, #a6ffea, #5cbfa6); box-shadow: 0 8px 25px rgba(166, 255, 234, 0.4); color: #22443a !important; }}
    .fact-none {{ background: linear-gradient(135deg, #c2ccde, #828fa3); box-shadow: 0 8px 20px rgba(194, 204, 222, 0.3); }}
    
    /* 초롱핑의 오로라 비밀 아카이브 보드 */
    .fact-card {{
        background: rgba(255, 255, 255, 0.04);
        border: 2px solid rgba(182, 140, 255, 0.3);
        border-radius: 22px;
        padding: 22px;
        color: #e1f0fa;
        box-shadow: inset 0 0 15px rgba(255, 255, 255, 0.03), 0 10px 30px rgba(0,0,0,0.15);
    }}
    .fact-card b {{ color: #b6ffd4; font-size: 17px; }}
    
    /* 파스텔 계기판 미터 스크린 */
    div[data-testid="stMetric"] {{
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(166, 255, 234, 0.2);
        border-radius: 18px;
        padding: 18px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        text-align: center;
    }}
    div[data-testid="stMetric"] label {{ color: #cbdced !important; font-weight: 600; font-size: 14px; }}
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{ color: #a6ffea !important; font-weight: 800; font-size: 24px !important; text-shadow: 0 0 10px rgba(166,255,234,0.4); }}
    
    /* 파스텔 파동 스캔 개방 버튼 */
    .stButton>button {{
        background: linear-gradient(90deg, #bda2ff, #ffb3db, #a2f0ff) !important;
        color: #312252 !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        border-radius: 20px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 8px 25px rgba(189, 162, 255, 0.3) !important;
        transition: all 0.3s ease;
        padding: 12px 0px !important;
    }}
    .stButton>button:hover {{
        transform: scale(1.01);
        box-shadow: 0 12px 35px rgba(255, 179, 219, 0.5) !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 샘플 데이터 셋 가동 체계
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    num_samples = 4000
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

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)
df = df.dropna(subset=["cluster", "위도", "경도"])
df["cluster"] = df["cluster"].astype(int)

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

RISK_STYLE = {
    "고위험": ("fact-high", "🌸🚨 [스타핑 레이더 대기경보]", "초롱핑의 궤도 반경에 흔들림 에너지가 집중 감지되었어츄! 스타 더스트 방어벽을 활성화해야 해!"),
    "중위험": ("fact-mid", "🌟⚡ [초롱핑 스캔! 아우라 전파]", "지파 에너지가 확인되었지만 심해성 성간 흐름이라 지상 버블 실드가 안전하게 흡수해 주는 중이야!"),
    "저위험": ("fact-low", "✨💚 [반짝안전! 파스텔 가든]", "포근하고 평화로운 별빛 가루가 은은하게 도는 안심 성층권 구역이야츄~ 자유롭게 탐사해도 좋아!"),
    "판단보류": ("fact-none", "🌌💫 [안개지대! 은하 외곽]", "슈팅스타팩트 미답사 은하 구역으로, 고요하고 신비로운 무중력 상태가 유지되는 곳이야!"),
}

# ═════════════════════════════════════════════════════════════
# 사이드바 콘솔
# ═════════════════════════════════════════════════════════════
st.sidebar.markdown("### 🪐 오로라 싱크 팩트 조절")
K_NEIGHBORS = st.sidebar.slider("🔮 주파수 매핑 별빛 노드 (k)", 5, 50, 20)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div style="background: rgba(255,255,255,0.03); padding: 12px; border-radius: 12px; border: 1px solid rgba(200, 180, 255, 0.2);">
        <small style="color: #cbdced;"><b>🛰️ 궤도 안착 동기화 요원</b><br>
        • 초롱핑 (오로라 스크린 동기화)<br>
        • 스타핑 (매직 스타 게이트 개방)<br>
        • 하츄핑 (러블리 우주 파동 매칭)</small>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="shooting-star-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT ✨</h1>
        <p>파스텔빛 은하수 오로라 주파수를 맞춰 지정한 성간 좌표의 흔들림을 아기자기하게 포착합니다츄!</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 좌표 입력 패드
# ═════════════════════════════════════════════════════════════
st.subheader("🎯 슈팅스타 레이더 오로라 크로스헤어 타겟점")
cx, cy = st.columns(2)
with cx:
    lat = st.number_input("🌸 타겟 위도 좌표 (Latitude)", -90.0, 90.0, 36.5, 0.1, format="%.4f")
with cy:
    lon = st.number_input("🌌 타겟 경도 좌표 (Longitude)", -180.0, 180.0, 127.5, 0.1, format="%.4f")

# ═════════════════════════════════════════════════════════════
# 팩트 메인 구동 프로세스
# ═════════════════════════════════════════════════════════════
if st.button("🪐 슈팅스타 팩트 오픈! 파스텔 오로라 스캔 스타트!!", type="primary", use_container_width=True):
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
                cw[c_key] = cw.get(c_key, 0.0) + 1.0 / (dist[idx] + 25.0)
            except: continue
        
        if cw:
            dom_cluster = int(max(cw, key=cw.get))
            grade = grade_map.get(dom_cluster, "저위험")
        else:
            grade = "저위험"
            dom_cluster = 0

    css_cls, emoji, desc = RISK_STYLE[grade]
    st.divider()

    # 상단 요약 피드 대시보드
    r1, r2 = st.columns([1.1, 1.3])
    with r1:
        sub = f"🌌 오로라 스타 궤도 {dom_cluster}번 채널 안착" if dom_cluster is not None else "🪐 미탐사 안개 성운 영역"
        st.markdown(
            f'<div class="risk-badge {css_cls}">{emoji}<br>'
            f'<span style="font-size:14px; font-weight:700; letter-spacing:0.5px; opacity:0.9;">{sub}</span></div>',
            unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f'<div class="fact-card"><b>🛰️ 초롱핑의 오로라 정밀 홀로그램 피드</b><br><span style="color:#e2f0fd;">{desc}</span><br><br>'
            f'<b>📍 조준 팩트 성간 은하축</b> : 위도 {lat:.4f}°, 경도 {lon:.4f}°<br>'
            f'<b>☄️ 가장 가까운 지동 진원 에너지 코어</b> : {nearest_km:,.1f} km</div>',
            unsafe_allow_html=True,
        )

    # 파스텔 스크린 계기판 메트릭스
    st.markdown("<br>", unsafe_allow_html=True)
    m1, m2, m3 = st.columns(3)
    m1.metric("🔮 은하 오로라 평균 규모", f"Mag {near['규모'].mean():.2f}")
    m2.metric("🌌 내부 레이더 진원깊이", f"{near['진원깊이'].mean():.0f} km")
    m3.metric("✨ 초롱핑 궤도 동기화 영향도", f"{near['영향도'].mean():.1f} 핑")

    # ═════════════════════════════════════════════════════════════
    # 🛰️ 슈팅스타팩트 하드웨어 프레임 안에 3D 지형 투사
    # ═════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🔮 슈팅스타팩트 홀로그램 LCD 입체 투사 기믹 스크린")
    
    st.markdown('<div class="star-fact-hardware-case"><div class="star-fact-lcd-screen">', unsafe_allow_html=True)
    
    show = df.sample(min(3500, len(df)), random_state=42).copy()
    
    PASTEL_COLOR_MAP = {
        "고위험": [255, 160, 180, 180], 
        "중위험": [255, 220, 140, 180], 
        "저위험": [140, 240, 220, 180]
    }
    show['color'] = show['cluster'].map(lambda c: PASTEL_COLOR_MAP.get(grade_map.get(int(c)), [190, 180, 240, 130]))

    ring_data = [{"lon": lon, "lat": lat, "radius": rf * 160000} for rf in [0.7, 1.4, 2.3]]
    ring_df = pd.DataFrame(ring_data)
    target_df = pd.DataFrame([{"lon": lon, "lat": lat}])

    layers = [
        pdk.Layer(
            'ScatterplotLayer',
            data=show,
            get_position='[경도, 위도]',
            get_color='color',
            get_radius=55000,
            pickable=True
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=ring_df,
            get_position='[lon, lat]',
            get_radius='radius',
            get_color=[160, 230, 255, 60],
            filled=False,
            stroked=True,
            line_width_min_pixels=2
        ),
        pdk.Layer(
            'ScatterplotLayer',
            data=target_df,
            get_position='[lon, lat]',
            get_color=[255, 235, 150, 255],
            get_radius=240000,
            filled=True,
            stroked=True,
            line_width_min_pixels=3,
            get_line_color=[240, 160, 220, 255]
        )
    ]

    view_state = pdk.ViewState(
        latitude=lat,
        longitude=lon,
        zoom=2.6,
        pitch=45,
        bearing=10
    )

    r = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_style='mapbox://styles/mapbox/dark-v10',
        tooltip={"text": "성간 주파수 등급: {cluster}\n위도: {위도}°\n경도: {경도}°"}
    )
    st.pydeck_chart(r)
    st.markdown('</div></div>', unsafe_allow_html=True)

    # ═════════════════════════════════════════════════════════════
    # 2D 매직스타 레이더 주파수 도표 (폰트 완벽 적용)
    # ═════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 초롱핑의 오로라 주파수 추적 궤도 도표")
    
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor='#161032')
    ax.set_facecolor('#1a1c4b')
    
    ax.tick_params(colors='#bce9ff', labelsize=10)
    ax.xaxis.label.set_color('#bda2ff')
    ax.yaxis.label.set_color('#bda2ff')
    ax.title.set_color('#e8dbff')

    HEX_MAP = {"고위험": "#ff9ebb", "중위험": "#ffcf87", "저위험": "#a6ffea"}
    for c in sorted(df["cluster"].unique()):
        s_ = df[df["cluster"] == c]
        g_name = grade_map.get(int(c), "저위험")
        ax.scatter(s_["경도"], s_["위도"], s=10, alpha=0.55,
                   color=HEX_MAP.get(g_name, "#b8b2d6"),
                   label=f"성간 오로라 노드 {c} ({g_name})")
                   
    ax.scatter(lon, lat, c="#fff2b2", s=380, marker="*",
                edgecolors="#d476aa", linewidths=2.5, zorder=12, label="팩트 조준점")
               
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    ax.set_xlabel("스페이스 경도 축 (Longitude)", fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_ylabel("스페이스 위도 축 (Latitude)", fontsize=11, fontweight='bold', fontfamily=KOREAN_FONT)
    ax.set_title("✨ 슈팅스타 레이더 파스텔 오로라 주파수 도표 ✨", fontsize=14, fontweight='bold', pad=15, fontfamily=KOREAN_FONT)
    
    legend = ax.legend(fontsize=10, loc="lower left")
    plt.setp(legend.get_texts(), color='white', fontfamily=KOREAN_FONT)
    legend.get_frame().set_facecolor('#161032')
    legend.get_frame().set_edgecolor('#bda2ff')
    legend.get_frame().set_linewidth(1.2)
    
    ax.grid(color='#bda2ff', alpha=0.12, linestyle='--')
    st.pyplot(fig)
    plt.close(fig)
