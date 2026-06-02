import os
import urllib.request
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 및 서버 다운 원천 차단 (웹폰트 실시간 다운로드)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_korean_font_secure():
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
    os.makedirs(font_dir, exist_ok=True)
    font_path = os.path.join(font_dir, "NanumGothic.ttf")
    
    if not os.path.exists(font_path):
        url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
        try:
            urllib.request.urlretrieve(url, font_path)
        except Exception:
            return fm.FontProperties()
            
    try:
        fm.fontManager.addfont(font_path)
        prop = fm.FontProperties(fname=font_path)
        mpl.rcParams["font.family"] = prop.get_name()
        mpl.rcParams["axes.unicode_minus"] = False
        return prop
    except:
        return fm.FontProperties()

font_prop = load_korean_font_secure()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# ✨ [SyntaxError 해결] f-string 버그를 방지하기 위해 일반 문자열 처리
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit+Spaces&family=Pretendard:wght@700;900&display=swap');

    /* 🪐 파스텔톤 오로라 우주 무드 배경 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #cbd5e1 0%, #eef2fa 30%, #e0d7f7 65%, #fbcfe8 100%) !important;
        color: #3b3a57 !important;
    }
    
    .stMainBlockContainer {
        background: radial-gradient(circle at 20% 30%, rgba(167, 139, 250, 0.45) 0%, transparent 60%),
                    radial-gradient(circle at 80% 70%, rgba(244, 114, 182, 0.4) 0%, transparent 60%);
        padding: 30px 60px !important;
        position: relative;
    }

    /* 🌟 상단에서 은은하게 쏟아져 내리는 하얀 별빛 애니메이션 */
    .stMainBlockContainer::before {
        content: "★";
        position: absolute;
        top: -20px;
        left: 20%;
        color: rgba(255, 255, 255, 0.95);
        font-size: 14px;
        animation: fallStars 6s infinite linear;
        pointer-events: none;
        z-index: 99;
    }
    .stMainBlockContainer::after {
        content: "✧  ✨  ✦  *";
        position: absolute;
        top: -30px;
        left: 60%;
        color: rgba(255, 255, 255, 0.9);
        font-size: 16px;
        letter-spacing: 140px;
        animation: fallStars 9s infinite linear;
        pointer-events: none;
        z-index: 99;
    }
    @keyframes fallStars {
        0% { transform: translateY(-20px) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(850px) rotate(360deg); opacity: 0; }
    }

    /* 메인 타이틀 바 */
    .photo-top-header {
        background: rgba(255, 255, 255, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 26px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(165, 180, 252, 0.25);
        margin-bottom: 30px;
        backdrop-filter: blur(12px);
    }
    .photo-top-header h1 { margin: 0; font-size: 26px; font-weight: 900; color: #4c4475; }

    /* 🔮 슈팅스타팩트 입체 스테이지 구역 (중앙 배치) */
    .shooting-star-factory-stage {
        position: relative;
        width: 480px;
        height: 480px;
        margin: 20px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* ⭐ 배경에 깔리는 화려한 대형 입체 별 모양 스탠드 */
    .star-base-platform {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffd6e8 0%, #ffedd5 50%, #ccffd8 100%);
        clip-path: polygon(50% 0%, 61% 35%, 98% 35%, 68% 57%, 79% 91%, 50% 70%, 21% 91%, 32% 57%, 2% 35%, 39% 35%);
        box-shadow: 0 20px 50px rgba(147, 51, 234, 0.25);
        transform: scale(0.95);
        border: 4px solid #ffffff;
        z-index: 1;
    }

    /* 💖 분홍 리본 + 날개가 결합된 슈팅스타팩트 메인 구체 프레임 */
    .fact-main-sphere-body {
        position: absolute;
        width: 360px;
        height: 360px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fce7f3 40%, #f472b6 85%, #db2777 100%);
        border-radius: 50%;
        border: 10px solid #ffffff;
        box-shadow: 0 25px 60px rgba(219, 39, 119, 0.35), inset -6px -6px 20px rgba(0,0,0,0.08);
        z-index: 5;
    }

    /* 👼 양옆으로 펼쳐진 천사 금빛 날개 장식 */
    .fact-angel-wing-left {
        position: absolute;
        left: -50px;
        top: 120px;
        width: 90px;
        height: 100px;
        background: linear-gradient(-45deg, #fef08a 0%, #facc15 60%, #eab308 100%);
        border: 4px solid #ffffff;
        border-radius: 90px 10px 90px 90px;
        transform: rotate(-15deg);
        box-shadow: -5px 10px 20px rgba(0,0,0,0.15);
        z-index: 4;
    }
    .fact-angel-wing-right {
        position: absolute;
        right: -50px;
        top: 120px;
        width: 90px;
        height: 100px;
        background: linear-gradient(45deg, #fef08a 0%, #facc15 60%, #eab308 100%);
        border: 4px solid #ffffff;
        border-radius: 10px 90px 90px 90px;
        transform: rotate(15deg);
        box-shadow: 5px 10px 20px rgba(0,0,0,0.15);
        z-index: 4;
    }

    /* 🎀 정중앙을 장식하는 커다란 분홍색 리본 장식 */
    .fact-center-pink-ribbon {
        position: absolute;
        left: 50%;
        top: -15px;
        transform: translateX(-50%);
        width: 140px;
        height: 50px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 4px solid #ffffff;
        border-radius: 30px;
        z-index: 12;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    }

    /* 🗺️ [초핵심] 3D 지도를 구체 내부로 봉인하는 언더레이어 바인더 */
    .map-inside-hologram-binder {
        position: absolute;
        left: 35px;
        top: 35px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 8;
        background: #090521;
    }

    /* 🛡️ 지도가 사각형으로 튀어나오는 현상을 덮어서 물리적으로 삭제하는 오로라 유리 가림막 */
    .fact-glass-lens-mask {
        position: absolute;
        left: 35px;
        top: 35px;
        width: 270px;
        height: 270px;
        border-radius: 50%;
        /* 중심부는 투명하게 비춰 지도를 보여주고, 모서리 테두리는 마스킹 */
        background: radial-gradient(circle 125px at center, transparent 96%, #ffffff 100%),
                    radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.4) 0%, rgba(128, 222, 234, 0.2) 50%, rgba(192, 132, 252, 0.3) 100%);
        border: 6px solid #fef08a; /* 실물 기기의 금빛 내부 링 무드 반영 */
        box-shadow: inset 0 0 30px rgba(0, 242, 254, 0.55), 0 0 25px rgba(232, 121, 249, 0.5);
        z-index: 10; /* 지도 위에 배치됨 */
        pointer-events: none; /* 지도의 마우스 드래그 이벤트를 방해하지 않고 관통 */
    }

    /* 하단 피드 디자인 */
    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.8);
        border: 2px solid #ffffff;
        border-radius: 24px;
        padding: 24px 30px;
        box-shadow: 0 15px 40px rgba(148, 163, 184, 0.15);
        margin-top: 25px;
        backdrop-filter: blur(8px);
    }
    .danger-tag { font-weight: 900; padding: 5px 14px; border-radius: 12px; color: white; }
    .tag-high { background: #ff7675; }
    .tag-mid { background: #facc15; color: #333; }
    .tag-low { background: #4ade80; color: #111; }

    .stButton>button {
        background: linear-gradient(90deg, #fbcfe8 0%, #c7d2fe 100%) !important;
        color: #4338ca !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 6px 20px rgba(199, 210, 254, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [데이터 프로세싱 엔진]
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    np.random.seed(42)
    num_samples = 2000
    df = pd.DataFrame({
        '위도': np.random.uniform(-50, 50, num_samples),
        '경도': np.random.uniform(-160, 160, num_samples),
        '규모': np.random.uniform(1.5, 7.0, num_samples),
        '진원깊이': np.random.uniform(5, 550, num_samples),
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
    df = load_pure_quake_data()

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)
df = df.dropna(subset=["cluster", "위도", "경도"])

agg = df.groupby("cluster")[FEATURES].mean()
score = ((agg["규모"] - agg["규모"].min()) / (agg["규모"].max() - agg["규모"].min() + 1e-5) * 0.5 + 
         (1 - (agg["진원깊이"] - agg["진원깊이"].min()) / (agg["진원깊이"].max() - agg["진원깊이"].min() + 1e-5)) * 0.5)
order = score.sort_values(ascending=False).index.tolist()
labels = ["고위험군", "중위험군", "저위험군"]

grade_map = {}
for i, c in enumerate(order):
    grade_map[int(c)] = labels[i]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# ═════════════════════════════════════════════════════════════
# 메인 대시보드 레이아웃 시작
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#6b7280; font-size:13px; margin-top:5px; font-weight:700;">
            (오로라 스페이스 팩트 지진 위험군 정밀 분석 시스템)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🎯 레이더 타겟 크로스헤어 좌표 설정")
cx, cy = st.columns(2)
with cx:
    lat = st.number_input("💖 타겟 위도 (Latitude)", -90.0, 90.0, 36.5, step=0.1)
with cy:
    lon = st.number_input("🌌 타겟 경도 (Longitude)", -180.0, 180.0, 127.5, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🪐 슈팅스타 팩트 개방 및 지진 위험군 데이터 매핑 스캔 시작", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    near_idx = np.argsort(dist)[:20]
    nearest_km = float(dist[near_idx[0]])
    
    cw = {}
    for idx in near_idx:
        c_val = int(df.iloc[idx]["cluster"])
        cw[c_val] = cw.get(c_val, 0.0) + 1.0 / (dist[idx] + 10.0)
    dom_cluster = int(max(cw, key=cw.get))
    final_grade = grade_map.get(dom_cluster, "저위험군")

    # [좌측]: 슈팅스타팩트 실물 본체와 돔 내부 지도 / [우측]: 깔끔한 분석 차트
    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 내부 3D 홀로그램 투사")
        
        # 실물 팩트 모양의 HTML 스케일 조립 시작
        st.markdown(
            f"""
            <div class="shooting-star-factory-stage">
                <div class="star-base-platform"></div>
                <div class="fact-angel-wing-left"></div>
                <div class="fact-angel-wing-right"></div>
                <div class="fact-main-sphere-body">
                    <div class="fact-center-pink-ribbon"></div>
                    <div class="fact-glass-lens-mask"></div>
                    
                    <div class="map-inside-hologram-binder">
            """, 
            unsafe_allow_html=True
        )
        
        show_df = df.sample(min(1200, len(df)), random_state=42).copy()
        PASTEL_COLOR = {
            "고위험군": [255, 118, 117, 220],
            "중위험군": [250, 204, 21, 220],
            "저위험군": [74, 222, 128, 220]
        }
        show_df['color'] = show_df['cluster'].map(lambda c: PASTEL_COLOR.get(grade_map.get(int(c)), [180, 180, 220, 140]))
        
        layers = [
            pdk.Layer(
                'ScatterplotLayer',
                data=show_df,
                get_position='[경도, 위도]',
                get_color='color',
                get_radius=110000,
                pickable=True
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lon": lon, "lat": lat}]),
                get_position='[lon, lat]',
                get_color=[255, 255, 255, 255],
                get_radius=380000,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[219, 39, 119, 255]
            )
        ]
        
        # 팩트 내부 유리 구체의 질감을 살리기 위해 맵 스타일을 다크 테마로 세팅
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=1.5, pitch=40, bearing=0),
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip={"text": "위험분류: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor='none')
        ax.set_facecolor((1, 1, 1, 0.55)) 
        
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=18, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        ax.scatter(lon, lat, c="#ffffff", s=350, marker="*", edgecolors="#db2777", linewidths=2.5, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#cbd5e1', linestyle='-', linewidth=0.8)
        
        # 다운로드된 폰트 할당으로 리눅스 서버 한글 자막 깨짐 해결
        ax.set_xlabel("타겟 경도", fontproperties=font_prop, fontsize=11, color="#334155", fontweight='bold')
        ax.set_ylabel("타겟 위도", fontproperties=font_prop, fontsize=11, color="#334155", fontweight='bold')
        
        legend_obj = ax.legend(loc='upper right', framealpha=0.7)
        if legend_obj:
            for text in legend_obj.get_texts():
                text.set_fontproperties(font_prop)
                text.set_size(10)
            
        ax.tick_params(colors='#334155', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#1e1b4b;">🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#475569; line-height:1.7; font-size:14px; margin:0;">
                초롱핑 감지와 정밀 특성창 플로텍이 S도라 자역과 제나이서 오로라 암이 짜리를 만도할 수 있는 아로와 조람이 만든 <b>{final_grade}</b>을 말합니다.
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 위험 격자 스펙트럼 스캔이 완료되었으며, 인근 실제 지진 중심핵 코어 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
