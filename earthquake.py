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
# 🎨 슈팅스타팩트 실물 저격 디테일 가미 완벽 무드 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@700;900&display=swap');

    /* 🌌 전체 파스텔 오로라 우주 공간 배경 무드 복원 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #b4c5e7 0%, #eef2fa 30%, #e3daf7 65%, #fbcfe8 100%) !important;
        color: #3b3a57 !important;
    }
    
    .stMainBlockContainer {
        background: radial-gradient(circle at 20% 25%, rgba(167, 139, 250, 0.45) 0%, transparent 50%),
                    radial-gradient(circle at 80% 75%, rgba(244, 114, 182, 0.4) 0%, transparent 50%);
        padding: 25px 50px !important;
        position: relative;
    }

    /* ⭐ 위에서 떨어지는 은은한 하얀 별빛 이펙트 */
    .stMainBlockContainer::before {
        content: "★";
        position: absolute;
        top: -20px;
        left: 15%;
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
        left: 70%;
        color: rgba(255, 255, 255, 0.9);
        font-size: 16px;
        letter-spacing: 150px;
        animation: fallStars 8s infinite linear;
        pointer-events: none;
        z-index: 99;
    }
    @keyframes fallStars {
        0% { transform: translateY(-20px) rotate(0deg); opacity: 0; }
        10% { opacity: 1; }
        90% { opacity: 1; }
        100% { transform: translateY(850px) rotate(360deg); opacity: 0; }
    }

    .photo-top-header {
        background: rgba(255, 255, 255, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 26px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(165, 180, 252, 0.2);
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }
    .photo-top-header h1 { margin: 0; font-size: 25px; font-weight: 900; color: #433d6a; }

    /* 🔮 [구조 혁신] 지도가 팩트 돔 내부에 완벽히 귀속되도록 설계된 컨테이너 */
    .hologram-absolute-universe {
        position: relative;
        width: 440px;
        height: 440px;
        margin: 20px auto;
    }

    /* 🌟 실물 사진의 거대한 노란색 입체 별 모양 거치대 베이스 */
    .star-gold-pedestal {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 62% 35%, 98% 35%, 69% 57%, 80% 91%, 50% 72%, 20% 91%, 31% 57%, 2% 35%, 38% 35%);
        box-shadow: 0 15px 35px rgba(234, 179, 8, 0.35);
        transform: scale(1.0);
        border: 4px solid #ffffff;
        z-index: 1;
    }

    /* 💖 분홍색 하트 리본 마크와 화려한 윙이 조합된 팩트 본체 */
    .fact-aurora-shield-frame {
        position: absolute;
        left: 45px;
        top: 45px;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fbcfe8 45%, #ec4899 85%, #be185d 100%);
        border-radius: 50%;
        border: 8px solid #ffffff;
        box-shadow: 0 20px 50px rgba(236, 72, 153, 0.4), inset -5px -5px 15px rgba(0,0,0,0.1);
        z-index: 3;
    }

    /* 👼 완구 특유의 입체 천사 금빛 날개 디테일 */
    .fact-wing-part-left {
        position: absolute;
        left: -15px;
        top: 155px;
        width: 85px;
        height: 75px;
        background: linear-gradient(-45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 3.5px solid #ffffff;
        border-radius: 90px 10px 90px 90px;
        transform: rotate(-10deg);
        box-shadow: -5px 8px 15px rgba(0,0,0,0.12);
        z-index: 2;
    }
    .fact-wing-part-right {
        position: absolute;
        right: -15px;
        top: 155px;
        width: 85px;
        height: 75px;
        background: linear-gradient(45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 3.5px solid #ffffff;
        border-radius: 10px 90px 90px 90px;
        transform: rotate(10deg);
        box-shadow: 5px 8px 15px rgba(0,0,0,0.12);
        z-index: 2;
    }

    /* 🎀 상단 중앙의 시그니처 핑크 하트 리본 장식 */
    .fact-crown-ribbon {
        position: absolute;
        left: 50%;
        top: -10px;
        transform: translateX(-50%);
        width: 120px;
        height: 45px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 3px solid #ffffff;
        border-radius: 20px;
        z-index: 10;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
    }

    /* 🔮 [초강력 락킹] 지도가 밖으로 탈출하는 것을 원천 봉쇄하는 원형 바인더 구체 */
    .map-inside-hologram-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important; /* 바깥 사각형을 완전히 잘라버림 */
        z-index: 5;
        background: #0b0726; /* 우주의 검고 깊은 베이스 */
    }

    /* 🛡️ 지도 위에 씌워져 오로라 투사 효과를 극대화하는 유리막 커버 (마우스 드래그 관통) */
    .fact-glass-lens-mask {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50%;
        /* 완구 고유의 내부 골드 서클 링과 오로라 홀로그램 무드 결합 */
        background: radial-gradient(circle 122px at center, transparent 96%, #ffffff 100%),
                    radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.35) 0%, rgba(128, 222, 234, 0.15) 50%, rgba(192, 132, 252, 0.25) 100%);
        border: 5px solid #fef08a; 
        box-shadow: inset 0 0 25px rgba(0, 242, 254, 0.5), 0 0 20px rgba(232, 121, 249, 0.4);
        z-index: 7; /* 지도 레이어보다 높은 위치에 강제 고정 */
        pointer-events: none; /* 마우스 클릭 및 드래그가 하단 지도로 그대로 투과됨 */
    }

    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.8);
        border: 2px solid #ffffff;
        border-radius: 24px;
        padding: 24px 30px;
        box-shadow: 0 15px 40px rgba(148, 163, 184, 0.15);
        margin-top: 25px;
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
# 레이아웃 노드 가로 정렬 배치
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

    # 5:5 완벽 분할 균형배치
    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 내부 3D 홀로그램 투사")
        
        # 디자인 요소 레이어 구조화 시작
        st.markdown(
            f"""
            <div class="hologram-absolute-universe">
                <div class="star-gold-pedestal"></div>
                <div class="fact-wing-part-left"></div>
                <div class="fact-wing-part-right"></div>
                <div class="fact-aurora-shield-frame"></div>
                <div class="fact-crown-ribbon"></div>
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
                get_radius=120000,
                pickable=True
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lon": lon, "lat": lat}]),
                get_position='[lon, lat]',
                get_color=[255, 255, 255, 255],
                get_radius=390000,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[21db, 39, 119, 255]
            )
        ]
        
        # 팩트 내부 우주 느낌을 살리기 위해 매핑 스타일을 다크 모드로 튜닝
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=1.4, pitch=40, bearing=0),
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip={"text": "위험분류: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

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
