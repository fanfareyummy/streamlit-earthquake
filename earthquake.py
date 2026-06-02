import os
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 및 ValueError 원천 차단 장치 (객체 반환형)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def setup_korean_font_object():
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
                return fm.FontProperties(fname=path)
            except Exception:
                pass
    return fm.FontProperties(family="sans-serif")

# 폰트 속성 객체를 전역으로 명확히 획득
font_prop = setup_korean_font_object()
font_prop.set_weight('bold')

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 슈팅스타팩트 실물 완벽 싱크로 & Pydeck 강제 크롭 원형 봉인 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@700;900&display=swap');

    /* 🌌 파스텔 오로라 우주 공간 배경 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #a6d5ff 0%, #d5f1fe 25%, #fbe3f1 50%, #fbe8d5 75%, #ffffff 100%) !important;
        color: #3b3a57 !important;
    }
    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.35) 0%, transparent 60%),
                    radial-gradient(circle at 85% 75%, rgba(255, 105, 180, 0.35) 0%, transparent 60%);
        padding: 30px 60px !important;
    }

    /* 최상단 헤더 타이틀 팩 */
    .photo-top-header {
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 26px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(165, 180, 252, 0.25);
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
    }
    .photo-top-header h1 { margin: 0; font-size: 26px; font-weight: 900; color: #4c4475; }

    /* 🔮 [구조 개혁] 슈팅스타팩트 및 3D 지도를 수용하는 마스터 입체 스테이지 */
    .shooting-star-factory-stage {
        position: relative;
        width: 440px;
        height: 440px;
        margin: 30px auto;
    }

    /* 🌟 실물 일러스트의 '대형 황금색 입체 별 모양 거치대 베이스' 완벽 구현 */
    .star-gold-pedestal-base {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 62% 35%, 98% 35%, 69% 57%, 80% 91%, 50% 72%, 20% 91%, 31% 57%, 2% 35%, 38% 35%);
        box-shadow: 0 15px 35px rgba(234, 179, 8, 0.4);
        border: 4px solid #ffffff;
        z-index: 1;
    }

    /* 👼 실물 양옆의 천사 날개 파츠 장식 기하학 드로잉 */
    .fact-wing-left-part {
        position: absolute;
        left: -35px;
        top: 165px;
        width: 110px;
        height: 85px;
        background: linear-gradient(-45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 110px 15px 110px 110px;
        transform: rotate(-12deg);
        box-shadow: -8px 10px 20px rgba(0,0,0,0.15);
        z-index: 2;
    }
    .fact-wing-right-part {
        position: absolute;
        right: -35px;
        top: 165px;
        width: 110px;
        height: 85px;
        background: linear-gradient(45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 15px 110px 110px 110px;
        transform: rotate(12deg);
        box-shadow: 8px 10px 20px rgba(0,0,0,0.15);
        z-index: 2;
    }

    /* 💖 화려한 컴팩트 본체 외부 핫핑크 하트 하우징 */
    .fact-pink-heart-shield {
        position: absolute;
        left: 45px;
        top: 45px;
        width: 350px;
        height: 350px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fbcfe8 45%, #ec4899 85%, #be185d 100%);
        border-radius: 50%;
        border: 10px solid #ffffff;
        box-shadow: 0 25px 55px rgba(236, 72, 153, 0.45);
        z-index: 3;
    }

    /* 🎀 본체 상단부의 핑크색 하트 왕관 리본 완장 리본 장식 */
    .fact-top-crown-ribbon {
        position: absolute;
        left: 50%;
        top: -5px;
        transform: translateX(-50%);
        width: 130px;
        height: 50px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 4px solid #ffffff;
        border-radius: 25px;
        z-index: 12;
        box-shadow: 0 5px 12px rgba(0,0,0,0.18);
    }

    /* 🔒 [초강력 솔루션] Pydeck 사각형 프레임을 무조건 동그라미로 깎아서 가두는 절대 링 바인더 */
    .map-inside-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5; /* 별 스탠드와 리본 프레임 내부 한가운데에 위치시킴 */
        background: #090521;
        clip-path: circle(50% at 50% 50%) !important; /* 웹킷 브라우저 및 iframe 유출 원천 차단 */
        -webkit-clip-path: circle(50% at 50% 50%) !important;
    }
    
    /* Streamlit이 생성하는 고유의 사각 위젯 래퍼를 한 번 더 원형 강제 제어 */
    .map-inside-binder > div, 
    .map-inside-binder [data-testid="stPydeckChart"], 
    .map-inside-binder iframe {
        border-radius: 50% !important;
        clip-path: circle(50% at 50% 50%) !important;
        width: 270px !important;
        height: 270px !important;
        overflow: hidden !important;
    }

    /* ✨ 3D 지구 홀로그램을 지켜주는 크리스탈 글래스 돔 렌즈막 코팅 */
    .fact-crystal-glass-lens {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50%;
        background: radial-gradient(circle 122px at center, transparent 94%, #ffffff 100%),
                    radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.45) 0%, rgba(128, 222, 234, 0.1) 50%, rgba(232, 121, 249, 0.25) 100%);
        border: 6px solid #fef08a; /* 내부 미니 골드 메탈 서클 라인 */
        box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.65), 0 0 25px rgba(232, 121, 249, 0.45);
        z-index: 8; /* 레이아웃상 지도 위를 완벽하게 감싸 보호하는 형태 */
        pointer-events: none; /* 지도의 회전/확대 마우스 드래깅 인풋을 그대로 투과 */
    }

    /* 하단 알림판 가독성 디자인 */
    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.85);
        border: 2px solid #ffffff;
        border-radius: 24px;
        padding: 24px 30px;
        box-shadow: 0 15px 40px rgba(148, 163, 184, 0.15);
        margin-top: 25px;
        backdrop-filter: blur(10px);
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
# 📊 [데이터 가동 알고리즘 파트]
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
# 레이아웃 노드 드로잉
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

    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 내부 3D 홀로그램 투사")
        
        # HTML 팩트 완구 하우징 조립 프레임 출력
        st.markdown(
            f"""
            <div class="shooting-star-factory-stage">
                <div class="star-gold-pedestal-base"></div>
                <div class="fact-wing-left-part"></div>
                <div class="fact-wing-right-part"></div>
                <div class="fact-pink-heart-shield"></div>
                <div class="fact-top-crown-ribbon"></div>
                <div class="fact-crystal-glass-lens"></div>
                
                <div class="map-inside-binder">
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
                get_line_color=[219, 39, 119, 255] # [완벽 교정] 오타 문자열 '21db' 완전 박멸 완료! 순수 정수 리터럴 매핑
            )
        ]
        
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
        
        # [완벽 교정] fontproperties 에 더 이상 문자열이 아닌 올바른 폰트 객체 font_prop을 직접 전달하여 ValueError 원인 제거!
        ax.set_xlabel("타겟 경도", fontproperties=font_prop, fontsize=11, color="#334155")
        ax.set_ylabel("타겟 위도", fontproperties=font_prop, fontsize=11, color="#334155")
        
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
