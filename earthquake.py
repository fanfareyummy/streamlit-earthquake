import os
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 및 에러 없는 초간단 안전 한글 설정 (ValueError 완전 해결)
# ═════════════════════════════════════════════════════════════
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 5기 실물 싱크로: 3D 지도를 중심에 가두는 입체 레이어 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@700;900&display=swap');

    /* 🌌 우주 오로라 배경 무드 */
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

    /* 🔮 슈팅스타팩트 레이아웃 제어실 (지도를 감싸는 마스터 컨테이너) */
    .shooting-star-factory-stage {
        position: relative;
        width: 360px;
        height: 360px;
        margin: 40px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* ⭐ 완구 실물의 대형 황금색 별 스탠드 베이스 (지도의 한참 아래 z-index: 1 배치) */
    .star-gold-pedestal-base {
        position: absolute;
        width: 440px;
        height: 440px;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 62% 35%, 98% 35%, 69% 57%, 80% 91%, 50% 72%, 20% 91%, 31% 57%, 2% 35%, 38% 35%);
        box-shadow: 0 15px 35px rgba(234, 179, 8, 0.3);
        border: 4px solid #ffffff;
        z-index: 1; 
        pointer-events: none;
    }

    /* 👼 양옆의 천사 날개 파츠 (z-index: 2) */
    .fact-wing-left-part {
        position: absolute;
        left: -80px;
        top: 130px;
        width: 120px;
        height: 90px;
        background: linear-gradient(-45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 120px 15px 120px 120px;
        transform: rotate(-12deg);
        box-shadow: -8px 10px 20px rgba(0,0,0,0.12);
        z-index: 2;
        pointer-events: none;
    }
    .fact-wing-right-part {
        position: absolute;
        right: -80px;
        top: 130px;
        width: 120px;
        height: 90px;
        background: linear-gradient(45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 15px 120px 120px 120px;
        transform: rotate(12deg);
        box-shadow: 8px 10px 20px rgba(0,0,0,0.12);
        z-index: 2;
        pointer-events: none;
    }

    /* 💖 지도를 원형으로 크롭해서 정확하게 중앙 안쪽에 가두는 링 가이드 시스템 (z-index: 4) */
    .map-inside-binder {
        position: relative;
        width: 280px;
        height: 280px;
        border-radius: 50% !important;
        overflow: hidden !important;
        background: #090521;
        box-shadow: 0 0 30px rgba(236, 72, 153, 0.5);
        z-index: 4;
        clip-path: circle(50% at 50% 50%) !important;
        -webkit-clip-path: circle(50% at 50% 50%) !important;
    }
    
    /* Streamlit이 임의로 생성해 밀어넣는 Pydeck 고유 사각 뷰포트를 강제로 동그라미 처리 */
    .map-inside-binder [data-testid="stPydeckChart"],
    .map-inside-binder .stPydeckChart,
    .map-inside-binder div,
    .map-inside-binder canvas {
        border-radius: 50% !important;
        clip-path: circle(50% at 50% 50%) !important;
        -webkit-clip-path: circle(50% at 50% 50%) !important;
        width: 280px !important;
        height: 280px !important;
        overflow: hidden !important;
    }

    /* 🛡️ 지도 위를 포근하게 덮어서 일체감을 주는 오로라 크리스탈 투명 글래스 돔 (z-index: 6) */
    .fact-crystal-glass-lens {
        position: absolute;
        width: 284px;
        height: 284px;
        border-radius: 50%;
        background: radial-gradient(circle 125px at center, transparent 93%, #ffffff 100%),
                    radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.4) 0%, rgba(128, 222, 234, 0.08) 50%, rgba(232, 121, 249, 0.2) 100%);
        border: 8px solid #ec4899; /* 실물 본체의 핫핑크 하트 내부 링 질감 */
        box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.6), 0 0 25px rgba(232, 121, 249, 0.4);
        z-index: 6; 
        pointer-events: none; /* 이 막이 있어도 아래쪽 지도를 마우스로 돌릴 수 있도록 관통 */
    }

    /* 🎀 본체 맨 상단 중앙에 자리한 미니 골드 리본 크라운 마크 (z-index: 8) */
    .fact-top-crown-ribbon {
        position: absolute;
        top: -45px;
        width: 110px;
        height: 40px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 3.5px solid #ffffff;
        border-radius: 20px;
        z-index: 8;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
        pointer-events: none;
    }

    /* 상황 하단 안내판 디자인 */
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
# 📊 [데이터 분석 코어 코딩 파트]
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
# 메인 노드 화면 렌더링 시작
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
        
        # [구조 혁명] 팩트 프레임을 레이어로 나누어 지도 위아래에 정교하게 배치
        st.markdown(
            f"""
            <div class="shooting-star-factory-stage">
                <div class="star-gold-pedestal-base"></div>
                <div class="fact-wing-left-part"></div>
                <div class="fact-wing-right-part"></div>
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
                get_line_color=[219, 39, 119, 255]
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
        
        # 폰트 에러를 방지하고 한글을 깔끔하게 표기하는 안정화된 Matplotlib 드로잉 코드
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
        
        # 한글 깨짐 및 객체 충돌 없이 문자열 설정 반영 완료
        ax.set_xlabel("타겟 경도", fontsize=11, color="#334155", fontweight='bold')
        ax.set_ylabel("타겟 위도", fontsize=11, color="#334155", fontweight='bold')
        ax.set_title("지진 관측 데이터 매트릭스", fontsize=12, color="#1e1b4b", fontweight='bold')
        
        legend_obj = ax.legend(loc='upper right', framealpha=0.7)
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
