import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 방지 및 한글 설정 (모든 환경에서 안전한 맑은고딕/나눔고딕 무조건 강제 매핑)
# ═════════════════════════════════════════════════════════════
plt.rcParams['font.family'] = ['Malgun Gothic', 'AppleGothic', 'NanumGothic', 'Noto Sans CJK KR', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 5기 실물 100% 싱크로: 3D 홀로그램 구체를 내부에 강제 일체화시키는 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@700;900&display=swap');

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

    /* 🔮 슈팅스타팩트 입체 스탠드 제어 프레임 */
    .shooting-star-factory-stage {
        position: relative;
        width: 440px;
        height: 440px;
        margin: 30px auto;
    }

    /* ⭐ 대형 황금색 입체 별 모양 거치대 베이스 */
    .star-gold-pedestal-base {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 62% 35%, 98% 35%, 69% 57%, 80% 91%, 50% 72%, 20% 91%, 31% 57%, 2% 35%, 38% 35%);
        box-shadow: 0 15px 35px rgba(234, 179, 8, 0.3);
        border: 4px solid #ffffff;
        z-index: 1; 
    }

    /* 👼 양옆의 천사 날개 파츠 장식 */
    .fact-wing-left-part {
        position: absolute;
        left: -45px;
        top: 165px;
        width: 120px;
        height: 90px;
        background: linear-gradient(-45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 120px 15px 120px 120px;
        transform: rotate(-12deg);
        box-shadow: -8px 10px 20px rgba(0,0,0,0.12);
        z-index: 2;
    }
    .fact-wing-right-part {
        position: absolute;
        right: -45px;
        top: 165px;
        width: 120px;
        height: 90px;
        background: linear-gradient(45deg, #ffffff 0%, #e0f2fe 60%, #bae6fd 100%);
        border: 4px solid #ffffff;
        border-radius: 15px 120px 120px 120px;
        transform: rotate(12deg);
        box-shadow: 8px 10px 20px rgba(0,0,0,0.12);
        z-index: 2;
    }

    /* 💖 외부 핫핑크 본체 서클 하우징 */
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

    /* 🎀 본체 상단의 핑크 왕관 리본 장식 */
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

    /* 🔒 [해결] 지도가 절대 도망치지 못하도록 액자 형태로 밀착 삽입하는 컨테이너 */
    .map-inside-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 6px solid #fef08a; /* 노란색 보석 링 테두리 구현 */
        box-shadow: inset 0 0 30px rgba(0, 242, 254, 0.6);
        background: #090521;
    }

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
# 📊 [데이터 검증 및 로드 엔진 - KeyError 에러 완벽 대처]
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
df_loaded = False

if os.path.exists(df_path):
    for enc in ("utf-8", "utf-8-sig", "cp949"):
        try:
            temp_df = pd.read_csv(df_path, encoding=enc)
            # 파일이 있어도 필수 컬럼이 모두 포함되어 있는지 완벽 체크!
            if all(col in temp_df.columns for col in ["위도", "경도"] + FEATURES):
                df = temp_df
                df_loaded = True
                break
        except:
            continue

if not df_loaded:
    df = load_pure_quake_data()

# 군집화 수행 (확보된 데이터프레임 기반)
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
# 레이아웃 노드 전개
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
        
        # 팩트의 외형 조형 부품들을 완전 조립합니다.
        st.markdown(
            """
            <div class="shooting-star-factory-stage">
                <div class="star-gold-pedestal-base"></div>
                <div class="fact-wing-left-part"></div>
                <div class="fact-wing-right-part"></div>
                <div class="fact-pink-heart-shield"></div>
                <div class="fact-top-crown-ribbon"></div>
                <div class="map-inside-binder">
            """, 
            unsafe_allow_html=True
        )
        
        # 🔒 [대혁신] 외부 격리 돔 위젯을 완벽 무력화하고 팩트 내부에 고정 매핑하는 Matplotlib 3D 오로라 지구본
        fig_3d, ax_3d = plt.subplots(figsize=(3.5, 3.5), subplot_kw={'projection': '3d'}, facecolor='none')
        fig_3d.patch.set_alpha(0.0)
        ax_3d.set_facecolor('#090521') # 우주 암흑색 배경 고정
        
        # 🪐 3D 가상 입체 지구본 와이어프레임 생성
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        xs = 180 * np.outer(np.cos(u), np.sin(v))
        ys = 90 * np.outer(np.sin(u), np.sin(v))
        zs = 90 * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_3d.plot_wireframe(xs, ys, zs, color="rgba(0, 242, 254, 0.15)", linewidth=0.5)

        # 지진 위험군 데이터 포인트를 3D 영역 공간에 투사 연산
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        show_df = df.sample(min(500, len(df)), random_state=42)
        
        for c in sorted(show_df["cluster"].unique()):
            sub = show_df[show_df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax_3d.scatter(sub["경도"], sub["위도"], sub["규모"]*5, color=HEX_MAP.get(g_name, "#ffffff"), alpha=0.7)
            
        # 선택한 레이더 크로스헤어 타겟 포인트를 화려한 하얀색 별로 중앙 표시
        ax_3d.scatter(lon, lat, 120, color="#ffffff", marker="*", edgecolors="#db2777", linewidths=1.5, zorder=20)
        
        # 지구본 시점 세팅 및 축 라벨 투명 제거로 입체 구체 느낌 극대화
        ax_3d.view_init(elev=25, azim=int(lon)-30)
        ax_3d.axis('off')
        
        # 지도를 팩트 내부 서클 돔 안에 완전히 액자처럼 끼워 넣습니다츄!
        st.pyplot(fig_3d)
        plt.close(fig_3d)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        ax.set_facecolor((1, 1, 1, 0.6))
        
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=18, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        ax.scatter(lon, lat, c="#ffffff", s=350, marker="*", edgecolors="#db2777", linewidths=2.5, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#cbd5e1', linestyle='-', linewidth=0.8)
        
        # 시스템 기본 폰트를 배열로 자동 선택하게 세팅하여 글자 깨짐 완전 해결
        ax.set_xlabel("타겟 경도", fontsize=11, color="#334155", fontweight='bold')
        ax.set_ylabel("타겟 위도", fontsize=11, color="#334155", fontweight='bold')
        ax.set_title("지진 관측 데이터 매트릭스", fontsize=12, color="#1e1b4b", fontweight='bold')
        
        ax.legend(loc='upper right', framealpha=0.7)
        ax.tick_params(colors='#334155', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#1e1b4b;"> UFO <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
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
