import os
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 방지 장치 (시스템 기본 고딕 폰트 강제 매핑)
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
            except Exception: pass
    return "sans-serif"

KOREAN_FONT = setup_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🖼️ 사진 속 디자인 100% 똑같이 구현하는 CSS (에러 해결 완료)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@500;700;900&display=swap');

    /* 전체 배경: 사진 속 신비로운 은은한 파스텔 오로라 은하수 */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #a3bded 0%, #eaf0ff 50%, #dad6ec 100%) !important;
        color: #333355 !important;
    }}
    .stMainBlockContainer {{
        background: radial-gradient(circle at 10% 20%, rgba(186, 201, 255, 0.4) 0%, transparent 50%),
                    radial-gradient(circle at 90% 80%, rgba(255, 214, 231, 0.4) 0%, transparent 50%);
        padding: 20px 40px !important;
    }}

    /* 1. 최상단 메인 타이틀 바 (사진 속 그 하얀 투명 둥근 박스) */
    .photo-top-header {{
        background: rgba(255, 255, 255, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 20px;
        padding: 15px 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(190, 196, 224, 0.2);
        margin-bottom: 30px;
        backdrop-filter: blur(10px);
    }}
    .photo-top-header h1 {{
        margin: 0;
        font-size: 26px;
        font-weight: 800;
        color: #4a4375;
        letter-spacing: -0.5px;
    }}

    /* 2. 타겟 좌표 표시창 (중앙 상단의 미니멀 파스텔 타질 인풋창 스타일) */
    .coord-display-box {{
        background: linear-gradient(90deg, #ffe3ec, #e3f2fd);
        border-radius: 30px;
        padding: 8px 25px;
        text-align: center;
        font-weight: 700;
        color: #4b5563;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        border: 2px solid #ffffff;
        margin-bottom: 5px;
    }}

    /* 3. 메인 작업 레이아웃 공간 (사진과 똑같은 가로 배치 구성) */
    .hologram-stage {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        position: relative;
        width: 100%;
        min-height: 500px;
        margin-top: 20px;
    }}

    /* 4. [좌측] 사진 속 열려있는 슈팅스타팩트 완구 하드웨어 구현 */
    .star-fact-device-body {{
        position: relative;
        width: 320px;
        height: 380px;
        background: radial-gradient(circle at center, #fff6f9 0%, #f3e5f5 60%, #e1bee7 100%);
        border: 10px solid #ffffff;
        border-radius: 100px 100px 80px 80px;
        box-shadow: -15px 20px 40px rgba(120, 100, 150, 0.3), inset -5px -5px 15px rgba(0,0,0,0.1);
        transform: perspective(800px) rotateY(25deg) rotateX(10deg);
    }}
    /* 기기 왼쪽의 금빛 날개 엠블럼 데코 조형 */
    .star-fact-device-body::before {{
        content: "";
        position: absolute;
        left: -35px;
        top: 30%;
        width: 45px;
        height: 120px;
        background: linear-gradient(135deg, #ffe082, #ffb300);
        border-radius: 50px 10px 10px 50px;
        box-shadow: -5px 5px 10px rgba(0,0,0,0.15);
    }}
    /* 하단 열린 키패드 액정 화면 */
    .star-fact-inner-lcd {{
        position: absolute;
        bottom: 25px;
        left: 20px;
        right: 20px;
        height: 160px;
        background: linear-gradient(180deg, #110e29 0%, #1a237e 100%);
        border: 4px solid #ce93d8;
        border-radius: 20px;
        box-shadow: inset 0 0 20px rgba(0,255,221,0.5);
        padding: 10px;
        color: #80deea;
        font-family: 'Orbit', sans-serif;
        font-size: 11px;
    }}

    /* 5. 홀로그램 투사 빛줄기 효과 (f-string 단일 중괄호 버그 원인 해결) */
    .hologram-light-beam {{
        position: absolute;
        left: 180px;
        bottom: 120px;
        width: 300px;
        height: 320px;
        background: linear-gradient(45deg, rgba(179, 157, 219, 0.4) 0%, rgba(128, 222, 234, 0.2) 60%, transparent 100%);
        clip-path: polygon(0% 100%, 30% 100%, 100% 0%, 30% 0%);
        pointer-events: none;
        z-index: 1;
        animation: pulseBeam 3s infinite alternate;
    }}
    @keyframes pulseBeam {{
        0% {{ opacity: 0.6; }}
        100% {{ opacity: 0.9; }}
    }}

    /* 6. [중앙] 공중에 붕 떠있는 3D 홀로그램 원형 지구본 프레임 */
    .globe-floating-container {{
        position: absolute;
        left: 360px;
        top: 30px;
        width: 420px;
        height: 420px;
        border-radius: 50%;
        background: radial-gradient(circle at center, transparent 40%, rgba(128, 222, 234, 0.15) 70%, transparent 100%);
        box-shadow: 0 0 50px rgba(186, 104, 200, 0.2);
        z-index: 2;
        padding: 10px;
    }}

    /* 7. [하단] 지진 위험군 매칭 텍스트 보드 */
    .photo-bottom-card {{
        background: rgba(255, 255, 255, 0.8);
        border: 2px solid #fff;
        border-radius: 20px;
        padding: 20px 25px;
        color: #444;
        box-shadow: 0 10px 30px rgba(150, 150, 180, 0.15);
        margin-top: 30px;
    }}
    .photo-bottom-card h3 {{
        margin-top: 0;
        font-size: 19px;
        color: #2b2b52;
    }}
    .danger-tag {{
        font-weight: 900;
        padding: 3px 10px;
        border-radius: 10px;
        color: white;
    }}
    .tag-high {{ background: #ff7675; }}
    .tag-mid {{ background: #ffeaa7; color: #555; }}
    .tag-low {{ background: #55efc4; color: #222; }}

    /* 실행 스캔 버튼 스타일 */
    .stButton>button {{
        background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%) !important;
        color: #4a4375 !important;
        font-weight: 800 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 6px 20px rgba(166, 193, 238, 0.4) !important;
        padding: 10px 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [지진 위험군 분석] 데이터 처리 시스템
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
# 레이아웃 배치 시작
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#75759e; font-size:14px; margin-top:4px; font-weight:600;">(오로라 스페이스 팩트 지진 위험군 탐지 시스템)</div>
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
    near_df = df.iloc[near_idx]
    nearest_km = float(dist[near_idx[0]])
    
    cw = {}
    for idx in near_idx:
        c_val = int(df.iloc[idx]["cluster"])
        cw[c_val] = cw.get(c_val, 0.0) + 1.0 / (dist[idx] + 10.0)
    dom_cluster = int(max(cw, key=cw.get))
    final_grade = grade_map.get(dom_cluster, "저위험군")

    col_left_stage, col_right_graph = st.columns([7, 5])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 스크린")
        
        st.markdown(
            f"""
            <div class="hologram-stage">
                <div class="star-fact-device-body">
                    <div class="star-fact-inner-lcd">
                        <span style="color:#fff; font-weight:700;">[FACT SYSTEM]</span><br>
                        SYS.STATUS: ONLINE<br>
                        LAT SYNC : {lat:.2f}<br>
                        LON SYNC : {lon:.2f}<br>
                        CORE_DIST: {nearest_km:.1f}km<br>
                        <span style="color:#ff8a80;">⚠ RISK_LEVEL DETECTED</span>
                    </div>
                </div>
                <div class="hologram-light-beam"></div>
                <div class="globe-floating-container">
            """, 
            unsafe_allow_html=True
        )
        
        show_df = df.sample(min(1200, len(df)), random_state=42).copy()
        PASTEL_COLOR = {
            "고위험군": [255, 118, 117, 210],
            "중위험군": [255, 234, 167, 210],
            "저위험군": [85, 239, 196, 210]
        }
        show_df['color'] = show_df['cluster'].map(lambda c: PASTEL_COLOR.get(grade_map.get(int(c)), [200, 200, 200, 150]))
        
        layers = [
            pdk.Layer(
                'ScatterplotLayer',
                data=show_df,
                get_position='[경도, 위도]',
                get_color='color',
                get_radius=70000,
                pickable=True
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lon": lon, "lat": lat}]),
                get_position='[lon, lat]',
                get_color=[255, 242, 178, 255],
                get_radius=280000,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[212, 118, 170, 255]
            )
        ]
        
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=2.2, pitch=45, bearing=15),
            map_style='mapbox://styles/mapbox/light-v10',
            tooltip={"text": "지진분석군: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        # 렌더링 에러 해결: rgba 스트링을 Matplotlib이 이해하는 4원소 튜플 (R, G, B, Alpha)로 명시적 전환
        fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor='none')
        ax.set_facecolor((1, 1, 1, 0.4)) 
        
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#ffeaa7", "저위험군": "#55efc4"}
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=12, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        ax.scatter(lon, lat, c="#ffeaa7", s=300, marker="*", edgecolors="#e84393", linewidths=2, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#dcdde1', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel("타겟 경도", fontfamily=KOREAN_FONT, fontsize=10, color="#4f5d75")
        ax.set_ylabel("타겟 위도", fontfamily=KOREAN_FONT, fontsize=10, color="#4f5d75")
        ax.tick_params(colors='#4f5d75', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    # ═════════════════════════════════════════════════════════════
    # 하단 정보 텍스트 피드 창
    # ═════════════════════════════════════════════════════════════
    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3>🛸 <b>초롱핑의 오로라 정밀 지진 위험군 피드</b></h3>
            <p style="font-size:16px; font-weight:700;">
                [ ⚡ 격자 신호 감지 결과: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#555; line-height:1.6; font-size:14px;">
                선택하신 위도 {lat:.4f}°, 경도 {lon:.4f}° 좌표 반경 내부의 격자 진원 분포 신호를 연산한 결과, 
                가장 지배적인 데이터 요소가 지진 분포 밀집 집단인 <b>{final_grade}</b> 패턴과 정밀하게 매칭되는 것으로 분석되었습니다. 
                가장 인근에 기록된 실제 지진 중심 코어(진앙지)와의 거리는 약 <b>{nearest_km:,.1f} km</b> 입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
