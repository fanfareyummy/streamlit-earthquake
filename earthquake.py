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
# 🎨 사진 속 홀로그램 원 + 디테일 팩트 완벽 구현 CSS (중괄호 이중 처리)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@500;700;900&display=swap');

    /* 전체 배경: 사진 속 은은한 오로라 배경 */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #b4c5e7 0%, #eef2fa 40%, #e3daf7 70%, #f7e3ef 100%) !important;
        color: #333355 !important;
    }}
    .stMainBlockContainer {{
        background: radial-gradient(circle at 15% 25%, rgba(195, 175, 255, 0.35) 0%, transparent 50%),
                    radial-gradient(circle at 85% 75%, rgba(255, 195, 220, 0.35) 0%, transparent 50%);
        padding: 20px 40px !important;
    }}

    /* 최상단 투명 둥근 타이틀 바 */
    .photo-top-header {{
        background: rgba(255, 255, 255, 0.55);
        border: 2px solid rgba(255, 255, 255, 0.7);
        border-radius: 24px;
        padding: 18px 30px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(180, 185, 215, 0.2);
        margin-bottom: 30px;
        backdrop-filter: blur(12px);
    }}
    .photo-top-header h1 {{
        margin: 0;
        font-size: 25px;
        font-weight: 900;
        color: #433d6a;
        letter-spacing: 0.5px;
    }}

    /* 메인 작업 스테이지 가로 배치 컨테이너 */
    .hologram-stage {{
        display: flex;
        position: relative;
        width: 100%;
        height: 480px;
        margin-top: 10px;
    }}

    /* ✨ [좌측] 슈팅스타팩트 실물 완구 하드웨어 초정밀 디테일 강화 */
    .star-fact-device-body {{
        position: absolute;
        left: 10px;
        bottom: 10px;
        width: 320px;
        height: 340px;
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #f6e8f7 40%, #e8c7ed 80%, #d5a3dd 100%);
        border: 12px solid #ffffff;
        border-radius: 90px 90px 75px 75px;
        box-shadow: -18px 22px 45px rgba(110, 90, 140, 0.35), 
                    inset -4px -4px 12px rgba(0,0,0,0.08),
                    0 0 25px rgba(232, 199, 237, 0.6);
        transform: perspective(900px) rotateY(28deg) rotateX(12deg);
        z-index: 2;
    }}
    
    /* 팩트 좌측 측면: 사진 속 입체적인 금빛 날개 조형 디테일 */
    .star-fact-device-body::before {{
        content: "";
        position: absolute;
        left: -42px;
        top: 25%;
        width: 50px;
        height: 130px;
        background: linear-gradient(135deg, #ffe082 0%, #ffa000 60%, #ff8f00 100%);
        border-radius: 60px 15px 15px 60px;
        border: 3px solid #ffffff;
        box-shadow: -6px 6px 14px rgba(90, 50, 120, 0.25);
    }}

    /* 팩트 우측 측면 내부 크림 옐로우 힌지 포인트 조형 고도화 */
    .star-fact-device-body::after {{
        content: "";
        position: absolute;
        right: -10px;
        top: 40%;
        width: 16px;
        height: 45px;
        background: #fff9db;
        border: 3px solid #fff;
        border-radius: 0 10px 10px 0;
    }}

    /* 팩트 하단 내부: 3줄 파동 메인 계기판 LCD 스크린 */
    .star-fact-inner-lcd {{
        position: absolute;
        bottom: 20px;
        left: 20px;
        right: 20px;
        height: 150px;
        background: linear-gradient(180deg, #0d0a21 0%, #161b54 100%);
        border: 4px solid #d09ee3;
        border-radius: 18px;
        box-shadow: inset 0 0 25px rgba(0,255,221,0.45), 0 0 15px rgba(208,158,227,0.4);
        padding: 12px;
        color: #80deea;
        font-family: 'Orbit', sans-serif;
        font-size: 11px;
        line-height: 1.6;
    }}

    /* ✨ [중앙] 핵심 수정: 사진처럼 팩트 LCD에서 대각선 위로 뻗어나가는 홀로그램 투사 빛줄기 */
    .hologram-light-beam {{
        position: absolute;
        left: 190px;
        bottom: 110px;
        width: 260px;
        height: 280px;
        background: linear-gradient(45deg, rgba(206, 147, 216, 0.45) 0%, rgba(128, 222, 234, 0.25) 50%, transparent 100%);
        clip-path: polygon(0% 100%, 35% 100%, 100% 0%, 40% 0%);
        pointer-events: none;
        z-index: 3;
    }}

    /* ✨ [핵심 원 기믹] 3D 지도를 완벽하게 가두어 둥실 띄우는 빛나는 '홀로그램 원형 구체' 컨테이너 */
    .globe-floating-sphere {{
        position: absolute;
        left: 350px;
        top: 20px;
        width: 380px;
        height: 380px;
        border-radius: 50% !important; /* 무조건 원형 유지 */
        overflow: hidden !important;   /* 중요: 내부 3D 지도가 사각형으로 튀어나오는 것 차단 */
        border: 6px solid rgba(255, 255, 255, 0.7);
        background: radial-gradient(circle at center, transparent 30%, rgba(128, 222, 234, 0.1) 70%, rgba(206, 147, 216, 0.2) 100%);
        box-shadow: 0 0 40px rgba(128, 222, 234, 0.5), 
                    0 0 70px rgba(218, 166, 247, 0.4),
                    inset 0 0 30px rgba(255, 255, 255, 0.6);
        z-index: 4;
        animation: floatSphere 3.5s infinite alternate ease-in-out;
    }}
    
    /* 홀로그램 구체가 공중에 기분 좋게 둥실둥실 떠오르는 애니메이션 효과 */
    @keyframes floatSphere {{
        0% {{ transform: translateY(0px); }}
        100% {{ transform: translateY(-12px); }}
    }}

    /* 내부 Pydeck 차트 프레임도 강제로 원형 박스 크기에 동기화 고정 */
    .globe-floating-sphere .stPydeckChart {{
        border-radius: 50% !important;
        overflow: hidden !important;
        width: 100% !important;
        height: 100% !important;
    }}

    /* [하단] 지진 위험군 종합 피드 정보 판넬 */
    .photo-bottom-card {{
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid #ffffff;
        border-radius: 22px;
        padding: 22px 28px;
        color: #444;
        box-shadow: 0 12px 35px rgba(140, 145, 175, 0.12);
        margin-top: 25px;
        backdrop-filter: blur(8px);
    }}
    .photo-bottom-card h3 {{
        margin-top: 0;
        font-size: 18px;
        color: #2b2b52;
    }}
    .danger-tag {{
        font-weight: 900;
        padding: 4px 12px;
        border-radius: 12px;
        color: white;
        font-size: 14px;
    }}
    .tag-high {{ background: #ff7675; box-shadow: 0 4px 10px rgba(255,118,117,0.3); }}
    .tag-mid {{ background: #ffeaa7; color: #555; box-shadow: 0 4px 10px rgba(255,234,167,0.3); }}
    .tag-low {{ background: #55efc4; color: #222; box-shadow: 0 4px 10px rgba(85,239,196,0.3); }}

    /* 인터페이스 버튼 스타일링 */
    .stButton>button {{
        background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%) !important;
        color: #4a4375 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 6px 18px rgba(166, 193, 238, 0.35) !important;
        padding: 11px 0 !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [본질 집중] 지진 위험군 클러스터링 코어 데이터 엔진
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
# 레이더 가동 화면 레이아웃
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#716b94; font-size:13px; margin-top:5px; font-weight:700; letter-spacing:0.3px;">
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

    # [좌측 복합 홀로그램 스테이지]와 [우측 격자 2D 그래프] 분할 배치
    col_left_stage, col_right_graph = st.columns([7, 5])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 원형 투사")
        
        # HTML 뼈대 생성 (팩트 바디 + 광선 + 원형 홀로그램 가두기 틀)
        st.markdown(
            f"""
            <div class="hologram-stage">
                <div class="star-fact-device-body">
                    <div class="star-fact-inner-lcd">
                        <span style="color:#e1f5fe; font-weight:900; letter-spacing:0.5px;">[STATION_ACTIVE]</span><br>
                        팩트동기화: ONLINE<br>
                        위도좌표축: {lat:.3f}°<br>
                        경도좌표축: {lon:.3f}°<br>
                        인근지동핵: {nearest_km:.1f}km<br>
                        <span style="color:#ffa726; font-weight:700;">🔮 HOLOGRAM SYSTEM READY</span>
                    </div>
                </div>
                
                <div class="hologram-light-beam"></div>
                
                <div class="globe-floating-sphere">
            """, 
            unsafe_allow_html=True
        )
        
        # 3D 피덱 지구본 세팅
        show_df = df.sample(min(1200, len(df)), random_state=42).copy()
        PASTEL_COLOR = {
            "고위험군": [255, 118, 117, 220],
            "중위험군": [255, 234, 167, 220],
            "저위험군": [85, 239, 196, 220]
        }
        show_df['color'] = show_df['cluster'].map(lambda c: PASTEL_COLOR.get(grade_map.get(int(c)), [180, 180, 220, 140]))
        
        layers = [
            pdk.Layer(
                'ScatterplotLayer',
                data=show_df,
                get_position='[경도, 위도]',
                get_color='color',
                get_radius=85000,
                pickable=True
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lon": lon, "lat": lat}]),
                get_position='[lon, lat]',
                get_color=[255, 238, 130, 255],
                get_radius=320000,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[232, 67, 147, 255]
            )
        ]
        
        # 둥근 구체 핏에 맞도록 피치 및 스케일 조정된 뷰 스테이트
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=2.0, pitch=40, bearing=12),
            map_style='mapbox://styles/mapbox/light-v10',
            tooltip={"text": "지진분석군: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        st.markdown("</div></div>", unsafe_allow_html=True) # 홀로그램 스테이지 태그 마감

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        # 2D 격자 차트 드로잉 (안정적인 투명 배경색 튜플 구조)
        fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor='none')
        ax.set_facecolor((1, 1, 1, 0.45)) 
        
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#ffeaa7", "저위험군": "#55efc4"}
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=14, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        # 사진과 매칭되는 골드 크로스헤어 타겟 별 표식 마킹
        ax.scatter(lon, lat, c="#ffeaa7", s=320, marker="*", edgecolors="#e84393", linewidths=2.5, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#dcdde1', linestyle='-', linewidth=0.8)
        
        ax.set_xlabel("타겟 경도", fontfamily=KOREAN_FONT, fontsize=10, color="#4f5d75")
        ax.set_ylabel("타겟 위도", fontfamily=KOREAN_FONT, fontsize=10, color="#4f5d75")
        ax.tick_params(colors='#4f5d75', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    # ═════════════════════════════════════════════════════════════
    # 하단 피드백 텍스트창 (사진 속 정보창 레이아웃 매칭)
    # ═════════════════════════════════════════════════════════════
    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3>🛸 <b>초롱핑의 오로라 정밀 지진 위험군 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 격자 신호 분석 결과: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#4b5563; line-height:1.6; font-size:14px; margin:0;">
                지정한 타겟 위도 {lat:.4f}°, 경도 {lon:.4f}° 공간 스펙트럼 내부의 지동 진원 깊이와 규모 격자를 추적 연산했습니다츄.
                현재 좌표는 밀집 위험군 집단 분류상 <b>{final_grade}</b> 영역에 가장 가깝게 위치해 있는 상태입니다. 
                가장 가까운 실제 지진 활동성 진원 코어(진앙) 데이터 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>로 계산되었습니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
