import os
import streamlit as st
import numpy as np
import pandas as pd
import pydeck as pdk
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib as mpl

# ═════════════════════════════════════════════════════════════
# 폰트 깨짐 철통 방어 장치 (Matplotlib 한글 무조건 매핑)
# ═════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def setup_korean_font():
    mpl.rcParams["axes.unicode_minus"] = False
    # 플랫폼별 절대 깨지지 않는 기본 한글 고딕 패밀리 명시적 선언
    system_fonts = ["Malgun Gothic", "AppleGothic", "NanumGothic", "sans-serif"]
    
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
            
    # 파일 경로 매핑 실패 시 시스템 폰트 이름으로 강제 적용
    mpl.rc("font", family=system_fonts[0])
    return system_fonts[0]

KOREAN_FONT = setup_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 3D 지도 강제 원형 가두기 + 하얀색 떨어지는 별빛 효과 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@500;700;900&display=swap');

    /* 전체 배경 및 오로라 무드 질감 */
    html, body, [data-testid="stAppViewContainer"] {{
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #b4c5e7 0%, #eef2fa 40%, #e3daf7 70%, #f7e3ef 100%) !important;
        color: #333355 !important;
        overflow-x: hidden;
    }}
    .stMainBlockContainer {{
        background: radial-gradient(circle at 15% 25%, rgba(195, 175, 255, 0.35) 0%, transparent 50%),
                    radial-gradient(circle at 85% 75%, rgba(255, 195, 220, 0.35) 0%, transparent 50%);
        padding: 20px 40px !important;
        position: relative;
    }}

    /* 🌟 하늘에서 떨어지는 하얀색 별빛 입자 레이아웃 효과 */
    .stMainBlockContainer::before {{
        content: "✨";
        position: absolute;
        top: -20px;
        left: 20%;
        color: rgba(255, 255, 255, 0.8);
        font-size: 16px;
        animation: fallStars 6s infinite linear;
        pointer-events: none;
        z-index: 10;
    }}
    .stMainBlockContainer::after {{
        content: "✧  * ✨  .  *";
        position: absolute;
        top: -40px;
        left: 60%;
        color: rgba(255, 255, 255, 0.7);
        font-size: 20px;
        letter-spacing: 120px;
        animation: fallStars 9s infinite linear;
        pointer-events: none;
        z-index: 10;
    }}

    @keyframes fallStars {{
        0% {{ transform: translateY(-50px) rotate(0deg); opacity: 0; }}
        10% {{ opacity: 1; }}
        90% {{ opacity: 1; }}
        100% {{ transform: translateY(800px) rotate(360deg); opacity: 0; }}
    }}

    /* 최상단 투명 타이틀 바 */
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

    /* 메인 홀로그램 통합 스테이지 판넬 */
    .hologram-stage {{
        display: flex;
        position: relative;
        width: 100%;
        height: 480px;
        margin-top: 10px;
    }}

    /* 🔮 슈팅스타팩트 실물 완구 하드웨어 입체 디테일 고도화 */
    .star-fact-device-body {{
        position: absolute;
        left: 5px;
        bottom: 20px;
        width: 310px;
        height: 350px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #f7e9f8 45%, #e9cbe0 80%, #d8aadc 100%);
        border: 10px solid #ffffff;
        border-radius: 85px 85px 70px 70px;
        box-shadow: -15px 25px 40px rgba(100, 85, 135, 0.3), 
                    inset -4px -4px 15px rgba(0,0,0,0.06),
                    0 0 20px rgba(235, 205, 240, 0.5);
        transform: perspective(1000px) rotateY(26deg) rotateX(10deg);
        z-index: 2;
    }}
    
    /* 완구 금빛 사이드 윙 엠블럼 */
    .star-fact-device-body::before {{
        content: "";
        position: absolute;
        left: -38px;
        top: 26%;
        width: 44px;
        height: 125px;
        background: linear-gradient(135deg, #ffe082 0%, #ffb300 50%, #ffa000 100%);
        border-radius: 50px 12px 12px 50px;
        border: 3px solid #ffffff;
        box-shadow: -5px 5px 12px rgba(80, 45, 110, 0.2);
    }}

    /* 오른쪽 크림치즈 옐로우 마이크&힌지 접합 핀 */
    .star-fact-device-body::after {{
        content: "";
        position: absolute;
        right: -9px;
        top: 42%;
        width: 14px;
        height: 40px;
        background: #fffbe6;
        border: 3px solid #ffffff;
        border-radius: 0 8px 8px 0;
    }}

    /* 완구 내부 정밀 상태 계기판 LCD 스크린 */
    .star-fact-inner-lcd {{
        position: absolute;
        bottom: 22px;
        left: 18px;
        right: 18px;
        height: 145px;
        background: linear-gradient(180deg, #0b081d 0%, #13184b 100%);
        border: 3.5px solid #dca7ed;
        border-radius: 16px;
        box-shadow: inset 0 0 20px rgba(0,255,221,0.4), 0 0 10px rgba(220,167,237,0.3);
        padding: 12px;
        color: #80deea;
        font-family: 'Orbit', sans-serif;
        font-size: 11px;
        line-height: 1.6;
    }}

    /* 팩트 액정에서 대각선 공간으로 터져 나오는 광선 기둥 */
    .hologram-light-beam {{
        position: absolute;
        left: 180px;
        bottom: 110px;
        width: 250px;
        height: 290px;
        background: linear-gradient(45deg, rgba(220, 167, 237, 0.4) 0%, rgba(128, 222, 234, 0.2) 60%, transparent 100%);
        clip-path: polygon(0% 100%, 40% 100%, 100% 0%, 50% 0%);
        pointer-events: none;
        z-index: 1;
    }}

    /* 🔮 핵심 변경: 3D 지도를 뚫리지 않게 강제 잠금하는 홀로그램 유리 구체 프레임 구조 */
    .globe-outer-frame {{
        position: absolute;
        left: 330px;
        top: 20px;
        width: 390px;
        height: 390px;
        border-radius: 50% !important;
        border: 6px solid rgba(255, 255, 255, 0.85);
        box-shadow: 0 0 35px rgba(128, 222, 234, 0.6), 
                    0 0 60px rgba(220, 166, 245, 0.45),
                    inset 0 0 40px rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.03);
        z-index: 5; /* 지도 위에 씌워 오로라 광택 유리 가두리 효과 연출 */
        pointer-events: none; 
        animation: floatSphere 3.2s infinite alternate ease-in-out;
    }}

    /* Pydeck 컴포넌트가 강제로 안착하는 내부 구역 바인딩 박스 */
    .globe-inner-map-binder {{
        position: absolute;
        left: 336px;
        top: 26px;
        width: 378px;
        height: 378px;
        border-radius: 50% !important;
        overflow: hidden !important; /* 내부 사각형 지도 완벽 제거 마스킹 */
        z-index: 4;
        animation: floatSphere 3.2s infinite alternate ease-in-out;
    }}

    /* 이중 마스킹 적용: Streamlit 내부 프레임까지 원형으로 동기화 */
    .globe-inner-map-binder > div, .globe-inner-map-binder [data-testid="stPydeckChart"] {{
        border-radius: 50% !important;
        overflow: hidden !important;
    }}

    @keyframes floatSphere {{
        0% {{ transform: translateY(0px); }}
        100% {{ transform: translateY(-14px); }}
    }}

    /* 하단 상황 알림판 */
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

    /* 컨트롤 실행 스캔 버튼 */
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
# 📊 [지진 위험군 분석] 정밀 클러스터링 알고리즘 로직
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
# 레이어 제어 및 타겟팅 인터페이스
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

    # [좌측 팩트 스테이지 공간] 및 [우측 2D 분포 그래프] 수평 배치 
    col_left_stage, col_right_graph = st.columns([7, 5])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 원형 투사")
        
        # HTML 가이드 조형 생성 (팩트 바디 + 홀로그램 투사 빔 배치)
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
                
                <div class="globe-inner-map-binder">
            """, 
            unsafe_allow_html=True
        )
        
        # Pydeck 공간 지도 가동 및 데이터 색상 할당
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
        
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=2.0, pitch=40, bearing=12),
            map_style='mapbox://styles/mapbox/light-v10',
            tooltip={"text": "지진분석군: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        
        # 지도 컴포넌트 상단에 유리 원형 셸 광택 효과 레이어를 강제로 뒤덮어 입체감 고정
        st.markdown(
            """
                </div>
                <div class="globe-outer-frame"></div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor='none')
        ax.set_facecolor((1, 1, 1, 0.45)) 
        
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#ffeaa7", "저위험군": "#55efc4"}
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=14, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        ax.scatter(lon, lat, c="#ffeaa7", s=320, marker="*", edgecolors="#e84393", linewidths=2.5, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#dcdde1', linestyle='-', linewidth=0.8)
        
        # 안전한 명시적 한글 폰트 주입 및 스타일링
        ax.set_xlabel("타겟 경도", fontfamily=KOREAN_FONT, fontsize=11, color="#4f5d75", fontweight='bold')
        ax.set_ylabel("타겟 위도", fontfamily=KOREAN_FONT, fontsize=11, color="#4f5d75", fontweight='bold')
        
        # 범례 한글 인코딩 보호
        ax.legend(prop={'family': KOREAN_FONT, 'size': 9, 'weight': 'bold'}, loc='upper right', framealpha=0.6)
        ax.tick_params(colors='#4f5d75', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    # ═════════════════════════════════════════════════════════════
    # 하단 분석 결과 피드창
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
