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
# 🔮 [복원] 파스텔 우주 배경 + 별빛 효과 + 슈팅스타팩트 실물 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit&family=Pretendard:wght@700;900&display=swap');

    /* 🪐 파스텔톤 우주 오로라 배경 무드 복원 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #b4c5e7 0%, #eef2fa 40%, #e3daf7 70%, #f7e3ef 100%) !important;
        color: #333355 !important;
    }
    
    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(195, 175, 255, 0.4) 0%, transparent 50%),
                    radial-gradient(circle at 85% 75%, rgba(255, 195, 220, 0.4) 0%, transparent 50%);
        padding: 20px 40px !important;
        position: relative;
    }

    /* 🌟 위에서 은은하게 떨어지는 하얀색 별빛 애니메이션 복원 */
    .stMainBlockContainer::before {
        content: "★";
        position: absolute;
        top: -20px;
        left: 25%;
        color: rgba(255, 255, 255, 0.9);
        font-size: 14px;
        animation: fallStars 7s infinite linear;
        pointer-events: none;
        z-index: 99;
    }
    .stMainBlockContainer::after {
        content: "✧  ✨  * ✦";
        position: absolute;
        top: -30px;
        left: 65%;
        color: rgba(255, 255, 255, 0.85);
        font-size: 16px;
        letter-spacing: 130px;
        animation: fallStars 10s infinite linear;
        pointer-events: none;
        z-index: 99;
    }
    @keyframes fallStars {
        0% { transform: translateY(-20px) rotate(0deg); opacity: 0; }
        15% { opacity: 1; }
        85% { opacity: 1; }
        100% { transform: translateY(800px) rotate(360deg); opacity: 0; }
    }

    /* 상단 헤더 */
    .photo-top-header {
        background: rgba(255, 255, 255, 0.65);
        border: 2px solid rgba(255, 255, 255, 0.8);
        border-radius: 24px;
        padding: 18px 30px;
        text-align: center;
        box-shadow: 0 10px 35px rgba(180, 185, 215, 0.2);
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
    }
    .photo-top-header h1 { margin: 0; font-size: 26px; font-weight: 900; color: #433d6a; keyframe-name: glow; }

    /* 🔮 슈팅스타팩트 메인 스테이지 배치 구역 */
    .hologram-outer-stage {
        position: relative;
        width: 100%;
        height: 420px;
        margin-bottom: 20px;
    }

    /* 💖 핵심: 슈팅스타팩트 기기 본체 완벽 입체화 */
    .star-fact-device-body {
        position: absolute;
        left: 10px;
        bottom: 15px;
        width: 300px;
        height: 360px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #f7e9f8 45%, #e9cbe0 80%, #d8aadc 100%);
        border: 9px solid #ffffff;
        border-radius: 85px 85px 75px 75px;
        box-shadow: -15px 25px 40px rgba(100, 85, 135, 0.25), inset -4px -4px 15px rgba(0,0,0,0.05);
        transform: perspective(1000px) rotateY(18deg) rotateX(5deg);
        z-index: 10;
    }
    /* 팩트 왼쪽 노란 왕관 날개 장식 버튼 */
    .star-fact-device-body::before {
        content: "";
        position: absolute;
        left: -35px;
        top: 25%;
        width: 42px;
        height: 120px;
        background: linear-gradient(135deg, #ffe082 0%, #ffb300 50%, #ffa000 100%);
        border-radius: 50px 12px 12px 50px;
        border: 3px solid #ffffff;
    }
    /* 팩트 하단 미래지향적 내부 LCD 모니터 창 */
    .star-fact-inner-lcd {
        position: absolute;
        bottom: 25px;
        left: 20px;
        right: 20px;
        height: 140px;
        background: linear-gradient(180deg, #0b081d 0%, #13184b 100%);
        border: 3px solid #dca7ed;
        border-radius: 16px;
        box-shadow: inset 0 0 20px rgba(0,255,221,0.4);
        padding: 12px;
        color: #80deea;
        font-family: 'Orbit', sans-serif;
        font-size: 11px;
        line-height: 1.6;
    }

    /* 모니터에서 뿜어져 나오는 신비로운 광선 효과 */
    .hologram-light-beam {
        position: absolute;
        left: 180px;
        bottom: 120px;
        width: 220px;
        height: 260px;
        background: linear-gradient(45deg, rgba(220, 167, 237, 0.25) 0%, rgba(128, 222, 234, 0.08) 60%, transparent 100%);
        clip-path: polygon(0% 100%, 45% 100%, 100% 0%, 55% 0%);
        pointer-events: none;
        z-index: 5;
    }

    /* 🗺️ Pydeck 지도가 담기는 원형 바인더 구체 */
    .map-under-binder {
        position: absolute;
        left: 330px;
        top: 15px;
        width: 380px;
        height: 380px;
        z-index: 2;
        animation: floatSphere 3.5s infinite alternate ease-in-out;
    }

    /* 🛡️ [고장 해결] 지도가 사각형으로 깨지는 걸 덮어서 막아버리는 최상위 오로라 프레임 렌즈 */
    .hologram-top-lens-frame {
        position: absolute;
        left: 330px;
        top: 15px;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle 178px at center, transparent 98%, #ffffff 100%),
                    radial-gradient(circle 182px at center, transparent 100%, rgba(238, 242, 250, 0.98) 100%);
        border-radius: 50%;
        box-shadow: 0 0 0 8px #ffffff, 0 0 35px rgba(128, 222, 234, 0.6), 0 0 55px rgba(220, 166, 245, 0.4);
        z-index: 20; /* 지도 위에 배치되어 사각형 모서리를 원형으로 마스킹 */
        pointer-events: none; /* 드래그 이벤트를 지도로 관통 */
        animation: floatSphere 3.5s infinite alternate ease-in-out;
    }

    @keyframes floatSphere {
        0% { transform: translateY(0px); }
        100% { transform: translateY(-14px); }
    }

    /* 하단 결과 피드 카드 */
    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid #ffffff;
        border-radius: 22px;
        padding: 22px 28px;
        box-shadow: 0 12px 35px rgba(140, 145, 175, 0.12);
        margin-top: 25px;
        backdrop-filter: blur(8px);
    }
    .danger-tag { font-weight: 900; padding: 4px 12px; border-radius: 12px; color: white; }
    .tag-high { background: #ff7675; }
    .tag-mid { background: #ffeaa7; color: #555; }
    .tag-low { background: #55efc4; color: #222; }

    /* 대형 스캔 버튼 스타일 */
    .stButton>button {
        background: linear-gradient(90deg, #fbc2eb 0%, #a6c1ee 100%) !important;
        color: #4a4375 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 5px 15px rgba(166, 193, 238, 0.4);
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
# 레이아웃 배치 시작
# ═════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#716b94; font-size:13px; margin-top:5px; font-weight:700;">
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

    # 가로 배치 영역 분할
    col_left_stage, col_right_graph = st.columns([7, 5])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 원형 투사")
        
        # 팩트 본체 및 도넛 마스크 렌즈 레이어를 한 묶음으로 묶어 깨짐 우회
        st.markdown(
            f"""
            <div class="hologram-outer-stage">
                <div class="star-fact-device-body">
                    <div class="star-fact-inner-lcd">
                        <span style="color:#e1f5fe; font-weight:900;">[STATION_ACTIVE]</span><br>
                        팩트동기화: ONLINE<br>
                        위도좌표축: {lat:.3f}°<br>
                        경도좌표축: {lon:.3f}°<br>
                        인근지동핵: {nearest_km:.1f}km<br>
                        <span style="color:#ffa726; font-weight:700;">🔮 HOLOGRAM SYSTEM READY</span>
                    </div>
                </div>
                <div class="hologram-light-beam"></div>
                <div class="hologram-top-lens-frame"></div>
                
                <div class="map-under-binder">
            """, 
            unsafe_allow_html=True
        )
        
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
                get_radius=90000,
                pickable=True
            ),
            pdk.Layer(
                'ScatterplotLayer',
                data=pd.DataFrame([{"lon": lon, "lat": lat}]),
                get_position='[lon, lat]',
                get_color=[255, 238, 130, 255],
                get_radius=340000,
                stroked=True,
                line_width_min_pixels=3,
                get_line_color=[232, 67, 147, 255]
            )
        ]
        
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=2.1, pitch=35, bearing=10),
            map_style='mapbox://styles/mapbox/light-v10',
            tooltip={"text": "위험분류: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        fig, ax = plt.subplots(figsize=(5.5, 4.8), facecolor='none')
        ax.set_facecolor((1, 1, 1, 0.45)) 
        
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#ffeaa7", "저위험군": "#55efc4"}
        for c in sorted(df["cluster"].unique()):
            sub_set = df[df["cluster"] == c]
            g_name = grade_map.get(int(c), "저위험군")
            ax.scatter(sub_set["경도"], sub_set["위도"], s=15, alpha=0.6,
                       color=HEX_MAP.get(g_name, "#b2bec3"), label=g_name)
        
        ax.scatter(lon, lat, c="#ffeaa7", s=320, marker="*", edgecolors="#e84393", linewidths=2.5, zorder=10)
        
        ax.set_xlim(lon-30, lon+30)
        ax.set_ylim(lat-30, lat+30)
        ax.grid(True, color='#dcdde1', linestyle='-', linewidth=0.8)
        
        # 다운로드 완료된 폰트 직접 맵핑으로 에러/깨짐 동시 해결
        ax.set_xlabel("타겟 경도", fontproperties=font_prop, fontsize=11, color="#4f5d75", fontweight='bold')
        ax.set_ylabel("타겟 위도", fontproperties=font_prop, fontsize=11, color="#4f5d75", fontweight='bold')
        
        legend_obj = ax.legend(loc='upper right', framealpha=0.6)
        if legend_obj:
            for text in legend_obj.get_texts():
                text.set_fontproperties(font_prop)
                text.set_size(9.5)
            
        ax.tick_params(colors='#4f5d75', labelsize=9)
        
        st.pyplot(fig)
        plt.close(fig)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#2b2b52;">🛸 <b>초롱핑의 오로라 정밀 지진 위험군 피드</b></h3>
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
