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
        "C:/Windows/Fonts/malgunbd.ttf",
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
            except Exception:
                pass
    return "sans-serif"

KOREAN_FONT = setup_korean_font()
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🌌 [캐치! 티니핑 시즌 5: 슈팅스타팩트] 오로라 스페이스 팩트 테마 (CSS)
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 위험군 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbit+Spaces&family=Pretendard:wght@700;900&display=swap');

    /* 전체 배경: 화려한 파스텔 오로라 우주 무드 */
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #a6d5ff 0%, #d5f1fe 25%, #fbe3f1 50%, #fbe8d5 75%, #ffffff 100%) !important;
        color: #3b3a57 !important;
    }
    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.4) 0%, transparent 60%),
                    radial-gradient(circle at 85% 75%, rgba(255, 105, 180, 0.4) 0%, transparent 60%);
        padding: 30px 60px !important;
    }

    /* 최상단 타이틀 바 */
    .photo-top-header {
        background: rgba(255, 255, 255, 0.7);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 26px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 12px 40px rgba(165, 180, 252, 0.25);
        margin-bottom: 30px;
        backdrop-filter: blur(12px);
    }}
    .photo-top-header h1 { margin: 0; font-size: 26px; font-weight: 900; color: #4c4475; }

    /* 슈팅스타팩트 입체 스테이지 구역 */
    .shooting-star-factory-stage {
        position: relative;
        width: 100%;
        height: 600px;
        margin-bottom: 20px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* 💖 핵심: 슈팅스타팩트 실물 디테일 구현 (윙, 스타, 보석 장식) */
    .fact-wing-star-body {
        position: absolute;
        width: 480px;
        height: 480px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fce7f3 35%, #ec4899 75%, #be185d 100%);
        border-radius: 50%;
        border: 12px solid #ffffff;
        box-shadow: 0 30px 70px rgba(219, 39, 119, 0.4), inset -8px -8px 25px rgba(0,0,0,0.1);
        z-index: 10;
        transform: scale(0.9);
    }
    .fact-wing-star-body::before, .fact-wing-star-body::after {
        content: "";
        position: absolute;
        top: 25%;
        width: 140px;
        height: 120px;
        background: linear-gradient(135deg, #fef08a 0%, #facc15 60%, #eab308 100%);
        border: 5px solid #ffffff;
        border-radius: 50px 15px 50px 50px;
        z-index: 5;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    }
    .fact-wing-star-body::before { left: -75px; transform: rotate(-15deg); }
    .fact-wing-star-body::after { right: -75px; transform: rotate(15deg); border-radius: 15px 50px 50px 50px; }

    .fact-center-jewel {
        position: absolute;
        left: 50%;
        top: -20px;
        transform: translateX(-50%);
        width: 180px;
        height: 70px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 5px solid #ffffff;
        border-radius: 35px;
        z-index: 12;
        box-shadow: 0 8px 18px rgba(0,0,0,0.15);
    }

    /* 📦 [수정 완료] 지도를 돔 내부로 봉인하는 언더레이어 바인더 */
    .map-inside-binder {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 380px;
        height: 380px;
        border-radius: 50% !important; /* 무조건 원형 유지 */
        overflow: hidden !important; /* 중요: 내부 지도가 돔 밖으로 튀어나오는 것 차단 */
        z-index: 8;
        background: #090521;
    }

    /* 🛡️ 지도가 샌드박스를 뚫고 사각형으로 깨지는 것을 막기 위해 유리막 쉴드 씌우기 */
    .fact-glass-lens {
        position: absolute;
        left: 50%;
        top: 50%;
        transform: translate(-50%, -50%);
        width: 380px;
        height: 380px;
        border-radius: 50%;
        /* 중심부는 투명하게 비춰 지도를 보여주고, 모서리 테두리는 마스킹 */
        background: radial-gradient(circle 178px at center, transparent 96%, #ffffff 100%),
                    radial-gradient(circle at 35% 35%, rgba(255, 255, 255, 0.35) 0%, rgba(128, 222, 234, 0.15) 50%, rgba(192, 132, 252, 0.25) 100%);
        border: 8px solid #fef08a; /* 실물 기기의 금빛 내부 링 */
        box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.6), 0 0 30px rgba(232, 121, 249, 0.5);
        z-index: 10; /* 지도 위에 배치됨 */
        pointer-events: none; /* 지도의 마우스 드래그 이벤트를 방해하지 않고 관통 */
    }

    /* 🌌 화려한 홀로그램 에너지 구체 (지구본 수용) */
    .holographic-universe-sphere {
        position: absolute;
        left: 330px;
        top: 330px;
        width: 440px;
        height: 440px;
        border-radius: 50% !important;
        overflow: hidden !important;
        box-shadow: 0 0 80px rgba(128, 222, 234, 0.8), 0 0 120px rgba(220, 166, 245, 0.6), inset 0 0 50px rgba(255, 255, 255, 0.8);
        border: 10px solid #ffffff;
        background: rgba(255, 255, 255, 0.05);
        z-index: 20;
    }

    /* 하단 상황 알림판 */
    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.8);
        border: 2px solid #ffffff;
        border-radius: 22px;
        padding: 22px 28px;
        color: #444;
        box-shadow: 0 10px 30px rgba(150, 150, 180, 0.15);
        margin-top: 25px;
        backdrop-filter: blur(8px);
    }
    .photo-bottom-card h3 { margin-top: 0; color: #1e1b4b; }
    .danger-tag { font-weight: 900; padding: 4px 12px; border-radius: 12px; color: white; }
    .tag-high { background: #ff7675; }
    .tag-mid { background: #facc15; color: #333; }
    .tag-low { background: #4ade80; color: #111; }

    /* 스캔 버튼 스타일 */
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
# 샘플 데이터 셋 가동 체계
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_data():
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
    df = load_sample_data()

# 지진 대피 위험군 클러스터링 산출
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)
df = df.dropna(subset=["cluster", "위도", "경도"])

# 위험 등급 연산 및 인덱스 매핑
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
        <div style="color:#6b7280; font-size:13px; margin-top:5px; font-weight:700;">
            (오로라 스페이스 팩트 지진 위험군 정밀 분석 시스템)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# 좌표 입력 패널
st.markdown("### 🎯 레이더 타겟 크로스헤어 좌표 설정")
cx, cy = st.columns(2)
with cx:
    lat = st.number_input("💖 타겟 위도 (Latitude)", -90.0, 90.0, 36.5, step=0.1)
with cy:
    lon = st.number_input("🌌 타겟 경도 (Longitude)", -180.0, 180.0, 127.5, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# 메인 팩트 구동 스테이지 배치
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
        
        # [수정 완료] f-string 중괄호 에러 수정을 위해 div를 감싸는 f-string에 이중 중괄호 사용
        st.markdown(
            f"""
            <div class="shooting-star-factory-stage">
                <div class="fact-wing-star-body">
                    <div class="fact-crown-jewel"></div>
                    <div class="fact-glass-lens"></div>
                    
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
                get_line_color=[21db, 39, 119, 255]
            )
        ]
        
        # 팩트 내부 우주 질감을 살리기 위해 맵 스타일을 다크 테마로 세팅
        r = pdk.Deck(
            layers=layers,
            initial_view_state=pdk.ViewState(latitude=lat, longitude=lon, zoom=1.4, pitch=40, bearing=0),
            map_style='mapbox://styles/mapbox/dark-v10',
            tooltip={"text": "위험분류: {cluster}\n위도: {위도}\n경도: {경도}"}
        )
        st.pydeck_chart(r)
        
        st.markdown("</div></div></div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        # 가독성 높은 차트 드로잉 (파스텔 & 화이트 배합)
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
