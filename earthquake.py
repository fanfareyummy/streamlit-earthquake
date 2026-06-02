import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Matplotlib (우측 2D 그래프용) 한글 깨짐 방지 범용 세팅
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 슈팅스타팩트 리얼 마스터피스: 고퀄리티 깃털 날개 및 투명 홀로그램 프레임
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

    /* 🔮 슈팅스타팩트 본체 무대 */
    .shooting-star-factory-stage {
        position: relative;
        width: 440px;
        height: 440px;
        margin: 40px auto;
    }

    /* ⭐ 대형 황금색 입체 별 모양 거치대 */
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

    /* 👼 [초고퀄리티 변신] 오로라 펄 레이어드가 들어간 세련된 요정의 천사 날개 - LEFT */
    .fact-wing-left-part {
        position: absolute;
        left: -85px;
        top: 130px;
        width: 170px;
        height: 130px;
        background: linear-gradient(-45deg, #ffffff 20%, #e0f2fe 50%, #fbcfe8 80%, #f472b6 100%);
        border: 4px solid #ffffff;
        /* 여러 장의 깃털이 겹친 고급스러운 형상 컷팅 */
        clip-path: polygon(100% 70%, 85% 85%, 65% 95%, 40% 100%, 15% 95%, 0% 80%, 2% 55%, 12% 30%, 35% 10%, 70% 0%, 90% 15%, 85% 45%);
        transform: rotate(-15deg);
        filter: drop-shadow(-8px 12px 16px rgba(236, 72, 153, 0.45));
        z-index: 2;
    }
    /* 깃털 디테일 선 구현 */
    .fact-wing-left-part::before {
        content: '';
        position: absolute;
        width: 100%; height: 100%;
        border-radius: inherit;
        background: radial-gradient(circle at 80% 50%, transparent 50%, rgba(255, 255, 255, 0.8) 53%, transparent 56%);
    }

    /* 👼 [초고퀄리티 변신] 오로라 펄 레이어드가 들어간 세련된 요정의 천사 날개 - RIGHT */
    .fact-wing-right-part {
        position: absolute;
        right: -85px;
        top: 130px;
        width: 170px;
        height: 130px;
        background: linear-gradient(45deg, #ffffff 20%, #e0f2fe 50%, #fbcfe8 80%, #f472b6 100%);
        border: 4px solid #ffffff;
        clip-path: polygon(0% 70%, 15% 85%, 35% 95%, 60% 100%, 85% 95%, 100% 80%, 98% 55%, 88% 30%, 65% 10%, 30% 0%, 10% 15%, 15% 45%);
        transform: rotate(18deg);
        filter: drop-shadow(8px 12px 16px rgba(236, 72, 153, 0.45));
        z-index: 2;
    }
    .fact-wing-right-part::before {
        content: '';
        position: absolute;
        width: 100%; height: 100%;
        background: radial-gradient(circle at 20% 50%, transparent 50%, rgba(255, 255, 255, 0.8) 53%, transparent 56%);
    }

    /* 💖 외부 마법 아우라 분홍색 서클 실드 */
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

    /* 🎀 본체 상단 럭셔리 골드/핑크 리본 크라운 마크 */
    .fact-top-crown-ribbon {
        position: absolute;
        left: 50%;
        top: -10px;
        transform: translateX(-50%);
        width: 140px;
        height: 55px;
        background: linear-gradient(180deg, #f472b6 0%, #db2777 100%);
        border: 4px solid #ffffff;
        border-radius: 28px;
        z-index: 12;
        box-shadow: 0 6px 14px rgba(0,0,0,0.2);
    }

    /* 🔒 [배경 투명화 완료] 완벽한 오로라 스페이스 테두리 링 안착 돔 */
    .map-inside-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 6px solid #fef08a; /* 내부 주얼리 링 테두리 */
        box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.6);
        background: radial-gradient(circle, #1e1b4b 0%, #090521 100%);
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
# 📊 [데이터 처리 모듈 및 AI 군집화 분석]
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
            if all(col in temp_df.columns for col in ["위도", "경도"] + FEATURES):
                df = temp_df
                df_loaded = True
                break
        except: continue

if not df_loaded:
    df = load_pure_quake_data()

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
# 프론트엔드 노드 출력 부
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
        
        # 데이터 전달 가공
        show_df = df.sample(min(400, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            points_js.append(f"{{lat: {row['위도']}, lon: {row['경도']}, color: '{HEX_MAP[g_name]}', size: {row['규모']}}}")
        points_js_str = ",\n".join(points_js)

        # 🪐 [대혁신 치트키] 투명 배경 + 마우스로 드래그하여 직접 회전할 수 있는 HTML5 Canvas 3D 인터랙티브 홀로그램
        three_js_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: transparent; font-family: sans-serif; }}
                canvas {{ background: transparent; width: 100vw; height: 100vh; cursor: grab; }}
                canvas:active {{ cursor: grabbing; }}
            </style>
        </head>
        <body>
            <canvas id="globeCanvas"></canvas>
            <script>
                const canvas = document.getElementById('globeCanvas');
                const ctx = canvas.getContext('2d');
                
                let width = canvas.width = 270;
                let height = canvas.height = 270;
                
                let rotationX = 0.3;
                let rotationY = {np.radians(lon)};
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                const points = [
                    {points_js_str}
                ];
                
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', size: 8 }};

                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    
                    let radius = 95;
                    let x = radius * Math.cos(rLat) * Math.sin(rLon);
                    let y = radius * Math.sin(rLat);
                    let z = radius * Math.cos(rLat) * Math.cos(rLon);
                    
                    // X축 회전 적용
                    let cosX = Math.cos(rotationX);
                    let sinX = Math.sin(rotationX);
                    let ry = y * cosX - z * sinX;
                    let rz = y * sinX + z * cosX;
                    
                    return {{ x: x + width/2, y: -ry + height/2, depth: rz }};
                }}

                function draw() {{
                    ctx.clearRect(0, 0, width, height);
                    
                    // 1. 와이어프레임 가상 위경도 가이드라인 투명 패스 드로잉
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.12)';
                    ctx.lineWidth = 0.7;
                    for (let l = -60; l <= 60; l += 30) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 10) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 2. 깊이(z-index) 기준으로 정렬하여 앞쪽에 있는 포인트 우선 입체 렌더링
                    let allPoints = [...points, targetPoint];
                    allPoints.forEach(p => p._proj = project(p.lat, p.lon));
                    allPoints.sort((a, b) => b._proj.depth - a._proj.depth);
                    
                    allPoints.forEach(p => {{
                        let proj = p._proj;
                        if (proj.depth > -30) {{ // 앞면 영역 가시화
                            let alpha = Math.max(0.2, (proj.depth + 95) / 190);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                // 타겟 포인트는 하얀색 빛나는 크로스헤어 서클로 특수 묘사
                                ctx.arc(proj.x, proj.y, 7, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 10;
                                ctx.shadowColor = '#db2777';
                            }} else {{
                                ctx.arc(proj.x, proj.y, Math.max(2, p.size * 0.8), 0, 2 * Math.PI);
                                ctx.fillStyle = p.color;
                                ctx.shadowBlur = 0;
                            }}
                            ctx.globalAlpha = alpha;
                            ctx.fill();
                        }}
                    }});
                    ctx.globalAlpha = 1.0;
                    ctx.shadowBlur = 0;
                    requestAnimationFrame(draw);
                }}

                // 드래그 마우스 인터랙션 이벤트 바인딩
                window.addEventListener('mousedown', e => {{
                    isDragging = true;
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }});
                
                window.addEventListener('mousemove', e => {{
                    if (!isDragging) return;
                    let deltaX = e.clientX - previousMousePosition.x;
                    let deltaY = e.clientY - previousMousePosition.y;
                    
                    rotationY += deltaX * 0.007;
                    rotationX += deltaY * 0.007;
                    rotationX = Math.max(-Math.PI/3, Math.min(Math.PI/3, rotationX));
                    
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }});
                
                window.addEventListener('mouseup', () => isDragging = false);
                
                draw();
            </script>
        </body>
        </html>
        """
        components.html(three_js_code, height=270, width=270, scroller=False)
        
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
        
        # 폰트를 직접 지정하지 않는 범용 방식으로 인자 전달 변경 완료
        ax.set_xlabel("Target Longitude", fontsize=11, color="#334155", fontweight='bold')
        ax.set_ylabel("Target Latitude", fontsize=11, color="#334155", fontweight='bold')
        ax.set_title("Earthquake Risk Cluster Matrix", fontsize=12, color="#1e1b4b", fontweight='bold')
        
        ax.legend(loc='upper right', framealpha=0.7)
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
