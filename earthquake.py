import os
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 2D 서포트 그래프 레이아웃 전용 범용 인코딩 세팅
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['axes.unicode_minus'] = False

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 [슈팅스타팩트 리얼 완구 퀄리티] 결이 살아있는 3D 입체 깃털 날개 CSS 마스터링
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

    /* 🔮 슈팅스타팩트 마법 조립 기단 무대 */
    .shooting-star-factory-stage {
        position: relative;
        width: 440px;
        height: 440px;
        margin: 40px auto;
    }

    /* ⭐ 대형 황금색 입체 별 마크 스탠드 베이스 */
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

    /* 👼 [대변신] 종이 느낌 제로! 결이 살아있는 입체 마법 깃털 조형 파츠 세트 */
    /* 왼쪽 천사 날개 마스터 그룹 */
    .wing-group-left {
        position: absolute;
        left: -85px;
        top: 110px;
        width: 180px;
        height: 160px;
        z-index: 2;
        filter: drop-shadow(-8px 12px 18px rgba(236, 72, 153, 0.5));
    }
    /* 겹겹이 쌓이는 세부 개별 깃털(Feather) 묘사 */
    .feather-l1 {
        position: absolute; top: 0; right: 0; width: 140px; height: 70px;
        background: linear-gradient(-60deg, #ffffff 30%, #e0f2fe 70%, #bae6fd 100%);
        border: 3px solid #ffffff;
        border-radius: 150px 10px 120px 80px;
        transform: rotate(-15deg);
    }
    .feather-l2 {
        position: absolute; top: 35px; right: 15px; width: 130px; height: 65px;
        background: linear-gradient(-60deg, #ffffff 30%, #fbcfe8 70%, #f472b6 100%);
        border: 3px solid #ffffff;
        border-radius: 140px 10px 110px 70px;
        transform: rotate(-5deg);
    }
    .feather-l3 {
        position: absolute; top: 70px; right: 35px; width: 110px; height: 60px;
        background: linear-gradient(-60deg, #ffffff 20%, #e0f2fe 60%, #7dd3fc 100%);
        border: 3px solid #ffffff;
        border-radius: 120px 10px 90px 60px;
        transform: rotate(8deg);
    }

    /* 오른쪽 천사 날개 마스터 그룹 */
    .wing-group-right {
        position: absolute;
        right: -85px;
        top: 110px;
        width: 180px;
        height: 160px;
        z-index: 2;
        filter: drop-shadow(8px 12px 18px rgba(236, 72, 153, 0.5));
    }
    .feather-r1 {
        position: absolute; top: 0; left: 0; width: 140px; height: 70px;
        background: linear-gradient(60deg, #ffffff 30%, #e0f2fe 70%, #bae6fd 100%);
        border: 3px solid #ffffff;
        border-radius: 10px 150px 80px 120px;
        transform: rotate(15deg);
    }
    .feather-r2 {
        position: absolute; top: 35px; left: 15px; width: 130px; height: 65px;
        background: linear-gradient(60deg, #ffffff 30%, #fbcfe8 70%, #f472b6 100%);
        border: 3px solid #ffffff;
        border-radius: 10px 140px 70px 110px;
        transform: rotate(5deg);
    }
    .feather-r3 {
        position: absolute; top: 70px; left: 35px; width: 110px; height: 60px;
        background: linear-gradient(60deg, #ffffff 20%, #e0f2fe 60%, #7dd3fc 100%);
        border: 3px solid #ffffff;
        border-radius: 10px 120px 60px 90px;
        transform: rotate(-8deg);
    }

    /* 💖 외부 오로라 분홍색 서클 아우라 하우징 */
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

    /* 🎀 본체 맨 위를 장식하는 핑크 리본 크라운 완장 마크 */
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

    /* 🔒 [투명 오로라 돔 링] 내부 홀로그램이 표출되는 글래스 원형 코어 안착 영역 */
    .map-inside-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 270px;
        height: 270px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 6px solid #fef08a; /* 노란색 팩트 주얼리 링 내부 마감 */
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
# 📊 [데이터 세팅 및 인공지능 지진 클러스터 연산부]
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
# 메인 제어 보드 디자인 노드 전개
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
        
        # 풍성해진 겹겹의 깃털 날개 그룹을 무대에 바인딩 조립합니다.
        st.markdown(
            """
            <div class="shooting-star-factory-stage">
                <div class="star-gold-pedestal-base"></div>
                
                <div class="wing-group-left">
                    <div class="feather-l1"></div>
                    <div class="feather-l2"></div>
                    <div class="feather-l3"></div>
                </div>
                
                <div class="wing-group-right">
                    <div class="feather-r1"></div>
                    <div class="feather-r2"></div>
                    <div class="feather-r3"></div>
                </div>
                
                <div class="fact-pink-heart-shield"></div>
                <div class="fact-top-crown-ribbon"></div>
                <div class="map-inside-binder">
            """, 
            unsafe_allow_html=True
        )
        
        # 3D 렌더링에 실시간 데이터 전달 처리
        show_df = df.sample(min(450, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            points_js.append(f"{{lat: {row['위도']}, lon: {row['경도']}, color: '{HEX_MAP[g_name]}', size: {row['규모']}}}")
        points_js_str = ",\n".join(points_js)

        # 🪐 [해결] 완전히 투명해진 오로라 바탕 + 마우스 회전 제어식 HTML5 네이티브 3D 코어 엔진
        # scroller 파라미터를 사용하지 않는 안전 규격 컴포넌트로 완벽 대체 완료!
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
                    
                    let cosX = Math.cos(rotationX);
                    let sinX = Math.sin(rotationX);
                    let ry = y * cosX - z * sinX;
                    let rz = y * sinX + z * cosX;
                    
                    return {{ x: x + width/2, y: -ry + height/2, depth: rz }};
                }}

                function draw() {{
                    ctx.clearRect(0, 0, width, height);
                    
                    // 1. 투명도가 적용된 화사한 오로라 지구본 그리드 라인 투사
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.15)';
                    ctx.lineWidth = 0.8;
                    for (let l = -60; l <= 60; l += 30) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 10) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 2. 입체 레이어 뎁스 정렬 알고리즘 작동 (글자 깨짐 및 투명도 버그 원천 차단)
                    let allPoints = [...points, targetPoint];
                    allPoints.forEach(p => p._proj = project(p.lat, p.lon));
                    allPoints.sort((a, b) => b._proj.depth - a._proj.depth);
                    
                    allPoints.forEach(p => {{
                        let proj = p._proj;
                        if (proj.depth > -35) {{ 
                            let alpha = Math.max(0.2, (proj.depth + 95) / 190);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, 8, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 12;
                                ctx.shadowColor = '#db2777';
                            }} else {{
                                ctx.arc(proj.x, proj.y, Math.max(2.5, p.size * 0.9), 0, 2 * Math.PI);
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
        # [수정 전] components.html(..., scroller=False) -> TypeError 유발 항목 제거 완료!
        components.html(three_js_code, height=270, width=270, scrolling=False)
        
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
        
        # HTML 캔버스가 아닌 우측 격자는 웹 안전 영어 레이블링으로 깔끔하고 안전하게 마감
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
