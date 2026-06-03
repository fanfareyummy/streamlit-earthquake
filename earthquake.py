import os
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 [슈팅스타팩트 오리지널 폼] 삐져나온 핑크 제거 및 돔 정중앙 매립 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #a6d5ff 0%, #d5f1fe 25%, #fbe3f1 50%, #fbe8d5 75%, #ffffff 100%) !important;
        color: #3b3a57 !important;
        overflow-x: hidden;
    }
    
    /* ✨ 하늘에서 은은하게 쏟아져 내리는 마법 별빛 가루 이펙트 */
    .stAppViewContainer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 999;
        background-image: 
            radial-gradient(circle at 15% 15%, #fff 1.5px, transparent 2.5px),
            radial-gradient(circle at 65% 25%, #fff 2px, transparent 3px),
            radial-gradient(circle at 40% 45%, #ffb6c1 2px, transparent 3.5px),
            radial-gradient(circle at 25% 65%, #fff 1px, transparent 2px),
            radial-gradient(circle at 85% 75%, #e0f2fe 2.5px, transparent 4.5px);
        background-size: 500px 500px;
        animation: magicStarFall 14s linear infinite;
        opacity: 0.75;
    }
    @keyframes magicStarFall {
        0% { background-position: 0px 0px; }
        100% { background-position: 40px 500px; }
    }

    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.18) 0%, transparent 60%),
                    radial-gradient(circle at 85% 75%, rgba(255, 105, 180, 0.18) 0%, transparent 60%);
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

    /* 🔮 슈팅스타팩트 절대 좌표 통합 프레임 무대 */
    .shooting-star-factory-stage {
        position: relative;
        width: 460px;
        height: 460px;
        margin: 20px auto;
        overflow: visible;
    }

    /* ⭐ 대형 황금 별 스탠드 베이스 파츠 */
    .star-gold-pedestal-base {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 63% 34%, 98% 34%, 71% 56%, 82% 90%, 50% 73%, 18% 90%, 29% 56%, 2% 34%, 37% 34%);
        box-shadow: 0 12px 30px rgba(234, 179, 8, 0.25);
        border: 4px solid #ffffff;
        z-index: 1; 
    }

    /* 👼 카툰 만화영화 스타일의 부드럽고 풍성한 요정의 겹날개 구현 */
    .fairy-wing-bundle-left {
        position: absolute; left: -85px; top: 120px; width: 200px; height: 180px; z-index: 2;
        filter: drop-shadow(-8px 12px 18px rgba(244, 114, 182, 0.5));
    }
    .fairy-wing-bundle-right {
        position: absolute; right: -85px; top: 120px; width: 200px; height: 180px; z-index: 2;
        filter: drop-shadow(8px 12px 18px rgba(244, 114, 182, 0.5));
    }
    .fairy-feather { position: absolute; background: #fff; border: 2.5px solid #fff; }
    
    /* 곡선미를 살린 카툰풍 라운드 깃털 배치 */
    .f-left-1 { width: 150px; height: 68px; top: 0; right: 0; background: linear-gradient(-45deg, #ffffff, #e2f5ff, #bae6fd); border-radius: 150px 30px 130px 90px; transform: rotate(-12deg); }
    .f-left-2 { width: 135px; height: 62px; top: 42px; right: 15px; background: linear-gradient(-45deg, #ffffff, #fce7f3, #f472b6); border-radius: 130px 25px 115px 80px; transform: rotate(-2deg); z-index: 2; }
    .f-left-3 { width: 115px; height: 55px; top: 82px; right: 35px; background: linear-gradient(-45deg, #ffffff, #e0f2fe, #7dd3fc); border-radius: 110px 20px 95px 70px; transform: rotate(8deg); }

    .f-right-1 { width: 150px; height: 68px; top: 0; left: 0; background: linear-gradient(45deg, #ffffff, #e2f5ff, #bae6fd); border-radius: 30px 150px 90px 130px; transform: rotate(12deg); }
    .f-right-2 { width: 135px; height: 62px; top: 42px; left: 15px; background: linear-gradient(45deg, #ffffff, #fce7f3, #f472b6); border-radius: 25px 130px 80px 115px; transform: rotate(2deg); z-index: 2; }
    .f-right-3 { width: 115px; height: 55px; top: 82px; left: 35px; background: linear-gradient(45deg, #ffffff, #e0f2fe, #7dd3fc); border-radius: 20px 110px 70px 95px; transform: rotate(-8deg); }

    /* 💖 외부 오로라 분홍색 콤팩트 하우징 코어 (상단에 불필요한 캡슐 요소 완전 분리 삭제) */
    .fact-pink-heart-shield {
        position: absolute;
        left: 45px;
        top: 45px;
        width: 370px;
        height: 370px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fbcfe8 45%, #ec4899 85%, #be185d 100%);
        border-radius: 50%;
        border: 9px solid #ffffff;
        box-shadow: 0 22px 50px rgba(236, 72, 153, 0.4);
        z-index: 3;
    }

    /* 🔒 [철통 매립 고정] 홀로그램 지구가 팩트 스크린 정중앙 내부 서클 안에 들어가도록 유도 */
    .map-inside-binder {
        position: absolute;
        left: 85px;
        top: 85px;
        width: 290px;
        height: 290px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 5px solid #fef08a;
        box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.75);
        background: #06031a;
    }
    
    /* 코딩 텍스트 노출 현상을 완전히 제어하기 위한 iframe 컨테이너 규격 강제 고정 */
    .hologram-iframe-container {
        width: 290px !important;
        height: 290px !important;
        border: none !important;
        overflow: hidden !important;
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
# 📊 [인공지능 클러스터링 기반 지진 연산 모듈]
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    np.random.seed(42)
    num_samples = 2000
    df = pd.DataFrame({
        '위도': np.random.uniform(-50, 50, num_samples),
        '경도': np.random.uniform(-160, 160, num_samples),
        '규모': np.random.uniform(1.5, 7.5, num_samples),
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
grade_map = {int(c): labels[i] for i, c in enumerate(order)}

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# ═════════════════════════════════════════════════════════════
# 메인 콘솔 UI 컨트롤 패널
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
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 동기화")
        
        show_df = df.sample(min(450, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{HEX_MAP[g_name]}', size: {float(row['규모'])}}}"
            points_js.append(p_str)
        
        points_js_str = ",\n".join(points_js)

        # 🪐 HTML 컴포넌트에 스트림릿 코드가 얽혀 텍스트가 표시되던 현상을 원천 차단한 순수 내부 스크립트
        three_js_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; overflow: hidden; background: transparent; }}
                canvas {{ background: transparent; width: 290px; height: 290px; cursor: grab; }}
                canvas:active {{ cursor: grabbing; }}
            </style>
        </head>
        <body>
            <canvas id="globeCanvas"></canvas>
            <script>
                const canvas = document.getElementById('globeCanvas');
                const ctx = canvas.getContext('2d');
                let width = canvas.width = 290;
                let height = canvas.height = 290;
                
                let rotationX = 0.35;
                let rotationY = {np.radians(lon)};
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                const points = [{points_js_str}];
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', size: 8.5 }};

                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    let radius = 105;
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
                    
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.65)';
                    ctx.lineWidth = 1.3;
                    for (let l = -60; l <= 60; l += 20) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 10) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    let allPoints = [...points, targetPoint];
                    allPoints.forEach(p => p._proj = project(p.lat, p.lon));
                    allPoints.sort((a, b) => b._proj.depth - a._proj.depth);
                    
                    allPoints.forEach(p => {{
                        let proj = p._proj;
                        if (proj.depth > -35) {{ 
                            let alpha = Math.max(0.4, (proj.depth + 105) / 210);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, 8.5, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 15;
                                ctx.shadowColor = '#db2777';
                            }} else {{
                                ctx.arc(proj.x, proj.y, Math.max(2.8, p.size * 1.0), 0, 2 * Math.PI);
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

        # 외부 삐져나옴과 덩어리 오염 버그를 100% 걷어낸 완전 차폐형 팩트 쉘 프레임워크
        html_shell = f"""
        <div class="shooting-star-factory-stage">
            <div class="star-gold-pedestal-base"></div>
            <div class="fairy-wing-bundle-left">
                <div class="fairy-feather f-left-1"></div>
                <div class="fairy-feather f-left-2"></div>
                <div class="fairy-feather f-left-3"></div>
            </div>
            <div class="fairy-wing-bundle-right">
                <div class="fairy-feather f-right-1"></div>
                <div class="fairy-feather f-right-2"></div>
                <div class="fairy-feather f-right-3"></div>
            </div>
            <div class="fact-pink-heart-shield"></div>
            <div class="map-inside-binder">
                <iframe srcdoc="{three_js_code.replace('"', '&quot;')}" class="hologram-iframe-container" frameborder="0" scrolling="no"></iframe>
            </div>
        </div>
        """
        st.markdown(html_shell, unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 도표")
        
        chart_points = []
        for c in sorted(df["cluster"].unique()):
            sub_set = df[(df["cluster"] == c) & (df["경도"].between(lon-30, lon+30)) & (df["위도"].between(lat-30, lat+30))].sample(min(150, len(df)), replace=True)
            g_name = grade_map.get(int(c), "저위험군")
            for _, r in sub_set.iterrows():
                chart_points.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{HEX_MAP[g_name]}', label: '{g_name}'}}")
        chart_points_str = ",\n".join(chart_points)

        canvas_chart_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Pretendard', sans-serif; margin: 0; background: rgba(255,255,255,0.75); border-radius: 20px; padding: 15px; }}
            </style>
        </head>
        <body>
            <canvas id="chartCanvas" width="460" height="380"></canvas>
            <script>
                const cvs = document.getElementById('chartCanvas');
                const ctx = cvs.getContext('2d');
                const pts = [{chart_points_str}];
                
                ctx.strokeStyle = '#cbd5e1';
                ctx.lineWidth = 0.5;
                for(let i=40; i<460; i+=50) {{
                    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 340); ctx.stroke();
                }}
                for(let j=20; j<340; j+=40) {{
                    ctx.beginPath(); ctx.moveTo(40, j); ctx.lineTo(460, j); ctx.stroke();
                }}
                
                ctx.fillStyle = '#4c4475';
                ctx.font = 'bold 12px Pretendard';
                ctx.fillText("타겟 주변 경도 범위 (Longitude)", 170, 368);
                
                ctx.save();
                ctx.translate(15, 200);
                ctx.rotate(-Math.PI/2);
                ctx.fillText("타겟 주변 위도 범위 (Latitude)", -80, 0);
                ctx.restore();
                
                pts.forEach(p => {{
                    let cx = 40 + ((p.x - ({lon-30})) / 60) * 400;
                    let cy = 340 - ((p.y - ({lat-30})) / 60) * 320;
                    if(cx >= 40 && cx <= 460 && cy >= 0 && cy <= 340) {{
                        ctx.beginPath();
                        ctx.arc(cx, cy, 4, 0, 2*Math.PI);
                        ctx.fillStyle = p.color;
                        ctx.fill();
                    }}
                }});
                
                let tx = 40 + (30 / 60) * 400;
                let ty = 340 - (30 / 60) * 320;
                ctx.beginPath();
                ctx.arc(tx, ty, 9, 0, 2*Math.PI);
                ctx.fillStyle = '#ffffff';
                ctx.strokeStyle = '#db2777';
                ctx.lineWidth = 3;
                ctx.fill(); ctx.stroke();
            </script>
        </body>
        </html>
        """
        components.html(canvas_chart_html, height=400, scrolling=False)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#1e1b4b;">🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#475569; line-height:1.7; font-size:14px; margin:0;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 위험 격자 스펙트럼 스캔이 완료되었으며, 
                인근 실제 지진 중심핵 코어 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
