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
# 🎨 [슈팅스타팩트 원본 사진 기반] 완벽한 카툰 요정 날개 및 마법 별빛 이펙트 세팅
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
        overflow-x: hidden;
    }
    
    /* ✨ [부활] 하늘에서 끊임없이 떨어지는 몽환적인 별빛 효과 */
    .stAppViewContainer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 999;
        background-image: 
            radial-gradient(circle at 20% 10%, #fff 1px, transparent 2px),
            radial-gradient(circle at 75% 20%, #fff 2px, transparent 3px),
            radial-gradient(circle at 50% 40%, #ffb6c1 1.5px, transparent 2px),
            radial-gradient(circle at 30% 60%, #fff 2px, transparent 3px),
            radial-gradient(circle at 80% 80%, #e0f2fe 2px, transparent 4px);
        background-size: 550px 550px;
        animation: starDustFall 12s linear infinite;
        opacity: 0.8;
    }
    @keyframes starDustFall {
        0% { background-position: 0px 0px; }
        100% { background-position: 100px 550px; }
    }

    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.25) 0%, transparent 60%),
                    radial-gradient(circle at 85% 75%, rgba(255, 105, 180, 0.25) 0%, transparent 60%);
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

    /* 🔮 슈팅스타팩트 마법 거치대 무대 (크기 확대 대응) */
    .shooting-star-factory-stage {
        position: relative;
        width: 500px;
        height: 500px;
        margin: 20px auto;
    }

    /* ⭐ 대형 황금색 입체 별 마크 스탠드 베이스 */
    .star-gold-pedestal-base {
        position: absolute;
        width: 100%;
        height: 100%;
        background: linear-gradient(135deg, #ffe082 0%, #facc15 40%, #eab308 100%);
        clip-path: polygon(50% 0%, 63% 35%, 98% 35%, 70% 57%, 81% 91%, 50% 74%, 19% 91%, 30% 57%, 2% 35%, 37% 35%);
        box-shadow: 0 15px 35px rgba(234, 179, 8, 0.3);
        border: 4px solid #ffffff;
        z-index: 1; 
    }

    /* 👼 [대개조] 원본 완구 사진과 일치하는 풍성한 카툰 요정의 날개 구현 */
    .fairy-wing-container-left {
        position: absolute; left: -85px; top: 140px; width: 220px; height: 180px; z-index: 2;
        filter: drop-shadow(-10px 14px 20px rgba(244, 114, 182, 0.6));
    }
    .fairy-wing-container-right {
        position: absolute; right: -125px; top: 140px; width: 220px; height: 180px; z-index: 2;
        filter: drop-shadow(10px 14px 20px rgba(244, 114, 182, 0.6));
    }
    
    /* 겹겹이 풍성하게 쌓이는 카툰 깃털 묘사 */
    .cartoon-feather-l1 {
        position: absolute; top: 0; right: 20px; width: 160px; height: 60px;
        background: linear-gradient(-45deg, #ffffff, #e0f2fe, #bae6fd);
        border: 3px solid #fff; border-radius: 160px 20px 140px 60px; transform: rotate(-10deg);
    }
    .cartoon-feather-l2 {
        position: absolute; top: 35px; right: 40px; width: 140px; height: 55px;
        background: linear-gradient(-45deg, #ffffff, #fbcfe8, #f472b6);
        border: 3px solid #fff; border-radius: 140px 20px 120px 55px; transform: rotate(-2deg);
    }
    .cartoon-feather-l3 {
        position: absolute; top: 70px; right: 60px; width: 120px; height: 50px;
        background: linear-gradient(-45deg, #ffffff, #e0f2fe, #7dd3fc);
        border: 3px solid #fff; border-radius: 120px 20px 100px 50px; transform: rotate(8deg);
    }

    .cartoon-feather-r1 {
        position: absolute; top: 0; left: 20px; width: 160px; height: 60px;
        background: linear-gradient(45deg, #ffffff, #e0f2fe, #bae6fd);
        border: 3px solid #fff; border-radius: 20px 160px 60px 140px; transform: rotate(10deg);
    }
    .cartoon-feather-r2 {
        position: absolute; top: 35px; left: 40px; width: 140px; height: 55px;
        background: linear-gradient(45deg, #ffffff, #fbcfe8, #f472b6);
        border: 3px solid #fff; border-radius: 20px 140px 55px 120px; transform: rotate(2deg);
    }
    .cartoon-feather-r3 {
        position: absolute; top: 70px; left: 60px; width: 120px; height: 50px;
        background: linear-gradient(45deg, #ffffff, #e0f2fe, #7dd3fc);
        border: 3px solid #fff; border-radius: 20px 120px 50px 100px; transform: rotate(-8deg);
    }

    /* 💖 외부 오로라 분홍색 하우징 원형 실드 (팩트 본체) */
    .fact-pink-heart-shield {
        position: absolute;
        left: 50px;
        top: 50px;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fbcfe8 45%, #ec4899 85%, #be185d 100%);
        border-radius: 50%;
        border: 10px solid #ffffff;
        box-shadow: 0 25px 55px rgba(236, 72, 153, 0.45);
        z-index: 3;
    }

    /* 🔒 [확대 개조] 대형 투명 홀로그램 구체가 표출되는 센터 돔 영역 */
    .map-inside-binder {
        position: absolute;
        left: 80px;
        top: 80px;
        width: 340px;
        height: 340px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 6px solid #fef08a;
        box-shadow: inset 0 0 40px rgba(0, 242, 254, 0.7);
        background: radial-gradient(circle, #120e3d 0%, #050214 100%);
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
        '위도': np.random.uniform(-60, 60, num_samples),
        '경도': np.random.uniform(-180, 180, num_samples),
        '규모': np.random.uniform(1.5, 7.5, num_samples),
        '진원깊이': np.random.uniform(5, 600, num_samples),
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
# 메인 콘솔 UI 레이아웃 빌드
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

    col_left_stage, col_right_graph = st.columns([1.1, 0.9])
    
    with col_left_stage:
        st.write("#### 🔮 3D 홀로그램 투사 (마우스 드래그 회전 가능)")
        
        # [해결] 꼬이던 핑크 블록과 이상한 문자열을 완벽히 제거하고 일체형 쉘로 정리
        html_shell = f"""
        <div class="shooting-star-factory-stage">
            <div class="star-gold-pedestal-base"></div>
            <div class="fairy-wing-container-left">
                <div class="cartoon-feather-l1"></div><div class="cartoon-feather-l2"></div><div class="cartoon-feather-l3"></div>
            </div>
            <div class="fairy-wing-container-right">
                <div class="cartoon-feather-r1"></div><div class="cartoon-feather-r2"></div><div class="cartoon-feather-r3"></div>
            </div>
            <div class="fact-pink-heart-shield"></div>
            <div class="map-inside-binder" id="inject-target"></div>
        </div>
        """
        st.markdown(html_shell, unsafe_allow_html=True)
        
        show_df = df.sample(min(500, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            points_js.append(f"{{lat: {row['위도']}, lon: {row['경도']}, color: '{HEX_MAP[g_name]}', size: {row['규모']}}}")
        points_js_str = ",\n".join(points_js)

        # 🪐 [크기 업그레이드 & 선명도 대폭 강화] 340px 대형 투명 인터랙티브 홀로그램 스크립트
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
                let width = canvas.width = 340;
                let height = canvas.height = 340;
                
                let rotationX = 0.4;
                let rotationY = {np.radians(lon)};
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                const points = [{points_js_str}];
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', size: 9 }};

                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    let radius = 125; // 더 시원하게 확대된 반지름 지름비
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
                    
                    // ✨ [색상 보정] 선명하고 진해진 고밀도 네온 비비드 사이언 블루 격자
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.6)';
                    ctx.lineWidth = 1.4;
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
                        if (proj.depth > -40) {{ 
                            let alpha = Math.max(0.4, (proj.depth + 125) / 250);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, 9, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 18;
                                ctx.shadowColor = '#ec4899';
                            }} else {{
                                ctx.arc(proj.x, proj.y, Math.max(3.0, p.size * 1.1), 0, 2 * Math.PI);
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
                    rotationY += deltaX * 0.006;
                    rotationX += deltaY * 0.006;
                    rotationX = Math.max(-Math.PI/3, Math.min(Math.PI/3, rotationX));
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }});
                window.addEventListener('mouseup', () => isDragging = false);
                draw();
            </script>
        </body>
        </html>
        """
        # 정밀 안착 바인딩 스페이싱 계산 적용
        st.markdown(
            """
            <style>
            .map-inside-binder iframe { pointer-events: auto !important; }
            </style>
            <div style="position: absolute; transform: translate(80px, -425px); z-index: 99;">
            """, unsafe_allow_html=True
        )
        components.html(three_js_code, height=340, width=340, scrolling=False)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right_graph:
        st.write("#### 📊 글로벌 지진 분포 매트릭스")
        
        # 🗺️ [대변신 해결] 좁게 갇힌 축소 지도가 아니라, 전 세계 전체 범위(-180~180, -90~90) 스케일로 투사
        # 요청하신 제목도 깨끗하게 날려버렸습니다!
        chart_points = []
        sub_set_all = df.sample(min(400, len(df)), random_state=42)
        for _, r in sub_set_all.iterrows():
            g_name = grade_map.get(int(r['cluster']), "저위험군")
            chart_points.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{HEX_MAP[g_name]}'}}")
        chart_points_str = ",\n".join(chart_points)

        canvas_chart_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: 'Pretendard', sans-serif; margin: 0; background: rgba(255,255,255,0.75); border-radius: 20px; padding: 20px; }}
            </style>
        </head>
        <body>
            <canvas id="chartCanvas" width="460" height="380"></canvas>
            <script>
                const cvs = document.getElementById('chartCanvas');
                const ctx = cvs.getContext('2d');
                const pts = [{chart_points_str}];
                
                // 🗺️ 전 세계 표준 그리드 망 구축
                ctx.strokeStyle = '#cbd5e1';
                ctx.lineWidth = 0.5;
                for(let i=40; i<=460; i+=60) {{
                    ctx.beginPath(); ctx.moveTo(i, 10); ctx.lineTo(i, 330); ctx.stroke();
                }}
                for(let j=10; j<=330; j+=40) {{
                    ctx.beginPath(); ctx.moveTo(40, j); ctx.lineTo(460, j); ctx.stroke();
                }}
                
                // 100% 안 깨지는 순수 브라우저 웹폰트 한글 축 표기
                ctx.fillStyle = '#4c4475';
                ctx.font = 'bold 13px Pretendard';
                ctx.fillText("전세계 경도 범위 (-180° ~ 180°)", 160, 365);
                
                ctx.save();
                ctx.translate(15, 200);
                ctx.rotate(-Math.PI/2);
                ctx.fillText("전세계 위도 범위 (-90° ~ 90°)", -80, 0);
                ctx.restore();
                
                // 포인트 전역 매핑
                pts.forEach(p => {{
                    let cx = 40 + ((p.x + 180) / 360) * 400;
                    let cy = 330 - ((p.y + 90) / 180) * 320;
                    if(cx >= 40 && cx <= 460 && cy >= 10 && cy <= 330) {{
                        ctx.beginPath();
                        ctx.arc(cx, cy, 3.5, 0, 2*Math.PI);
                        ctx.fillStyle = p.color;
                        ctx.fill();
                    }}
                }});
                
                // 현재 설정한 타겟 크로스헤어를 지도 위에 대형 보석 별로 실시간 마킹
                let tx = 40 + (({lon} + 180) / 360) * 400;
                let ty = 330 - (({lat} + 90) / 180) * 320;
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
        components.html(canvas_chart_html, height=410, scrolling=False)

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
