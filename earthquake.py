import os
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import json

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 [슈팅스타팩트: 텍토닉 마스터 에디션] 럭셔리 완구 디자인 & 리얼 지진대 탑재
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #bbf2f6 0%, #cfd9df 30%, #e2d1f9 60%, #fbc7d4 100%) !important;
        color: #3b3a57 !important;
        overflow-x: hidden;
    }
    
    /* ✨ 마법의 오로라 입체 별가루 효과 */
    .stAppViewContainer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 999;
        background-image: 
            radial-gradient(circle at 15% 20%, rgba(255,255,255,0.9) 1.5px, transparent 2.5px),
            radial-gradient(circle at 80% 40%, rgba(255,255,255,0.8) 2px, transparent 3.5px),
            radial-gradient(circle at 45% 70%, #ffd1ff 2.5px, transparent 5px);
        background-size: 350px 350px;
        opacity: 0.75;
    }

    .photo-top-header {
        background: rgba(255, 255, 255, 0.82);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 30px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 15px 45px rgba(218, 119, 242, 0.25);
        margin-bottom: 30px;
        backdrop-filter: blur(15px);
    }
    .photo-top-header h1 { margin: 0; font-size: 28px; font-weight: 900; color: #4a3e7d; text-shadow: 1px 1px 2px rgba(255,255,255,0.8); }

    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.88);
        border: 2px solid #ffffff;
        border-radius: 24px;
        padding: 26px 32px;
        box-shadow: 0 15px 45px rgba(148, 163, 184, 0.2);
        margin-top: 30px;
        backdrop-filter: blur(12px);
    }
    .danger-tag { font-weight: 900; padding: 6px 16px; border-radius: 12px; color: white; }
    .tag-high { background: #ff7675; box-shadow: 0 4px 10px rgba(255,118,117,0.4); }
    .tag-mid { background: #facc15; color: #333; box-shadow: 0 4px 10px rgba(250,204,21,0.4); }
    .tag-low { background: #4ade80; color: #111; box-shadow: 0 4px 10px rgba(74,222,128,0.4); }

    .stButton>button {
        background: linear-gradient(90deg, #fad0c4 0%, #ffd1ff 100%) !important;
        color: #5c3ea6 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 8px 25px rgba(250, 208, 196, 0.6);
        transition: all 0.25s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.01) translateY(-2px);
        box-shadow: 0 12px 30px rgba(250, 208, 196, 0.85);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [리얼 지진학 데이터 모델 모듈]
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_plate_boundary_earthquakes():
    np.random.seed(77)
    num_samples = 1500
    
    # 실제 지구의 핵심 지진대 및 판의 경계 궤적 (불의 고리 노드 세트)
    # [위도, 경도] 순서 리스트
    ring_of_fire_nodes = [
        [36.5, 138.0], [40.0, 143.0], [52.0, 160.0], [55.0, -160.0], [60.0, -140.0], # 쿠릴-알류샨 가호
        [45.0, -125.0], [35.0, -120.0], [20.0, -105.0], [10.0, -85.0], [-15.0, -75.0], [-35.0, -72.0], # 미서부-칠레 해구
        [-45.0, 165.0], [-20.0, 175.0], [-5.0, 150.0], [-2.0, 120.0], [5.0, 125.0], [15.0, 120.0], # 오세아니아-필리핀
        [-8.0, 110.0], [-6.0, 100.0], [25.0, 95.0], [35.0, 75.0], [37.0, 45.0], [38.0, 25.0] # 인도네시아-알프스-터키
    ]
    
    lats, lons = [], []
    for _ in range(num_samples):
        # 무작위로 하나의 판의 경계 기준 노드를 선택한 뒤 주변으로 응력 확산 처리
        node = ring_of_fire_nodes[np.random.randint(len(ring_of_fire_nodes))]
        lats.append(node["lat"] if isinstance(node, dict) else node[0] + np.random.normal(0, 3.2))
        lons.append(node["lon"] if isinstance(node, dict) else node[1] + np.random.normal(0, 4.0))
        
    df = pd.DataFrame({
        '위도': np.clip(lats, -85, 85),
        '경도': np.clip(lons, -180, 180),
        '규모': np.random.uniform(2.5, 8.2, num_samples),
        '진원깊이': np.random.uniform(8, 650, num_samples),
        '영향도': np.random.uniform(10, 100, num_samples),
    })
    return df

df = load_plate_boundary_earthquakes()
X = df[FEATURES]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = KMeans(n_clusters=3, random_state=42, n_init=10)
df["cluster"] = model.fit_predict(X_scaled)

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
# 메인 콘솔 UI 레이아웃
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#7a6fbe; font-size:14px; margin-top:5px; font-weight:700;">
            (오로라 스페이스 팩트 판의 경계 및 리얼 지진대 홀로그램 분석 콘솔)
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
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 동기화 (판의 경계 지진 지도)")
        
        show_df = df.sample(min(500, len(df)), random_state=77)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{HEX_MAP[g_name]}', size: {float(row['규모'])}}}"
            points_js.append(p_str)
        points_js_str = ",\n".join(points_js)

        # 🪐 [밤티 왜곡 해결 마스터 스크립트] 실제 리얼 지진학 판의 경계 벨트를 주입한 고화질 500줄급 샌드박스 코드
        compact_master_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{
                    margin: 0; padding: 0;
                    background: transparent;
                    display: flex; justify-content: center; align-items: center;
                    width: 500px; height: 500px;
                    overflow: hidden;
                }}
                
                /* 🔮 완벽한 1:1 좌우대칭 슈팅스타 팩트 하우징 무대 */
                .shooting-star-pact-container {{
                    position: relative;
                    width: 480px; height: 480px;
                    display: flex; justify-content: center; align-items: center;
                }}

                /* 👑 상단 크라운 스타 안테나 보석 파츠 (정중앙 앵커링) */
                .compact-top-crown {{
                    position: absolute;
                    top: 12px; left: 50%;
                    width: 65px; height: 35px;
                    background: linear-gradient(180deg, #ff9ebb 0%, #ec4899 100%);
                    transform: translateX(-50%);
                    border-radius: 30px 30px 10px 10px;
                    border: 4px solid #ffffff;
                    box-shadow: 0 6px 14px rgba(236, 72, 153, 0.45);
                    z-index: 2;
                }}
                .compact-top-crown::after {{
                    content: '⭐';
                    position: absolute;
                    top: -4px; left: 50%;
                    transform: translateX(-50%);
                    font-size: 15px;
                }}

                /* ⭐ 하단 도금 황금 스탠드 베이스 받침대 */
                .star-gold-pedestal-base {{
                    position: absolute;
                    bottom: 12px; left: 50%;
                    width: 330px; height: 105px;
                    background: linear-gradient(135deg, #ffe57f 0%, #ffca28 50%, #ffb300 100%);
                    transform: translateX(-50%);
                    clip-path: polygon(18% 0%, 82% 0%, 100% 100%, 0% 100%);
                    border-radius: 0 0 30px 30px;
                    border-bottom: 5px solid #ffffff;
                    box-shadow: 0 10px 25px rgba(255, 179, 0, 0.4);
                    z-index: 1; 
                }}

                /* 👼 [좌우 데칼코마니 완벽 대칭] 리얼 오로라 실크 날개 파츠 */
                .fairy-wing-left {{
                    position: absolute; left: 5px; top: 150px;
                    width: 115px; height: 175px;
                    background: linear-gradient(to left, rgba(255,255,255,0.98), rgba(226,245,255,0.85), rgba(186,230,253,0.7));
                    border: 4px solid #ffffff;
                    border-radius: 140px 30px 90px 120px;
                    box-shadow: -8px 12px 25px rgba(186, 230, 253, 0.45);
                    z-index: 2;
                }}
                .fairy-wing-right {{
                    position: absolute; right: 5px; top: 150px;
                    width: 115px; height: 175px;
                    background: linear-gradient(to right, rgba(255,255,255,0.98), rgba(255,231,243,0.85), rgba(244,114,182,0.7));
                    border: 4px solid #ffffff;
                    border-radius: 30px 140px 120px 90px;
                    box-shadow: 8px 12px 25px rgba(244, 114, 182, 0.45);
                    z-index: 2;
                }}

                /* 💖 오로라 크롬 핑크 메인 실드 하우징 바디 */
                .fact-pink-heart-shield {{
                    position: relative;
                    width: 360px; height: 360px;
                    background: radial-gradient(circle at 30% 30%, #ffffff 0%, #ffb6c1 25%, #f472b6 55%, #db2777 85%, #9d174d 100%);
                    border-radius: 50%;
                    border: 9px solid #ffffff;
                    box-shadow: 
                        inset 0 -15px 25px rgba(0,0,0,0.22),
                        inset 0 15px 25px rgba(255,255,255,0.65),
                        0 20px 45px rgba(219, 39, 119, 0.5);
                    display: flex; justify-content: center; align-items: center;
                    z-index: 3;
                }}

                /* ✨ 내부 금도금 베젤 스타 아우라 링 */
                .fact-inner-gold-ring {{
                    position: relative;
                    width: 300px; height: 300px;
                    border-radius: 50%;
                    background: transparent;
                    border: 6px solid #facc15;
                    box-shadow: 0 0 18px #facc15, inset 0 0 12px rgba(234, 179, 8, 0.6);
                    display: flex; justify-content: center; align-items: center;
                }}

                /* 🔒 밤티/일그러짐을 완전 차단한 정원 스크린 코어 돔 */
                .map-inside-binder {{
                    position: relative;
                    width: 276px; height: 276px;
                    border-radius: 50%;
                    overflow: hidden;
                    border: 4px solid #ffffff;
                    box-shadow: inset 0 0 35px rgba(0, 242, 254, 0.9);
                    background: linear-gradient(135deg, #070422 0%, #02010d 100%);
                }}

                canvas {{
                    position: absolute;
                    top: 0; left: 0;
                    width: 276px; height: 276px;
                    cursor: grab;
                }}
                canvas:active {{ cursor: grabbing; }}
            </style>
        </head>
        <body>
            <div class="shooting-star-pact-container">
                <div class="compact-top-crown"></div>
                <div class="star-gold-pedestal-base"></div>
                <div class="fairy-wing-left"></div>
                <div class="fairy-wing-right"></div>
                
                <div class="fact-pink-heart-shield">
                    <div class="fact-inner-gold-ring">
                        <div class="map-inside-binder">
                            <canvas id="globeCanvas" width="276" height="276"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const canvas = document.getElementById('globeCanvas');
                const ctx = canvas.getContext('2d');
                let size = 276;
                let centerPoint = size / 2;
                
                let rotationX = 0.35;
                let rotationY = {np.radians(lon)};
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                const points = [{points_js_str}];
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', size: 9.5 }};

                // 🌋 [리얼 텍토닉 맵 플레이트 수식] 밤티 현상을 부수고 지구의 실제 주요 '판의 경계선(지진 벨트)' 벡터 주입
                const tectonicPlates = [
                    // 환태평양 지진대 서측 (뉴질랜드-인도네시아-필리핀-일본-쿠릴열도)
                    [[-45,165], [-35,178], [-20,175], [-12,160], [-5,150], [0,135], [-7,120], [-8,110], [5,125], [15,120], [25,122], [35,135], [40,143], [52,160], [55,170]],
                    // 환태평양 지진대 동측 (알류샨 열도-알래스카-북미 서부-칠레 해구 가호)
                    [[55,170], [55,-160], [60,-140], [50,-128], [35,-120], [20,-105], [10,-85], [2,-78], [-15,-75], [-35,-72], [-45,-75], [-55,-68]],
                    // 알프스-히말라야-인도네시아 대지진 벨트
                    [[38,25], [37,35], [37,45], [30,55], [32,65], [35,75], [28,85], [25,95], [10,98], [-2,100], [-6,105], [-7,110]],
                    // 대서양 중앙 해령 벨트 (지구 반대편 윤곽선 참고용)
                    [[65,-18], [50,-30], [25,-45], [0,-20], [-20,-12], [-40,-15]]
                ];

                // 📐 고차원 구체 사영 공간 렌더링 매트릭스 수식
                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    let radius = 106; 
                    
                    let x = radius * Math.cos(rLat) * Math.sin(rLon);
                    let y = radius * Math.sin(rLat);
                    let z = radius * Math.cos(rLat) * Math.cos(rLon);
                    
                    let cosX = Math.cos(rotationX);
                    let sinX = Math.sin(rotationX);
                    let ry = y * cosX - z * sinX;
                    let rz = y * sinX + z * cosX;
                    
                    return {{ x: x + centerPoint, y: -ry + centerPoint, depth: rz }};
                }}

                function draw() {{
                    ctx.clearRect(0, 0, size, size);
                    
                    // 1단계: 실제 3D 지진대 판의 경계선(Tectonic Plate Lines) 매핑선 드로잉
                    ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)'; // 지진대 고유의 붉은 레이더 컬러 마킹
                    ctx.lineWidth = 1.8;
                    tectonicPlates.forEach(plate => {{
                        let startNode = true;
                        ctx.beginPath();
                        for(let i=0; i<plate.length; i++) {{
                            let p = project(plate[i][0], plate[i][1]);
                            if (p.depth > -35) {{ 
                                if (startNode) {{
                                    ctx.moveTo(p.x, p.y);
                                    startNode = false;
                                }} else {{
                                    ctx.lineTo(p.x, p.y);
                                }}
                            }}
                        }}
                        ctx.stroke();
                    }});

                    // 2단계: 홀로그램 보조 위선/경선 그리드 스캔 망 투사
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.4)';
                    ctx.lineWidth = 0.9;
                    for (let l = -60; l <= 60; l += 20) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 15) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 3단계: 지진대 위에 완벽 정렬 밀착 처리된 3D 지진 포인트 연산 출력
                    let allPoints = [...points, targetPoint];
                    allPoints.forEach(p => p._proj = project(p.lat, p.lon));
                    allPoints.sort((a, b) => b._proj.depth - a._proj.depth);
                    
                    allPoints.forEach(p => {{
                        let proj = p._proj;
                        if (proj.depth > -40) {{ 
                            let alpha = Math.max(0.35, (proj.depth + 106) / 212);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, 9.5, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 15;
                                ctx.shadowColor = '#ffffff';
                            }} else {{
                                ctx.arc(proj.x, proj.y, Math.max(3.0, p.size * 1.1), 0, 2 * Math.PI);
                                ctx.fillStyle = p.color;
                                ctx.shadowBlur = 0;
                            }}
                            ctx.globalAlpha = alpha;
                            ctx.fill();
                        }}
                    }});
                    ctx.shadowBlur = 0;
                    ctx.globalAlpha = 1.0;
                    requestAnimationFrame(draw);
                }}

                canvas.addEventListener('mousedown', e => {{
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
        components.html(compact_master_html, width=500, height=500, scrolling=False)

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 레이더")
        
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
                body {{ font-family: 'Pretendard', sans-serif; margin: 0; background: rgba(255,255,255,0.82); border-radius: 20px; padding: 15px; }}
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
            <h3 style="margin-top:0; color:#1e1b4b;">🛸 <b>초롱핑의 오로라 판 경계 트래킹 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#475569; line-height:1.7; font-size:14px; margin:0;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 지반 하부의 판 경계(Tectonic Plate Boundary) 활성 스트레스 스캔이 완료되었습니다츄.
                가장 가까운 활성 불의 고리 지진대 단층선 코어 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>로 계산되었습니다츄!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
