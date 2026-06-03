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
# 🎨 [슈팅스타팩트: 파스텔 럭셔리 에디션] 스트림릿 통합 프레임 CSS
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        /* ✨ 원본 사진의 화사하고 부드러운 파스텔 그라데이션 */
        background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 40%, #fef08a 100%) !important;
        color: #475569 !important;
        overflow-x: hidden;
    }
    
    /* 🌠 실시간으로 밤하늘에 떨어지는 별빛(슈팅스타) 이펙트 구현 */
    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: absolute;
        top: -100px; left: -100px; width: 200%; height: 200%;
        pointer-events: none;
        z-index: 1;
        background-image: 
            radial-gradient(circle at 10% 10%, rgba(255,255,255,1) 1.5px, transparent 3px),
            radial-gradient(circle at 40% 20%, rgba(255,255,255,0.9) 2px, transparent 4px),
            radial-gradient(circle at 70% 15%, rgba(254,240,138,1) 1px, transparent 2.5px),
            radial-gradient(circle at 25% 60%, rgba(244,114,182,0.8) 2px, transparent 5px),
            radial-gradient(circle at 85% 40%, rgba(147,197,253,0.9) 1.5px, transparent 4px);
        background-size: 500px 500px;
        animation: pastelShootingStars 18s linear infinite;
        opacity: 0.75;
    }

    @keyframes pastelShootingStars {
        0% { transform: translate(0, 0); }
        100% { transform: translate(250px, 500px); }
    }

    .photo-top-header {
        background: rgba(255, 255, 255, 0.65);
        border: 2px solid rgba(255, 255, 255, 0.9);
        border-radius: 30px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 15px 35px rgba(244, 114, 182, 0.15);
        margin-bottom: 30px;
        backdrop-filter: blur(20px);
        position: relative;
        z-index: 2;
    }
    .photo-top-header h1 { margin: 0; font-size: 28px; font-weight: 900; color: #db2777; text-shadow: 1px 1px 4px rgba(255,255,255,0.8); }

    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid rgba(244, 114, 182, 0.3);
        border-radius: 24px;
        padding: 26px 32px;
        box-shadow: 0 15px 35px rgba(244, 114, 182, 0.1);
        margin-top: 30px;
        backdrop-filter: blur(15px);
    }
    .danger-tag { font-weight: 900; padding: 6px 16px; border-radius: 12px; color: white; }
    .tag-high { background: #ff7675; box-shadow: 0 4px 10px rgba(255,118,117,0.3); }
    .tag-mid { background: #facc15; color: #475569; box-shadow: 0 4px 10px rgba(250,204,21,0.3); }
    .tag-low { background: #4ade80; color: #475569; box-shadow: 0 4px 10px rgba(74,222,128,0.3); }

    .stButton>button {
        background: linear-gradient(90deg, #fbcfe8 0%, #c084fc 100%) !important;
        color: #4c1d95 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 8px 20px rgba(192, 132, 252, 0.3);
        transition: all 0.25s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.01) translateY(-2px);
        box-shadow: 0 12px 25px rgba(192, 132, 252, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [대륙 윤곽 알고리즘] - 사진 속 실제 세계지도 라인 복원 벡터 데이터
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    np.random.seed(42)
    
    # 전달해주신 세계지도 이미지의 실제 핵심 지진대 플롯 라인 좌표 커스텀 매핑
    seismic_zones = [
        # [위도, 경도, 생성 비율]
        [61.0, -147.0, 0.12],   # 알래스카 베링해 라인
        [45.0, -122.0, 0.10],   # 북미 서안 오레곤 단층
        [36.0, -119.0, 0.25],   # 미국 캘리포니아 (초고밀집 레드 스팟 존)
        [19.0, -102.0, 0.08],   # 중미 멕시코 연안 라인
        [-12.0, -77.0, 0.08],   # 남미 페루 해구 경계선
        [-33.0, -71.0, 0.10],   # 칠레 서안 환태평양 판 라인
        [40.0, 142.0, 0.12],    # 일본 동쪽 해구 (우측 블루 스팟 핵심 라인)
        [14.0, 121.0, 0.07],    # 필리핀 지대 대륙선
        [-20.0, 175.0, 0.08],   # 오세아니아 통가 제도
        [-41.0, 174.0, 0.05],   # 뉴질랜드 경계선
        [38.0, 22.0, 0.05],     # 지중해 유럽 남부 라인
    ]
    
    num_samples = 2400
    lats, lons, magnitudes, depths, impacts = [], [], [], [], []
    
    for _ in range(num_samples):
        zone_idx = np.random.choice(len(seismic_zones), p=[z[2] for z in seismic_zones] / np.sum([z[2] for z in seismic_zones]))
        base_lat, base_lon, _ = seismic_zones[zone_idx]
        
        # 지도에서 선으로 밀집되어 퍼지도록 가우시안 노이즈 제어
        lat_noise = np.random.normal(0, 2.2)
        lon_noise = np.random.normal(0, 2.5)
        
        lat = np.clip(base_lat + lat_noise, -90.0, 90.0)
        lon = base_lon + lon_noise
        if lon > 180: lon -= 360
        if lon < -180: lon += 360
        
        # 미서부와 일본 열도 중심의 클러스터 분리 특성 부여
        if base_lon in [142.0, -119.0]:
            mag = np.random.uniform(4.5, 7.9)
            depth = np.random.uniform(80, 500)
        else:
            mag = np.random.uniform(1.5, 4.8)
            depth = np.random.uniform(5, 70)
            
        impact = mag * 10 + np.random.uniform(5, 25)
        
        lats.append(lat)
        lons.append(lon)
        magnitudes.append(mag)
        depths.append(depth)
        impacts.append(impact)

    return pd.DataFrame({'위도': lats, '경도': lons, '규모': magnitudes, '진원깊이': depths, '영향도': impacts})

df = load_pure_quake_data()
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
# 메인 콘솔 제어판 UI
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#f472b6; font-size:14px; margin-top:5px; font-weight:700;">
            (오로라 스페이스 팩트 지진 위험군 정밀 분석 마스터 시스템)
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
    lon = st.number_input("🌌 타겟 경도 (Longitude)", -180.0, 180.0, -120.0, step=0.1)

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
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 동기화 (파스텔 대칭 완성판)")
        
        show_df = df.sample(min(2000, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        # JavaScript 배열 문자열 빌드 (괄호 충돌 방지 가공)
        points_js_items = []
        for _, row in show_df.iterrows():
            g_name = grade_map.get(int(row["cluster"]), "저위험군")
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{HEX_MAP[g_name]}', size: {float(row['규모'])}}}"
            points_js_items.append(p_str)
        points_js_str = ",\n".join(points_js_items)

        # 🪐 [완벽 대칭 파스텔 팩트 하우징 + 대륙 라인 3D 회전 기술 코딩]
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
                
                /* 완벽 좌우 대칭형 하우징 컨테이너 */
                .shooting-star-pact-container {{
                    position: relative;
                    width: 460px; height: 460px;
                    display: flex; justify-content: center; align-items: center;
                }}

                /* 상단 골드 미니 윙 왕관 */
                .compact-top-crown {{
                    position: absolute;
                    top: 5px; left: 50%;
                    width: 55px; height: 22px;
                    background: linear-gradient(180deg, #ffe57f 0%, #ffca28 100%);
                    transform: translateX(-50%);
                    border-radius: 15px 15px 5px 5px;
                    border: 3px solid #ffffff;
                    box-shadow: 0 4px 10px rgba(255, 202, 40, 0.5);
                    z-index: 4;
                }}
                .compact-top-crown::after {{
                    content: '⭐';
                    position: absolute;
                    top: -5px; left: 50%;
                    transform: translateX(-50%);
                    font-size: 12px;
                }}

                /* 하단 대칭형 황금 받침대 */
                .star-gold-pedestal-base {{
                    position: absolute;
                    bottom: 15px; left: 50%;
                    width: 240px; height: 65px;
                    background: linear-gradient(180deg, #ffca28 0%, #ffb300 100%);
                    transform: translateX(-50%);
                    clip-path: polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);
                    border-radius: 0 0 25px 25px;
                    border-bottom: 4px solid #ffffff;
                    box-shadow: 0 8px 20px rgba(255, 179, 0, 0.35);
                    z-index: 1; 
                }}

                /* 좌측 파스텔 윙 (완벽 대칭) */
                .fairy-wing-left {{
                    position: absolute; left: 10px; top: 155px;
                    width: 100px; height: 150px;
                    background: linear-gradient(to left, rgba(255,255,255,0.95), rgba(186,230,253,0.9));
                    border: 4px solid #ffffff;
                    border-radius: 120px 40px 80px 100px;
                    box-shadow: -5px 10px 20px rgba(186, 230, 253, 0.4);
                    z-index: 2;
                }}
                
                /* 우측 파스텔 윙 (완벽 대칭) */
                .fairy-wing-right {{
                    position: absolute; right: 10px; top: 155px;
                    width: 100px; height: 150px;
                    background: linear-gradient(to right, rgba(255,255,255,0.95), rgba(186,230,253,0.9));
                    border: 4px solid #ffffff;
                    border-radius: 40px 120px 100px 80px;
                    box-shadow: 5px 10px 20px rgba(186, 230, 253, 0.4);
                    z-index: 2;
                }}

                /* 중앙 라운드 메인 핑크 하우징 */
                .fact-pink-heart-shield {{
                    position: relative;
                    width: 350px; height: 350px;
                    background: linear-gradient(135deg, #fbcfe8 0%, #f472b6 45%, #db2777 100%);
                    border-radius: 50%;
                    border: 8px solid #ffffff;
                    box-shadow: 
                        inset 0 -10px 20px rgba(0,0,0,0.15),
                        inset 0 10px 20px rgba(255,255,255,0.5),
                        0 15px 35px rgba(219, 39, 119, 0.35);
                    display: flex; justify-content: center; align-items: center;
                    z-index: 3;
                }}

                /* 내측 대칭 황금 링 가이드 */
                .fact-inner-gold-ring {{
                    position: relative;
                    width: 290px; height: 290px;
                    border-radius: 50%;
                    background: transparent;
                    border: 5px solid #facc15;
                    box-shadow: 0 0 12px #facc15, inset 0 0 8px rgba(234, 179, 8, 0.5);
                    display: flex; justify-content: center; align-items: center;
                }}

                /* 🔮 사진 속 파스텔 톤 구체 홀로그램 스페이스 구동 바인더 */
                .map-inside-binder {{
                    position: relative;
                    width: 266px; height: 266px;
                    border-radius: 50%;
                    overflow: hidden;
                    border: 4px solid #ffffff;
                    box-shadow: inset 0 0 30px rgba(34, 211, 238, 0.8);
                    /* 심해 다크블루가 아닌 부드럽고 영롱한 투명 에메랄드 스페이스 블루 */
                    background: radial-gradient(circle at 50% 50%, #083344 0%, #155e75 60%, #0f172a 100%);
                }}

                canvas {{
                    position: absolute;
                    top: 0; left: 0;
                    width: 266px; height: 266px;
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
                            <canvas id="globeCanvas" width="266" height="266"></canvas>
                        </div>
                    </div>
                </div>
            </div>

            <script>
                const canvas = document.getElementById('globeCanvas');
                const ctx = canvas.getContext('2d');
                let size = 266;
                let centerPoint = size / 2;
                
                let rotationX = 0.4;
                let rotationY = -({np.radians(lon)}) + Math.PI; 
                let isDragging = false;
                let previousMousePosition = {{ x: 0, y: 0 }};
                
                const points = [{points_js_str}];
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', size: 9.5 }};

                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    let radius = 130;
                    
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
                    
                    // 영롱한 오로라 광륜(위도 단층선) 입체 표출
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.35)';
                    ctx.lineWidth = 0.8;
                    for (let l = -60; l <= 60; l += 20) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 10) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    let tragedies = [...points, targetPoint];
                    
                    // 투영 좌표 매핑 알고리즘
                    for(let i=0; i<tragedies.length; i++) {{
                        tragedies[i]._proj = project(tragedies[i].lat, tragedies[i].lon);
                    }}
                    
                    // 3D 대륙 경계 왜곡 억제용 깊이 정렬 버퍼링
                    tragedies.sort(function(a, b) {{ return b._proj.depth - a._proj.depth; }});
                    
                    // 대륙 구도 스케치용 드로잉 루프
                    for(let i=0; i<tragedies.length; i++) {{
                        let p = tragedies[i];
                        let proj = p._proj;
                        if (proj.depth > -30) {{ 
                            let alpha = Math.max(0.2, (proj.depth + 130) / 260);
                            ctx.beginPath();
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, 8.0, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.shadowBlur = 12;
                                ctx.shadowColor = '#ffffff';
                            }} else {{
                                // 대륙 라인을 따라 촘촘한 밀집도를 보정하는 도트 크기 가중치 적용
                                ctx.arc(proj.x, proj.y, Math.max(2.4, p.size * 0.85), 0, 2 * Math.PI);
                                ctx.fillStyle = p.color;
                                ctx.shadowBlur = 0;
                            }}
                            ctx.globalAlpha = alpha;
                            ctx.fill();
                        }}
                    }}
                    
                    ctx.shadowBlur = 0;
                    ctx.globalAlpha = 1.0;
                    requestAnimationFrame(draw);
                }}

                canvas.addEventListener('mousedown', function(e) {{
                    isDragging = true;
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }});
                
                window.addEventListener('mousemove', function(e) {{
                    if (!isDragging) return;
                    let deltaX = e.clientX - previousMousePosition.x;
                    let deltaY = e.clientY - previousMousePosition.y;
                    rotationY += deltaX * 0.007;
                    rotationX += deltaY * 0.007;
                    rotationX = Math.max(-Math.PI/3, Math.min(Math.PI/3, rotationX));
                    previousMousePosition = {{ x: e.clientX, y: e.clientY }};
                }});
                
                window.addEventListener('mouseup', function() {{ isDragging = false; }});
                draw();
            </script>
        </body>
        </html>
        """
        components.html(compact_master_html, width=500, height=500, scrolling=False)

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
                body {{ font-family: 'Pretendard', sans-serif; margin: 0; background: rgba(255,255,255,0.7); border-radius: 20px; padding: 15px; color:#475569; }}
            </style>
        </head>
        <body>
            <canvas id="chartCanvas" width="460" height="380"></canvas>
            <script>
                const cvs = document.getElementById('chartCanvas');
                const ctx = cvs.getContext('2d');
                const pts = [{chart_points_str}];
                
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
                ctx.lineWidth = 0.5;
                for(let i=40; i<460; i+=50) {{
                    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 340); ctx.stroke();
                }}
                for(let j=20; j<340; j+=40) {{
                    ctx.beginPath(); ctx.moveTo(40, j); ctx.lineTo(460, j); ctx.stroke();
                }}
                
                ctx.fillStyle = '#db2777';
                ctx.font = 'bold 12px Pretendard';
                ctx.fillText("타겟 주변 경도 범위 (Longitude)", 170, 368);
                
                ctx.save();
                ctx.translate(15, 200);
                ctx.rotate(-Math.PI/2);
                ctx.fillText("타겟 주변 위도 범위 (Latitude)", -80, 0);
                ctx.restore();
                
                for(let i=0; i<pts.length; i++) {{
                    let p = pts[i];
                    let cx = 40 + ((p.x - ({lon-30})) / 60) * 400;
                    let cy = 340 - ((p.y - ({lat-30})) / 60) * 320;
                    if(cx >= 40 && cx <= 460 && cy >= 0 && cy <= 340) {{
                        ctx.beginPath();
                        ctx.arc(cx, cy, 4, 0, 2*Math.PI);
                        ctx.fillStyle = p.color;
                        ctx.fill();
                    }}
                }}
                
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
            <h3 style="margin-top:0; color:#db2777;">🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#64748b; line-height:1.7; font-size:14px; margin:0;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 위험 격자 스펙트럼 스캔이 완료되었으며, 
                인근 실제 지진 중심핵 코어 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
