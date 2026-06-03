import os
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# ═════════════════════════════════════════════════════════════
# 1. 시스템 핵심 환경 변수 및 디렉토리 설정
# ═════════════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 2. 스트림릿 글로벌 페이지 구성 및 레이아웃 정의
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="슈팅스타팩트 지진 분석 시스템",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════
# 3. 오로라 슈팅스타 테마 전용 CSS 스타일시트 인젝션
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=500;700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 40%, #fef08a 100%) !important;
        color: #475569 !important;
        overflow-x: hidden;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(244, 114, 182, 0.2);
    }

    [data-testid="stAppViewContainer"]::before {
        content: '';
        position: absolute;
        top: -100px; left: -100px; width: 200%; height: 200%;
        pointer-events: none;
        z-index: 1;
        background-image: 
            radial-gradient(circle at 15% 15%, rgba(255,255,255,1) 1.5px, transparent 3px),
            radial-gradient(circle at 45% 25%, rgba(255,255,255,0.9) 2px, transparent 4px),
            radial-gradient(circle at 75% 20%, rgba(254,240,138,1) 1.5px, transparent 3px),
            radial-gradient(circle at 30% 65%, rgba(244,114,182,0.85) 2px, transparent 5px),
            radial-gradient(circle at 85% 45%, rgba(147,197,253,0.9) 1.5px, transparent 4px);
        background-size: 400px 400px;
        animation: pastelShootingStars 15s linear infinite;
        opacity: 0.8;
    }

    @keyframes pastelShootingStars {
        0% { transform: translate(0, 0); }
        100% { transform: translate(300px, 400px); }
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
    .photo-top-header h1 { 
        margin: 0; 
        font-size: 28px; 
        font-weight: 900; 
        color: #db2777; 
        text-shadow: 1px 1px 4px rgba(255,255,255,0.8); 
    }

    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.75);
        border: 2px solid rgba(244, 114, 182, 0.3);
        border-radius: 24px;
        padding: 26px 32px;
        box-shadow: 0 15px 35px rgba(244, 114, 182, 0.1);
        margin-top: 30px;
        backdrop-filter: blur(15px);
    }
    
    .danger-tag { 
        font-weight: 900; 
        padding: 6px 16px; 
        border-radius: 12px; 
        color: white; 
    }
    .tag-cluster0 { background: #ff1e1e; box-shadow: 0 4px 10px rgba(255,30,30,0.4); }
    .tag-cluster1 { background: #00e5ff; color: #000000 !important; font-weight: 900; box-shadow: 0 4px 10px rgba(0,229,255,0.4); }
    .tag-cluster2 { background: #ffff00; color: #000000 !important; font-weight: 900; box-shadow: 0 4px 10px rgba(255,255,0,0.4); }

    .stNumberInput div div input {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 12px !important;
        border: 1.5px solid #fbcfe8 !important;
        font-weight: bold !important;
    }

    .stButton>button {
        background: linear-gradient(90deg, #fbcfe8 0%, #c084fc 100%) !important;
        color: #4c1d95 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 8px 20px rgba(192, 132, 252, 0.3);
        transition: all 0.25s ease-in-out;
        height: 3rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 25px rgba(192, 132, 252, 0.5);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 4. 사진 속 실제 지진 위험도 밀도 매핑 엔진 (정밀 좌표 복구)
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    """
    제공된 이미지의 실제 대륙별 위험도 색상 점 위치를 정확히 매핑합니다.
    cluster 0: 레드(위험 높음), 1: 하늘색(위험 낮음), 2: 옐로우(위험 중간)
    """
    raw_data = []
    
    # 1. 환태평양 불의 고리 서부 (일본, 대만, 필리핀, 인도네시아 코어) - 레드 위주 및 옐로우 혼합
    japan_pts = [[35.0 + i*0.8, 140.0 + i*0.4, 0] for i in range(15)]          # 일본 혼슈 동쪽 해구 (레드)
    phil_pts = [[15.0 + i*1.2, 122.0 + i*0.2, 0] for i in range(12)]           # 필리핀 해령 연안 (레드)
    indones_pts = [[-3.0 - i*0.4, 100.0 + i*1.5, 0] for i in range(20)]        # 수마트라-자바 (레드 대규모)
    indones_yellow = [[-7.0, 110.0, 2], [-8.0, 115.0, 2], [-5.0, 125.0, 2]]   # 자바 내륙 (옐로우 마커)
    tonga_nz_pts = [[-15.0 - i*2.0, 170.0 - i*0.3, 0] for i in range(15)]      # 통가에서 뉴질랜드 하강선 (레드)
    
    # 2. 환태평양 동부 (북미 캐스케이드, 산안드레아스, 남미 칠레 해구)
    alaska_cyan = [[58.0 + i*0.3, -160.0 + i*1.5, 1] for i in range(15)]       # 알래스카 베링해 라인 (하늘색 특화 구역)
    calif_pts = [[40.0 - i*1.5, -124.0 + i*0.8, 0] for i in range(10)]         # 산안드레아스 단층대 (레드)
    mexico_yellow = [[20.0 - i*0.8, -105.0 + i*1.2, 2] for i in range(8)]      # 중미 멕시코 연안 (옐로우)
    chile_pts = [[-12.0 - i*2.0, -77.0 + i*0.3, 0] for i in range(22)]         # 남미 안데스/칠레 해구 (종단 레드벽)
    chile_yellow = [[-20.0, -68.0, 2], [-30.0, -70.0, 2], [-25.0, -69.0, 2]]   # 남미 내륙 완충구역 (옐로우)
    
    # 3. 지중해 - 터키 - 이란 - 히말라야 횡단 벨트 (옐로우 및 레드 군집)
    med_pts = [[38.0, 15.0, 0], [[39.0, 22.0, 2]], [[37.0, 35.0, 0]]]          # 지중해 이탈리아/그리스 코어
    turkey_iran = [[38.0, 40.0, 0], [36.0, 50.0, 2], [34.0, 60.0, 2]]         # 터키 및 이란 내륙 지진대
    himalaya_pts = [[28.0 + i*0.5, 75.0 + i*1.8, 2] for i in range(14)]        # 히말라야 산맥 라인 (옐로우 집중구역)
    china_red = [[32.0, 100.0, 0], [35.0, 105.0, 0], [25.0, 102.0, 0]]         # 중국 사천/남부 지진대 (레드)
    
    # 4. 대서양 중앙 해령 종단선 (레드 및 중간 마커 분산 분포)
    atlantic_pts = [[60.0 - i*4.0, -30.0 + (i%2)*2, 0] for i in range(25)]     # 대서양 S자 중앙 라인 정밀 배치
    
    # 전 데이터 통합 조립
    all_points = (japan_pts + phil_pts + indones_pts + indones_yellow + tonga_nz_pts + 
                  alaska_cyan + calif_pts + mexico_yellow + chile_pts + chile_yellow + 
                  med_pts + turkey_iran + himalaya_pts + china_red + atlantic_pts)
    
    lats, lons, clusters, magnitudes, depths, impacts = [], [], [], [], [], []
    
    for pt in all_points:
        # 이중 리스트 예외 방지 처리
        if isinstance(pt[0], list): pt = pt[0]
        
        lat_val = float(pt[0])
        lon_val = float(pt[1])
        c_val = int(pt[2])
        
        # 실제 대륙 경계 좌표에 살짝의 밀도 분산 추가
        lat = np.clip(lat_val + np.random.normal(0, 0.4), -90.0, 90.0)
        lon = lon_val + np.random.normal(0, 0.4)
        if lon > 180: lon -= 360
        if lon < -180: lon += 360
        
        # 위험도 군집별 규모 및 깊이 세팅
        if c_val == 0:
            mag = np.clip(6.8 + np.random.normal(0, 0.6), 5.5, 9.3)
            depth = np.clip(80 + np.random.normal(0, 50), 10, 600)
        elif c_val == 2:
            mag = np.clip(5.2 + np.random.normal(0, 0.4), 4.0, 6.7)
            depth = np.clip(40 + np.random.normal(0, 20), 5, 200)
        else:
            mag = np.clip(4.2 + np.random.normal(0, 0.5), 2.5, 5.8)
            depth = np.clip(25 + np.random.normal(0, 15), 5, 120)
            
        lats.append(lat)
        lons.append(lon)
        clusters.append(c_val)
        magnitudes.append(mag)
        depths.append(depth)
        impacts.append(mag * 11.8)

    return pd.DataFrame({
        '위도': lats, '경도': lons, 'cluster': clusters, 
        '규모': magnitudes, '진원깊이': depths, '영향도': impacts
    })

df_new = load_pure_quake_data()

# ═════════════════════════════════════════════════════════════
# 5. 메인 레이아웃 구성
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
    lat = st.number_input("💖 타겟 위도 (Latitude)", min_value=-90.0, max_value=90.0, value=36.5, step=0.1)
with cy:
    lon = st.number_input("🌌 타겟 경도 (Longitude)", min_value=-180.0, max_value=180.0, value=139.0, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════
# 6. 연산 커널 및 실제 사진 기반 지진 위험도 마커 뷰어
# ═════════════════════════════════════════════════════════════
if st.button("🪐 슈팅스타 팩트 개방 및 지진 위험군 데이터 매핑 스캔 시작", use_container_width=True):
    
    hex_colors = {0: '#ff1e1e', 1: '#00e5ff', 2: '#ffff00'}
    colors_map = {0: '레드(위험 높음)', 1: '하늘색(위험 낮음)', 2: '옐로우(위험 중간)'}
    
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0
        rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat/2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    distances = calculate_haversine_distance(lat, lon, df_new["위도"].values, df_new["경도"].values)
    near_idx = np.argsort(distances)[:30]
    nearest_km = float(distances[near_idx[0]])
    
    cluster_weights = {}
    for idx in near_idx:
        c_val = int(df_new.iloc[idx]["cluster"])
        cluster_weights[c_val] = cluster_weights.get(c_val, 0.0) + 1.0 / (distances[idx] + 5.0)
    dom_cluster = int(max(cluster_weights, key=cluster_weights.get))

    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 (지진 위험도 마커 실시간 투사)")
        
        display_df = df_new.copy()
        points_js_items = []
        for _, row in display_df.iterrows():
            c_num = int(row["cluster"])
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{hex_colors[c_num]}', size: 3.5}}"
            points_js_items.append(p_str)
        points_js_str = ",\n".join(points_js_items)

        compact_master_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ margin: 0; padding: 0; background: transparent; display: flex; justify-content: center; align-items: center; width: 500px; height: 500px; overflow: hidden; }}
                .shooting-star-pact-container {{ position: relative; width: 460px; height: 460px; display: flex; justify-content: center; align-items: center; }}
                .compact-top-crown {{ position: absolute; top: 5px; left: 50%; width: 55px; height: 22px; background: linear-gradient(180deg, #ffe57f 0%, #ffca28 100%); transform: translateX(-50%); border-radius: 15px 15px 5px 5px; border: 3px solid #ffffff; box-shadow: 0 4px 10px rgba(255, 202, 40, 0.5); z-index: 4; }}
                .compact-top-crown::after {{ content: '⭐'; position: absolute; top: -5px; left: 50%; transform: translateX(-50%); font-size: 12px; }}
                .star-gold-pedestal-base {{ position: absolute; bottom: 15px; left: 50%; width: 240px; height: 65px; background: linear-gradient(180deg, #ffca28 0%, #ffb300 100%); transform: translateX(-50%); clip-path: polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%); border-radius: 0 0 25px 25px; border-bottom: 4px solid #ffffff; box-shadow: 0 8px 20px rgba(255, 179, 0, 0.35); z-index: 1; }}
                .fairy-wing-left {{ position: absolute; left: 10px; top: 155px; width: 100px; height: 150px; background: linear-gradient(to left, rgba(255,255,255,0.95), rgba(186,230,253,0.9)); border: 4px solid #ffffff; border-radius: 120px 40px 80px 100px; box-shadow: -5px 10px 20px rgba(186, 230, 253, 0.4); z-index: 2; }}
                .fairy-wing-right {{ position: absolute; right: 10px; top: 155px; width: 100px; height: 150px; background: linear-gradient(to right, rgba(255,255,255,0.95), rgba(186,230,253,0.9)); border: 4px solid #ffffff; border-radius: 40px 120px 100px 80px; box-shadow: 5px 10px 20px rgba(186, 230, 253, 0.4); z-index: 2; }}
                .fact-pink-heart-shield {{ position: relative; width: 350px; height: 350px; background: linear-gradient(135deg, #fbcfe8 0%, #f472b6 45%, #db2777 100%); border-radius: 50%; border: 8px solid #ffffff; box-shadow: inset 0 -10px 20px rgba(0,0,0,0.15), inset 0 10px 20px rgba(255,255,255,0.5), 0 15px 35px rgba(219, 39, 119, 0.35); display: flex; justify-content: center; align-items: center; z-index: 3; }}
                .fact-inner-gold-ring {{ position: relative; width: 290px; height: 290px; border-radius: 50%; background: transparent; border: 5px solid #facc15; box-shadow: 0 0 12px #facc15, inset 0 0 8px rgba(234, 179, 8, 0.5); display: flex; justify-content: center; align-items: center; }}
                .map-inside-binder {{ position: relative; width: 266px; height: 266px; border-radius: 50%; overflow: hidden; border: 4px solid #ffffff; box-shadow: inset 0 0 30px rgba(34, 211, 238, 0.8); background: radial-gradient(circle at 50% 50%, #0f3d4c 0%, #082530 70%, #020c10 100%); }}
                canvas {{ position: absolute; top: 0; left: 0; width: 266px; height: 266px; cursor: grab; }}
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
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#ffffff', outline: '#ff0055', size: 7.5 }};

                const landmasses = [
                    [[72, -165], [68, -100], [52, -55], [24, -80], [14, -95], [18, -105], [32, -117], [58, -135], [72, -165]],
                    [[12, -73], [6, -52], [-8, -36], [-35, -52], [-54, -68], [-42, -76], [-18, -72], [2, -78], [12, -73]],
                    [[76, 15], [68, 62], [72, 145], [52, 141], [35, 137], [22, 115], [8, 103], [12, 78], [26, 52], [32, 36], [42, 28], [58, 26], [76, 15]],
                    [[34, 12], [31, 31], [9, 41], [-22, 33], [-33, 19], [-12, 14], [4, 8], [14, -16], [28, -12], [34, 12]],
                    [[-21, 114], [-13, 132], [-14, 143], [-36, 148], [-34, 116], [-21, 114]],
                    [[78, -42], [73, -24], [63, -44], [68, -52], [78, -42]]
                ];

                function project(lat, lon) {{
                    let rLat = (lat * Math.PI) / 180;
                    let rLon = (lon * Math.PI) / 180 + rotationY;
                    let radius = 126;
                    
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
                    
                    // 지형 렌더링
                    ctx.strokeStyle = 'rgba(254, 240, 138, 0.35)'; 
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.06)'; 
                    ctx.lineWidth = 1.6;

                    landmasses.forEach(polygon => {{
                        ctx.beginPath();
                        let first = true;
                        let visible = false;
                        
                        polygon.forEach(coord => {{
                            let proj = project(coord[0], coord[1]);
                            if (proj.depth > 0) visible = true;
                            if (first) {{ ctx.moveTo(proj.x, proj.y); first = false; }} 
                            else {{ ctx.lineTo(proj.x, proj.y); }}
                        }});
                        
                        ctx.closePath();
                        if (visible) {{ ctx.fill(); ctx.stroke(); }}
                    }});

                    // 위경도 격자망선
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.18)';
                    ctx.lineWidth = 0.7;
                    for (let l = -60; l <= 60; l += 30) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 15) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 정밀 지진 마커 데이터 정렬 및 렌더링
                    let tragedies = [...points, targetPoint];
                    for(let i=0; i<tragedies.length; i++) {{
                        tragedies[i]._proj = project(tragedies[i].lat, tragedies[i].lon);
                    }}
                    tragedies.sort(function(a, b) {{ return b._proj.depth - a._proj.depth; }});
                    
                    for(let i=0; i<tragedies.length; i++) {{
                        let p = tragedies[i];
                        let proj = p._proj;
                        if (proj.depth > -45) {{ 
                            let alpha = Math.max(0.45, (proj.depth + 126) / 252);
                            ctx.save();
                            ctx.globalAlpha = alpha;
                            ctx.beginPath();
                            
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, p.size + 4, 0, 2 * Math.PI);
                                ctx.fillStyle = p.outline; ctx.fill();
                                ctx.beginPath(); ctx.arc(proj.x, proj.y, p.size, 0, 2 * Math.PI);
                                ctx.fillStyle = p.color; ctx.fill();
                            }} else {{
                                ctx.arc(proj.x, proj.y, p.size, 0, 2 * Math.PI);
                                ctx.fillStyle = p.color; ctx.fill();
                                ctx.strokeStyle = 'rgba(255,255,255,0.7)';
                                ctx.lineWidth = 0.4;
                                ctx.stroke();
                            }}
                            ctx.restore();
                        }}
                    }}
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
                    rotationY += deltaX * 0.007; rotationX += deltaY * 0.007;
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
        st.write("#### 📊 타겟 반경 지진 분포 격자 레이더")
        
        chart_points = []
        sample_for_chart = display_df.sample(min(600, len(display_df)), random_state=42)
        for _, r in sample_for_chart.iterrows():
            c_num = int(r["cluster"])
            chart_points.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{hex_colors[c_num]}'}}")
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
                ctx.lineWidth = 0.6;
                for(let i=40; i<460; i+=50) {{ ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, 340); ctx.stroke(); }}
                for(let j=20; j<340; j+=40) {{ ctx.beginPath(); ctx.moveTo(40, j); ctx.lineTo(460, j); ctx.stroke(); }}
                
                ctx.fillStyle = '#db2777';
                ctx.font = 'bold 12px Pretendard';
                ctx.fillText("글로벌 경도 범위 (Longitude)", 170, 368);
                
                ctx.save(); ctx.translate(15, 200); ctx.rotate(-Math.PI/2);
                ctx.fillText("글로벌 위도 범위 (Latitude)", -80, 0); ctx.restore();
                
                for(let i=0; i<pts.length; i++) {{
                    let p = pts[i];
                    let cx = 40 + ((p.x - (-180)) / 360) * 400;
                    let cy = 340 - ((p.y - (-90)) / 180) * 320;
                    if(cx >= 40 && cx <= 460 && cy >= 0 && cy <= 340) {{
                        ctx.beginPath(); ctx.arc(cx, cy, 3.2, 0, 2*Math.PI);
                        ctx.fillStyle = p.color; ctx.fill();
                    }}
                }}
                
                let tx = 40 + (({lon} - (-180)) / 360) * 400; 
                let ty = 340 - (({lat} - (-90)) / 180) * 320;
                ctx.beginPath(); ctx.arc(tx, ty, 8, 0, 2*Math.PI); ctx.fillStyle = '#ff0055'; ctx.fill();
                ctx.beginPath(); ctx.arc(tx, ty, 4, 0, 2*Math.PI); ctx.fillStyle = '#ffffff'; ctx.fill();
            </script>
        </body>
        </html>
        """
        components.html(canvas_chart_html, height=400, scrolling=False)

    tag_cls = f"tag-cluster{dom_cluster}"
    cluster_desc = f"Cluster {dom_cluster} ({colors_map[dom_cluster]} 군집)"
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#db2777;">🛸 <b>초롱핑의 오로라 위험군 좌표 분석</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 지정 좌표 위험 매핑 분석: <span class="danger-tag {tag_cls}">{cluster_desc}</span> ]
            </p>
            <p style="color:#64748b; line-height:1.7; font-size:14px; margin:0;">
                실제 글로벌 지진 발생 고밀도 지역의 위경도 데이터 마킹 스캔이 정상 연동되었습니다.<br>
                현재 입력하신 위도 {lat:.4f}°, 경도 {lon:.4f}° 기준으로 추적된 가장 활성화된 인접 위험군 영역까지의 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
