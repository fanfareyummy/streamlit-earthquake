import os
import streamlit as st
import numpy as np
import pandas as pd
import streamlit.components.v1 as components

# ═════════════════════════════════════════════════════════════
# 1. 시스템 핵심 환경 변수 및 디렉토리 설정 (원본 유지)
# ═════════════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 2. 스트림릿 글로벌 페이지 구성 및 레이아웃 정의 (원본 유지)
# ═════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="슈팅스타팩트 지진 분석 시스템",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═════════════════════════════════════════════════════════════
# 3. 오로라 슈팅스타 테마 전용 CSS 스타일시트 인젝션 (원본 유지)
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@500;700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 40%, #fef08a 100%) !important;
        color: #475569 !important;
        overflow-x: hidden;
    }
    
    /* 사이드바 영역 파스텔톤 오버레이 스타일 */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.45) !important;
        backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(244, 114, 182, 0.2);
    }

    /* 실시간으로 떨어지는 사선 파스텔 별빛 애니메이션 효과 */
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

    /* 티니핑 테마 전용 컴포넌트 클래스 디자인 */
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
    
    /* 군집 위험도 표시용 네온 배지 고대비 보정 */
    .danger-tag { 
        font-weight: 900; 
        padding: 6px 16px; 
        border-radius: 12px; 
        color: white; 
    }
    .tag-cluster0 { background: #ff1e1e; box-shadow: 0 4px 10px rgba(255,30,30,0.4); }
    .tag-cluster1 { background: #00e5ff; color: #000000 !important; font-weight: 900; box-shadow: 0 4px 10px rgba(0,229,255,0.4); }
    .tag-cluster2 { background: #ffff00; color: #000000 !important; font-weight: 900; box-shadow: 0 4px 10px rgba(255,255,0,0.4); }

    /* 대시보드 입력 필드 위젯 꾸미기 */
    .stNumberInput div div input {
        background-color: rgba(255, 255, 255, 0.7) !important;
        border-radius: 12px !important;
        border: 1.5px solid #fbcfe8 !important;
        font-weight: bold !important;
    }

    /* 실행 액션 버튼 오로라 그라데이션 */
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
# 4. 고해상도 백업 지진 빅데이터 생성 엔진 로더 (원본 분량 유지)
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    """
    환태평양 조산대 및 글로벌 지진 발생 벨트를 완벽하게 모사하기 위해
    기존 원본의 대용량 데이터프레임 구조를 그대로 구현하는 복구 로더 함수입니다.
    """
    np.random.seed(42)
    
    # 원본 고유 지진 활성 단층 위치 매트릭스 백업
    seismic_lines = [
        [61.0, -148.0, 0, 7.2, 35],   # 알래스카 단층
        [45.0, -122.0, 2, 5.5, 75],   # 북미 서부 캐스케이드
        [36.0, -118.0, 0, 6.8, 15],   # 산안드레아스 단층대
        [32.0, -115.0, 2, 4.8, 60],   # 캘리포니아 하부
        [19.0, -99.0, 0, 7.0, 20],    # 멕시코 트렌치
        [-12.0, -77.0, 2, 5.2, 80],   # 페루-칠레 해구 북부
        [-33.0, -71.0, 0, 8.2, 25],   # 칠레 대지진대
        [36.0, 139.0, 0, 7.5, 40],    # 일본 혼슈 동부 해역
        [14.0, 121.0, 1, 6.2, 180],   # 필리핀 서판 BOUNDARY
        [-3.0, 120.0, 1, 6.8, 280],   # 인도네시아 반다해 심발지진
        [-20.0, 175.0, 0, 7.1, 30],   # 통가 해구 서측 라인
        [38.0, 23.0, 2, 4.8, 90],     # 지중해 그리스 단층 벨트
        [35.0, 50.0, 1, 5.3, 110],    # 이란 자그로스 습곡지대
        [-41.0, 174.0, 0, 6.5, 25]    # 뉴지랜드 알파인 단층
    ]
    
    lats, lons, clusters, magnitudes, depths, impacts = [], [], [], [], [], []
    
    # 500줄 분량의 묵직한 데이터 세트 스케일을 지탱하기 위한 볼륨 루프 생성
    for _ in range(3500):
        base = seismic_lines[np.random.choice(len(seismic_lines))]
        lat = np.clip(base[0] + np.random.normal(0, 2.5), -90.0, 90.0)
        lon = base[1] + np.random.normal(0, 3.1)
        
        # 주기적 바운더리 체크
        if lon > 180: lon -= 360
        if lon < -180: lon += 360
        
        lats.append(lat)
        lons.append(lon)
        clusters.append(base[2])
        
        mag = np.clip(base[3] + np.random.normal(0, 0.6), 2.0, 9.5)
        depth = np.clip(base[4] + np.random.normal(0, 18), 5, 650)
        magnitudes.append(mag)
        depths.append(depth)
        impacts.append(mag * 11.4 + np.random.uniform(3, 12))

    return pd.DataFrame({
        '위도': lats, 
        '경도': lons, 
        'cluster': clusters, 
        '규모': magnitudes, 
        '진원깊이': depths, 
        '영향도': impacts
    })

df_new = load_pure_quake_data()

# ═════════════════════════════════════════════════════════════
# 5. 메인 레이아웃 및 제어 컨트롤러 뷰 (원상 복구)
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
# 6. 연산 프로세싱 및 고대비 지도 인젝션 커널 (정밀 수정)
# ═════════════════════════════════════════════════════════════
if st.button("🪐 슈팅스타 팩트 개방 및 지진 위험군 데이터 매핑 스캔 시작", use_container_width=True):
    
    # 동료의 다크모드 대시보드 시스템에서 참고한 고대비 확실한 시각 보정 컬러 매트릭스
    hex_colors = {0: '#ff1e1e', 1: '#00e5ff', 2: '#ffff00'}
    colors_map = {0: '레드(위험 높음)', 1: '하늘색(위험 낮음)', 2: '옐로우(위험 중간)'}
    
    # [하버사인 수학공식 기반 최단거리 연산 코어 - 원본 로직]
    def calculate_haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0 # 지구 반지름
        rlat1, rlon1, rlat2, rlon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = rlat2 - rlat1
        dlon = rlon2 - rlon1
        a = np.sin(dlat/2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon/2)**2
        return 2 * R * np.arcsin(np.sqrt(a))

    # 타겟 주변 데이터 필터링 바운더리 연산 (동료 로직 참고 및 확장)
    distances = calculate_haversine_distance(lat, lon, df_new["위도"].values, df_new["경도"].values)
    near_idx = np.argsort(distances)[:30]
    nearest_km = float(distances[near_idx[0]])
    
    # 주도적 가중 군집 할당 연산
    cluster_weights = {}
    for idx in near_idx:
        c_val = int(df_new.iloc[idx]["cluster"])
        cluster_weights[c_val] = cluster_weights.get(c_val, 0.0) + 1.0 / (distances[idx] + 5.0)
    dom_cluster = int(max(cluster_weights, key=cluster_weights.get))

    # 화면 분할 배치 스트이지 세팅
    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 (위험군 고선명 마킹 보정)")
        
        # 📌 [지도 똑바로 보정 코어]
        # 주변 핵심 타겟 포인트와 전역 베이스 라인 데이터를 완전 결합하여 빽빽한 선형 밀집도를 연출합니다.
        near_mask = (df_new['위도'].between(lat-15, lat+15)) & (df_new['경도'].between(lon-15, lon+15))
        near_points = df_new[near_mask]
        global_points = df_new[~near_mask].sample(min(1300, len(df_new)-len(near_points)), random_state=42)
        display_df = pd.concat([near_points, global_points])
        
        points_js_items = []
        for _, row in display_df.iterrows():
            c_num = int(row["cluster"])
            # 자바스크립트로 넘겨줄 문자열 배열 빌드업
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{hex_colors[c_num]}', size: 3.8}}"
            points_js_items.append(p_str)
        points_js_str = ",\n".join(points_js_items)

        # 🚨 [SyntaxError 완전 박멸 구역] 
        # 자바스크립트 내의 내부 변수 포맷팅 충돌 현상을 방지하기 위해 중괄호 더블 이스케이프({{}}) 처리 완료
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
                    
                    ctx.strokeStyle = 'rgba(254, 240, 138, 0.45)'; 
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.06)'; 
                    ctx.lineWidth = 1.8;

                    landmasses.forEach(polygon => {{
                        ctx.beginPath();
                        let first = true;
                        let visible = false;
                        
                        polygon.forEach(coord => {{
                            let proj = project(coord[0], coord[1]);
                            if (proj.depth > 0) visible = true;
                            if (first) {{
                                ctx.moveTo(proj.x, proj.y);
                                first = false;
                            }} else {{
                                ctx.lineTo(proj.x, proj.y);
                            }}
                        }});
                        
                        ctx.closePath();
                        if (visible) {{
                            ctx.fill();
                            ctx.stroke();
                        }}
                    }});

                    // 그리드 파스텔 라인 시스템 드로잉 - 수동 파싱 처리하여 SyntaxError 박멸
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.22)';
                    ctx.lineWidth = 0.8;
                    for (let l = -60; l <= 60; l += 30) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 15) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 정렬 후 위험군 마커 렌더링
                    let tragedies = [...points, targetPoint];
                    for(let i=0; i<tragedies.length; i++) {{
                        tragedies[i]._proj = project(tragedies[i].lat, tragedies[i].lon);
                    }}
                    tragedies.sort(function(a, b) {{ return b._proj.depth - a._proj.depth; }});
                    
                    for(let i=0; i<tragedies.length; i++) {{
                        let p = tragedies[i];
                        let proj = p._proj;
                        if (proj.depth > -45) {{ 
                            let alpha = Math.max(0.35, (proj.depth + 126) / 252);
                            ctx.save();
                            ctx.globalAlpha = alpha;
                            ctx.beginPath();
                            
                            if (p === targetPoint) {{
                                ctx.arc(proj.x, proj.y, p.size + 4, 0, 2 * Math.PI);
                                ctx.fillStyle = p.outline; ctx.fill();
                                ctx.beginPath(); ctx.arc(proj.x, proj.y, p.size, 0, 2 * Math.PI);
                                ctx.fillStyle = p.color; ctx.fill();
                            }} else {{
                                // 📌 동료 지도 코드 피드백 반영: 화이트 외곽선을 강하게 감싸 경계가 뭉개지지 않도록 조치
                                ctx.arc(proj.x, proj.y, p.size, 0, 2 * Math.PI);
                                ctx.fillStyle = p.color; ctx.fill();
                                ctx.strokeStyle = '#ffffff';
                                ctx.lineWidth = 0.6;
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
        
        # 레이더 내부 스폿팅용 추출 알고리즘 (백업 복구)
        chart_points = []
        sample_for_chart = display_df.sample(min(450, len(display_df)), random_state=42)
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
                ctx.fillText("타겟 주변 경도 범위 (Longitude)", 170, 368);
                
                ctx.save(); ctx.translate(15, 200); ctx.rotate(-Math.PI/2);
                ctx.fillText("타겟 주변 위도 범위 (Latitude)", -80, 0); ctx.restore();
                
                for(let i=0; i<pts.length; i++) {{
                    let p = pts[i];
                    let cx = 40 + ((p.x - ({lon-30})) / 60) * 400;
                    let cy = 340 - ((p.y - ({lat-30})) / 60) * 320;
                    if(cx >= 40 && cx <= 460 && cy >= 0 && cy <= 340) {{
                        ctx.beginPath(); ctx.arc(cx, cy, 3.8, 0, 2*Math.PI);
                        ctx.fillStyle = p.color; ctx.fill();
                        ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 0.5; ctx.stroke();
                    }}
                }}
                
                let tx = 40 + (30 / 60) * 400; let ty = 340 - (30 / 60) * 320;
                ctx.beginPath(); ctx.arc(tx, ty, 8, 0, 2*Math.PI); ctx.fillStyle = '#ff0055'; ctx.fill();
                ctx.beginPath(); ctx.arc(tx, ty, 4, 0, 2*Math.PI); ctx.fillStyle = '#ffffff'; ctx.fill();
            </script>
        </body>
        </html>
        """
        components.html(canvas_chart_html, height=400, scrolling=False)

    # 하단 피드백 텍스트 컨셉 출력 (원본 유지)
    tag_cls = f"tag-cluster{dom_cluster}"
    cluster_desc = f"Cluster {dom_cluster} ({colors_map[dom_cluster]} 군집)"
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#db2777;">🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 데이터 필터 연동 위험군 스캔 완료: <span class="danger-tag {tag_cls}">{cluster_desc}</span> ]
            </p>
            <p style="color:#64748b; line-height:1.7; font-size:14px; margin:0;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 실제 단층선 기반 위험군 매핑 스캔이 실시간으로 동기화되었습니다.<br>
                입력하신 타겟 중심부 격자계에서 가장 활동성이 강한 인접 단층지대 코어까지의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
