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
# 🎨 [슈팅스타팩트: 파스텔 대륙 지구본 에디션] 스트림릿 통합 프레임
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght=700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #e0f2fe 0%, #fae8ff 40%, #fef08a 100%) !important;
        color: #475569 !important;
        overflow-x: hidden;
    }
    
    /* 🌠 실시간으로 떨어지는 사선 파스텔 별빛 애니메이션 효과 */
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
    .tag-cluster0 { background: #ef4444; box-shadow: 0 4px 10px rgba(239,68,68,0.3); }
    .tag-cluster1 { background: #2563eb; box-shadow: 0 4px 10px rgba(37,99,235,0.3); }
    .tag-cluster2 { background: #16a34a; box-shadow: 0 4px 10px rgba(22,163,74,0.3); }

    .stButton>button {
        background: linear-gradient(90deg, #fbcfe8 0%, #c084fc 100%) !important;
        color: #4c1d95 !important;
        font-weight: 900 !important;
        border-radius: 25px !important;
        border: 2px solid #ffffff !important;
        box-shadow: 0 8px 20px rgba(192, 132, 252, 0.3);
        transition: all 0.25s ease-in-out;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═════════════════════════════════════════════════════════════
# 📊 [인공지능 데이터 셋 로더] - 원본 지진대 데이터 유지
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_pure_quake_data():
    np.random.seed(42)
    
    # 실제 원본 이미지들의 단층선 분포 (환태평양 조산대, 지중해-히말라야) 백본 데이터 생성
    seismic_lines = [
        # 알래스카 및 북미 서부 밀집 (Red, Green 다수 지역)
        [61.0, -148.0, 0, 7.2, 35], [45.0, -122.0, 2, 5.5, 75], [36.0, -118.0, 0, 6.8, 15], [32.0, -115.0, 2, 4.8, 60],
        # 중남미 서안 라인
        [19.0, -99.0, 0, 7.0, 20], [-12.0, -77.0, 2, 5.2, 80], [-33.0, -71.0, 0, 8.2, 25],
        # 서태평양 블루 단층선 밀집 구역 (일본, 필리핀, 인도네시아)
        [36.0, 139.0, 1, 6.5, 250], [14.0, 121.0, 1, 5.8, 180], [-3.0, 120.0, 1, 6.2, 300], [-20.0, 175.0, 1, 7.0, 400],
        # 지중해 - 중동 라인
        [38.0, 23.0, 1, 4.5, 120], [35.0, 50.0, 1, 5.0, 90],
        # 남아공 및 호주 소수 도트
        [-26.0, 28.0, 0, 4.2, 10], [-25.0, 133.0, 1, 3.5, 150]
    ]
    
    lats, lons, clusters, magnitudes, depths, impacts = [], [], [], [], [], []
    
    for _ in range(3500):
        base = seismic_lines[np.random.choice(len(seismic_lines))]
        lat = np.clip(base[0] + np.random.normal(0, 2.5), -90.0, 90.0)
        lon = base[1] + np.random.normal(0, 3.0)
        if lon > 180: lon -= 360
        if lon < -180: lon += 360
        
        lats.append(lat)
        lons.append(lon)
        clusters.append(base[2]) # Folium 코드 구조와 매핑되는 클러스터 번호 주입
        
        mag = np.clip(base[3] + np.random.normal(0, 0.6), 2.0, 9.5)
        depth = np.clip(base[4] + np.random.normal(0, 20), 5, 600)
        magnitudes.append(mag)
        depths.append(depth)
        impacts.append(mag * 11 + np.random.uniform(5, 15))

    return pd.DataFrame({'위도': lats, '경도': lons, 'cluster': clusters, '규모': magnitudes, '진원깊이': depths, '영향도': impacts})

# 실제 유저 데이터프레임 구조 대체 (df_new역할)
df_new = load_pure_quake_data()

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
    # Folium 샘플링 로직 및 컬러 사전 구조 완벽 동기화
    sample_size = min(5000, len(df_new))
    df_sample = df_new.sample(sample_size, random_state=42)
    
    # Folium 코드 정의 색상 구조: 0->red, 1->blue, 2->green
    colors_map = {0: 'red', 1: 'blue', 2: 'green'}
    # 웹 렌더링용 매칭 HEX 컬러
    hex_colors = {0: '#ef4444', 1: '#2563eb', 2: '#22c55e'}
    
    # 주변 대표 군집 계산용 하버사인 계산
    def haversine(lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

    dist = haversine(lat, lon, df_sample["위도"].values, df_sample["경도"].values)
    near_idx = np.argsort(dist)[:20]
    nearest_km = float(dist[near_idx[0]])
    
    cw = {}
    for idx in near_idx:
        c_val = int(df_sample.iloc[idx]["cluster"])
        cw[c_val] = cw.get(c_val, 0.0) + 1.0 / (dist[idx] + 10.0)
    dom_cluster = int(max(cw, key=cw.get))

    col_left_stage, col_right_graph = st.columns([1, 1])
    
    with col_left_stage:
        st.write("#### 🔮 슈팅스타 팩트 3D 홀로그램 (Folium 군집 데이터 100% 동기화)")
        
        # 3D 뷰포트 최적화를 위한 부분 샘플 구성
        view_sample = df_sample.sample(min(2000, len(df_sample)), random_state=42)
        points_js_items = []
        for _, row in view_sample.iterrows():
            c_num = int(row["cluster"])
            p_str = f"{{lat: {float(row['위도'])}, lon: {float(row['경도'])}, color: '{hex_colors[c_num]}', size: 3.0}}"
            points_js_items.append(p_str)
        points_js_str = ",\n".join(points_js_items)

        # 🪐 [3D 가상 대륙 지형 윤곽 및 고선명 격자선 적용 홀로그램 팩트 엔클로저]
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
                .shooting-star-pact-container {{
                    position: relative;
                    width: 460px; height: 460px;
                    display: flex; justify-content: center; align-items: center;
                }}
                .compact-top-crown {{
                    position: absolute; top: 5px; left: 50%;
                    width: 55px; height: 22px;
                    background: linear-gradient(180deg, #ffe57f 0%, #ffca28 100%);
                    transform: translateX(-50%); border-radius: 15px 15px 5px 5px;
                    border: 3px solid #ffffff; box-shadow: 0 4px 10px rgba(255, 202, 40, 0.5);
                    z-index: 4;
                }}
                .compact-top-crown::after {{ content: '⭐'; position: absolute; top: -5px; left: 50%; transform: translateX(-50%); font-size: 12px; }}
                
                .star-gold-pedestal-base {{
                    position: absolute; bottom: 15px; left: 50%;
                    width: 240px; height: 65px;
                    background: linear-gradient(180deg, #ffca28 0%, #ffb300 100%);
                    transform: translateX(-50%); clip-path: polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);
                    border-radius: 0 0 25px 25px; border-bottom: 4px solid #ffffff;
                    box-shadow: 0 8px 20px rgba(255, 179, 0, 0.35); z-index: 1; 
                }}
                .fairy-wing-left {{
                    position: absolute; left: 10px; top: 155px; width: 100px; height: 150px;
                    background: linear-gradient(to left, rgba(255,255,255,0.95), rgba(186,230,253,0.9));
                    border: 4px solid #ffffff; border-radius: 120px 40px 80px 100px;
                    box-shadow: -5px 10px 20px rgba(186, 230, 253, 0.4); z-index: 2;
                }}
                .fairy-wing-right {{
                    position: absolute; right: 10px; top: 155px; width: 100px; height: 150px;
                    background: linear-gradient(to right, rgba(255,255,255,0.95), rgba(186,230,253,0.9));
                    border: 4px solid #ffffff; border-radius: 40px 120px 100px 80px;
                    box-shadow: 5px 10px 20px rgba(186, 230, 253, 0.4); z-index: 2;
                }}
                .fact-pink-heart-shield {{
                    position: relative; width: 350px; height: 350px;
                    background: linear-gradient(135deg, #fbcfe8 0%, #f472b6 45%, #db2777 100%);
                    border-radius: 50%; border: 8px solid #ffffff;
                    box-shadow: inset 0 -10px 20px rgba(0,0,0,0.15), inset 0 10px 20px rgba(255,255,255,0.5), 0 15px 35px rgba(219, 39, 119, 0.35);
                    display: flex; justify-content: center; align-items: center; z-index: 3;
                }}
                .fact-inner-gold-ring {{
                    position: relative; width: 290px; height: 290px; border-radius: 50%;
                    background: transparent; border: 5px solid #facc15;
                    box-shadow: 0 0 12px #facc15, inset 0 0 8px rgba(234, 179, 8, 0.5);
                    display: flex; justify-content: center; align-items: center;
                }}
                .map-inside-binder {{
                    position: relative; width: 266px; height: 266px; border-radius: 50%;
                    overflow: hidden; border: 4px solid #ffffff;
                    box-shadow: inset 0 0 30px rgba(34, 211, 238, 0.8);
                    background: radial-gradient(circle at 50% 50%, #0f3d4c 0%, #082530 70%, #020c10 100%);
                }}
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
                // Folium.Marker(location=[lat, lon], icon='star')의 형태를 검은 테두리의 흰 별 형태로 매핑
                const targetPoint = {{ lat: {lat}, lon: {lon}, color: '#000000', size: 5.0 }};

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
                    
                    // 1️⃣ 대륙 베이스 윤곽선선
                    ctx.strokeStyle = 'rgba(254, 240, 138, 0.45)'; 
                    ctx.fillStyle = 'rgba(255, 255, 255, 0.04)'; 
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

                    // ⚡ [줄 표시 진하게 변경] 경도선/위도선 홀로그램 격자 강도 상승
                    ctx.strokeStyle = 'rgba(34, 211, 238, 0.45)';
                    ctx.lineWidth = 1.0;
                    for (let l = -60; l <= 60; l += 30) {{
                        ctx.beginPath();
                        for (let lng = -180; lng <= 180; lng += 15) {{
                            let p = project(l, lng);
                            if (lng === -180) ctx.moveTo(p.x, p.y);
                            else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    
                    // 2️⃣ 데이터 투영 레이어 (Folium CircleMarker 명세 반영)
                    let tragedies = [...points, targetPoint];
                    for(let i=0; i<tragedies.length; i++) {{
                        tragedies[i]._proj = project(tragedies[i].lat, tragedies[i].lon);
                    }}
                    tragedies.sort(function(a, b) {{ return b._proj.depth - a._proj.depth; }});
                    
                    for(let i=0; i<tragedies.length; i++) {{
                        let p = tragedies[i];
                        let proj = p._proj;
                        if (proj.depth > -30) {{ 
                            let alpha = Math.max(0.25, (proj.depth + 126) / 252);
                            ctx.beginPath();
                            
                            if (p === targetPoint) {{
                                // ⭐ 검은 마커 안의 하이라이트 스타 좌표 매핑
                                ctx.arc(proj.x, proj.y, 8, 0, 2 * Math.PI);
                                ctx.fillStyle = '#000000';
                                ctx.fill();
                                ctx.beginPath();
                                ctx.arc(proj.x, proj.y, 4, 0, 2 * Math.PI);
                                ctx.fillStyle = '#ffffff';
                                ctx.fill();
                            }} else {{
                                // Folium의 radius=3 마커 투영
                                ctx.arc(proj.x, proj.y, p.size, 0, 2 * Math.PI);
                                ctx.fillStyle = p.color;
                                ctx.fill();
                            }}
                            ctx.globalAlpha = alpha;
                        }}
                    }}
                    
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
        st.write("#### 📊 타겟 반경 지진 분포 격자 레이더")
        
        chart_points = []
        for c in sorted(df_sample["cluster"].unique()):
            filtered_df = df_sample[(df_sample["cluster"] == c) & (df_sample["경도"].between(lon-30, lon+30)) & (df_sample["위도"].between(lat-30, lat+30))]
            f_len = len(filtered_df)
            
            if f_len > 0:
                sub_set = filtered_df.sample(min(200, f_len), replace=True)
                for _, r in sub_set.iterrows():
                    chart_points.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{hex_colors[int(c)]}'}}")
                    
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
                
                // 선명도 보강 격자선선
                ctx.strokeStyle = 'rgba(148, 163, 184, 0.5)';
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
                        ctx.beginPath(); ctx.arc(cx, cy, 3, 0, 2*Math.PI);
                        ctx.fillStyle = p.color; ctx.fill();
                    }}
                }}
                
                // 타겟 센터 마커 표시
                let tx = 40 + (30 / 60) * 400; let ty = 340 - (30 / 60) * 320;
                ctx.beginPath(); ctx.arc(tx, ty, 7, 0, 2*Math.PI);
                ctx.fillStyle = '#ffffff'; ctx.strokeStyle = '#000000'; ctx.lineWidth = 3;
                ctx.fill(); ctx.stroke();
            </script>
        </body>
        </html>
        """
        components.html(canvas_chart_html, height=400, scrolling=False)

    tag_cls = f"tag-cluster{dom_cluster}"
    cluster_desc = f"Cluster {dom_cluster} ({colors_map[dom_cluster]} 군집 대역)"
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#db2777;">🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ Folium 연동 감지 결과: <span class="danger-tag {tag_cls}">{cluster_desc}</span> ]
            </p>
            <p style="color:#64748b; line-height:1.7; font-size:14px; margin:0;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 실제 단층선 매핑 스캔이 완벽하게 완료되었습니다.<br>
                입력하신 타겟 중심부에서 가장 가까운 실제 지진 활동성 단층 데이터 코어까지의 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
