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
# 🎨 [슈팅스타팩트 최종판] 3D 지도 팩트 내부 안착 및 카툰 요정 날개 구현
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

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
    
    /* ✨ 하늘에서 떨어지는 마법 별빛 조각 (Star Dust) */
    .stAppViewContainer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 999;
        background-image: 
            radial-gradient(circle at 20% 10%, #fff 1px, transparent 2px),
            radial-gradient(circle at 75% 20%, #fff 2.5px, transparent 3.5px),
            radial-gradient(circle at 50% 40%, #ffb6c1 2px, transparent 3px),
            radial-gradient(circle at 30% 60%, #fff 1.5px, transparent 2.5px),
            radial-gradient(circle at 80% 80%, #e0f2fe 3px, transparent 5px);
        background-size: 600px 600px;
        animation: magicStarFall 10s linear infinite;
        opacity: 0.7;
    }
    @keyframes magicStarFall {
        0% { background-position: 0px 0px; }
        100% { background-position: 50px 600px; }
    }

    .stMainBlockContainer {
        background: radial-gradient(circle at 15% 25%, rgba(138, 43, 226, 0.2) 0%, transparent 60%),
                    radial-gradient(circle at 85% 75%, rgba(255, 105, 180, 0.2) 0%, transparent 60%);
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

    /* 🔮 슈팅스타팩트 입체 스테이지 (확대 및 고정) */
    .shooting-star-factory-stage {
        position: relative;
        width: 480px;
        height: 480px;
        margin: 20px auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    /* ⭐ 대형 황금 별 스탠드 베이스 */
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

    /* 👼 [리뉴얼] 화려하고 풍성한 카툰 요정 깃털 날개 조형 */
    .wing-group-left {
        position: absolute; left: -95px; top: 120px; width: 220px; height: 180px; z-index: 2;
        filter: drop-shadow(-10px 14px 20px rgba(236, 72, 153, 0.5));
    }
    .wing-group-right {
        position: absolute; right: -95px; top: 120px; width: 220px; height: 180px; z-index: 2;
        filter: drop-shadow(10px 14px 20px rgba(236, 72, 153, 0.5));
    }
    .feather { position: absolute; background: #fff; border: 3px solid #fff; }
    
    .l-f1 { width: 160px; height: 65px; top: 0; right: 0; background: linear-gradient(-60deg, #fff, #e0f2fe, #bae6fd); border-radius: 160px 15px 140px 80px; transform: rotate(-15deg); }
    .l-f2 { width: 140px; height: 60px; top: 40px; right: 20px; background: linear-gradient(-60deg, #fff, #fbcfe8, #f472b6); border-radius: 140px 15px 120px 70px; transform: rotate(-5deg); z-index: 2; }
    .l-f3 { width: 120px; height: 55px; top: 80px; right: 40px; background: linear-gradient(-60deg, #fff, #e0f2fe, #7dd3fc); border-radius: 120px 15px 100px 60px; transform: rotate(10deg); }

    .r-f1 { width: 160px; height: 65px; top: 0; left: 0; background: linear-gradient(60deg, #fff, #e0f2fe, #bae6fd); border-radius: 15px 160px 80px 140px; transform: rotate(15deg); }
    .r-f2 { width: 140px; height: 60px; top: 40px; left: 20px; background: linear-gradient(60deg, #fff, #fbcfe8, #f472b6); border-radius: 15px 140px 70px 120px; transform: rotate(5deg); z-index: 2; }
    .r-f3 { width: 120px; height: 55px; top: 80px; left: 40px; background: linear-gradient(60deg, #fff, #e0f2fe, #7dd3fc); border-radius: 15px 120px 60px 90px; transform: rotate(-10deg); }

    /* 💖 팩트 본체 서클 하우징 */
    .fact-pink-heart-shield {
        position: absolute;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle at 35% 35%, #ffffff 0%, #fbcfe8 45%, #ec4899 85%, #be185d 100%);
        border-radius: 50%;
        border: 10px solid #ffffff;
        box-shadow: 0 25px 60px rgba(236, 72, 153, 0.45);
        z-index: 3;
    }

    /* 🔒 [안착형] 지도가 팩트 내부에 완벽히 봉인되는 코어 바인더 */
    .map-inside-binder {
        position: relative;
        width: 290px;
        height: 290px;
        border-radius: 50% !important;
        overflow: hidden !important;
        z-index: 5;
        border: 6px solid #fef08a; /* 노란색 보석 링 */
        box-shadow: inset 0 0 40px rgba(0, 242, 254, 0.7);
        background: #090521;
        display: flex;
        justify-content: center;
        align-items: center;
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
# 📊 [데이터 연산부]
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_quake_data():
    np.random.seed(42)
    num_samples = 2000
    df = pd.DataFrame({
        '위도': np.random.uniform(-70, 70, num_samples),
        '경도': np.random.uniform(-180, 180, num_samples),
        '규모': np.random.uniform(1.5, 8.0, num_samples),
        '진원깊이': np.random.uniform(5, 650, num_samples),
        '영향도': np.random.uniform(10, 100, num_samples),
    })
    return df

df = load_quake_data()
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
# 메인 콘솔 UI
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

st.markdown("### 🎯 타겟 좌표 설정")
cx, cy = st.columns(2)
with cx:
    lat = st.number_input("💖 타겟 위도", -90.0, 90.0, 36.5, step=0.1)
with cy:
    lon = st.number_input("🌌 타겟 경도", -180.0, 180.0, 127.5, step=0.1)

if st.button("🪐 슈팅스타 팩트 분석 모드 가동", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    near_idx = np.argsort(dist)[:20]
    nearest_km = float(dist[near_idx[0]])
    
    cw = {}
    for idx in near_idx:
        c_val = int(df.iloc[idx]["cluster"])
        cw[c_val] = cw.get(c_val, 0.0) + 1.0 / (dist[idx] + 10.0)
    dom_cluster = int(max(cw, key=cw.get))
    final_grade = grade_map.get(dom_cluster, "저위험군")

    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.write("#### 🔮 3D 홀로그램 분석")
        
        # 팩트 본체 구성 (지도가 내부 Binder 안에 정확히 배치됨)
        st.markdown(
            f"""
            <div class="shooting-star-factory-stage">
                <div class="star-gold-pedestal-base"></div>
                <div class="wing-group-left">
                    <div class="feather l-f1"></div><div class="feather l-f2"></div><div class="feather l-f3"></div>
                </div>
                <div class="wing-group-right">
                    <div class="feather r-f1"></div><div class="feather r-f2"></div><div class="feather r-f3"></div>
                </div>
                <div class="fact-pink-heart-shield"></div>
                <div class="map-inside-binder" id="hologram-target">
            """, unsafe_allow_html=True
        )
        
        # 3D 홀로그램 지구본 데이터 전달
        show_df = df.sample(min(500, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        
        points_js = []
        for _, r in show_df.iterrows():
            g = grade_map.get(int(r["cluster"]), "저위험군")
            points_js.append(f"{{lat: {r['위도']}, lon: {r['경도']}, color: '{HEX_MAP[g]}', size: {r['규모']}}}")
        
        # [완벽 조치] 투명 배경 + 선명한 레이더 격자가 특징인 HTML5 3D 엔진
        three_js_code = f"""
        <!DOCTYPE html>
        <html>
        <head><style>body {{ margin: 0; overflow: hidden; background: transparent; }} canvas {{ background: transparent; cursor: grab; }}</style></head>
        <body><canvas id="c"></canvas>
        <script>
            const canvas = document.getElementById('c'); const ctx = canvas.getContext('2d');
            let w = canvas.width = 290; let h = canvas.height = 290;
            let rotX = 0.4, rotY = {np.radians(lon)};
            const pts = [{", ".join(points_js)}];
            const target = {{ lat: {lat}, lon: {lon}, color: '#fff', size: 9 }};

            function project(lat, lon) {{
                let rLat = (lat * Math.PI) / 180, rLon = (lon * Math.PI) / 180 + rotY;
                let r = 110, x = r * Math.cos(rLat) * Math.sin(rLon), y = r * Math.sin(rLat), z = r * Math.cos(rLat) * Math.cos(rLon);
                let cX = Math.cos(rotX), sX = Math.sin(rotX);
                let ry = y * cX - z * sX, rz = y * sX + z * cX;
                return {{ x: x + w/2, y: -ry + h/2, depth: rz }};
            }}

            function draw() {{
                ctx.clearRect(0, 0, w, h);
                ctx.strokeStyle = 'rgba(0, 242, 254, 0.7)'; ctx.lineWidth = 1.3;
                for (let l = -60; l <= 60; l += 20) {{
                    ctx.beginPath();
                    for (let lng = -180; lng <= 180; lng += 10) {{
                        let p = project(l, lng); if (lng === -180) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
                    }}
                    ctx.stroke();
                }}
                let all = [...pts, target];
                all.forEach(p => p._z = project(p.lat, p.lon));
                all.sort((a, b) => b._z.depth - a._z.depth);
                all.forEach(p => {{
                    if (p._z.depth > -40) {{
                        ctx.beginPath(); ctx.arc(p._z.x, p._z.y, Math.max(2.5, p.size * 1.1), 0, 7);
                        ctx.fillStyle = p.color; ctx.globalAlpha = Math.max(0.4, (p._z.depth + 110) / 220);
                        if (p === target) {{ ctx.shadowBlur = 15; ctx.shadowColor = '#ec4899'; }}
                        ctx.fill(); ctx.shadowBlur = 0;
                    }}
                }});
                requestAnimationFrame(draw);
            }}
            window.addEventListener('mousedown', e => {{ isDragging = true; prevPos = {{ x: e.clientX, y: e.clientY }}; }});
            window.addEventListener('mousemove', e => {{
                if (!isDragging) return;
                rotY += (e.clientX - prevPos.x) * 0.007; rotX += (e.clientY - prevPos.y) * 0.007;
                prevPos = {{ x: e.clientX, y: e.clientY }};
            }});
            window.addEventListener('mouseup', () => isDragging = false);
            draw();
        </script></body></html>
        """
        components.html(three_js_code, height=290, width=290, scrolling=False)
        st.markdown("</div></div>", unsafe_allow_html=True)

    with c2:
        st.write("#### 📊 글로벌 지진 격자 매트릭스")
        
        # [해결] 확대맵 ❌ → 한글 깨짐 없는 글로벌 맵 ⭕, 제목 제거
        chart_points = []
        for _, r in df.sample(min(450, len(df)), random_state=42).iterrows():
            g = grade_map.get(int(r['cluster']), "저위험군")
            chart_points.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{HEX_MAP[g]}'}}")
        
        canvas_2d_html = f"""
        <!DOCTYPE html>
        <html>
        <head><style>body {{ font-family: 'Pretendard', sans-serif; margin: 0; background: rgba(255,255,255,0.8); border-radius: 20px; padding: 20px; }}</style></head>
        <body><canvas id="c2" width="460" height="380"></canvas>
        <script>
            const cvs = document.getElementById('c2'); const ctx = cvs.getContext('2d');
            const pts = [{", ".join(chart_points)}];
            ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 0.5;
            for(let i=40; i<=460; i+=60) {{ ctx.beginPath(); ctx.moveTo(i, 10); ctx.lineTo(i, 330); ctx.stroke(); }}
            for(let j=10; j<=330; j+=40) {{ ctx.beginPath(); ctx.moveTo(40, j); ctx.lineTo(460, j); ctx.stroke(); }}
            ctx.fillStyle = '#4c4475'; ctx.font = 'bold 13px Pretendard';
            ctx.fillText("전세계 경도 범위 (-180° ~ 180°)", 160, 365);
            ctx.save(); ctx.translate(15, 200); ctx.rotate(-Math.PI/2); ctx.fillText("전세계 위도 범위 (-90° ~ 90°)", -80, 0); ctx.restore();
            pts.forEach(p => {{
                let cx = 40 + ((p.x + 180) / 360) * 400, cy = 330 - ((p.y + 90) / 180) * 320;
                ctx.beginPath(); ctx.arc(cx, cy, 3.5, 0, 7); ctx.fillStyle = p.color; ctx.fill();
            }});
            let tx = 40 + (({lon} + 180) / 360) * 400, ty = 330 - (({lat} + 90) / 180) * 320;
            ctx.beginPath(); ctx.arc(tx, ty, 9, 0, 7); ctx.fillStyle = '#fff'; ctx.strokeStyle = '#db2777'; ctx.lineWidth = 3; ctx.fill(); ctx.stroke();
        </script></body></html>
        """
        components.html(canvas_2d_html, height=410, scrolling=False)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3>🛸 <b>초롱핑의 오로라 정밀 홀로그램 피드</b></h3>
            <p> [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ] </p>
            <p style="color:#475569; font-size:14px; line-height:1.7;">
                지정한 위도 {lat:.4f}°, 경도 {lon:.4f}° 내부의 위험 격자 스캔 결과 <b>{final_grade}</b> 등급이 감지되었습니다츄.
                인근 지진 중심 코어 영역과의 최단 이격 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄.
            </p>
        </div>
        """, unsafe_allow_html=True
    )
