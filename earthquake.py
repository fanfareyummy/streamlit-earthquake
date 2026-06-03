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
# 🎨 [슈팅스타팩트: 퍼펙트 시메트리] 비대칭 완전 해결 & 럭셔리 디자인
# ═════════════════════════════════════════════════════════════
st.set_page_config(page_title="슈팅스타팩트 지진 분석 시스템", page_icon="🔮", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Pretendard:wght@700;900&display=swap');

    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Pretendard', sans-serif !important;
        background: linear-gradient(135deg, #fdfcfb 0%, #e2d1f9 100%) !important;
        color: #3b3a57 !important;
        overflow-x: hidden;
    }
    
    /* ✨ 마법 별빛 가루 효과 */
    .stAppViewContainer::before {
        content: '';
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        pointer-events: none;
        z-index: 999;
        background-image: 
            radial-gradient(circle at 10% 10%, #fff 1px, transparent 2px),
            radial-gradient(circle at 90% 20%, #fff 1.5px, transparent 2.5px),
            radial-gradient(circle at 50% 50%, #ffd1ff 2px, transparent 4px);
        background-size: 300px 300px;
        opacity: 0.6;
    }

    .photo-top-header {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 30px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(186, 104, 200, 0.15);
        margin-bottom: 25px;
        backdrop-filter: blur(10px);
        border: 1px solid #ffffff;
    }
    .photo-top-header h1 { margin: 0; font-size: 26px; font-weight: 900; color: #5e4b8d; }

    .photo-bottom-card {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 24px;
        padding: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
        margin-top: 20px;
        border: 1px solid #fff;
    }
    .danger-tag { font-weight: 900; padding: 5px 12px; border-radius: 10px; color: white; }
    .tag-high { background: #ff7675; }
    .tag-mid { background: #facc15; color: #333; }
    .tag-low { background: #4ade80; color: #111; }
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
# 메인 콘솔 UI
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#8e84ad; font-size:14px; font-weight:700;">오로라 팩트 지진 위험군 정밀 감지 시스템</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### 🎯 레이더 타겟 설정")
c_in1, c_in2 = st.columns(2)
with c_in1:
    lat = st.number_input("💖 타겟 위도", -90.0, 90.0, 36.5, step=0.1)
with c_in2:
    lon = st.number_input("🌌 타겟 경도", -180.0, 180.0, 127.5, step=0.1)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🪐 슈팅스타 팩트 분석 가동", use_container_width=True):
    dist = haversine(lat, lon, df["위도"].values, df["경도"].values)
    near_idx = np.argsort(dist)[:20]
    nearest_km = float(dist[near_idx[0]])
    cw = {}
    for idx in near_idx:
        c_val = int(df.iloc[idx]["cluster"])
        cw[c_val] = cw.get(c_val, 0.0) + 1.0 / (dist[idx] + 10.0)
    dom_cluster = int(max(cw, key=cw.get))
    final_grade = grade_map.get(dom_cluster, "저위험군")

    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.write("#### 🔮 팩트 홀로그램 (100% 대칭 교정)")
        
        show_df = df.sample(min(400, len(df)), random_state=42)
        HEX_MAP = {"고위험군": "#ff7675", "중위험군": "#facc15", "저위험군": "#4ade80"}
        points_js = [f"{{lat: {r['위도']}, lon: {r['경도']}, color: '{HEX_MAP[grade_map[int(r['cluster'])]]}', size: {r['규모']}}}" for _, r in show_df.iterrows()]

        # 🪐 [대혁신] 모든 그래픽 요소를 하나의 중심점에 정렬한 통합 HTML/CSS/JS 샌드박스
        master_pact_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ margin: 0; background: transparent; display: flex; justify-content: center; align-items: center; width: 500px; height: 500px; overflow: hidden; }}
                .pact-stage {{ position: relative; width: 450px; height: 450px; display: flex; justify-content: center; align-items: center; }}
                
                /* ⭐ 하단 황금 받침대 (정중앙 정렬) */
                .gold-base {{
                    position: absolute; bottom: 0; width: 280px; height: 120px;
                    background: linear-gradient(135deg, #fff9c4, #fbc02d, #f9a825);
                    clip-path: polygon(20% 0%, 80% 0%, 100% 100%, 0% 100%);
                    border-radius: 0 0 30px 30px; border-bottom: 6px solid #fff; z-index: 1;
                }}

                /* 👼 고퀄리티 좌우 대칭 요정 날개 */
                .wing {{ position: absolute; width: 130px; height: 180px; background: linear-gradient(to bottom, #fff, #e1f5fe, #fce4ec); border: 3px solid #fff; z-index: 2; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }}
                .wing-l {{ left: -30px; border-radius: 150px 20px 100px 130px; }}
                .wing-r {{ right: -30px; border-radius: 20px 150px 130px 100px; }}

                /* 💖 메인 핑크 보석 바디 (중앙) */
                .main-body {{
                    position: relative; width: 360px; height: 360px;
                    background: radial-gradient(circle at 30% 30%, #ffffff 0%, #fce4ec 20%, #f06292 60%, #ad1457 100%);
                    border-radius: 50%; border: 10px solid #fff;
                    box-shadow: 0 15px 40px rgba(173, 20, 87, 0.4), inset 0 -10px 20px rgba(0,0,0,0.2);
                    display: flex; justify-content: center; align-items: center; z-index: 3;
                }}

                /* ✨ 골드 이너 링 (중앙) */
                .inner-gold {{
                    width: 290px; height: 290px; border-radius: 50%;
                    border: 6px solid #ffd700; box-shadow: 0 0 15px #ffd700;
                    display: flex; justify-content: center; align-items: center;
                }}

                /* 🔒 지구본 매립 영역 (중앙) */
                .display-dome {{
                    width: 270px; height: 270px; border-radius: 50%; overflow: hidden;
                    background: #050112; border: 4px solid #fff;
                    box-shadow: inset 0 0 30px rgba(0, 242, 254, 0.8); position: relative;
                }}

                canvas {{ cursor: grab; }}
            </style>
        </head>
        <body>
            <div class="pact-stage">
                <div class="gold-base"></div>
                <div class="wing wing-l"></div>
                <div class="wing wing-r"></div>
                <div class="main-body">
                    <div class="inner-gold">
                        <div class="display-dome">
                            <canvas id="globe" width="270" height="270"></canvas>
                        </div>
                    </div>
                </div>
            </div>
            <script>
                const canvas = document.getElementById('globe');
                const ctx = canvas.getContext('2d');
                let rotX = 0.4, rotY = {np.radians(lon)}, isDrag = false, lastPos = {{x:0, y:0}};
                const pts = [{",".join(points_js)}];
                const target = {{lat: {lat}, lon: {lon}, color: '#fff', size: 9}};

                function project(la, lo) {{
                    let rLa = (la * Math.PI)/180, rLo = (lo * Math.PI)/180 + rotY;
                    let r = 100, x = r * Math.cos(rLa) * Math.sin(rLo);
                    let y = r * Math.sin(rLa), z = r * Math.cos(rLa) * Math.cos(rLo);
                    let cX = Math.cos(rotX), sX = Math.sin(rotX);
                    let ry = y * cX - z * sX, rz = y * sX + z * cX;
                    return {{ x: x + 135, y: -ry + 135, d: rz }};
                }}

                function draw() {{
                    ctx.clearRect(0,0,270,270);
                    ctx.strokeStyle = 'rgba(0, 242, 254, 0.6)'; ctx.lineWidth = 1.2;
                    for(let i=-60; i<=60; i+=20) {{
                        ctx.beginPath();
                        for(let j=-180; j<=180; j+=10) {{
                            let p = project(i, j); if(j===-180) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
                        }}
                        ctx.stroke();
                    }}
                    let all = [...pts, target];
                    all.forEach(p => p._z = project(p.lat, p.lon));
                    all.sort((a,b) => b._z.d - a._z.d);
                    all.forEach(p => {{
                        if(p._z.d > -40) {{
                            ctx.globalAlpha = Math.max(0.4, (p._z.d + 100)/200);
                            ctx.beginPath(); ctx.arc(p._z.x, p._z.y, Math.max(2, p.size*0.9), 0, 7);
                            ctx.fillStyle = p.color; ctx.fill();
                            if(p===target) {{ ctx.shadowBlur=15; ctx.shadowColor="#db2777"; ctx.stroke(); }}
                        }}
                    }});
                    ctx.shadowBlur=0; requestAnimationFrame(draw);
                }}
                canvas.onmousedown = e => {{ isDrag=true; lastPos={{x:e.clientX, y:e.clientY}}; }};
                window.onmousemove = e => {{ if(!isDrag) return; rotY+=(e.clientX-lastPos.x)*0.01; rotX+=(e.clientY-lastPos.y)*0.01; lastPos={{x:e.clientX, y:e.clientY}}; }};
                window.onmouseup = () => isDrag=false;
                draw();
            </script>
        </body>
        </html>
        """
        components.html(master_pact_html, width=500, height=500, scrolling=False)

    with col_right:
        st.write("#### 📊 타겟 정밀 확대 레이더")
        chart_pts = []
        sub = df[(df["경도"].between(lon-30, lon+30)) & (df["위도"].between(lat-30, lat+30))].sample(min(200, len(df)), replace=True)
        for _, r in sub.iterrows():
            chart_pts.append(f"{{x: {r['경도']}, y: {r['위도']}, color: '{HEX_MAP[grade_map[int(r['cluster'])]]}'}}")
        
        canvas_2d = f"""
        <html><body style="margin:0; font-family:sans-serif; background:rgba(255,255,255,0.7); border-radius:20px; padding:15px;">
            <canvas id="c" width="460" height="380"></canvas>
            <script>
                const ctx = document.getElementById('c').getContext('2d');
                const pts = [{",".join(chart_pts)}];
                ctx.strokeStyle="#cbd5e1"; ctx.lineWidth=0.5;
                for(let i=40; i<460; i+=60) {{ ctx.beginPath(); ctx.moveTo(i,0); ctx.lineTo(i,340); ctx.stroke(); }}
                ctx.fillStyle="#5e4b8d"; ctx.font="bold 12px sans-serif";
                ctx.fillText("경도 (Longitude)", 200, 370);
                pts.forEach(p => {{
                    let x = 40 + ((p.x - ({lon-30}))/60)*400, y = 340 - ((p.y - ({lat-30}))/60)*300;
                    ctx.beginPath(); ctx.arc(x,y,4,0,7); ctx.fillStyle=p.color; ctx.fill();
                }});
                ctx.beginPath(); ctx.arc(240, 190, 10, 0, 7); ctx.fillStyle="#fff"; ctx.strokeStyle="#db2777"; ctx.lineWidth=3; ctx.fill(); ctx.stroke();
            </script>
        </body></html>
        """
        components.html(canvas_2d, height=400, scrolling=False)

    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3>🛸 <b>초롱핑의 오로라 정밀 피드</b></h3>
            <p style="font-size:18px;">[ ⚡ 감지 등급: <span class="danger-tag {"tag-high" if final_grade=="고위험군" else "tag-mid" if final_grade=="중위험군" else "tag-low"}">{final_grade}</span> ]</p>
            <p>지정한 좌표에서 가장 가까운 지진 활동점까지의 거리는 약 <b>{nearest_km:,.1f} km</b>입니다츄. 데이터 스캔이 성공적으로 완료되었습니다!</p>
        </div>
        """, unsafe_allow_html=True
    )
