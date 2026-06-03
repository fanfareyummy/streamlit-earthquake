import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import streamlit.components.v1 as components  # 👈 누락되었던 컴포넌트 복구!

APP_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ["영향도", "규모", "진원깊이"]

# ═════════════════════════════════════════════════════════════
# 🎨 [슈팅스타팩트: 최종 안정화 에디션] 3색 지진 완벽 동기화
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

    /* 🔮 완벽한 1:1 좌우대칭 슈팅스타 팩트 하우징 메인 외곽 프레임 구조 */
    .pact-total-frame {
        position: relative;
        width: 100%;
        max-width: 600px;
        margin: 0 auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .compact-top-crown {
        position: absolute;
        top: -20px; left: 50%;
        width: 80px; height: 40px;
        background: linear-gradient(180deg, #ff9ebb 0%, #ec4899 100%);
        transform: translateX(-50%);
        border-radius: 40px 40px 10px 10px;
        border: 4px solid #ffffff;
        box-shadow: 0 6px 14px rgba(236, 72, 153, 0.45);
        z-index: 10;
    }
    .compact-top-crown::after {
        content: '⭐';
        position: absolute;
        top: -2px; left: 50%;
        transform: translateX(-50%);
        font-size: 16px;
    }
    .star-gold-pedestal-base {
        position: absolute;
        bottom: -25px; left: 50%;
        width: 420px; height: 120px;
        background: linear-gradient(135deg, #ffe57f 0%, #ffca28 50%, #ffb300 100%);
        transform: translateX(-50%);
        clip-path: polygon(15% 0%, 85% 0%, 100% 100%, 0% 100%);
        border-radius: 0 0 35px 35px;
        border-bottom: 6px solid #ffffff;
        box-shadow: 0 12px 30px rgba(255, 179, 0, 0.45);
        z-index: 1; 
    }
    .fairy-wing-left {
        position: absolute; left: -50px; top: 120px;
        width: 130px; height: 210px;
        background: linear-gradient(to left, rgba(255,255,255,0.98), rgba(226,245,255,0.9), rgba(186,230,253,0.75));
        border: 4px solid #ffffff;
        border-radius: 160px 40px 100px 140px;
        box-shadow: -8px 12px 25px rgba(186, 230, 253, 0.45);
        z-index: 2;
    }
    .fairy-wing-right {
        position: absolute; right: -50px; top: 120px;
        width: 130px; height: 210px;
        background: linear-gradient(to right, rgba(255,255,255,0.98), rgba(255,231,243,0.9), rgba(244,114,182,0.75));
        border: 4px solid #ffffff;
        border-radius: 40px 160px 140px 100px;
        box-shadow: 8px 12px 25px rgba(244, 114, 182, 0.45);
        z-index: 2;
    }
    .fact-pink-heart-shield {
        position: relative;
        width: 500px; height: 500px;
        background: radial-gradient(circle at 30% 30%, #ffffff 0%, #ffb6c1 25%, #f472b6 55%, #db2777 85%, #9d174d 100%);
        border-radius: 50%;
        border: 10px solid #ffffff;
        box-shadow: 
            inset 0 -15px 30px rgba(0,0,0,0.25),
            inset 0 15px 30px rgba(255,255,255,0.7),
            0 25px 55px rgba(219, 39, 119, 0.55);
        display: flex;
        justify-content: center;
        align-items: center;
        z-index: 5;
    }
    .fact-inner-gold-ring {
        position: relative;
        width: 440px; height: 440px;
        border-radius: 50%;
        background: transparent;
        border: 8px solid #facc15;
        box-shadow: 0 0 22px #facc15, inset 0 0 15px rgba(234, 179, 8, 0.6);
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .map-inside-binder {
        position: relative;
        width: 408px; height: 408px;
        border-radius: 50%;
        overflow: hidden;
        border: 5px solid #ffffff;
        box-shadow: inset 0 0 40px rgba(0, 242, 254, 0.85);
        background: #bcf0f7;
    }

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
    .tag-high { background: #ff4d4d; box-shadow: 0 4px 10px rgba(255,77,77,0.4); }
    .tag-mid { background: #2ed573; box-shadow: 0 4px 10px rgba(46,213,115,0.4); }
    .tag-low { background: #1e3799; box-shadow: 0 4px 10px rgba(30,55,153,0.4); }

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
# 📊 [사진 기반 글로벌 지진 고정밀 데이터셋 엔진]
# ═════════════════════════════════════════════════════════════
@st.cache_data
def load_perfect_image_earthquakes():
    np.random.seed(99)
    num_samples = 2200
    
    image_accurate_zones = [
        [62.0, -150.0], [55.0, -162.0], [53.0, -170.0], 
        [36.0, -119.5], [34.0, -118.0], [40.0, -124.0], 
        [19.5, -155.5],                               
        [9.0, -83.0], [15.0, -90.0], [10.0, -70.0],    
        [-15.0, -75.0], [-33.0, -71.5], [-45.0, -73.0], 
        [36.5, 138.0], [35.0, 142.0], [43.0, 145.0],   
        [38.0, 23.5], [36.0, 26.0], [39.0, 35.0],      
        [-8.0, 115.0], [-5.0, 102.0], [5.0, 125.0],    
        [-41.0, 174.0], [-20.0, 168.0], [-15.0, 178.0], 
        [-30.0, 26.0], [-33.9, 18.4], [64.0, -18.0]     
    ]
    
    lats, lons = [], []
    for _ in range(num_samples):
        core = image_accurate_zones[np.random.randint(len(image_accurate_zones))]
        lats.append(core[0] + np.random.normal(0, 2.5))
        lons.append(core[1] + np.random.normal(0, 2.9))
        
    df = pd.DataFrame({
        '위도': np.clip(lats, -85, 85),
        '경도': np.clip(lons, -180, 180),
        '규모': np.random.uniform(2.0, 8.8, num_samples),
        '진원깊이': np.random.uniform(5, 700, num_samples),
        '영향도': np.random.uniform(5, 100, num_samples),
    })
    return df

df = load_perfect_image_earthquakes()
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
df["danger_grade"] = df["cluster"].map(grade_map)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat2 - lat1)/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin((lon2 - lon1)/2)**2))

# ═════════════════════════════════════════════════════════════
# 메인 헤더 콘솔 UI
# ═════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="photo-top-header">
        <h1>✨ CATCH! TEENIEPING: SHOOTING STAR AURA FACT</h1>
        <div style="color:#7a6fbe; font-size:14px; margin-top:5px; font-weight:700;">
            (오로라 스페이스 팩트 프로덕션 3색 월드 레이더 지진 분석 콘솔)
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
        st.write("#### 🔮 슈팅스타 팩트 정밀 3D 글로벌 맵 (완벽 동기화)")
        
        COLOR_DISCRETE_MAP = {"고위험군": "#ff4d4d", "중위험군": "#2ed573", "저위험군": "#1e3799"}
        
        fig = go.Figure()
        
        for g_type, color in COLOR_DISCRETE_MAP.items():
            sub_df = df[df["danger_grade"] == g_type]
            fig.add_trace(go.Scattergeo(
                lon=sub_df["경도"],
                lat=sub_df["위도"],
                mode="markers",
                name=g_type,
                marker=dict(
                    size=4.5,
                    color=color,
                    opacity=0.85,
                    line=dict(width=0.3, color="white")
                )
            ))
            
        fig.add_trace(go.Scattergeo(
            lon=[lon],
            lat=[lat],
            mode="markers",
            name="조준 타겟",
            marker=dict(
                size=14,
                color="#ffffff",
                symbol="diamond",
                line=dict(width=2.5, color="#db2777")
            )
        ))

        fig.update_layout(
            geo=dict(
                projection_type="orthographic", 
                showland=True,
                landcolor="#f5f6fa",           
                showocean=True,
                oceancolor="#bcf0f7",          
                showcountries=False,
                showcoastlines=True,
                coastlinecolor="rgba(255,255,255,0.4)",
                projection_rotation=dict(lon=lon, lat=lat, roll=0), 
                lataxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)"),
                lonaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.2)")
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            width=408,
            height=408,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False
        )
        
        st.markdown(
            """
            <div class="pact-total-frame">
                <div class="compact-top-crown"></div>
                <div class="star-gold-pedestal-base"></div>
                <div class="fairy-wing-left"></div>
                <div class="fairy-wing-right"></div>
                <div class="fact-pink-heart-shield">
                    <div class="fact-inner-gold-ring">
                        <div class="map-inside-binder">
            """,
            unsafe_allow_html=True
        )
        
        st.plotly_chart(fig, use_container_width=False, config={'displayModeBar': False})
        
        st.markdown(
            """
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_right_graph:
        st.write("#### 📊 타겟 반경 지진 분포 격자 레이더")
        
        chart_points = []
        HEX_MAP = {"고위험군": "#ff4d4d", "중위험군": "#2ed573", "저위험군": "#1e3799"}
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
        components.html(canvas_chart_html, height=410, scrolling=False)

    tag_cls = "tag-high" if final_grade == "고위험군" else ("tag-mid" if final_grade == "중위험군" else "tag-low")
    
    st.markdown(
        f"""
        <div class="photo-bottom-card">
            <h3 style="margin-top:0; color:#1e1b4b;">🛸 <b>초롱핑의 오로라 리얼 이미지 고정밀 매핑 피드</b></h3>
            <p style="font-size:16px; font-weight:700; margin-bottom:12px;">
                [ ⚡ 초롱핑 감지: <span class="danger-tag {tag_cls}">{final_grade}</span> ]
            </p>
            <p style="color:#475569; line-height:1.7; font-size:14px; margin:0;">
                업로드해 주신 이미지 레이아웃 사양서의 3색(🔴 빨강, 🟢 초록, 🔵 파랑) 군집 구조와 완벽히 일치하는 리얼 맵 연산이 완료되었습니다츄.
                지정한 정밀 타겟 위도 {lat:.4f}°, 경도 {lon:.4f}° 기준, 최인접 활성 단층 지진대 포인트와의 최단 도달 거리는 <b>{nearest_km:,.1f} km</b>로 스캔되었습니다츄!
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
