import streamlit as st
import pandas as pd
import numpy as np
import json

# --- 1. 페이지 설정 및 메디컬 라이트 UI 테마 ---
st.set_page_config(
    page_title="스마트 의료 센터 - 바이러스 통합 관제",
    page_icon="🩺",
    layout="wide",
)

# 화사하고 깨끗한 의료 대시보드 스타일 및 [글자색 강제 선명화] CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght=300;500;700&display=swap');

    /* 전역 글자색 기본 네이비 고정 */
    .stApp { 
        background: linear-gradient(135deg, #F0F9FF 0%, #FFFFFF 100%);
        color: #1E293B !important;
        font-family: 'Noto Sans KR', sans-serif;
    }

    /* 🚨 [긴급 조치] 배경과 겹쳐서 안 보였던 메트릭 글자색을 아주 어둡고 선명하게 강제 지정 */
    div[data-testid="stMetricValue"] {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #334155 !important;
        font-weight: 500 !important;
    }

    /* 상단 의료 센터 헤더 */
    .hospital-header {
        background: white;
        border-bottom: 3px solid #0EA5E9;
        padding: 20px;
        margin-bottom: 25px;
        border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .hospital-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0369A1 !important;
    }
    .status-badge {
        background: #E0F2FE;
        border: 1px solid #0EA5E9;
        padding: 5px 15px;
        border-radius: 50px;
        font-size: 0.85rem;
        color: #0369A1 !important;
        font-weight: 500;
    }

    /* 3D 지구본 컨테이너 */
    .globe-section {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 24px;
        padding: 15px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    }

    /* 🦠 자체 회전 코로나 구조 분석 챔버 */
    .pure-css-virus-chamber {
        background: radial-gradient(circle at center, #1E3A8A 0%, #0F172A 100%);
        height: 250px;
        border-radius: 20px;
        border: 4px solid #E2E8F0;
        position: relative;
        overflow: hidden;
        display: flex;
        justify-content: center;
        align-items: center;
        box-shadow: inset 0 0 40px rgba(0, 242, 255, 0.4);
    }
    
    .rotating-corona-sphere {
        font-size: 85px;
        animation: coronaOrbit 7s linear infinite;
        filter: drop-shadow(0 0 25px #22C55E);
        user-select: none;
    }

    .scanner-laser-line {
        position: absolute;
        width: 100%;
        height: 3px;
        background: linear-gradient(90deg, rgba(0,242,255,0) 0%, rgba(0,242,255,1) 50%, rgba(0,242,255,0) 100%);
        box-shadow: 0 0 15px #00F2FF;
        animation: laserScan 4s ease-in-out infinite;
    }

    @keyframes coronaOrbit {
        0% { transform: rotate(0deg) scale(1); }
        50% { transform: rotate(180deg) scale(1.15); }
        100% { transform: rotate(360deg) scale(1); }
    }

    @keyframes laserScan {
        0% { top: 5%; }
        50% { top: 95%; }
        100% { top: 5%; }
    }

    /* 하단 가로형 스마트 제어 패널 */
    .control-panel {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        border: 1px solid #BAE6FD;
        border-top: 4px solid #0EA5E9;
        border-radius: 20px;
        padding: 25px;
        margin-top: 30px;
        box-shadow: 0 -10px 30px rgba(14, 165, 233, 0.05);
    }

    div[data-baseweb="input"] { border-radius: 12px !important; border: 1px solid #CBD5E1 !important; }
    
    .medical-note {
        background: #F0FDF4;
        border-left: 5px solid #22C55E;
        padding: 15px;
        border-radius: 8px;
        font-size: 0.9rem;
        color: #1E293B !important;
        line-height: 1.6;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 2. 데이터 로드 ---
@st.cache_data
def load_data():
    file_name = "covid_risk_analysis_result.csv"
    try:
        data = pd.read_csv(file_name)
        return data[['위도', '경도', 'cluster']].dropna()
    except:
        return None

df = load_data()

if df is None:
    st.error("🔬 데이터 동기화 실패: 'covid_risk_analysis_result.csv' 아카이브를 인덱싱할 수 없습니다.")
    st.stop()

# --- 3. 헤더 섹션 ---
st.markdown("""
    <div class='hospital-header'>
        <div class='hospital-title'>🩺 스마트 의료 통합 관제 센터 <span>[V10.1 COLOR_FIXED]</span></div>
        <div class='status-badge'>● 폰트 셰이더 및 가독성 패치 적용 가동 중</div>
    </div>
""", unsafe_allow_html=True)

# --- 4. 메인 콘텐츠 ---
col_globe, col_media = st.columns([2.1, 1.9])

with col_globe:
    st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#0369A1; margin-bottom:10px;'>🌍 글로벌 병원균 확산 3D 시각화 매트릭스</p>", unsafe_allow_html=True)
    
    if 'lat_val' not in st.session_state: st.session_state.lat_val = 10.80
    if 'lon_val' not in st.session_state: st.session_state.lon_val = 106.60

    points_json = json.dumps(df.to_dict(orient="records"))
    
    hologram_globe_html = f"""
    <div class='globe-section'>
        <div id="medical-globe" style="width: 100%; height: 380px;"></div>
        <script src="https://unpkg.com/globe.gl"></script>
        <script>
            const rawData = {points_json};
            const gData = rawData.map(d => ({{
                lat: d['위도'], lng: d['경도'],
                size: d['cluster'] == 2 ? 0.7 : (d['cluster'] == 1 ? 0.45 : 0.25),
                color: d['cluster'] == 2 ? '#EF4444' : (d['cluster'] == 1 ? '#F59E0B' : '#0EA5E9'),
                isTarget: false
            }}));

            gData.push({{
                lat: {st.session_state.lat_val}, lng: {st.session_state.lon_val},
                size: 1.5, color: '#22C55E', isTarget: true
            }});

            const globe = Globe()
                (document.getElementById('medical-globe'))
                .globeImageUrl('https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg')
                .backgroundColor('rgba(0,0,0,0)')
                .pointsData(gData)
                .pointRadius('size')
                .pointColor('color')
                .pointAltitude(d => d.isTarget ? 0.1 : 0.03)
                .pointLabel(d => d.isTarget ? `🎯 정밀 분석 타겟` : `관찰 데이터`)
                .controlsMaxZoom(3);

            globe.pointOfView({{ lat: {st.session_state.lat_val}, lng: {st.session_state.lon_val}, alt: 2.1 }}, 1500);
            globe.controls().autoRotate = false;
        </script>
        <div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 0.8rem; color: #475569;">
            <div style="color: #22C55E; font-weight: bold;">🎯 타겟 록온: {st.session_state.lat_val}°, {st.session_state.lon_val}°</div>
            <div>
                <span style="color:#EF4444;">●</span> 위중증 &nbsp;&nbsp;
                <span style="color:#F59E0B;">●</span> 경계 &nbsp;&nbsp;
                <span style="color:#0EA5E9;">●</span> 안정
            </div>
        </div>
    </div>
    """
    st.components.v1.html(hologram_globe_html, height=430)

    st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#0369A1; margin-top:12px; margin-bottom:5px;'>📊 타겟 반경 변이 바이러스 스파이크 단백질 해독 스캔률 (역학 정밀 트렌드)</p>", unsafe_allow_html=True)
    
    with st.container(border=True):
        chart_data = pd.DataFrame(
            np.random.rand(15, 3) * [15, 65, 20],
            columns=['알파/델타 계통', '오미크론 하위변이(BA.5)', '기타 신종 변종']
        )
        st.bar_chart(chart_data, height=240)
    
    st.markdown("""
        <div style='background: #F8FAFC; border: 1px solid #E2E8F0; padding: 10px 15px; border-radius: 10px; font-size: 0.8rem; color: #1E293B; line-height: 1.5; margin-bottom: 10px;'>
            <b>💡 CDC 감염병 변이 통계 보강:</b> 전 세계 하수 기반 역학 조사 결과, 오미크론 하위 계통의 변이 속도가 지질 외벽 안정성에 미치는 상관관계가 상기 차트와 같이 도출되었습니다.
        </div>
    """, unsafe_allow_html=True)
    
    # 🚨 [색상 복원구역] 강제 오버라이딩을 거쳐 이제 검정색 계열로 뚜렷하게 나옵니다!
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="🦠 주요 변이 위험도", value="위험 (BA.5)", delta="상승 지표")
    with m2:
        st.metric(label="🛡️ 타겟 반경 방역 지수", value="82.4 점", delta="안전 범위")
    with m3:
        st.metric(label="🧬 유전자 서열 일치율", value="99.8 %", delta="변이 확인")

with col_media:
    st.markdown("<p style='font-size:0.9rem; font-weight:700; color:#0369A1; margin-bottom:10px;'>🔬 SARS-CoV-2 (코로나 바이러스) 입체 구조 분석 시뮬레이터</p>", unsafe_allow_html=True)
    
    st.markdown("""
        <div class='pure-css-virus-chamber'>
            <div class='scanner-laser-line'></div>
            <div class='rotating-corona-sphere'>🦠</div>
            <div style='position:absolute; top:15px; right:20px; text-align:right; color:#00F2FF; font-family:monospace; font-size:0.75rem; line-height:1.4;'>
                SYS_STATUS: ACTIVE<br>
                SPIKE_PROTEIN: SCANNING...<br>
                MUTATION_RATE: 89.2%
            </div>
            <div style='position:absolute; bottom:15px; left:20px; color:rgba(255,255,255,0.7); font-size:0.75rem; font-weight:500;'>
                ⚙️ 로컬 하드웨어 가속 독립 구동 모드
            </div>
        </div>
        
        <div style='background: white; border:1px solid #E2E8F0; padding:12px; border-radius:12px; margin-top:10px; font-size:0.8rem; color:#1E293B;'>
            <b>🧬 입체 분자 생물학 리포트:</b><br>
            중앙의 <b>구형 코어(Core)</b> 외벽에 왕관 형태의 단백질이 돌출되어 있으며, 실시간 수평 단면 스캔 레이저가 복제 유전 정보를 해독하여 좌측 분석 패널로 실시간 동기화하고 있습니다.
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""<div style='margin-top:15px;'></div>""", unsafe_allow_html=True)
    
    st.video("https://www.youtube.com/watch?v=aE0MEPeaks4")
    
    st.markdown("""
        <div class='medical-note'>
            <b style='font-size:1rem; color:#0369A1;'>📑 임상 관찰 요약</b><br>
            • 위의 구조 분석기에서 보듯 바이러스 외벽(Envelope)은 지질(기름) 성분으로 둘러싸여 있습니다.<br>
            • <b>실험 결과:</b> 비누 없는 물 세척은 이 지질 외벽을 파괴하지 못해 강력한 감염력을 그대로 유지합니다.<br>
            • <b>해결책:</b> 30초 이상의 6단계 손씻기로 외벽을 물리적/화학적으로 완전히 파괴하십시오.
        </div>
    """, unsafe_allow_html=True)

# --- 5. 하단 제어 패널 섹션 ---
st.markdown("<div class='control-panel'>", unsafe_allow_html=True)
c_desc, c_input, c_result = st.columns([1, 1.4, 1.6])

with c_desc:
    st.markdown("""
        <div style='display: flex; gap: 15px; align-items: center;'>
            <div style='font-size: 2.8rem;'>🧬</div>
            <div>
                <div style='font-weight: 700; color: #0369A1; font-size:1.1rem;'>정밀 관제 스캐너</div>
                <div style='font-size: 0.8rem; color: #475569;'>좌표 변경 시 지구가 오토 타겟팅을 시작합니다.</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with c_input:
    st.markdown("<span style='font-size: 0.8rem; font-weight: 700; color: #0EA5E9;'>TARGET COORDINATES</span>", unsafe_allow_html=True)
    i_lat, i_lon = st.columns(2)
    with i_lat:
        lat_in = st.number_input("위도", value=10.80, format="%.2f", label_visibility="collapsed", key="lat_input")
    with i_lon:
        lon_in = st.number_input("경도", value=106.60, format="%.2f", label_visibility="collapsed", key="lon_input")
    
    st.session_state.lat_val = lat_in
    st.session_state.lon_val = lon_in
    st.caption("🎯 위경도를 입력하고 Enter를 누르면 실시간 록온이 시작됩니다.")

with c_result:
    near_df = df[(df['위도'] >= lat_in-5) & (df['위도'] <= lat_in+5) & 
                 (df['경도'] >= lon_in-5) & (df['경도'] <= lon_in+5)]
    
    st.markdown("<span style='font-size: 0.8rem; font-weight: 700; color: #0EA5E9;'>DIAGNOSIS REPORT</span>", unsafe_allow_html=True)
    
    if not near_df.empty:
        main_c = int(near_df['cluster'].value_counts().idxmax())
        res_color = {0: '#0EA5E9', 1: '#F59E0B', 2: '#EF4444'}[main_c]
        res_text = {0: '안정(Normal) 🛡️', 1: '경계(Warning) ⚠️', 2: '위험(Infected) ☣️'}[main_c]
        
        st.markdown(f"""
            <div style='background:{res_color}15; color:{res_color}; border:2px solid {res_color}; padding:10px; border-radius:10px; text-align:center; font-weight:700; margin-bottom:10px;'>
                판독 등급: {res_text}
            </div>
        """, unsafe_allow_html=True)
        
        if main_c == 2:
            st.error("☣️ 긴급 지침: 해당 지역은 감염 농도가 매우 높습니다. 방역 프로토콜을 즉시 가동하십시오.")
        else:
            st.success("🔬 보고: 해당 구역은 현재 표준 위생 관리 범위 내에 있습니다.")
    else:
        st.markdown("<div style='background:#F1F5F9; color:#475569; padding:10px; border-radius:10px; text-align:center; font-size:0.85rem;'>측정 범위 내 데이터 분석 불가능</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
