import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import numpy as np

# 페이지 설정
st.set_page_config(
    page_title="지방 지진 데이터 분석 시스템",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 데이터 불러오기
try:
    df_new = pd.read_csv("earthquake.csv")
except FileNotFoundError:
    st.error("데이터 파일('earthquake.csv')을 찾을 수 없습니다.")
    st.stop()

# 위험도 사전
risk_dict = {0: '높음', 1: '낮음', 2: '중간'}

# [보정] 원본 이미지처럼 고대비로 꽉 찬 시인성을 주기 위해 형광 발색 HEX 코드로 변경
# Cluster 0: 위험도 높음 (붉은색 계열)
# Cluster 1: 위험도 낮음 (밝은 파랑/하늘색 계열)
# Cluster 2: 위험도 중간 (노란색/연두색 계열)
colors = {0: '#FF1E1E', 1: '#00E5FF', 2: '#FFFF00'}

# 전체 인터페이스 다크 모드 스타일 레이아웃
st.markdown("""
<style>
.stApp {
    background-color: #111217 !important;
}
body {
    background-color: #111217;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1, h2, h3, p, span, label {
    color: #ffffff !important;
}
[data-testid="stSidebar"] {
    background-color: #1e2029 !important;
    border-right: 1px solid #2d313f;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stButton>button {
    background: linear-gradient(135deg, #cc0000, #ff3333) !important;
    color: white !important;
    border-radius: 4px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
    letter-spacing: 1px;
    width: 100%;
    box-shadow: 0 0 10px rgba(255, 51, 51, 0.4);
}
[data-testid="stSidebar"] .stButton>button:hover {
    background: linear-gradient(135deg, #ee0000, #ff5555) !important;
    box-shadow: 0 0 15px rgba(255, 51, 51, 0.7);
}
.stTitle {
    color: #ff3333 !important;
    font-weight: 800;
    letter-spacing: -0.5px;
}
.stAlert {
    background-color: #1a1c23 !important;
    border: 1px solid #2d313f !important;
    border-radius: 6px;
}
</style>
""", unsafe_allow_html=True)

# 사이드바 구성
st.sidebar.title("🚨 지진 속보 모니터")
st.sidebar.markdown("---")
st.sidebar.subheader("진앙지 위치 입력")
lat = st.sidebar.number_input("위도 (Latitude)", value=37.5, format="%.4f")
lon = st.sidebar.number_input("경도 (Longitude)", value=127.0, format="%.4f")
st.sidebar.markdown("---")
analysis_button = st.sidebar.button("위험도 분석 실행")

# 메인 화면
st.title("🖥️ 지진 파동 정보 분석 시스템")
st.markdown("---")
st.write("본 시스템은 입력하신 위도와 경도 주변의 과거 지진 데이터를 분석하여 지진 위험도를 평가합니다. 공식적인 지진 예보가 아니며, 참고 자료로 활용하시기 바랍니다.")

if analysis_button:
    # 1. 사용자가 지정한 반경 계산 (원본 범위 계산 로직 유지)
    near_df = df_new[
        (df_new['위도'] >= lat - 5) &
        (df_new['위도'] <= lat + 5) &
        (df_new['경도'] >= lon - 5) &
        (df_new['경도'] <= lon + 5)
    ]

    if len(near_df) == 0:
        st.sidebar.warning("해당 위치 주변에 충분한 지진 데이터가 없습니다. 분석 범위를 넓혀보세요.")
    else:
        # 군집 비율 및 대표 위험군 연산
        cluster_ratio = near_df['cluster'].value_counts(normalize=True)
        main_cluster = cluster_ratio.idxmax()
        danger_level = risk_dict[main_cluster]
        
        if danger_level == '높음':
            st.error(f"⚠️ **[지진속보] 예상 위험도: 높음** — 강한 진동 유의 및 추가 여진에 대비하십시오.")
        elif danger_level == '중간':
            st.warning(f"⚠️ **[지진정보] 예상 위험도: 중간** — 주변 지역에 흔들림이 감지될 수 있습니다.")
        else:
            st.success(f"✅ **[지진정보] 예상 위험도: 낮음** — 현재 시점 기준 특이 진동 징후가 낮습니다.")

        # 지도 레이어 생성 (Esri 다크 위성뷰 적용)
        esri_satellite = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
        attr = "Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS"
        
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=6, 
            tiles=esri_satellite, 
            attr=attr
        )

        # -----------------------------------------------------------------
        # 진앙지 중심 지진파(동심원) 속보 그래픽 레이어
        # -----------------------------------------------------------------
        wave_radii = [25, 60, 110, 170]
        wave_opacities = [0.8, 0.5, 0.25, 0.1]
        wave_dashes = [None, "5, 5", "6, 10", "10, 15"]

        for r, op, dash in zip(wave_radii, wave_opacities, wave_dashes):
            folium.Circle(
                location=[lat, lon],
                radius=r * 1000,
                color="#FF3333",
                weight=2 if dash else 3,
                fill=True,
                fill_color="#FF3333",
                fill_opacity=op * 0.12,
                dash_array=dash
            ).add_to(m)

        # -----------------------------------------------------------------
        # 🔥 [핵심 똑바로 보정] 위험군 포인트 누락 방지 매핑 로직
        # -----------------------------------------------------------------
        # 전체 데이터 중 랜덤 500개만 뿌리면 분석 구역 정보가 전부 지워집니다.
        # 타겟 구역 안의 알짜배기 실제 데이터를 우선 추출하고 부족한 부분을 전체 샘플로 채웁니다.
        
        max_display_points = 600  # 화면 과부하를 막으면서 이미지를 꽉 채울 표출 개수
        
        # 현재 진앙지 주변(near_df) 데이터를 화면에 최우선적으로 100% 전부 확보합니다.
        near_points = near_df.copy()
        
        # 만약 주변 데이터가 너무 많다면 연산 성능을 위해 상한선 제한, 부족하다면 전체 데이터에서 채움
        if len(near_points) > max_display_points:
            display_df = near_points.sample(max_display_points, random_state=42)
        else:
            # 타겟 영역 데이터를 무조건 포함하고, 남는 자리에 배경용 전역 데이터를 추가 서포트
            remain_slots = max_display_points - len(near_points)
            global_sample = df_new.drop(near_points.index).sample(min(remain_slots, len(df_new) - len(near_points)), random_state=42)
            display_df = pd.concat([near_points, global_sample])

        # 과거 지진 포인트 마커 지도에 그리기 (시인성 극대화 셋업)
        for i in range(len(display_df)):
            row = display_df.iloc[i]
            cluster = int(row['cluster'])
            scale = float(row['규모'])
            
            # 원본 지도 이미지처럼 경계선이 뭉개지지 않고 선명하게 밀집되도록 투명도와 두께 최적화
            folium.CircleMarker(
                location=[row['위도'], row['경도']],
                radius=max(3, scale * 1.6),        # 미소 지진도 최소 크기(3)를 확보하여 점이 똑바로 보이게 설정
                color='#FFFFFF',                     # 테두리는 원본처럼 깔끔한 백색 마킹
                weight=0.6,                          # 경계선 굵기를 조절하여 군집 겹침 현상 시각화 강화
                fill=True,
                fill_color=colors[cluster],          # 위험군 분류 색상 매핑
                fill_opacity=0.85                    # 흐릿하지 않고 원본 지도처럼 진하게 표시하기 위해 밀도 상향
            ).add_to(m)

        # 사용자 위치 표시 (진앙지 에피센터 센터 앵커)
        folium.Marker(
            location=[lat, lon],
            popup="진앙지 (EPICENTER)",
            icon=folium.Icon(color='red', icon='play')
        ).add_to(m)

        # 스트림릿 지도 렌더링
        st_folium(m, use_container_width=True, height=600, returned_objects=[])

# 푸터
st.markdown("---")
st.markdown("<p style='text-align: center; color: #525866; font-size: 0.85rem;'>EARTHQUAKE EARLY WARNING MONITORING SYSTEM | © 2026 지진 데이터 분석팀.</p>", unsafe_allow_html=True)
