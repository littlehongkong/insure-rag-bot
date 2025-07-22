import streamlit as st
from app.api.routes import router
from app.core.insurance_analyzer import InsuranceAnalyzer

st.set_page_config(page_title="보험 정책 분석기", page_icon="📊")

# Initialize analyzer
analyzer = InsuranceAnalyzer()

# Main page
st.title("보험 정책 분석기")

# File upload section
uploaded_file = st.file_uploader("보험 약관 PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file is not None:
    # TODO: Implement file processing
    st.success("파일이 성공적으로 업로드되었습니다!")

# Query section
query = st.text_input("질문을 입력하세요:")
if st.button("분석하기"):
    if query:
        # TODO: Implement query processing
        st.write("분석 결과가 여기에 표시됩니다.")

# Comparison section
st.header("보험 정책 비교")
policy_a = st.text_input("보험사 A")
policy_b = st.text_input("보험사 B")

if st.button("비교하기"):
    if policy_a and policy_b:
        # TODO: Implement policy comparison
        st.write("비교 결과가 여기에 표시됩니다.")
