
from pathlib import Path
import pandas as pd
import streamlit as st

CSV_FILE = (
    Path(__file__).resolve().parent
    / "attendance"
    / "attendance.csv"
)

st.set_page_config(
    page_title="Face Attendance Dashboard",
    page_icon="📋",
    layout="wide"
)

st.title("Face Recognition Attendance System")
st.write("Student Attendance Dashboard")

if not CSV_FILE.exists():
    st.info("No attendance records found.")

else:
    df = pd.read_csv(CSV_FILE)

    if df.empty:
        st.info("No attendance records yet.")

    else:
        col1, col2 = st.columns(2)

        col1.metric("Total Attendance", len(df))

        col2.metric(
            "Total Students",
            df["student_id"].nunique()
        )

        selected_date = st.selectbox(
            "Select Date",
            ["All Dates"] + sorted(
                df["date"].unique().tolist()
            )
        )

        if selected_date == "All Dates":
            result = df
        else:
            result = df[
                df["date"] == selected_date
            ]

        st.dataframe(
            result,
            use_container_width=True
        )

        st.download_button(
            "Download Attendance CSV",
            result.to_csv(index=False),
            file_name="attendance.csv",
            mime="text/csv"
        )
      
