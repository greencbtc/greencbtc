import streamlit as st
import main

st.set_page_config(page_title="TML Green CBTC", layout="wide")
st.title("Train Energy Consumption Estimation")

st.sidebar.header("Calculation Parameters")
eff = st.sidebar.number_input("Motor Efficiency", 0.0, 1.0, 0.80, 0.05)
regen = st.sidebar.number_input("Regen Efficiency", 0.0, 1.0, 0.2, 0.05)

file = st.file_uploader("Upload your RTD .csv file", type="csv")
st.caption("**Note: Please ensure your file name includes the trainset ID (e.g., T56).")

if file:
    df, is_sp1900 = main.preprocess(file)
    df, kwh, dist, kwh_km  = main.calculate_energy(df, is_sp1900, regen, eff)
    main.export_results(df)
    
    cols = st.columns(4)
    train_name = "SP1900" if is_sp1900 else "TML C Train"
    cols[0].metric("Train Type", train_name)
    cols[1].metric("Running Distance", f"{dist:.4f} km")
    cols[2].metric("Total Energy", f"{kwh:.4f} kWh")
    cols[3].metric("kWh/car-km", f"{kwh_km:.4f} kWh/car-km")

    st.plotly_chart(main.plot(df))