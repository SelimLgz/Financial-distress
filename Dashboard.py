import streamlit as st
import pandas as pd
import pickle

# ====================================
# BASIC CONFIGURATION
# ====================================

st.title("💳 Simple Credit Calculator")
st.write("Enter client information to calculate default risk")

# ====================================
# 1. LOAD MODEL
# ====================================

st.sidebar.header("📂 Load Your Model")
uploaded_model = st.sidebar.file_uploader("Select your model (.pkl)", type=['pkl'])

if uploaded_model is None:
    st.warning("⚠️ Please upload your model in the sidebar on the left")
    st.stop()

# Load the model
model_package = pickle.load(uploaded_model)
model = model_package['model']
scaler = model_package['scaler']
feature_names = model_package['feature_names']

st.success("✅ Model successfully loaded!")

# ====================================
# 2. SIMPLE FORM
# ====================================

st.header("📝 Client Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    monthly_income = st.number_input("Monthly Income (€)", min_value=0, value=3000)
    debt_ratio = st.number_input("Debt Ratio (0 to 1)", min_value=0.0, max_value=2.0, value=0.3, step=0.01)
    revolving_util = st.number_input("Revolving Credit Utilization (0 to 1)", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)

with col2:
    num_credit_lines = st.number_input("Open Credit Lines", min_value=0, max_value=30, value=5)
    num_real_estate = st.number_input("Real Estate Loans", min_value=0, max_value=10, value=1)
    late_30_59 = st.number_input("30-59 Days Late", min_value=0, max_value=10, value=0)
    late_60_89 = st.number_input("60-89 Days Late", min_value=0, max_value=10, value=0)
    late_90 = st.number_input("90+ Days Late", min_value=0, max_value=10, value=0)

# ====================================
# 3. SCORE CALCULATION
# ====================================

if st.button("🔍 CALCULATE SCORE", type="primary", use_container_width=True):
    
    # Create client data
    client_data = {
        'age': age,
        'MonthlyIncome': monthly_income,
        'DebtRatio': debt_ratio,
        'RevolvingUtilizationOfUnsecuredLines': revolving_util,
        'NumberOfDependents': num_dependents,
        'NumberOfOpenCreditLinesAndLoans': num_credit_lines,
        'NumberRealEstateLoansOrLines': num_real_estate,
        'NumberOfTime30-59DaysPastDueNotWorse': late_30_59,
        'NumberOfTime60-89DaysPastDueNotWorse': late_60_89,
        'NumberOfTimes90DaysLate': late_90
    }
    
    # Convert to DataFrame
    client_df = pd.DataFrame([client_data])
    
    # Add missing features with 0
    for feature in feature_names:
        if feature not in client_df.columns:
            client_df[feature] = 0
    
    # Reorder columns
    client_df = client_df[feature_names]
    
    # Normalize
    client_scaled = scaler.transform(client_df)
    
    # Predict
    probability = model.predict_proba(client_scaled)[0][1]
    
    # ====================================
    # 4. DISPLAY RESULTS
    # ====================================
    
    st.markdown("---")
    st.header("📊 RESULTS")
    
    # Score in %
    score_pct = probability * 100
    
    # Determine risk level
    if score_pct >= 70:
        risk_level = "🔴 HIGH"
        risk_color = "red"
        decision = "❌ DECLINE"
    elif score_pct >= 30:
        risk_level = "🟡 MODERATE"
        risk_color = "orange"
        decision = "⚠️ REVIEW"
    else:
        risk_level = "🟢 LOW"
        risk_color = "green"
        decision = "✅ APPROVE"
    
    # Show results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Default Probability", f"{score_pct:.1f}%")
    
    with col2:
        st.markdown(f"### Risk Level")
        st.markdown(f"# {risk_level}")
    
    with col3:
        st.markdown(f"### Decision")
        st.markdown(f"# {decision}")
    
    # Visual progress bar
    st.progress(probability)
    
    # Simple explanations
    st.markdown("---")
    st.subheader("💡 Explanations")
    
    if score_pct < 30:
        st.success("""
        ✅ **Good profile**
        - Low default risk
        - Recommendation: Approve credit
        """)
    elif score_pct < 70:
        st.warning("""
        ⚠️ **Profile to watch**
        - Moderate default risk
        - Recommendation: Review in detail
        - Possible acceptance with conditions
        """)
    else:
        st.error("""
        🚨 **Risky profile**
        - High default risk
        - Recommendation: Decline credit
        """)
    
    # Risk factors
    st.markdown("### 🔍 Identified Factors")
    
    if late_90 > 0:
        st.error(f"🚨 {late_90} severe delay(s) of 90+ days")
    if revolving_util > 0.8:
        st.warning(f"⚠️ High credit utilization: {revolving_util*100:.0f}%")
    if debt_ratio > 0.4:
        st.warning(f"⚠️ High debt ratio: {debt_ratio*100:.0f}%")
    if monthly_income < 1500:
        st.warning(f"⚠️ Low monthly income: {monthly_income}€")
    
    if late_90 == 0 and revolving_util < 0.3 and debt_ratio < 0.3:
        st.success("✅ Good payment history and reasonable debt level")

# ====================================
# USAGE INSTRUCTIONS
# ====================================

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📖 How to use

1. **Upload your model** (.pkl) at the top
2. **Fill in** client information
3. **Click** on "Calculate Score"
4. **Check** results and decision

### 💡 Tips
- All fields are required
- Enter 0 if no late payments
- Score ranges from 0% (good) to 100% (bad)
""")

# ====================================
# LAUNCH INSTRUCTIONS
# ====================================

print("""
🚀 SIMPLE DASHBOARD READY!

USAGE:
1. Save this code into a file: dashboard.py
2. In the terminal: streamlit run dashboard.py
3. A web page will automatically open
4. Upload your .pkl model in the sidebar
5. Fill in the fields and click "Calculate"
6. To exit: Ctrl+C in the terminal
THAT'S IT! 😊
""")
