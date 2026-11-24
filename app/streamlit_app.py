import math

import requests
import streamlit as st

API_URL = "https://cnc-api-943785405592.us-central1.run.app/predict"

# overstrain limits by product type, from the dataset documentation
STRAIN_LIMITS = {"L": 11000, "M": 12000, "H": 13000}

# label, min, max, default, step for each numeric input (ranges match the training data)
INPUTS = {
    "air_temp_k": ("Air temperature (K)", 295.3, 304.5, 300.0, 0.1),
    "process_temp_k": ("Process temperature (K)", 305.7, 313.8, 310.0, 0.1),
    "rotational_speed_rpm": ("Rotational speed (rpm)", 1168, 2886, 1500, 1),
    "torque_nm": ("Torque (Nm)", 3.8, 76.6, 40.0, 0.1),
    "tool_wear_min": ("Tool wear (min)", 0, 253, 100, 1),
}

st.set_page_config(page_title="CNC Failure Predictor", page_icon="🛠️")

for name, (_, _, _, default, _) in INPUTS.items():
    st.session_state.setdefault(f"{name}_box", default)
    st.session_state.setdefault(f"{name}_slider", default)


def sync(source, target):
    st.session_state[target] = st.session_state[source]


def linked_input(name):
    label, low, high, _, step = INPUTS[name]
    box_key, slider_key = f"{name}_box", f"{name}_slider"
    st.number_input(
        label, min_value=low, max_value=high, step=step, key=box_key,
        on_change=sync, args=(box_key, slider_key),
        help=f"Allowed range: {low:,} to {high:,} (the range seen in the training data)",
    )
    st.slider(
        label, min_value=low, max_value=high, step=step, key=slider_key,
        on_change=sync, args=(slider_key, box_key), label_visibility="collapsed",
    )
    return st.session_state[box_key]


st.title("CNC Machine Failure Predictor")
st.write(
    "Enter the machine's current conditions by typing a value or dragging its slider, "
    "then click **Predict** to estimate the risk of failure on this run. "
    "The model was trained on the AI4I 2020 predictive maintenance dataset."
)

col1, col2 = st.columns(2)

with col1:
    product_type = st.selectbox("Product quality type", ["L", "M", "H"])
    air_temp = linked_input("air_temp_k")
    process_temp = linked_input("process_temp_k")

with col2:
    speed = linked_input("rotational_speed_rpm")
    torque = linked_input("torque_nm")
    tool_wear = linked_input("tool_wear_min")

payload = {
    "type": product_type,
    "air_temp_k": air_temp,
    "process_temp_k": process_temp,
    "rotational_speed_rpm": speed,
    "torque_nm": torque,
    "tool_wear_min": tool_wear,
}

if st.button("Predict", type="primary"):
    with st.spinner("Asking the model (the first request may take a few seconds)..."):
        try:
            response = requests.post(API_URL, json=payload, timeout=30)
            response.raise_for_status()
            st.session_state["result"] = response.json()
            st.session_state["predicted_payload"] = payload
        except requests.exceptions.RequestException as e:
            st.error(f"Could not reach the prediction API: {e}")
            st.stop()

if "result" not in st.session_state:
    st.stop()

if st.session_state["predicted_payload"] != payload:
    st.info("Inputs changed since the last prediction. Click **Predict** to update.")

shown = st.session_state["predicted_payload"]
probability = st.session_state["result"]["failure_probability"]
prediction = st.session_state["result"]["prediction"]

st.divider()
st.metric("Failure probability", f"{probability:.1%}")
st.progress(min(probability, 1.0))

if prediction == 1:
    st.error("⚠️ High failure risk. Inspect the machine before running.")
else:
    st.success("✅ Low failure risk. Safe to run.")

# same physics as the engineered features in src/features.py
temp_diff = shown["process_temp_k"] - shown["air_temp_k"]
power = shown["torque_nm"] * (shown["rotational_speed_rpm"] * 2 * math.pi / 60)
strain = shown["tool_wear_min"] * shown["torque_nm"]
strain_limit = STRAIN_LIMITS[shown["type"]]

heat_risk = temp_diff < 8.6 and shown["rotational_speed_rpm"] < 1380
power_risk = power < 3500 or power > 9000
strain_risk = strain > strain_limit
wear_risk = shown["tool_wear_min"] >= 200

st.subheader("What's driving this prediction")
st.caption(
    "Each failure type in the dataset follows a physical rule. "
    "These are the values behind those rules for the settings that were predicted."
)

c1, c2 = st.columns(2)

with c1:
    st.metric("Temperature gap (K)", f"{temp_diff:.1f}")
    if heat_risk:
        st.error("Heat buildup risk: gap under 8.6 K at under 1,380 rpm")
    else:
        st.success("Heat: OK (risk when gap < 8.6 K and speed < 1,380 rpm)")

    st.metric("Power (W)", f"{power:,.0f}")
    if power_risk:
        st.error("Power risk: outside the 3,500 to 9,000 W safe range")
    else:
        st.success("Power: OK (safe range 3,500 to 9,000 W)")

with c2:
    st.metric("Strain (tool wear × torque)", f"{strain:,.0f}")
    if strain_risk:
        st.error(f"Overstrain risk: above the {strain_limit:,} limit for type {shown['type']}")
    else:
        st.success(f"Strain: OK (limit {strain_limit:,} for type {shown['type']})")

    st.metric("Tool wear (min)", f"{shown['tool_wear_min']}")
    if wear_risk:
        st.error("Tool wear risk: tools wear out between 200 and 240 min")
    else:
        st.success("Tool wear: OK (tools wear out between 200 and 240 min)")

with st.expander("Request sent to the API"):
    st.json(shown)
