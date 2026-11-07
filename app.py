"""Streamlit chat UI for the Weather-Bot. Run with: streamlit run app.py"""

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from agent import ask_weather_bot
from guardrails import (
    MAX_INPUT_CHARS,
    MAX_REQUESTS_PER_MINUTE,
    MAX_REQUESTS_PER_SESSION,
    REFUSAL_MESSAGE,
    RateLimiter,
    check_input,
)

st.set_page_config(page_title="Weather-Bot", page_icon="🌤️", layout="wide")
st.title("🌤️ Weather-Bot")
st.caption("Ask about current weather or a 5-day forecast for any city. Powered by grok-4.7 + OpenWeather.")

# --- Session state -----------------------------------------------------------
defaults = {
    "messages": [],        # {"role", "content", "weather"}
    "limiter": RateLimiter(),
    "security_log": [],    # {"time", "type", "message"}
    "cities": [],          # cities looked up this session
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def log_block(block_type: str, text: str):
    st.session_state.security_log.append({
        "time": datetime.now().strftime("%I:%M:%S %p"),
        "type": block_type,
        "message": text[:60] + ("..." if len(text) > 60 else ""),
    })


# --- Sidebar -----------------------------------------------------------------
EXAMPLES = [
    "What's the weather in Amarillo right now?",
    "Give me the 5-day forecast for Amarillo",
    "What's the weather in Dallas?",
]

with st.sidebar:
    st.header("Session stats")
    c1, c2 = st.columns(2)
    c1.metric("Requests", f"{st.session_state.limiter.total}/{MAX_REQUESTS_PER_SESSION}")
    c2.metric("Blocked", len(st.session_state.security_log))
    if st.session_state.cities:
        st.caption("Cities looked up: " + ", ".join(st.session_state.cities))

    st.divider()
    st.header("Try asking")
    for q in EXAMPLES:
        if st.button(q, width="stretch"):
            st.session_state.pending = q

    st.divider()
    st.header("Security features")
    st.markdown(
        "- Weather-only questions\n"
        "- Prompt injection filter\n"
        f"- {MAX_INPUT_CHARS} character message limit\n"
        f"- Rate limit: {MAX_REQUESTS_PER_MINUTE}/min, {MAX_REQUESTS_PER_SESSION}/session\n"
        "- Max 3 tool-call rounds\n"
        "- API keys stored in .env, never shown"
    )

    with st.expander(f"Security log ({len(st.session_state.security_log)})"):
        if not st.session_state.security_log:
            st.caption("No blocked requests yet.")
        for entry in reversed(st.session_state.security_log):
            st.markdown(f"**{entry['time']}** - {entry['type']}  \n`{entry['message']}`")

    st.divider()
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
        st.rerun()


# --- Weather display helpers -------------------------------------------------
def show_current(data: dict):
    st.markdown(f"**📍 Current weather: {data['city']}** ({data['condition']})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Temperature", f"{data['temperature_f']}°F")
    c2.metric("Feels like", f"{data['feels_like_f']}°F")
    c3.metric("Humidity", f"{data['humidity_pct']}%")
    c4.metric("Wind", f"{data['wind_speed_mph']} mph")


def show_forecast(data: dict):
    st.markdown(f"**📅 5-day forecast: {data['city']}**")
    cols = st.columns(len(data["forecast"]))
    for col, day in zip(cols, data["forecast"]):
        col.metric(day["date"], f"{day['high_f']}° / {day['low_f']}°")
        col.caption(day["condition"])

    # High / low temperature chart
    df = pd.DataFrame(data["forecast"])
    df = df.melt(id_vars="date", value_vars=["high_f", "low_f"], var_name="Type", value_name="Temp (°F)")
    df["Type"] = df["Type"].map({"high_f": "High", "low_f": "Low"})
    chart = (
        alt.Chart(df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:N", sort=None, title=None),
            y=alt.Y("Temp (°F):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("Type:N", scale=alt.Scale(domain=["High", "Low"], range=["#e4572e", "#2e86de"])),
            tooltip=["date", "Type", "Temp (°F)"],
        )
        .properties(height=220)
    )
    st.altair_chart(chart, width="stretch")


def show_weather(results: list[dict]):
    for r in results:
        data = r["data"]
        if "error" in data:
            continue
        with st.container(border=True):
            if r["tool"] == "get_current_weather":
                show_current(data)
            elif r["tool"] == "get_weather_forecast":
                show_forecast(data)


# --- Chat history ------------------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("weather"):
            show_weather(msg["weather"])

# --- New message -------------------------------------------------------------
typed = st.chat_input("e.g. What's the weather in Amarillo right now?")
prompt = typed or st.session_state.pop("pending", None)

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        weather = []
        allowed, reason = st.session_state.limiter.allow()   # 1. rate limit
        if not allowed:
            log_block("Rate limit", prompt)
        else:
            allowed, reason = check_input(prompt)            # 2. input validation
            if not allowed:
                kind = "Message too long" if len(prompt) > MAX_INPUT_CHARS else "Prompt injection"
                log_block(kind, prompt)

        if not allowed:
            reply = reason                                   # blocked: LLM never called
        else:
            with st.spinner("Checking the weather..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = ask_weather_bot(prompt, history)    # 3. agent (+ output filter)
            reply, weather = result["reply"], result["weather"]

            if REFUSAL_MESSAGE in reply:
                log_block("Off-topic (refused by model)", prompt)
            if reply.startswith("🚫"):
                log_block("Output filter", prompt)
            for r in weather:
                city = r["data"].get("city")
                if city and city not in st.session_state.cities:
                    st.session_state.cities.append(city)

        st.markdown(reply)
        show_weather(weather)

    st.session_state.messages.append({"role": "assistant", "content": reply, "weather": weather})
    st.rerun()  # refresh so the sidebar stats update
