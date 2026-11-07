"""Streamlit chat UI for the Weather-Bot. Run with: streamlit run app.py"""

import streamlit as st

from agent import ask_weather_bot
from guardrails import RateLimiter, check_input

st.set_page_config(page_title="Weather-Bot", page_icon="🌤️")
st.title("🌤️ Weather-Bot")
st.caption("Ask about current weather or a 5-day forecast for any city. Powered by grok-4.7 + OpenWeather.")

# --- Session state -----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # {"role", "content", "weather"}
if "limiter" not in st.session_state:
    st.session_state.limiter = RateLimiter()


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
if prompt := st.chat_input("e.g. What's the weather in Amarillo right now?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        weather = []
        allowed, reason = st.session_state.limiter.allow()  # 1. rate limit
        if allowed:
            allowed, reason = check_input(prompt)           # 2. input validation
        if not allowed:
            reply = reason                                  # blocked: LLM never called
        else:
            with st.spinner("Checking the weather..."):
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                result = ask_weather_bot(prompt, history)   # 3. agent (+ output filter)
            reply, weather = result["reply"], result["weather"]

        st.markdown(reply)
        show_weather(weather)

    st.session_state.messages.append({"role": "assistant", "content": reply, "weather": weather})
