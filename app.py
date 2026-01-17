
import streamlit as st
import google.generativeai as genai

# -------------------------------
# 1. Setup Gemini API
# -------------------------------
genai.configure(api_key="AIzaSyAoygbh3qoHAallbamXG6JZJCmyIN09ykM")

model = genai.GenerativeModel("models/gemini-2.5-flash")

# -------------------------------
# 2. Streamlit Page Setup
# -------------------------------
st.set_page_config(page_title="Heart Disease Info Chatbot", page_icon="❤️", layout="wide")
st.title("❤️ Heart Disease Information Chatbot")

st.markdown("""
This chatbot provides **general information** about heart health and heart disease.  
⚠️ *It cannot diagnose or provide medical treatment.*  
Please consult a healthcare professional for personal medical concerns.
""")

# -------------------------------
# 3. Initialize chat history
# -------------------------------
if "chat_session" not in st.session_state:
    # create the first message as "system prompt" but using allowed user role
    st.session_state.chat_session = model.start_chat(history=[
        {
            "role": "user",
            "parts": [{
                "text": (
                    "You are a safe heart disease information chatbot. "
                    "You provide general information about symptoms, risks, and lifestyle tips. "
                    "Do NOT provide diagnosis, treatment, medication guidance, or emergencies advice. "
                    "For medical concerns, tell the user to consult a licensed doctor."
                )
            }]
        }
    ])

# -------------------------------
# 4. Chat input box
# -------------------------------
user_message = st.chat_input("Ask me anything about heart disease...")

# -------------------------------
# 5. Handle user messages
# -------------------------------
if user_message:
    st.chat_message("user").markdown(user_message)

    response = st.session_state.chat_session.send_message(
        {"role": "user", "parts": [{"text": user_message}]}
    )

    bot_reply = response.text
    st.chat_message("assistant").markdown(bot_reply)

# -------------------------------
# (Optional) Show chat history debug
# -------------------------------
# st.write(st.session_state.chat_session._history)
