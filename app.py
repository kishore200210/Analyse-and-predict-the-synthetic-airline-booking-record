import os
import re
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

DATA_PATH = Path("clarity_bookings_dataset.csv")

keyword_rules = {
    "Refund Issues": [r"\brefund\b", r"\bchargeback\b", r"\breturn\b"],
    "Schedule Change": [r"\bdelay\b", r"\breschedul\b", r"\bchange flight\b", r"\btime change\b"],
    "Baggage": [r"\bbaggage\b", r"\blost luggage\b", r"\bdamaged bag\b"],
    "Pricing Error": [r"\bprice\b", r"\bfare\b", r"\bcharged\b", r"\bextra fee\b"],
    "Ticketing Issues": [r"\bticket\b", r"\bbooking error\b", r"\bconfirmation\b", r"\breservation\b"],
    "Customer Service": [r"\bagent\b", r"\bcustomer service\b", r"\bcall center\b"],
}


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rule_label(text: str) -> str:
    text = clean_text(text)
    for label, patterns in keyword_rules.items():
        if any(re.search(p, text) for p in patterns):
            return label
    return "Other"


def classify_with_groq(text: str) -> str:
    api_key = os.getenv("GROQ_API_KEY")
    model = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

    if not api_key:
        return rule_label(text)

    try:
        client = Groq(api_key=api_key)
        prompt = f"""
Classify this complaint into one category only:
Refund Issues, Schedule Change, Baggage, Pricing Error,
Ticketing Issues, Customer Service, Other.
Complaint: {text}
Return only the category name.
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a careful complaint categorizer."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=30,
        )
        return response.choices[0].message.content.strip().splitlines()[-1].strip()
    except Exception:
        return rule_label(text)


def main():
    st.set_page_config(page_title="Complaint Summary Demo", page_icon="🧾", layout="wide")
    st.title("Complaint Summary Demo")
    st.write("Paste one or more complaint texts and get a quick support summary.")

    sample_text = """Refund not processed after 30 days
Flight delayed by 3 hours, requesting compensation
Extra baggage fee charged incorrectly
Schedule change not communicated
Unable to add extra baggage through portal"""

    user_text = st.text_area("Enter complaint text(s), one per line:", value=sample_text, height=180)

    if st.button("Generate Summary"):
        lines = [line.strip() for line in user_text.splitlines() if line.strip()]

        if not lines:
            st.warning("Please enter at least one complaint text.")
            st.stop()

        rows = []
        for text in lines:
            category = classify_with_groq(text)
            rows.append({"complaint": text, "keyword_category": rule_label(text), "llm_category": category})

        df = pd.DataFrame(rows)

        st.subheader("Result Table")
        st.dataframe(df, use_container_width=True)

        summary = df['llm_category'].value_counts().reset_index()
        summary.columns = ['Category', 'Count']

        st.subheader("Category Summary")
        st.bar_chart(summary.set_index('Category'))

        st.subheader("Support Summary")
        top = summary.head(3).to_dict('records')
        for item in top:
            st.write(f"- {item['Category']}: {item['Count']} complaint(s)")

        st.write("This prototype is a simple support-team summary view for complaint batches.")


if __name__ == "__main__":
    main()
