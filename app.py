import streamlit as st
import os
from generate import generate_music

st.set_page_config(
    page_title="AI Music Generator",
    page_icon="🎵",
    layout="centered"
)

st.title("🎵 AI Music Generator")
st.write("Click the button below to generate a new MIDI music file using the trained AI model.")

if st.button("🎼 Generate Music"):
    with st.spinner("Generating music... Please wait..."):
        try:
            output_file = generate_music()

            st.success("✅ Music generated successfully!")

            with open(output_file, "rb") as file:
                st.download_button(
                    label="⬇️ Download MIDI File",
                    data=file,
                    file_name=os.path.basename(output_file),
                    mime="audio/midi"
                )

        except Exception as e:
            st.error(f"Error: {e}")