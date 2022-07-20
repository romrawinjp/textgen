import streamlit as st

st.title("💬🦜 Text Generator")
st.subheader("GPT2 Thai Text Generator")

text = st.text_input("Adding some text")

# load model
from chargpt.generator import generate

# def generator():
#     if text != "":
#         text_gen = generate(text)
#         # print(text_gen)
#         for i in text_gen:
#             st.write(i)  
#     else: 
#         st.write("No text input...")

# generate text 
# st.button("Generate...", on_click=generator)
if st.button("Generate...") and text != "":
    text_gen = generate(text)
    # print(text_gen)
    for i in text_gen:
        st.write(i)  
else: 
    st.write("No text input...")
# try: 
#     st.button("Generate...", on_click=generator)
# except:
#     st.error("Add some text")
    