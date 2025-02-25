import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Y3X_BBSR.prescribed_health_assistance import prescription_generator

class Y3XHealthAssistance:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load model and tokenizer on GPU
        self.model_name = "BioMistral/BioMistral-7B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)

    def generate_output(self):
        st.title('Health Assistance Y3X Innovatech Bhubaneswar')

        with st.form(key='health_form'):
            name = st.text_input("Enter Your Name")
            age = st.number_input("Enter Your Age:", min_value=0)
            height = st.number_input("Enter Your Height (cm):", min_value=0.0)
            weight = st.number_input("Enter Your Weight (kg):", min_value=0.0)
            spo2 = st.number_input("Enter Your SpO2 (%):", min_value=0.0, max_value=100.0)
            temperature = st.number_input("Enter Your Body Temperature (°C):", min_value=0.0)
            ecg_results = st.text_input("Enter Your Model Results:")
            medical_history = st.text_input("Enter Your Previous Medical History:")

            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            if all([name, age, height, weight, spo2, temperature, ecg_results, medical_history]):
                prompt_text = (
                    f"You are an expert medical assistant. Based on the user's input, provide detailed health recommendations.\n\n"
                    f"Name: {name}\n"
                    f"Age: {age}\n"
                    f"Height: {height} cm\n"
                    f"Weight: {weight} kg\n"
                    f"SpO2 Level: {spo2}%\n"
                    f"Body Temperature: {temperature}°C\n"
                    f"ECG Results: {ecg_results}\n"
                    f"Medical History: {medical_history}\n\n"
                    f"Provide personalized health advice regarding physical health, necessary precautions, and potential concerns."
                )

                # Tokenize and move input to GPU
                inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

                with torch.no_grad():
                    output = self.model.generate(**inputs, max_new_tokens=512)

                response = self.tokenizer.decode(output[0], skip_special_tokens=True)

                output_path = f'{name}_health_report.pdf'
                prescription = prescription_generator(
                    name,
                    age,
                    height,
                    weight,
                    spo2,
                    temperature,
                    medical_history,
                    ecg_results,
                    response,
                    output_path
                )
                prescription.create_prescription_pdf()

                with open(output_path, "rb") as file:
                    st.download_button("Download Health Report", file, file_name=output_path, mime="application/pdf")

                st.write(response)
            else:
                st.error("Please fill in all the fields.")


if __name__ == "__main__":
    app =Y3XHealthAssistance()
    app.generate_output()
