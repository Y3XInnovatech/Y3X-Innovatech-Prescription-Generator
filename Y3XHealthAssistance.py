import streamlit as st ## type:ignore
from llama_cpp import Llama ##type:ignore
from prescribed_health_assistance import prescription_generator

class Y3XHealthAssistance:
    def __init__(self):
        # Load the Llama model from GGUF
        # self.llm = Llama.from_pretrained(
        #     repo_id="mav23/AlpaCare-llama1-7b-GGUF",
        #     filename="alpacare-llama1-7b.Q2_K.gguf"
        # )
        self.llm = Llama(model_path=r"C:\Users\swain\Downloads\alpacare-llama1-7b.Q2_K.gguf", n_ctx=2048)


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
                     f"You are an expert doctor. Based on the patient's health data, provide medical advice in minimum 150 words.\n\n"
                    f"Patient Information:\n"
                    f"- Name: {name}\n"
                    f"- Age: {age}\n"
                    f"- Height: {height} cm\n"
                    f"- Weight: {weight} kg\n"
                    f"- SpO2 Level: {spo2}%\n"
                    f"- Body Temperature: {temperature}°C\n"
                    f"- ECG Results: {ecg_results}\n"
                    f"- Medical History: {medical_history}\n\n"
                    f"🩺 **Doctor's Advice:**"
                )

                # Generate response using llama_cpp
                output = self.llm(
                    prompt=prompt_text,
                    max_tokens=512,
                    echo=False
                )

                response = output["choices"][0]["text"]
                print(response)

                # Create and download the prescription PDF
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
    app = Y3XHealthAssistance()
    app.generate_output()
