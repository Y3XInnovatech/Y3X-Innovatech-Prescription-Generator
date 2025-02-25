from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
class prescription_generator():
    def __init__(self, name, age, height, weight, spo2, temperature, previous_medical_history, ecg_model_results, health_guidance, output_path):
        self.name = name
        self.height = height
        self.age = age
        self.weight = weight
        self.spo2 = spo2
        self.temperature = temperature
        self.previous_medical_history = previous_medical_history
        self.ecg_model_resuls = ecg_model_results
        self.patient_details =  {"Name": self.name, "Age":self.age, "Height":self.height, "Weight":self.weight, "SPO2":self.spo2, "ECG MODEL RESULTS":self.ecg_model_resuls, "Medical History":self.previous_medical_history}
        self.output_path = output_path
        self.health_guidance = health_guidance
                
    def split_text_to_fit_width(self, canvas, text, max_width):
        """
        Helper function to wrap text to fit within a given width.
        """
        lines = []
        words = text.split()
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            if canvas.stringWidth(test_line, "Helvetica", 12) <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        return lines
    
    def create_prescription_pdf(self):
        c = canvas.Canvas(self.output_path, pagesize=A4)
        width, height = A4  # Get page dimensions

        title = "Patient Prescription"
        c.setFont("Helvetica-Bold", 14)
        text_width = c.stringWidth(title, "Helvetica-Bold", 14)  
        title_x = (width - text_width) / 2 
        c.drawString(title_x, height - 50, title) 

        y = height - 80  
        c.setFont("Helvetica", 12)
        x = 50  

        # Calculate space needed for patient details
        num_details = len(self.patient_details)
        space_needed = num_details * 20 + 30  # 20px per line + extra space

        for key, value in self.patient_details.items():
            c.drawString(x, y, f"{key}: {value}")
            y -= 20  
            if y < 50:
                c.showPage() 
                y = height - 50  

        # Add extra spacing before printing responses
        y -= 30  # Ensuring a gap before the response section

        ### Printing the response
        max_text_width = width - 50
        wrapped_lines = self.split_text_to_fit_width(c, self.health_guidance, max_text_width)

        for line in wrapped_lines:
            if y < 50:  # If the space is less, move to a new page
                c.showPage()
                y = height - 50  
                c.setFont("Helvetica", 12)  # Reset font after page change

            c.drawString(50, y, line)  
            y -= 20  

        c.save()  
