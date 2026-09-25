import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# =========================================================
# DATASET
# =========================================================

data = {
    "Study_Hours": [
        2, 5, 1, 6, 4, 3, 7, 2, 5, 8,
        1, 4, 6, 3, 7, 2, 5, 8, 4, 6
    ],

    "Attendance": [
        65, 90, 55, 95, 80, 70, 98, 60, 88, 96,
        50, 75, 92, 68, 97, 58, 85, 99, 78, 91
    ],

    "Assignment_Score": [
        50, 85, 40, 92, 75, 60, 95, 45, 82, 96,
        35, 70, 88, 55, 94, 42, 80, 98, 72, 90
    ],

    "Internal_Marks": [
        45, 82, 38, 90, 70, 55, 94, 42, 78, 92,
        30, 68, 85, 50, 91, 40, 76, 96, 65, 88
    ],

    "Final_Marks": [
        48, 85, 35, 93, 72, 58, 96, 40, 80, 95,
        32, 70, 87, 52, 94, 38, 78, 97, 68, 90
    ]
}

df = pd.DataFrame(data)


# =========================================================
# TRAIN MODEL
# =========================================================

X = df[
    [
        "Study_Hours",
        "Attendance",
        "Assignment_Score",
        "Internal_Marks"
    ]
]

y = df["Final_Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)


# =========================================================
# RECOMMENDATION SYSTEM
# =========================================================

def generate_recommendations(
    study_hours,
    attendance,
    assignment_score,
    internal_marks,
    predicted_marks
):

    recommendations = []

    if study_hours < 3:
        recommendations.append(
            "Increase your daily study time to at least 3-4 hours."
        )

    if attendance < 75:
        recommendations.append(
            "Improve your attendance. Try to maintain at least 75% attendance."
        )

    if assignment_score < 60:
        recommendations.append(
            "Complete assignments regularly and improve your assignment scores."
        )

    if internal_marks < 50:
        recommendations.append(
            "Focus more on internal assessments and revise regularly."
        )

    if predicted_marks < 50:
        recommendations.append(
            "Follow a regular study schedule to improve your performance."
        )

    elif predicted_marks < 75:
        recommendations.append(
            "Your performance is moderate. Practice more to improve your marks."
        )

    else:
        recommendations.append(
            "Good performance! Continue your current study habits."
        )

    return recommendations


# =========================================================
# PREDICTION
# =========================================================

def predict_student(
    student_name,
    study_hours,
    attendance,
    assignment_score,
    internal_marks
):

    if not student_name:
        student_name = "Student"

    student_input = pd.DataFrame({
        "Study_Hours": [study_hours],
        "Attendance": [attendance],
        "Assignment_Score": [assignment_score],
        "Internal_Marks": [internal_marks]
    })

    predicted_marks = model.predict(student_input)[0]

    predicted_marks = max(
        0,
        min(100, predicted_marks)
    )

    if predicted_marks >= 75:
        performance = "Excellent"

    elif predicted_marks >= 60:
        performance = "Good"

    elif predicted_marks >= 50:
        performance = "Average"

    else:
        performance = "Needs Improvement"

    recommendations = generate_recommendations(
        study_hours,
        attendance,
        assignment_score,
        internal_marks,
        predicted_marks
    )

    recommendation_text = "\n".join(
        "• " + recommendation
        for recommendation in recommendations
    )

    return (
        f"{predicted_marks:.2f} / 100",
        performance,
        recommendation_text
    )


# =========================================================
# CUSTOM CSS
# =========================================================

css = """

body {
    background: #f4f7fb;
}

.gradio-container {
    max-width: 1400px !important;
    margin: auto !important;
}

.header {
    background: linear-gradient(135deg, #172554, #2563eb);
    color: white;
    padding: 28px;
    border-radius: 18px;
    margin-bottom: 20px;
}

.header h1 {
    color: white;
    font-size: 32px;
    margin-bottom: 5px;
}

.header p {
    color: #dbeafe;
    font-size: 16px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    border: 1px solid #e5e7eb;
}

.stat-card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    border: 1px solid #e5e7eb;
}

.stat-number {
    font-size: 28px;
    font-weight: bold;
    color: #2563eb;
}

.stat-label {
    color: #64748b;
    font-size: 14px;
}

"""

# =========================================================
# DASHBOARD
# =========================================================

with gr.Blocks(
    title="AI Student Performance Prediction",
    css=css,
    theme=gr.themes.Soft()
) as demo:

    # HEADER

    gr.HTML("""
    <div class="header">
        <h1>🎓 AI Student Performance Prediction</h1>
        <p>
            Personalized Learning Recommendation System
        </p>
    </div>
    """)

    # STATISTICS

    with gr.Row():

        gr.HTML("""
        <div class="stat-card">
            <div class="stat-number">20</div>
            <div class="stat-label">Students in Dataset</div>
        </div>
        """)

        gr.HTML("""
        <div class="stat-card">
            <div class="stat-number">5</div>
            <div class="stat-label">Academic Features</div>
        </div>
        """)

        gr.HTML("""
        <div class="stat-card">
            <div class="stat-number">AI</div>
            <div class="stat-label">Prediction Model</div>
        </div>
        """)

        gr.HTML("""
        <div class="stat-card">
            <div class="stat-number">100</div>
            <div class="stat-label">Maximum Marks</div>
        </div>
        """)

    gr.Markdown("## 🔮 Student Performance Prediction")

    # MAIN AREA

    with gr.Row():

        # LEFT SIDE

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown("### 📝 Student Details")

            student_name = gr.Textbox(
                label="Student Name",
                placeholder="Enter student name"
            )

            study_hours = gr.Number(
                label="Study Hours per Day",
                value=2,
                minimum=0,
                maximum=24
            )

            attendance = gr.Number(
                label="Attendance (%)",
                value=50,
                minimum=0,
                maximum=100
            )

            assignment_score = gr.Number(
                label="Assignment Score",
                value=55,
                minimum=0,
                maximum=100
            )

            internal_marks = gr.Number(
                label="Internal Marks",
                value=45,
                minimum=0,
                maximum=100
            )

            predict_button = gr.Button(
                "🔮 Predict Performance",
                variant="primary",
                size="lg"
            )

        # RIGHT SIDE

        with gr.Column(
            scale=1,
            elem_classes="card"
        ):

            gr.Markdown("### 🤖 AI Prediction Result")

            predicted_marks = gr.Textbox(
                label="Predicted Final Marks",
                interactive=False
            )

            performance = gr.Textbox(
                label="Performance Level",
                interactive=False
            )

            recommendations = gr.Textbox(
                label="💡 Personalized Recommendations",
                lines=8,
                interactive=False
            )

    # INFORMATION

    gr.Markdown("---")

    gr.Markdown("""
    ## 📊 How the AI Works

    The system uses **Linear Regression** to predict a student's
    final marks using:

    - 📚 Study Hours
    - 📅 Attendance
    - 📝 Assignment Score
    - 📖 Internal Marks

    Based on the predicted marks, the system also provides
    personalized learning recommendations.
    """)

    # BUTTON ACTION

    predict_button.click(
        predict_student,

        inputs=[
            student_name,
            study_hours,
            attendance,
            assignment_score,
            internal_marks
        ],

        outputs=[
            predicted_marks,
            performance,
            recommendations
        ]
    )


# =========================================================
# RENDER DEPLOYMENT
# =========================================================

import os

port = int(os.environ.get("PORT", 7860))

demo.launch(
    server_name="0.0.0.0",
    server_port=port
)
