import gradio as gr
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# -----------------------------
# Dataset
# -----------------------------

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


# -----------------------------
# Train AI Model
# -----------------------------

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


# -----------------------------
# Recommendation System
# -----------------------------

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
            "Focus more on internal assessments and revise your subjects regularly."
        )

    if predicted_marks < 50:
        recommendations.append(
            "Your predicted performance is low. Follow a regular study schedule."
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


# -----------------------------
# Prediction Function
# -----------------------------

def predict_student(
    student_name,
    study_hours,
    attendance,
    assignment_score,
    internal_marks
):

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


# -----------------------------
# Gradio Website
# -----------------------------

with gr.Blocks(
    title="AI Student Performance Prediction"
) as demo:

    gr.Markdown(
        "# 🎓 AI Student Performance Prediction"
    )

    gr.Markdown(
        "### Personalized Learning Recommendation System"
    )

    gr.Markdown(
        "Enter the student's academic details to predict final performance."
    )

    with gr.Row():

        with gr.Column():

            gr.Markdown("### 📝 Student Details")

            student_name = gr.Textbox(
                label="Student Name",
                placeholder="Enter student name"
            )

            study_hours = gr.Number(
                label="Study Hours",
                value=2
            )

            attendance = gr.Number(
                label="Attendance (%)",
                value=50
            )

            assignment_score = gr.Number(
                label="Assignment Score",
                value=55
            )

            internal_marks = gr.Number(
                label="Internal Marks",
                value=45
            )

            predict_button = gr.Button(
                "🔮 Predict Performance",
                variant="primary"
            )

        with gr.Column():

            gr.Markdown("### 🤖 AI Prediction")

            predicted_marks = gr.Textbox(
                label="Predicted Final Marks"
            )

            performance = gr.Textbox(
                label="Performance Level"
            )

            recommendations = gr.Textbox(
                label="💡 Personalized Recommendations",
                lines=8
            )

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


demo.launch()
