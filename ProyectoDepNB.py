# Paso 1: Librerías necesarias
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import numpy as np
import sys

# Paso 2: Cargar datos combinados
file_path = "Dataset_Limpio.csv"  # Ajusta la ruta si es necesario
df = pd.read_csv(file_path)

# Paso 3: Preprocesamiento
df = df.dropna(subset=["texto_completo", "Severity Level"])
X = df["texto_completo"]
y = df["Severity Level"]

# Paso 4: Codificación de etiquetas
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Paso 5: División de datos y vectorización
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
vectorizer = TfidfVectorizer(stop_words="english", max_features=1000)
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Paso 6: Entrenamiento del modelo Naive Bayes
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# Guardar el modelo y los objetos necesarios

import joblib

joblib.dump(model, "modelo_naivebayes.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(le, "label_encoder.pkl")
# Paso 7: Evaluación
print("\n📊 Reporte de clasificación:")
print(classification_report(y_test, model.predict(X_test_vect), target_names=le.classes_))

# Paso 8: Interacción con el usuario
preguntas_previas = [
    "How have you been feeling lately?",
    "Do you feel like you enjoy things as usual?",
    "Have you felt exhausted or hopeless recently?"
]

def phq9_to_severity(score):
    if score <= 4:
        return "Minimal"
    elif score <= 9:
        return "Mild"
    elif score <= 14:
        return "Moderate"
    elif score <= 19:
        return "Moderately Severe"
    else:
        return "Severe"

def recomendaciones_por_nivel(nivel):
    if nivel == "Minimal":
        return "✅ Keep maintaining your well-being with healthy habits and social interaction."
    elif nivel == "Mild":
        return "🧘‍♂️ Try light physical activity, maintain routine, and stay connected to people you trust."
    elif nivel == "Moderate":
        return "📝 Consider journaling or talking to a counselor or therapist."
    elif nivel == "Moderately Severe":
        return "⚠️ It's advisable to contact a mental health professional or support line. Talking to someone you trust is essential."
    elif nivel == "Severe":
        return "🚨 Please seek help from a mental health service or contact a support organization immediately. You're not alone."

phq9_questions = [
    "Little interest or pleasure in doing things",
    "Feeling down, depressed, or hopeless",
    "Trouble falling or staying asleep, or sleeping too much",
    "Feeling tired or having little energy",
    "Poor appetite or overeating",
    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
    "Trouble concentrating on things, such as reading the newspaper or watching television",
    "Moving or speaking so slowly that other people could have noticed? Or the opposite — being so fidgety or restless that you have been moving around a lot more than usual",
    "Thoughts that you would be better off dead or of hurting yourself in some way"
]

while True:
    print("\n🧠 Let's talk about how you're feeling. You can type 'exit' anytime.\n")

    respuestas = []
    for q in preguntas_previas:
        r = input(f"❓ {q}\n> ")
        if r.lower() == "exit":
            print("👋 Session ended.")
            sys.exit()
        respuestas.append(r)

    texto_usuario = " ".join(respuestas)
    entrada_vect = vectorizer.transform([texto_usuario])
    pred = model.predict(entrada_vect)
    nivel = le.inverse_transform(pred)[0]

    print(f"\n🧠 Estimated depression level from initial answers: **{nivel}**")
    print(f"💡 Recommendation: {recomendaciones_por_nivel(nivel)}")

    if nivel in ["Moderate", "Moderately Severe", "Severe"]:
        print("\n📋 Let's apply the full PHQ-9 to get a clearer picture:")
        print("Each question refers to how often you have been bothered by the following problems over the last 2 weeks.")
        print("Respond with:")
        print("  0 - Not at all")
        print("  1 - Several days")
        print("  2 - More than half the days")
        print("  3 - Nearly every day\n")

        puntaje_total = 0
        for i, question in enumerate(phq9_questions, 1):
            while True:
                print(f"PHQ-{i}: {question}")
                respuesta = input("Your answer (0-3): ")
                if respuesta in ["0", "1", "2", "3"]:
                    puntaje_total += int(respuesta)
                    break
                else:
                    print("⚠️ Invalid input. Please enter 0, 1, 2, or 3.")

        nivel_final = phq9_to_severity(puntaje_total)
        print(f"\n✅ PHQ-9 Score: {puntaje_total}")
        print(f"🧠 Final clinical level: **{nivel_final}**")
        print(f"💡 Recommendation: {recomendaciones_por_nivel(nivel_final)}")
    else:
        print("\n✅ Based on your answers, a full PHQ-9 may not be necessary right now.")

    continuar = input("\n🔁 Do you want to try again? (yes/no)\n> ")
    if continuar.lower() != "yes":
        print("👋 Thank you for using the depression screener.")
        break
