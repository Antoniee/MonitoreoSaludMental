from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters, ConversationHandler
import joblib

# Cargar modelo, vectorizador y codificador
modelo = joblib.load("modelo_naivebayes.pkl")
vectorizador = joblib.load("vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Preguntas iniciales
preguntas_previas = [
    "How have you been feeling lately?",
    "Do you feel like you enjoy things as usual?",
    "Have you felt exhausted or hopeless recently?"
]

# Preguntas PHQ-9
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

# Reglas de severidad del PHQ-9
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

# Recomendaciones según nivel
def recomendaciones_por_nivel(nivel):
    return {
        "Minimal": "✅ Keep maintaining your well-being with healthy habits and social interaction.",
        "Mild": "🧘‍♂️ Try light physical activity, maintain routine, and stay connected to people you trust.",
        "Moderate": "📝 Consider journaling or talking to a counselor or therapist.",
        "Moderately Severe": "⚠️ It's advisable to contact a mental health professional or support line. Talking to someone you trust is essential.",
        "Severe": "🚨 Please seek help from a mental health service or contact a support organization immediately. You're not alone."
    }.get(nivel, "Take care.")

# Estados de conversación
INICIALES, PHQ9 = range(2)

# Diccionario de usuarios
usuarios = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    usuarios[user_id] = {
        "fase": "inicial",
        "respuestas": [],
        "indice": 0,
        "nivel": None,
        "phq_score": 0
    }
    await update.message.reply_text("🧠 Hi! I'm here to help check on your mental health.")
    await update.message.reply_text(preguntas_previas[0])
    return INICIALES

async def manejar_iniciales(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    usuario = usuarios[user_id]
    respuesta = update.message.text
    usuario["respuestas"].append(respuesta)
    usuario["indice"] += 1

    if usuario["indice"] < len(preguntas_previas):
        await update.message.reply_text(preguntas_previas[usuario["indice"]])
        return INICIALES
    else:
        texto_completo = " ".join(usuario["respuestas"])
        entrada_vect = vectorizador.transform([texto_completo])
        pred = modelo.predict(entrada_vect)
        nivel = label_encoder.inverse_transform(pred)[0]
        usuario["nivel"] = nivel

        await update.message.reply_text(f"🧠 Estimated depression level: *{nivel}*", parse_mode="Markdown")
        await update.message.reply_text(f"💡 Recommendation: {recomendaciones_por_nivel(nivel)}")

        if nivel in ["Moderate", "Moderately Severe", "Severe"]:
            await update.message.reply_text("\n📋 Now, let's continue with the full PHQ-9. Answer each with:\n0 - Not at all\n1 - Several days\n2 - More than half the days\n3 - Nearly every day")
            usuario["fase"] = "phq9"
            usuario["indice"] = 0
            return await siguiente_phq(update, context)
        else:
            await update.message.reply_text("✅ No need for PHQ-9 at this time. Take care!")
            usuarios.pop(user_id)
            return ConversationHandler.END

async def siguiente_phq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    usuario = usuarios[user_id]
    i = usuario["indice"]
    if i < len(phq9_questions):
        await update.message.reply_text(f"PHQ-{i+1}: {phq9_questions[i]}")
        return PHQ9
    else:
        puntaje = usuario["phq_score"]
        nivel_final = phq9_to_severity(puntaje)
        await update.message.reply_text(f"✅ PHQ-9 Total Score: {puntaje}")
        await update.message.reply_text(f"🧠 Final clinical level: *{nivel_final}*", parse_mode="Markdown")
        await update.message.reply_text(f"💡 Recommendation: {recomendaciones_por_nivel(nivel_final)}")
        usuarios.pop(user_id)
        return ConversationHandler.END

async def manejar_phq(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    usuario = usuarios[user_id]
    respuesta = update.message.text.strip()

    if respuesta not in ["0", "1", "2", "3"]:
        await update.message.reply_text("⚠️ Please respond with 0, 1, 2, or 3.")
        return PHQ9

    usuario["phq_score"] += int(respuesta)
    usuario["indice"] += 1
    return await siguiente_phq(update, context)

async def cancelar(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Conversation canceled.")
    usuarios.pop(update.effective_user.id, None)
    return ConversationHandler.END

# Configuración del bot
TOKEN = "Ingrese su Token Aquí"
app = ApplicationBuilder().token(TOKEN).build()

conv_handler = ConversationHandler(
    entry_points=[CommandHandler("start", start)],
    states={
        INICIALES: [MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_iniciales)],
        PHQ9: [MessageHandler(filters.TEXT & ~filters.COMMAND, manejar_phq)],
    },
    fallbacks=[CommandHandler("cancel", cancelar)],
)

app.add_handler(conv_handler)
app.run_polling()
