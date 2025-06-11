import os
import joblib
import pandas as pd
import numpy as np
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
import logging
import pickle
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variables to store loaded models and preprocessors
best_model = None
scaler = None
cat_columns = None
num_features = None

# Questions and their expected answer types
QUESTIONS = [
    {"key": "gender", "question": "👤 Jinsingiz? (Gender)", "options": ["Male", "Female"]},
    {"key": "SeniorCitizen", "question": "👴 Keksa mijozsinmi? (Senior Citizen)", "options": ["Yes", "No"]},
    {"key": "Partner", "question": "💑 Sherigingiz bormi? (Partner)", "options": ["Yes", "No"]},
    {"key": "Dependents", "question": "👨‍👩‍👧‍👦 Qaramog'ingiz bormi? (Dependents)", "options": ["Yes", "No"]},
    {"key": "tenure", "question": "📅 Necha oy mijoz bo'lgansiz? (Tenure in months)", "options": None},
    {"key": "PhoneService", "question": "📞 Telefon xizmatidan foydalanasizmi? (Phone Service)", "options": ["Yes", "No"]},
    {"key": "MultipleLines", "question": "📱 Bir nechta telefon liniyasiga egasizmi? (Multiple Lines)", "options": ["Yes", "No", "No phone service"]},
    {"key": "InternetService", "question": "🌐 Internet xizmati turi? (Internet Service)", "options": ["DSL", "Fiber optic", "No"]},
    {"key": "OnlineSecurity", "question": "🔒 Onlayn xavfsizlik xizmatidan foydalanasizmi? (Online Security)", "options": ["Yes", "No", "No internet service"]},
    {"key": "OnlineBackup", "question": "💾 Onlayn zaxira nusxa xizmatidan foydalanasizmi? (Online Backup)", "options": ["Yes", "No", "No internet service"]},
    {"key": "DeviceProtection", "question": "🛡️ Qurilma himoyasi xizmatidan foydalanasizmi? (Device Protection)", "options": ["Yes", "No", "No internet service"]},
    {"key": "TechSupport", "question": "🔧 Texnik yordam xizmatidan foydalanasizmi? (Tech Support)", "options": ["Yes", "No", "No internet service"]},
    {"key": "StreamingTV", "question": "📺 TV oqim xizmatidan foydalanasizmi? (Streaming TV)", "options": ["Yes", "No", "No internet service"]},
    {"key": "StreamingMovies", "question": "🎬 Film oqim xizmatidan foydalanasizmi? (Streaming Movies)", "options": ["Yes", "No", "No internet service"]},
    {"key": "Contract", "question": "📄 Shartnoma turi? (Contract)", "options": ["Month-to-month", "One year", "Two year"]},
    {"key": "PaperlessBilling", "question": "📋 Qog'ozsiz hisob-kitobdan foydalanasizmi? (Paperless Billing)", "options": ["Yes", "No"]},
    {"key": "PaymentMethod", "question": "💳 To'lov usuli? (Payment Method)", "options": ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]},
    {"key": "MonthlyCharges", "question": "💰 Oylik to'lov miqdori? (Monthly Charges in $)", "options": None},
    {"key": "TotalCharges", "question": "💵 Jami to'lov miqdori? (Total Charges in $)", "options": None}
]

MAIN_MENU = InlineKeyboardMarkup([
    [InlineKeyboardButton("🔮 Bashorat qilish", callback_data="predict"),
     InlineKeyboardButton("ℹ️ Yordam", callback_data="help")],
    [InlineKeyboardButton("📊 Bot haqida", callback_data="about")]
])

RESULT_MENU = InlineKeyboardMarkup([
    [InlineKeyboardButton("🔮 Yana bashorat qilish", callback_data="predict"),
     InlineKeyboardButton("🏠 Bosh menyu", callback_data="main_menu")]
])

WELCOME_TEXT = (
    "🤖 <b>Mijoz Ketishi Bashorat Boti</b>\n\n"
    "Salom! Men mijozlarning ketish ehtimolini bashorat qila olaman.\n\n"
    "📋 <b>Mavjud buyruqlar:</b>\n"
    "/start - Botni ishga tushirish\n"
    "/predict - Mijoz ketishi bashoratini ko'rish\n"
    "/help - Yordam\n\n"
    "Boshlash uchun <b>Bashorat qilish</b> tugmasini bosing!"
)

HELP_TEXT = (
    "ℹ️ <b>Yordam</b>\n\n"
    "1. <b>Bashorat qilish</b> tugmasini bosing yoki /predict yuboring.\n"
    "2. 19 ta savolga javob bering (variantlar tugma sifatida chiqadi).\n"
    "3. Natijani oling va kerak bo'lsa, yana bashorat qiling.\n\n"
    "Bosh menyuga qaytish uchun <b>Bosh menyu</b> tugmasini bosing."
)

ABOUT_TEXT = (
    "📊 <b>Bot haqida</b>\n\n"
    "Bu bot Telco mijozlarining ketish ehtimolini bashorat qilish uchun yaratilgan.\n"
    "Model: Machine Learning (Logistic Regression yoki boshqalar)\n"
    "Loyiha: final_project\n\n"
    "Bosh menyuga qaytish uchun <b>Bosh menyu</b> tugmasini bosing."
)

def load_models():
    """Load all required model components"""
    global best_model, scaler, cat_columns, num_features
    
    try:
        # Try to load all components
        best_model = joblib.load('../models/best_churn_model.pkl')
        logger.info("✅ Model loaded successfully!")
        
        # Try to load scaler, but handle if it doesn't exist or isn't fitted
        try:
            scaler = joblib.load('../models/scaler.pkl')
            logger.info("✅ Scaler loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Scaler not found or invalid: {e}")
            scaler = None
        
        # Try to load feature lists
        try:
            cat_columns = joblib.load('../models/cat_columns.pkl')
            logger.info("✅ Categorical columns loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Categorical columns not found: {e}")
            cat_columns = None
        
        try:
            num_features = joblib.load('../models/num_features.pkl')
            logger.info("✅ Numerical features loaded successfully!")
        except Exception as e:
            logger.warning(f"⚠️ Numerical features not found: {e}")
            num_features = None
        
        return True
    except Exception as e:
        logger.error(f"❌ Error loading main model: {e}")
        return False

def predict_churn(customer_data):
    """
    Predict customer churn based on input data
    
    Args:
        customer_data (dict): Dictionary containing customer information
        
    Returns:
        tuple: (prediction_text, probability_percentage)
    """
    try:
        # Convert to DataFrame
        df = pd.DataFrame([customer_data])
        logger.info(f"Input data: {customer_data}")
        
        # Convert categorical 'Yes'/'No' to binary for SeniorCitizen if needed
        if 'SeniorCitizen' in df.columns:
            df['SeniorCitizen'] = df['SeniorCitizen'].map({'Yes': 1, 'No': 0})
        
        # Handle TotalCharges - convert to numeric and handle missing values
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
        
        # Handle MonthlyCharges - ensure it's numeric
        if 'MonthlyCharges' in df.columns:
            df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
            df['MonthlyCharges'] = df['MonthlyCharges'].fillna(0.0)
        
        # Handle tenure - ensure it's numeric
        if 'tenure' in df.columns:
            df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce')
            df['tenure'] = df['tenure'].fillna(0.0)
        
        # Define default numerical and categorical columns if not loaded
        if num_features is None:
            numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        else:
            numeric_cols = list(num_features) if hasattr(num_features, '__iter__') else num_features
        
        if cat_columns is None:
            categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                              'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                              'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                              'PaperlessBilling', 'PaymentMethod']
        else:
            categorical_cols = list(cat_columns) if hasattr(cat_columns, '__iter__') else cat_columns
        
        # Handle numeric features
        numeric_features = []
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].fillna(0.0)
                numeric_features.append(col)
        
        # Scale numeric features only if scaler is available and properly fitted
        if scaler is not None and numeric_features:
            try:
                # Check if scaler is fitted by trying to transform a sample
                test_data = np.array([[0.0] * len(numeric_features)])
                scaler.transform(test_data)  # This will raise an error if not fitted
                
                # If we get here, scaler is fitted, so use it
                df[numeric_features] = scaler.transform(df[numeric_features])
                logger.info("✅ Numeric features scaled successfully")
            except Exception as e:
                logger.warning(f"⚠️ Scaler not fitted or error in scaling: {e}")
                logger.info("📊 Using unscaled numeric features")
        else:
            logger.info("📊 No scaler available, using unscaled numeric features")
        
        # Handle categorical features with one-hot encoding
        categorical_features = []
        for col in categorical_cols:
            if col in df.columns:
                categorical_features.append(col)
        
        # Apply one-hot encoding for categorical columns
        if categorical_features:
            df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=False)
        else:
            df_encoded = df.copy()
        
        logger.info(f"Encoded features shape: {df_encoded.shape}")
        logger.info(f"Encoded features: {list(df_encoded.columns)}")
        
        # Get all possible feature names that the model expects
        expected_features = None
        if hasattr(best_model, 'feature_names_in_'):
            expected_features = best_model.feature_names_in_
        elif hasattr(best_model, 'coef_'):
            # For models like LogisticRegression, we might need to infer features
            logger.warning("⚠️ Model doesn't have feature_names_in_, using current features")
        
        if expected_features is not None:
            logger.info(f"Model expects {len(expected_features)} features")
            
            # Ensure all expected features are present
            for feature in expected_features:
                if feature not in df_encoded.columns:
                    df_encoded[feature] = 0
            
            # Select only the features the model expects and in the correct order
            df_encoded = df_encoded[expected_features]
            logger.info(f"Final feature set shape: {df_encoded.shape}")
        
        # Make prediction
        prediction = best_model.predict(df_encoded)[0]
        probability = best_model.predict_proba(df_encoded)[0]
        
        # Get churn probability (probability of class 1)
        churn_probability = probability[1] if len(probability) > 1 else probability[0]
        
        # Format result
        if prediction == 1:
            prediction_text = "🚨 Ketadi"
            emoji = "⚠️"
        else:
            prediction_text = "✅ Qoladi"
            emoji = "😊"
        
        probability_percentage = f"{churn_probability * 100:.1f}"
        
        logger.info(f"Prediction: {prediction_text}, Probability: {probability_percentage}%")
        
        return prediction_text, probability_percentage, emoji
        
    except Exception as e:
        logger.error(f"❌ Error in prediction: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return "❌ Xatolik", "0.0", "❌"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    if update.message:
        await update.message.reply_text(WELCOME_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")
    elif update.callback_query:
        await update.callback_query.edit_message_text(WELCOME_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")

async def predict_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /predict command and start the question sequence"""
    
    # Check if models are loaded
    if best_model is None:
        if not load_models():
            error_message = "❌ Model yuklashda xatolik. Iltimos, keyinroq urinib ko'ring."
            if update.message:
                await update.message.reply_text(error_message)
            elif update.callback_query:
                await update.callback_query.edit_message_text(error_message)
            return
    
    # Initialize user data
    context.user_data['current_question'] = 0
    context.user_data['answers'] = {}
    context.user_data['in_prediction'] = True
    
    # Ask first question
    await ask_question(update, context)

async def ask_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Ask the current question to the user"""
    current_q = context.user_data['current_question']
    
    if current_q >= len(QUESTIONS):
        # All questions answered, make prediction
        await make_prediction(update, context)
        return
    
    question_data = QUESTIONS[current_q]
    question_text = f"❓ <b>Savol {current_q + 1}/19</b>\n\n{question_data['question']}"
    options = question_data['options']
    
    if options:
        # Create inline buttons for options (max 3 per row)
        keyboard = []
        for i in range(0, len(options), 3):
            row = [InlineKeyboardButton(option, callback_data=f"answer_{option}") 
                   for option in options[i:i+3]]
            keyboard.append(row)
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(question_text, reply_markup=reply_markup, parse_mode="HTML")
        else:
            await update.message.reply_text(question_text, reply_markup=reply_markup, parse_mode="HTML")
    else:
        # Numeric input question
        question_text += "\n\n💡 <i>Raqam kiriting:</i>"
        if update.callback_query:
            await update.callback_query.edit_message_text(question_text, parse_mode="HTML")
        else:
            await update.message.reply_text(question_text, parse_mode="HTML")

async def handle_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle user's text answer to current question"""
    if not context.user_data.get('in_prediction', False):
        return

    current_q = context.user_data['current_question']
    answer = update.message.text.strip()

    if current_q >= len(QUESTIONS):
        return

    question_data = QUESTIONS[current_q]
    question_key = question_data['key']

    # Only handle numeric questions here
    if question_data['options'] is not None:
        await update.message.reply_text("❌ Iltimos, tugmalardan birini tanlang.")
        return
    else:
        # Validate numeric input
        try:
            float(answer)
        except ValueError:
            await update.message.reply_text("❌ Iltimos, to'g'ri raqam kiriting.")
            return

    # Store answer
    context.user_data['answers'][question_key] = answer
    context.user_data['current_question'] += 1

    # Progress indicator
    progress = f"✅ {current_q + 1}/19 savol javoblandi"

    if context.user_data['current_question'] < len(QUESTIONS):
        await update.message.reply_text(progress)
        await ask_question(update, context)
    else:
        await update.message.reply_text(f"{progress}\n\n🔄 Bashorat tayyorlanmoqda...")
        await make_prediction(update, context)

async def handle_inline_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle inline button answer"""
    query = update.callback_query
    await query.answer()
    
    # Handle menu callbacks
    if query.data in ["predict", "help", "about", "main_menu"]:
        await menu_callback(update, context)
        return
    
    # Handle answer callbacks
    if not query.data.startswith("answer_"):
        return
        
    if not context.user_data.get('in_prediction', False):
        return
    
    answer = query.data.replace("answer_", "")
    current_q = context.user_data['current_question']
    
    if current_q >= len(QUESTIONS):
        return
        
    question_data = QUESTIONS[current_q]
    question_key = question_data['key']

    # Store answer
    context.user_data['answers'][question_key] = answer
    context.user_data['current_question'] += 1

    # Progress indicator
    progress = f"✅ {current_q + 1}/19 savol javoblandi"

    if context.user_data['current_question'] < len(QUESTIONS):
        await query.edit_message_text(progress)
        await ask_question(update, context)
    else:
        await query.edit_message_text(f"{progress}\n\n🔄 Bashorat tayyorlanmoqda...")
        await make_prediction(update, context)

async def make_prediction(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Make the final prediction based on collected answers"""
    try:
        customer_data = context.user_data['answers']
        prediction_text, probability_percentage, emoji = predict_churn(customer_data)
        
        # Create detailed result message
        result_message = (
            f"🎯 <b>Bashorat natijasi:</b>\n\n"
            f"{emoji} <b>Mijoz holati:</b> {prediction_text}\n"
            f"📊 <b>Ketish ehtimoli:</b> {probability_percentage}%\n\n"
        )
        
        # Add interpretation
        prob_float = float(probability_percentage)
        if prob_float < 30:
            result_message += "💚 <b>Yaxshi!</b> Mijoz qolish ehtimoli yuqori."
        elif prob_float < 70:
            result_message += "⚠️ <b>Ehtiyot bo'ling!</b> O'rtacha xavf darajasi."
        else:
            result_message += "🚨 <b>Diqqat!</b> Mijozni yo'qotish xavfi yuqori."
        
        if update.message:
            await update.message.reply_text(result_message, reply_markup=RESULT_MENU, parse_mode="HTML")
        elif update.callback_query:
            await update.callback_query.edit_message_text(result_message, reply_markup=RESULT_MENU, parse_mode="HTML")
        
        # Clear user data
        context.user_data.clear()
        
    except Exception as e:
        logger.error(f"❌ Error making prediction: {e}")
        error_message = "❌ Bashorat qilishda xatolik yuz berdi. Iltimos, qaytadan urinib ko'ring."
        
        if update.message:
            await update.message.reply_text(error_message, reply_markup=MAIN_MENU)
        elif update.callback_query:
            await update.callback_query.edit_message_text(error_message, reply_markup=MAIN_MENU)
        
        context.user_data.clear()

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /cancel command"""
    context.user_data.clear()
    await update.message.reply_text(
        "❌ Bashorat jarayoni bekor qilindi.\n\nYangi bashorat uchun /predict buyrug'ini yuboring yoki tugmani bosing.", 
        reply_markup=MAIN_MENU
    )

async def menu_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle menu button callbacks"""
    query = update.callback_query
    data = query.data

    if data == "predict":
        context.user_data.clear()  # Clear any existing data
        await predict_command(update, context)
    elif data == "help":
        await help_command(update, context)
    elif data == "about":
        await about_command(update, context)
    elif data == "main_menu":
        context.user_data.clear()  # Clear any existing data
        await start(update, context)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    if update.message:
        await update.message.reply_text(HELP_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")
    elif update.callback_query:
        await update.callback_query.edit_message_text(HELP_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")

async def about_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /about command"""
    if update.message:
        await update.message.reply_text(ABOUT_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")
    elif update.callback_query:
        await update.callback_query.edit_message_text(ABOUT_TEXT, reply_markup=MAIN_MENU, parse_mode="HTML")

def main():
    """Main function to run the bot"""
    
    # Get bot token from environment variable or use the one provided
    TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    
    if not TOKEN:
        logger.error("❌ TELEGRAM_BOT_TOKEN environment variable not set!")
        return
    
    # Load models at startup
    if not load_models():
        logger.error("❌ Could not load main model. Bot will attempt to load models when needed.")
    
    # Create application
    application = Application.builder().token(TOKEN).build()
    
    # Add handlers in correct order
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("about", about_command))
    application.add_handler(CommandHandler("predict", predict_command))
    application.add_handler(CommandHandler("cancel", cancel))
    
    # Callback query handlers
    application.add_handler(CallbackQueryHandler(handle_inline_answer))
    
    # Message handlers (should be last)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_answer))
    
    # Start bot
    logger.info("🚀 Bot started successfully!")
    print("🤖 Telegram Customer Churn Prediction Bot is running...")
    print("📊 Ready to predict customer churn!")
    
    # Run the bot
    application.run_polling()

if __name__ == '__main__':
    main()