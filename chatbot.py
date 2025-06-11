import joblib
import spacy
import re
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud


nlp = spacy.load("en_core_web_sm") # English Spacy


model_choice = input(
    "Which exam score prediction model do you want to use?\n" +
    "Type 'rf' for RandomForest, 'lr' for LinearRegression: "
).strip().lower()
if model_choice == "lr":
    print("You chose Linear Regression.")
    model = joblib.load("student_score_linearregression.pkl")
else:
    print("Using Random Forest (default).")
    model = joblib.load("student_score_randomforest.pkl")

impute_vals = joblib.load("impute_defaults.pkl") 

features = {
    "age": None,
    "gender": None,
    "study_hours_per_day": None,
    "social_media_hours": None,
    "netflix_hours": None,
    "part_time_job": None,
    "attendance_percentage": None,
    "sleep_hours": None,
    "diet_quality": None,
    "exercise_frequency": None,
    "parental_education_level": None,
    "internet_quality": None,
    "mental_health_rating": None,
    "extracurricular_participation": None,
}

all_text = []
sentiment_scores = []

custom_stopwords = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
    'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
    'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those',
    'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while',
    'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between',
    'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
    'under', 'again', 'further', 'then', 'once', 'here', 'there',
    'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very',
    's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
])

def parse_input(user_input):
    all_text.append(user_input)
    sentiment_scores.append(TextBlob(user_input).sentiment.polarity)

    doc = nlp(user_input.lower())

    #lemmatized tokens as string
    lemmatized_tokens = [token.lemma_ for token in doc]
    token_texts = [token.text for token in doc]

    # Extract features
    if "year" in lemmatized_tokens or "age" in lemmatized_tokens:
        for token in doc:
            if token.like_num:
                features["age"] = int(token.text)
                break

    if "study" in lemmatized_tokens:
        match = re.search(r"(\d+\.?\d*)\s*hour", user_input.lower())
        if match:
            features["study_hours_per_day"] = float(match.group(1))
    if "sleep" in lemmatized_tokens:
        match = re.search(r"(\d+\.?\d*)\s*hour", user_input.lower())
        if match:
            features["sleep_hours"] = float(match.group(1))
    if "social" in lemmatized_tokens:
        match = re.search(r"(\d+\.?\d*)\s*hour", user_input.lower())
        if match:
            features["social_media_hours"] = float(match.group(1))
    if "netflix" in lemmatized_tokens or "tv" in lemmatized_tokens:
        match = re.search(r"(\d+\.?\d*)\s*hour", user_input.lower())
        if match:
            features["netflix_hours"] = float(match.group(1))
    if "job" in lemmatized_tokens:
        features["part_time_job"] = "Yes" if "yes" in lemmatized_tokens else "No"
    if "percent" in lemmatized_tokens or "%" in user_input:
        match = re.search(r"(\d+\.?\d*)", user_input.lower())
        if match:
            features["attendance_percentage"] = float(match.group(1))
    if "diet" in lemmatized_tokens:
        for quality in ["poor", "fair", "good"]:
            if quality in lemmatized_tokens:
                features["diet_quality"] = quality.capitalize()
    if "exercise" in lemmatized_tokens:
        match = re.search(r"(\d+)", user_input.lower())
        if match:
            features["exercise_frequency"] = int(match.group(1))
    if "parent" in lemmatized_tokens or "education" in lemmatized_tokens:
        for level in ["high school", "bachelor", "master", "none"]:
            if level in user_input.lower():
                features["parental_education_level"] = level.capitalize()
    if "internet" in lemmatized_tokens:
        for quality in ["poor", "average", "good"]:
            if quality in lemmatized_tokens:
                features["internet_quality"] = quality.capitalize()
    if "mental" in lemmatized_tokens or "stress" in lemmatized_tokens:
        match = re.search(r"(\d+)", user_input.lower())
        if match:
            features["mental_health_rating"] = int(match.group(1))
    if "extracurricular" in lemmatized_tokens or "activity" in lemmatized_tokens or "club" in lemmatized_tokens:
        features["extracurricular_participation"] = "Yes" if "yes" in lemmatized_tokens else "No"
    if "female" in lemmatized_tokens or "male" in lemmatized_tokens:
        features["gender"] = "Female" if "female" in lemmatized_tokens else "Male"

def get_prediction():
    input_vals = {}
    for f in features:
        val = features[f]
        if val is None:
            val = impute_vals[f]
        input_vals[f] = val
    input_df = pd.DataFrame([input_vals])
    pred = model.predict(input_df)[0]
    return max(0, min(pred, 100))

def summarize_conversation():
    if sentiment_scores:
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        print(f"Overall sentiment score: {avg_sentiment:.2f}")
        if avg_sentiment > 0.1:
            print("Your overall mood seems to be quite positive")
        elif avg_sentiment < -0.1:
            print("It seems like you're having a tough time. Take care")
        else:
            print("Your mood appears to be neutral.")

    text = ' '.join(all_text)
    doc = nlp(text.lower())
    lemmatized = ' '.join([token.lemma_ for token in doc if token.lemma_ not in custom_stopwords and token.is_alpha])
    wordcloud = WordCloud(width=600, height=400).generate(lemmatized)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud of Conversation (Lemmatized)")
    plt.show()

print("Welcome to the Student Exam Score Predictor Bot!")
print("You can start by telling me things like your age, study habits, sleep hours, etc.")
print("***Please type numbers (1,2,3) and follow with 'hours' afterwards***")
print("Type 'exit' to quit and get your score prediction and summary.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break

    parse_input(user_input)
    print("Bot: Thanks! Got that.")

    missing = [f for f, v in features.items() if v is None]
    if missing:
        print(f"Bot: Could you also tell me about: {', '.join(missing)}?")

missing = [f for f, v in features.items() if v is None]
if missing:
    print("\nBot: Note, you didn't provide these: " + ", ".join(missing) +
          ".\nThe prediction may be less accurate as a result.")

score = get_prediction()
print(f"\n Predicted Exam Score: {score:.2f}")

summarize_conversation()