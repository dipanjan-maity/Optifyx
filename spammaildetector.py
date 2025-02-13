{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 6,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qbQNPLNDyhtO",
        "outputId": "a109059f-9489-4e07-977e-be3c9161f70f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n",
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Package punkt_tab is already up-to-date!\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Model Accuracy: 97.49%\n",
            "Classification Report:\n",
            "               precision    recall  f1-score   support\n",
            "\n",
            "           0       0.97      1.00      0.99       965\n",
            "           1       1.00      0.81      0.90       150\n",
            "\n",
            "    accuracy                           0.97      1115\n",
            "   macro avg       0.99      0.91      0.94      1115\n",
            "weighted avg       0.98      0.97      0.97      1115\n",
            "\n",
            "Enter an email message to check if it's Spam or Ham: yahoo you have the winner\n",
            "\n",
            "Prediction: Ham\n"
          ]
        }
      ],
      "source": [
        "import pandas as pd\n",
        "import re\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.tokenize import word_tokenize\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.naive_bayes import MultinomialNB\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "# Load dataset\n",
        "df = pd.read_csv(\"/content/spam.csv\", encoding=\"latin-1\")[[\"v1\", \"v2\"]]\n",
        "df = df.rename(columns={\"v1\": \"label\", \"v2\": \"message\"})\n",
        "\n",
        "# Encode labels (ham -> 0, spam -> 1)\n",
        "df[\"label\"] = df[\"label\"].map({\"ham\": 0, \"spam\": 1})\n",
        "\n",
        "# Download necessary NLTK resources\n",
        "nltk.download(\"stopwords\")\n",
        "nltk.download(\"punkt\")\n",
        "nltk.download(\"wordnet\")\n",
        "# Download the 'punkt_tab' resource\n",
        "nltk.download('punkt_tab') # This line is added to download the missing resource.\n",
        "\n",
        "# Initialize stopwords and lemmatizer\n",
        "stop_words = set(stopwords.words(\"english\"))\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "\n",
        "# Text preprocessing function\n",
        "def preprocess_text(text):\n",
        "    text = text.lower()  # Convert to lowercase\n",
        "    text = re.sub(r\"[^\\w\\s]\", \"\", text)  # Remove punctuation\n",
        "    words = word_tokenize(text)  # Tokenize words\n",
        "    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Remove stopwords & lemmatize\n",
        "    return \" \".join(words)\n",
        "\n",
        "# Apply preprocessing\n",
        "df[\"cleaned_message\"] = df[\"message\"].apply(preprocess_text)\n",
        "\n",
        "# Convert text into numerical features using TF-IDF\n",
        "vectorizer = TfidfVectorizer(max_features=5000)\n",
        "X = vectorizer.fit_transform(df[\"cleaned_message\"]).toarray()\n",
        "y = df[\"label\"]\n",
        "\n",
        "# Split dataset into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Train a Naive Bayes classifier\n",
        "model = MultinomialNB()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Make predictions\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "print(f\"Model Accuracy: {accuracy * 100:.2f}%\")\n",
        "print(\"Classification Report:\\n\", classification_report(y_test, y_pred))\n",
        "\n",
        "# Function to predict if an email is spam or ham\n",
        "def predict_spam():\n",
        "    user_email = input(\"Enter an email message to check if it's Spam or Ham: \")\n",
        "    email = preprocess_text(user_email)\n",
        "    email_vector = vectorizer.transform([email]).toarray()\n",
        "    prediction = model.predict(email_vector)\n",
        "    print(\"\\nPrediction:\", \"Spam\" if prediction[0] == 1 else \"Ham\")\n",
        "\n",
        "# Run user input test\n",
        "predict_spam()\n"
      ]
    }
  ]
}
