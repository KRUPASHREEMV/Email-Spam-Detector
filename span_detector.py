import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.exceptions import NotFittedError

# Ensure necessary NLTK resources are downloaded
nltk.download('stopwords')
nltk.download('punkt')

# Initialize PorterStemmer
ps = PorterStemmer()

# Function to preprocess the input text
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize the text

    y = []
    for i in text:
        if i.isalnum():  # Only keep alphanumeric tokens
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:  # Remove stopwords and punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # Apply stemming

    return " ".join(y)

# Load the trained TF-IDF vectorizer and the spam detection model
try:
    tfidf = pickle.load(open('C:\\Users\\HP\\Desktop\\span_detector\\vectorizer.pkl', 'rb'))
    model = pickle.load(open('C:\\Users\\HP\\Desktop\\span_detector\\model.pkl', 'rb'))
except FileNotFoundError as e:
    st.error(f"File not found: {e}. Please make sure 'vectorizer.pkl' and 'model.pkl' exist.")
except Exception as e:
    st.error(f"An error occurred while loading model files: {e}")

# Streamlit app title
st.title("Email/SMS Spam Classifier")

# Input text area for the message
input_sms = st.text_area("Enter the message")

# Predict button
if st.button('Predict'):
    if input_sms:  # Check if any input is provided
        try:
            # 1. Preprocess the input text
            transformed_sms = transform_text(input_sms)
            # 2. Vectorize the text input
            vector_input = tfidf.transform([transformed_sms])
            # 3. Predict the result using the trained model
            result = model.predict(vector_input)[0]
            # 4. Display the prediction result
            if result == 1:
                st.header("Spam")
            else:
                st.header("Not Spam")
        except NotFittedError:
            st.error("The model is not fitted. Please ensure the model is trained with the training data before making predictions.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.error("Please enter a message to classify.")
