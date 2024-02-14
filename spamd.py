import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

# Load the SMS Spam Collection dataset
url = 'C:\\Users\\MADIHA HAFSA\\OneDrive\\Documents\\7th sem\\MINI_PROJECT\\smsspamcollection\\SMSSpamCollection'
df = pd.read_csv(url, sep='\t', names=['label', 'message'])

# Map labels to binary values (0 for ham, 1 for spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Split the dataset into training and test sets
X = df['message'].values
y = df['label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize the text
tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Pad sequences
max_length = 100
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')

# Build the neural network model
model = Sequential([
    Embedding(input_dim=5000, output_dim=16, input_length=max_length),
    Flatten(),
    Dense(24, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_pad, y_train, epochs=5, batch_size=32, validation_split=0.1)

loss, accuracy = model.evaluate(X_test_pad, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

# Function to predict spam or ham for a given message
def predict_spam_or_ham(message):
    # Preprocess the input message
    sequence = tokenizer.texts_to_sequences([message])
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post', truncating='post')
    
    # Predict the class (spam or ham)
    prediction = model.predict(padded_sequence)[0][0]
    if prediction >= 0.5:
        return 'spam'
    else:
        return 'ham'

# Customizable input for user
while True:
    user_input = input("Enter a message (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    else:
        prediction = predict_spam_or_ham(user_input)
        print(f"The message is predicted as: {prediction}\n")

print("Thank you for using the spam detection system!")
