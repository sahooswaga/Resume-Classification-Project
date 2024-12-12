# Libraries imported at the top 
import streamlit as st
import os
import re
import pandas as pd
import textract
import streamlit as st
from pathlib import Path
from docx import Document
from PyPDF2 import PdfReader
import re  # Import regex for pattern matching
import seaborn as sns
sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from sklearn.preprocessing import normalize
import string
import numpy as np

import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud, STOPWORDS

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


from sklearn.ensemble import RandomForestClassifier


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

from sklearn.metrics.pairwise import cosine_similarity

# Example skills list (replace with your actual skills list)
skills_list = ['python', 'sql', 'workday', 'admin', 'Peoplesoft','People soft','reactjs','react']


##################-------------------------Functions made--------------------##############################
# Function to extract text from a document using textract, with fallbacks for .docx and .pdf files
def extract_text(file_path):
    try:
        # Attempt to extract text using textract
        text = textract.process(file_path).decode("utf-8")
        return text
    except Exception as e:
        print(f"Error extracting text from {file_path} with textract: {e}")
        
        # If it's a .docx file, try using python-docx as a fallback
        if file_path.lower().endswith('.docx'):
            try:
                doc = Document(file_path)
                text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                return text
            except Exception as docx_error:
                print(f"Error extracting text from {file_path} with python-docx: {docx_error}")
        
        # If it's a .pdf file, try using PyPDF2 as a fallback
        if file_path.lower().endswith('.pdf'):
            try:
                pdf_text = []
                with open(file_path, "rb") as pdf_file:
                    reader = PdfReader(pdf_file)
                    for page in reader.pages:
                        pdf_text.append(page.extract_text())
                return "\n".join(pdf_text)
            except Exception as pdf_error:
                print(f"Error extracting text from {file_path} with PyPDF2: {pdf_error}")
        
        # Return empty string if extraction fails
        return ""

# Function to extract email from text
def extract_email(text):
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else ""

# Function to extract phone number from text
def extract_phone_number(text):
    phone_pattern = r'(\+?\d{1,4}[\s-]?)?(\(?\d{3}\)?[\s.-]?)?\d{3}[\s.-]?\d{4}'
    match = re.search(phone_pattern, text)
    return match.group(0) if match else ""

# Function to extract total experience from text (heuristic approach)
def extract_experience(text):
    # Split the text into lines and take the first 4-5 lines
    lines = text.split('\n')
    first_lines = "\n".join(lines[:50])  # Take the first 5 lines (adjust if you want only 4)
    # last_lines = "\n".join(lines[50:])  # Take the first 5 lines (adjust if you want only 4)
    
    # Regex to match patterns like "6 years", "3.5 years", "12 months", etc.
    experience_pattern = r'(?:more than|over)?\s*(\d+(\.\d+)?\+?)\s*(years|year|yr|yrs|months|month|mo|mos)'
    matches = re.findall(experience_pattern, first_lines, re.IGNORECASE)
    
    # Aggregate total years of experience if multiple matches are found
    total_years = 0
    for match in matches:
        value = match[0].replace('+', '').strip()  # Remove '+' and any extra spaces
        try:
            value = float(value)  # Convert to a float
        except ValueError:
            continue  # Skip invalid values if they can't be converted
        unit = match[2].lower()  # The unit part (e.g., "years", "months")

        # Convert months to years if needed
        if 'month' in unit or 'mo' in unit:
            value /= 12  # Convert months to years
        
        total_years += value

    return f"{total_years:.1f} years" if total_years > 0 else " 3 years"

def extract_skills(normalized_text):

    extracted_skills = []
    for skill in skills_list:
        if skill.lower() in normalized_text: #case insensitive matching
            extracted_skills.append(skill)
    return extracted_skills


# Function to gather file data, extract text, and additional information
def collect_files_and_extract_text(folder_path):
    data = []  # List to hold file data for CSV

    # Walk through the folder to find files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Skip temporary files that start with ~$
            if file.startswith('~$'):
                continue
            
            file_path = os.path.join(root, file)
            folder_name = Path(file_path).parent.name  # Last folder name in path (for 'Category' column)

            # Extract text from the document
            document_text = extract_text(file_path)

            # Extract email, phone number, and experience
            email = extract_email(document_text)
            phone_number = extract_phone_number(document_text)
            total_experience = extract_experience(document_text)
            skill = extract_skills(document_text)

            # Add row data to the list (file name, path, category, document text, email, phone, experience)
            data.append({
                "File Path": file_path,
                "Candidate Resume": file,
                "Category": folder_name,  # Assuming the category is the last folder in the path
                "Document Data Extracted": document_text,  # Adding extracted text as a new column
                "Email": email,
                "Phone Number": phone_number,
                "Total Experience": total_experience,
                "Skills": skill
            })

    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)

    # Remove rows where 'Category' is 'Resume_Docx' (if needed)
    df = df[df['Category'] != 'Resume_Docx']

    # Save the DataFrame to CSV (or return for further use)
    df.to_csv('document_data.csv', index=False)
    return df

# Function to gather file data, extract text, and additional information
def extract_text_from_file(uploaded_file):
    try:
        # Use textract to extract text based on file type
       
        text = textract.process(io.BytesIO(bytes_data)).decode('utf-8')
        return text
    except Exception as e:
        # Handle exceptions during text extraction
        print(f"Error extracting text: {e}")
        return None


####---------------------------------------------EDA Function -----------------------##########
def perform_eda(df):
    st.write("Data Shape:", df.shape)
    st.write("Data Types:", df.dtypes)
    st.write("Descriptive Statistics:", df.describe())

    # Example visualizations
    st.subheader("Here is the category counts provided for the data set")
    st.write(df['Category'].value_counts())
    #Word Count Analysis
    df['Word Count'] = df['Document Data Extracted'].dropna().apply(lambda x: len(x.split()))
    print("\nWord Count Statistics:")
    print(df['Word Count'].describe())
    
    plt.figure(figsize=(10, 6), facecolor=None)
    sns.histplot(df['Word Count'], kde=True, bins=30, color='green')
    plt.title("Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)
    
def normalize_text(text):
    stop_words = set(stopwords.words('english'))
    # Check if the input is a string
    if isinstance(text, str):
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        text = "".join([char for char in text if char not in string.punctuation])
        
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        
        # Stop word removal
        stop_words = set(stopwords.words('english'))
        words = [word for word in words if word not in stop_words]
        
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text = " ".join(lemmatized_words)
        return text
    else:
        return str(text)   

def remove_emails_numbers_links(text):
    """Removes email addresses, numbers, and links from a given text.

    Args:
        text: The input text string.

    Returns:
        The cleaned text string with emails, numbers, and links removed.
    """
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove URLs (http/https/www)
    text = re.sub(r'http\S+|https\S+|www\S+', '', text)

    return text

def remove_unnecessary_words(text):
    unnecessary_words = ["Ltd", "inbound", "outbound","project","using","etc","group","Hyderabad","summary","related","good","knowledge","ltd","experience","involved","client","pvt","creation","role","employee","integration","system","data","application","worked","customer","position","activity","skill","creating","various","description","working","worked"]  # Replace with your list
    words = text.split()
    cleaned_words = [word for word in words if word not in unnecessary_words]
    return " ".join(cleaned_words)
#--------------------------------------#-------------------Adding Swagatika Code of Naive Bayes Model-------------------------#
def Naive_fit():
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  
    st.write("Here is the naive bayes model completed by Swagatika")
    X = vectorizer.fit_transform(df['Cleaned Data'])  
    y = df['Category']  
    print("Shape of TF-IDF Features:", st.write(X.shape))
    #Splitting the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    st.write("Training Set Size:", X_train.shape)
    st.write("Testing Set Size:", X_test.shape)
    
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    
    y_pred = nb_model.predict(X_test)
    
    # Metrics
    st.write("\nClassification Report:")
    st.write(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)
    # Accuracy Score
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"\nAccuracy Score: {accuracy:.2f}")

    #Visualizing the Confusion Matrix
    plt.figure(figsize=(10, 6), facecolor=None)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

    from sklearn.model_selection import cross_val_score
    
    cv_scores = cross_val_score(nb_model, X, y, cv=5, scoring='accuracy')
    st.write("Cross-Validation Accuracy Scores:", cv_scores)
    print("Mean CV Accuracy:", cv_scores.mean())
    
    '''Cross-Validation Accuracy Scores: [1.         1.         1.         1.         0.86666667]
    Mean CV Accuracy: 0.9733333333333334
    The cross-validation results indicate that our model is performing very well,
     with a mean accuracy of 97.33%. However, the slight drop in one of the folds (86.67% accuracy)
     suggests that our model might be overfitting to some folds.'''
     
     #Analyzing Class Distribution 
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # for train_index, test_index in skf.split(X, int(y)):
    #     st.write("Train Class Distribution:", np.bincount(y.iloc[train_index]))
    #     st.write("Test Class Distribution:", np.bincount(y.iloc[test_index]))
    
    #Feature Engineering
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2), stop_words='english')
    
    #Evaluating on a Holdout Set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_val, X_holdout, y_val, y_holdout = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    nb_model.fit(X_train, y_train)
    y_val_pred = nb_model.predict(X_val)
    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=nb_model.classes_)
    plt.figure(figsize=(10,6), facecolor=None)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=nb_model.classes_, yticklabels=nb_model.classes_)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)

def random_forest_fit():

    
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')  
    st.write("Here is the Knn bayes model completed by Swagatika")
    X = vectorizer.fit_transform(df['Cleaned Data'])  
    y = df['Category']  
    print("Shape of TF-IDF Features:", st.write(X.shape))
    #Splitting the Data into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    '''--------------------------------Random Forest Model--------------------------------------'''
    
    from sklearn.ensemble import RandomForestClassifier
    
    #Initializing the Random Forest Model
    rf = RandomForestClassifier(random_state=42)
    
    #Defining the Hyperparameter Grid
    param_grid = {
    'n_estimators': [100, 200],  # Number of trees in the forest
    'max_depth': [10, 20],  # Maximum depth of the trees
    'min_samples_split': [2, 5],  # Minimum samples required to split a node
    'min_samples_leaf': [1, 2],    # Minimum samples required at each leaf node
    'criterion': ['gini']  # Splitting criterion
    }
    
    #Performing GridSearchCV for Hyperparameter Tuning
    grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', verbose=1, n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    
    #Retrieving Best Model
    best_rf_model = grid_search_rf.best_estimator_
    st.write("\nBest Hyperparameters for Random Forest:")
    st.write(grid_search_rf.best_params_)
    
    #Model Evaluation on Test Set
    y_pred_rf = best_rf_model.predict(X_test)
    
    #Accuracy
    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    st.write(f"\nRandom Forest Test Accuracy: {rf_accuracy:.2f}")
    
    #Classification Report
    st.write("\nClassification Report - Random Forest:")
    st.write(classification_report(y_test, y_pred_rf))
    
    #Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    plt.figure(figsize=(10,6), facecolor=None)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=best_rf_model.classes_, yticklabels=best_rf_model.classes_)
    plt.title("Confusion Matrix - Random Forest")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    st.pyplot(plt)
    #Cross-Validation for Generalization
    from sklearn.model_selection import cross_val_score
    #Cross-Validation to Evaluate Generalization
    cv_scores_rf = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='accuracy')
    st.write("\nCross-Validation Accuracy Scores for Random Forest:")
    st.write(cv_scores_rf)
    st.write(f"Mean CV Accuracy: {cv_scores_rf.mean():.2f}")
    
    #Extracting Feature Names from the Vectorizer
    feature_names = vectorizer.get_feature_names_out()
    
    #Getting Feature Importances from the Model
    feature_importances = best_rf_model.feature_importances_
    
    #Creating a DataFrame for Visualization
    features_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    #Plotting the Top 20 Features
    plt.figure(figsize=(12,8), facecolor=None)
    sns.barplot(x='Importance', y='Feature', data=features_df.head(20))
    plt.title("Top 20 Feature Importances - Random Forest")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    st.pyplot(plt)

def log_reg_and_xgboost_fit():
    st.write("Here is the logistic regression and xgboost code")
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Remove the 'Peoplesoft' category (single-instance class)
    filtered_resumes = df[df['Category'] != 'Peoplesoft']
    
    # Extract features and labels
    X = filtered_resumes['Cleaned Data']
    y = filtered_resumes['Category']
    
    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Text vectorization using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    log_reg = LogisticRegression(max_iter=1000, random_state=42)
    log_reg.fit(X_train_tfidf, y_train)

    y_pred_log_reg = log_reg.predict(X_test_tfidf)
    y_pred_prob_log_reg = log_reg.predict_proba(X_test_tfidf)
    log_reg_acc = accuracy_score(y_test, y_pred_log_reg)
    # Train accuracy
    train_accuracy_log_reg = log_reg.score(X_train_tfidf, y_train)
    # Test accuracy
    test_accuracy_log_reg = log_reg.score(X_test_tfidf, y_test)
    st.write("\nLogistic Regression Accuracy:", log_reg_acc)
    st.write(f"Train Accuracy (Logistic Regression): {train_accuracy_log_reg:.2f}")
    st.write(f"Test Accuracy (Logistic Regression): {test_accuracy_log_reg:.2f}")
    # Classification report
    st.write("Logistic Regression Classification Report:")
    st.write(classification_report(y_test, y_pred_log_reg, target_names=label_encoder.classes_))
    conf_matrix_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    st.write("\nLogistic Regression Confusion Matrix:\n", conf_matrix_log_reg)

    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Compute the confusion matrix
    cm_log_reg = confusion_matrix(y_test, y_pred_log_reg)
    
    # Plot the confusion matrix
    plt.figure(figsize=(12,8), facecolor=None)
    sns.heatmap(cm_log_reg, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Logistic Regression')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(plt)
    # ROC Curve and AUC
    fpr_log_reg, tpr_log_reg, _ = roc_curve(y_test, y_pred_prob_log_reg[:, 1], pos_label=1)
    roc_auc_log_reg = auc(fpr_log_reg, tpr_log_reg)

    # Plot ROC Curve
    plt.figure(figsize=(12,8), facecolor=None)
    plt.plot(fpr_log_reg, tpr_log_reg, label=f'Logistic Regression (AUC = {roc_auc_log_reg:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve - Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    st.pyplot(plt)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train_tfidf, y_train)
    y_pred_xgb = xgb.predict(X_test_tfidf)
    y_pred_prob_xgb = xgb.predict_proba(X_test_tfidf)
    xgb_acc = accuracy_score(y_test, y_pred_xgb)
    # Train accuracy
    train_accuracy_xgb = xgb.score(X_train_tfidf, y_train)
    # Test accuracy
    test_accuracy_xgb = xgb.score(X_test_tfidf, y_test)
    st.write("\nXGBoost Accuracy:", xgb_acc)
    st.write(f"Test Accuracy (XGBoost): {test_accuracy_xgb:.2f}")
    st.write(f"Train Accuracy (XGBoost): {train_accuracy_xgb:.2f}")
    st.write("XGBoost Classification Report:")
    st.write(classification_report(y_test, y_pred_xgb, target_names=label_encoder.classes_))
    conf_matrix_xgb = confusion_matrix(y_test, y_pred_xgb)
    st.write("\nXGBoost Confusion Matrix:\n", conf_matrix_xgb)
    import seaborn as sns
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix
    
    # Compute the confusion matrix for XGBoost
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    
    # Plot the confusion matrix as a heatmap
    plt.figure(figsize=(12,8), facecolor=None)
    sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', 
                xticklabels=label_encoder.classes_, 
                yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - XGBoost')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    st.pyplot(plt)

    # ROC Curve and AUC
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_prob_xgb[:, 1], pos_label=1)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    # Plot ROC Curve
    plt.figure(figsize=(12,8), facecolor=None)
    plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.title('ROC Curve - XGBoost')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    st.pyplot(plt)


def dec_tree_fit():
    from sklearn.model_selection import train_test_split
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import confusion_matrix, accuracy_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import f1_score, classification_report, precision_score, recall_score
    
    st.write("Decision Tree code by Anushka")
    x = df['Cleaned Data'].values
    y = df['Category'].values
    x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=45, test_size=0.25,stratify=y)
    tfidf_vector = TfidfVectorizer(sublinear_tf=True,stop_words='english')
    x_train = tfidf_vector.fit_transform(x_train)
    x_test = tfidf_vector.transform(x_test)

    model_DT = DecisionTreeClassifier(criterion='gini')
    model_DT.fit(x_train, y_train)
    y_pred = model_DT.predict(x_test)
    accuracy_DT = accuracy_score(y_test, y_pred)
    print('Accuracy of training set : {:.2f}'.format(model_DT.score(x_train, y_train)))
    print('Accuracy of  test set    : {:.2f}'.format(model_DT.score(x_test, y_test)))
    print("Classification report for classifier %s:\n%s\n" % (model_DT,classification_report(y_test, y_pred)))
    nb_score = model_DT.score(x_test, y_test)
    nb_cm = confusion_matrix(y_test, y_pred)
    
    precision_DT = round(precision_score(y_test,y_pred,average = 'macro'),2)
    recall_DT= round(recall_score(y_test,y_pred, average = 'macro'),2)
    f1_DT = round(f1_score(y_test,y_pred, average = 'macro'),2)
    accuracy_DT = round(accuracy_score(y_test,y_pred),2)
    cm = confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(12,8), facecolor=None)
    plt.title('Curve - Decision Tree')
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    sns.heatmap(cm,annot=True,xticklabels=['PeopleSoft','React JS Developer','SQL Developer','Workday'],
            yticklabels=['Dog','Not Dog'])
    st.pyplot(plt)
    
def normalize_document(doc):
    if isinstance(doc, str):
        doc = doc.lower()
        doc = ''.join(char for char in doc if char not in string.punctuation)
        tokens = nltk.word_tokenize(doc)
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return ' '.join(tokens)
    else:
        return ''
def preprocess_texts(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) #remove punctuation
    tokens = nltk.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)
        
def multinb():    # Load your resume data (replace 'your_resume_data.csv' with your actual file)
    try:
        df = pd.read_csv('document_data.csv')
    except FileNotFoundError:
        print("Error: 'your_resume_data.csv' not found. Please provide the correct file path.")
        exit()

    # Preprocess the 'Resume_str' column
    df['Resume_str'] = df['Document Data Extracted'].astype(str) #Ensure column is string type
    df['Processed_Resume'] = df['Resume_str'].apply(preprocess_texts)
    

    
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['Processed_Resume'], df['Category'], test_size=0.2, random_state=42
    )
    
    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Adjust max_features if needed
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    # Train a model (Multinomial Naive Bayes)
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
        
    # Make predictions on the test set
    y_pred = model.predict(X_test_tfidf)

    resume_to_correct = "Tanna Sujatha"
    index_to_correct = df[df['Document Data Extracted'].str.contains(resume_to_correct, case=False)].index

    if not index_to_correct.empty:
        # Assuming there's only one matching resume, if not use a loop to correct multiple
        correct_index = index_to_correct[0]
        df.loc[correct_index, 'Category'] = 'Peoplesoft'  # Correct the category
    
        # Now, retrain the model with the corrected category
        X_train, X_test, y_train, y_test = train_test_split(
            df['Processed_Resume'], df['Category'], test_size=0.2, random_state=42
        )
        tfidf_vectorizer = TfidfVectorizer(max_features=5000)
        X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
        model = MultinomialNB()
        model.fit(X_train_tfidf, y_train)
    
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"Accuracy: {accuracy}")
        st.write(classification_report(y_test, y_pred))
        # Predict categories for all candidates in the DataFrame
        df['Predicted_Category'] = model.predict(tfidf_vectorizer.transform(df['Processed_Resume']))
        predicted_categories = df['Predicted_Category'].unique()
        for category in predicted_categories:
            category_table = df[df['Predicted_Category'] == category]
            st.write(f"\nTable for Category: {category}")
            st.table(category_table[['Candidate Resume', 'Category', 'Predicted_Category']])

    # ... (Rest of your code)
    else:
        print(f"Resume '{resume_to_correct}' not found in the DataFrame.")
    
    

    

    # Display the DataFrame with predicted categories
    # st.write(df[['Document Data Extracted', 'Candidate Resume', 'Predicted_Category']])

# Streamlit UI
# ----------------------- Page Title and icon ------------------
st.set_page_config(page_title="HR Resume AI", page_icon=":tada:", layout="wide")

# ----------------------- Greet Header ---------------------
with st.container():
    # --------------Header Section- ---------------
    st.title("Project Resume Classification") 
    

    # Folder path input
    st.subheader("Step 1: Please Select your Resumes Folder below for Creating a valid data for EDA Process")

    option = st.selectbox(
        'How would you like to load your resumes?',
        ('Load Folder', 'Choose a Resume'))
    
    if option == 'Load Folder':
        st.write('You selected Load Folder.')
        # Code to handle folder loading
        # Example:
        folder_path = st.text_input("Enter folder path")
        if st.button("Load Folder"):
          # Load data from the specified folder
            st.write(f"Loading data from {folder_path}")

            # Button to process and generate CSV
            # Check if folder exists
            if os.path.exists(folder_path):
            # Collect files and extract text
                df = collect_files_and_extract_text(folder_path)
                # Show the dataframe in Streamlit
                st.subheader("Extracted Data")
                st.dataframe(df.head(5))
                st.success("CSV file generated successfully!")
                # EDA Section
                try:
                    df = pd.read_csv(r"C:\Users\Lenovo\Dassde\Code\document_data.csv") # Replace with your data loading code 
                    df['Category'] = df['Category'].replace('Peoplesoft Resume', 'Peoplesoft')
                    st.dataframe(df)
                    st.write("Performing EDA on the data set and providing Word cloud for it")
                    perform_eda(df)
                    df['Cleaned Data'] = df['Document Data Extracted'].apply(normalize_text)
                    df['Cleaned Data'] = df['Cleaned Data'].apply(remove_emails_numbers_links)
                    df['Cleaned Data'] = df['Cleaned Data'].apply(remove_unnecessary_words)
                            
                    # Generating the word cloud
                    text = " ".join(df['Cleaned Data'].astype(str))
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                                    
                    # Displaying the word cloud
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis('off')
                    st.pyplot(plt)  # Display the wordcloud in Streamlit
                    
                    Naive_fit()
                    random_forest_fit()
                    log_reg_and_xgboost_fit()
                    dec_tree_fit()
                    multinb()
                
                except FileNotFoundError:
                    st.error("File not found. Please upload a file first or provide a valid file path.")
                    df = None                                   
        
    elif option == 'Choose a Resume':
        st.write('You selected Choose a Resume.')
        # Code to handle resume upload
        # Example:
        uploaded_file = st.text_input("Enter resume")
        st.write("Dummy path : C://Users/Lenovo/Dassde/Resumes_Classification/Resumes_Docx1/Peoplesoft Resume/Peoplesoft Admin_Varkala Vikas.docx")
        if uploaded_file is not None:        
            extracted_text = extract_text(uploaded_file)
            if extracted_text:
                st.write("Word Cloud for the selected CV:")
                #Further processing of the extracted text
                newtext = ""
                newtext = normalize_text(extracted_text)
                # st.write(newtext)
                # newtext = remove_emails_numbers_links(newtext)
                newtext = remove_unnecessary_words(newtext)            
                # Generating the word cloud
                
                wordcloud = WordCloud(width=800, height=400, background_color="white").generate(newtext)

                # Display the generated image:
                plt.figure(figsize=(10, 5), facecolor=None)
                plt.imshow(wordcloud)
                plt.axis("off")
                plt.tight_layout(pad=0)
                st.pyplot(plt)


