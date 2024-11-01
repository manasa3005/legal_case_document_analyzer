import pandas as pd
import re
import os
import google.generativeai as genai
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from streamlit.components.v1 import html
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=api)
model = genai.GenerativeModel('gemini-pro')

# Load SBERT model for sentence embeddings
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to preprocess the data
def preprocess(data):
    selected_columns = ['Title', 'timestamp','document']
    df = data[selected_columns]

    df["content"] = df['title'] + ' This case happend on: ' + df['timestamp'].fillna('') + ' ' + df['document'].fillna('') 

    def clean_text(text):
        text = text.replace('\n', ' ')
        text = re.sub(r'\(US\d{7,}\)', '', text)
        text = re.sub(r'\s+', ' ', text)
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)
        unique_sentences = list(dict.fromkeys(sentences))
        cleaned_text = ' '.join(unique_sentences)
        return cleaned_text

    df['content'] = df['content'].apply(clean_text)
    result_df = df[['Title', 'content']]
    return result_df

# Function to generate summaries, calculate similarity, and provide novelty assessment
def generate_combined_results_and_novelty(data, user_idea):
    user_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', user_idea)
    user_embeddings = sbert_model.encode(user_sentences, convert_to_tensor=True)
    user_idea_embedding = sbert_model.encode(user_idea, convert_to_tensor=True)

    combined_results = []

    for index, row in data.iterrows():
        case_title = row['Title']
        case_content = row['content']
        
        # Generate summary for the case
        summary_response = model.generate_content("Imagine yourself as a legal analyst and summarize the following case in a concise paragraph: " + case_content)
        summary = summary_response[0].text if summary_response else "No summary available"

        # Generate conclusion for the case
        conclusion_response = model.generate_content("As a legal analyst, provide a concise conclusion for the following case: " + case_content)
        conclusion = conclusion_response[0].text if conclusion_response else "No conclusion available"

        # Identify persons involved in the case
        persons_response = model.generate_content("Identify and list any persons involved in the following case: " + case_content)
        persons_involved = persons_response[0].text if persons_response else "No persons identified"

        # Calculate similarity score
        case_embedding = sbert_model.encode(case_content, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(user_idea_embedding, case_embedding).item()

        # Sentence-level comparison between user sentences and existing cases
        case_sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', case_content)
        case_embeddings = sbert_model.encode(case_sentences, convert_to_tensor=True)

        similar_sentences = []
        for i, user_embedding in enumerate(user_embeddings):
            cosine_scores = util.pytorch_cos_sim(user_embedding, case_embeddings)[0]
            if cosine_scores.max().item() > 0.60:
                similar_sentences.append(f'*{user_sentences[i]}')

        # Combine all results for this case
        combined_results.append({
            'Case Title': case_title,
            'Summary': summary,
            'Conclusion': conclusion,
            'Persons Involved': persons_involved,
            'Similarity Score': similarity_score,
            'Similar Sentences': '\n'.join(similar_sentences),
        })

    output_df = pd.DataFrame(combined_results)
    return output_df


# Streamlit app function
def main():
    st.title('Legal case Analyzer')

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    user_idea = st.text_area("Enter your case for comparison:")

    # Limit user idea to 250 words
    if user_idea:
        word_count = len(user_idea.split())
        if word_count > 250:
            st.error(f"Your idea exceeds the 250-word limit. It currently has {word_count} words.")
        else:
            if uploaded_file is not None:
                df = pd.read_excel(uploaded_file)
                preprocessed_df = preprocess(df)

                # Generate combined results and get novelty feedback
                combined_result_df, novelty_feedback = generate_combined_results_and_novelty(preprocessed_df, user_idea)

                #st.write("Generated Summaries, Similarity Scores, and Sentence Comparison:")
                st.write(combined_result_df)

                

                # Download the combined result as an Excel file
                result_file = combined_result_df.to_csv(index=False)
                st.download_button(label="Download Results", data=result_file, file_name="case_summary_similarity_comparison.csv", mime="text/csv")

if __name__ == "__main__":
    main()