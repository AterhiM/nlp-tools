import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Load the model
model = SentenceTransformer("../similarity_search/models/")

# Streamlit app
st.title('Document Similarity App')

# Upload first file
uploaded_file_1 = st.file_uploader("Choose the first Excel file (SIC sections)", type="xlsx")
if uploaded_file_1:
    # Load the first dataset
    df1 = pd.read_excel(uploaded_file_1)
    st.write("First dataset (SIC sections):")
    st.write(df1)

# Upload second file
uploaded_file_2 = st.file_uploader("Choose the second Excel file (Company descriptions)", type="xlsx")
if uploaded_file_2:
    # Load the second dataset
    df2 = pd.read_excel(uploaded_file_2)
    st.write("Second dataset (Company descriptions):")
    st.write(df2)

if uploaded_file_1 and uploaded_file_2:
    # Extract descriptions and classes
    sic_sections = df1['class'].tolist()
    sic_descriptions = df1['description'].tolist()
    company_classes = df2['class'].tolist()
    company_comment = df2["Comments"].fillna("").tolist()
    company_label = df2["Is Section C ?"].tolist()
    company_descriptions = df2['description'].tolist()

    # Compute embeddings for SIC descriptions
    sic_embeddings = model.encode(sic_descriptions)

    # Compute embeddings for company descriptions
    company_embeddings = model.encode(company_descriptions)

    # Compute cosine similarity
    similarity_matrix = (1 + cosine_similarity(company_embeddings, sic_embeddings)) / 2

    # Prepare a DataFrame to show the results
    results = []
    for i, company_class in enumerate(company_classes):
        max_similarity_idx = similarity_matrix[i].argmax()
        results.append({
            'Company Class': company_class,
            'Company Description': company_descriptions[i],
            'Company Label': company_label[i],
            "Comment": company_comment[i],
            'Most Similar SIC Section': sic_sections[max_similarity_idx],
            'Most Similar SIC Description': sic_descriptions[max_similarity_idx],
            'Similarity Score': similarity_matrix[i, max_similarity_idx]
        })

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Sort the DataFrame by similarity score in descending order
    df_results = df_results.sort_values(by='Similarity Score', ascending=False)

    # Display the DataFrame
    st.write("Similarity Scores:")
    st.dataframe(df_results)

    # Option to download results as a CSV file
    st.download_button(
        label="Download results as CSV",
        data=df_results.to_csv(index=False),
        file_name='similarity_scores.csv',
        mime='text/csv'
    )
