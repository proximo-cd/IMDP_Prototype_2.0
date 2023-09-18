import os
import asyncio
import streamlit as st
from sklearn.cluster import KMeans
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer.aio import DocumentAnalysisClient
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from azure.storage.blob import BlobServiceClient, generate_blob_sas
from datetime import datetime, timedelta
import re

# Set the environment variables
os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"] = "https://cog-pcm-pdf-sbx-eu-1.cognitiveservices.azure.com/"
os.environ["AZURE_FORM_RECOGNIZER_KEY"] = "e03cac118c80488ab9f8a0453d4d2ed8"

# Initialize BlobServiceClient
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=demostorageproximo;AccountKey=W6oLVC6X/SHfMb8S+9TGoZRRqNxQMpofPVcag3+L5J3vZUhO2EqRkQm7fx3lacUWc9tnGb1YIRMh+AStTE1VkQ==;EndpointSuffix=core.windows.net")

st.title('IMDP Document Analysis Prototype')

st.write("""
Upload a document to analyze and visualize its content.
""")

uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)

if uploaded_files:
    st.write("Files uploaded successfully!")

    async def analyze_layout_async(blob_data):
        # Access the environment variables
        endpoint = os.environ["AZURE_FORM_RECOGNIZER_ENDPOINT"]
        key = os.environ["AZURE_FORM_RECOGNIZER_KEY"]

        document_analysis_client = DocumentAnalysisClient(
            endpoint=endpoint, credential=AzureKeyCredential(key)
        )

        async with document_analysis_client:
            poller = await document_analysis_client.begin_analyze_document(
                "prebuilt-layout", document=blob_data
            )
            result = await poller.result()

        # Extracted text data
        text_data = " ".join([line.content for page in result.pages for line in page.lines])

        return text_data

    @st.cache_data
    def analyze_file(file_bytes):
        return asyncio.run(analyze_layout_async(file_bytes))

    # List to store file information
    file_info_list = []

    for uploaded_file in uploaded_files:
        # Analyze the uploaded file using Form Recognizer
        file_bytes = uploaded_file.read()
        text_data = analyze_file(file_bytes)

        # Preprocess the text data
        stop_words = set(open('stopwords.txt').read().split())
        tokenized_data = [re.findall(r'\b\w+\b', text.lower()) for text in text_data.split('.')]
        tokenized_data = [[word for word in text if word not in stop_words] for text in tokenized_data]

        # Upload file to Blob Storage
        container_name = "mydatacontainer"
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=uploaded_file.name)
        uploaded_file.seek(0)  # Reset file pointer to the beginning
        blob_client.upload_blob(uploaded_file.read(), overwrite=True)

        # Generate SAS token for the blob
        sas_token = generate_blob_sas(
            account_name="demostorageproximo",
            container_name=container_name,
            blob_name=uploaded_file.name,
            account_key="W6oLVC6X/SHfMb8S+9TGoZRRqNxQMpofPVcag3+L5J3vZUhO2EqRkQm7fx3lacUWc9tnGb1YIRMh+AStTE1VkQ==",
            permission="r",
            expiry=datetime.utcnow() + timedelta(hours=1)
        )

        # Generate blob URL with SAS token
        blob_url_with_sas = f"{blob_client.url}?{sas_token}"

        # Store file information
        file_info = {"file_name": uploaded_file.name, "text_data": text_data, "tokenized_data": tokenized_data, "blob_url": blob_url_with_sas}
        file_info_list.append(file_info)

    # Get all text data
    all_text_data = [file_info['text_data'] for file_info in file_info_list]

    # Step 1: Feature extraction using TF-IDF
    vectorizer = TfidfVectorizer(max_df=1.0, max_features=10000, min_df=1, stop_words='english')
    X = vectorizer.fit_transform(all_text_data)

    # Step 2: Topic Modeling using LDA
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(X)
    lda_topics = lda.transform(X)

    # Step 3: Clustering using KMeans
    @st.cache_data
    def apply_kmeans(_X):
        kmeans = KMeans(n_clusters=min(5, len(all_text_data)))  # Adjust the number of clusters based on your data
        kmeans.fit(_X)
        return kmeans.labels_

    cluster_labels = apply_kmeans(lda_topics)

    # Step 4: Assign cluster labels to each document (this label can be updated manually later)
    doc_type_labels = {
        0: "Resume",
        1: "Invoice",
        2: "Medical Report",
        3: "User Manual",
        4: "Research Paper",
        5: "News Article",
        6: "Legal Document",
        7: "Email",
        8: "Financial Report",
        9: "Ad/Marketing Material",
    }

    for i, file_info in enumerate(file_info_list):
        file_info['doc_type'] = doc_type_labels.get(cluster_labels[i], "Unknown")

    # Display a sidebar with file names
    st.sidebar.title("Files")
    file_options = [(i, file_info['file_name']) for i, file_info in enumerate(file_info_list)]
    selected_file_index, selected_file_name = st.sidebar.radio("Choose a file", file_options)

    # Get the selected file info
    selected_file_info = file_info_list[selected_file_index]

    # Display the details of the selected file
    st.write(f"File: {selected_file_info['file_name']}")
    st.write(f"Document Type: {selected_file_info['doc_type']}")

    # Display Word Cloud
    wordcloud = WordCloud(width=800, height=400, background_color ='white').generate(selected_file_info['text_data'])
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    st.write("Extracted Text:")
    st.write(selected_file_info['text_data'])

    # Display link to download the document from blob storage
    st.write(f"[Download file from Blob Storage]({selected_file_info['blob_url']})")
