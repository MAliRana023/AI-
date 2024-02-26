import base64
import pickle
import streamlit as st
from wordcloud import WordCloud
import io
import PyPDF2
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Define the main function of your Streamlit app
def main():
    # Set the title of your Streamlit app
    st.title("Word Cloud Generator")

    # Add a selectbox to choose the input type
    input_type = st.selectbox("Select input type", ["Text", "PDF"])

    # Add a text input section
    if input_type == "Text":
        # Add a text input widget for users to enter their text
        user_input = st.text_area("Enter your text here:")

        # Add an option to remove stop words
        remove_stopwords_option = st.checkbox("Remove stop words")

        if remove_stopwords_option:
            # Print the table of stop words
            st.subheader("Table of Stop Words")
            stop_words_table = stopwords.words('english')
            st.write(stop_words_table)

            # Remove stop words from the user input
            user_input = remove_stopwords(user_input)

        # Add a button to generate the word cloud
        if st.button("Generate Word Cloud"):
            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(user_input)

            # Display the word cloud using Streamlit
            st.image(wordcloud.to_array(), use_column_width=True)

            # Add a download button for the word cloud image
            st.markdown(download_button(wordcloud.to_image(), "wordcloud.png", "Download Word Cloud Image"), unsafe_allow_html=True)

    # Add a PDF upload section
    if input_type == "PDF":
        # Add a file uploader for users to upload a PDF file
        pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

        if pdf_file is not None:
            # Read the PDF file
            pdf_text = read_pdf(pdf_file)

            # Add an option to remove stop words
            remove_stopwords_option = st.checkbox("Remove stop words")

            if remove_stopwords_option:
                # Print the table of stop words
                st.subheader("Table of Stop Words")
                stop_words_table = stopwords.words('english')
                st.write(stop_words_table)

                # Remove stop words from the PDF text
                pdf_text = remove_stopwords(pdf_text)

            # Generate the word cloud
            wordcloud = WordCloud(width=800, height=400, background_color="white").generate(pdf_text)

            # Display the word cloud using Streamlit
            st.image(wordcloud.to_array(), use_column_width=True)

            # Add a download button for the word cloud image
            st.markdown(download_button(wordcloud.to_image(), "wordcloud.png", "Download Word Cloud Image"), unsafe_allow_html=True)

# Function to read text from a PDF file
def read_pdf(pdf_file):
    # Create a PDF reader object
    pdf_reader = PyPDF2.PdfFileReader(pdf_file)

    # Initialize an empty string to store the text
    text = ""

    # Iterate through each page of the PDF
    for page_num in range(pdf_reader.numPages):
        # Extract text from the current page
        page = pdf_reader.getPage(page_num)
        text += page.extractText()

    # Close the PDF file
    pdf_file.close()

    return text

# Function to remove stop words from text
def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = nltk.word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

# Function to add a download button for downloading images
def download_button(object_to_download, download_filename, button_text):
    """
    Generates a link to download the given object_to_download.

    Parameters:
        object_to_download: The object to be downloaded.
        download_filename (str): The filename to save the object as when downloaded.
        button_text (str): The text to display on the download button.

    Examples:
        download_button(df, 'dataframe.csv', 'Download CSV File')
    """
    if isinstance(object_to_download, bytes):
        object_to_download = object_to_download.decode()

    # Convert the Image object to bytes
    img_byte_array = io.BytesIO()
    object_to_download.save(img_byte_array, format='PNG')
    img_bytes = img_byte_array.getvalue()

    # Generate the download link
    href = f'<a href="data:image/png;base64,{base64.b64encode(img_bytes).decode()}" download="{download_filename}">{button_text}</a>'
    return href

# Call the main function to run the Streamlit app
if __name__ == "__main__":
    main()
