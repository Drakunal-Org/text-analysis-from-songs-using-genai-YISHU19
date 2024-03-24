#it should be a streamlit application for analyzing text from songs using GenAI, you should be able to upload any song and the AI model should be able to answer questions based on that
import streamlit as st
from pydub import AudioSegment
import speech_recognition as sr
import os
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS
from io import BytesIO
import subprocess
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def mp3_to_text(mp3_file, directory=os.getcwd()):
    text = ""

    recognizer = sr.Recognizer()

    # Convert MP3 file to WAV format
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(os.path.join(directory, "temp.wav"), format="wav")

    # Read WAV file
    with sr.AudioFile(os.path.join(directory, "temp.wav")) as source:
        audio_data = recognizer.record(source)

    # If the 'temp.wav' file was not found, create it
    if not os.path.exists(os.path.join(directory, "temp.wav")):
        raise FileNotFoundError("The 'temp.wav' file is not found. Please check the working directory.")

    texts = recognizer.recognize_google(audio_data)

    text += f"Audio: {mp3_file}\nText: {texts}\n\n"

    if os.path.exists(os.path.join(directory, "temp.wav")):
        os.remove(os.path.join(directory, "temp.wav"))

    return text.strip()


#function for splitting text into chunks of data
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


#saving of the text in the form of embeddings in a vector database  file
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


#Prompt template for langchain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the text that you get, 
    make sure to provide all the details, 
    \n\nFor example,\nExample 1 -Diplay the text, 
    the python code would be print(text)
    \nExample 2 - What are the text about?, 
    the output would be the summary of the text
    If the answer is not in
    provided context just say, 
    "Answer is not available in the context", 
    don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=1)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


#response based on the user input and the value from vector database
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    # print(response)
    sound_file = BytesIO()
    tts = gTTS(response["output_text"], lang='en')
    tts.write_to_fp(sound_file)
    st.audio(sound_file)
    st.write("Reply: ", response["output_text"])




#frontend portion

def main():
    st.set_page_config("text 101!")
    st.header("🧙🏻What are the audio about!!🧙🏻")

    user_question = st.text_input("Ask anything related to the audio")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        song = st.file_uploader("Upload your mp3 songs", accept_multiple_files=True)

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                if song:
                    raw_text = mp3_to_text(song[0], directory=os.path.dirname(song[0].name))
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
                    

if __name__ == "__main__":
    main()