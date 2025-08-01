import os
import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda

# Load OpenAI key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit config
st.set_page_config(page_title="🎥 YouTube RAG Assistant")
st.title("🎬 YouTube Video Q&A")

def extract_video_id(url):
    import re
    import urllib.parse
    parsed = urllib.parse.urlparse(url)
    if parsed.hostname == "youtu.be":
        return parsed.path[1:]
    elif parsed.hostname in ["www.youtube.com", "youtube.com"]:
        if parsed.path == "/watch":
            from urllib.parse import parse_qs
            return parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/embed/"):
            return parsed.path.split("/")[2]
    return None

def fetch_transcript(video_id):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
        return " ".join(chunk["text"] for chunk in transcript)
    except (TranscriptsDisabled, NoTranscriptFound):
        return None
    except Exception as e:
        st.warning(f"Transcript API failed: {e}")
        return None

def build_chain(transcript_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.create_documents([transcript_text])
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(docs, embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant.
Answer the question using the context below.
If the answer isn't available in the context, say you don't know.

Context:
{context}

Question:
{question}
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnableParallel({
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough()
        })
        | prompt
        | ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)
        | StrOutputParser()
    )

    return chain

# UI
video_url = st.text_input("🔗 Enter YouTube video URL:")

if video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL")
    else:
        with st.spinner("🔍 Fetching transcript..."):
            transcript = fetch_transcript(video_id)

        if transcript:
            st.success("✅ Transcript loaded successfully!")
            qa_chain = build_chain(transcript)

            if "messages" not in st.session_state:
                st.session_state.messages = []

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            if question := st.chat_input("Ask anything about the video..."):
                st.chat_message("user").markdown(question)
                with st.chat_message("assistant"):
                    response = qa_chain.invoke(question)
                    st.markdown(response)

                st.session_state.messages.append({"role": "user", "content": question})
                st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            st.warning("⚠️ This video doesn't have captions. Transcription not supported on Streamlit Cloud.")
