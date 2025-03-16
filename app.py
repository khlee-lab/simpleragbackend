import streamlit as st
import requests
from datetime import datetime
import os
import time
import json
import base64
from typing import Optional

# Configuration - set this to your backend URL
# Default is http://localhost:8000 but can be changed via environment variable
BACKEND_API_URL = os.environ.get("BACKEND_API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Contract AI Assistant",
    page_icon="📄",
    layout="wide"
)

# Add a configuration section at the top of the app
st.sidebar.title("Contract AI Assistant")

# Backend configuration expandable section
with st.sidebar.expander("⚙️ Backend Configuration"):
    current_backend = st.text_input("Backend API URL", 
                                  value=BACKEND_API_URL, 
                                  key="backend_url")
    
    if st.button("Save Configuration"):
        BACKEND_API_URL = current_backend
        st.success(f"Backend URL updated to: {BACKEND_API_URL}")

# Helper functions for API calls
def upload_pdf(file) -> dict:
    """Upload a PDF file to the backend."""
    try:
        files = {"file": (file.name, file, "application/pdf")}
        response = requests.post(f"{BACKEND_API_URL}/upload", files=files)
        return response.json()
    except Exception as e:
        st.error(f"Error uploading PDF: {str(e)}")
        return {"success": False, "message": str(e)}

def check_indexer_status() -> dict:
    """Check the status of the indexer."""
    try:
        response = requests.get(f"{BACKEND_API_URL}/indexer-status")
        return response.json()
    except Exception as e:
        st.error(f"Error checking indexer status: {str(e)}")
        return {"success": False, "message": str(e)}

def chat_with_pdf(prompt: str) -> dict:
    """Send a chat request to the backend."""
    try:
        response = requests.post(
            f"{BACKEND_API_URL}/chat",
            params={"prompt": prompt}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error sending chat request: {str(e)}")
        return {"success": False, "message": str(e)}

def get_pdf_content(query: str = "") -> dict:
    """Get PDF content from the backend."""
    try:
        response = requests.get(
            f"{BACKEND_API_URL}/pdf-content",
            params={"query": query}
        )
        return response.json()
    except Exception as e:
        st.error(f"Error getting PDF content: {str(e)}")
        return {"success": False, "message": str(e)}

# Initialize session state
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_status" not in st.session_state:
    st.session_state.current_status = None

# UI Layout
st.title("Contract AI Assistant")
st.markdown("""
Upload your contract documents (PDF) and chat with the AI to get insights and answers about the content.
""")

# Sidebar for file upload and status
with st.sidebar:
    st.header("Document Management")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file and (not st.session_state.uploaded_file or 
                         st.session_state.uploaded_file.name != uploaded_file.name):
        st.session_state.uploaded_file = uploaded_file
        with st.spinner("Uploading document..."):
            result = upload_pdf(uploaded_file)
            
            if result.get("success"):
                st.success(f"Document uploaded: {uploaded_file.name}")
                st.session_state.current_status = "indexing"
                # Reset chat history when new document is uploaded
                st.session_state.chat_history = []
            else:
                st.error(f"Failed to upload: {result.get('message')}")
    
    # Status checker
    st.subheader("Document Processing Status")
    if st.button("Check Status"):
        with st.spinner("Checking status..."):
            status_result = check_indexer_status()
            
            if status_result.get("success"):
                status = status_result.get("data", {}).get("indexing_status", "unknown")
                
                if status == "success":
                    st.session_state.current_status = "ready"
                    st.success("✅ Document is indexed and ready for chatting")
                elif status == "not_found":
                    st.warning("⚠️ No documents found. Please upload a document first.")
                    st.session_state.current_status = "not_found"
                elif status.startswith("error"):
                    st.error(f"❌ Error: {status}")
                    st.session_state.current_status = "error"
                else:
                    st.info(f"⏳ Processing: {status}")
                    st.session_state.current_status = "indexing"
            else:
                st.error(f"Failed to check status: {status_result.get('message')}")
    
    # Display current status
    if st.session_state.current_status:
        status_emoji = {
            "ready": "✅",
            "indexing": "⏳",
            "not_found": "⚠️",
            "error": "❌",
            None: "❓"
        }.get(st.session_state.current_status, "❓")
        
        st.info(f"Current status: {status_emoji} {st.session_state.current_status}")

# Main area for chat
st.header("Chat with your Document")

# Chat interface
if st.session_state.uploaded_file is None:
    st.info("Please upload a document first using the sidebar.")
else:
    # Display chat history
    chat_container = st.container()
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"**You**: {message['content']}")
            else:
                st.markdown(f"**AI**: {message['content']}")
                
                # Show sources if available
                if "sources" in message:
                    with st.expander("View Sources"):
                        for source in message["sources"]:
                            st.text(f"📄 {source.get('filename', 'Unknown file')}")
                
                st.divider()
    
    # Input for new message
    prompt = st.text_area("Ask a question about your document:", height=100)
    
    if st.button("Send", disabled=st.session_state.current_status != "ready"):
        if not prompt:
            st.warning("Please enter a question!")
        else:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Send request to backend
            with st.spinner("AI is thinking..."):
                chat_response = chat_with_pdf(prompt)
                
                if chat_response.get("success"):
                    response_data = chat_response.get("data", {})
                    answer = response_data.get("answer", "No answer provided.")
                    sources = response_data.get("sources", [])
                    
                    # Add AI response to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                    
                    # Force a rerun to update the chat display
                    st.experimental_rerun()
                else:
                    st.error(f"Error: {chat_response.get('message')}")
                    st.session_state.chat_history.append({
                        "role": "assistant", 
                        "content": f"Sorry, I encountered an error: {chat_response.get('message')}",
                    })
    
    # Document viewer section
    if st.session_state.current_status == "ready":
        with st.expander("View Document Content"):
            content_response = get_pdf_content("*")
            
            if content_response.get("success"):
                docs = content_response.get("data", {}).get("documents", [])
                
                if docs:
                    doc_selector = st.selectbox(
                        "Select document to view:",
                        [doc.get("filename", f"Document {i+1}") for i, doc in enumerate(docs)]
                    )
                    
                    selected_doc = next(
                        (doc for doc in docs if doc.get("filename") == doc_selector), 
                        docs[0]
                    )
                    
                    st.text_area(
                        "Document content:",
                        value=selected_doc.get("content", "No content available"),
                        height=400,
                        disabled=True
                    )
                else:
                    st.info("No document content available.")
            else:
                st.error(f"Error loading document content: {content_response.get('message')}")

# Footer
st.markdown("---")
st.caption("Contract AI Assistant © 2023")
