import streamlit as st
import ollama

# Streamlit app setup
st.title("Causal Inference Tutor")
st.subheader("Powered by Ollama")

# Sidebar for configuration
st.sidebar.header("Model Settings")
model_name = st.sidebar.text_input("Model Name", "causal-inference-tutor")
system_prompt = st.sidebar.text_area(
    "System Prompt",
    "You are my causal inference tutor. You are going to teach me about important concepts in causal inference based on my questions.",
)
stream_mode = st.sidebar.checkbox("Stream Responses", value=True)

# Input prompt from the user
user_prompt = st.text_area(
    "Enter your question:", placeholder="Ask anything about causal inference..."
)

if st.button("Get Answer"):
    # Create custom model if it doesn't exist
    with st.spinner("Setting up your custom model..."):
        ollama.create(
            model=model_name,
            from_="llama3.2:latest",
            system=system_prompt,
        )

    # Generate response
    with st.spinner("Generating response..."):
        response = ollama.generate(
            model=model_name,
            prompt=user_prompt,
            stream=stream_mode,
        )

        # Display response
        if stream_mode:
            st.text("Response:")
            response_text = ""
            for chunk in response:
                response_text += chunk["response"]
                st.text(response_text)
        else:
            st.text("Response:")
            st.write(response["response"])
