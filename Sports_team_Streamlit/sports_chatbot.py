import streamlit as st
import uuid
import boto3
from botocore.exceptions import ClientError
import time
from datetime import datetime, timedelta

# Constants
AGENT_ID = "A8BOBK3KW2"
AGENT_ALIAS_ID = "ULQOQRBUIW"
COST_PER_INPUT_TOKEN = 0.00025
COST_PER_OUTPUT_TOKEN = 0.00025

def apply_custom_css():
    """Apply custom CSS styling"""
    st.markdown("""
        <style>
        /* Main container styling */
        .block-container {
            padding: 2rem 1rem 1rem 1rem;
            max-width: 100%;
        }

        /* Column styling */
        [data-testid="column"] {
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Metric styling */
        [data-testid="stMetricValue"] {
            font-size: 15px;
        }

        div[data-testid="stMetricLabel"] > label {
            font-size: 12px;
        }

        .stMetric {
            padding: 1px;
            margin-bottom: 5px;
        }

        /* Header styling */
        h1 {
            font-size: 1.5rem !important;
            padding-bottom: 1rem !important;
            margin: 0 !important;
        }

        /* Button styling */
        .stButton button {
            width: 100%;
            margin: 0;
        }

        /* Chat container styling */
        .stChatFloatingInputContainer {
            padding-bottom: 1rem;
        }

        /* Dark mode adjustments */
        .stApp {
            background-color: #0E1117;
        }
        </style>
    """, unsafe_allow_html=True)

def count_tokens(text):
    """Rough approximation of token count"""
    return len(text.split())

def format_time(seconds):
    """Format seconds into a readable duration"""
    return str(timedelta(seconds=round(seconds)))

def init_state():
    """Initialize session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.citations = []
        st.session_state.trace = {}
        st.session_state.total_cost = 0
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.session_start_time = time.time()
        st.session_state.total_prompt_time = 0
        st.session_state.initialized = True

def invoke_agent(prompt):
    """Invoke Bedrock agent and return response"""
    start_time = time.time()
    try:
        client = boto3.session.Session().client(service_name="bedrock-agent-runtime")
        response = client.invoke_agent(
            agentId=AGENT_ID,
            agentAliasId=AGENT_ALIAS_ID,
            enableTrace=True,
            sessionId=st.session_state.session_id,
            inputText=prompt,
        )

        output_text = ""
        citations = []
        trace = {}

        for event in response.get("completion"):
            if "chunk" in event:
                chunk = event["chunk"]
                output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations = citations + chunk["attribution"]["citations"]

            if "trace" in event:
                for trace_type in ["preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        if trace_type not in trace:
                            trace[trace_type] = []
                        trace[trace_type].append(event["trace"]["trace"][trace_type])

        execution_time = time.time() - start_time
        st.session_state.total_prompt_time += execution_time

        return {
            "output_text": output_text,
            "citations": citations,
            "trace": trace,
            "execution_time": execution_time
        }

    except ClientError as e:
        st.error(f"Error invoking agent: {str(e)}")
        return None

def render_session_analytics(container):
    """Render session analytics in the given container"""
    session_duration = time.time() - st.session_state.session_start_time
    container.metric("Session Duration", format_time(session_duration))
    container.metric("Total Input Tokens", st.session_state.total_input_tokens)
    container.metric("Total Output Tokens", st.session_state.total_output_tokens)
    container.metric("Total Tokens", st.session_state.total_input_tokens + st.session_state.total_output_tokens)
    container.metric("Total Cost ($)", f"{st.session_state.total_cost:.4f}")
    container.metric("Total Processing Time", format_time(st.session_state.total_prompt_time))
    
    if container.button("Reset Session", use_container_width=True, key="reset_session"):
        st.session_state.clear()
        init_state()
        st.rerun()

def render_chat_interface(container):
    """Render main chat interface in the given container"""
    # Display chat messages
    for message in st.session_state.messages:
        with container.chat_message(message["role"]):
            container.markdown(message["content"])
    
    # Chat input
    if prompt := container.chat_input("Enter your message"):
        process_user_input(prompt, container)

def process_user_input(prompt, container):
    """Process user input and generate response"""
    input_tokens = count_tokens(prompt)
    input_cost = input_tokens * COST_PER_INPUT_TOKEN
    
    st.session_state.total_input_tokens += input_tokens
    st.session_state.total_cost += input_cost
    
    # Display user message
    with container.chat_message("user"):
        container.write(prompt)
    
    # Process assistant response
    with container.chat_message("assistant"):
        response_placeholder = container.empty()
        response_placeholder.markdown("Thinking...")
        
        response = invoke_agent(prompt)
        if response:
            process_agent_response(response, prompt, input_tokens, input_cost, response_placeholder)

def process_agent_response(response, prompt, input_tokens, input_cost, placeholder):
    """Process and display agent response"""
    output_text = response["output_text"]
    output_tokens = count_tokens(output_text)
    output_cost = output_tokens * COST_PER_OUTPUT_TOKEN
    
    st.session_state.total_output_tokens += output_tokens
    st.session_state.total_cost += output_cost
    
    # Add citations if present
    if response["citations"]:
        citation_text = "\n\n**Sources:**"
        for idx, citation in enumerate(response["citations"], 1):
            for ref in citation["retrievedReferences"]:
                citation_text += f"\n{idx}. {ref['location']['s3Location']['uri']}"
        output_text += citation_text
    
    # Display response
    placeholder.markdown(output_text)
    
    # Store messages
    store_messages(prompt, output_text, input_tokens, output_tokens, input_cost, output_cost, response)

def store_messages(prompt, output_text, input_tokens, output_tokens, input_cost, output_cost, response):
    """Store messages and metrics in session state"""
    st.session_state.messages.extend([
        {
            "role": "user",
            "content": prompt,
            "metrics": {
                "tokens": input_tokens,
                "cost": input_cost,
                "timestamp": datetime.now().strftime("%H:%M:%S")
            }
        },
        {
            "role": "assistant",
            "content": output_text,
            "metrics": {
                "tokens": output_tokens,
                "cost": output_cost,
                "execution_time": response["execution_time"],
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "trace": response["trace"]
            }
        }
    ])
    
    st.session_state.citations = response["citations"]
    st.session_state.trace = response["trace"]

def render_prompt_analytics(container):
    """Render prompt analytics in the given container"""
    for i in range(0, len(st.session_state.messages), 2):
        if i + 1 < len(st.session_state.messages):
            user_msg = st.session_state.messages[i]
            assistant_msg = st.session_state.messages[i + 1]
            
            with container.expander(f"Prompt {i//2 + 1} - {user_msg['metrics']['timestamp']}", expanded=True):
                col1, col2 = st.columns(2, gap="small")
                
                with col1:
                    st.metric("Input Tokens", user_msg['metrics']['tokens'])
                    st.metric("Processing Time", f"{assistant_msg['metrics']['execution_time']:.2f}s")
                
                with col2:
                    st.metric("Output Tokens", assistant_msg['metrics']['tokens'])
                    total_cost = user_msg['metrics']['cost'] + assistant_msg['metrics']['cost']
                    st.metric("Total Cost", f"${total_cost:.4f}")
                
                if st.button(f"View Raw Trace #{i//2 + 1}", use_container_width=True):
                    st.json(assistant_msg['metrics']['trace'])

def main():
    """Main application function"""
    # Apply custom CSS
    apply_custom_css()
    
    # Initialize session state
    init_state()

    # Create a container for the entire app
    with st.container():
        # Create columns with adjusted ratios
        left_sidebar, main_chat, right_analytics = st.columns(
            [0.7, 1.6, 0.7],  # Adjusted ratios
            gap="small"
        )
        
        # Render each section
        with left_sidebar:
            st.title("Session Analytics")  # Changed to st.title
            render_session_analytics(st)
        
        with main_chat:
            st.title("Sports Poem Chatbot") 
            render_chat_interface(st)
        
        with right_analytics:
            st.title("Prompt Analytics")
            render_prompt_analytics(st)

if __name__ == "__main__":
    main()




