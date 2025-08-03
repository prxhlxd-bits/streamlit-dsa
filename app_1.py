import streamlit as st
import base64
from together import Together
from PIL import Image
import io
import time
import re
import subprocess
from contextlib import redirect_stdout

# Configure the page
st.set_page_config(
    page_title="DSA Question Solver",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling and mobile responsiveness
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        color: #495057;
        font-weight: 500;
        transition: background-color 0.3s, color 0.3s;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #e9ecef;
    }
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    .upload-container {
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9ff;
        margin: 1rem 0;
    }
    .question-container, .code-container, .output-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .code-editor-container {
        background-color: #1e1e1e;
        color: #d4d4d4;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-family: 'Courier New', Courier, monospace;
    }
    .step-header {
        background: linear-gradient(90deg, #007bff, #0056b3);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.1rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .test-case-pass {
        color: #28a745;
        font-weight: bold;
    }
    .test-case-fail {
        color: #dc3545;
        font-weight: bold;
    }
    .regeneration-section {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    .code-history-item {
        background-color: #f1f3f4;
        border-left: 3px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 5px;
    }
    @media (max-width: 768px) {
        .main { padding: 1rem; }
        .stTabs [data-baseweb="tab"] { padding-left: 10px; padding-right: 10px; font-size: 0.9rem; }
        .upload-container { padding: 1rem; }
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = ""
if 'formatted_question' not in st.session_state:
    st.session_state.formatted_question = ""
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""
if 'edited_code' not in st.session_state:
    st.session_state.edited_code = ""
if 'test_cases' not in st.session_state:
    st.session_state.test_cases = []
if 'run_output' not in st.session_state:
    st.session_state.run_output = ""
if 'code_history' not in st.session_state:
    st.session_state.code_history = []
if 'generation_count' not in st.session_state:
    st.session_state.generation_count = 0
if 'user_feedback' not in st.session_state:
    st.session_state.user_feedback = ""

# --- API and Helper Functions ---

def handle_api_error(e):
    st.error(f"An API error occurred: {str(e)}")
    if "authentication" in str(e).lower():
        st.warning("Please check if your Together AI API key is correct and has sufficient credits.")

def extract_text_from_images(images, api_key):
    """Extracts text from a list of images using Llama-Vision."""
    if not api_key:
        st.error("Please enter your Together AI API key in the sidebar.")
        return ""
    try:
        client = Together(api_key=api_key)
        prompt = "Extract all text from the image. Ignore UI elements like buttons or unrelated text."
        all_text = []
        for i, image in enumerate(images):
            with st.spinner(f"Analyzing image {i+1}/{len(images)}..."):
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                base64_image = base64.b64encode(img_bytes.getvalue()).decode('utf-8')
                
                response = client.chat.completions.create(
                    model="meta-llama/Llama-Vision-Free",
                    messages=[
                        {"role": "system", "content": "You are an expert OCR reader for coding platforms."},
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                        ]},
                    ]
                )
                all_text.append(response.choices[0].message.content)
        return "\n\n---\n\n".join(all_text)
    except Exception as e:
        handle_api_error(e)
        return ""

def format_question_with_llama(raw_text, api_key):
    """Formats the raw text into a structured question using Llama 3.3."""
    if not api_key:
        st.error("Please enter your Together AI API key in the sidebar.")
        return ""
    try:
        client = Together(api_key=api_key)
        prompt = f"""
            Based on the following text extracted from one or more images, please structure it into a clear and well-formatted coding problem.
            
            The output should strictly follow this format:
            ### Problem Description
            [Write the question relevant text as it is here. Do not change the wording also.]

            ### Tasks
            [List the specific tasks or requirements.]

            ### Examples
            [Provide at least one example with clear Input and Output.]
            Example 1:
            Input:
            [Sample Input]
            Output:
            [Sample Output]

            ### Explanation (Optional)
            [If provided in the source, add any explanation for the examples.]

            Here is the raw text:
            ---
            {raw_text}
            ---
        """
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        handle_api_error(e)
        return ""

def generate_solution(question, api_key, previous_attempts=None, error_feedback=None, user_comments=None):
    """Generates a code solution for the given question, optionally considering previous attempts."""
    if not api_key:
        st.error("Please enter your Together AI API key in the sidebar.")
        return ""
    try:
        client = Together(api_key=api_key)
        
        # Base prompt
        prompt = f"""
            Please provide a complete and runnable Python code solution for the following programming problem.
            The code should be self-contained and ready to execute.
            Do not include any explanations, analysis, or markdown formatting, only the raw Python code.

            Problem:
            {question}
        """
        
        # Add context from previous attempts if available
        if previous_attempts and error_feedback:
            prompt += f"""
            
            IMPORTANT: Previous attempts have failed. Please learn from these mistakes:
            
            Previous failed attempts:
            {previous_attempts}
            
            Error feedback from test runs:
            {error_feedback}
            """
            
            if user_comments:
                prompt += f"""
                
                Additional user feedback:
                {user_comments}
                """
                
            prompt += """
            
            Please generate a NEW and IMPROVED solution that addresses these issues. 
            Make sure to:
            1. Fix any logical errors from previous attempts
            2. Handle edge cases properly
            3. Follow the exact input/output format specified in the examples
            4. Test your logic mentally before providing the solution
            """
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Slightly higher for regeneration to get different approaches
        )
        
        # Extract only the code block
        code = response.choices[0].message.content
        code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return code.strip() # Fallback if no markdown block is found
    except Exception as e:
        handle_api_error(e)
        return ""

def extract_test_cases(formatted_question):
    """Extracts input/output pairs from the formatted question using regex."""
    test_cases = []
    # Regex to find examples, then extract Input and Output blocks
    example_blocks = re.findall(r'Example \d+:(.*?)($|Example \d+:|###)', formatted_question, re.DOTALL)
    
    for block, _ in example_blocks:
        input_match = re.search(r'Input:\s*\n(.*?)\n\s*Output:', block, re.DOTALL)
        output_match = re.search(r'Output:\s*\n(.*)', block, re.DOTALL)
        
        if input_match and output_match:
            test_cases.append({
                "input": input_match.group(1).strip(),
                "expected_output": output_match.group(1).strip()
            })
    return test_cases

def run_code(code_string, test_cases):
    """Executes the given code against a list of test cases and captures the output."""
    results = []
    for i, case in enumerate(test_cases):
        try:
            # Execute the code as a subprocess for safety
            process = subprocess.run(
                ['python', '-c', code_string],
                input=case['input'],
                text=True,
                capture_output=True,
                timeout=10  # 10-second timeout to prevent infinite loops
            )
            
            stdout = process.stdout.strip()
            stderr = process.stderr.strip()
            
            result = {"case": i + 1, "input": case['input']}
            if stderr:
                result["status"] = "Error"
                result["output"] = stderr
            else:
                result["output"] = stdout
                if stdout == case['expected_output']:
                    result["status"] = "Passed"
                else:
                    result["status"] = "Failed"
                    result["expected"] = case['expected_output']
            results.append(result)
            
        except subprocess.TimeoutExpired:
            results.append({"case": i + 1, "input": case['input'], "status": "Error", "output": "Execution timed out."})
        except Exception as e:
            results.append({"case": i + 1, "input": case['input'], "status": "Error", "output": f"An unexpected error occurred: {str(e)}"})
            
    return results

def get_failed_test_summary(run_results):
    """Creates a summary of failed tests for LLM feedback."""
    failed_summary = []
    for result in run_results:
        if result['status'] in ['Failed', 'Error']:
            summary = f"Test Case {result['case']}:\n"
            summary += f"Input: {result['input']}\n"
            summary += f"Your Output: {result['output']}\n"
            if 'expected' in result:
                summary += f"Expected Output: {result['expected']}\n"
            summary += f"Status: {result['status']}\n"
            failed_summary.append(summary)
    return "\n---\n".join(failed_summary)

def add_to_code_history(code, results, generation_num):
    """Adds code attempt and results to history."""
    history_item = {
        "generation": generation_num,
        "code": code,
        "results": results,
        "timestamp": time.strftime("%H:%M:%S")
    }
    st.session_state.code_history.append(history_item)

# --- UI Layout ---

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #007bff; margin-bottom: 0.5rem;">🧠 DSA Question Solver</h1>
    <p style="color: #6c757d; font-size: 1.1rem;">From Image to Executable Code</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔑 Configuration")
    api_key = st.text_input("Together AI API Key", type="password", value=st.session_state.api_key)
    if api_key:
        st.session_state.api_key = api_key
        if 'api_key_configured' not in st.session_state:
            st.success("API Key configured!")
            st.session_state.api_key_configured = True
    
    # Code generation history in sidebar
    if st.session_state.code_history:
        st.markdown("### 📚 Generation History")
        for i, item in enumerate(st.session_state.code_history):
            passed_tests = sum(1 for r in item['results'] if r['status'] == 'Passed')
            total_tests = len(item['results'])
            with st.expander(f"Gen {item['generation']} ({passed_tests}/{total_tests}) - {item['timestamp']}"):
                st.code(item['code'][:200] + "..." if len(item['code']) > 200 else item['code'], language="python")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["1. Upload Images", "2. Format Question", "3. Generate Code", "4. Run & Edit Code"])

with tab1:
    st.markdown('<div class="step-header">Step 1: Upload Question Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose up to 4 images of the DSA question", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    
    if uploaded_files:
        st.session_state.uploaded_images = [Image.open(file) for file in uploaded_files]
        st.markdown(f'<div class="success-message">✅ {len(st.session_state.uploaded_images)} image(s) uploaded!</div>', unsafe_allow_html=True)
        cols = st.columns(len(st.session_state.uploaded_images))
        for i, img in enumerate(st.session_state.uploaded_images):
            cols[i].image(img, caption=f"Image {i+1}", use_container_width=True)

    if st.session_state.uploaded_images and st.button("🔍 Extract Text from Images", type="primary", use_container_width=True):
        raw_text = extract_text_from_images(st.session_state.uploaded_images, st.session_state.api_key)
        if raw_text:
            st.session_state.extracted_text = raw_text
            st.success("Text extracted! Please proceed to the next tab to format the question.")

with tab2:
    st.markdown('<div class="step-header">Step 2: Format Question with AI</div>', unsafe_allow_html=True)
    if not st.session_state.extracted_text:
        st.info("Please upload images and extract text in Step 1.")
    else:
        with st.expander("📄 View Raw Extracted Text", expanded=False):
            st.text_area("", st.session_state.extracted_text, height=200, disabled=True)

        if st.button("🤖 Format Question with Llama 3.3", type="primary", use_container_width=True):
            with st.spinner("Formatting question... This may take a moment."):
                formatted = format_question_with_llama(st.session_state.extracted_text, st.session_state.api_key)
                if formatted:
                    st.session_state.formatted_question = formatted
                    st.session_state.test_cases = extract_test_cases(formatted)
                    st.success("Question formatted successfully! Proceed to the next step.")

        if st.session_state.formatted_question:
            st.markdown("### ✨ AI-Formatted Question")
            st.markdown(st.session_state.formatted_question, unsafe_allow_html=True)
            if st.session_state.test_cases:
                st.success(f"Successfully extracted {len(st.session_state.test_cases)} test case(s).")
            else:
                st.warning("Could not automatically extract test cases. Please check the formatted question.")

with tab3:
    st.markdown('<div class="step-header">Step 3: Generate Code Solution</div>', unsafe_allow_html=True)
    if not st.session_state.formatted_question:
        st.info("Please format the question in Step 2 first.")
    else:
        st.markdown("#### Using this question:")
        st.markdown(f'<div class="question-container">{st.session_state.formatted_question[:300]}...</div>', unsafe_allow_html=True)
        
        # Generate initial solution button
        if st.session_state.generation_count == 0:
            button_text = "🚀 Generate Initial Solution"
        else:
            button_text = f"🔄 Generate Solution (Attempt #{st.session_state.generation_count + 1})"
            
        if st.button(button_text, type="primary", use_container_width=True):
            with st.spinner("Generating solution..."):
                # Prepare context for regeneration
                previous_attempts = None
                error_feedback = None
                
                if st.session_state.generation_count > 0 and st.session_state.code_history:
                    # Get previous attempts
                    previous_attempts = "\n\n".join([f"Attempt {item['generation']}:\n{item['code']}" for item in st.session_state.code_history])
                    
                    # Get error feedback from last attempt
                    if st.session_state.run_output:
                        error_feedback = get_failed_test_summary(st.session_state.run_output)
                
                solution = generate_solution(
                    st.session_state.formatted_question, 
                    st.session_state.api_key,
                    previous_attempts=previous_attempts,
                    error_feedback=error_feedback,
                    user_comments=st.session_state.user_feedback
                )
                
                if solution:
                    st.session_state.generation_count += 1
                    st.session_state.generated_code = solution
                    st.session_state.edited_code = solution # Initialize editor with generated code
                    st.session_state.user_feedback = ""  # Clear previous feedback
                    st.success(f"Solution generated (Attempt #{st.session_state.generation_count})! You can now run or edit it in the next tab.")
        
        if st.session_state.generated_code:
            st.markdown(f"### 💡 Generated Code (Attempt #{st.session_state.generation_count})")
            st.code(st.session_state.generated_code, language="python")

with tab4:
    st.markdown('<div class="step-header">Step 4: Run & Edit Code</div>', unsafe_allow_html=True)
    if not st.session_state.generated_code:
        st.info("Please generate a code solution in Step 3 first.")
    else:
        st.markdown("### 🛠️ Code Editor")
        edited_code = st.text_area(
            "Edit your code here:",
            value=st.session_state.edited_code,
            height=400,
            key="code_editor",
            label_visibility="collapsed"
        )
        st.session_state.edited_code = edited_code

        if not st.session_state.test_cases:
            st.warning("No test cases were extracted. Cannot run the code.")
        else:
            if st.button("▶️ Run Code on Test Cases", type="primary", use_container_width=True):
                with st.spinner("Running code..."):
                    run_results = run_code(st.session_state.edited_code, st.session_state.test_cases)
                    st.session_state.run_output = run_results
                    
                    # Add to history
                    add_to_code_history(st.session_state.edited_code, run_results, st.session_state.generation_count)

        if st.session_state.run_output:
            st.markdown("### 📊 Execution Results")
            
            # Summary stats
            passed_tests = sum(1 for r in st.session_state.run_output if r['status'] == 'Passed')
            total_tests = len(st.session_state.run_output)
            
            if passed_tests == total_tests:
                st.success(f"🎉 All {total_tests} test cases passed! Great job!")
            else:
                st.error(f"❌ {passed_tests}/{total_tests} test cases passed")
                
                # Show regeneration section for failed tests
                st.markdown('<div class="regeneration-section">', unsafe_allow_html=True)
                st.markdown("### 🔄 Code Regeneration")
                st.markdown("Since some test cases failed, you can provide feedback to improve the solution:")
                
                user_feedback = st.text_area(
                    "Add your comments about what might be wrong or what approach to try:",
                    value=st.session_state.user_feedback,
                    height=100,
                    placeholder="e.g., 'The algorithm should handle negative numbers', 'Try using a different sorting approach', 'The output format seems incorrect'..."
                )
                st.session_state.user_feedback = user_feedback
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔄 Regenerate Code with Feedback", type="secondary", use_container_width=True):
                        # Switch to tab 3 for regeneration
                        st.info("Click on the 'Generate Code' tab and then click the generate button to create an improved solution based on the test failures and your feedback.")
                
                with col2:
                    if st.button("📋 Copy Failed Test Info", use_container_width=True):
                        failed_info = get_failed_test_summary(st.session_state.run_output)
                        st.code(failed_info, language="text")
                        
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed test results
            for result in st.session_state.run_output:
                status_color = "green" if result['status'] == 'Passed' else "red"
                with st.container():
                    st.markdown(f"**Test Case {result['case']}**: <span class='test-case-{result['status'].lower()}'>{result['status']}</span>", unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    col1.text_area("Input", value=result['input'], height=100, disabled=True, key=f"input_{result['case']}_{st.session_state.generation_count}")
                    col2.text_area("Your Output", value=result['output'], height=100, disabled=True, key=f"output_{result['case']}_{st.session_state.generation_count}")
                    if result['status'] == 'Failed':
                        col3.text_area("Expected Output", value=result.get('expected', ''), height=100, disabled=True, key=f"expected_{result['case']}_{st.session_state.generation_count}")
                    st.markdown("---")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Built with ❤️ using Streamlit and Together AI</p>
</div>
""", unsafe_allow_html=True)