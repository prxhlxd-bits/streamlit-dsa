import streamlit as st
import base64
from together import Together
from PIL import Image
import io
import time
import re
import subprocess
import requests
import json

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
    .question-container, .code-container, .output-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .debugger-container {
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
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
    .test-case-pass { color: #28a745; font-weight: bold; }
    .test-case-fail { color: #dc3545; font-weight: bold; }
    .test-case-error { color: #ffc107; font-weight: bold; }
    .test-case-container {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
st.session_state.setdefault('together_api_key', "")
st.session_state.setdefault('gemini_api_key', "")
st.session_state.setdefault('uploaded_images', [])
st.session_state.setdefault('extracted_text', "")
st.session_state.setdefault('formatted_question', "")
st.session_state.setdefault('edited_question', "")
st.session_state.setdefault('generated_code', "")
st.session_state.setdefault('edited_code', "")
st.session_state.setdefault('test_cases', [])
st.session_state.setdefault('custom_test_cases', [])
st.session_state.setdefault('run_output', [])
st.session_state.setdefault('error_feedback', "")


# --- API and Helper Functions ---

def handle_together_api_error(e):
    st.error(f"An API error occurred with Together AI: {str(e)}")
    if "authentication" in str(e).lower():
        st.warning("Please check if your Together AI API key is correct and has sufficient credits.")

def handle_gemini_api_error(response):
    """Handles errors from the Gemini API."""
    try:
        error_info = response.json()
        st.error(f"An API error occurred with Gemini: {error_info.get('error', {}).get('message', 'Unknown error')}")
    except json.JSONDecodeError:
        st.error(f"An API error occurred with Gemini. Status code: {response.status_code}, Response: {response.text}")


def extract_text_from_images(images, api_key):
    """Extracts text from images using Together AI's Llama-Vision."""
    if not api_key:
        st.error("Please enter your Together AI API key in the sidebar.")
        return ""
    try:
        client = Together(api_key=api_key)
        prompt = "Extract all text from the image. Focus on the question, description, examples, and constraints. Ignore UI elements like buttons or unrelated text."
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
        handle_together_api_error(e)
        return ""


def format_question_with_llama(raw_text, api_key):
    """Formats raw text into a structured question using Together AI."""
    if not api_key:
        st.error("Please enter your Together AI API key in the sidebar.")
        return ""
    try:
        client = Together(api_key=api_key)
        prompt = f"""
            Based on the following text extracted from one or more images, please structure it into a clear and well-formatted coding problem.
            The output should strictly follow this format:
            ### Problem Description
            [Provide a clear and concise description of the problem.]
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
            Here is the raw text: --- {raw_text} ---
        """
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        handle_together_api_error(e)
        return ""


def generate_solution_with_gemini(question, gemini_api_key, previous_code=None, error_message=None, user_feedback=None):
    """Generates or regenerates a code solution using the Gemini API."""
    if not gemini_api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        return ""

    model_name = "gemini-2.5-flash-preview-05-20"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_api_key}"

    # Base prompt for initial generation
    prompt = f"""
        Please provide a complete and runnable Python code solution for the following programming problem.
        The code should be self-contained and ready to execute.
        Do not include any explanations, analysis, or markdown formatting, only the raw Python code.
        Problem: {question}
    """

    # Modify prompt for debugging and regeneration
    if previous_code and error_message:
        prompt = f"""
            You are an expert Python programmer debugging a solution.
            The previous code attempt failed. Please analyze the problem, the faulty code, the error, and the user's feedback to provide a corrected, runnable Python solution.
            Only output the raw, corrected Python code.

            **Original Problem:**
            {question}

            **Faulty Code:**
            ```python
            {previous_code}
            ```

            **Error Message:**
            {error_message}

            **User Feedback (optional):**
            {user_feedback if user_feedback else "No feedback provided."}
        """

    payload = {"contents": [{"parts": [{"text": prompt}]}]}
    headers = {'Content-Type': 'application/json'}

    try:
        response = requests.post(api_url, headers=headers, json=payload)
        if response.status_code == 200:
            data = response.json()
            code = data['candidates'][0]['content']['parts'][0]['text']
            code_match = re.search(r'```python\n(.*?)```', code, re.DOTALL)
            if code_match:
                return code_match.group(1).strip()
            return code.strip()
        else:
            handle_gemini_api_error(response)
            return ""
    except requests.exceptions.RequestException as e:
        st.error(f"Network error when calling Gemini API: {e}")
        return ""


def extract_test_cases(formatted_question):
    """Extracts input/output pairs from the formatted question using regex."""
    test_cases = []
    example_blocks = re.findall(r'Example \d+:(.*?)($|Example \d+:|###)', formatted_question, re.DOTALL)
    for block, _ in example_blocks:
        input_match = re.search(r'Input:\s*\n(.*?)\n\s*Output:', block, re.DOTALL)
        output_match = re.search(r'Output:\s*\n(.*)', block, re.DOTALL)
        if input_match and output_match:
            test_cases.append({
                "input": input_match.group(1).strip(),
                "expected_output": output_match.group(1).strip(),
                "type": "extracted"
            })
    return test_cases


def add_custom_test_case():
    """Add a new custom test case to the session state."""
    if 'custom_test_cases' not in st.session_state:
        st.session_state.custom_test_cases = []
    st.session_state.custom_test_cases.append({
        "input": "",
        "expected_output": "",
        "type": "custom"
    })


def remove_custom_test_case(index):
    """Remove a custom test case at the given index."""
    if 0 <= index < len(st.session_state.custom_test_cases):
        st.session_state.custom_test_cases.pop(index)


def get_all_test_cases():
    """Get all test cases (extracted + custom)."""
    all_cases = st.session_state.test_cases.copy()
    all_cases.extend(st.session_state.custom_test_cases)
    return all_cases


def run_code(code_string, test_cases):
    """Executes code against test cases and captures output/errors."""
    results = []
    for i, case in enumerate(test_cases):
        try:
            process = subprocess.run(
                ['python', '-c', code_string],
                input=case['input'], text=True, capture_output=True, timeout=10
            )
            stdout, stderr = process.stdout.strip(), process.stderr.strip()
            result = {
                "case": i + 1, 
                "input": case['input'],
                "type": case.get('type', 'extracted')
            }
            if stderr:
                result.update({"status": "Error", "output": stderr})
            else:
                result.update({"output": stdout})
                result["status"] = "Passed" if stdout == case['expected_output'] else "Failed"
                if result["status"] == "Failed":
                    result["expected"] = case['expected_output']
            results.append(result)
        except subprocess.TimeoutExpired:
            results.append({
                "case": i + 1, 
                "input": case['input'], 
                "status": "Error", 
                "output": "Execution timed out.",
                "type": case.get('type', 'extracted')
            })
        except Exception as e:
            results.append({
                "case": i + 1, 
                "input": case['input'], 
                "status": "Error", 
                "output": f"An unexpected error occurred: {str(e)}",
                "type": case.get('type', 'extracted')
            })
    return results

# --- UI Layout ---

st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="color: #007bff; margin-bottom: 0.5rem;">🧠 DSA Question Solver</h1>
    <p style="color: #6c757d; font-size: 1.1rem;">From Image to Executable Code</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    st.session_state.together_api_key = st.text_input("Together AI API Key", type="password", value=st.session_state.together_api_key, help="For text extraction and question formatting.")
    st.session_state.gemini_api_key = st.text_input("Gemini API Key", type="password", value=st.session_state.gemini_api_key, help="For code generation.")

# Main content tabs
tabs = ["1. Upload Images", "2. Format Question", "3. Generate Code", "4. Run & Edit Code"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

with tab1:
    st.markdown('<div class="step-header">Step 1: Upload Question Images</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Choose up to 4 images", type=['png', 'jpg', 'jpeg'], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.uploaded_images = [Image.open(file) for file in uploaded_files]
        st.success(f"{len(st.session_state.uploaded_images)} image(s) uploaded!")
        cols = st.columns(len(st.session_state.uploaded_images))
        for i, img in enumerate(st.session_state.uploaded_images):
            cols[i].image(img, caption=f"Image {i+1}", use_column_width=True)
    if st.session_state.uploaded_images and st.button("🔍 Extract Text", type="primary", use_container_width=True):
        st.session_state.extracted_text = extract_text_from_images(st.session_state.uploaded_images, st.session_state.together_api_key)
        if st.session_state.extracted_text:
            st.success("Text extracted! Proceed to the next tab.")

with tab2:
    st.markdown('<div class="step-header">Step 2: Format and Edit Question</div>', unsafe_allow_html=True)
    if not st.session_state.extracted_text:
        st.info("Please upload and extract text in Step 1.")
    else:
        with st.expander("📄 View Raw Extracted Text"):
            st.text_area("", st.session_state.extracted_text, height=200, disabled=True)
        
        if st.button("🤖 Format Question with AI", type="primary", use_container_width=True):
            with st.spinner("Formatting question..."):
                st.session_state.formatted_question = format_question_with_llama(st.session_state.extracted_text, st.session_state.together_api_key)
                st.session_state.edited_question = st.session_state.formatted_question
                st.session_state.test_cases = extract_test_cases(st.session_state.formatted_question)
        
        if st.session_state.formatted_question:
            st.markdown("### ✨ AI-Formatted Question")
            with st.expander("📖 View AI-Formatted Question", expanded=True):
                st.markdown(st.session_state.formatted_question)
            
            st.markdown("### ✏️ Edit Question")
            st.session_state.edited_question = st.text_area(
                "Edit the formatted question here:",
                value=st.session_state.edited_question,
                height=400,
                help="You can modify the AI-formatted question before proceeding to code generation."
            )
            
            if st.button("🔄 Update Test Cases from Edited Question", use_container_width=True):
                st.session_state.test_cases = extract_test_cases(st.session_state.edited_question)
                st.success(f"Updated! Extracted {len(st.session_state.test_cases)} test case(s) from edited question.")
            
            if st.session_state.test_cases:
                st.markdown("### 📋 Extracted Test Cases")
                for i, case in enumerate(st.session_state.test_cases):
                    with st.container():
                        st.markdown(f"**Test Case {i+1}:**")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.text_area(f"Input {i+1}", value=case['input'], height=100, disabled=True, key=f"extracted_input_{i}")
                        with col2:
                            st.text_area(f"Expected Output {i+1}", value=case['expected_output'], height=100, disabled=True, key=f"extracted_output_{i}")

with tab3:
    st.markdown('<div class="step-header">Step 3: Generate Code with Gemini</div>', unsafe_allow_html=True)
    if not st.session_state.edited_question:
        st.info("Please format and edit the question in Step 2.")
    else:
        with st.expander("📋 Question to be used for code generation"):
            st.markdown(st.session_state.edited_question)
        
        if st.button("🚀 Generate Solution with Gemini", type="primary", use_container_width=True):
            with st.spinner("Generating solution with Gemini..."):
                solution = generate_solution_with_gemini(st.session_state.edited_question, st.session_state.gemini_api_key)
                if solution:
                    st.session_state.generated_code = solution
                    st.session_state.edited_code = solution
                    st.success("Solution generated! Proceed to the next tab to run it.")
        
        if st.session_state.generated_code:
            st.markdown("### 💡 Generated Code")
            st.code(st.session_state.generated_code, language="python")

with tab4:
    st.markdown('<div class="step-header">Step 4: Test Cases, Run & Debug</div>', unsafe_allow_html=True)
    if not st.session_state.generated_code:
        st.info("Please generate a solution in Step 3.")
    else:
        # Test Cases Management Section
        st.markdown("### 📋 Test Cases Management")
        
        # Display extracted test cases
        if st.session_state.test_cases:
            st.markdown("#### Extracted Test Cases")
            for i, case in enumerate(st.session_state.test_cases):
                with st.container():
                    st.markdown(f'<div class="test-case-container">', unsafe_allow_html=True)
                    st.markdown(f"**Extracted Test Case {i+1}:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.text_area(f"Input", value=case['input'], height=80, disabled=True, key=f"ext_display_input_{i}")
                    with col2:
                        st.text_area(f"Expected Output", value=case['expected_output'], height=80, disabled=True, key=f"ext_display_output_{i}")
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Custom test cases section
        st.markdown("#### Custom Test Cases")
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("➕ Add Custom Test Case"):
                add_custom_test_case()
        
        # Display and edit custom test cases
        if st.session_state.custom_test_cases:
            for i, case in enumerate(st.session_state.custom_test_cases):
                with st.container():
                    st.markdown(f'<div class="test-case-container">', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns([3, 3, 1])
                    with col1:
                        st.session_state.custom_test_cases[i]['input'] = st.text_area(
                            f"Custom Input {i+1}", 
                            value=case['input'], 
                            height=80, 
                            key=f"custom_input_{i}"
                        )
                    with col2:
                        st.session_state.custom_test_cases[i]['expected_output'] = st.text_area(
                            f"Custom Expected Output {i+1}", 
                            value=case['expected_output'], 
                            height=80, 
                            key=f"custom_output_{i}"
                        )
                    with col3:
                        st.write("")  # Add some spacing
                        st.write("")  # Add some spacing
                        if st.button("🗑️", key=f"remove_custom_{i}", help="Remove this test case"):
                            remove_custom_test_case(i)
                            st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
        
        # Code Editor Section
        st.markdown("### 🛠️ Code Editor")
        st.session_state.edited_code = st.text_area("Edit your code here:", value=st.session_state.edited_code, height=400, label_visibility="collapsed")

        # Get all test cases for execution
        all_test_cases = get_all_test_cases()
        
        if all_test_cases and st.button("▶️ Run Code on All Test Cases", type="primary", use_container_width=True):
            st.session_state.run_output = run_code(st.session_state.edited_code, all_test_cases)
            st.session_state.error_feedback = ""  # Clear old feedback on new run

        # Enhanced Debugger UI
        if st.session_state.run_output:
            failed_cases = [r for r in st.session_state.run_output if r['status'] in ['Error', 'Failed']]
            
            if failed_cases:
                with st.container():
                    st.markdown('<div class="debugger-container">', unsafe_allow_html=True)
                    st.error(f"Issues detected in {len(failed_cases)} test case(s).")
                    
                    # Group failed cases by type
                    error_cases = [r for r in failed_cases if r['status'] == 'Error']
                    failed_test_cases = [r for r in failed_cases if r['status'] == 'Failed']
                    
                    if error_cases:
                        st.markdown("**Runtime Errors:**")
                        for error in error_cases:
                            st.markdown(f"- **Test Case {error['case']} ({error.get('type', 'extracted').title()})**: {error['output']}")
                    
                    if failed_test_cases:
                        st.markdown("**Wrong Output:**")
                        for fail in failed_test_cases:
                            st.markdown(f"- **Test Case {fail['case']} ({fail.get('type', 'extracted').title()})**: Expected `{fail['expected']}`, got `{fail['output']}`")
                    
                    st.session_state.error_feedback = st.text_area(
                        "Add comments or hints for Gemini to fix the code:", 
                        value=st.session_state.error_feedback,
                        height=100,
                        placeholder="e.g., 'The algorithm should handle edge cases like empty arrays' or 'Consider using a different data structure'"
                    )

                    if st.button("🔧 Fix Code with Gemini", use_container_width=True):
                        with st.spinner("Gemini is analyzing and fixing the code..."):
                            # Create detailed error context for Gemini
                            error_details = []
                            for result in failed_cases:
                                if result['status'] == 'Error':
                                    error_details.append(f"Test Case {result['case']} ({result.get('type', 'extracted').title()}) - Runtime Error: {result['output']}")
                                elif result['status'] == 'Failed':
                                    error_details.append(f"Test Case {result['case']} ({result.get('type', 'extracted').title()}) - Wrong Output: Expected '{result['expected']}', Got '{result['output']}'")
                            
                            error_context = "\n".join(error_details)
                            
                            new_code = generate_solution_with_gemini(
                                st.session_state.edited_question, 
                                st.session_state.gemini_api_key,
                                previous_code=st.session_state.edited_code,
                                error_message=error_context,
                                user_feedback=st.session_state.error_feedback
                            )
                            if new_code:
                                st.session_state.edited_code = new_code
                                st.session_state.generated_code = new_code
                                st.session_state.run_output = []  # Clear old results
                                st.success("Gemini generated a new solution! It's now in the editor above.")
                                st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.success("🎉 All test cases passed!")

        # Display detailed results
        if st.session_state.run_output:
            st.markdown("### 📊 Detailed Execution Results")
            for result in st.session_state.run_output:
                with st.container():
                    case_type_badge = f"({result.get('type', 'extracted').title()})"
                    st.markdown(f"**Test Case {result['case']} {case_type_badge}**: <span class='test-case-{result['status'].lower()}'>{result['status']}</span>", unsafe_allow_html=True)
                    
                    if result['status'] == 'Failed':
                        cols = st.columns(3)
                        cols[0].text_area("Input", value=result['input'], height=100, disabled=True, key=f"result_input_{result['case']}")
                        cols[1].text_area("Your Output", value=result['output'], height=100, disabled=True, key=f"result_output_{result['case']}")
                        cols[2].text_area("Expected", value=result.get('expected', ''), height=100, disabled=True, key=f"result_expected_{result['case']}")
                    else:
                        cols = st.columns(2)
                        cols[0].text_area("Input", value=result['input'], height=100, disabled=True, key=f"result_input_2col_{result['case']}")
                        cols[1].text_area("Output", value=result['output'], height=100, disabled=True, key=f"result_output_2col_{result['case']}")
                    
                    st.markdown("---")

# Footer
st.markdown("""
---
<div style="text-align: center; color: #6c757d; padding: 1rem;">
    <p>Built with ❤️ using Streamlit, Together AI, and Gemini</p>
</div>
""", unsafe_allow_html=True)
