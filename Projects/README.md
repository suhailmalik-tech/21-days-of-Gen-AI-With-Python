.

🤖 Agentic Chatbot with Stateful Memory
An interactive, lightning-fast AI chatbot built with LangChain, LangGraph, and Streamlit. Featuring dynamic agentic workflows, branching decision loops, and persistent conversational memory.

✨ Features
Stateful Agent Workflows: Built using LangGraph to handle complex branching logic, loops, and fallback decisions.

Persistent Memory: Maintains contextual user history across dynamic graph nodes for natural, multi-turn conversations.

Streamlit UI: Intuitive, responsive web application for seamless interaction.

Fast & Lightweight: Optimized workflow pipeline designed for minimal latency.

🛠️ Tech Stack
Language: Python 3.8+

Orchestration & Agents: LangChain, LangGraph

Frontend / UI: Streamlit

Version Control: Git

🚀 Getting Started
Follow these steps to set up and run the project locally on your machine.

Prerequisites
Ensure you have the following installed on your system:

Python 3.8+

Git

Visual Studio Code (or your preferred editor)

💻 Local Setup & Installation
Step 1: Clone the Repository
Bash
git clone https://github.com/suhailmalik-tech/21-days-of-Gen-AI-With-Python.git
cd 21-days-of-Gen-AI-With-Python/Projects
Step 2: Create and Activate a Virtual Environment
On Windows (PowerShell / Command Prompt):

Bash
python -m venv venv
venv\Scripts\activate
On macOS / Linux:

Bash
python3 -m venv venv
source venv/bin/activate
Step 3: Install Dependencies
Bash
pip install -r requirements.txt
(Optional) Verify installed packages:

Bash
pip list
🎯 Running the Application
Launch the Streamlit app with the following command:

Bash
streamlit run AgenticChatbot.py
Once the server starts, open your browser and navigate to:
👉 http://localhost:8501

📂 Project Structure
Plaintext
├── AgenticChatbot.py    # Main Streamlit UI and agent workflow initialization
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
