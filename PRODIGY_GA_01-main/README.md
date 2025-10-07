# textgen_gpt-2
### **Setup and Installation**

To get this project up and running, follow these simple steps.

1.  **Create a Virtual Environment** 🐍: It's best practice to use a virtual environment to manage project dependencies. This command creates a new one named `.venv` in your project directory.
    ```bash
    python3 -m venv .venv
    ```
    
2.  **Activate the Virtual Environment**: Before installing anything, you need to activate the virtual environment you just created. This ensures all packages are installed within it and not globally.
    ```bash
    source .venv/bin/activate
    ```
---
### **Running the Application**

Once you've set up your environment, you're ready to install the necessary packages and run the application.

1.  **Install Dependencies** 📦: Use pip to install all the required libraries listed in the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```
    
2.  **Start the Streamlit App** 🚀: Finally, run the Streamlit application using the following command. This will open the app in your default web browser.
    ```bash
    streamlit run app.py
    ```
