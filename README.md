# 2025-CAS-RPM-NN-GenAI-Workshop

## Setup
  
1. **Clone the Repository:**

```bash
git clone https://github.com/bettyzhu912/2025-CAS-RPM-NN-GenAI-Workshop.git
```
      
2. **Check you have Python installed:**

Make sure you have Python >= 3.9 installed. If you're on a work computer, you may need admin privileges to install Python.

You can check you have python by running:
```bash
python --version
```

3. **Install the required packages:**

Mac/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

4. **Run the notebook:**

```bash
jupyter notebook
```

Then you can open the notebook in your browser.

5. **API Keys:**

This notebook uses an OpenAI-based approach for simplicity. Given the low token usage, it should only cost a few cents to run.

- If you're new to the OpenAI API, [sign up for an account](https://platform.openai.com/signup).
- Follow the [Quickstart](https://platform.openai.com/docs/quickstart) to retrieve your API key.

You can paste that key into the notebook in the cell that starts with 

```python
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
```

Make sure to replace `YOUR_API_KEY` with your actual API key and uncomment the line (remove the `#` at the beginning of the line).

> ⚠️ **Work Computer SSL Issues:** Some corporate networks may block access to OpenAI’s API, leading to SSL verification errors. If you encounter this issue, consider:  
> - Running the notebook on a personal computer.  
> - Using the [hosted Colab version](https://colab.research.google.com/drive/15uvMEzytBbf65HBhjh82-mHDTK6GnYt5) (ensure your VPN is off).  
> - Checking with your IT team for potential workarounds, such as configuring SSL settings or using a company-approved proxy.

# Part 1: ...

# Part 2: Bridging Data Divides Workbook

This repo offers a hands-on workbook and interactive notebook examples that illustrate key concepts such as retrieval-augmented generation, embeddings, and structured outputs. It’s tailored for actuaries and data professionals seeking practical approaches to unify and analyze unstructured data, apply AI techniques to real-world challenges, and integrate these insights into existing insurance workflows.

### Hosted Colab Version:

For those who don't want to setup Python, you can use the hosted Colab version of the notebook. Note this does require a Google account and it may be blocked by your company's security policies.

[Open In Colab](https://colab.research.google.com/drive/15uvMEzytBbf65HBhjh82-mHDTK6GnYt5)
