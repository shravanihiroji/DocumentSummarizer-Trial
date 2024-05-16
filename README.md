•	Developed the code for document summarizer on VS Code.
•	Created a virtual environment in VS Code as venv using conda create -p venv python==3.10
•	Activate the environment using conda activate venv
•	Create Google API key by going to the website www.makersuite.google.com/app/apikey
•	Create a .env file and save the API key in the variable GOOGLE_API_KEY
•	Create requirements file to install all the required libraries like streamlit, google-generativeai, python-dotenv, langchain, PyPDF2, faiss-cpu and langchain_google_genai
•	Install the libraries using pip install -r requirements.txt
•	Imported all the required libraries in the code
•	To read the PDF we use PyPDF2 importing the PDF Reader
•	Used RecursiveCharacterTextSplitter as we would read the PDF and split the text to convert it into vectors
•	For vectorizing the text used GoogleGenerativeAIEmbeddings
•	FAISS is used for vector embeddings
•	Load_qa_chain and prompt template is used to chat with the PDFs
•	Load_dotenv used to load the environment variable
•	Configure the Gen AI environment with GOOGLE_API_KEY
•	Created a function to get the text data from the PDFs in which PDFs are read using PdfReader. This will go through all the pages of the PDF and extract the text.
•	The text is then divided into multiple chunks using a function. Here we use RecursiveTestSplitter to split the text into smaller chunks and a chunk size of 10000 with an overlap of 1000 is kept.
•	Now we would convert the chunks into vectors using a function. This is done by using Goggle GenAI embeddings by using embedding-001 as the technique for doing the embeddings. These embeddings are stored in a vector store using FAISS locally. This would create a pickle file.
•	Created a function for giving the prompt to the developed application. The prompt message specifies what is needed by the application to be done. We use Gemini pro model over here. The two inputs to the prompt would be our context and the question. We use chain type as “stuff” to get conversational chain.
•	Creating a function for taking the user input in which we load the embeddings and FAISS index from the local. Here we do similarity search to get the response. This is related to the text box, where the question is read, similarity check is done in the FAISS vectors and then the conversational chain is given as the response basis on the new similarity search. This way we get the response.
•	Creating a Streamlit app to create a front end for the application.
•	This app will allow the user to import PDFs and then ask any question about the PDF.
•	If the question is not related to the PDF, it will return “Content not found”.
