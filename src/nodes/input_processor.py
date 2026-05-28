import fitz  # PyMuPDF

def input_processor(state):

    # Check if user gave a PDF file
    if state.document_path is not None:

        try:
            # Open the PDF
            doc = fitz.open(state.document_path)

            text = ""

            # Read every page
            for page in doc:
                text += page.get_text()

            # Close the PDF
            doc.close()

            # Save extracted text into state
            state.document_text = text

        except Exception as e:

            # If error happens
            state.document_text = f"Error: {e}"

    # Return updated state
    return state