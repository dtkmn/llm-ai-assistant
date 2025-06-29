import logging
import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import RequestEntityTooLarge
import DocumentQA

# Initialize Flask app and CORS
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.logger.setLevel(logging.ERROR)

# Define paths relative to the application root
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'uploads')
ALLOWED_EXTENSIONS = {'pdf'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

# Ensure directory exists with write permissions
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

qa_system = DocumentQA.DocumentQA()

# Define the route for the index page
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')  # Render the index.html template

# Define the route for processing messages
@app.route('/process-message', methods=['POST'])
def process_message_route():
    user_message = request.json['userMessage']  # Extract the user's message from the request
    print('user_message', user_message)

    bot_response = qa_system.query(user_message)  # Process the user's message using the worker module

    # Return the bot's response as JSON
    return jsonify({
        "botResponse": bot_response
    }), 200

# Define the route for processing documents
@app.route('/process-document', methods=['POST'])
def process_document_route():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({
            "botResponse": "It seems like the file was not uploaded correctly, can you try "
                           "again. If the problem persists, try using a different file"
        }), 400

    file = request.files['file']  # Extract the uploaded file from the request
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)  # Save the file

    qa_system.process_document(file_path)  # Process the document using the worker module

    # Return a success message as JSON
    return jsonify({
        "botResponse": "Thank you for providing your PDF document. I have analyzed it, so now you can ask me any "
                       "questions regarding it!"
    }), 200

# Custom error handler for 413 Request Entity Too Large
@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    # You can dynamically get the limit from app.config if you want
    # max_size_mb = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
    # For now, let's hardcode it based on the current setting or make it general
    message = f"Error: File exceeds the maximum allowed size of {app.config['MAX_CONTENT_LENGTH'] // (1024*1024)}MB."
    return jsonify(botResponse=message, error=True), 413

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, port=7860, host='0.0.0.0')
