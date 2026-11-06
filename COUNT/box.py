from boxsdk import OAuth2, Client
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading
import os
import tempfile
import tkinter as tk
from tkinter import messagebox
from dotenv import load_dotenv
from tqdm import tqdm


from COUNT import ui, tracking

# Load environment variables
load_dotenv()
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:1410/'  # This must match what's configured in your Box app

# Global variable to store the authorization code
authorization_code = None


class AuthCodeHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global authorization_code
        print(f"Received request: {self.path}")  # Debug output

        # Parse the query parameters
        query = urlparse(self.path).query
        if query:
            params = parse_qs(query)
            if 'code' in params:
                authorization_code = params['code'][0]
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b'Authorization successful! You can close this window.')
                return

        self.send_response(400)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'Authorization failed!')

    # Suppress server logs to console
    def log_message(self, format, *args):
        return


class BoxAPI_App:
    def __init__(self, master):
        """UI for box settings"""
        self.master = master
        self.file = tk.StringVar(value='')
        self.folder = tk.StringVar(value='')
        self.create_toolbar()
        self.create_widgets()

    def create_toolbar(self):
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        help_ = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='Help', menu=help_)
        help_.add_command(label='Help', command=self.help)

    def create_widgets(self):
        justification = "w"
        # File selection button
        tk.Label(self.master, text="Enter individual file_id to process", anchor=justification).grid(sticky="w", row=1,
                                                                                                     column=0,
                                                                                                     padx=10, pady=25)
        self.file_entry = tk.Entry(self.master, textvariable=self.file)
        self.file_entry.grid(row=1, column=1, padx=10, pady=25)

        tk.Label(self.master, text="or", anchor=justification).grid(sticky="w", row=2, column=0,padx=5, pady=5)

        tk.Label(self.master, text="Enter folder_id to process", anchor=justification).grid(sticky="w", row=3,
                                                                                            column=0,padx=5, pady=5)
        self.folder_entry = tk.Entry(self.master, textvariable=self.folder)
        self.folder_entry.grid(row=3, column=1, padx=10, pady=25)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="      Run     ", command=self.confirm_selections, height=2, width=20)
        self.confirm_button.grid(row=4, column=0, columnspan=3, padx=10, pady=10)

    def confirm_selections(self):
        self.master.destroy()

    def help(self):
        messagebox.showinfo("Help", "IDs are the numbers in the URL: \nEx: https://iastate.app.box.com/folder/123456789\nfolder_id=123456789")


def create_ui():
    root = tk.Tk()
    root.title("Connect to Box")
    app = BoxAPI_App(root)
    root.mainloop()
    return app


def start_server():
    print(f"Starting server on http://localhost:1410/")
    server = HTTPServer(('localhost', 1410), AuthCodeHandler)
    server.handle_request()  # Handle one request then exit


def authenticate_with_box():
    # Create OAuth2 object
    oauth = OAuth2(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET
    )

    # Get authorization URL - use the correct redirect URI here
    auth_url, csrf_token = oauth.get_authorization_url(REDIRECT_URI)

    # Open the URL in a browser
    print(f"Opening browser to authenticate with Box...")
    print(f"Auth URL: {auth_url}")  # Debug output
    webbrowser.open(auth_url)

    # Start local server to receive the redirect
    server_thread = threading.Thread(target=start_server)
    server_thread.start()
    server_thread.join()

    # Wait for the authorization code
    if authorization_code:
        print("Received authorization code. Authenticating...")
        # Authenticate with the code
        access_token, refresh_token = oauth.authenticate(authorization_code)
        print("Authentication successful!")

        # Create and return the client
        return Client(oauth)
    else:
        raise Exception("Failed to get authorization code")

def download_file_from_box(client, file_id):
    file_object = client.file(file_id)
    file_info = client.file(file_id).get()
    file_size = file_info.size

    print(f'File "{file_info.name}" has a size of {file_size} bytes')

    # Create a temporary file
    with tempfile.NamedTemporaryFile(suffix='.nd2', delete=False) as temp_file:
        temp_file_path = temp_file.name

        # Write binary content to the temporary file
        print(f"Downloading {file_info.name} bytes in chunks and writing to {temp_file_path}")
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Writing file") as pbar:
            # Download in chunks so a large file never has to fit in memory at once
            chunk_size = 64 * 1024 * 1024  # 64 MB
            byte_range_start = 0

            # Loop through the file in byte ranges. Box byte ranges are inclusive at both ends: (0, 9) is 10 bytes.
            while byte_range_start < file_size:
                byte_range_end = min(byte_range_start + chunk_size, file_size) - 1
                byte_range = (byte_range_start, byte_range_end)

                # Download the chunk of data from Box
                chunk_data = file_object.content(byte_range=byte_range)

                # Write the chunk to the temp file
                temp_file.write(chunk_data)

                # Update the start of the next byte range
                byte_range_start = byte_range_end + 1

                pbar.update(len(chunk_data))

    if os.path.getsize(temp_file_path) != file_size:
        raise IOError(f"Download of {file_info.name} is incomplete: "
                      f"{os.path.getsize(temp_file_path)} of {file_size} bytes")
    print(f"File written to {temp_file_path}")
    return temp_file_path


def process_box_file(client, file_id, file_name, app):
    """Download one .nd2 file from Box to a temporary file, track it, save the CSVs, then delete the temp file."""
    temp_file_path = download_file_from_box(client, file_id)
    try:
        # COUNT onto downloaded temp file
        counted_objects = tracking.nd2_mog_contours(temp_file_path, app)
        stem = os.path.splitext(file_name)[0]
        tracking.export_to_csv(counted_objects, os.path.join(app.csv_folder_path.get(), f"{stem}_results.csv"))
        tracking.export_trajectories_to_csv(
            counted_objects.values(), os.path.join(app.csv_folder_path.get(), f"{stem}_trajectory_results.csv"))
    finally:
        # Clean up the temporary file when done
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            print(f"Temp file removed: {temp_file_path}")


def main():
    # Get user settings for COUNT. In this mode the file/folder buttons are ignored: files come from Box.
    app = ui.create_ui(require_files=False)
    if not app.confirmed:
        print("No files processed.")
        return

    # Authenticate with Box
    client = authenticate_with_box()

    # Choose file/folder located in box
    box_app = create_ui()

    # Get files from box
    if box_app.file.get():  # Was a file id provided
        file_id = box_app.file.get()
        file_info = client.file(file_id).get()
        process_box_file(client, file_id, file_info.name, app)

    if box_app.folder.get():  # Was a folder id provided
        for item in client.folder(box_app.folder.get()).get_items():  # Go through items in the folder
            print(item.name)
            if item.type == 'file' and item.name.lower().endswith('.nd2'):  # skip sub-folders and other files
                process_box_file(client, item.id, item.name, app)


if __name__ == "__main__":
    main()