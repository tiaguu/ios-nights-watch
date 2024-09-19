import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse
import os

app = FastAPI()

@app.get("/health", status_code=200)
def health_check():
    return True

@app.post("/upload")
async def upload_application(file: UploadFile = File(...)):
    try:
        file_location = f"uploads/{file.filename}"  # Save the file to the 'uploads' directory
        os.makedirs(os.path.dirname(file_location), exist_ok=True)  # Ensure directory exists
        # Write to file in chunks
        with open(file_location, "wb+") as file_object:
            while content := await file.read(1024 * 1024):  # Read in chunks of 1MB
                file_object.write(content)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not upload the file: {e}"})
    
@app.post("/upload/malware")
async def upload_malware_application(file: UploadFile = File(...)):
    try:
        file_location = f"malware/{file.filename}"  # Save the file to the 'uploads' directory
        os.makedirs(os.path.dirname(file_location), exist_ok=True)  # Ensure directory exists
        # Write to file in chunks
        with open(file_location, "wb+") as file_object:
            while content := await file.read(1024 * 1024):  # Read in chunks of 1MB
                file_object.write(content)
        return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not upload the file: {e}"})
    
@app.get("/download/{number}")
async def download_goodware(number: int):
    try:
        files_dir = "goodware"
        files = os.listdir(files_dir)
        alphabetical_files = sorted(files)
        download_path = os.path.join(files_dir, alphabetical_files[number])
        return FileResponse(download_path, media_type='application/zip', filename=alphabetical_files[number])
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not download the file: {e}"})
    
@app.get("/download/malware/{number}")
async def download_malware(number: int):
    try:
        files_dir = "malware"
        files = os.listdir(files_dir)
        alphabetical_files = sorted(files)
        download_path = os.path.join(files_dir, alphabetical_files[number])
        return FileResponse(download_path, media_type='application/zip', filename=alphabetical_files[number])
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not download the file: {e}"})
    
@app.get("/download/opcodes/malware/{number}")
async def download_opcodes_malware(number: int):
    try:
        files_dir = "malware-opcodes"
        files = os.listdir(files_dir)
        sorted_files = sorted(files, key=lambda x: os.path.getsize(os.path.join(files_dir, x)))
        download_path = os.path.join(files_dir, sorted_files[number])
        return FileResponse(download_path, media_type='text/plain', filename=sorted_files[number])
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not download the file: {e}"})
    
@app.get("/download/opcodes/goodware/{number}")
async def download_opcodes_goodware(number: int):
    try:
        files_dir = "goodware-opcodes"
        files = os.listdir(files_dir)
        sorted_files = sorted(files, key=lambda x: os.path.getsize(os.path.join(files_dir, x)))
        download_path = os.path.join(files_dir, sorted_files[number])
        return FileResponse(download_path, media_type='text/plain', filename=sorted_files[number])
    except Exception as e:
        return JSONResponse(status_code=400, content={"message": f"Could not download the file: {e}"})