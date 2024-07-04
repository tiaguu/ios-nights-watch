import uvicorn
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import JSONResponse
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