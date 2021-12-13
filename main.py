import subprocess
from typing import List
import os
import uvicorn
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from core.config import TMP_DIRECTORY
from routers import extraction

app = FastAPI()
app.include_router(extraction.router)

subprocesses: List[subprocess.Popen] = []



@app.on_event("shutdown")
def shutdown_event():
    """ Executes tasks if this program stops. """
    for process in subprocesses:
        process.kill()

    # Deletes all files in the tmp directories
    for dir, _, files in os.walk(TMP_DIRECTORY):
        for file in files:
            file_path = os.path.join(dir, file)
            if os.path.exists(file_path):
                os.remove(file_path)



@app.exception_handler(StarletteHTTPException)
async def custom_http_exception_handler(request, exc):
    """ Redirects every wrong Request to the docs. """
    return RedirectResponse("/docs")

if __name__ == "__main__":
    uvicorn.run(app, port=8002, host='0.0.0.0')
