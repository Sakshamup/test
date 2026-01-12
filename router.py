from fastapi import APIRouter, UploadFile, File, HTTPException
import tempfile
import os

from app import analyze_emr_file

router = APIRouter(
    prefix="/emr",
    tags=["EMR"]
)


@router.post("/analyze")
async def analyze_emr(file: UploadFile = File(...)):
    """
    Analyze veterinary EMR / lab reports (PDF or Image)
    Supported species: Dogs and Cats
    """

    filename = file.filename.lower()

    if not filename.endswith((
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".bmp",
        ".tiff",
        ".webp"
    )):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Upload PDF or image."
        )

    temp_path = None

    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(filename)[1]
        ) as tmp:
            tmp.write(await file.read())
            temp_path = tmp.name

        # Call EMR business logic
        result = analyze_emr_file(temp_path)

        return {
            "status": "success",
            "data": result
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)

