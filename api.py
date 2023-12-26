from fastapi import FastAPI, File, UploadFile, HTTPException
import tensorflow as tf
import json 
from model_definition import SegmentationModel 

app = FastAPI() 

model = SegmentationModel().model
model.load_weights('cancer_weights.h5') 

@app.post('/')
async def scoring_endpoint(data: UploadFile = File(...)) -> dict:
    try:
        # Validate file type and size
        if data.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=415, detail="Unsupported media type. Please upload a JPEG or PNG image.")
        
        # Get file size
        file_size = data.file.fileno()
        if file_size > 5 * 1024 * 1024:  # 5 MB limit
            raise HTTPException(status_code=413, detail="File size exceeds the limit. Please upload a smaller image.")
        
        image_bytes = await data.read()
        image = tf.io.decode_image(image_bytes)
        
        # Make predictions
        yhat = model.predict(tf.expand_dims(image, axis=0))
        
        return {"prediction": json.dumps(yhat.tolist())}
    
    except Exception as e:
        # Handle exceptions and return an error response
        return {"error": f"An error occurred: {str(e)}"}
