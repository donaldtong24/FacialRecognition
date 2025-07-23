import cv2
import os

# importing the required libraries
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image as PILImage
import psycopg2
import os

from IPython.display import Image, display

# Create folder if it doesn't exist
if not os.path.exists('stored-faces'):
    os.makedirs('stored-faces')

# Load the Haar cascade file
alg = "haarcascade_frontalface_default.xml"
haar_cascade = cv2.CascadeClassifier(alg)

# Load the image in color
file_name = r"oldheads.jpg"
img = cv2.imread(file_name)
if img is None:
    print("Image not found. Please check the file path.")
    exit()

# Convert to grayscale for face detection
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.05, minNeighbors=1, minSize=(50, 50)
)

print(f"Found {len(faces)} face(s).")

# Save cropped faces
for i, (x, y, w, h) in enumerate(faces):
    cropped_face = img[y:y+h, x:x+w]
    target_path = f'stored-faces/{i}.jpg'
    cv2.imwrite(target_path, cropped_face)
    print(f"Saved: {target_path}")


#step 2 Embeddings calculation
#Calculate embeddings from the faces and pushing to PostgreSQL, you'll need to change the <SERVICE_URI> parameter with the PostgreSQL Service URI

# connecting to the database - replace the SERVICE URI with the service URI
conn = psycopg2.connect("")

for filename in os.listdir("stored-faces"):
    # opening the image
    img = PILImage.open("stored-faces/" + filename)
    # loading the `imgbeddings`
    ibed = imgbeddings()
    # calculating the embeddings
    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("""
    SELECT filename, embedding <#> %s::vector AS distance
    FROM pictures
    ORDER BY distance ASC
    LIMIT 1
""", (embedding[0].tolist(),))
    print(filename)
conn.commit()


#Step 3: Calculate embeddings on a new picture
#Find the face and calculate the embeddings on the picture
# loading the face image path into file_name variable
file_name = "slimReaper.jpg"  # replace <INSERT YOUR FACE FILE NAME> with the path to your image
# opening the image
img = PILImage.open(file_name)
# loading the `imgbeddings`
ibed = imgbeddings()
# calculating the embeddings
embedding = ibed.to_embeddings(img)

#Step 4: Find similar images by querying the Postgresql database using pgvector

cur = conn.cursor()
string_representation = "["+ ",".join(str(x) for x in embedding[0].tolist()) +"]"
# Find the most similar image in the database using pgvector
cur.execute("""
    SELECT filename, embedding <#> %s::vector AS distance
    FROM pictures
    ORDER BY distance ASC
    LIMIT 1
""", (embedding[0].tolist(),))
row = cur.fetchone()
if row:
    display(Image(filename="stored-faces/"+row[0]))
    print(f"Most similar image: {row[0]}")
else:
    print("No similar images found.")
cur.close()