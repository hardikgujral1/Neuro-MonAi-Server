import requests
from requests_toolbelt.multipart import decoder

# Call MONAI Label API
url = "http://localhost:8000/infer/brats_mri_segmentation"
files = {"file": open("combined_4mod.nii.gz", "rb")}
data = {"params": "{}"}

response = requests.post(url, files=files, data=data)

# Decode multipart response
multipart_data = decoder.MultipartDecoder.from_response(response)

for part in multipart_data.parts:
    content_disposition = part.headers.get(b'Content-Disposition', b'').decode()
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[1].strip('"')
        with open(filename, "wb") as f:
            f.write(part.content)
        print(f"Segmentation saved as: {filename}")
