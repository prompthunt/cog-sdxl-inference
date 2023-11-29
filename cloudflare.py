import requests
from requests_toolbelt.multipart.encoder import MultipartEncoder
import uuid

CLOUDFLARE_ENDPOINT = "https://api.cloudflare.com/client/v4/accounts"


def upload_to_cloudflare(id, image_path, cloudflare_account, cloudflare_api_key):
    MAX_RETRIES = 5
    retries = 0

    print("UPLOADING", id)

    while retries <= MAX_RETRIES:
        try:
            with open(image_path, "rb") as file:
                multipart_data = MultipartEncoder(
                    fields={"file": (id, file, "image/png")}
                )

                response = requests.post(
                    f"{CLOUDFLARE_ENDPOINT}/{cloudflare_account}/images/v1",
                    headers={
                        "Authorization": f"Bearer {cloudflare_api_key}",
                        "Content-Type": multipart_data.content_type,
                    },
                    data=multipart_data,
                    timeout=90,
                )

            if response.status_code != 200:
                print(response.text)
                raise Exception(response.text)

            data = response.json()

            public_variant = next(
                (v for v in data["result"]["variants"] if "public" in v), None
            )
            return public_variant or data["result"]["variants"][0]
        except Exception as error:
            if retries == MAX_RETRIES:
                print("ERROR", error)
                return handle_cloudflare_error(error, id)
            retries += 1
            print(f"Attempt {retries} failed. Retrying...")

    return ""


def get_watermarked_image(
    image_url, logo_width, cloudflare_account, cloudflare_api_key
):
    try:
        id = str(uuid.uuid4())
        watermark_service_url = f"https://watermark.picstudioimages.com/?imageUrl={image_url}&logoWidth={logo_width}"

        response = requests.get(watermark_service_url)
        if response.status_code != 200:
            raise Exception("Error getting watermarked image")

        # Assuming the watermark service returns the image directly
        # If it returns a URL or JSON, modify this part accordingly
        watermarked_image_data = response.content

        # Call to upload to Cloudflare (assuming the function can handle binary data)
        uploaded_image_url = upload_to_cloudflare(
            id,
            watermarked_image_data,
            cloudflare_account,
            cloudflare_api_key,
        )

        return uploaded_image_url
    except Exception as error:
        print(error)
        raise


def handle_cloudflare_error(error, id):
    # Implement error handling here
    pass
