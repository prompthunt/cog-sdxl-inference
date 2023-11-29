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

        print(f"Generated watermark service URL: {watermark_service_url}")

        response = requests.get(watermark_service_url)
        print(f"Watermark service response status: {response.status_code}")

        if response.status_code != 200:
            print(
                f"Failed to get watermarked image. Status code: {response.status_code}, Response: {response.text}"
            )
            raise Exception("Error getting watermarked image")

        watermarked_image_data = response.content
        print(
            f"Received watermarked image data. Size: {len(watermarked_image_data)} bytes"
        )

        uploaded_image_url = upload_to_cloudflare(
            id, watermarked_image_data, cloudflare_account, cloudflare_api_key
        )
        print(f"Uploaded to Cloudflare. URL: {uploaded_image_url}")

        return uploaded_image_url
    except Exception as error:
        print(f"Error in get_watermarked_image: {error}")
        raise


def handle_cloudflare_error(error, id):
    # Implement error handling here
    pass
