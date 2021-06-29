import re
import time
import tempfile

from google.cloud import storage
from google.cloud import vision
from google.cloud import texttospeech
from google.protobuf import json_format

vision_client = vision.ImageAnnotatorClient()
storage_client = storage.Client()
speech_client = texttospeech.TextToSpeechClient()


def p2a_gcs_trigger(file, context):
    print("Started `p2a_gcs_trigger` function")

    file_name = file["name"]  # use in prod
    print("file name: {}".format(file_name))
    # file_name = "MY_SAMPLE_FILE_NAME" # use locally to test
    if not file_name.lower().endswith(".pdf"):
        print("Finished `p2a_gcs_trigger` function because file is not pdf")
        return
    bucket = None
    file_blob = None
    while bucket == None or file_blob == None:  # retry
        bucket = storage_client.get_bucket(file["bucket"])  # use in prod
        # bucket = storage_client.get_bucket("MY_AWESOME_BUCKET_NAME") # use locally to test
        file_blob = bucket.get_blob(file_name)
        time.sleep(1)

    # PDF to TEXT
    text_list_from_pdf = None
    # if file_name.lower().endswith(".pdf"):
    text_list_from_pdf = p2a_pdf_to_text(bucket, file_blob)

    # Convert TEXT to SPEECH
    # while text_list_from_pdf == None:
    #     text_list_from_pdf = p2a_pdf_to_text(bucket, file_blob)
    #     time.sleep(1)
    print("before text to speech")
    p2a_text_to_speech(bucket, file_name, text_list_from_pdf)
    print("after text to speech")

    print("Finished `p2a_gcs_trigger` function")


def p2a_pdf_to_text(bucket, pdf_blob):
    """Converts PDF file provided to TEXT"""
    """Returns a list of strings, each string is a page"""
    print("Started `p2a_pdf_to_text` function")

    # Supported mime_types are: 'application/pdf' and 'image/tiff'
    mime_type = 'application/pdf'
    # How many pages should be grouped into each json output file
    batch_size = 100

    # Feature to use, in this case `DOCUMENT_TEXT_DETECTION`
    feature = vision.types.Feature(
        type=vision.enums.Feature.Type.DOCUMENT_TEXT_DETECTION)

    # Config for input
    gcs_source_uri = "gs://{}/{}".format(bucket.name, pdf_blob.name)
    gcs_source = vision.types.GcsSource(uri=gcs_source_uri)
    input_config = vision.types.InputConfig(
        gcs_source=gcs_source, mime_type=mime_type)

    # Config for output
    pdf_id = pdf_blob.name.replace(".pdf", "")
    gcs_destination_uri = "gs://{}/{}".format(bucket.name, pdf_id + ".")
    gcs_destination = vision.types.GcsDestination(uri=gcs_destination_uri)
    output_config = vision.types.OutputConfig(
        gcs_destination=gcs_destination, batch_size=batch_size)

    # Call the API
    async_request = vision.types.AsyncAnnotateFileRequest(
        features=[feature], input_config=input_config, output_config=output_config)
    async_response = vision_client.async_batch_annotate_files(requests=[
                                                              async_request])
    print("Started OCR for file {}".format(pdf_blob.name))
    async_response.result(timeout=540)
    print("Finished OCR for file {}".format(pdf_blob.name))

    # Prepare downloading `json` file of that contains response
    match = re.match(r'gs://([^/]+)/(.+)', gcs_destination_uri)
    prefix = match.group(2)
    blob_to_download = bucket.get_blob(
        pdf_blob.name.replace(".pdf", ".output-1-to-2.json"))

    # Download `json` file and decode the file
    print('Started downloading blob as JSON string')
    json_string = blob_to_download.download_as_string()
    print('Finished downloading blob as JSON string')
    print('Started decoding JSON string')
    decoded_json = json_format.Parse(
        json_string, vision.types.AnnotateFileResponse())
    print('Finished decoding JSON string')

    # Combine all pages into one string
    full_text_pages_list = []
    all_pages_reponses = decoded_json.responses
    for page_response in all_pages_reponses:
        pageText = page_response.full_text_annotation.text
        full_text_pages_list.append(pageText)
        print(u'Added page number {} to the list'.format(
            page_response.context.page_number))

    # `return` TEXT of PDF
    print("Finished `p2a_pdf_to_text` function")
    return full_text_pages_list


def p2a_text_to_speech(bucket, file_name, text_list_from_pdf):
    print("Started `p2a_text_to_speech` function")
    # Setup
    speech_file_name = file_name.replace('.pdf', '.mp3')
    temp_dir = tempfile.gettempdir()
    temp_speech_file_name = 'temp_output.mp3'
    temp_speech_file_path = '{}/{}'.format(temp_dir, temp_speech_file_name)

    # Convert list of text(strings) to one text(string)
    print('Started converting the list to one string')
    text = ' '.join(text_list_from_pdf)
    print('Finished converting the list to one string')

    # Set the text input to be synthesized
    synthesis_input = texttospeech.types.SynthesisInput(text=text)

    # Build the voice request
    voice = texttospeech.types.VoiceSelectionParams(
        language_code="en-US"
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.types.AudioConfig(
        audio_encoding=texttospeech.enums.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    print('Started converting the string to speech')
    response = speech_client.synthesize_speech(
        synthesis_input, voice, audio_config
    )
    print('Finished converting the string to speech')

    # Saving `.mp3` file to google storage bucket
    print('Started saving binary to .mp3 file')
    # The response's audio_content is binary.
    with open(temp_speech_file_path, "wb") as out:
        # Write the response to the output file.
        out.write(response.audio_content)
        print(u'Audio content written to file "{}"'.format(temp_speech_file_path))
    print('Finished saving binary to .mp3 file')

    # Upload `.mp3` file to google storage bucket
    print('Started uploading .mp3 file to goog storage bucket')
    speech_blob = bucket.blob(speech_file_name)
    with open(temp_speech_file_path, "rb") as speech_file:
        speech_blob.upload_from_file(speech_file)
    print('Finished uploading .mp3 file to goog storage bucket')

    print("Finished `p2a_text_to_speech` function")
