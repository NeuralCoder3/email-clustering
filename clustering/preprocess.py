import eml_parser
import os
from tqdm import tqdm
import html2text


def preprocess(in_folder, out_folder, verbose=False):
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    parser = eml_parser.EmlParser(
        include_raw_body=True, include_attachment_data=False)
    files = os.listdir(in_folder)
    if verbose:
        print("Preprocessing E-Mails...")
        files = tqdm(files)
    for filename in files:
        if filename.endswith(".eml"):
            try:
                with open(os.path.join(in_folder, filename), 'rb') as fhdl:
                    raw_email = fhdl.read()
                parsed_eml = parser.decode_email_bytes(raw_email)
                with open(os.path.join(out_folder, filename), 'w') as fhdl:
                    fhdl.write("from: " + parsed_eml['header']['from']+"\n")
                    fhdl.write(
                        "to: " + ", ".join(parsed_eml['header']['to'])+"\n")
                    fhdl.write("subject: " +
                               parsed_eml['header']['subject']+"\n")
                    body = parsed_eml['body'][0]['content']
                    body = html2text.html2text(body)
                    fhdl.write(body)
            except Exception as e:
                print("Error parsing file: ", filename)
                continue
