import datetime
import itertools
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from stanza.server import CoreNLPClient
import global_options
from culture.file_util import file_to_list, line_counter
from culture.preprocess import preprocessor


def process_line(args):
    line, lineID, preprocessor = args
    try:
        sentences_processed, doc_sent_ids = preprocessor.process_document(
            line, lineID)
        return "\n".join(sentences_processed), "\n".join(doc_sent_ids)
    except Exception as e:
        print(f"Exception in line: {lineID} - {e}")
        return "", ""  # Return empty strings on exception to maintain output consistency


def process_largefile_multithreaded(input_file, output_file, input_file_ids, output_index_file, preprocessor, chunk_size, max_workers, start_index=None):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Initialize start_index if None
    start_index = start_index or 0

    try:
        with open(output_index_file, 'r') as file:
            lines = set([line[:line.find(".F")+2] for line in file])
    except FileNotFoundError:
        lines = set()
        # Ensure output files are empty if starting fresh
        open(output_file, 'w').close()
        open(output_index_file, 'w').close()

    # Update start_index based on processed documents, if not explicitly set
    if start_index == 0:
        for index, id in enumerate(input_file_ids):
            if id in lines:
                start_index = index + 1
            else:
                break

    print(
        f"Starting from index {start_index}. Processing {len(input_file_ids) - start_index} documents out of {len(input_file_ids)}")

    with open(input_file, 'r', encoding='utf-8') as f_in, \
            open(output_file, 'a', encoding='utf-8') as f_out, \
            open(output_index_file, 'a', encoding='utf-8') as f_index_out:

        for i, batch in enumerate(itertools.islice(itertools.zip_longest(*[f_in] * chunk_size), start_index // chunk_size, None)):
            lines = [x for x in batch if x is not None]
            lineIDs = input_file_ids[i * chunk_size +
                                     start_index: (i + 1) * chunk_size + start_index]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(
                    process_line, (line, lineID, preprocessor)): lineID for line, lineID in zip(lines, lineIDs)}
                for future in as_completed(futures):
                    sentences, ids = future.result()
                    if sentences and ids:
                        f_out.write(sentences + '\n')
                        f_index_out.write(ids + '\n')

            print(
                f"Processed chunk {i+1 + start_index // chunk_size} at {datetime.datetime.now()}")


if __name__ == "__main__":
    with CoreNLPClient(
        properties={
            "ner.applyFineGrained": "false",
            "annotators": "tokenize,ssplit,pos,lemma,ner,depparse",
        },
        memory=global_options.RAM_CORENLP,
        threads=global_options.N_CORES,
        timeout=12000000,
        endpoint="http://localhost:9002",
        be_quiet=True,
    ) as client:
        preprocessor_instance = preprocessor(client)
        input_file_path = Path(global_options.DATA_FOLDER,
                               "input", "documents.txt")
        output_file_path = Path(
            global_options.DATA_FOLDER, "processed", "parsed", "documents.txt")
        output_index_file_path = Path(
            global_options.DATA_FOLDER, "processed", "parsed", "document_ids.txt")
        input_file_ids = file_to_list(
            Path(global_options.DATA_FOLDER, "input", "document_ids.txt"))

        assert line_counter(input_file_path) == len(
            input_file_ids), "Input file and ID file line counts do not match."

        process_largefile_multithreaded(
            input_file=input_file_path,
            output_file=output_file_path,
            input_file_ids=input_file_ids,
            output_index_file=output_index_file_path,
            preprocessor=preprocessor_instance,
            chunk_size=global_options.PARSE_CHUNK_SIZE,
            max_workers=global_options.N_CORES,
            start_index=None  # or specify a start index if needed
        )
