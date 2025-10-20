from pathlib import Path
from langchain_community.document_loaders import JSONLoader, PyPDFLoader, NotebookLoader, CSVLoader

def load_data(data_dir: str):
    data_path = Path(data_dir).resolve()
    print(f'Data Path: {data_dir}')
    docs = []

    # PDF Files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f'Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}')
    for file in pdf_files:
        print(f'Loading PDF {file}')
        try:
            loader = PyPDFLoader(str(file))
            loaded = loader.load()
            for page in loaded:
                page.metadata['source_file'] = file.name
                page.metadata['file_type'] = 'pdf'
            print(f'Loaded PDF: {file}')
            docs.extend(loaded)
        except Exception as e:
            print(f'Cound not load PDF : {file}, {e}')
            raise

    # JSON
    json_files = list(data_path.glob('**/*.json'))
    print(f'Found {len(json_files)} JSON files: {[str(f) for f in json_files]}')
    for file in json_files:
        print(f'Loading JSON {file}')
        try:
            # loader = JSONLoader(
            #     file_path=file,
            #     jq_schema='.',
            #     text_content=False
            # )
            loader = JSONLoader(str(file), jq_schema='.')
            loaded = loader.load()
            for page in loaded:
                page.metadata['source_file'] = file.name
                page.metadata['file_type'] = 'json'
            print(f'Loaded JSON: {file}')
            docs.extend(loaded)
        except Exception as e:
            print(f'Cound not load JSON : {file}, {e}')
            raise
    
    # IPYNB
    notebook_files = list(data_path.glob('**/*.ipynb'))
    print(f'Found {len(notebook_files)} ipynb files: {[str(f) for f in notebook_files]}')
    for file in notebook_files:
        print(f'Loading ipynb {file}')
        try:
            # loader = NotebookLoader(
            #     file,
            #     include_outputs=True,
            #     max_output_length=1000,
            #     remove_newline=True,
            # )
            loader = NotebookLoader(str(file))
            loaded = loader.load()
            for page in loaded:
                page.metadata['source_file'] = file.name
                page.metadata['file_type'] = 'ipynb'
            print(f'Loaded ipynb: {file}')
            docs.extend(loaded)
        except Exception as e:
            print(f'Cound not load ipynb : {file}, {e}')
            raise
    
    # CSV
    csv_files = list(data_path.glob('**/*.csv'))
    print(f'Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}')
    for file in csv_files:
        print(f'Loading CSV {file}')
        try:
            # loader = CSVLoader(
            #     file_path=file,
            #     csv_args={
            #         'delimiter': ',',
            #         'quotechar': '"',
            #     },
            #     encoding='latin-1'
            # )
            loader = CSVLoader(str(file))
            loaded = loader.load()
            for page in loaded:
                page.metadata['source_file'] = file.name
                page.metadata['file_type'] = 'csv'
            print(f'Loaded CSV: {file}')
            docs.extend(loaded)
        except Exception as e:
            print(f'Cound not load CSV : {file}, {e}')
            raise

    return docs