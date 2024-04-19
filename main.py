import argparse
from huggingface_hub.hf_api import HfFolder
from utils.helper_functions import get_excel_schema, read_table_file
from utils.config import TaskEnum, HF_TOKEN, BASE_MODEL_NAME
from models.CodeLlaMa import CodeLlaMa
from utils.initital_system import load_global_schema


def run(**kwargs):
    parser = argparse.ArgumentParser(description="Process filename and question.")
    parser.add_argument("--filename", help="Name of the file to process.", default=None)
    parser.add_argument("--question", help="The question to process.")

    args = parser.parse_args()

    assert isinstance(kwargs["CodeLlaMa"], CodeLlaMa), "CodeLlaMa not found"

    ################################
    # Load LLM

    ################################
    # Classifier
    task = kwargs["CodeLlaMa"].classify(question=args.question, filename=args.filename)

    if args.filename is not None:
        assert (
            len(args.filename.split(".")) == 2
        ), "Any file must contain filename and extension."
        filename, extension = args.filename.split(".")

    if task is TaskEnum.SQL_QUERY:
        if args.filename is None:
            schema = load_global_schema()
            print(kwargs["CodeLlaMa"].sql_query(question=args.question, schema=schema))
        elif extension in ["xlsx", "csv"]:
            schema = get_excel_schema(args.filename)
            print(kwargs["CodeLlaMa"].sql_query(schema=schema, question=args.question))

    elif task is TaskEnum.GENERAL_QA:
        if extension in ["png", "jpg"]:
            table_string = kwargs["MATCHA"].response()
            kwargs["CodeLlaMa"].response(
                question=args.question, table_string=table_string
            )
        # Run langchain
        pass

    elif task is TaskEnum.PLOT_CHART:
        if extension in ["xlsx", "csv"]:
            df = read_table_file(args.filename)
            kwargs["CodeLlaMa"].plot_chart(question=args.question, table=df)
        else:
            print("Table not found!")


def main():
    # init global schema by importing database

    HfFolder.save_token(HF_TOKEN["TeeA"])
    # load MATCHA
    # laod LLM
    codellama_model = CodeLlaMa(base_model_name=BASE_MODEL_NAME)
    # while True:
    run(CodeLlaMa=codellama_model)


if __name__ == "__main__":
    main()
