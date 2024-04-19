import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    PeftModel,
)
import bitsandbytes as bnb
from utils.initital_system import load_global_schema
from utils.config import TaskEnum, MAX_MEMORY, RETRAIN_QLORA, LORA_DIR, LORA_RANK


def find_all_linear_names(model):
    cls = (
        bnb.nn.Linear4bit
    )  # if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class CodeLlaMa:
    def __init__(self, base_model_name):
        self.SCRIPT = {
            "sql_query": """<s>[INST] Sinh ra câu sql từ câu hỏi tương ứng với schema được cung cấp [/INST] ###schema: {sample[f'schema_{level}']}, ###câu hỏi: {sample[f'question_{level}']}, ###câu sql: {sample[f'query_{level}']}</s>"""
        }
        self.base_model_name = base_model_name

        ### LLM config
        self.model = None
        self.bnb_config = None
        self.n_gpus = torch.cuda.device_count()
        self.max_memory = f"{MAX_MEMORY}MB"
        self.retrain_qlora = RETRAIN_QLORA
        self.lora_dir = LORA_DIR
        self.lora_rank = LORA_RANK

    def generate(self, input):
        if self.model is None:
            print("For the first time, loading model is waiting more time")
            self.load_model()
        return self.model

    def load_model(self):
        print("Loading model...")
        # set quantization configuration to load large model with less GPU memory
        # this requires the bitsandbytes library
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model_name,
            quantization_config=self.bnb_config,
            device_map="auto",  # dispatch efficiently the model on the available ressources
            max_memory={i: self.max_memory for i in range(self.n_gpus)},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_name, use_auth_token=True
        )
        # tokenizer = BloomTokenizerFast.from_pretrained(model_name, use_auth_token=True)
        ### Needed for LLaMA tokenizer
        self.pad_token = self.eos_token

        if self.retrain_qlora is True:
            peft_model_id_visquad_lora = self.lora_dir
            self.model = PeftModel.from_pretrained(
                self.model, peft_model_id_visquad_lora, is_trainable=True
            )

        modules = find_all_linear_names(self.model)

        if self.lora_rank != 0:
            self.lora_config = LoraConfig(
                r=self.lora_rank,  # dimension of the updated matrices
                lora_alpha=256,  # parameter for scaling
                target_modules=modules,
                lora_dropout=0.1,  # dropout probability for layers
                bias="none",
                task_type="CAUSAL_LM",
            )

            # 1 - Enabling gradient checkpointing to reduce memory usage during fine-tuning
        self.model.gradient_checkpointing_enable()

        if self.retrain_qlora is False:
            # 2 - Using the prepare_model_for_kbit_training method from PEFT
            self.model = prepare_model_for_kbit_training(self.model)
            # 3 - apply config
            try:
                # If config is exist
                self.model = get_peft_model(self.model, self.lora_config)
                self.model.print_trainable_parameters()
            except:
                pass
        else:
            self.model.enable_input_require_grads()
            self.model.print_trainable_parameters()
            # 4 - # Print information about the percentage of trainable parameters
        print("Load model completed")

    def response(
        self, question: str, schema: str = "GLOBAL_SCHEMA", table_string: str = None
    ) -> str:
        if table_string is not None:
            response = self.generate(self.SCRIPT.format(table_string, question))
            return response
        if schema == "GLOBAL_SCHEMA":
            schema = load_global_schema()
        response = self.generate(self.SCRIPT.format(schema, question))
        # postprocess
        return response

    def classify(self, question: str, filename: str) -> str:
        # LLM response combine with post process
        print("Classifying")
        return TaskEnum.SQL_QUERY

    def plot_chart(self, question: str, table: pd.DataFrame) -> str:
        print("Ploting chart")
        return ""

    def sql_query(self, question: str, schema: str):

        prompt = f"""[INST] Given an input question, generate a SQL query by choosing one or multiple of the following tables. If not available directly, infer the primary and foreign keys from the schema information above. [/INST] Answer the following question with the context below: ###Context: {schema}, ###Question: {question}, ###Answer: """
        encodeds = tokenizer(prompt, return_tensors="pt")

        generated_ids = model.generate(
            inputs=encodeds["input_ids"].to("cuda"),
            attention_mask=encodeds["attention_mask"],
            do_sample=False,
            temperature=0.1,
            top_k=1,
            max_new_tokens=100,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        decoded = tokenizer.batch_decode(generated_ids)

        return decoded[0]
