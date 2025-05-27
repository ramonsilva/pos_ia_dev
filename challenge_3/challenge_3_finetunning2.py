from datasets import load_dataset
import torch
import os
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

try:
    from transformers import BitsAndBytesConfig

    QUANTIZATION_AVAILABLE = True
except ImportError:
    print("⚠️ bitsandbytes não instalado - quantização não disponível")
    QUANTIZATION_AVAILABLE = False
from datasets import Dataset
import gc

# Configurações de memória
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'


def setup_model_and_tokenizer(model_name="meta-llama/Llama-2-7b-hf", use_quantization=True):
    """Configura modelo e tokenizer com otimizações de memória"""

    # Carrega tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Detecta dispositivo e ajusta dtype
    if torch.cuda.is_available():
        torch_dtype = torch.float16
        device_map = "auto"
    elif torch.backends.mps.is_available():
        torch_dtype = torch.float32  # MPS funciona melhor com float32
        device_map = "mps"
    else:
        torch_dtype = torch.float32
        device_map = "cpu"

    # Configuração com ou sem quantização
    if use_quantization and torch.cuda.is_available():  # Quantização apenas para CUDA
        try:
            # Configuração de quantização (4-bit)
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16
            )

            # Carrega modelo com quantização
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=True
            )
            print("Modelo carregado com quantização 4-bit")

        except ImportError:
            print("bitsandbytes não disponível, carregando sem quantização...")
            use_quantization = False

    if not use_quantization or not torch.cuda.is_available():
        # Carrega modelo sem quantização (mais compatível)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print(f"Modelo carregado sem quantização ({torch_dtype})")

    return model, tokenizer


def setup_lora_config():
    """Configura LoRA para fine-tuning eficiente"""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,  # IMPORTANTE: deve ser False para treinamento
        r=8,  # Rank - menor = mais eficiente
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",  # Adiciona configuração de bias
        use_rslora=False,  # Desabilita RSLoRA para compatibilidade
    )


def tokenize_function_optimized(examples, tokenizer, max_length=512):
    """Função de tokenização otimizada para instruction tuning"""

    # Processa em lotes menores para economizar memória
    batch_size = min(100, len(examples["prompt"]))

    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for i in range(0, len(examples["prompt"]), batch_size):
        batch_prompts = examples["prompt"][i:i + batch_size]
        batch_completions = examples["completion"][i:i + batch_size]

        for prompt, completion in zip(batch_prompts, batch_completions):
            # Combina prompt e completion
            full_text = f"{prompt} {completion}{tokenizer.eos_token}"

            # Tokeniza
            tokenized = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
                return_attention_mask=True
            )

            # Para instruction tuning, vamos mascarar apenas o prompt
            prompt_length = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])

            labels = tokenized["input_ids"].copy()
            # Mascara tokens do prompt (-100 = ignorado na loss)
            for j in range(min(prompt_length, len(labels))):
                labels[j] = -100

            all_input_ids.append(tokenized["input_ids"])
            all_attention_masks.append(tokenized["attention_mask"])
            all_labels.append(labels)

    return {
        "input_ids": all_input_ids,
        "attention_mask": all_attention_masks,
        "labels": all_labels
    }


def create_training_args():
    """Cria argumentos de treinamento otimizados baseado no dispositivo"""

    # Detecta dispositivo disponível
    if torch.cuda.is_available():
        device_type = "cuda"
        use_fp16 = True
        use_bf16 = False
        pin_memory = True
    elif torch.backends.mps.is_available():
        device_type = "mps"
        use_fp16 = False  # MPS não suporta fp16 mixed precision
        use_bf16 = False  # MPS não suporta bf16
        pin_memory = False
    else:
        device_type = "cpu"
        use_fp16 = False
        use_bf16 = False
        pin_memory = False

    print(f" Configurando para dispositivo: {device_type}")
    if use_fp16:
        print("Usando FP16 precision")
    else:
        print("Usando FP32 precision (sem mixed precision)")

    return TrainingArguments(
        output_dir="./llama-finetuned",

        # Configurações de batch e memória
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=16,  # Batch efetivo = 16

        # Otimizações de memória (ajustadas por dispositivo)
        gradient_checkpointing=True,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_pin_memory=pin_memory,

        # Configurações de treinamento
        num_train_epochs=3,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,

        # Salvamento e logging
        save_strategy="steps",
        save_steps=500,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,

        # Otimizações adicionais
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        # Configurações específicas para MPS
        use_mps_device=(device_type == "mps"),

        # Para DeepSpeed (opcional - apenas CUDA)
        # deepspeed="zero_stage2_config.json" if device_type == "cuda" else None
    )


def train_llama_efficiently(dataset, model_name="meta-llama/Llama-2-7b-hf", use_quantization=None):
    """Função principal para treinamento eficiente do Llama"""

    # Auto-detecta se deve usar quantização
    if use_quantization is None:
        use_quantization = QUANTIZATION_AVAILABLE

    print("Configurando modelo e tokenizer...")
    model, tokenizer = setup_model_and_tokenizer(model_name, use_quantization)

    print("Configurando LoRA...")
    lora_config = setup_lora_config()
    model = get_peft_model(model, lora_config)

    # IMPORTANTE: Habilitar gradientes para parâmetros LoRA
    print(" Habilitando gradientes para treinamento...")
    model.train()  # Coloca em modo de treinamento

    # Verificar se parâmetros LoRA têm gradientes habilitados
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"Parâmetros treináveis: {trainable_params:,} de {all_param:,} ({100 * trainable_params / all_param:.2f}%)")

    if trainable_params == 0:
        raise ValueError("Nenhum parâmetro configurado para treinamento! Verifique a configuração LoRA.")

    print("Tokenizando dataset...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function_optimized(x, tokenizer),
        batched=True,
        batch_size=100,
        remove_columns=dataset["train"].column_names
    )

    print("Configurando treinamento...")
    training_args = create_training_args()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    print("Iniciando treinamento...")

    # Limpa cache antes do treinamento
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # Treina
    trainer.train()

    # Salva modelo final
    print("Salvando modelo...")
    trainer.save_model()
    tokenizer.save_pretrained("./llama-finetuned")

    return trainer


# Exemplo de uso
if __name__ == "__main__":
    # Supondo que você tem seu dataset
    # dataset = your_tokenized_dataset
    dataset = load_dataset('json', data_files={'train': 'finetuning_data/train_llama.jsonl',
                                               'validation': 'finetuning_data/val_llama.jsonl'})

    # Treina o modelo
    trainer = train_llama_efficiently(dataset, use_quantization=False)

    print("Script configurado! Use train_llama_efficiently(dataset) para treinar.")


# Configurações adicionais para máxima eficiência
def setup_memory_efficient_training():
    """Configurações extras para economizar memória"""

    # Limita threads para evitar overhead
    torch.set_num_threads(4)

    # Configurações de ambiente
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["OMP_NUM_THREADS"] = "4"

    # Para sistemas com pouca RAM
    import gc
    gc.collect()

