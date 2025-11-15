import torch
import psutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import listarestrutura as lt
import time
import os
from torch.cuda import OutOfMemoryError

# Contador de prompts processados
prompts_testados = 0

# --------------------------------------------------------------
# Função de limpeza segura de CUDA (só executa se CUDA existir)
# --------------------------------------------------------------
def limpar_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# Limpa GPU uma vez no início
limpar_cuda()

# --------------------------------------------------------------
# Exibir uso de memória
# --------------------------------------------------------------
def mostrar_uso_memoria():
    processo = psutil.Process()
    mem_info = processo.memory_info().rss / (1024 ** 2)
    print(f" Memória RAM usada: {mem_info:.2f} MB")

    if device == "cuda":
        memoria_gpu_usada = torch.cuda.memory_allocated(0) / (1024 ** 2)
        memoria_gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"Memória GPU usada: {memoria_gpu_usada:.2f} MB / {memoria_gpu_total:.2f} MB totais")

# --------------------------------------------------------------
# Executa o modelo com um prompt (com fallback em OOM)
# --------------------------------------------------------------
def executarModelo(input_text, max_new_tokens=350):
    global prompts_testados
    prompts_testados += 1

    # Prepara entrada (cria tensores e move para o device do modelo)
    inputs = tokenizer(input_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    limpar_cuda()

    # tentativa principal
    try:
        tempo_execucao = time.perf_counter()
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.15,
            top_p=0.9
        )
        tempo_execucao = time.perf_counter() - tempo_execucao

    except (OutOfMemoryError, RuntimeError) as e:
        # Captura OOM e tenta fallback
        print("!! OOM detectado durante generate():", str(e))
        limpar_cuda()
        # 1) Tentar reduzir tokens e re-rodar
        try:
            print("Tentando reduzir max_new_tokens para 128 e reexecutar...")
            tempo_execucao = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=min(128, max_new_tokens),
                temperature=0.15,
                top_p=0.9
            )
            tempo_execucao = time.perf_counter() - tempo_execucao
        except Exception as e2:
            print("Falha na segunda tentativa:", e2)
            limpar_cuda()
            # 2) fallback simples: reduzir o prompt a um prefixo (evita divisão complexa)
            short_input = input_text[:4000]  # pegar os primeiros caracteres
            print("Fazendo fallback com prompt reduzido (prefixo)...")
            inputs2 = tokenizer(short_input, return_tensors="pt")
            inputs2 = {k: v.to(model.device) for k, v in inputs2.items()}
            try:
                tempo_execucao = time.perf_counter()
                outputs = model.generate(
                    **inputs2,
                    max_new_tokens=64,
                    temperature=0.15,
                    top_p=0.9
                )
                tempo_execucao = time.perf_counter() - tempo_execucao
            except Exception as e3:
                # última tentativa falhou — retornar mensagem de erro
                print("Fallback final falhou:", e3)
                return f"[ERRO: geração falhou por OOM ou outro problema: {e3}]"

    # Decodifica
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Exibe relatório
    print("\n_____________________________________________")
    print(f"Prompt testado nº: {prompts_testados}")
    print(f"\nTempo de execução do modelo: {tempo_execucao:.2f} segundos")
    mostrar_uso_memoria()
    print("\n" + decoded + "\n_____________________________________________ ")

    return decoded

# --------------------------------------------------------------
# Caminho do modelo local
# --------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, "Llama-3.2-1B-Instruct")

print("Caminho do modelo:", model_path)
print("Diretório atual:", base_dir)

# --------------------------------------------------------------
# Detectar GPU
# --------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"GPU detectada: {torch.cuda.get_device_name(0)}")
else:
    print("Nenhuma GPU detectada, rodando na CPU")

# --------------------------------------------------------------
# Carregar tokenizer e modelo (QUANTIZADO 4-bit) - UMA ÚNICA VEZ
# --------------------------------------------------------------
print("Carregando modelo local...")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    local_files_only=True
)

print("Modelo carregado com sucesso!")
mostrar_uso_memoria()

# --------------------------------------------------------------
# Processar entradas (loop)
# --------------------------------------------------------------
for i in range(1, 5):
    with open(f"../entradas/entrada{i}.txt", "r", encoding="utf-8") as f:
        arquivo = f.read()

    prompt = f"""

    {arquivo}

    Com base nas dependências, scripts e estrutura, descreva:
    1. Quais tecnologias principais o projeto utiliza (ex: Node.js, React, Express, Prisma, etc.).
    2. Qual o tipo de arquitetura que o projeto segue (ex: Monolítico, MVC, Microservices, Event-Driven, Client-Server, etc.).
    3. Quais diretórios ou pacotes indicam separação de responsabilidades (ex: 'server', 'frontend', 'collector').
    4. Como ocorre o fluxo de execução em desenvolvimento (baseando-se nos scripts `dev:*` e `setup`).
    5. Se o projeto parece voltado para execução local, em nuvem, ou ambos.
    6. Quais boas práticas ou convenções de arquitetura são evidentes (ex: uso de Prisma para ORM, divisão entre back e front, uso de Docker, etc.).
    7. Gere um pequeno resumo final explicando o propósito geral do projeto e seu padrão arquitetural principal.

    Responda em português de forma clara OBJETIVA, não repetitiva além de Avaliar a confiabilidade da sua resposta.
    """

    print(prompt)

    os.makedirs("respostas", exist_ok=True)
    arquivo_saida = f"respostas/resposta{i}.txt"

    resposta = executarModelo(prompt)

    with open(arquivo_saida, "w", encoding="utf-8") as f:
        f.write(resposta)
