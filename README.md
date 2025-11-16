# Anything-LLM – Análise de Padrões Arquiteturais

## 📌 1. Equipe
Abaixo está a tabela detalhada de contribuição dos integrantes:

| Nome                                   | Matrícula    | Contribuição                                                                                                |
| -------------------------------------- | ------------ | ----------------------------------------------------------------------------------------------------------- |
| **Breno Henrique Do Carmo Santos**     | 202200078737 | Pesquisa, teste de modelos, padronização dos modelos, elaboração do tutorial e edição do vídeo e documento. |
| **Carla Stefany R. Santos**            | 202400060148 | Pesquisa, teste de modelos, elaboração do tutorial, edição e formatação do documento.                       |
| **Fernanda Karoliny Santos Silva**     | 202200092431 | Pesquisa, edição do documento, formatação do documento.                                                     |
| **João Paulo Menezes Machado**         | 202300038743 | Pesquisa, análise do projeto, edição do documento.                                                          |
| **João Vinícius De Almeida Argolo**    | 202200025573 | Pesquisa, teste de modelos, elaboração do tutorial e edição do documento.                                   |
| **José Arthur Calixto Da Rocha Costa** | 202300038770 | Pesquisa, teste de modelos, elaboração do tutorial e edição do documento.                                   |
| **Vinícius Azevedo Peroba**            | 201900076892 | Pesquisa, análise do projeto, apuração dos testes, edição do documento.                                     |
| **Wendel Alexsander Gomes Menezes**    | 202300027740 | Pesquisa, análise do projeto, apuração dos testes, edição do documento.                                     |

Este repositório contém a implementação completa da Atividade 1 de Engenharia de Software II (2025.2), cujo objetivo é **analisar padrões arquiteturais ao longo da evolução de um projeto real do GitHub utilizando modelos de linguagem (LLMs)**.

A equipe desenvolveu um pipeline automatizado capaz de:
- Baixar e analisar um repositório alvo;
- Processar seu conteúdo (código-fonte, documentação e estrutura);
- Executar **5 modelos do Hugging Face** para identificar padrões arquiteturais;
- Comparar a eficácia dos modelos;
- Gerar resultados em txt.

---
## ▶️ Vídeo Da apresentação da equipe:
- https://drive.google.com/drive/folders/1k_cFVe-c416gxxkyP5GDc-b5KLNITyY1?usp=sharing



## 📌 2. Projeto Analisado

Este estudo utiliza como alvo o repositório:
> **[Anything-LLM](https://github.com/Mintplex-Labs/anything-llm)**

O Anything LLM é uma plataforma open source que permite criar assistentes de IA capazes de conversar com usuários e compreender documentos, integrando grandes modelos de linguagem com bases de conhecimento locais. Ela transforma arquivos e textos em dados pesquisáveis, possibilitando que o chatbot responda com base nesses conteúdos e até execute tarefas automatizadas por meio de agentes de IA. Pode ser usada localmente ou via servidor, oferecendo flexibilidade, privacidade e personalização para empresas e desenvolvedores que desejam construir seus próprios sistemas de chat inteligentes.

O projeto foi escolhido por apresentar:
- Estrutura modular clara;
- Utilização de diversas camadas arquiteturais;
- Evolução consistente ao longo do tempo;
- Relação direta com processamento de linguagem natural.

---

## 📌 3. Estrutura do Repositório
```
├── Data/ # Scripts dos modelos indivídualizados no google colabutilizados no Colab
│   ├── Modelo_bart_large_cnn_e_bart_large_mnli.ipynb
│   ├── bart_karge_mnli.ipynb
│   └── modelo_Qwen.ipynb
│
├── entradas/            # Arquivos de entrada utilizados pela equipe
│
├── src/                 # Código-fonte principal
│   ├── Modelos/         # Scripts dos modelos utilizados
│   └── RunModels.py     # Execução central dos modelos
│
├── resultados/          # Resultados gerados pelos modelos
│   ├── BART_CNN+MNLI.txt
│   ├── Bart_MNLI.txt
│   ├── Llama.txt
│   ├── Qwen.txt
│   └── all_minilm_l6_v2.txt
│
├── docs/                # Documentação do projeto
│   └── Breno_Henrique_Carla_Stefany_Fernanda_Karoliny_João_Paulo_José_Arthur_Vinicius_Azevedo_Wendel_Alexsander_Atividade_1.pdf
│
├── requirements.txt     # Dependências do projeto
└── README.md            # Este documento

```
## 📌 4. Modelos Utilizados
O estudo utilizou **cinco modelos** do Hugging Face para ampliar a diversidade de análise:

Modelos utilizados:
- **bart-large-cnn** – sumarização e auxílio na compreensão estrutural
- **bart-large-mnli** – classificação e verificação de correspondência entre padrões
- **Llama-3.2-1B-Instruct** – análise arquitetural orientada a prompts
- **Qwen2.5-Coder-1.5B-Instruct** – excelente para análise de arquivos de código
- **all-MiniLM-L6-v2** – geração de embeddings para similaridade e agrupamento

Cada modelo recebeu como entrada fragmentos do projeto alvo e produziu como saída:
- Classes detectadas
- Categorias (Controller, Service, Repository, Utils...)
- Justificativa textual
- Relações arquiteturais

Os resultados estão disponíveis em `resultados`.

## 📌 5. Requisitos do Sistema
### ✔️ Dependências
Instale todas as dependências com:
```
pip install -r requirements.txt
```
Principais libs:
- transformers
- sentence-transformers
- torch
- accelerate
- bitsandbytes
- tqdm
- scikit-learn
- GitPython

### ✔️ Infraestrutura
Este projeto foi executado com os seguintes recursos:
- GPU: **NVIDIA GeForce RTX 3070 (8 GB VRAM)**
- CPU: **Ryzen 7 5700x3d**
- RAM: **32 GB**
- Ambiente Python 3.12

Os modelos foram rodados individualmente no google Colab utilizando um plano gratuito com a seguinte configuração:

- System RAM: 12.7 GB
- GPU: T4 GPU
- GPU RAM: 15.0 GB
- Ambiente Python 3.12

> A infraestrutura é importante pois modelos maiores podem estourar a memória em máquinas com menos recursos

---

## 📌 6. Como Executar o Projeto
### **1️⃣ Clonar o repositório**
```
git clone https://github.com/FernandaKaroliny/Engenharia_SoftwareII_2025-2_T02_Anything_llm
cd Engenharia_SoftwareII_2025-2_T02_Anything_llm
```

### **2️⃣ Instalar dependências**
```
pip install -r requirements.txt
```

### **3️⃣ Rodar o pipeline principal**
```
python src/run.py
```

Este script realiza:
1. Clonagem do projeto alvo;
2. Extração de arquivos relevantes;
3. Processamento e limpeza;
4. Execução dos modelos Hugging Face;
5. Salvamento dos resultados em `resultados/`.

---

## 📌 7. Resultados
Os resultados são gerados automaticamente após a escolha do modelo:
- `resultados` → Lista de padrões arquiteturais detectados

Além disso, o PDF do tutorial contém uma análise aprofundada, incluindo:
- Comportamento dos modelos
- Vantagens e limitações

---

## 📌 8. Tutorial (Documento Escrito)
O arquivo PDF completo está em:
```
docs/tutorial.pdf
```
Ele contém:
- Introdução
- Fundamentação Teórica
- Metodologia
- Resultados e Análise
- Resultados por Modelo
- Tutorial Individual por modelo
- Discussão dos padrões identificados
- Conclusões



---


