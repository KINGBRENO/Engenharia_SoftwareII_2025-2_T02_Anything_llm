import os
import sys
import subprocess
import time

# ===========================================
# CONFIGURAÇÃO DOS SCRIPTS DOS MODELOS
# ===========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

os.makedirs(os.path.join(BASE_DIR, "resultados"), exist_ok=True)
MODEL_SCRIPTS = {
    "1": ("BART (bart-large-cnn + bart-large-mnli)", os.path.join(BASE_DIR, "Modelos/BART_CNN+MNLI.py")), # OK
    "2": ("LLaMA", "/Modelos/llama.py"),
    "3": ("Qwen", "/Modelos/qwen.py"),
    "4": ("Mistral", "/Modelos/mistral.py"),
    "5": ("T5", "/Modelos/modelo_t5.py"),
    "0": ("Sair", None)
}

# ===========================================
# FUNÇÕES
# ===========================================

def limpar_terminal():
    os.system("cls" if os.name == "nt" else "clear")


def mostrar_menu():
    print("=======================================")
    print("        SELECIONE O MODELO LLM         ")
    print("=======================================\n")

    for key, (nome, _) in MODEL_SCRIPTS.items():
        print(f" {key} — {nome}")

    print("\n=======================================\n")


def executar_modelo(script):
    if not os.path.exists(script):
        print(f"\n❌ ERRO: O script '{script}' não foi encontrado!")
        return

    print(f"\n🚀 Executando modelo via: {script}\n")
    time.sleep(1)

    try:
        subprocess.run([sys.executable, script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ O modelo encontrou um erro:\n{e}")
    except Exception as e:
        print(f"\n❌ Falha inesperada ao executar o modelo:\n{e}")


# ===========================================
# PROGRAMA PRINCIPAL
# ===========================================

def main():
    while True:
        limpar_terminal()
        mostrar_menu()
        escolha = input("👉 Digite o número da opção: ").strip()

        if escolha not in MODEL_SCRIPTS:
            print("\n⚠️ Opção inválida, tente novamente...")
            time.sleep(1.5)
            continue

        nome, script = MODEL_SCRIPTS[escolha]

        if escolha == "0":
            print("\n👋 Encerrando...")
            break

        print(f"\nModelo selecionado: {nome}\n")
        executar_modelo(script)

        input("\nPressione ENTER para voltar ao menu...")


if __name__ == "__main__":
    main()
