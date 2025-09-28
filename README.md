# Trabalho 1 - Redes Neurais (NES)
Autor: Mariana Lins dos Santos 
Professor: Eduardo Adame  
Tema: CIFAR-10 (1 canal - R/G/B)

---

## 📌 Descrição
Este projeto implementa e compara um modelo **MLP baseline** e um **CNN** no dataset CIFAR-10 (com apenas 1 canal de cor).  
Inclui análise de overfitting, regularização (Dropout/L2), Grad-CAM, matriz de confusão e aplicação interativa via **Streamlit**.

---

## 📂 Estrutura do repositório
- `notebooks/T11_CIFAR10.ipynb`: notebook principal com exploração, MLP e CNN.  
- `src/models.py`: definição das arquiteturas.  
- `src/train.py`: rotina de treino/teste com salvamento dos modelos.  
- `src/app.py`: aplicação interativa em Streamlit.  
- `figures/`: figuras usadas no relatório (curvas, matrizes, Grad-CAM).  
- `requirements.txt`: dependências do projeto.  

---

## Detalhes de Treino
- Épocas: 30 (default, EarlyStopping com paciência 5)
- Sementes: fixadas (padrão 42)
- Parâmetros: CNN ≈ X mil, MLP ≈ Y mil (obtido via `model.summary()`)

## 🚀 Instalação
```bash
pip install -r requirements.txt
