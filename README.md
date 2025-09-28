# Trabalho 1: Avaliação 1 - Redes Neurais (NES) UFAL - Módulo 4

Autor: Mariana Lins dos Santos
Professor: Eduardo Adame
Tema: CIFAR-10 (1 canal - R/G/B)

---

## 📌 Descrição

Este projeto implementa e compara um modelo **MLP baseline** e uma **CNN** no dataset CIFAR-10 (utilizando apenas 1 canal de cor).
Inclui:

* análise de overfitting
* regularização (Dropout/L2)
* Grad-CAM para interpretabilidade
* matriz de confusão e análise de erros
* aplicação interativa via **Streamlit**

---

## 📂 Estrutura do repositório

* `notebooks/T11_CIFAR10.ipynb`: notebook principal com exploração, MLP e CNN.
* `src/models.py`: definição das arquiteturas.
* `src/train.py`: rotina de treino/teste com salvamento dos modelos.
* `src/viz.py`: funções para curvas, matriz de confusão e Grad-CAM.
* `src/app.py`: aplicação interativa em Streamlit.
* `figures/`: figuras usadas no relatório (curvas, matrizes, Grad-CAM).
* `requirements.txt`: dependências do projeto.

---

## ⚙️ Detalhes de Treino

* Épocas: **30** (default, com EarlyStopping paciência = 5)
* Sementes: **fixadas** (42, 123, 999)
* Parâmetros: **MLP ≈ 150 mil** | **CNN ≈ 545 mil** (ambos < 1M, conforme restrição do enunciado)

---

## 🚀 Instalação

```bash
pip install -r requirements.txt
```

---

## 🏋️‍♂️ Treinamento

Treinar o **MLP**:

```bash
python src/train.py --model mlp --epochs 30 --seed 42
```

Treinar a **CNN**:

```bash
python src/train.py --model cnn --epochs 30 --seed 42
```

---

## 💻 Interface Interativa (Streamlit)

Após o treino, rode a aplicação:

```bash
streamlit run src/app.py
```

Funcionalidades:

* Upload de imagens externas
* Exibição das probabilidades por classe
* Visualização do Grad-CAM

---

## 📊 Resultados Principais

* A CNN supera o MLP em acurácia média (≈ 99,1% vs 98,0%)
* Grad-CAM mostra regiões relevantes para classificação
* Matriz de confusão evidencia confusões entre classes semelhantes

---

### 📊 Visualizações
Funções de visualização estão em `src/viz.py`. Exemplos:

```python
from src.viz import plot_curves, plot_confusion_matrix, plot_gradcam
