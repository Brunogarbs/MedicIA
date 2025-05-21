# MedicIA
MedicIA é um chatbot desenvolvido para auxiliar na triagem médica, oferecendo suporte inicial ao identificar possíveis condições de saúde com base nos sintomas fornecidos pelo usuário.

## 🧠 Sobre o Projeto
O objetivo do MedicIA é fornecer uma ferramenta acessível que possa orientar usuários na identificação preliminar de possíveis doenças, facilitando a tomada de decisão sobre buscar atendimento médico profissional.

## ⚙️ Tecnologias Utilizadas
- Python
- Bibliotecas de Processamento de Linguagem Natural (NLP)
- Modelos de Machine Learning
- TF-IDF para vetorização de texto

## 📁 Estrutura do Projeto
- main.py: Arquivo principal que executa o chatbot.
- treinamento_2.py: Script utilizado para treinar o modelo de machine learning.
- dataset_expansao.csv: Conjunto de dados contendo sintomas e diagnósticos utilizados para treinamento.
- modelo_sintomas.pkl: Modelo treinado salvo para uso no chatbot.
- vectorizer_tfidf.pkl: Vetorizador TF-IDF salvo para transformar entradas de texto.
- requirements.txt: Lista de dependências necessárias para executar o projeto.

## 🚀 Como Executar
1. Clone o repositório:
```
git clone https://github.com/Brunogarbs/MedicIA.git
cd MedicIA
```
2. Crie um ambiente virtual (opcional, mas recomendado):
```
python -m venv venv
source venv/bin/activate 
```
3. Instale as dependências:
```
pip install -r requirements.txt
```
4. Execute o chatbot:
```
python main.py
```
## 🧪 Como Treinar o Modelo
Caso deseje treinar o modelo novamente com o conjunto de dados fornecido:
```
python treinamento_2.py
```
Isso gerará os arquivos modelo_sintomas.pkl e vectorizer_tfidf.pkl atualizados.

