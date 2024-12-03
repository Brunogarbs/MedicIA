import telebot
import time
from telebot.types import InlineKeyboardMarkup, InlineKeyboardButton
from sklearn.metrics import accuracy_score, classification_report
from decouple import config
import re
import psycopg2
import spacy
import joblib
from datetime import datetime

# Obter a data atual
data_atendimento = datetime.now()

# Função para conectar ao banco de dados
def conectar_banco():
    try:
        # Dados da conexão
        conexao = psycopg2.connect(
            dbname="MedicDB",    # Nome do banco de dados
            user="postgres",    # Usuário do banco
            password="dK7JKtOFOnaVTKHf",  # Senha do banco
            host="elusively-concrete-boxer.data-1.use1.tembo.io",       # Host (ex.: localhost ou endereço do servidor)
            port="5432"            # Porta do PostgreSQL (geralmente 5432)
        )
        print("Conexão ao banco de dados estabelecida com sucesso.")
        return conexao
    except psycopg2.Error as erro:
        print(f"Erro ao conectar ao banco de dados: {erro}")
        return None

# Conectar ao banco de dados
conexao = conectar_banco()

# Acesso a API do Telegram
bot = telebot.TeleBot('7532292572:AAFU9oBVcGGrQ3VOsEmUW-eLPAJGMPGBWtM')
grupo_id = '-1002358397560'#ID Grupo

# Dicionário para armazenar dados temporários dos usuários
user_data = {}

# Carregar o modelo e o vetor TF-IDF
modelo = joblib.load('modelo_sintomas.pkl')
vectorizer = joblib.load('vectorizer_tfidf.pkl')

# Inicializa o SpaCy com o modelo de português
nlp = spacy.load("pt_core_news_sm")

# Dicionário com tratamentos para cada doença
tratamentos = {
    "COVID-19": "Mantenha-se em isolamento, hidrate-se, e utilize máscara.",
    "Intoxicação Alimentar": "Mantenha-se hidratado e consuma alimentos leves.",
    "DENGUE": "Mantenha-se hidratado",
    "AVC": "Procure imediatamente um atendimento médico. Evite alimentos, líquidos ou medicamentos, pois isso pode agravar o quadro"
}

# Função para verificar os sintomas e sugerir um tratamento
def verificar_tratamento(sintomas_usuario):
    sintomas_processados = vectorizer.transform([sintomas_usuario])
    
    # Prever a doença
    diagnostico = modelo.predict(sintomas_processados)[0]
    
    # Obter as probabilidades de cada classe
    probabilidades = modelo.predict_proba(sintomas_processados)[0]
    confianca = max(probabilidades) * 100  # Confiança na previsão (em %)
    
    # Obter o tratamento
    tratamento = tratamentos.get(diagnostico, "Não foi possível identificar um tratamento. Consulte um médico.")
    
    return diagnostico, tratamento, confianca

# Função para extrair nomes próprios usando SpaCy
def extrair_nome(texto):
    doc = nlp(texto)
    # Lista de expressões para ignorar
    ignorar = {"olá", "oi", "e aí", "boa tarde", "bom dia", "boa noite"}
    
    # Verifica se há entidades reconhecidas pelo SpaCy
    for entidade in doc.ents:
        if entidade.label_ == "PER":  # "PER" é o rótulo para pessoas em português
            nome = entidade.text.lower().strip()  # Normaliza para comparação
            if nome not in ignorar:  
                return entidade.text
    return None

@bot.callback_query_handler(func=lambda call: True)
def botoes(call):
    if 'genero_' in call.data:
        genero = call.data.split('_')[1]
        user_id = int(call.data.split('_')[2])

        # Armazenar o gênero do usuário
        user_data[user_id]['genero'] = genero

        contato_emergencia(call.message)

    elif call.data.startswith("dor_"):
        user_id = call.from_user.id
        intensidade_dor = int(call.data.split("_")[1])
        user_data[user_id]['intensidade_dor'] = intensidade_dor
        bot.send_message(user_id, f"Você informou que a intensidade da dor é {intensidade_dor}.")
        time.sleep(3)
        bot.send_message(user_id, "Agora, poderia descrever seus sintomas com mais detalhes?")
        bot.register_next_step_handler(call.message, capturar_sintomas)

    elif call.data.startswith("urgencia_"):
        user_id = int(call.data.split("_")[2])
        urgencia = call.data.split("_")[1]
        mensagens_urgencia = {
            "vermelho": "Emergência máxima! Será atendido imediatamente!",
            "laranja": "Urgência alta. Será atendido em até 10 minutos!",
            "amarelo": "Urgência moderada. Será atendido em até 1 hora.",
            "verde": "Urgência baixa. Será atendido em até 2 horas.",
            "azul": "Não urgente. Será atendido em até 4 horas.",
        }
        
        bot.send_message(grupo_id, f"A urgência do paciente foi classificada como: {mensagens_urgencia[urgencia]}.")
        
        # Salvar os dados do atendimento, incluindo a urgência
        id_atendimento = salvar_atendimento(user_id, urgencia)
        
        bot.send_message(user_id, f"Sua classificação de urgência é: *{mensagens_urgencia[urgencia]}*. ID de atendimento: *{id_atendimento}*.", parse_mode="Markdown")
        time.sleep(4)
        bot.send_message(user_id, f"Aguarde o atendimento com o médico. Caso queira reiniciar o díalogo, digite /start ou clique aqui. 😊")

@bot.message_handler(commands=['start'])
def boas_vindas(message):
    user_id = message.from_user.id
    user_data[user_id] = {'nome': None, 'chat_id_inicial': user_id}  # Incluindo o chat_id_inicial no dicionário
    bot.send_message(message.chat.id, "Olá! Bem-vindo ao sistema de triagem do hospital. Para iniciar o atendimento, vamos coletar algumas informações.")
    time.sleep(2)
    bot.send_message(message.chat.id, "Por favor, informe seu nome completo:")
    bot.register_next_step_handler(message, armazenar_nome)

def armazenar_nome(message):
    nome = extrair_nome(message.text)
    user_id = message.from_user.id

    # Atualiza o nome e preserva o chat_id_inicial
    user_data[user_id] = user_data.get(user_id, {})
    user_data[user_id]['nome'] = nome
    user_data[user_id]['chat_id_inicial'] = message.chat.id
    if nome:
        bot.send_message(message.chat.id, f"Obrigado, *{nome}*. Agora, informe sua idade:",parse_mode="Markdown")
        bot.register_next_step_handler(message, valida_maior_de_idade)
    else:
        bot.send_message(message.chat.id, "Não consegui identificar o seu nome. Poderia digitar novamente? Com a primeira letra do seu nome em maiúsculo ou apenas seu nome.")
        bot.register_next_step_handler(message, armazenar_nome)

def valida_maior_de_idade(message):
    user_id = message.from_user.id
    try:
        idade = int(message.text)
        user_data[user_id]['idade'] = idade

        if idade < 18:
            bot.send_message(message.chat.id, "Você é menor de idade. Informe o nome do seu responsável:")
            bot.register_next_step_handler(message, responsavel)
        else:
            markup = InlineKeyboardMarkup()
            botao_masculino = InlineKeyboardButton("Masculino", callback_data=f"genero_masculino_{message.chat.id}")
            botao_feminino = InlineKeyboardButton("Feminino", callback_data=f"genero_feminino_{message.chat.id}")
            botao_outro = InlineKeyboardButton("Outro", callback_data=f"genero_outro_{message.chat.id}")
            markup.add(botao_masculino, botao_feminino, botao_outro)

            bot.send_message(message.chat.id, "Por favor, selecione seu gênero:", reply_markup=markup)

    except ValueError:
        bot.send_message(message.chat.id, "Por favor, informe um valor numérico válido para a idade.")
        bot.register_next_step_handler(message, valida_maior_de_idade)

def responsavel(message):
    user_id = message.from_user.id
    nome_responsavel = extrair_nome(message.text)
    user_data[user_id]['responsavel'] = nome_responsavel  # Armazenando o nome do responsável

    bot.send_message(message.chat.id, f"Obrigado. O responsável informado é: *{nome_responsavel}*.",parse_mode="Markdown")
    time.sleep(2)
    bot.send_message(message.chat.id, "Qual é o parentesco do responsável com você?")
    bot.register_next_step_handler(message, parentesco_responsavel)

def parentesco_responsavel(message):
    user_id = message.from_user.id
    parentesco = message.text
    user_data[user_id]['parentesco'] = parentesco  # Armazenando o parentesco

    bot.send_message(message.chat.id, f"Obrigado. O responsável é: *{user_data[user_id]['responsavel']}*, e o parentesco é: *{parentesco}*. Agora, vamos coletar mais informações.",parse_mode="Markdown")
    
    markup = InlineKeyboardMarkup()
    botao_masculino = InlineKeyboardButton("Masculino", callback_data=f"genero_masculino_{message.chat.id}")
    botao_feminino = InlineKeyboardButton("Feminino", callback_data=f"genero_feminino_{message.chat.id}")
    botao_outro = InlineKeyboardButton("Outro", callback_data=f"genero_outro_{message.chat.id}")
    markup.add(botao_masculino, botao_feminino, botao_outro)

    time.sleep(2)
    bot.send_message(message.chat.id, "Por favor, selecione seu gênero:", reply_markup=markup)  # Chama a função para coletar gênero

def contato_emergencia(message):
    bot.send_message(message.chat.id, "Por favor, informe o *NOME* do seu contato de emergência.",parse_mode="Markdown")
    bot.register_next_step_handler(message, telefone_contato_emergencia)

def telefone_contato_emergencia(message):
    user_id = message.from_user.id
    nome_contato = extrair_nome(message.text)
    user_data[user_id]['contato_emergencia'] = {'nome': nome_contato}
    bot.send_message(message.chat.id, "Agora, por favor, informe o *TELEFONE* do seu contato de emergência. _(ex: (11) 12345-6789)_", parse_mode="Markdown")
    bot.register_next_step_handler(message, salvar_contato_emergencia)

def salvar_contato_emergencia(message):
    user_id = message.from_user.id
    telefone = message.text.strip()
    
    # Expressão regular para validar o telefone no formato 
    telefone_pattern = r'^\(\d{2}\)\s\d{5}-\d{4}$'
    
    if re.match(telefone_pattern, telefone):
        # Se o telefone for válido, armazena no dicionário
        user_data[user_id]['contato_emergencia']['telefone'] = telefone
        bot.send_message(message.chat.id, "Obrigado. Agora, você poderia informar o nº da sua carteirinha SUS?")
        bot.register_next_step_handler(message, dados_sus)
    else:
        # Se o telefone não for válido, pede para o usuário tentar novamente
        bot.send_message(message.chat.id, "O telefone fornecido não é válido. Por favor, insira um telefone no formato correto (ex: (11) 12345-6789)")
        bot.register_next_step_handler(message, salvar_contato_emergencia)

def dados_sus(message):
    user_id = message.from_user.id
    num_sus = message.text
    if len(num_sus) == 15 and num_sus.isdigit():
        user_data[user_id]['numero_sus'] = num_sus
        bot.send_message(message.chat.id, "Muito obrigado! Agora vamos verificar sua dor.")
        
        markup = InlineKeyboardMarkup(row_width=5)
        botoes = [InlineKeyboardButton(str(i), callback_data=f"dor_{i}") for i in range(11)]
        markup.add(*botoes)
        time.sleep(3)
        bot.send_message(message.chat.id, "Em uma escala de 0 a 10, qual é a intensidade da sua dor?", reply_markup=markup)

    else:
        bot.send_message(message.chat.id, "Por favor, informe um número de cartão SUS válido com 15 dígitos.")
        bot.register_next_step_handler(message, dados_sus)

def capturar_sintomas(message):
    user_id = message.from_user.id
    sintomas = message.text
    user_data[user_id]['sintomas'] = sintomas
    
    # Obter previsão, tratamento e confiança
    diagnostico, tratamento, confianca = verificar_tratamento(sintomas)

    print(confianca)
    print(diagnostico)
    
    if confianca <= 80.0:
        tratamento = ("Não foi possível identificar um tratamento. Consulte um médico.")
        diagnostico = ("Não foi possivel encontrar um diagnostioco")

    user_data[user_id]['diagnostico'] = diagnostico
    user_data[user_id]['tratamento'] = tratamento

    
    # Mensagem para médicos
    mensagem_para_medicos = (
        f"⚕️ *Nova Triagem Médica*\n"
        f"👤 *Paciente*: {user_data[user_id]['nome']}\n"
        f"🧑‍🦱 *Idade*: {user_data[user_id]['idade']}\n"
        f"⚧️ *Gênero*: {user_data[user_id]['genero']}\n"
        f"📞 *Contato de Emergência*: {user_data[user_id]['contato_emergencia']['nome']} - {user_data[user_id]['contato_emergencia']['telefone']}\n"
        f"🖋 *Número do SUS*: {user_data[user_id]['numero_sus']}\n"
        f"🤕 *Intensidade da dor*: {user_data[user_id]['intensidade_dor']}\n"
        f"📋 *Sintomas*: {sintomas}\n"
        f"🔍 *Possível diagnóstico*: {diagnostico}\n"
        f"🩺 *Tratamento sugerido*: {tratamento}"
    )
    
    bot.send_message(grupo_id, mensagem_para_medicos, parse_mode="Markdown")
   

    # Enviar botões para classificar a urgência para o grupo
    markup = InlineKeyboardMarkup()
    botao_vermelho = InlineKeyboardButton("🔴 Vermelho", callback_data=f"urgencia_vermelho_{user_id}")
    botao_laranja = InlineKeyboardButton("🟠 Laranja", callback_data=f"urgencia_laranja_{user_id}")
    botao_amarelo = InlineKeyboardButton("🟡 Amarelo", callback_data=f"urgencia_amarelo_{user_id}")
    botao_verde = InlineKeyboardButton("🟢 Verde", callback_data=f"urgencia_verde_{user_id}")
    botao_azul = InlineKeyboardButton("🔵 Azul", callback_data=f"urgencia_azul_{user_id}")

    markup.add(botao_vermelho, botao_laranja, botao_amarelo, botao_verde, botao_azul)

    # Enviando os botões para o grupo, e não para o chat do usuário
    bot.send_message(grupo_id, "Médico, classifique a urgência do paciente:", reply_markup=markup)

    # Mensagem para o usuário confirmando os sintomas e o tratamento sugerido
    bot.send_message(message.chat.id, f"Você informou os seguintes sintomas: *{sintomas}.*",parse_mode="Markdown")
    time.sleep(4)
    bot.send_message(message.chat.id, f"Aqui está um tratamento preventivo:*{tratamento}*.", parse_mode="Markdown")
    time.sleep(2)
    bot.send_message(message.chat.id, "As informações foram enviadas para a classificação do médico.")
    time.sleep(2)
    bot.send_message(message.chat.id, "Aguarde um momento...")

def salvar_atendimento(user_id, urgencia):
    """Salva os dados do paciente, responsável, contato de emergência e atendimento."""
    try:
        cursor = conexao.cursor()
        
        # Dados do paciente
        nome = user_data[user_id]['nome']
        idade = user_data[user_id]['idade']
        genero = user_data[user_id]['genero']
        numero_sus = user_data[user_id]['numero_sus']
        
        # Dados do responsável (opcionais)
        responsavel = user_data[user_id].get('responsavel', 'Não Informado')
        parentesco = user_data[user_id].get('parentesco', 'Não Informado')
        
        # Dados do contato de emergência
        contato_emergencia = user_data[user_id]['contato_emergencia']['nome']
        telefone_emergencia = user_data[user_id]['contato_emergencia']['telefone']
        
        # Dados do atendimento
        chat_id = user_data[user_id]['chat_id_inicial']
        intensidade_dor = user_data[user_id]['intensidade_dor']
        sintomas = user_data[user_id]['sintomas']
        classificacao = urgencia
        tratamento = user_data[user_id]['tratamento']
        possivel_doenca = user_data[user_id]['diagnostico']

        # Inserir dados do paciente
        cursor.execute("""
            INSERT INTO pacientes (nome, idade, genero, numero_sus)
            VALUES (%s, %s, %s, %s);
        """, (nome, idade, genero, numero_sus))  # Use %s como placeholder
        
        # Obter o ID do paciente
        cursor.execute("SELECT currval(pg_get_serial_sequence('pacientes', 'id'));")
        id_paciente = cursor.fetchone()[0]

        # Inserir dados do responsável, se houver
        if responsavel and parentesco:
            cursor.execute("""
                INSERT INTO Responsaveis (nome, parentesco, paciente_id)
                VALUES (%s, %s, %s);
            """, (responsavel, parentesco, id_paciente))
        
        # Inserir dados do contato de emergência
        cursor.execute("""
            INSERT INTO ContatosEmergencia (nome, telefone, paciente_id)
            VALUES (%s, %s, %s);
        """, (contato_emergencia, telefone_emergencia, id_paciente))
        
            # SQL para inserir o atendimento e retornar o ID gerado
        sql_atendimento = """
            INSERT INTO Atendimentos (paciente_id, chat_id, intensidade_dor, sintomas, classificacao, 
                                    tratamento, possivel_doenca, data_atendimento)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
        """

        # Executar o comando e recuperar o ID
        cursor.execute(sql_atendimento, (id_paciente, chat_id, intensidade_dor, sintomas, classificacao, tratamento, possivel_doenca, data_atendimento))
        atendimento_id = cursor.fetchone()[0]
        # Confirmar transação
        conexao.commit()
        
        return atendimento_id
    except Exception as e:
        print(f"Erro ao salvar atendimento: {e}")
        conexao.rollback()
        return None


# Inicia o bot
bot.polling()
