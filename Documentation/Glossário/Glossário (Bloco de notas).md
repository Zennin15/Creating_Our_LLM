**CAPÍTULO 1:**



**Termos Técnicos**



1\. AI (Artificial Intelligence): Campo de pesquisa que estuda máquinas que conseguem resolver problemas que requerem inteligência semelhante ao do ser humano.



2\. Autoregressive Models: Modelos que incorporam saídas anteriores como entradas para as futuras predições.



3\. Classification Fine-Tuning: Subcategoria do fine-tuning onde é apresentado textos rotulados.



4\. Deep Learning: Subcampo do Machine Learning onde se faz o uso de redes neurais (três ou mais) para aprendizado profundo de máquina.



5\. Expert Systems: Programa de computador que usa a inteligência artificial para imitar a capacidade de decisão de um especialista humano.



6\. Fine-Tuning: Segunda etapa do treinamento de uma LLM, onde os dados utilizados são mais específicos para o contexto em que a LLM será inserida.



7\. Fuzzy Logic: Forma de lógica multivalorada em que os valores de verdade variam entre 0 e 1, permitindo representar graus de verdade parcial em vez de apenas verdadeiro ou falsos, imitando o raciocínio humano para lidar com imprecisões e conceitos vagos.



8\. Genetic Algorithms: Método computacional para a resolução de problemas baseado na seleção artificial e biologia.



9\. Instruction Fine-Tuning: Subcategoria do fine-tuning onde a LLM é alimentada com uma instrução para realizar uma tarefa e o resultado esperado ao completá-la.



10\. LLM (Large Language Model): É uma rede neural criada para compreender e responder textos humanos.



11\. Machine Learning: Subcampo da AI que estuda os algoritmos usados para implementar uma AI.



12\. NLP: Natural Language Processing.



13\. Parâmetros: Pesos ajustáveis em uma rede neural que são otimizados durante o treinamento para prever a próxima palavra em uma sequência.



14\. Pretraining: Primeira etapa do treinamento de uma LLM, onde é utilizado um grande volume de dados brutos sem rótulos.



15\. Rule-Based Systems: É um programa de AI que usa condições (if-then) humanas para armazenar conhecimento, resolver problemas e tomar decisões.



16\. Self-Attention Mechanism: Permite ao modelo de IA pesar a importância de diferentes palavras ou tokens em uma sequência.



17\. Symbolic Reasoning: Conceito da inteligência artificial simbólica onde problemas, lógica e informações são representadas com símbolos e regras humanas explícitas.



18\. Tokens: Unidade de texto que um modelo lê.



19\. Transformer: Arquitetura de uma rede neural.



**Palavras Estrangeiras**



1\. Encompasses: Abrange, envolve, inclui.



2\. Harnesses: Aproveita.



3\. narrower: mais estreito.



4\. Tailored: sob medida.



5\. Underpins: Sustenta, suporta, mantém.



6\. Ushered: Inaugurou, conduziu.



**CAPÍTULO 2:**



1. Byte Pair Encoding (Codificação de Pares de Bytes): Algoritmo que implementa o processo de tokenização. Utilizado por modelos GPT.



Função: Codificar (token -> ID) 

&#x09;Exemplo de código: tiktoken.get\_encoding("gpt2")

&#x09;

&#x09;Decodificar (ID -> token)

&#x09;Exemplo de código: tiktoken.get\_decoding("gpt2")



2\. Embedding (incorporação): Processo onde convertemos dados em um vetor representante. Existem diversos tipos de embedding, o que será utilizado durante o projeto será o word embedding. 



Função: Modelos de rede neural, como as LLMs, não conseguem processar dados brutos diretamente. Os dados brutos são incompatíveis com as operações matemáticas usadas para implementar e treinar as redes neurais. Por isso, transformamos os dados em vetores com valores decimais representando os dados de interesse. 



3\. Input-Target pairs (Pares de Entrada-Alvo): Dados de entrada e alvo no processo de treinamento de uma LLM.



Função: São usados para o modelo prever a próxima palavra em uma sequência (target) dado um entrada (input). Os dados que estão além do target são mascarados, forçando o modelo prever uma palavra por vez.  



4\. Tokenizing (tokenização): Processo onde transformamos todas as palavras e caracteres especiais (como sinais de pontuação) de um texto de entrada em tokens individuais.



Função: Os tokens individuais podem ser posteriormente mapeados com token IDs, uma etapa necessária para o embedding. Além disso, LLMs GPT-like são treinadas palavra por palavra, sendo assim a separação em tokens permite ao modelo identificar separadamente cada palavra e caractere em um texto de entrada, facilitando o entendimento do contexto e a predição da próxima palavra na sequência.



5\. Token ID: Número único de identificação de um token individual.



Função: Por serem valores inteiros, são transformados em um vetor pelo método embedding. 



6\. Vocabulary (vocabulário): Um vocabulário onde são mapeados (ou relacionados) os tokens e seus respectivos IDs, semelhante a um dicionário.



Função: Serve como uma tabela ou lista onde o modelo pode consultar o ID de um token e vice-versa.



7\. Word embeddings (incorporação de palavra): Tipo de embedding onde utilizamos dados textuais que são utilizados para o treinamento de LLMs.



Função: São utilizados para treinar uma LLM com dados textuais transformados em vetores representantes.  





&#x20;





