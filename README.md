# Classificação de Spam com Máquina de Vetores de Suporte (SVM)
```
⠀⠀⢀⣀⠤⠿⢤⢖⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⡔⢩⠂⠀⠒⠗⠈⠀⠉⠢⠄⣀⠠⠤⠄⠒⢖⡒⢒⠂⠤⢄⠀⠀⠀⠀
⠇⠤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠈⠀⠈⠈⡨⢀⠡⡪⠢⡀⠀
⠈⠒⠀⠤⠤⣄⡆⡂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠢⠀⢕⠱⠀
⠀⠀⠀⠀⠀⠈⢳⣐⡐⠐⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠁⠇                    <- n detecta isso kkkkkkkkkkkkkkkkkkkk
⠀⠀⠀⠀⠀⠀⠀⠑⢤⢁⠀⠆⠀⠀⠀⠀⠀⢀⢰⠀⠀⠀⡀⢄⡜⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠘⡦⠄⡷⠢⠤⠤⠤⠤⢬⢈⡇⢠⣈⣰⠎⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⣃⢸⡇⠀⠀⠀⠀⠀⠈⢪⢀⣺⡅⢈⠆⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠶⡿⠤⠚⠁⠀⠀⠀⢀⣠⡤⢺⣥⠟⢡⠃⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠉⠀⠀⠀⠀⠀⠀⠀⠀
```

A ideia desse projeto foi bem simples: usar uma Máquina de Vetores de Suporte (SVM) pra classificar e-mails como spam ou não spam. O dataset que usei é o famoso Spambase, que tem um monte de características numéricas extraídas de e-mails, como a frequência de palavras como "free" e "money", ou a quantidade de caracteres especiais tipo "$" e "!". Basicamente, o modelo tenta aprender a diferença entre um e-mail normal e aqueles chatos que ficam enchendo sua caixa de entrada.

O legal é que a SVM é um modelo bem robusto e consegue lidar bem com dados de alta dimensionalidade (e o Spambase tem 57 características, então é um bom teste). No final, o modelo alcançou um F1-Score médio de 92,76%, o que é bem legal, considerando que o dataset tem um leve desbalanceamento (mais e-mails normais do que spam).

O que tem aqui?
Só tem um arquivo principal, o main.py, que é onde toda a mágica acontece. Nele, você vai encontrar o código pra carregar o dataset, treinar o modelo SVM e avaliar o desempenho. Não tem muita frescura, é tudo bem direto ao ponto.

### Como rodar o código?

Primeiro, clone o repositório:

```bash
git clone https://github.com/seu-usuario/nome-do-repositorio.git
```

Depois, instale as dependências (se você já tem o Python instalado, é só rodar):

```bash
pip install scikit-learn pandas numpy
```

Agora é só rodar o código:

```bash
python main.py
```

E pronto! O código vai treinar o modelo e te mostrar as métricas de desempenho, como acurácia, precisão, revocação e F1-Score. Se você quiser brincar com os parâmetros da SVM, é só mexer no código e ver como o modelo se comporta.

### Referências

Se você quiser saber mais sobre o dataset ou a SVM, dá uma olhada nesses links:

- [Spambase no UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/spambase)
- [Documentação do Scikit-learn](https://scikit-learn.org/stable/)

