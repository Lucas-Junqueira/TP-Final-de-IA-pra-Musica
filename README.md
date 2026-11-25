# Geração de piano pop condicionada por emoção com *classifier-free guidance*

Trabalho final de **IA para Geração de Música** (UFMG, 2026/2).

O sistema gera trechos de piano pop em MIDI (16–32 compassos) condicionados a um dos quatro quadrantes emocionais de Russell:

| Quadrante | Emoção |
|---|---|
| Q1 | feliz |
| Q2 | tensa |
| Q3 | triste |
| Q4 | calma |

O estudo mede como o **classifier-free guidance (CFG)**, aplicado só na amostragem, afeta três coisas: a aderência à emoção, a qualidade e a diversidade. Valência e arousal são analisados separadamente.

> **Pergunta de pesquisa:** quanto o CFG aumenta a aderência à emoção, quanto isso custa em qualidade e diversidade, e o efeito é diferente para valência e arousal?

---

## Resultados principais

Cada sistema gerou 100 amostras por quadrante, com a mesma semente, temperatura 1,0 e top-p 0,95. A aderência é medida por um classificador de emoção treinado só no treino do EMOPIA+. A qualidade é a *overlap area* (OA) média das métricas do muspy contra o teste humano do mesmo quadrante.

| Sistema | 4 classes | Valência | Arousal | OA médio | Diversidade |
|---|---|---|---|---|---|
| Teste humano (teto do classificador) | 0,580 | 0,693 | 0,841 | 0,847¹ | 0,525 |
| Markov ordem 4 | 0,860² | 0,880 | 0,973 | 0,480 | 0,233 |
| LSTM 2×512 | 0,403 | 0,510 | 0,790 | 0,717 | 0,538 |
| Transformer γ = 1 | 0,340 | 0,562 | 0,615 | 0,670 | 0,640 |
| Transformer γ = 2 | 0,438 | 0,580 | 0,738 | 0,666 | 0,642 |
| Transformer γ = 3 | 0,463 | 0,588 | 0,797 | 0,677 | 0,648 |
| **Transformer γ = 5** | **0,557** | 0,618 | 0,905 | **0,687** | 0,640 |
| Transformer γ = 10 | 0,583 | 0,623 | 0,955 | 0,678 | 0,623 |

¹ OA do treino humano contra o teste humano: a referência de "melhor possível".
² Acima do teto: o Markov reproduz traços superficiais que o classificador usa, mas tem a pior qualidade.

**Hipóteses do plano:**

| Hipótese | Resultado | Evidência |
|---|---|---|
| **H1**: o CFG aumenta a aderência | **Confirmada** | 4 classes de 0,34 para 0,56 (γ = 5, p < 10⁻⁴); satura a partir de γ ≈ 5 |
| **H2**: a qualidade e a diversidade caem a partir de algum γ | **Fraca** | sem custo até γ = 5; em γ = 8–10, sinais leves (KL sobe, mais tokens descartados, diversidade cai um pouco), mas o OA fica estável |
| **H3**: o arousal é mais fácil de controlar; o CFG ajuda mais a valência | **Metade** | o arousal vai de 0,62 a 0,96, mas a valência quase não se move (0,56 → 0,62, n.s.) |
| **H4**: o Transformer supera o Markov e a LSTM | **Em parte** | menor perplexidade no teste (3,59 contra 4,69 da LSTM) e bem melhor que o Markov; OA um pouco abaixo da LSTM |

**Por que o CFG controla o arousal e não a valência** (`results/figures/fig4_attributes.png`):

- Com γ maior, Q1 e Q2 ficam mais densos, mais fortes e mais rápidos; Q3 e Q4, mais esparsos, suaves e com notas longas. São justamente os atributos de arousal, e chegam a passar dos valores humanos em γ ≥ 8.
- A fração de peças em modo menor não muda com γ (~0,3–0,4 em todos os quadrantes). No teste humano ela vai de 0,16 (Q1/Q4) a 0,74 (Q2).
- O modelo não aprendeu a associação entre modo e emoção, e o CFG só amplifica o que o modelo já sabe.
- Consequência: com γ alto, Q1 (feliz) é quase sempre classificado como Q2 (tenso).

Todos os números, tabelas e figuras estão em [`results/`](results/) e são gerados por scripts.

---

## Estrutura

```
Data/download.sh      baixa EMOPIA+ e Pop1k7 (com checagem md5)
configs/              hiperparâmetros (YAML) de cada treino
src/overlap.py        sobreposição EMOPIA+ × Pop1k7 por conteúdo (protege o teste)
src/tokenize.py       MIDI → tokens REMI (miditok) + tokens especiais + janelas
src/dataset.py        Dataset/DataLoader: recortes, transposição ±3, condition dropout
src/model.py          Transformer decoder e LSTM em PyTorch puro (+ cache KV)
src/train.py          pré-treino e ajuste fino (AdamW, warmup + cosseno, AMP, early stopping)
src/sample.py         temperatura, top-k, top-p, CFG e a geração das amostras
src/markov.py         baseline de Markov (uma cadeia por quadrante) + filtro de gramática REMI
src/classifier.py     classificador de emoção (métrica de aderência)
src/evaluate.py       aderência, muspy + OA/KL, diversidade, memorização, perplexidade, atributos, figuras
src/render.py         áudio das amostras com FluidSynth
tests/                testes automatizados (pytest)
runs/                 checkpoints e logs de treino (gerados)
samples/              MIDIs gerados (gerados)
results/              tabelas, métricas e figuras (gerados)
```

---

## Como reproduzir

### 1. Ambiente

Python 3.12. Em Windows, crie o ambiente virtual.

```bash
python -m venv ~/venvs/tpfinal
source ~/venvs/tpfinal/Scripts/activate      # Windows (Git Bash); no Linux: bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # GPU; omita para CPU
pip install -r requirements.txt
python -m pytest tests                        # ~20 s em CPU
```

Rode tudo **a partir da raiz, como módulo** (`python -m src.x`). O arquivo `src/tokenize.py` esconderia o módulo `tokenize` da biblioteca padrão se fosse rodado como script.

### 2. Dados

```bash
bash Data/download.sh        # precisa de wget
python -m src.overlap        # lista as peças do Pop1k7 a excluir (Data/pop1k7_exclude.txt)
python -m src.tokenize       # Data/tokens/*.pkl + teste de ida e volta em samples/roundtrip/
```

### 3. Treino

Os tempos são de uma RTX 3050 Laptop (4 GB) com bf16.

```bash
python -m src.train configs/transformer_pretrain.yaml
python -m src.train configs/transformer_finetune.yaml   
python -m src.train configs/lstm_pretrain.yaml          
python -m src.train configs/lstm_finetune.yaml           
python -m src.markov --order 4                          
python -m src.classifier --target quadrant              # classificador de avaliação
```

Para retomar um treino interrompido, acrescente `--resume`. Para testar o pipeline inteiro com um modelo minúsculo, use `configs/tiny_pretrain.yaml` e `configs/tiny_finetune.yaml`.

### 4. Geração

```bash
python -m src.sample --systems markov lstm transformer          # γ ∈ {1; 1,5; 2; 3; 5}, ~1,5 h
python -m src.sample --systems transformer --gammas 8 10        # extensão de γ, ~40 min
```

Saída em `samples/<execução>/`: um MIDI por amostra (`Q{k}_{i}.mid`), mais `tokens.pkl` e `summary.json`.

### 5. Avaliação

```bash
python -m src.evaluate      # as métricas por amostra ficam em cache em results/cache/
```

| Arquivo | Conteúdo |
|---|---|
| `results/tables.md`, `results/table1.csv` | Tabela 1, acurácia por quadrante, OA por métrica, atributos, perplexidade |
| `results/metrics.json` | todos os números (incluindo IC de 95% e p-valores contra γ = 1) |
| `results/figures/fig1_tradeoff.png` | Figura 1: aderência × qualidade, por γ |
| `results/figures/fig2_valence_arousal.png` | Figura 2: valência e arousal em função de γ |
| `results/figures/fig3_loss_curves.png` | Figura 3: curvas de perda |
| `results/figures/fig4_attributes.png` | atributos musicais por quadrante em função de γ |
| `results/figures/confusion_matrices.png`, `oa_by_metric.png`, `pianoroll_*.png` | análises complementares |

### 6. Áudio

Precisa do [FluidSynth](https://github.com/FluidSynth/fluidsynth/releases). No Windows, baixe o `.zip` win10-x64 e indique o executável. O soundfont padrão é o MuseScore General (licença MIT), baixado automaticamente pelo muspy.

```bash
export FLUIDSYNTH="C:/caminho/para/fluidsynth/bin/fluidsynth.exe"   # ou --fluidsynth
python -m src.render        # ~1 min
```

O script grava `samples/audio/<sistema>/Q{k}_{i}.flac`, com o MIDI correspondente ao lado, e um `manifest.json`.
- Sistemas padrão: Transformer com γ = 5 e γ = 1, Markov e teste humano. Para outros, use `--systems`; os áudios atuais também incluem a LSTM.
- São as **primeiras** 2 amostras de cada quadrante, sem escolha manual, cortadas em 16 compassos (`--n`, `--bars`).

---

## Decisões de implementação

- **Tokens:** REMI com 16 posições por compasso, notas 21–108, 16 níveis de velocity e 32 de andamento, só peças 4/4. O vocabulário tem 193 tokens.
- **Formato das sequências:** `<BOS> <Q1..Q4|NULL> eventos... <EOS>`.
- **Transformer:** 6 camadas, d_model 256, 8 cabeças, contexto 1024, 5,05 M parâmetros.
  - Pré-treino no Pop1k7, sempre com `<NULL>`.
  - Ajuste fino no EMOPIA+ com condition dropout p = 0,15, que permite o CFG.
- **CFG:** `l = l_NULL + γ·(l_emoção − l_NULL)`, com as duas passadas no mesmo lote, seguido de temperatura e top-p.
- **Avaliação:** a qualidade é medida nos primeiros 16 compassos de cada amostra, porque os trechos humanos têm ~14 compassos.

## Limitações

- O classificador acerta 58% em 4 classes no teste humano, que tem só 88 trechos. O Markov e o CFG com γ alto mostram que o gerador pode explorar os traços que o classificador usa; um teste de audição confirmaria os resultados.
- Os modelos foram treinados com uma única semente.
- A memorização é medida por n-gramas exatos de 16 tokens e não detecta cópias transpostas.
- Há uma diferença grande entre os datasets: o modelo só pré-treinado tem perplexidade 18,7 no EMOPIA+.

## Dados e licenças

- [EMOPIA+](https://zenodo.org/records/13122742) (CC BY-NC-SA 4.0).
- [Pop1k7](https://zenodo.org/records/13167761), do *Compound Word Transformer*.
- Os dados não são versionados; baixe-os com `Data/download.sh`.

## Uso de IA

Ferramentas de IA generativa foram usadas como apoio neste trabalho, para:

- revisão dos textos escritos;
- documentação do código;
- pesquisa de informações;
- correção de bugs.

Todo o conteúdo produzido com apoio de IA foi revisado e validado pelos integrantes do grupo.

## Integrantes

- Cecília Junqueira V. M. Pereira
- Daniel Barreto Torres
- Erik Soares de Souza
- Felipe Lopes Gomide
- Leonardo Romano Andrade
- Lucas Junqueira Carvalhido