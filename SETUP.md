# Guia de Configuração e Primeira Execução

Este guia assume que você já extraiu o projeto na sua máquina (não no sandbox de
desenvolvimento) e está com um terminal aberto na raiz do projeto
(`tcc-table-recognition/`).

**Recomendação:** na primeira vez, siga só a trilha do **TableFormer** (Passo 2A +
Passo 3, pulando TabStructNet) com `sample_size` pequeno num dataset só
(sugestão: SciTSR, o mais simples de baixar). Se isso rodar, o resto é repetir o
mesmo padrão.

---

## Passo 1 — Monte os 4 ambientes virtuais

Os 4 algoritmos têm dependências que **não cabem no mesmo Python** — TabStructNet
exige uma versão bem mais antiga e SLANet usa PaddlePaddle. Ver `README.md` (seção
"Por que ambientes separados por modelo?") para o detalhe de por que isso é
necessário, não só implicância de versão.

| Ambiente | Python exigido | Observação |
|---|---|---|
| `.venv-tableformer` | 3.10+ | pode ser o Python normal da sua máquina |
| `.venv-tabletransformer` | 3.10+ | idem |
| `.venv-tabstructnet` | **3.7** | TensorFlow 1.15.5 e as anotações adiadas do projeto |
| `.venv-slanet` | 3.10+ | PaddlePaddle 3.0.0 CPU e PaddleOCR 2.10.0 PP-StructureV2 |

No Windows, use um instalador oficial do Python 3.7 para o ambiente TabStructNet. O
`uv` não distribui builds standalone dessa versão no Windows. A rotina de inferência
mantém dependências incompatíveis separadas e usa anotações adiadas no código comum
para permitir carregar os runners legados.

O OCR exige também o executável Tesseract, não apenas o pacote `pytesseract`. No
Windows, instale Tesseract OCR, adicione sua pasta ao `PATH` e confirme com
`tesseract --list-langs` que os idiomas definidos em `text_recovery.ocr_language`
estão instalados (por padrão `eng` e `por`). Se ele não estiver no `PATH`, defina
`text_recovery.tesseract_cmd` no YAML ou a variável `TESSERACT_CMD`, por exemplo
`C:\Program Files\Tesseract-OCR\tesseract.exe`.

Para cada ambiente Python 3.7, instale o conjunto comum compatível antes das
dependências específicas do modelo:

```powershell
py -3.7 -m venv .venv-tabstructnet
.venv-tabstructnet\Scripts\python.exe -m pip install -r environment\requirements-legacy-common.txt
.venv-tabstructnet\Scripts\python.exe -m pip install -r environment\requirements-tabstructnet.txt

py -3.11 -m venv .venv-slanet
.venv-slanet\Scripts\python.exe -m pip install -r environment\requirements-slanet.txt
```

TabStructNet requer `protobuf==3.20.3` e `numpy==1.18.5` para TensorFlow 1.15.5. O
`pycocotools==2.0.7` fornece um wheel Windows compatível; não instale outro pacote
`mrcnn` do PyPI, pois a implementação necessária já está em
`third_party/TabStructNet/mrcnn/`.

SLANet usa PaddleOCR 2.10.0/PP-StructureV2 com PaddlePaddle 3.0.0 em CPU. O
primeiro smoke test baixa automaticamente os modelos de inferência para OCR e
reconhecimento de tabela, além de alguns modelos auxiliares que PP-Structure inicializa
mesmo com layout e fórmulas desativados; confirme acesso aos servidores PaddleOCR e
espaço em disco antes de executar.

---

## Passo 2 — Baixe os checkpoints

| Modelo | O que fazer | Onde colocar |
|---|---|---|
| **TableFormer** | nada manual — baixa sozinho na 1ª execução via `huggingface_hub` | (automático, cache do HF) |
| **Table Transformer** | nada manual — mesma coisa | (automático) |
| **TabStructNet** | checkpoint `mask_rcnn_tab_0100.h5` (SciTSR) já presente nesta workspace; se precisar baixá-lo em outra máquina, o link está em `third_party/TabStructNet/trained_model/tab/logs/tab20200821T0923/note.txt` | `third_party/TabStructNet/trained_model/tab/logs/tab20200821T0923/` |
| **SLANet** | baixado automaticamente pelo PaddleOCR PP-StructureV2 conforme `language: en` e `table_algorithm: SLANet` | cache do PaddleOCR; não requer download manual |

---

## Passo 3 — Baixe os datasets que for usar

Você não precisa baixar os 4 — só os que for efetivamente testar. Para um teste
rápido, baixe só um.

| Dataset | Onde conseguir | Onde colocar | Estrutura esperada |
|---|---|---|---|
| **PubTabNet** | repositório oficial (`ibm-aur-nlp/PubTabNet`, ver link na licença do dataset) | `data/raw/pubtabnet/` | o `.jsonl` de anotação + imagens (por split) direto nessa pasta |
| **FinTabNet** | distribuição oficial IBM | `data/raw/fintabnet/` | mesmo padrão do PubTabNet |
| **SciTSR** | `github.com/Academic-Hammer/SciTSR` | `data/raw/scitsr/` | precisa ter `img/`, `structure/` e `chunk/` dentro |
| **Corporativo** | seus próprios PDFs | `data/corporate/` (PDFs) + `data/ground_truth/corporate/` (gabarito) | ver Passo 3B abaixo |

Depois de baixar, ajuste `configs/experiment_config.yaml` se o nome do arquivo de
anotação baixado for diferente do padrão (`annotations_filename`), e defina
`sample_size: 5` (ou outro número pequeno) no dataset que for testar primeiro, para
não esperar o dataset inteiro rodar antes de confirmar que está tudo funcionando.

### Passo 3B — Preparando o conjunto corporativo

Diferente dos datasets públicos, aqui não existe anotação pronta — são 2 etapas:

```bash
# Detecta, recorta e padroniza as páginas selecionadas no YAML.
# Windows:
.venv-tabletransformer/Scripts/python.exe -m src.data_prep.extract_table_regions
# Linux/macOS: .venv-tabletransformer/bin/python -m src.data_prep.extract_table_regions
```

O comando lê `table_detection.pages_by_pdf`, gera PNGs RGB padronizados em
`data/processed/corporate/` e grava `manifest.csv` com página, caixa original,
caixa com padding, confidence, status e nome do recorte. Detecções de tabelas
rotacionadas são registradas como `excluded_rotated`, nunca entram silenciosamente
na inferência; páginas sem detecção também ficam registradas. Para adicionar PDFs,
inclua-os em `data/corporate/` e adicione-os a `table_detection.pages_by_pdf` em
`configs/experiment_config.yaml`. Quando a lista existe, somente os PDFs listados
são processados; defina `pages_by_pdf: null` para processar todos os PDFs e páginas.

Os gabaritos dos dois recortes atuais já estão em `data/ground_truth/corporate/`.
Para novas imagens, crie o JSON correspondente com o mesmo nome-base e o formato
documentado em `src/data_prep/dataset_loaders/corporate_loader.py`. Valide os pares
antes de executar modelos:

```bash
python -m src.data_prep.validate_datasets --dataset corporate
```

---

## Passo 4 — Rode, um modelo por vez

**Nunca rode mais de um modelo no mesmo comando** — é por isso que existem 4 venvs
separados.

Antes do dataset completo, faça o smoke test de uma amostra. No conjunto corporativo,
a primeira imagem tem cabeçalho hierárquico e células mescladas; o comando informa uma
prévia limitada da saída bruta e as dimensões/células normalizadas, sem gravar métricas
ou planilhas:

```bash
# Windows (repita para cada modelo depois de instalar seu venv e checkpoint).
.venv-tableformer\Scripts\python.exe -m src.run_experiment --model tableformer --dataset corporate --smoke-test
.venv-tabletransformer\Scripts\python.exe -m src.run_experiment --model tabletransformer --dataset corporate --smoke-test
.venv-tabstructnet\Scripts\python.exe -m src.run_experiment --model tabstructnet --dataset corporate --smoke-test
.venv-slanet\Scripts\python.exe -m src.run_experiment --model slanet --dataset corporate --smoke-test
```

Só prossiga para a execução normal depois que cada smoke test mostrar uma tabela com
linhas, colunas e células. Os caminhos `checkpoint_path`, `variant`, `repo_dir` e parâmetros específicos
(`score_threshold`, `language`, `device`, `table_algorithm`) são lidos de
`configs/experiment_config.yaml` pelo runner correspondente.

### Benchmark pareado e inspeção das planilhas

Para executar todos os modelos nos mesmos splits e amostras limitadas pelo YAML,
use um identificador novo para cada rodada:

```powershell
.venv-tableformer\Scripts\python.exe -m src.evaluation.benchmark --run-id paired-20261022
```

O comando congela a lista exata de imagens em `sample_manifest.csv`, executa cada
modelo em seu venv em processos separados e registra hashes do código/configuração,
checkpoint, versão Python, `pip freeze`, parâmetros, comandos e logs. Uma falha de
um runner não apaga os resultados dos demais. Sob
`results/benchmarks/<run-id>/` ficam as métricas por amostra, resumo por
dataset/modelo, comparações pareadas, auditoria XLSX, metadados e logs. Médias,
desvios-padrão amostrais, intervalos bootstrap e teste exato pareado são
exploratórios com os tamanhos pequenos definidos neste experimento; não sustentam
conclusões de superioridade sem mais tabelas.

`xlsx_audit.csv` registra dimensões, texto preenchido, posições vazias, mesclagens,
células em negrito e uma prévia do conteúdo, para identificar planilhas que são
estruturalmente válidas mas não úteis para ETL.

Depois, rode cada modelo no seu ambiente como abaixo:

```bash
# Windows:
.venv-tableformer\Scripts\python.exe -m src.run_experiment --model tableformer --dataset scitsr
.venv-tabstructnet\Scripts\python.exe -m src.run_experiment --model tabstructnet --dataset scitsr
.venv-tabletransformer\Scripts\python.exe -m src.run_experiment --model tabletransformer --dataset scitsr
.venv-slanet\Scripts\python.exe -m src.run_experiment --model slanet --dataset scitsr

# Linux/macOS:
.venv-tableformer/bin/python -m src.run_experiment --model tableformer --dataset scitsr
.venv-tabstructnet/bin/python -m src.run_experiment --model tabstructnet --dataset scitsr
.venv-tabletransformer/bin/python -m src.run_experiment --model tabletransformer --dataset scitsr
.venv-slanet/bin/python -m src.run_experiment --model slanet --dataset scitsr
```

Tire o `--dataset scitsr` quando quiser rodar em todos os datasets habilitados no
config de uma vez.

---

## Passo 5 — Onde olhar o resultado

- `results/metrics/<dataset>_<modelo>.csv` — TEDS, S-TEDS e F1 por célula, por amostra.
- `results/spreadsheets/<dataset>/<modelo>/*.xlsx` — a tabela reconstruída de cada
  amostra, pronta para abrir no Excel.

---

## Se algo der errado

- **Erro de import ao rodar um modelo**: confira se você ativou/apontou para o venv
  certo (`.venv-<modelo>/bin/python`, não o Python do sistema).
- **TabStructNet reclamando de versão de `numpy`/`opencv`**: esse ambiente é sensível
  à versão exata (ver comentários em `requirements-tabstructnet.txt`).
- **Falha ao baixar modelos SLANet/OCR**: confira acesso ao servidor de modelos
  PaddleOCR; os arquivos são obtidos automaticamente pelo PP-StructureV2.
- Qualquer outra dúvida específica de cada modelo tem uma nota detalhada na
  docstring do respectivo arquivo em `src/models/<modelo>_runner.py`.

---

## Painel local de utilização

O painel exibe as execuções já registradas, resumos estatísticos, comparações
pareadas, resultados por atributo e a auditoria textual/estrutural dos XLSX. Também
permite validar os dados configurados ou iniciar um novo benchmark; nenhum modelo
é executado automaticamente ao abrir a página.

Inicie-o a partir da raiz do projeto usando o ambiente TableFormer (Python 3.10+):

```powershell
.venv-tableformer\Scripts\python.exe -m src.web_app
```

Abra `http://127.0.0.1:8765`. O serviço escuta apenas no computador local. Para
validar as imagens e os gabaritos antes de executar, clique em **Validar dados
agora** e consulte a tarefa/log exibido no painel. Para uma comparação, selecione
os runners e clique em **Iniciar nova execução**. O painel usa os splits, tamanhos
de amostra e parâmetros atuais de `configs/experiment_config.yaml`, iniciando cada
runner no venv correspondente. Uma nova execução nunca reutiliza um identificador
existente; os relatórios ficam em `results/benchmarks/<run-id>/`.
