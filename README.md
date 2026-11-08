# TCC — Análise de Algoritmos de Detecção de Estrutura de Tabela

Pipeline experimental para comparar algoritmos de **Table Structure Recognition (TSR)**
e **Table Recognition (TR)** — TableFormer, TabStructNet, Table Transformer e
SLANet (PaddleOCR PP-StructureV2) — quanto à qualidade da reconstrução de tabelas
em documentos não estruturados.

## Algoritmos do estudo e fontes verificadas

Cada algoritmo abaixo foi confirmado por inspeção direta do código-fonte oficial
(clone raso dos repositórios), não apenas pelo paper — útil para citar nas Seções 3/4
do TCC com segurança sobre o que de fato está disponível publicamente:

| Algoritmo | Paper | Repositório oficial | Pesos oficiais | Formato de saída real |
|---|---|---|---|---|
| **TableFormer** | Nassar et al., CVPR 2022 | [docling-project/docling-ibm-models](https://github.com/docling-project/docling-ibm-models) | Hugging Face (`ds4sd/docling-models`) | Estrutura HTML + tokens/caixas casados |
| **TabStructNet** | Raja; Mondal; Jawahar, ECCV 2020 | [sachinraja13/TabStructNet](https://github.com/sachinraja13/TabStructNet) | Google Drive (linkado no README oficial, treinado em SciTSR) | XML próprio (`<tablecell><start_row>...`) |
| **Table Transformer (TATR)** | Smock et al. (PubTables-1M), 2022 | [microsoft/table-transformer](https://github.com/microsoft/table-transformer) | Hugging Face (`microsoft/table-transformer-structure-recognition`) | Detecções de objetos (linha/coluna/spanning cell) |
| **SLANet** | PaddleOCR PP-StructureV2 | [PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | [Modelo English SLANet para PubTabNet](https://www.paddleocr.ai/v2.10.0/en/ppstructure/models_list.html), baixado pelo PaddleOCR | HTML estrutural + caixas das células |

**Nota importante sobre o TableFormer:** diferente do que se poderia supor, o
`TFPredictor` não recebe apenas a imagem da tabela — ele espera uma "iocr_page" já
com o texto extraído (OCR ou parsing nativo de PDF) e casa esse texto com a estrutura
prevista. Ou seja, o pipeline de preparação de dados (`src/data_prep/`) precisa incluir
uma etapa de extração de texto antes de chamar esse modelo especificamente. Os demais
algoritmos recebem a imagem da tabela; para SLANet, o HTML interno é usado apenas para
estrutura e o texto é recuperado pelo fluxo compartilhado do experimento.

## Status da comparação pública

A comparação concluída usou os mesmos 150 exemplos para os quatro modelos:
PubTabNet `val` (50), FinTabNet `test` (50) e SciTSR `test` (50). A amostragem
foi sem reposição, com semente 42. Os dados PubTabNet e FinTabNet vieram das
revisões fixadas `c6b519724c09c81dc4797b742826681ebf5c7dde` e
`1fe37f0f9478b4c571c51c5c4d11e3ef357e75c4`, respectivamente. O manifesto está
em `data/manifests/public-samples-seed42.csv`, com SHA-256
`81642f2e2a95c6ed76f2af87e8bf6e85a43f6deab57dd8156df31a610c0c61b9`.

| Dataset | TableFormer | TabStructNet | Table Transformer | SLANet | Interseção estrutural |
|---|---:|---:|---:|---:|---:|
| PubTabNet | 0,978 / 0,979 | 0,829 / 0,872 | 0,742 / 0,803 | 0,992 / 0,991 | 46/50 |
| FinTabNet | 0,985 / 0,992 | 0,869 / 0,905 | 0,555 / 0,581 | 0,763 / 0,788 | 33/50 |
| SciTSR | 0,968 / 0,974 | 0,759 / 0,818 | 0,691 / 0,773 | 0,947 / 0,951 | 39/50 |

As células da tabela mostram médias de S-TEDS / cell-F1 pareadas na interseção
estrutural comum de cada conjunto. A execução válida foi: PubTabNet, 47/50 para
TableFormer, 50/50 para TabStructNet, 49/50 para Table Transformer e 50/50 para
SLANet; FinTabNet, 36/50, 50/50, 47/50 e 50/50, respectivamente. As falhas de
TableFormer decorreram de texto associado a células fora da estrutura prevista;
as do Table Transformer foram saídas vazias. TEDS teve texto válido para todos
os modelos em 33 exemplos PubTabNet, 29 FinTabNet e 34 SciTSR. Os resultados de
PubTabNet e FinTabNet são descritivos das amostras selecionadas e não representam
os splits completos.

O repositório do TabStructNet está em `third_party/` e seu checkpoint SciTSR está
local. O modelo SLANet é baixado automaticamente pelo PaddleOCR na primeira execução.

## Status das métricas de avaliação

| Métrica | Implementada? | Testada? | Observação |
|---|---|---|---|
| `cell_f1.py` | ✅ | ✅ | Correspondência exata de posição e extensão (rowspan/colspan); detecta deslocamentos, spans incorretos e células ausentes. Não usa texto. |
| `teds.py` | ✅ | ✅ | Distância de edição de árvore com custo de texto/span e denominador alinhado à referência PubTabNet. Não representa coordenadas absolutas; lacunas internas viram nós `td` vazios. |
| `s_teds.py` | ✅ | ✅ | Ignora texto e `td`/`th`, pois a anotação pública de cabeçalho não é comparável. Avalia sequência de linhas/células e spans. |

**Validação contra a referência PubTabNet:** executamos `src/metric.py` oficial e a
implementação local nos mesmos HTMLs. Em controles sintéticos, os valores coincidem:
tabela idêntica `1.0`, uma substituição de texto `0.6667`, célula vazia/gap `0.75` e
mesclagem idêntica `1.0`. Na amostra real `PMC5755158_010_01.jpg` do split `val`, a
tabela intacta marcou `1.0` em ambas; após alterar um token em uma célula, a referência
marcou `0.996` e a implementação local `0.99375` (delta `-0.00225`). A diferença vem
da simplificação local, que descarta os nós de seção `thead`/`tbody` na normalização.
Os valores locais não devem ser comparados diretamente aos resultados publicados sem
essa ressalva.

**Interpretação e cabeçalhos:** S-TEDS e cell-F1 são as métricas estruturais
principais. S-TEDS neutraliza `td`/`th`; cell-F1 é a medida explícita de posição e
extensão, pois TEDS só percebe deslocamentos quando a árvore resultante muda (por
exemplo, quando aparece um `td` vazio). TEDS permanece fiel à referência e penaliza
`td` versus `th`. Como os conjuntos públicos não fornecem rótulo de cabeçalho
compatível, essa penalidade textual não deve ser interpretada como erro semântico
de cabeçalho.

O protocolo atual é `teds_grid_v3`: TEDS é limitado ao intervalo `[0, 1]`; S-TEDS e
cell-F1 são as leituras estruturais principais, e TEDS só deve ser interpretado
quando a amostra tem texto recuperado. Os CSVs guardam a fonte/configuração do texto
por amostra. Na comparação final, TEDS teve texto avaliado em menos amostras que as
métricas estruturais. A exportação XLSX é uma saída do fluxo, mas não foi avaliada
neste experimento.

## Estrutura do projeto

```
tcc-table-recognition/
├── environment/            # requirements.txt por modelo (ambientes isolados)
├── configs/                 # configuração central do experimento (YAML)
├── data/
│   ├── raw/                 # datasets públicos brutos (PubTabNet, FinTabNet, SciTSR)
│   ├── corporate/            # subconjunto de documentos corporativos reais
│   ├── processed/             # regiões de tabela extraídas/padronizadas (entrada dos modelos)
│   └── ground_truth/          # anotações de referência (HTML/estrutura) para avaliação
├── src/
│   ├── data_prep/            # extração da região da tabela + padronização de entrada
│   ├── models/                # um "runner" por algoritmo, com interface comum
│   ├── normalization/          # converte a saída de cada modelo para uma representação comum
│   ├── evaluation/             # métricas: TEDS, S-TEDS, F1 por célula
│   ├── export/                 # exportação da estrutura reconstruída para .xlsx
│   └── run_experiment.py       # orquestra: dados -> modelo -> normalização -> avaliação -> export
├── results/
│   ├── logs/                   # logs de execução por modelo/dataset
│   ├── metrics/                 # métricas calculadas (CSV/JSON)
│   └── spreadsheets/            # planilhas .xlsx geradas a partir das tabelas reconstruídas
└── notebooks/                   # análise exploratória e geração dos gráficos da Seção 6
```

## Por que ambientes separados por modelo?

Os quatro algoritmos vêm de repositórios com matrizes de dependências muito diferentes
— e em dois casos, francamente incompatíveis entre si:

- **TableFormer**: PyTorch 2.x atual, via `docling-ibm-models` (pip instalável).
- **Table Transformer**: PyTorch 2.x atual, via `transformers` (pip instalável).
- **TabStructNet**: TensorFlow 1.15 / Keras 2.2 (Mask R-CNN do matterport) — exige
  Python ≤ 3.7, incompatível com os ambientes acima.
- **SLANet**: PaddlePaddle/PaddleOCR 2.x — isolado em venv próprio para não misturar
  runtime Paddle com os ambientes PyTorch/TensorFlow.

Por isso, cada `environment/requirements-<modelo>.txt` define um ambiente virtual
próprio (e, no caso do TabStructNet, um interpretador Python mais antigo). O
`src/models/<modelo>_runner.py` correspondente é pensado para rodar **dentro** desse
ambiente e expor uma interface comum (`BaseModelRunner`), de forma que o restante do
pipeline (normalização, avaliação, exportação) seja agnóstico a qual modelo gerou a
saída — e agnóstico também ao formato bruto de cada saída (HTML, detecções de
objetos, XML do TabStructNet, ou HTML e caixas do SLANet), resolvido em
`src/normalization/normalize_output.py`.

## Como começar

**Guia passo a passo completo (ambientes, checkpoints, datasets, execução):
ver [`SETUP.md`](SETUP.md).** O resumo rápido abaixo cobre só a criação dos venvs.

```bash
# 1) Ambiente base (orquestração, normalização, avaliação, exportação)
python -m venv .venv-base
source .venv-base/bin/activate   # Windows: .venv-base\Scripts\activate
pip install -r environment/requirements-common.txt

# 2) Um ambiente por modelo
python -m venv .venv-tableformer
source .venv-tableformer/bin/activate
pip install -r environment/requirements-tableformer.txt

# SLANet / PP-StructureV2 (CPU)
python -m venv .venv-slanet
source .venv-slanet/bin/activate
pip install -r environment/requirements-slanet.txt
```

## Fluxo do pipeline (ver Seção 4.2 da metodologia)

1. `src/data_prep/extract_table_regions.py` — localiza/recorta a região da tabela no documento.
2. `src/data_prep/standardize_input.py` — padroniza formato/resolução de entrada para os modelos.
3. `src/models/<modelo>_runner.py` — executa o algoritmo selecionado sobre a entrada padronizada.
4. `src/normalization/normalize_output.py` — converte a saída (HTML, matriz, etc.) para uma
   representação tabular comum.
5. `src/evaluation/{teds,s_teds,cell_f1}.py` — compara a estrutura normalizada com o gabarito.
6. `src/export/spreadsheet_exporter.py` — gera a planilha `.xlsx` a partir da estrutura reconstruída.
7. `src/run_experiment.py` — orquestra as etapas acima e grava os resultados em
   `results/`. **Roda um modelo por vez** (`--model`, obrigatório), já que os 4
   ambientes têm dependências mutuamente incompatíveis — não é possível rodar os 4
   no mesmo processo Python. Uso (um comando por modelo, cada um no seu `.venv-*`):

   ```bash
   .venv-tableformer/bin/python -m src.run_experiment --model tableformer
   .venv-tabstructnet/bin/python -m src.run_experiment --model tabstructnet
   .venv-tabletransformer/bin/python -m src.run_experiment --model tabletransformer
   .venv-slanet/bin/python -m src.run_experiment --model slanet
   ```

   Por padrão processa todos os datasets habilitados no config; `--dataset <nome>`
   restringe a um só (útil para testar rapidamente antes de rodar tudo).

## Próximos passos sugeridos

- [x] Ambientes e checkpoints dos quatro modelos configurados; checkpoint local de
      TabStructNet e revisões/modelos de inferência fixados para TableFormer, TATR e
      SLANet.
- [x] Recortes corporativos, manifesto com página/coordenadas/confiança e dois
      gabaritos curados com nome-base correspondente.
- [x] Recuperação OCR/PDF e associação de texto às células; metadados da fonte
      textual gravados por amostra.
- [x] Métricas de controle, comparação com a referência PubTabNet e testes de
      conversão para `NormalizedTable`.
- [x] Validar todos os candidatos locais e salvar imagens, gabaritos, dimensões,
      spans e motivos de exclusão em `results/validation/datasets-20261023.json`.
- [x] Fixar a próxima amostra SciTSR em 50 tabelas, selecionadas com seed 42 e
      registradas pelo manifesto comum a todos os runners.
- [x] Acrescentar a estrutura gabaritada das 50 amostras selecionadas ao arquivo
      de atributos, marcando rótulos visuais não revisados como pendentes.
- [x] Executar os quatro runners nos 150 exemplos públicos do manifesto e atualizar
      as médias pareadas e taxas de execução válida.
- [x] Validar o SLANet no ambiente PaddleOCR nos 50 exemplos de PubTabNet,
      50 de FinTabNet e na rodada SciTSR já existente.
- [x] Registrar as taxas de execução válida e os motivos das falhas nos resultados
      e no texto do TCC.
- [ ] Revisar manualmente os atributos visuais pendentes do SciTSR antes de
      apresentar análises por atributo.
- [x] Atualizar o texto final do TCC com os resultados da rodada e as ressalvas
      metodológicas documentadas aqui.

## Seleção de datasets e protocolo da amostra

Para a comparação final, são usados três conjuntos públicos, com 50 amostras por
conjunto e manifesto compartilhado entre os quatro modelos:

| Dataset | Split avaliado | Amostras | Decisão metodológica |
|---|---|---:|---|
| PubTabNet | `val` | 50 | Seleção aleatória sem reposição da [conversão OTSL de PubTabNet](https://huggingface.co/datasets/docling-project/PubTabNet_OTSL), revisão `c6b519724c09c81dc4797b742826681ebf5c7dde`. O split `test` oficial não tem gabarito público. |
| FinTabNet | `test` | 50 | Seleção aleatória sem reposição da [conversão OTSL de FinTabNet](https://huggingface.co/datasets/docling-project/FinTabNet_OTSL), revisão `1fe37f0f9478b4c571c51c5c4d11e3ef357e75c4`. |
| SciTSR | `test` | 50 | Usa os gabaritos `structure/` e imagens correspondentes já presentes no dataset local; seleção aleatória reproduzível. |

As três seleções usam a semente 42. Para reproduzir as amostras, instale
`pyarrow==21.0.0` no ambiente de preparação e execute:

```powershell
python -m pip install pyarrow==21.0.0
python scripts/fetch_public_dataset_samples.py --seed 42 --sample-size 50
```

O script lê temporariamente os arquivos Parquet das revisões fixadas, salva as
imagens e anotações selecionadas e gera `data/manifests/public-samples-seed42.csv`,
com os índices de origem OTSL e os caminhos exatos das tabelas SciTSR. O split
`test` vazio do PubTabNet é intencional: não se deve renomear
exemplos de `val` como `test`. A comparação usa as amostras processadas por todos
os modelos em cada conjunto; os resultados não representam os splits completos.
SHA-256 do manifesto congelado: `81642f2e2a95c6ed76f2af87e8bf6e85a43f6deab57dd8156df31a610c0c61b9`.

### Anonimização das amostras corporativas

Antes de uma nova execução corporativa, preencha localmente
`data/privacy/corporate_redactions.yaml` com uma regra para cada PNG-fonte. O arquivo
é ignorado pelo Git. Para cada recorte, informe um identificador sem relação com o
nome original, retângulos `[x0, y0, x1, y1]` em pixels da imagem padronizada e as
células `[linha, coluna]` correspondentes que contêm os dados ocultados:

```yaml
samples:
  crop-source-a.png:
    sample_id: corp-001
    image_regions:
      - [10, 20, 150, 45]
    cells:
      - [0, 1]
```

Se for necessário extrair novamente as tabelas dos PDFs, mantenha também a seleção
de arquivos e páginas em `data/privacy/corporate_pages.yaml` (igualmente ignorado
pelo Git), no formato `pages_by_pdf: {<nome-local-do-pdf>: [<páginas físicas>]}`.
A extração grava apenas na área-fonte privada.

As coordenadas e células devem ser identificadas e revisadas por uma pessoa; o
programa não detecta automaticamente dados pessoais ou comerciais. Se a revisão
confirmar que um recorte não contém identificadores, use listas vazias para
`image_regions` e `cells` e acrescente `reviewed_no_sensitive_data: true`. Gere os
derivados antes de executar os modelos. Primeiro, registre a autorização formal e
marque `datasets.corporate.privacy.authorization_confirmed: true` no YAML; o padrão
é `false` e a rotina recusa processar os documentos sem essa confirmação:

```bash
python -m src.data_prep.anonymize_corporate
```

O comando falha se não houver regra para cada recorte ou se regiões, células ou
identificadores forem inválidos. As saídas ficam em
`data/processed/corporate-anonymized/` e
`data/ground_truth/corporate-anonymized/`; os originais permanecem na área privada.
As regiões são cobertas de forma irreversível, o texto das células correspondentes
é removido e o manifesto de saída contém apenas pseudônimos e metadados mínimos.
Para amostras assim marcadas, a recuperação textual é forçada para OCR sobre a imagem
tratada, sem leitura do texto vetorial do PDF. Revise PNGs, gabaritos, resultados e
planilhas antes de qualquer compartilhamento. A anonimização não substitui a
autorização formal para uso ou divulgação.

## Recuperação de texto e métricas

A comparação principal é **estrutural**: S-TEDS e F1 por célula são reportados para
todos os modelos. TEDS, que combina conteúdo e estrutura, é um resultado secundário
e só é calculado quando há palavras reconhecidas e geometria disponível para a
associação textual. O CSV registra `text_source`, engine, configuração, palavras
reconhecidas/atribuídas, cobertura de células e `text_score_status`; TEDS fica vazio
quando falta base textual suficiente, em vez de comparar células artificialmente
vazias.

Em documentos não anonimizados, `text_recovery.mode: auto` pode usar texto vetorial
corporativo via PyMuPDF. Para os recortes gerados pelo fluxo de anonimização, o
manifesto sinaliza o estado tratado e a recuperação é exclusivamente OCR sobre o PNG
mascarado. Datasets públicos e documentos escaneados usam Tesseract conforme
`ocr_language`, `ocr_page_segmentation_mode` e `ocr_min_confidence` no YAML.

O TableFormer recebe essas palavras e caixas no `iocr_page` antes da inferência. O
Table Transformer, TabStructNet e SLANet recebem a mesma fonte textual; após
reconstruir a estrutura, cada palavra é associada à célula com maior sobreposição
relativa à caixa da palavra, exigindo `min_word_overlap`. As caixas vêm das detecções
do Table Transformer, do XML do TabStructNet e das caixas de célula retornadas pelo
SLANet. O texto reconhecido internamente pelo PaddleOCR é descartado para manter a
mesma fonte OCR/PDF compartilhada pelo experimento.

## Análise por atributos

`data/ground_truth/sample_metadata.csv` contém metadados das amostras avaliadas e
configuradas. Linhas,
colunas e mesclagens vêm dos gabaritos; a profundidade de cabeçalho vem do HTML em
PubTabNet e foi revisada visualmente em FinTabNet, SciTSR e corporativo. A visibilidade
de regras, bandamento e origem digital/escaneada foi classificada por inspeção das
imagens. As colunas `structure_basis`, `merged_basis`, `header_basis`, `visual_basis`
e `origin_basis` indicam a fonte de cada grupo de rótulos.

As 50 amostras SciTSR do manifesto têm linhas, colunas, células e mesclagens calculadas
pelos gabaritos. Como o SciTSR não anota cabeçalhos e as imagens ainda não foram
revisadas visualmente, os campos correspondentes aparecem explicitamente como
`not_annotated`/`not_reviewed`; não são contados como negativos. O arquivo de metadados
também pode conter amostras de seleções anteriores; o `attribute_summary.csv` da
execução filtra estritamente pelo manifesto congelado. Atualize o arquivo após mudar
a amostra configurada com:

```bash
python -m src.data_prep.update_sample_metadata
```

A complexidade é uma faixa exploratória, não uma propriedade oficial do dataset:
`score = cell_count + 4 * merged_cell_count + 5 * max(header_rows - 1, 0)`. Scores
até 20 são `low`, até 80 são `medium` e acima de 80 são `high`. Na amostra congelada,
46 tabelas não têm mesclagens e 16 têm; 50 registros SciTSR têm atributos de cabeçalho
e visuais pendentes por falta de anotação/revisão. O resumo por atributo guarda o
número de exemplos por grupo e as contagens efetivamente avaliadas; grupos pequenos
ou não revisados não permitem conclusões confiáveis.

Gere médias por atributo, incluindo `sample_count` e `evaluated_count`, com:

```bash
python -m src.evaluation.summarize_by_attributes
```

O resultado é salvo em `results/metrics/by_attribute_summary.csv`. No estado atual,
há metadados para 17 amostras, mas os únicos scores existentes são cinco registros
SciTSR/TableFormer gerados antes da versão corrigida das métricas; o agregador os
exclui por não terem `metric_protocol`. Assim, os grupos mostram contagens e médias
vazias até os modelos serem executados novamente. Com no máximo cinco amostras por
dataset, os resultados por grupo são exploratórios e não sustentam alegações
estatísticas de superioridade.

`pytesseract` está nas dependências comuns, mas o executável Tesseract e os idiomas
`eng`/`por` precisam estar instalados no sistema e no `PATH`. Em Windows, instale o
Tesseract OCR e confirme com `tesseract --list-langs` que os idiomas configurados
estão disponíveis. Se o executável não estiver no `PATH`, configure
`text_recovery.tesseract_cmd` no YAML ou defina `TESSERACT_CMD` para o caminho do
executável (por exemplo `C:\Program Files\Tesseract-OCR\tesseract.exe`).

Antes de qualquer inferência, validar loaders, imagens, gabaritos e spans:

```bash
python -m src.data_prep.validate_datasets
```

O validador informa candidatos, amostras verificadas, carregadas e ignoradas por
dataset. Uma amostra inválida é relatada, mas não impede o relatório dos demais casos;
o comando termina com erro se algum dataset habilitado não tiver nenhuma amostra
válida.

## Carregamento dos datasets

Implementado em `src/data_prep/dataset_loaders/`, um módulo por formato de anotação
(PubTabNet e FinTabNet compartilham o mesmo módulo, pois usam exatamente o mesmo
schema). `run_experiment.py` despacha para o loader certo conforme o nome do dataset.

| Dataset | Loader | Formato confirmado via | Testado? |
|---|---|---|---|
| PubTabNet | `pubtabnet_fintabnet_loader.py` | Schema OTSL convertido para `html.structure.tokens` + `html.cells` | ✅ 50 imagens/gabaritos de `val` carregados e normalizados; `train` local também confere. |
| FinTabNet | `pubtabnet_fintabnet_loader.py` (mesmo módulo) | `docling-project/FinTabNet_OTSL`, com HTML e células originais | ✅ 50 imagens/gabaritos de `test` carregados e normalizados; `train` e `val` locais também conferidos. |
| SciTSR | `scitsr_loader.py` | Repositório oficial + amostras reais publicadas (`huggingface.co/datasets/bevaya/SciTSR-cc-by-nc-sa`) | ✅ 50 pares do split `test` usados na comparação final; imagens presentes e `row_span`/`col_span` válidos. |
| Corporativo | `corporate_loader.py` | Amostra local; derivados pseudonimizados por `anonymize_corporate.py` | ⏳ Regras manuais de anonimização e reexecução dos modelos necessárias. |

**Nota sobre `is_header`:** nem PubTabNet/FinTabNet (que usam `<td>` mesmo dentro de
`<thead>`) nem SciTSR (que não anota essa distinção) marcam `is_header=True` no
gabarito. Isso é relevante para o TEDS — se um algoritmo prever `<th>` para células
de cabeçalho (caso do Table Transformer), a comparação com o gabarito vai registrar
uma divergência de tag mesmo quando a previsão está semanticamente correta. Vale
citar essa limitação na Seção 6 (Resultados) se aparecer nos números.

**Sobre o "IBTR" removido:** a versão da metodologia (Seção 4.2 do PDF original)
citava um quarto dataset público chamado "IBTR" junto com PubTabNet, FinTabNet e
SciTSR. Pesquisei bastante — inclusive no survey do Kasem et al. (2024), citado como
referência nessa mesma seção — e não encontrei nenhum dataset de reconhecimento de
tabelas com esse nome. Após confirmar com o autor, foi removido da lista; o texto do
TCC ainda precisa ser ajustado para não citá-lo (ver item na lista de pendências).

## Detecção de tabela em documentos corporativos (PDF cru)

Diferente dos datasets públicos (já anotados), os documentos corporativos reais
chegam como PDF sem nenhuma indicação de onde a tabela está na página. Essa etapa é
resolvida em `src/data_prep/table_detector.py` + `extract_table_regions.py`:

1. `rasterize_pdf_page()` converte cada página do PDF em imagem via **PyMuPDF**
      (`pymupdf`) — sem depender de um binário externo como o Poppler. A resolução
      (DPI) é configurável no YAML.
2. `TableDetector` localiza a tabela na página usando o **Table Transformer**,
   variante de DETECÇÃO (`microsoft/table-transformer-detection`) — mesma classe
   (`TableTransformerForObjectDetection`) e biblioteca (`transformers`) já usadas
   para reconhecimento de estrutura em `tabletransformer_runner.py`, só muda o
   checkpoint e a tarefa. A API foi confirmada via documentação oficial do modelo.
3. `python -m src.data_prep.extract_table_regions` processa os PDFs e páginas
      listados em `table_detection.pages_by_pdf`, recorta com margem, converte para RGB,
      limita o lado maior a 1024 px e salva o `manifest.csv` com PDF de origem,
      índice/página física, índice da detecção, caixa do detector, caixa do recorte,
      confidence e status. A lista é uma allowlist: PDFs novos não são processados até
      serem adicionados; `pages_by_pdf: null` processa todos os documentos e páginas.
      O comando foi executado com o detector real nos dois PDFs corporativos; encontrou
      uma tabela por página selecionada com confidence aproximada de 0,9998. Páginas sem
      tabela ficam como `no_detection`; tabelas rotacionadas ficam como
      `excluded_rotated` e não são enviadas aos modelos.

As imagens no `processed_dir` são a entrada compartilhada pelos quatro runners;
assim, o experimento compara estrutura nos mesmos recortes, não diferenças de
detecção de página. `standardize_input.py` é aplicado na geração dos recortes
corporativos. Imagens dos datasets públicos são mantidas intactas para preservar os
dados de referência e passam pelos pré-processadores próprios dos runners (por
exemplo, processor do Transformers para Table Transformer e configuração interna de
redimensionamento do TableFormer).

**Limitação:** tabelas rotacionadas ainda não são corrigidas para TSR. Elas são
registradas no manifesto como `excluded_rotated`, preservando caixa e confidence,
mas não geram imagem nem entram na comparação. Assim a exclusão fica explícita e
contável.
