# Segmentação de Ventrículos e Análise de Alzheimer com RM
**O artigo completo relatando o desenvolvimento do trabalho está no repositório com o nome de "PAI_Trabalho.pdf"**

Aplicação desenvolvida para a disciplina de Processamento e Análise de Imagens, focada na extração automática de estruturas cerebrais em exames de ressonância magnética e na avaliação de possíveis marcadores associados à Doença de Alzheimer.

O projeto combina processamento de imagens, machine learning e deep learning para investigar a relação entre a morfologia ventricular e o quadro clínico dos pacientes.

## Objetivos

* Segmentar automaticamente os ventrículos laterais em imagens axiais do dataset OASIS-2.

* Extrair descritores geométricos relevantes (área, circularidade, excentricidade, etc.).

* Avaliar modelos rasos (Regressão Linear, XGBoost) para classificação e regressão.

* Treinar modelos profundos baseados na EfficientNet-B0 para análise direta das imagens.

## Metodologia Resumida
### Segmentação

* Pipeline completo incluindo:

* Normalização e realce de contraste (CLAHE)

* Skull stripping

* Clusterização via K-Means para identificar LCR

* Filtragem por área, centralidade e transformada de distância

* Extração de Features

* Geração automática de métricas morfológicas dos ventrículos para uso em modelos clássicos.

### Modelos de Aprendizado

* Regressão Linear + threshold tuning para classificação.

* XGBoost para regressão de idade.

* EfficientNet-B0 com fine-tuning parcial para classificação e regressão.

## Principais Resultados

* Os modelos rasos apresentaram baixo desempenho, indicando que descritores isolados dos ventrículos não são suficientes para o diagnóstico.
* A EfficientNet apresentou overfitting acentuado, consequência direta do número reduzido de pacientes disponíveis.
* Na regressão de idade, a rede convergiu para a média do conjunto, típico de cenários com pouca variabilidade amostral.

## Conclusões

Mesmo com um pipeline sólido e arquiteturas modernas, a limitação crítica foi o tamanho reduzido da base quando dividida rigorosamente por paciente — condição essencial para evitar data leakage em neuroimagem.
O estudo confirma a necessidade de datasets maiores e mais variados para que modelos profundos generalizem adequadamente em aplicações clínicas.
