# Estressando Cuda Cores

Este trabalho tem como objetivo causar um estressamento nas Cudas Cores. Uma Cuda Core é uma unidade de processamento dentro de uma GPU (placa gráfica) da NVIDIA que é projetada para executar operações em paralelo. 
O estressamento é realizado atráves de cálculos intensivos.

# ✒️ Autores

- Gabriel de Souza Borba
- João Vitor Schmidt
- Sâmela Hostins

# ⌨️ Principais componentes e funcionalidades do projeto:

### 1. Detecção e Configuração Automática da GPU
O projeto identifica automaticamente a placa de vídeo Nvidia disponível usando a biblioteca CUDA Toolkit:
- **`cudaGetDeviceCount`**: verifica quantas GPUs compatíveis com CUDA estão presentes no sistema.
- **`cudaGetDevice`**: seleciona a GPU específica para ser usada, configurando-a como ativa.
- **`cudaGetDeviceProperties`**: coleta informações detalhadas sobre a GPU, como a quantidade de memória de vídeo (VRAM), núcleos CUDA e frequência de operação. Essas informações são fundamentais para configurar o teste de estresse.

### 2. Proteção de Memória e Limites de Uso
O programa monitora o uso da GPU e recomenda que não se exceda 60% da capacidade de memória para evitar travamentos. Caso o uso ultrapasse esse limite, o programa pode travar ou interromper a execução. Você não deve colocar um valor maior que a capacidade de memória máxima, nem um valor menor ou igual a zero.

### 3. Processamento e Cálculos de Estresse
O usuário define a quantidade de cálculos para o teste de estresse. A aplicação cria um array para armazenar operações matemáticas de forma aleatória entre multiplicação e divisão, com números entre 1 e 1000, que são distribuídas para execução pelos núcleos CUDA. As operações são carregadas de uma só vez para gerar um estresse eficiente dos Cuda Cores.

### 4. Monitoramento de Performance e Consumo de Energia
Durante o teste, utilizamos o software MSI Afterburner para monitorar o consumo de energia da GPU (Power Draw) e outras métricas, como a frequência e o uso da memória. Devido ao processamento intensivo, é esperado um aumento temporário no consumo de energia. Mas a utilização desse software não é obrigatória, é apenas uma forma de monitorar o processo.

### 5. Execução do Kernel de Estresse
- **`StressKernel`**: envia operações matemáticas para o kernel do sistema, onde são processadas pelos núcleos CUDA da GPU.
- **`runNvidiaSmi`**: coleta informações detalhadas sobre o uso da GPU.
- **`monitorGPU`**: executa a visualização do desempenho em tempo real, permitindo o monitoramento do estado da GPU durante o estresse.

### 6. Resultados e Conclusão do Teste
Ao final do teste, o tempo total de execução é registrado, permitindo a análise da capacidade de processamento da GPU e da estabilidade sob carga intensa.


# 🚀 Começando

Essas instruções permitirão que você obtenha uma cópia do projeto em operação na sua máquina local para fins de teste.

### 📋 Pré-requisitos

- **Placa de Vídeo da NVIDIA**: Certifique-se de que seu computador ou notebook tenha uma GPU NVIDIA compatível com CUDA.

- **CUDA Toolkit**: Baixe e instale a versão 12.6.2 do CUDA Toolkit a partir do site oficial da NVIDIA: [Download CUDA Toolkit 12.6.2](https://developer.nvidia.com/cuda-toolkit-archive). Este kit contém as bibliotecas e ferramentas necessárias para desenvolvimento. Checar a etapa [Sobre o CUDA](#sobre-o-cuda) antes de fazer a instalação.

- **IDE para C++**: Utilize uma IDE que suporte C++, como o [Visual Studio](https://visualstudio.microsoft.com/pt-br/downloads/).

- **Compilador C++**: Instale o módulo do Visual Studio **Desenvolvimento para desktop com C++** pois ele instala um compilador, ou caso prefira instale um compilador de sua preferência.

- **Clonar o Projeto do GitHub**: Veja a seção [Como Clonar o Projeto](#como-clonar-o-projeto) para instruções sobre como clonar este repositório.


## 🖥️ Como Clonar o Projeto

Você pode clonar este projeto do GitHub de duas maneiras: usando o **Git Bash** ou baixando o repositório como um arquivo ZIP. Aqui estão as instruções para ambos os métodos:

### Opção 1: Clonar com Git Bash

1. **Instale o Git**: Se você ainda não tem o Git instalado, você pode baixá-lo [aqui](https://git-scm.com/downloads).

2. **Abra o Git Bash**: Após a instalação, abra o Git Bash. Você pode encontrá-lo no menu Iniciar ou na sua lista de aplicativos.

3. **Clone o Repositório**: Use o seguinte comando no Git Bash (substitua `URL_DO_REPOSITORIO` pela URL do repositório):

   ```bash
   git clone https://github.com/usuario/nome-do-repositorio.git

4. **Abra a IDE**: Inicie o Visual Studio Code (ou a IDE de sua preferência).

5. **Abrir a Pasta do Projeto**: Na IDE, vá até o menu `File` (Arquivo) e selecione `Open Folder...` (Abrir Pasta...). Navegue até a pasta onde você clonou o repositório e clique em `Select Folder` (Selecionar Pasta).


### Opção 2: Baixar como Arquivo ZIP

1. **Acesse o Repositório no GitHub**: Vá até a página do repositório que você deseja baixar.

2. **Clique em "Code"**: No canto superior direito da página, clique no botão verde "Code".

3. **Baixar ZIP**: Selecione a opção "Download ZIP". Isso baixará o repositório como um arquivo compactado.

4. **Extrair o ZIP**: Após o download, localize o arquivo ZIP no seu computador e extraia-o em uma pasta de sua escolha.

5. **Abra a IDE**: Inicie o Visual Studio Code (ou a IDE de sua preferência).

6. **Abrir a Pasta do Projeto**: Na IDE, vá até o menu `File` (Arquivo) e selecione `Open Folder...` (Abrir Pasta...). Navegue até a pasta onde você extraiu o repositório e clique em `Select Folder` (Selecionar Pasta).


## 🔧 Sobre o CUDA

É necessário ao baixar o CUDA fazer mais algumas etapas para completar seu ambiente de desenvolvimento.

1. **Extrair os arquivos da pasta do download no local correto**
Copie os arquivos extraídos para os diretórios do CUDA Toolkit:
- `cudnn.h` para `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\include`
- `cudnn.lib` para `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\lib\x64`
- `cudnn.dll` para `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin`


2. **Adicionar CUDA ao PATH**
   - Pesquise por "variáveis de ambiente" no Windows e abra as configurações.
   - Clique em "Variáveis de ambiente" e, na seção "Variáveis do sistema", encontre a variável `Path`.
   - Adicione as seguintes entradas (substitua X.X pela versão do CUDA que você instalou):
     ```
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\bin
     C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\libnvvp
     ```

2. **Verificar a Instalação do CUDA**

Para garantir que o CUDA foi instalado corretamente e está funcionando, siga os passos abaixo:

1. **Abra o Prompt de Comando ou PowerShell**.
2. **Verifique a versão do CUDA** executando o seguinte comando:
   ```bash
   nvcc --version

## ⚙️ Como Baixar e Instalar o Visual Studio para Compilar C++

**Passo 1:** Baixar o Instalador do Visual Studio
1. Acesse a página de download do Visual Studio: https://visualstudio.microsoft.com/
2. Clique em "Baixar Visual Studio" e escolha a edição desejada (Community, Professional ou Enterprise).

**Passo 2:** Iniciar o Instalador
1. Execute o arquivo baixado para iniciar o instalador do Visual Studio.
2. Após carregar, você verá uma lista de cargas de trabalho (workloads) disponíveis.

**Passo 3:** Selecionar o Módulo de Desenvolvimento para Desktop com C++
1. Na seção de cargas de trabalho, marque "Desenvolvimento para desktop com C++".
2. Confirme que a opção inclui:
   - Ferramentas do C++ (como compilador e depurador)
   - Ferramentas para desenvolvimento de aplicativos nativos em C++
3. (Opcional) Se desejar funcionalidades adicionais, você pode selecionar outras cargas de trabalho ou componentes individuais no menu "Componentes individuais".

**Passo 4:** Instalar o Visual Studio
1. Clique em "Instalar" para iniciar a instalação. Dependendo da sua conexão com a internet, esse processo pode levar alguns minutos.
2. Aguarde a conclusão da instalação. O Visual Studio será iniciado automaticamente após a instalação, se a opção "Iniciar após a instalação" estiver marcada.

**Passo 5:** Abrir o Visual Studio e Configurar um Projeto em C++
1. Abra o Visual Studio (se ainda não estiver aberto).
2. Na tela inicial, selecione "Criar um novo projeto".
3. Escolha um dos templates de projeto em C++ (por exemplo, "Aplicativo de Console em C++") e clique em "Avançar".
4. Defina um nome e um local para o projeto, e clique em "Criar" para começar.

# 📦 Como executar

- Abra um terminal, pode ser dentro da sua IDE ou fora
- Entre no caminho da pasta referente ao projeto aberto, como no exemplo abaixo referente ao caminho que encontra a minha pasta:
```bash
 cd C:\Users\Documents\EstressandoCudaCores
 ```
- Rode o executável (dependendo do número de interações pode demorar para conclusão):
```bash
nvcc -o stress_cuda stress_cuda.cu
 ```

