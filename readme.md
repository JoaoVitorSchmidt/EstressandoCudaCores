# Estressando Cuda Cores

Este trabalho tem como objetivo causar um estressamento nas Cudas Cores. Uma Cuda Core é uma unidade de processamento dentro de uma GPU (placa gráfica) da NVIDIA que é projetada para executar operações em paralelo. 

## 🚀 Começando

Essas instruções permitirão que você obtenha uma cópia do projeto em operação na sua máquina local para fins de teste.

### 📋 Pré-requisitos

- **Placa de Vídeo da NVIDIA**: Certifique-se de que seu computador ou notebook tenha uma GPU NVIDIA compatível com CUDA.

- **CUDA Toolkit**: Baixe e instale a versão 12.6.2 do CUDA Toolkit a partir do site oficial da NVIDIA: [Download CUDA Toolkit 12.6.2](https://developer.nvidia.com/cuda-toolkit-archive). Este kit contém as bibliotecas e ferramentas necessárias para desenvolvimento. Checar a etapa [Sobre o CUDA](#sobre-o-cuda) antes de fazer a instalação.

- **IDE para C++**: Utilize uma IDE que suporte C++, como o [Visual Studio Code (VSCode)](https://code.visualstudio.com/).

- **Extensões para VSCode ou IDE de preferência**: Instale o [C/C++ Extension Pack](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools) para suporte completo ao desenvolvimento em C++ no VSCode.

- **Clonar o Projeto do GitHub**: Veja a seção [Como Clonar o Projeto](#como-clonar-o-projeto) para instruções sobre como clonar este repositório.


### 🖥️ Como Clonar o Projeto

Você pode clonar este projeto do GitHub de duas maneiras: usando o **Git Bash** ou baixando o repositório como um arquivo ZIP. Aqui estão as instruções para ambos os métodos:

#### Opção 1: Clonar com Git Bash

1. **Instale o Git**: Se você ainda não tem o Git instalado, você pode baixá-lo [aqui](https://git-scm.com/downloads).

2. **Abra o Git Bash**: Após a instalação, abra o Git Bash. Você pode encontrá-lo no menu Iniciar ou na sua lista de aplicativos.

3. **Clone o Repositório**: Use o seguinte comando no Git Bash (substitua `URL_DO_REPOSITORIO` pela URL do repositório):

   ```bash
   git clone https://github.com/usuario/nome-do-repositorio.git

4. **Abra a IDE**: Inicie o Visual Studio Code (ou a IDE de sua preferência).

5. **Abrir a Pasta do Projeto**: Na IDE, vá até o menu `File` (Arquivo) e selecione `Open Folder...` (Abrir Pasta...). Navegue até a pasta onde você clonou o repositório e clique em `Select Folder` (Selecionar Pasta).


#### Opção 2: Baixar como Arquivo ZIP

1. **Acesse o Repositório no GitHub**: Vá até a página do repositório que você deseja baixar.

2. **Clique em "Code"**: No canto superior direito da página, clique no botão verde "Code".

3. **Baixar ZIP**: Selecione a opção "Download ZIP". Isso baixará o repositório como um arquivo compactado.

4. **Extrair o ZIP**: Após o download, localize o arquivo ZIP no seu computador e extraia-o em uma pasta de sua escolha.

5. **Abra a IDE**: Inicie o Visual Studio Code (ou a IDE de sua preferência).

6. **Abrir a Pasta do Projeto**: Na IDE, vá até o menu `File` (Arquivo) e selecione `Open Folder...` (Abrir Pasta...). Navegue até a pasta onde você extraiu o repositório e clique em `Select Folder` (Selecionar Pasta).


### 🔧 Sobre o CUDA

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

## ⚙️ C++



### ⌨️ Como esse projeto funciona?


```
Dar exemplos
```

## 📦 Como executar

Adicione notas adicionais sobre como implantar isso em um sistema ativo

## ✒️ Autores

- 
- João Vitor Schmidt
- Sâmela Hostins
