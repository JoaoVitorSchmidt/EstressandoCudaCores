#include <iostream>
#include <cmath>
#include <cuda_runtime.h>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <atomic>
#include <sstream>
#include <string>
#include <cstdio>
#include <iomanip>
#include <GLFW/glfw3.h>

__global__ void stressKernel(float *data, int size, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float val = data[idx];
        for (int i = 0; i < iterations; ++i) {
            val = sin(val) * cos(val) + sqrt(val);
            val += tanh(val) * exp(val) - log(val + 1.0f);
        }
        data[idx] = val;
    }
}

std::string runNvidiaSmi() {
    FILE* pipe = _popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.total,memory.used,gpu_name --format=csv,noheader,nounits", "r");
    if (!pipe) {
        std::cerr << "Não foi possível executar nvidia-smi." << std::endl;
        return "";
    }

    std::string result;
    char buffer[128];
    while (fgets(buffer, sizeof(buffer), pipe) != nullptr) {
        result += buffer;
    }
    _pclose(pipe);
    return result;
}

void drawBar(float x, float y, float width, float height, float value, const std::string& label) {
    glBegin(GL_QUADS);
    glColor3f(0.2f, 0.7f, 0.2f); // Verde claro
    glVertex2f(x, y);
    glVertex2f(x + width * value, y);
    glVertex2f(x + width * value, y + height);
    glVertex2f(x, y + height);
    glEnd();

    glColor3f(1.0f, 1.0f, 1.0f); // Texto branco
    glRasterPos2f(x + width / 2.0f - 0.05f, y + height + 0.05f);
    for (const char& c : label) {
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
    }
}

void monitorGPU(std::atomic<bool>& running) {
    if (!glfwInit()) {
        std::cerr << "Falha ao inicializar o GLFW." << std::endl;
        return;
    }

    GLFWwindow* window = glfwCreateWindow(600, 400, "Monitoramento da GPU", nullptr, nullptr);
    if (!window) {
        std::cerr << "Falha ao criar a janela GLFW." << std::endl;
        glfwTerminate();
        return;
    }

    glfwMakeContextCurrent(window);
    glClearColor(0.1f, 0.1f, 0.1f, 1.0f); // Cor de fundo

    while (running && !glfwWindowShouldClose(window)) {
        std::string output = runNvidiaSmi();
        if (output.empty()) {
            std::cerr << "Erro ao obter dados do nvidia-smi." << std::endl;
            return;
        }

        std::istringstream stream(output);
        std::string line;
        std::getline(stream, line);
        line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());

        size_t pos1 = line.find(',');
        size_t pos2 = line.find(',', pos1 + 1);
        size_t pos3 = line.find(',', pos2 + 1);
        size_t pos4 = line.find(',', pos3 + 1);
        size_t pos5 = line.find(',', pos4 + 1);
        size_t pos6 = line.find(',', pos5 + 1);

        int gpuUtil = std::stoi(line.substr(0, pos1));
        int memUtil = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
        int temp = std::stoi(line.substr(pos2 + 1, pos3 - pos2 - 1));
        std::string gpuName = line.substr(pos4 + 1, pos5 - pos4 - 1);

        glClear(GL_COLOR_BUFFER_BIT); // Limpar a tela antes de desenhar

        // Desenhando as barras de uso
        drawBar(-0.9f, 0.5f, 1.8f, 0.1f, gpuUtil / 100.0f, "Uso da GPU: " + std::to_string(gpuUtil) + "%");
        drawBar(-0.9f, 0.2f, 1.8f, 0.1f, memUtil / 100.0f, "Uso da Memória: " + std::to_string(memUtil) + "%");
        drawBar(-0.9f, -0.1f, 1.8f, 0.1f, temp / 100.0f / 100.0f, "Temperatura: " + std::to_string(temp) + "°C");

        // Exibindo o nome da GPU
        glRasterPos2f(-0.9f, 0.8f);
        for (const char& c : gpuName) {
            glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();

        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    glfwDestroyWindow(window);
    glfwTerminate();
}

int main() {
    int deviceCount;
    cudaError_t cudaStatus = cudaGetDeviceCount(&deviceCount);

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erro ao obter contagem de dispositivos CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return EXIT_FAILURE;
    }

    if (deviceCount == 0) {
        std::cerr << "Nenhuma GPU CUDA detectada." << std::endl;
        return EXIT_FAILURE;
    }

    int device;
    cudaDeviceProp deviceProp;
    cudaStatus = cudaGetDevice(&device);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erro ao obter dispositivo CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return EXIT_FAILURE;
    }

    cudaStatus = cudaGetDeviceProperties(&deviceProp, device);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erro ao obter propriedades do dispositivo CUDA: " << cudaGetErrorString(cudaStatus) << std::endl;
        return EXIT_FAILURE;
    }

    // Exibindo o nome da GPU
    std::cout << "Placa de Vídeo: " << deviceProp.name << std::endl;

    // Perguntando ao usuário quanto da memória ele deseja usar
    float userMemoryPercentage;
    std::cout << "Porcentagem de memória que deseja usar (0 a 100): ";
    std::cin >> userMemoryPercentage;

    if (userMemoryPercentage < 0 || userMemoryPercentage > 100) {
        std::cerr << "Valor inválido. A porcentagem deve estar entre 0 e 100." << std::endl;
        return EXIT_FAILURE;
    }

    size_t availableMem = deviceProp.totalGlobalMem;
    size_t memoryToUse = availableMem * (userMemoryPercentage / 100.0);

    // Garantir que a memória a ser usada seja um valor válido
    if (memoryToUse < 1) {
        std::cerr << "Memória insuficiente para o teste de estresse!" << std::endl;
        return EXIT_FAILURE;
    }

    // Correção do cálculo do safeArraySize
    size_t safeArraySize = memoryToUse / sizeof(float);
    
    std::cout << "Memória disponível para o teste: " << memoryToUse / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Safe Array Size: " << safeArraySize << " (número de elementos)" << std::endl;

    int threadsPerBlock = 256;  // Ajustando para um valor válido
    int numBlocks = (safeArraySize + threadsPerBlock - 1) / threadsPerBlock;

    // Exibir valores de numBlocks e threadsPerBlock
    std::cout << "Número de Blocos: " << numBlocks << std::endl;
    std::cout << "Threads por Bloco: " << threadsPerBlock << std::endl;

    // Verificando se a configuração é válida para a GPU
    if (numBlocks > deviceProp.maxGridSize[0] || threadsPerBlock > deviceProp.maxThreadsPerBlock) {
        std::cerr << "Configuração do kernel inválida: Número de blocos ou threads excede o limite da GPU." << std::endl;
        return EXIT_FAILURE;
    }

    float* h_data = new float[safeArraySize];
    float* d_data;
    cudaStatus = cudaMalloc(&d_data, safeArraySize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erro ao alocar memória na GPU: " << cudaGetErrorString(cudaStatus) << std::endl;
        return EXIT_FAILURE;
    }

    std::atomic<bool> running(true);
    std::thread gpuMonitorThread(monitorGPU, std::ref(running));

    stressKernel<<<numBlocks, threadsPerBlock>>>(d_data, safeArraySize, 500);
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "Erro ao sincronizar o dispositivo: " << cudaGetErrorString(cudaStatus) << std::endl;
    }

    running = false;
    gpuMonitorThread.join();

    cudaFree(d_data);
    delete[] h_data;

    return EXIT_SUCCESS;
}
