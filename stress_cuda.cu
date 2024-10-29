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
#ifdef _WIN32
    #include <conio.h>
#else
    #include <termios.h>
    #include <unistd.h>
    char getch() {
        termios oldt, newt;
        tcgetattr(STDIN_FILENO, &oldt);
        newt = oldt;
        newt.c_lflag &= ~(ICANON | ECHO);
        tcsetattr(STDIN_FILENO, TCSANOW, &newt);
        char ch = getchar();
        tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
        return ch;
    }
#endif

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
    FILE* pipe = _popen("nvidia-smi --query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.total,memory.used --format=csv,noheader,nounits", "r");
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

void monitorGPU(std::atomic<bool>& running) {
    while (running) {
        std::cout << "\n" << std::string(109, '-') << "\n";
        std::cout << "| GPU Utilization | Memory Utilization | Temperature | Power Draw | Total Memory | Used Memory | CUDA Cores |\n";
        std::cout << std::string(109, '-') << "\n";

        std::string output = runNvidiaSmi();
        if (output.empty()) {
            std::cerr << "Erro ao obter dados do nvidia-smi." << std::endl;
            return; 
        }

        std::istringstream stream(output);
        std::string line;
        while (std::getline(stream, line)) {
            line.erase(remove_if(line.begin(), line.end(), ::isspace), line.end());

            size_t pos1 = line.find(',');
            size_t pos2 = line.find(',', pos1 + 1);
            size_t pos3 = line.find(',', pos2 + 1);
            size_t pos4 = line.find(',', pos3 + 1);
            size_t pos5 = line.find(',', pos4 + 1);

            if (pos1 != std::string::npos && pos2 != std::string::npos && pos3 != std::string::npos && pos4 != std::string::npos && pos5 != std::string::npos) {
                int gpuUtil = std::stoi(line.substr(0, pos1));
                int memUtil = std::stoi(line.substr(pos1 + 1, pos2 - pos1 - 1));
                int temp = std::stoi(line.substr(pos2 + 1, pos3 - pos2 - 1));
                float power = std::stof(line.substr(pos3 + 1, pos4 - pos3 - 1));
                int totalMem = std::stoi(line.substr(pos4 + 1, pos5 - pos4 - 1));
                int usedMem = std::stoi(line.substr(pos5 + 1)); 

                int cudaCores = 0;
                int deviceCount;
                cudaGetDeviceCount(&deviceCount);

                for (int i = 0; i < deviceCount; ++i) {
                    cudaDeviceProp deviceProp;
                    cudaGetDeviceProperties(&deviceProp, i);
                    cudaCores += deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor;
                }

                std::cout << "| " 
                          << std::setw(14) << gpuUtil << "% "
                          << "| " << std::setw(17) << memUtil << "% "
                          << "| " << std::setw(9) << temp << " C "
                          << "| " << std::setw(8) << std::fixed << std::setprecision(2) << power << " W "
                          << "| " << std::setw(9) << totalMem << " MB "
                          << "| " << std::setw(8) << usedMem << " MB "
                          << "| " << std::setw(10) << cudaCores << " |\n";
            } else {
                std::cerr << "Erro ao analisar a linha: " << line << std::endl;
            }
        }

        std::cout << std::string(109, '-') << "\n";
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void handleError(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error in file '" << file << "' at line " << line << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (handleError(err, __FILE__, __LINE__))

int main() {
    while (true) {
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);

        if (deviceCount == 0) {
            std::cerr << "Nenhuma GPU CUDA detectada." << std::endl;
            return EXIT_FAILURE;
        }

        int device;
        cudaDeviceProp deviceProp;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "GPU detectada: " << deviceProp.name << "\n";
        std::cout << "Memoria Total (VRAM): " << deviceProp.totalGlobalMem / (1024 * 1024) << " MB\n";

        size_t availableMem = deviceProp.totalGlobalMem;
        size_t recommendedMem = availableMem * 1.0;
        std::cout << "Recomendado uso de 60% da total pra evitar travamentos: " 
                  << recommendedMem * 0.6 / (1024 * 1024) << " MB\n";

        size_t memoryToUse;
        std::cout << "Digite a quantidade de memoria a ser utilizada (em MB): ";
        std::cin >> memoryToUse;
        
        while (memoryToUse <= 0 || memoryToUse > availableMem / (1024 * 1024)) {
            std::cerr << "Quantidade de memoria invalida." << std::endl;
            std::cout << "Digite a quantidade de memoria a ser utilizada (em MB): ";
            std::cin >> memoryToUse;
        }

        int iterations;
        std::cout << "Digite o numero de cálculos a serem feitos (valores mais elevados para maior estresse): ";
        std::cin >> iterations;

        while (iterations <= 0) {
            std::cerr << "O numero de calculos deve ser um inteiro positivo." << std::endl;
            std::cout << "Digite o numero de cálculos a serem feitos (valores mais elevados para maior estresse): ";
            std::cin >> iterations;
        }

        const int safeArraySize = (memoryToUse * 1024 * 1024) / sizeof(float);

        float *d_data;
        HANDLE_ERROR(cudaMalloc(&d_data, safeArraySize * sizeof(float)));

        float *h_data = new float[safeArraySize];
        for (int i = 0; i < safeArraySize; i++) {
            h_data[i] = static_cast<float>(i % 1000);
        }
        HANDLE_ERROR(cudaMemcpy(d_data, h_data, safeArraySize * sizeof(float), cudaMemcpyHostToDevice));

        int threadsPerBlock = 1024; 
        int totalThreads = safeArraySize; 
        int numBlocks = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

        std::atomic<bool> running(true);
        std::thread monitorThread(monitorGPU, std::ref(running));

        auto start = std::chrono::high_resolution_clock::now();
        stressKernel<<<numBlocks, threadsPerBlock>>>(d_data, safeArraySize, iterations);
        HANDLE_ERROR(cudaDeviceSynchronize());

        running = false; 
        monitorThread.join(); 

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        cudaFree(d_data);
        delete[] h_data;

        std::cout << "Teste de estresse concluido em " << elapsed.count() << " segundos!" << std::endl;

        std::cout << "Pressione Backspace para encerrar o programa ou Enter para reiniciar..." << std::endl;
        char option = getch();
        
        if (option == 8) { // Backspace
            break;
        } else if (option == 13) { // Enter
            continue;
        }
    }
    return 0;
}
