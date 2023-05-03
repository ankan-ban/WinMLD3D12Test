// Minimal C++ example for using WinML with D3D12.
// 
// Goals:
//   - Avoid CPU <-> GPU transfers at each inference (TODO: also demonstrate how to use d3d12 copy queue to pipeline the copies)
//   - pipeline multiple inference requests to keep GPU occupied all the time.
// 
// Currently none of the above work :-/
// WinML runtime seems to block each session.Evaluate() call, and also copy the outputs to CPU internally
// 
// TODO: 
//  1. Fix the above issues
//  2. Update the test to load/store a real image (so that we can also test if it's working functionally).

const int warmupIterations = 100;
const int iterations = 100;

#include <stdio.h>
#include <vcruntime.h>
#include <windows.h>
#include <winrt/Windows.Foundation.h>
#include <winrt/Windows.Foundation.Collections.h>
#include <winrt/Windows.AI.MachineLearning.h>
#include <windows.ai.machinelearning.native.h>


using namespace winrt;
using namespace Windows::AI::MachineLearning;

#include "d3dx12.h"
#include <chrono>

void CreateD3D12Buffer(ID3D12Device *pDevice, const size_t size, ID3D12Resource** ppResource)
{
    D3D12_RESOURCE_DESC bufferDesc = {};
    bufferDesc.MipLevels = 1;
    bufferDesc.Format = DXGI_FORMAT_UNKNOWN;
    bufferDesc.Width = size;
    bufferDesc.Height = 1;
    bufferDesc.Flags = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
    bufferDesc.DepthOrArraySize = 1;
    bufferDesc.SampleDesc.Count = 1;
    bufferDesc.SampleDesc.Quality = 0;
    bufferDesc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    bufferDesc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    HRESULT hr = pDevice->CreateCommittedResource(
        &CD3DX12_HEAP_PROPERTIES(D3D12_HEAP_TYPE_DEFAULT),
        D3D12_HEAP_FLAG_NONE,
        &bufferDesc,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        nullptr,
        IID_PPV_ARGS(ppResource));

    if (FAILED(hr))
    {
        printf("\nFailed creating a resource\n");
        exit(0);
    }
}

int main()
{
    HRESULT hr = S_OK;

    ID3D12Device* pDevice;
    ID3D12CommandQueue* pCommandQueue;
    ID3D12Resource* pInput;
    ID3D12Resource* pOutput;

    // 1. Create Device
    hr = D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&pDevice));

    // 2. Create command queue
    D3D12_COMMAND_QUEUE_DESC queueDesc = {};
    queueDesc.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
    queueDesc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    hr = pDevice->CreateCommandQueue(&queueDesc, IID_PPV_ARGS(&pCommandQueue));

    // 3. Create d3d12 resources (to be used for input and output of the network)
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pInput);
    CreateD3D12Buffer(pDevice, 3 * 720 * 720 * sizeof(float), &pOutput);

    // Event and D3D12 Fence to manage CPU<->GPU sync (we want to keep 2 iterations in "flight")
    HANDLE hEvent = CreateEvent(nullptr, FALSE, FALSE, nullptr);
    ID3D12Fence* pFence = nullptr;
    pDevice->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&pFence));

    // 4. create LearningModelDevice from command queue	
    com_ptr<ILearningModelDeviceFactoryNative> dFactory =
        get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
    com_ptr<::IUnknown> spLearningDevice;
    hr = dFactory->CreateFromD3D12CommandQueue(pCommandQueue, spLearningDevice.put());
    LearningModelDevice pWinMLDevice = spLearningDevice.as<LearningModelDevice>();

    // 5. Load the onnx model from file
    auto model = LearningModel::LoadFromFilePath(L"fns-candy.onnx");    // fns-candy

    LearningModelSessionOptions options = {};
    // Important - always specify/override all named dimensions
    // (By default they are set to 1, but DML optimizations get turned off if a model
    //  has dynamic dimensions that are not explicitly specified).
    // This just crashes! TODO: figure out why?
    /*
    auto name = L"None";
    options.OverrideNamedDimension(name, (uint32_t)1);
    */
    options.BatchSizeOverride(1);       // What does this mean - given we can override anything using above API?
    options.CloseModelOnSessionCreation(true);

    LearningModelSession session(model, LearningModelDevice(pWinMLDevice), options);
    LearningModelBinding binding(session);

    // 6. Create WinML tensor Objects out of d3d12 resources and bind them to the model.
    TensorFloat inputTensor(nullptr), outputTensor(nullptr);

    com_ptr<ITensorStaticsNative> tensorfactory = get_activation_factory<TensorFloat, ITensorStaticsNative>();
    com_ptr<::IUnknown> spUnkTensor;
    int64_t shapes[4] = { 1, 3, 720, 720 };
    hr = tensorfactory->CreateFromD3D12Resource(pInput, shapes, 4, spUnkTensor.put());
    spUnkTensor.try_as(inputTensor);
    hr = tensorfactory->CreateFromD3D12Resource(pOutput, shapes, 4, spUnkTensor.put());
    spUnkTensor.try_as(outputTensor);

    binding.Bind(model.InputFeatures().GetAt(0).Name(), inputTensor);
    binding.Bind(model.OutputFeatures().GetAt(0).Name(), outputTensor);

    // 7. Run the model (schedule 100 iterations on the command queue for testing)
    // Warmup
    for (int i = 1; i <= warmupIterations; i++)
    {
        session.Evaluate(binding, L"RunId");
        pCommandQueue->Signal(pFence, i);
        pFence->SetEventOnCompletion(i, hEvent);    // immediately wait for the GPU results
        DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
    }
	

    // Actual run for benchmarking

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++)
    {
        session.Evaluate(binding, L"RunId");
        pCommandQueue->Signal(pFence, i + 1);

        // wait for (i-2)nd iteration (so that we have 2 iterations in flight)
        if (i > 1)
        {
            pFence->SetEventOnCompletion(i - 1, hEvent);
            DWORD retVal = WaitForSingleObject(hEvent, INFINITE);
        }
    }

    // Wait for the last iteration
    pFence->SetEventOnCompletion(iterations, hEvent);
    DWORD retVal = WaitForSingleObject(hEvent, INFINITE);

    auto end = std::chrono::high_resolution_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();

    // 9. Release d3d12 objects
    pFence->Release();
    pInput->Release();
    pOutput->Release();
    pCommandQueue->Release();
    pDevice->Release();

    printf("\nInference loop done. %d iterations in %g ms - avg: %g ms per iteration\n", iterations, duration, duration/iterations);
    return 0;
}